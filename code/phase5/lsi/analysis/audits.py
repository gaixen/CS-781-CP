"""Audits 1-4: data / label structure, predictions, explicit statute citations and the submission zips.

All audits are read-only; none of them changes the submissions.
"""
import glob
import json
import re
import time
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from lsi.data.loader import load_json, norm_docs
from lsi.eval.metrics import (GRID, _counts, all_metrics, decide, ece_flat, f1_labels, macro_f1, micro_f1, sigmoid, tune_global)
from lsi.training.submission import to_pred
from lsi.utils import md5_text

pd.set_option("display.width", 250, "display.max_columns", 40)


def audit_data(ctx):
    """AUDIT 1: label frequency / imbalance, cardinality, train->dev prior shift, co-occurrence, lengths, duplicates and leakage.

    Side effect: ``ctx.dup_test`` (test documents with an exact / near twin in train or dev) and ``ctx.audit_twins`` are filled.

    Args:
        ctx (Context): Pipeline context.

    Returns:
        Dict[str, list]: ``ctx.dup_test`` with keys ``train``, ``dev``, ``near``.
    """
    cfg, d = ctx.cfg, ctx.data
    train_docs, dev_docs, test_docs, LABELS, YTR, YDEV = d.train_docs, d.dev_docs, d.test_docs, d.labels, d.ytr, d.ydev
    L = d.n_labels
    H = md5_text
    fr, fd = YTR.sum(0), YDEV.sum(0)
    order = np.argsort(-fr)
    Co = YTR.T @ YTR
    print("== label frequency (train docs) ==")
    print("top-10   :", [(LABELS[j], int(fr[j])) for j in order[:10]])
    print("bottom-10:", [(LABELS[j], int(fr[j])) for j in order[-10:]])
    cs = np.cumsum(np.sort(fr)[::-1]) / fr.sum()
    print(f"max/min = {fr.max()/max(fr.min(),1):.1f}x | median {np.median(fr):.0f} | labels holding 50% of all positives: {int((cs < .5).sum() + 1)} of {L} | labels with dev support: {int((fd > 0).sum())}")
    card = lambda Y: dict(sorted(Counter(np.minimum(Y.sum(1).astype(int), 10)).items()))
    print("\n== labels per document (10 = 10+) ==\ntrain:", card(YTR), "\ndev  :", card(YDEV))
    ptr, pdv = fr / len(YTR), fd / len(YDEV)
    rel = np.log((pdv + 1e-9) / (ptr + 1e-9))
    print(f"\n== train->dev prior shift == L1 distance {np.abs(ptr - pdv).sum():.3f} | most shifted labels (dev/train rate):",
          [(LABELS[j], round(float(pdv[j] / max(ptr[j], 1e-9)), 2)) for j in np.argsort(-np.abs(rel))[:6]])
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    jac = {(i, j): Co[i, j] / max(fr[i] + fr[j] - Co[i, j], 1) for i, j in pairs}
    print("\n== co-occurrence == strongest pairs (Jaccard):", [(LABELS[i], LABELS[j], round(float(v), 2)) for (i, j), v in sorted(jac.items(), key=lambda x: -x[1])[:8]])
    print(f"pairs that never co-occur in train: {sum(Co[i, j] == 0 for i, j in pairs)}/{len(pairs)} | mean P(j|i) over co-occurring pairs: {np.mean([Co[i, j] / fr[i] for i, j in pairs if Co[i, j] > 0]):.2f}")
    print("\n== document length (words) ==")
    W = {k: np.array([len(x["text"].split()) for x in docs]) for k, docs in (("train", train_docs), ("dev", dev_docs), ("test", test_docs))}
    print(pd.DataFrame({k: dict(p10=np.percentile(v, 10), median=np.median(v), p90=np.percentile(v, 90), max=v.max(), empty_or_tiny=int((v < 20).sum())) for k, v in W.items()}).T.round(0).to_string())
    lim = cfg.inf_chunks * (cfg.chunk_tokens - cfg.overlap) * 0.75
    print(f"test/train median ratio {np.median(W['test'])/np.median(W['train']):.2f} | docs longer than {cfg.inf_chunks} chunks (~{int(lim)} words): train {np.mean(W['train'] > lim):.1%}, test {np.mean(W['test'] > lim):.1%}")
    print("\n== duplicates & leakage ==")
    nrm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    K = lambda t: H(nrm(t)[:300] + "|" + nrm(t)[-300:])
    Htr, Ktr = defaultdict(list), defaultdict(list)
    for i, x in enumerate(train_docs):
        Htr[H(x["text"])].append(i)
        Ktr[K(x["text"])].append(i)
    grp = [v for v in Htr.values() if len(v) > 1]
    kgrp = [v for v in Ktr.values() if len(v) > 1]
    same = lambda g: len({tuple(train_docs[i]["labels"]) for i in g}) == 1
    jac_g = lambda g: np.mean([len(set(train_docs[a]["labels"]) & set(train_docs[b]["labels"])) / max(len(set(train_docs[a]["labels"]) | set(train_docs[b]["labels"])), 1) for a in g for b in g if a < b])
    print(f"train: exact-duplicate groups {len(grp)} ({sum(len(g) for g in grp)} docs) | identical label sets {np.mean([same(g) for g in grp]) if grp else float('nan'):.1%} | mean label Jaccard {np.mean([jac_g(g) for g in grp]) if grp else float('nan'):.3f}")
    print(f"train: near-duplicate groups (same first+last 300 chars) {len(kgrp)} ({sum(len(g) for g in kgrp)} docs) | identical label sets {np.mean([same(g) for g in kgrp]) if kgrp else float('nan'):.1%} | mean label Jaccard {np.mean([jac_g(g) for g in kgrp]) if kgrp else float('nan'):.3f}")
    raw_dev = norm_docs(load_json(d.paths["dev"]))
    Hdev_raw = defaultdict(list)
    for x in raw_dev:
        Hdev_raw[H(x["text"])].append(x["labels"])
    te_h = [H(x["text"]) for x in test_docs]
    te_k = [K(x["text"]) for x in test_docs]
    tr_hset, tr_kset = set(Htr), set(Ktr)
    print(f"original dev.json rows {len(raw_dev)}: also in train {sum(H(x['text']) in Htr for x in raw_dev)} | also in test {sum(H(x['text']) in set(te_h) for x in raw_dev)}")
    agree = [(set(x["labels"]) == set(train_docs[Htr[H(x['text'])][0]]["labels"])) for x in raw_dev if H(x["text"]) in Htr]
    print(f"dev rows that exist in train: label sets identical to the train twin in {np.mean(agree) if agree else float('nan'):.1%} of {len(agree)}")
    n_tr, n_dv, n_nd = sum(h in tr_hset for h in te_h), sum(h in Hdev_raw for h in te_h), sum(k in tr_kset for k in te_k)
    print(f"TEST documents (of {len(test_docs)}) that are EXACT copies of a train doc: {n_tr} | of an original-dev row: {n_dv} | NEAR copies of a train doc: {n_nd}")
    print("  -> if the label-agreement above is ~100%, these test documents can be answered by label lookup (deterministic fix).")
    ctx.dup_test = dict(train=[i for i, h in enumerate(te_h) if h in tr_hset], dev=[i for i, h in enumerate(te_h) if h in Hdev_raw], near=[i for i, k in enumerate(te_k) if k in tr_kset])
    ctx.audit_twins = (Htr, te_h)
    return ctx.dup_test


def _topk_decode(P, k):
    """Predict the top-``k[i]`` labels of every document.

    Args:
        P (np.ndarray): Probabilities (n, L).
        k (Sequence[int]): Number of labels per document.

    Returns:
        np.ndarray: Boolean predictions.
    """
    order = np.argsort(-P, 1)
    keep = np.arange(P.shape[1])[None, :] < np.asarray(k)[:, None]
    B = np.zeros(P.shape, bool)
    np.put_along_axis(B, order, keep, 1)
    return B


def _ece_bins(P, Y, edges=(0, .05, .1, .2, .3, .5, .7, 1.0001)):
    """Reliability table (probability bin, count, mean predicted, observed rate).

    Args:
        P (np.ndarray): Probabilities.
        Y (np.ndarray): References.
        edges (Tuple[float, ...]): Bin edges.

    Returns:
        List[tuple]: One tuple per non-empty bin.
    """
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (P >= a) & (P < b)
        if m.sum():
            out.append((f"{a:g}-{min(b,1):g}", int(m.sum()), round(float(P[m].mean()), 3), round(float(Y[m].mean()), 3)))
    return out


def audit_predictions(ctx):
    """AUDIT 2: ceilings (oracle thresholds / cardinality), per-quartile quality, over/under-prediction, calibration, confusions,
    model agreement and dev-vs-test shift.

    Args:
        ctx (Context): Pipeline context after ``load_available`` and ``diagnostics``.

    Returns:
        dict: ``PD`` (dev probabilities), ``BDm`` / ``BTm`` (dev / test decisions at each model's global threshold) and ``SUMM`` (summary table).
    """
    d = ctx.data
    YDEV, TRAIN_FREQ, LABELS, L = d.ydev, d.train_freq, d.labels, d.n_labels
    qs = np.quantile(TRAIN_FREQ, [.25, .5, .75])
    BK = np.digitize(TRAIN_FREQ, qs)
    BN = ["Q1 rare", "Q2", "Q3", "Q4 frequent"]
    sup = YDEV.sum(0) > 0
    Co = d.ytr.T @ d.ytr
    dwords = np.array([len(x["text"].split()) for x in d.dev_docs])
    Yb = YDEV.astype(bool)

    def unseen_pairs(B):
        """Share of predicted label pairs that never co-occur in train.

        Args:
            B (np.ndarray): Boolean predictions.

        Returns:
            float: Share in [0, 1].
        """
        Bf = B.astype(np.float32)
        M = Bf.T @ Bf
        np.fill_diagonal(M, 0)
        return float(M[Co == 0].sum() / max(M.sum(), 1))

    SUMROWS, PD, BDm, BTm = [], {}, {}, {}

    def audit_exp(e, detail):
        """Audit one experiment (summary row; detailed print when ``detail``).

        Args:
            e (str): Experiment id.
            detail (bool): Print the full per-experiment report.

        Returns:
            None
        """
        Pd, Pt = sigmoid(ctx.zd[e] / ctx.temp[e]), sigmoid(ctx.zt[e] / ctx.temp[e])
        g, _ = tune_global(Pd, YDEV)
        Bd, Bt = decide(Pd, g), decide(Pt, g)
        PD[e], BDm[e], BTm[e] = Pd, Bd, Bt
        tp, fp, fn = _counts(Bd, YDEV)
        P_ = tp / np.maximum(tp + fp, 1)
        R_ = tp / np.maximum(tp + fn, 1)
        F_ = f1_labels(Bd, YDEV)
        ap = np.array([average_precision_score(YDEV[:, j], Pd[:, j]) if sup[j] else np.nan for j in range(L)])
        au = np.array([roc_auc_score(YDEV[:, j], Pd[:, j]) if sup[j] and YDEV[:, j].min() < 1 else np.nan for j in range(L)])
        orc = np.stack([f1_labels(Pd >= t, YDEV) for t in GRID]).max(0)
        kt = YDEV.sum(1).astype(int)
        Bk = _topk_decode(Pd, kt)
        sp_ = Pd.sum(1)
        sc_best = max(((c, macro_f1(_topk_decode(Pd, np.maximum(1, np.round(c * sp_)).astype(int)), YDEV)) for c in (0.7, 0.85, 1.0, 1.15, 1.3)), key=lambda x: x[1])
        cap = {kmax: macro_f1(decide(Pd, g) & _topk_decode(Pd, np.full(len(Pd), kmax)), YDEV) for kmax in (6, 8, 10)}
        emp = ~Bd.any(1)
        SUMROWS.append(dict(exp=e, thr=g, macroF1=F_[sup].mean(), macroP=P_[sup].mean(), macroR=R_[sup].mean(), microF1=micro_f1(Bd, YDEV), ceil_oracle_thr=orc[sup].mean(),
                            ceil_oracle_k=macro_f1(Bk, YDEV), topk_sumprob=sc_best[1], macroAP=np.nanmean(ap), macroAUC=np.nanmean(au), pred_per_doc=Bd.sum(1).mean(),
                            true_per_doc=kt.mean(), empty_pct=100 * emp.mean(), unseen_pair_pct=100 * unseen_pairs(Bd), ECE=ece_flat(Pd, YDEV)))
        if not detail:
            return
        print(f"\n{'='*20} [{e}] global thr {g} {'='*20}\nceilings: oracle per-label thresholds {orc[sup].mean():.4f} | oracle cardinality top-k {macro_f1(Bk, YDEV):.4f} (micro {micro_f1(Bk, YDEV):.4f}) | top-k with k=round({sc_best[0]}·Σp) {sc_best[1]:.4f} | threshold+cap(kmax 6/8/10): {', '.join(f'{v:.4f}' for v in cap.values())}")
        print("by label-frequency quartile (labels with dev support):")
        print(pd.DataFrame([dict(bucket=BN[b], labels=int(((BK == b) & sup).sum()), avg_train_freq=float(TRAIN_FREQ[(BK == b) & sup].mean()), F1=F_[(BK == b) & sup].mean(), P=P_[(BK == b) & sup].mean(),
                                 R=R_[(BK == b) & sup].mean(), AP=np.nanmean(ap[(BK == b) & sup]), pred_over_true=Bd[:, (BK == b)].sum() / max(YDEV[:, (BK == b)].sum(), 1)) for b in range(4)]).round(3).to_string(index=False))
        ratio = Bd.sum(0) / np.maximum(YDEV.sum(0), 1)
        ok = sup & (YDEV.sum(0) >= 20)
        print("most OVER-predicted (pred/true, F1, train n):", [(LABELS[j], round(float(ratio[j]), 1), round(float(F_[j]), 2), int(TRAIN_FREQ[j])) for j in np.argsort(-np.where(ok, ratio, -1))[:6]])
        print("most UNDER-predicted                        :", [(LABELS[j], round(float(ratio[j]), 2), round(float(F_[j]), 2), int(TRAIN_FREQ[j])) for j in np.argsort(np.where(ok, ratio, 9))[:6]])
        print("worst F1 labels (F1, P, R, train n)         :", [(LABELS[j], round(float(F_[j]), 2), round(float(P_[j]), 2), round(float(R_[j]), 2), int(TRAIN_FREQ[j])) for j in np.argsort(np.where(sup, F_, 9))[:8]])
        print("best F1 labels                              :", [(LABELS[j], round(float(F_[j]), 2), int(TRAIN_FREQ[j])) for j in np.argsort(-np.where(sup, F_, -1))[:6]])
        print(f"cardinality: Spearman(Σp, true #labels) {spearmanr(sp_, kt)[0]:.2f} | pred vs true #labels per doc:", {k: (int((Bd.sum(1)[kt == k]).mean().round()) if (kt == k).any() else None) for k in (1, 2, 3, 4, 5, 6, 8)})
        print(f"empty predictions: {100*emp.mean():.1f}% of dev docs | their true labels/doc {kt[emp].mean() if emp.any() else 0:.2f} | median max-prob {np.median(Pd[emp].max(1)) if emp.any() else float('nan'):.3f} (thr {g}) | share of all true positives living in empty docs {YDEV[emp].sum()/YDEV.sum():.1%}")
        tpd = (Bd & Yb).sum(1)
        docF = 2 * tpd / np.maximum(Bd.sum(1) + kt, 1)
        qd = np.quantile(dwords, np.linspace(0, 1, 6))
        bd = np.clip(np.digitize(dwords, qd[1:-1]), 0, 4)
        print("mean per-doc F1 by doc-length quintile (short->long):", [round(float(docF[bd == b].mean()), 3) for b in range(5)], "| by #true labels (1,2,3,4,5-6,7+):",
              [round(float(docF[m].mean()), 3) if m.any() else None for m in [kt == 1, kt == 2, kt == 3, kt == 4, (kt >= 5) & (kt <= 6), kt >= 7]])
        print("reliability (prob bin, n, mean predicted, observed):", _ece_bins(Pd, YDEV))
        print("ECE by quartile:", [round(ece_flat(Pd[:, BK == b], YDEV[:, BK == b]), 4) for b in range(4)])
        print(f"implausible label pairs (never co-occur in train): predicted {100*unseen_pairs(Bd):.1f}% of predicted pairs vs true {100*unseen_pairs(Yb):.1f}% of true pairs")
        FN, FPm = Yb & ~Bd, Bd & ~Yb
        M = FN.astype(np.float32).T @ FPm.astype(np.float32)
        np.fill_diagonal(M, 0)
        top = np.dstack(np.unravel_index(np.argsort(-M.ravel())[:8], M.shape))[0]
        print("top confusions (missed label -> wrongly predicted instead, #docs):", [(LABELS[i], LABELS[j], int(M[i, j])) for i, j in top if M[i, j] > 0])
        print("most FALSE POSITIVES:", [(LABELS[j], int(FPm[:, j].sum())) for j in np.argsort(-FPm.sum(0))[:6]], "| most MISSED:", [(LABELS[j], int(FN[:, j].sum())) for j in np.argsort(-FN.sum(0))[:6]])

    DETAIL = list(ctx.diag.sort_values("dev_macroF1_oof", ascending=False).index[:2])
    for e in ctx.avail:
        audit_exp(e, e in DETAIL)
    print("\n" + "=" * 20, "SUMMARY across experiments", "=" * 20)
    SUMM = pd.DataFrame(SUMROWS).set_index("exp")
    print(SUMM.round(4).to_string())
    print("\n== agreement between models ==")
    if len(ctx.avail) > 1:
        ks = list(ctx.avail)
        jd = lambda A, B_: float(np.mean((A & B_).sum(1) / np.maximum((A | B_).sum(1), 1)))
        print("mean per-doc Jaccard of predicted sets, DEV :")
        print(pd.DataFrame([[jd(BDm[a], BDm[b]) for b in ks] for a in ks], index=ks, columns=ks).round(3).to_string())
        print("TEST:")
        print(pd.DataFrame([[jd(BTm[a], BTm[b]) for b in ks] for a in ks], index=ks, columns=ks).round(3).to_string())
        anyb, allb = np.any([BDm[k] for k in ks], 0), np.all([BDm[k] for k in ks], 0)
        tpos = Yb.sum()
        print(f"true positives found by ANY model {(anyb & Yb).sum()/tpos:.3f} | by ALL models {(allb & Yb).sum()/tpos:.3f} | best single recall {max((BDm[k] & Yb).sum() for k in ks)/tpos:.3f} | false positives shared by ALL {(allb & ~Yb).sum()/max((anyb & ~Yb).sum(),1):.3f} of those made by ANY")
        print("macro-F1 of majority vote (>= half of the models):", round(macro_f1(np.mean([BDm[k] for k in ks], 0) >= 0.5, YDEV), 4), "| union:", round(macro_f1(anyb, YDEV), 4), "| intersection:", round(macro_f1(allb, YDEV), 4))
    print("\n== dev vs TEST shift (same thresholds) ==")
    for e in ctx.avail:
        Pd, Pt = PD[e], sigmoid(ctx.zt[e] / ctx.temp[e])
        rate_d, rate_t = BDm[e].mean(0), BTm[e].mean(0)
        r = rate_t / np.maximum(rate_d, 1e-6)
        sh = np.argsort(-np.abs(np.log(np.maximum(r, 1e-6))))[:4]
        print(f"[{e}] Σp/doc dev {Pd.sum(1).mean():.2f} test {Pt.sum(1).mean():.2f} (KS {ks_2samp(Pd.sum(1), Pt.sum(1))[0]:.3f}) | max-prob median dev {np.median(Pd.max(1)):.3f} test {np.median(Pt.max(1)):.3f} | per-label rate corr dev/test {np.corrcoef(rate_d, rate_t)[0,1]:.3f} | biggest rate changes (test/dev):", [(LABELS[j], round(float(r[j]), 2)) for j in sh])
    return dict(PD=PD, BDm=BDm, BTm=BTm, SUMM=SUMM)


CITE_RE = re.compile(r"\b(?i:(sections?|sec|s|articles?|art|rule|order|u/s))\s*\.?\s*(\d{1,4}[A-Za-z]?)\b(?:\s*(?:of\s+(?:the\s+)?|,\s*)?([A-Z]{2,6})\b)?")


def cite_keys(text):
    """Citation keys such as ``s302`` or ``s302|IPC`` mined from the first 300k characters.

    Args:
        text (str): Document text.

    Returns:
        set: Citation keys.
    """
    ks = set()
    for kind, num, act in CITE_RE.findall(text[:300000]):
        k = kind.lower()
        k = "a" if k.startswith("art") else "r" if k == "rule" else "o" if k == "order" else "s"
        b = f"{k}{num.lower()}"
        ks.add(b)
        if act:
            ks.add(b + "|" + act)
    return ks


def audit_citations(ctx, pred):
    """AUDIT 3: mine citation->statute rules from TRAIN only and test whether ``model OR rules`` helps on dev.

    Args:
        ctx (Context): Pipeline context.
        pred (dict): Output of :func:`audit_predictions` (``BDm``, ``BTm``).

    Returns:
        None
    """
    d = ctx.data
    YDEV, YTR, LABELS, L = d.ydev, d.ytr, d.labels, d.n_labels
    t0 = time.time()
    kl_tr, kl_dv, kl_te = ([cite_keys(x["text"]) for x in docs] for docs in (d.train_docs, d.dev_docs, d.test_docs))
    print(f"citation keys extracted in {time.time()-t0:.0f}s | docs with >=1 key: train {np.mean([len(k) > 0 for k in kl_tr]):.1%}, dev {np.mean([len(k) > 0 for k in kl_dv]):.1%}, test {np.mean([len(k) > 0 for k in kl_te]):.1%} | median keys/doc {np.median([len(k) for k in kl_tr]):.0f}")
    vocab, rows_, cols_ = {}, [], []
    for i, ks in enumerate(kl_tr):
        for k in ks:
            rows_.append(i)
            cols_.append(vocab.setdefault(k, len(vocab)))
    X = sp.csr_matrix((np.ones(len(rows_), np.float32), (rows_, cols_)), shape=(len(d.train_docs), max(len(vocab), 1)))
    df_ = np.asarray(X.sum(0)).ravel()
    freq = np.where(df_ >= 5)[0]
    Cm = np.asarray((X[:, freq].T @ sp.csr_matrix(YTR)).todense(), np.float32)
    prec = Cm / df_[freq][:, None]
    inv = {j: k for k, j in vocab.items()}
    kidx = {inv[freq[n]]: n for n in range(len(freq))}

    def rule_scores(kls, min_hits=3):
        """Max train precision of the citation rules firing in each document.

        Args:
            kls (List[set]): Citation keys per document.
            min_hits (int): Minimum train co-occurrences for a rule to count.

        Returns:
            np.ndarray: Rule score matrix (n_docs, L).
        """
        R = np.where((Cm >= min_hits), prec, 0)
        out = np.zeros((len(kls), L), np.float32)
        for i, ks in enumerate(kls):
            ix = [kidx[k] for k in ks if k in kidx]
            if ix:
                out[i] = R[ix].max(0)
        return out

    Rd = rule_scores(kl_dv)
    Yd = YDEV.astype(bool)
    base_e = max(ctx.avail, key=lambda e: ctx.diag.loc[e, "dev_macroF1_oof"])
    Bd0 = pred["BDm"][base_e]
    print(f"{len(vocab)} distinct keys, {len(freq)} frequent (df>=5); rule precision table is built on TRAIN only")
    for th in (0.5, 0.7, 0.8, 0.9):
        Br = Rd >= th
        tp_ = (Br & Yd).sum()
        print(f"rules alone @ precision>={th}: predicted positives/doc {Br.sum(1).mean():.2f} | precision {tp_/max(Br.sum(),1):.3f} | recall of dev positives {tp_/Yd.sum():.3f} | macro-F1 {macro_f1(Br, YDEV):.4f}")
    print(f"\nbase model {base_e} (global thr): macro-F1 {macro_f1(Bd0, YDEV):.4f} micro {micro_f1(Bd0, YDEV):.4f}")
    for th in (0.6, 0.8, 0.9):
        Bu = Bd0 | (Rd >= th)
        added = (Rd >= th) & ~Bd0
        print(f"model OR rules(>={th}): macro-F1 {macro_f1(Bu, YDEV):.4f} micro {micro_f1(Bu, YDEV):.4f} | added positives {added.sum()} of which correct {(added & Yd).sum()} ({(added & Yd).sum()/max(added.sum(),1):.1%})")
    Bt_rule = rule_scores(kl_te) >= 0.8
    print(f"on TEST: rules(>=0.8) would add {int((Bt_rule & ~pred['BTm'][base_e]).sum())} positives over {base_e} (dev equivalent: {int(((Rd >= 0.8) & ~Bd0).sum())} on {len(d.dev_docs)} docs)")
    print("labels best covered by high-precision rules (label, hits, dev positives):", [(LABELS[j], int(((Rd[:, j] >= 0.8) & Yd[:, j]).sum()), int(Yd[:, j].sum())) for j in np.argsort(-((Rd >= 0.8) & Yd).sum(0))[:8]])


def audit_zips(ctx):
    """AUDIT 4: labels per document, empty predictions, drift from the train prior and pairwise agreement of all submission zips.

    Args:
        ctx (Context): Pipeline context.

    Returns:
        pd.DataFrame: Per-zip statistics.
    """
    import os
    d = ctx.data
    Z = {}
    for zp in sorted(glob.glob(f"{ctx.cfg.submit_dir}/*.zip")):
        with zipfile.ZipFile(zp) as zf:
            pr = json.loads(zf.read("prediction.json"))
        M = np.zeros((len(pr), d.n_labels), bool)
        for i, x in enumerate(pr):
            for v in x["pred"]:
                M[i, d.l2i[v]] = True
        Z[os.path.basename(zp)[:60]] = M
    prior = d.ytr.mean(0)
    rows = []
    for n, M in Z.items():
        c = M.sum(1)
        rows.append(dict(zip=n, labels_per_doc=c.mean(), median=np.median(c), empty_pct=100 * (c == 0).mean(), gt8_pct=100 * (c > 8).mean(), labels_never_predicted=int((M.sum(0) == 0).sum()),
                         rate_L1_vs_train_prior=float(np.abs(M.mean(0) - prior).sum()), rate_corr_with_prior=float(np.corrcoef(M.mean(0), prior)[0, 1])))
    out = pd.DataFrame(rows).set_index("zip")
    print(out.round(3).to_string())
    ks = list(Z)[:10]
    if len(ks) > 1:
        jd = lambda A, B_: float(np.mean((A & B_).sum(1) / np.maximum((A | B_).sum(1), 1)))
        print("\nmean per-doc Jaccard between zips (first 10):")
        print(pd.DataFrame([[jd(Z[a], Z[b]) for b in ks] for a in ks], index=[k[:28] for k in ks], columns=[str(i) for i in range(len(ks))]).round(2).to_string())
    if ctx.dup_test:
        print("\ntest documents with an exact twin in train/dev (label-lookup candidates):", {k: len(v) for k, v in ctx.dup_test.items()})
        if ctx.dup_test["train"] and ks:
            Htr, te_h = ctx.audit_twins
            tw = [(i, d.train_docs[Htr[te_h[i]][0]]["labels"]) for i in ctx.dup_test["train"]]
            M = Z[ks[0]]
            print(f"for those {len(tw)} test docs: zip '{ks[0][:40]}' predicts on average {np.mean([M[i].sum() for i, _ in tw]):.2f} labels vs the twin's {np.mean([len(l) for _, l in tw]):.2f}; "
                  f"label-set Jaccard with the twin {np.mean([len(set(to_pred(ctx, M[[i]])[0]) & set(l)) / max(len(set(to_pred(ctx, M[[i]])[0]) | set(l)), 1) for i, l in tw]):.3f}")
    return out
