"""Read-only audits: where is the remaining headroom? (AUDIT v3, AUDIT v4 and the precision audit).

None of these functions trains a model that is used for submissions.
"""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401  (kept for parity with the notebook)
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pcr.data.preprocessing import normt
from pcr.rerank.common import apply_rules, csls, dual, post, sinkhorn
from pcr.retrieval.metrics import f1_at_k, gold_matrix
from pcr.utils import fill_nan, gstd, zs

pd.set_option("display.width", 230, "display.max_columns", 40)


def audit_v3(ctx, v4, exps, signals_dev):
    """AUDIT v3: precision by rank, location of missed gold, LR coefficients, feature-group ablation, signal complementarity,
    cross-split priors and false-positive anatomy of the v4 out-of-fold reranker.

    Args:
        ctx (PCRContext): Pipeline context.
        v4 (dict): Output of ``run_ltr_v4`` (``Xd, topd, yd, names, bestC, OO``).
        exps (List[str]): Dense experiment ids.
        signals_dev (Dict[str, np.ndarray]): Dev signals (``signals(ctx, 'dev', exps)``) used for the complementarity study.

    Returns:
        None
    """
    spl, cfg, K = ctx.spl, ctx.cfg, ctx.cfg.k
    Xd, topd, yd, names, bestC, OO = v4["Xd"], v4["topd"], v4["yd"], v4["names"], v4["bestC"], v4["OO"]
    SD = spl["dev"]
    cids = np.array(SD["cids"])
    qids = SD["qids"]
    nq, nc = len(qids), len(cids)
    G = gold_matrix("dev", spl)
    ng = G.sum(1)
    ceil = np.minimum(ng, K)
    FIN = post(fill_nan(OO[0]["blend"]), ctx, "dev", "csls", 4, 0.75, True)
    order = np.argsort(-FIN, 1)
    hit = np.take_along_axis(G, order[:, :K], 1)
    tpq = hit.sum(1)
    print("=" * 14, "1. FINAL OOF (blend+CSLS+rules)", {k: round(v, 4) for k, v in f1_at_k(FIN, spl, "dev", K).items()})
    print("precision by rank 1..7:", hit.mean(0).round(3), "| P@7", round(hit.mean(), 3), "| ceiling P@7", round(ceil.sum() / (K * nq), 3),
          "| queries 0 TP:", round((tpq == 0).mean(), 3), "| at ceiling:", round((tpq == ceil).mean(), 3))
    print(pd.DataFrame(dict(b=pd.cut(ng, [0, 1, 3, 5, 7, 100]), prec=hit.mean(1), ceil=ceil / K, tp=tpq)).groupby("b", observed=True).agg(
        prec=("prec", "mean"), ceil=("ceil", "mean"), tp=("tp", "mean"), n=("prec", "size")).round(3).to_string())
    print("\n" + "=" * 14, "2. WHERE ARE THE MISSED GOLD? (rank of gold under final scores)")
    rk = np.argsort(order, 1)[G]
    bins = [-1, 6, 19, 49, 99, 10 ** 6]
    print(pd.Series(pd.cut(rk, bins, labels=["top7", "8-20", "21-50", "51-100", ">100"])).value_counts(normalize=True).sort_index().round(3).to_string())
    print("\n" + "=" * 14, "3. LR COEFFICIENTS (full dev fit)")
    Fz = Xd.shape[2]
    sc_ = StandardScaler().fit(Xd.reshape(-1, Fz))
    lr_ = LogisticRegression(C=bestC, max_iter=5000, random_state=0).fit(sc_.transform(Xd.reshape(-1, Fz)), yd.reshape(-1))
    print(pd.Series(lr_.coef_[0], index=names).sort_values(key=abs, ascending=False).round(3).to_string())
    print("\n" + "=" * 14, "4. FEATURE-GROUP ABLATION (LR, OOF seed0, raw | +CSLS k3 a.75)")
    GRP = {"new lexical L_*": lambda n: n.startswith("L_"), "dense E1-E4": lambda n: n in ("E1", "E2", "E3", "E4"), "bm25": lambda n: n == "bm25",
           "tfidf(bigram_q)": lambda n: n == "tfidf", "rules cite/fwd/rev/future/known": lambda n: n in ("cite", "fwd", "rev", "future", "known"),
           "year soft/ydiff": lambda n: n in ("soft_year", "ydiff")}

    def abl(keep):
        """OOF F1 (raw and with CSLS) of an LR trained on a subset of the features.

        Args:
            keep (List[int]): Feature indices to keep.

        Returns:
            Tuple[float, float]: ``(raw_f1, csls_f1)``.
        """
        S = np.full((nq, nc), np.nan, np.float32)
        for tr, va in KFold(5, shuffle=True, random_state=0).split(np.arange(nq)):
            Xa = Xd[:, :, keep]
            F_ = len(keep)
            s = StandardScaler().fit(Xa[tr].reshape(-1, F_))
            m = LogisticRegression(C=bestC, max_iter=5000, random_state=0).fit(s.transform(Xa[tr].reshape(-1, F_)), yd[tr].reshape(-1))
            for i in va:
                S[i, topd[i]] = m.decision_function(s.transform(Xa[i]))
        L = fill_nan(S)
        return round(f1_at_k(L, spl, "dev", K)["f1"], 4), round(f1_at_k(csls(L, 3, 0.75), spl, "dev", K)["f1"], 4)

    print("all features:", abl(list(range(len(names)))))
    for g, fn in GRP.items():
        print(f"drop {g:34s}", abl([i for i, n in enumerate(names) if not fn(n)]))
    print("\n" + "=" * 14, "5. SIGNAL COMPLEMENTARITY (TP@7 over dev, gold links =", int(G.sum()), ")")
    sg = {k: v for k, v in signals_dev.items() if not k.endswith("_ms")}
    sg.update({"L_" + k: v["dev"] for k, v in ctx.lex2.items()})
    tops = {}
    for k, v in sg.items():
        o = np.argsort(-v, 1)[:, :K]
        tops[k] = o
        print(f"  {k:14s} TP@7={int(np.take_along_axis(G, o, 1).sum())}")
    un = np.zeros_like(G)
    for o in tops.values():
        for i in range(nq):
            un[i, o[i]] = True
    print("  reranker TP@7 =", int(hit.sum()), "| oracle union of all signals' top-7 =", int((un & G).sum()), "| gold within reranker top-20 =", int((rk < 20).sum()), "| top-50 =", int((rk < 50).sum()))
    print("\n" + "=" * 14, "6. CROSS-SPLIT 'CLAIMED CANDIDATE' PRIOR")
    raw = ctx.raw
    tr_pool = {d["id"] for d in raw["train_c"]}
    tr_gold = {g for d in raw["train"] for g in d["gold"]}
    tr_q = {d["id"] for d in raw["train"]}
    dv_gold = {g for d in raw["dev"] for g in d["gold"]}
    isg = G.any(0)
    for nm, S_ in (("in train pool", tr_pool), ("gold of a TRAIN query", tr_gold), ("is a train query", tr_q)):
        m = np.array([c in S_ for c in cids])
        print(f"  dev cands {nm:24s}: {m.mean()*100:5.1f}% of pool | gold-for-some-dev-query rate: in-set {isg[m].mean():.3f} vs not-in-set {isg[~m].mean():.3f}")
    fp_ids = [cids[order[i, r]] for i in range(nq) for r in range(K) if not hit[i, r]]
    print(f"  top-7 false positives that are gold of a TRAIN query: {np.mean([c in tr_gold for c in fp_ids]):.3f} | overall dev pool: {np.mean([c in tr_gold for c in cids]):.3f}")
    tcs = spl["test"]["cids"]
    print(f"  TEST pool coverage: in train pool {np.mean([c in tr_pool for c in tcs]):.3f} | gold of a train query {np.mean([c in tr_gold for c in tcs]):.3f} | gold of a dev query {np.mean([c in dv_gold for c in tcs]):.3f}")
    print("\n" + "=" * 14, "7. FP ANATOMY AFTER CSLS")
    oj = order[:, :K]
    fpm = ~hit
    owners = G.sum(0)
    indeg = np.bincount(oj.ravel(), minlength=nc)
    print(f"  FPs={int(fpm.sum())} | FP is gold for another dev query: {np.mean(owners[oj][fpm] > 0):.3f} | distinct retrieved {int((indeg>0).sum())}/{nc} | max in-degree {indeg.max()} | "
          f"never retrieved but gold for someone: {int(((indeg==0)&(owners>0)).sum())}")
    ql = np.array([len(SD["qtok"][q]) for q in qids])
    print(pd.DataFrame(dict(b=pd.qcut(ql, 4), f1=2 * tpq / (K + ng))).groupby("b", observed=True).f1.agg(["mean", "count"]).round(3).T.to_string())


def audit_v4(ctx, B, state_has_meta=True):
    """AUDIT v4: which NEW information (case names, act / section, proper nouns, years, passage MaxSim) adds on top of stage B?

    Args:
        ctx (PCRContext): Pipeline context.
        B (np.ndarray): NaN-filled stage-B dev score matrix (``fill_nan(two_stage.Bz)``).
        state_has_meta (bool): Unused flag kept for API stability.

    Returns:
        pd.DataFrame: Standalone F1, AUC inside top-30 and additive effect of each new signal.
    """
    import time
    spl, cfg, K = ctx.spl, ctx.cfg, ctx.cfg.k
    MC = cfg.max_chars
    t0 = time.time()
    S_ = spl["dev"]
    qids_, cids_ = S_["qids"], S_["cids"]
    nq_, nc_ = len(qids_), len(cids_)
    Gd_ = gold_matrix("dev", spl)
    QT = [S_["qtext"][q] for q in qids_]
    CT = [S_["ctext"][c] for c in cids_]
    CASE = re.compile(r"\b([A-Z][A-Za-z.&'\-]{2,}(?:\s+[A-Z][A-Za-z.&'\-]{2,}){0,3})\s+(?:v\.?|vs\.?|versus)\s+([A-Z][A-Za-z.&'\-]{2,}(?:\s+[A-Z][A-Za-z.&'\-]{2,}){0,3})")
    ACTSEC = re.compile(r"[Ss]ections?\s+(\d+[A-Za-z]?)(?:\s*\(\w+\))*[^.;]{0,80}?\b(?:of|under)\s+the\s+((?:[A-Z][A-Za-z]+\s+){1,6}(?:Act|Code))")
    ACTN = re.compile(r"\b((?:[A-Z][A-Za-z]+\s+){1,5}(?:Act|Code))\b")
    CAPS = re.compile(r"\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,}){1,3}\b")
    YRS = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")
    FN = {"case_names": lambda t: ["|".join(sorted((normt(a)[:30], normt(b)[:30]))) for a, b in CASE.findall(t[:MC])] or ["_"],
          "act_section": lambda t: [f"{normt(a)}|{s}" for s, a in ACTSEC.findall(t[:MC])] or ["_"],
          "act_names": lambda t: [normt(a) for a in ACTN.findall(t[:MC])] or ["_"],
          "proper_nouns": lambda t: [normt(a) for a in CAPS.findall(t[:MC])] or ["_"],
          "years": lambda t: YRS.findall(t[:MC]) or ["_"]}

    def tok_sim(fn, max_df=0.3):
        """TF-IDF similarity over extracted tokens and the mean token counts per document.

        Args:
            fn (Callable[[str], List[str]]): Token extractor.
            max_df (float): Maximum document-frequency ratio.

        Returns:
            Tuple[np.ndarray, float, float]: ``(similarity, mean_cand_tokens, mean_query_tokens)``.
        """
        a = [fn(t) for t in QT]
        b = [fn(t) for t in CT]
        v = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, min_df=1, max_df=max_df, dtype=np.float32).fit(a + b)
        return (v.transform(a) @ v.transform(b).T).toarray().astype(np.float32), float(np.mean([sum(x != "_" for x in s) for s in b])), float(np.mean([sum(x != "_" for x in s) for s in a]))

    NEW = {}
    for k, fn in FN.items():
        NEW[k], cm, qm = tok_sim(fn, 0.5 if k == "years" else 0.3)
        print(f"  {k:13s} tokens/doc  query={qm:.1f}  cand={cm:.1f}  ({time.time()-t0:.0f}s)")

    def wins(t, w=250):
        """Split a text into 250-word windows (at most 40).

        Args:
            t (str): Text.
            w (int): Window size in words.

        Returns:
            List[str]: Windows.
        """
        ws = t[:MC].split()
        return [" ".join(ws[i:i + w]) for i in range(0, max(len(ws), 1), w)][:40]

    hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), stop_words="english", alternate_sign=False, norm=None, dtype=np.float32)
    qw, qo, cw, co = [], [], [], []
    for i, t in enumerate(QT):
        w_ = wins(t)
        qw += w_
        qo += [i] * len(w_)
    for j, t in enumerate(CT):
        w_ = wins(t)
        co.append(len(cw))
        cw += w_
    Cw_raw = hv.transform(cw)
    tt = TfidfTransformer(sublinear_tf=True).fit(Cw_raw)
    Cw = tt.transform(Cw_raw).T.tocsc()
    Qw = tt.transform(hv.transform(qw)).tocsr()
    qo = np.array(qo)
    co = np.array(co)
    MX = np.full((nq_, nc_), -1, np.float32)
    SM = np.zeros((nq_, nc_), np.float32)
    cnt = np.bincount(qo, minlength=nq_).astype(np.float32)
    for s in range(0, Qw.shape[0], 400):
        blk = (Qw[s:s + 400] @ Cw).toarray()
        R = np.maximum.reduceat(blk, co, axis=1)
        idx = qo[s:s + 400]
        np.maximum.at(MX, idx, R)
        np.add.at(SM, idx, R)
    NEW["maxsim_win"] = MX
    NEW["coverage_win"] = SM / np.maximum(cnt, 1)[:, None]
    print(f"  windows done ({time.time()-t0:.0f}s)")
    base = f1_at_k(B, spl, "dev", K)["f1"]

    def auc_top(F, Bm, m=30):
        """Mean AUC of a feature inside the top-m of the base ranking.

        Args:
            F (np.ndarray): Feature matrix.
            Bm (np.ndarray): Base score matrix defining the top-m.
            m (int): Depth.

        Returns:
            float: Mean per-query AUC (queries with both classes only).
        """
        top = np.argsort(-Bm, 1)[:, :m]
        out = []
        for i in range(nq_):
            y = Gd_[i, top[i]]
            ny = y.sum()
            if 0 < ny < m:
                r = rankdata(F[i, top[i]])
                out.append((r[y].sum() - ny * (ny + 1) / 2) / (ny * (m - ny)))
        return float(np.mean(out))

    rows = [dict(feat="[ref] stage-B score", standalone=round(base, 4), auc_top30=round(auc_top(B, B), 3))]
    for k, F in NEW.items():
        z_ = zs(F)
        r = dict(feat=k, standalone=round(f1_at_k(F, spl, "dev", K)["f1"], 4), auc_top30=round(auc_top(F, B), 3))
        for w in (0.25, 0.5, 1.0):
            r[f"B+{w}*z"] = round(f1_at_k(B + w * z_, spl, "dev", K)["f1"], 4)
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"\nbaseline stage-B dev F1 = {base:.4f}")
    print(df.to_string(index=False))
    return df


def precision_audit(ctx, tuned_cfg, tuner, S_test=None):
    """Where does precision go? Pool / gold structure, tuned-LTR OOF diagnostics, hubs, post-processing simulations and rule diagnostics.

    Args:
        ctx (PCRContext): Pipeline context.
        tuned_cfg (dict): Tuned LTR config from ``LTRTuner.run()['cfg']``.
        tuner (LTRTuner): The tuner object (provides ``build2``).
        S_test (Optional[np.ndarray]): Latest test score matrix for the hub-structure check.

    Returns:
        None
    """
    spl, K = ctx.spl, ctx.cfg.k
    raw, meta = ctx.raw, ctx.meta
    SD = spl["dev"]
    cids = np.array(SD["cids"])
    qids = SD["qids"]
    nq, nc = len(qids), len(cids)
    G = gold_matrix("dev", spl)
    print("=" * 16, "1. POOL / GOLD STRUCTURE")
    for s in ("train", "dev"):
        Q, Cc = raw[s], raw[s + "_c"]
        qset = {d["id"] for d in Q}
        pool = {d["id"] for d in Cc}
        links = [g for d in Q for g in d["gold"]]
        cnt = Counter(links)
        uni = set(links)
        ql = sum(g in qset for g in links)
        print(f"[{s}] queries={len(Q)} pool={len(pool)} (non-query docs {len(pool - qset)}) | gold links={len(links)} unique gold docs={len(uni)} | "
              f"links pointing at OTHER QUERIES={ql} ({100 * ql / len(links):.2f}%) | links outside pool={sum(g not in pool for g in links)}")
        print(f"      non-query pool docs that are gold for nobody={len((pool - qset) - uni)} | gold multiplicity {dict(sorted(Counter(cnt.values()).items()))} mean={np.mean(list(cnt.values())):.2f}")
    print("NOTE: every query id was removed from the pool -> gold links pointing at another query can never be retrieved.")
    print("\n" + "=" * 16, "2. CURRENT RERANKER (out-of-fold, tuned cfg)", tuned_cfg)
    X, top, names = tuner.build2("dev", tuned_cfg["R"], tuned_cfg["bw"], tuned_cfg["drop"])
    from pcr.rerank.ltr_dev import make_ranker
    y = np.array([[cids[j] in SD["gold"][q] for j in top[i]] for i, q in enumerate(qids)]).astype(int)
    F_ = X.shape[2]
    SCV = np.full((nq, nc), -1e9, np.float32)
    for tr, va in KFold(5, shuffle=True, random_state=0).split(np.arange(nq)):
        sc = StandardScaler().fit(X[tr].reshape(-1, F_))
        m = make_ranker(tuned_cfg["kind"], tuned_cfg["C"]).fit(sc.transform(X[tr].reshape(-1, F_)), y[tr].reshape(-1))
        for i in va:
            SCV[i, top[i]] = m.decision_function(sc.transform(X[i]))
    print("dev OOF:", {k: round(v, 4) for k, v in f1_at_k(SCV, spl, "dev", K).items()})
    order = np.argsort(-SCV, 1)[:, :K]
    hit = np.take_along_axis(G, order, 1)
    ng = G.sum(1)
    tpq = hit.sum(1)
    ceil = np.minimum(ng, K)
    print("precision by rank position 1..7:", hit.mean(0).round(3), "| P@7", round(hit.mean(), 3), "| ceiling", round(ceil.sum() / (K * nq), 3),
          "| queries with 0 TP:", round((tpq == 0).mean(), 3), "| queries at ceiling:", round((tpq == ceil).mean(), 3))
    print("\n" + "=" * 16, "3. FALSE-POSITIVE ANATOMY")
    pi, pj, ph = np.repeat(np.arange(nq), K), order.ravel(), hit.ravel()
    owners = G.sum(0)
    indeg = np.bincount(pj, minlength=nc)
    CR = np.argsort(np.argsort(-SCV, 0), 0)
    fp = ~ph
    print(f"FPs={fp.sum()} of {len(ph)} | FP candidate is gold for ANOTHER dev query: {(owners[pj][fp] > 0).mean():.3f} | gold for nobody: {(owners[pj][fp] == 0).mean():.3f}")
    print(pd.DataFrame(dict(indeg=np.minimum(indeg[pj], 5), hit=ph)).groupby("indeg").hit.agg(["mean", "count"]).round(3).T.to_string())
    print(pd.DataFrame(dict(colrank=np.minimum(CR[pi, pj], 5), hit=ph)).groupby("colrank").hit.agg(["mean", "count"]).round(3).T.to_string())
    print("\n" + "=" * 16, "4. COLUMN-AWARE POST-PROCESSING (simulated on OOF dev scores)")
    L = np.where(SCV > -1e8, SCV, np.nan)
    L = np.nan_to_num(L, nan=np.nanmin(L) - 1.0)
    f = lambda M: {k: round(v, 4) for k, v in f1_at_k(M, spl, "dev", K).items()}
    res = [("raw", f(L))]
    for tau in (0.5, 1, 2):
        for lam in (0.25, 0.5, 1):
            res.append((f"dual-softmax tau={tau} lam={lam}", f(dual(L, tau, lam))))
    for tau in (0.5, 1, 2):
        res.append((f"sinkhorn tau={tau}", f(sinkhorn(L, G, tau))))
    for k in (3, 10):
        for a in (0.25, 0.5, 1.0):
            res.append((f"CSLS-col k={k} a={a}", f(csls(L, k, a))))
    for mrk in (1, 2, 3):
        res.append((f"keep only if query in candidate's top-{mrk} queries", f(np.where(CR < mrk, L, L - 50))))
    print(pd.DataFrame([dict(method=a, **b) for a, b in res]).sort_values("f1", ascending=False).head(14).to_string(index=False))
    print("\n" + "=" * 16, "5. RULES")
    rf = ctx.rf["dev"]
    fut, kn, cite = rf["future"] > 0, rf["known"] > 0, rf["cite"] > 0
    print(f"known-year pairs={kn.mean():.3f} | gold violating chronology={int(fut[G].sum())} | FPs in top-7 that are 'future'={int((fut[pi, pj] & ~ph).sum())}/{int((~ph).sum())}")
    print(f"citation hits: flagged pairs={int(cite.sum())} gold among them={int(cite[G].sum())} | flagged pairs inside current top-7={int(cite[pi, pj].sum())}")
    ex = ctx.ex["dev"]
    print(f"REVERSE-CITATION rule: fwd pairs={int(ex['fwd'].sum())} gold={int(ex['fwd'][G].sum())} | rev pairs={int(ex['rev'].sum())} gold={int(ex['rev'][G].sum())} "
          f"FPs in top-7={int(((ex['rev'] > 0)[pi, pj] & ~ph).sum())}")
    if S_test is not None:
        ot = np.argsort(-S_test, 1)[:, :K]
        id_t = np.bincount(ot.ravel(), minlength=S_test.shape[1])
        print("\n" + "=" * 16, "7. TEST-SIDE")
        print(f"test: distinct candidates retrieved={int((id_t > 0).sum())}/{S_test.shape[1]} | in-degree: mean={id_t[id_t > 0].mean():.2f} max={id_t.max()} | retrieved by >=3 queries={int((id_t >= 3).sum())}")
        print(f"dev : distinct retrieved={(indeg > 0).mean():.3f} | candidates retrieved by >=3 queries={int((indeg >= 3).sum())}")
