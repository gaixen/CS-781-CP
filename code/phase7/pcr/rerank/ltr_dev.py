"""Learning-to-rank rerankers fitted on the dev split (honest estimates via query-grouped K-fold CV).

* ``run_ltr_v1``  (section 10c): logistic regression over dense, MaxSim, BM25, TF-IDF and rule features.
* ``LTRTuner``    (section 10d): depth / pre-fusion / model / regularisation / feature-group tuning with repeated CV.
* ``run_ltr_v3``  : + CITATOR-INFO / soft-year features, CSLS and hard rules.
* ``run_ltr_v4``  : + lexical suite (FIX A), LR + HGB blend, CSLS / dual-softmax / rules.

Everything is deterministic (fixed solver / seeds, no shuffling in the grouped CV of v1).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from pcr.rerank.common import fit2, fit_lr, post, signals, stack_top, csls, apply_rules
from pcr.retrieval.metrics import f1_at_k, label_matrix
from pcr.submission import validate_submission, write_submission
from pcr.utils import fill_nan, gstd, zs


# =====================================================================================================
# 10c - first learning-to-rank reranker
# =====================================================================================================
def build_v1(ctx, exps, split):
    """Feature tensor of the first reranker: z-score + log-rank of every signal + rule features.

    Args:
        ctx (PCRContext): Pipeline context.
        exps (List[str]): Dense experiment ids.
        split (str): ``dev`` | ``test``.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]: ``(X, top, pre_fusion_scores, feature_names)`` where
        ``X`` has shape (n_queries, r_topk, n_features) and ``top`` holds the candidate column indices.
    """
    sg, rf = signals(ctx, split, exps), ctx.rf[split]
    dense = [k for k in sg if k not in ("bm25", "tfidf")]
    pre = sum(sg[k] for k in dense) / len(dense) + 0.5 * sg["bm25"]
    top = np.argsort(-pre, 1)[:, :ctx.cfg.r_topk]
    rk = {k: np.argsort(np.argsort(-v, 1), 1) for k, v in sg.items()}
    X = []
    for i in range(pre.shape[0]):
        j = top[i]
        X.append(np.stack([sg[k][i, j] for k in sg] + [-np.log1p(rk[k][i, j]) for k in sg] + [rf[k][i, j] for k in rf], 1))
    return np.stack(X), top, pre, list(sg) + ["rank_" + k for k in sg] + list(rf)


def run_ltr_v1(ctx, exps):
    """Fit the first LR reranker on dev (C chosen by 5-fold query-grouped CV) and write ``LTR_rerank_C*`` submission.

    Args:
        ctx (PCRContext): Pipeline context.
        exps (List[str]): Dense experiment ids (E1-E4 with MaxSim scores available).

    Returns:
        dict: ``{"best_C", "cv", "scores_test", "zip"}``.
    """
    cfg, spl = ctx.cfg, ctx.spl
    Xd, topd, pre_d, fn = build_v1(ctx, exps, "dev")
    cids_d = np.array(spl["dev"]["cids"])
    nqd = Xd.shape[0]
    y = label_matrix(topd, spl, "dev")
    print(f"features={Xd.shape[2]} | pre-fusion dev F1@{cfg.k}={f1_at_k(pre_d, spl, 'dev', cfg.k)['f1']:.4f} | oracle recall inside top-{cfg.r_topk}="
          f"{y.sum()/sum(len(spl['dev']['gold'][q]) for q in spl['dev']['qids']):.3f}")

    def cv_f1(C):
        """5-fold query-grouped CV F1@K for a regularisation strength.

        Args:
            C (float): Inverse regularisation strength.

        Returns:
            Dict[str, float]: ``f1``, ``p``, ``r`` of the out-of-fold scores.
        """
        Scv = np.full((nqd, len(cids_d)), -1e9, np.float32)
        idx = np.arange(nqd)
        for tr, va in GroupKFold(5).split(idx, groups=idx):
            sc, clf = fit_lr(Xd[tr], y[tr], C)
            for i in va:
                Scv[i, topd[i]] = clf.decision_function(sc.transform(Xd[i]))
        return f1_at_k(Scv, spl, "dev", cfg.k)

    CV = {C_: cv_f1(C_) for C_ in (0.01, 0.03, 0.1, 0.3, 1.0)}
    print(pd.DataFrame(CV).T.round(4).to_string())
    bestC = max(CV, key=lambda c: CV[c]["f1"])
    print("best C:", bestC, "-> CV F1@7 =", round(CV[bestC]["f1"], 4))
    sc, clf = fit_lr(Xd, y, bestC)
    print(pd.Series(clf.coef_[0], index=fn).sort_values(key=abs, ascending=False).head(12).round(3).to_string())
    Xt, topt, _, _ = build_v1(ctx, exps, "test")
    S_t = np.full((len(spl["test"]["qids"]), len(spl["test"]["cids"])), -1e9, np.float32)
    for i in range(Xt.shape[0]):
        S_t[i, topt[i]] = clf.decision_function(sc.transform(Xt[i]))
    zp = write_submission(f"LTR_rerank_C{bestC}", S_t, spl, cfg)
    validate_submission(zp, spl, cfg.k)
    return dict(best_C=bestC, cv=CV, scores_test=S_t, zip=zp)


# =====================================================================================================
# 10d - tuning (depth / pre-fusion / model / regularisation / feature ablation, repeated CV)
# =====================================================================================================
def make_ranker(kind, C):
    """Instantiate a pointwise classifier used by the tuner.

    Args:
        kind (str): ``lr_l2`` | ``lr_l2_bal`` | ``lr_l1`` | ``hgb``.
        C (float): Inverse regularisation strength (ignored by ``hgb``).

    Returns:
        sklearn estimator: Un-fitted classifier.
    """
    if kind == "lr_l2":
        return LogisticRegression(C=C, max_iter=5000, random_state=0)
    if kind == "lr_l2_bal":
        return LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=0)
    if kind == "lr_l1":
        return LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=5000, random_state=0)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


class LTRTuner:
    """Repeated-CV tuning of the first-stage LTR reranker (section 10d).

    Args:
        ctx (PCRContext): Pipeline context.
        exps (List[str]): Dense experiment ids.
    """

    def __init__(self, ctx, exps):
        self.ctx, self.exps = ctx, exps
        self.SG = {s: signals(ctx, s, exps) for s in ("dev", "test")}
        self.RK = {s: {k: np.argsort(np.argsort(-v, 1), 1) for k, v in self.SG[s].items()} for s in self.SG}
        self.CIDS = {s: np.array(ctx.spl[s]["cids"]) for s in self.SG}
        self.cfg = dict(R=150, bw=0.5, kind="lr_l2", C=0.1, drop=())
        self.log = []

    def build2(self, split, R, bw, drop=()):
        """Feature tensor for a reranking depth / pre-fusion BM25 weight / dropped feature groups.

        Args:
            split (str): ``dev`` | ``test``.
            R (int): Reranking depth.
            bw (float): BM25 weight in the pre-fusion.
            drop (Tuple[str, ...]): Feature groups to drop (substring match on feature names).

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: ``(X, top, feature_names)``.
        """
        sg, rk, rf = self.SG[split], self.RK[split], self.ctx.rf[split]
        dense = [k for k in sg if k not in ("bm25", "tfidf")]
        pre = sum(sg[k] for k in dense) / len(dense) + bw * sg["bm25"]
        top = np.argsort(-pre, 1)[:, :R]
        feats = {**sg, **{"rank_" + k: -np.log1p(v) for k, v in rk.items()}, **rf}
        names = [n for n in feats if not any(d in n for d in drop)]
        return stack_top(feats, names, top), top, names

    def labels(self, top):
        """Dev labels of the selected candidates.

        Args:
            top (np.ndarray): Candidate indices (n_queries, R).

        Returns:
            np.ndarray: Binary label matrix.
        """
        return label_matrix(top, self.ctx.spl, "dev")

    def cv_score(self, cfg, seeds=(0, 1, 2)):
        """Mean / std dev F1@K of repeated shuffled 5-fold CV for a configuration.

        Args:
            cfg (dict): Keys ``R, bw, kind, C, drop``.
            seeds (Iterable[int]): Fold-shuffling seeds.

        Returns:
            Tuple[float, float]: ``(mean_f1, std_f1)``.
        """
        spl, k = self.ctx.spl, self.ctx.cfg.k
        X, top, _ = self.build2("dev", cfg["R"], cfg["bw"], cfg["drop"])
        y = self.labels(top)
        nq, R, F = X.shape
        out = []
        for sd in seeds:
            Scv = np.full((nq, len(self.CIDS["dev"])), -1e9, np.float32)
            for tr, va in KFold(5, shuffle=True, random_state=sd).split(np.arange(nq)):
                sc = StandardScaler().fit(X[tr].reshape(-1, F))
                m = make_ranker(cfg["kind"], cfg["C"]).fit(sc.transform(X[tr].reshape(-1, F)), y[tr].reshape(-1))
                f = (lambda Z: m.decision_function(Z)) if hasattr(m, "decision_function") else (lambda Z: m.predict_proba(Z)[:, 1])
                for i in va:
                    Scv[i, top[i]] = f(sc.transform(X[i]))
            out.append(f1_at_k(Scv, spl, "dev", k)["f1"])
        return float(np.mean(out)), float(np.std(out))

    def stage(self, name, grid):
        """Evaluate a grid of config updates and accept the best one only if it gains > 0.2 F1 points.

        Args:
            name (str): Stage name (for logging).
            grid (List[dict]): Config updates to try.

        Returns:
            None
        """
        res = []
        for upd in grid:
            c = {**self.cfg, **upd}
            try:
                m, s = self.cv_score(c)
            except Exception as e:
                print("  skip", upd, e)
                continue
            res.append((m, s, upd))
            self.log.append((name, c, m, s))
        res.sort(key=lambda r: -r[0])
        print(f"\n[{name}]")
        for m, s, u in res[:6]:
            print(f"  {m:.4f} ± {s:.4f}  {u}")
        if res[0][0] > self.cv_score(self.cfg)[0] + 0.002:      # only accept clear gains (>0.2 pt) to avoid chasing noise
            self.cfg = {**self.cfg, **res[0][2]}
        print("  -> cfg:", self.cfg)

    def run(self):
        """Run the full tuning procedure, write the tuned-LTR submission and the top-15 csv.

        Args:
            None

        Returns:
            dict: ``{"cfg", "scores_test", "zip", "model"}`` with the tuned configuration and the test score matrix.
        """
        ctx = self.ctx
        cfg_c, spl = ctx.cfg, ctx.spl
        base = self.cv_score(self.cfg)
        print("baseline", self.cfg, "-> CV F1 %.4f ± %.4f" % base)
        self.log = [("baseline", dict(self.cfg), *base)]
        self.stage("depth + pre-fusion", [dict(R=R, bw=b) for R in (50, 100, 150, 250, 400) for b in (0.0, 0.5, 1.0)])
        self.stage("model + regularisation", [dict(kind=k, C=C) for k in ("lr_l2", "lr_l2_bal", "lr_l1") for C in (0.01, 0.03, 0.1, 0.3, 1.0)] + [dict(kind="hgb", C=1.0)])
        for g in ("rank_", "tfidf", "_ms", "cite", "name", "future", "gap", "known"):          # greedy feature-group ablation
            c = {**self.cfg, "drop": tuple(sorted(set(self.cfg["drop"]) | {g}))}
            m, s = self.cv_score(c)
            self.log.append((f"drop {g}", c, m, s))
            keep = m >= self.cv_score(self.cfg)[0] - 0.001
            print(f"drop '{g}': {m:.4f} ± {s:.4f} -> {'DROPPED (no loss)' if keep else 'kept'}")
            self.cfg = c if keep else self.cfg
        cfg = self.cfg
        print("\nFINAL cfg:", cfg)
        fm, fs = self.cv_score(cfg, seeds=(10, 11, 12, 13, 14))
        bm, bs = self.cv_score(dict(R=150, bw=0.5, kind="lr_l2", C=0.1, drop=()), seeds=(10, 11, 12, 13, 14))
        print(f"10c-style baseline: {bm:.4f} ± {bs:.4f} | tuned: {fm:.4f} ± {fs:.4f} (same dev queries -> gains below ~1 pt are within noise; selection on dev is still optimistic)")
        pd.DataFrame([dict(stage=a, **{k: str(v) for k, v in b.items()}, cv_f1=c, std=d) for a, b, c, d in self.log]).sort_values(
            "cv_f1", ascending=False).head(15).to_csv(f"{cfg_c.submit_dir}/ltr_tuning_top15.csv", index=False)
        Xd2, topd2, nm = self.build2("dev", cfg["R"], cfg["bw"], cfg["drop"])
        y2 = self.labels(topd2)
        F2 = Xd2.shape[2]
        sc2 = StandardScaler().fit(Xd2.reshape(-1, F2))
        m2 = make_ranker(cfg["kind"], cfg["C"]).fit(sc2.transform(Xd2.reshape(-1, F2)), y2.reshape(-1))
        sf = (lambda Z: m2.decision_function(Z)) if hasattr(m2, "decision_function") else (lambda Z: m2.predict_proba(Z)[:, 1])
        Xt2, topt2, _ = self.build2("test", cfg["R"], cfg["bw"], cfg["drop"])
        S_t2 = np.full((len(spl["test"]["qids"]), len(spl["test"]["cids"])), -1e9, np.float32)
        for i in range(Xt2.shape[0]):
            S_t2[i, topt2[i]] = sf(sc2.transform(Xt2[i]))
        zp = write_submission(f"LTR_tuned_{cfg['kind']}_R{cfg['R']}_C{cfg['C']}_bw{cfg['bw']}", S_t2, spl, cfg_c)
        validate_submission(zp, spl, cfg_c.k)
        if cfg["kind"].startswith("lr"):
            print(pd.Series(m2.coef_[0], index=nm).sort_values(key=abs, ascending=False).head(12).round(3).to_string())
        return dict(cfg=cfg, scores_test=S_t2, zip=zp, model=m2)


# =====================================================================================================
# v3 - header features (CITATOR INFO / soft year) + CSLS + rules
# =====================================================================================================
def feats_v3(ctx, exps, split):
    """Feature tensor of LTR v3 (signals + chronology / citation + CITATOR-INFO and soft-year features).

    Args:
        ctx (PCRContext): Pipeline context (``ctx.rf`` and ``ctx.ex`` must be filled).
        exps (List[str]): Dense experiment ids.
        split (str): ``dev`` | ``test``.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: ``(X, top, feature_names)`` with ``r_ltr`` candidates per query.
    """
    sg = {k: v for k, v in signals(ctx, split, exps).items() if not k.endswith("_ms")}
    rf = ctx.rf[split]
    allf = {**sg, "future": rf["future"], "known": rf["known"], "cite": rf["cite"], **ctx.ex[split]}
    pre = sum(sg[e] for e in exps) / len(exps) + 1.0 * sg["tfidf"]
    top = np.argsort(-pre, 1)[:, :ctx.cfg.r_ltr]
    names = list(allf)
    return stack_top(allf, names, top), top, names


def run_ltr_v3(ctx, exps):
    """LTR v3: LR over header features with CSLS and rule post-processing chosen by repeated CV on dev.

    Args:
        ctx (PCRContext): Pipeline context.
        exps (List[str]): Dense experiment ids.

    Returns:
        dict: ``{"table", "chosen", "scores_test", "zip"}``.
    """
    cfg, spl = ctx.cfg, ctx.spl
    Xd, topd, names = feats_v3(ctx, exps, "dev")
    yd = label_matrix(topd, spl, "dev")
    nqd = Xd.shape[0]
    cids_d = spl["dev"]["cids"]
    print("pre-fusion oracle recall in top-%d: %.3f | features: %s" % (cfg.r_ltr, yd.sum() / sum(len(spl['dev']['gold'][q]) for q in spl['dev']['qids']), names))

    def oof(seed):
        """Out-of-fold LR scores on dev for one fold-shuffling seed.

        Args:
            seed (int): KFold random state.

        Returns:
            np.ndarray: Score matrix with NaN outside the reranking depth.
        """
        S = np.full((nqd, len(cids_d)), np.nan, np.float32)
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(nqd)):
            sc, m = fit_lr(Xd[tr], yd[tr])
            for i in va:
                S[i, topd[i]] = m.decision_function(sc.transform(Xd[i]))
        return S

    OO = [fill_nan(oof(s)) for s in (0, 1, 2)]
    res = []
    rules = lambda L, split: apply_rules(L, ctx, split)
    for use_r in (False, True):
        for k in (0, 2, 3, 5):
            for a in ((0,) if k == 0 else (0.25, 0.5, 0.75, 1.0)):
                f = np.mean([f1_at_k(rules(csls(L, k, a), "dev") if use_r else (csls(L, k, a) if k else L), spl, "dev", cfg.k)["f1"] for L in OO])
                res.append(dict(rules=use_r, csls_k=k, a=a, cv_f1=round(float(f), 4)))
    RES = pd.DataFrame(res).sort_values("cv_f1", ascending=False)
    print(RES.head(10).to_string(index=False))
    print("\nreference (LTR raw, no rules, no CSLS):", RES[(RES.csls_k == 0) & (~RES.rules)].cv_f1.values)
    b = RES.iloc[0]
    print("chosen:", dict(b))
    sc, m = fit_lr(Xd, yd)
    Xt, topt, _ = feats_v3(ctx, exps, "test")
    S_t = np.full((len(spl["test"]["qids"]), len(spl["test"]["cids"])), np.nan, np.float32)
    for i in range(Xt.shape[0]):
        S_t[i, topt[i]] = m.decision_function(sc.transform(Xt[i]))
    S_t = fill_nan(S_t)
    if b.csls_k > 0:
        S_t = csls(S_t, int(b.csls_k), float(b.a))
    if b.rules:
        S_t = rules(S_t, "test")
    zp = write_submission(f"LTRv3_csls{int(b.csls_k)}_a{b.a}_rules{int(b.rules)}", S_t, spl, cfg)
    validate_submission(zp, spl, cfg.k)
    return dict(table=RES, chosen=b, scores_test=S_t, zip=zp)


# =====================================================================================================
# v4 - lexical suite + LR/HGB blend + CSLS / dual-softmax / rules
# =====================================================================================================
def feats_v4(ctx, exps, split):
    """Feature tensor of LTR v4 (v3 features + lexical suite channels).

    Args:
        ctx (PCRContext): Pipeline context (``ctx.lex2`` must hold the lexical suite for dev and test).
        exps (List[str]): Dense experiment ids.
        split (str): ``dev`` | ``test``.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: ``(X, top, feature_names)``.
    """
    sg = {k: v for k, v in signals(ctx, split, exps).items() if not k.endswith("_ms")}
    rf = ctx.rf[split]
    lx = {}
    for k, v in ctx.lex2.items():
        M = v[split]
        lx["L_" + k] = zs(M) if k in ("bi_q_nostop", "tri_q", "head_bi") else M
    lx["L_cocite_any"] = (ctx.lex2["cocite"][split] > 0.02).astype(np.float32)
    lx["L_stat_any"] = (ctx.lex2["stat"][split] > 0.3).astype(np.float32)
    allf = {**sg, **lx, "future": rf["future"], "known": rf["known"], "cite": rf["cite"], **ctx.ex[split]}
    pre = 0.5 * np.mean([sg[e] for e in exps], 0) + sg["tfidf"] + lx["L_tri_q"]
    top = np.argsort(-pre, 1)[:, :ctx.cfg.r_ltr]
    names = list(allf)
    return stack_top(allf, names, top), top, names


def run_ltr_v4(ctx, exps):
    """LTR v4: LR + HGB (and their blend) with CSLS / dual-softmax / rules chosen on out-of-fold dev scores.

    Args:
        ctx (PCRContext): Pipeline context.
        exps (List[str]): Dense experiment ids.

    Returns:
        dict: ``Xd, topd, yd, names, bestC, OO`` (out-of-fold score dicts for seeds 0-2), ``table``, ``chosen``,
        ``scores_test`` and ``zip``. ``OO[0]['blend']`` is reused by the two-stage reranker.
    """
    cfg, spl = ctx.cfg, ctx.spl
    Xd, topd, names = feats_v4(ctx, exps, "dev")
    yd = label_matrix(topd, spl, "dev")
    nqd = Xd.shape[0]
    cdev = len(spl["dev"]["cids"])
    print(f"features={Xd.shape[2]} | oracle recall in top-{cfg.r_ltr}: {yd.sum()/sum(len(spl['dev']['gold'][q]) for q in spl['dev']['qids']):.3f}")

    def oof2(seed, C):
        """Out-of-fold LR / HGB / blend scores on dev.

        Args:
            seed (int): KFold random state.
            C (float): LR regularisation.

        Returns:
            Dict[str, np.ndarray]: Score matrices for ``lr``, ``hgb`` and ``blend``.
        """
        out = {k: np.full((nqd, cdev), np.nan, np.float32) for k in ("lr", "hgb")}
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(nqd)):
            for kind in out:
                f = fit2(kind, Xd[tr], yd[tr], C)
                for i in va:
                    out[kind][i, topd[i]] = f(Xd[i])
        out["blend"] = gstd(out["lr"]) + gstd(out["hgb"])
        return out

    bestC = max((0.03, 0.1, 0.3), key=lambda C: np.mean([f1_at_k(fill_nan(oof2(s, C)["lr"]), spl, "dev", cfg.k)["f1"] for s in (0,)]))
    print("LR C:", bestC)
    OO = [oof2(s, bestC) for s in (0, 1, 2)]
    res = []
    for base in ("lr", "hgb", "blend"):
        Ls = [fill_nan(o[base]) for o in OO]
        cfgs = [("none", 0, 0)] + [("csls", k, a) for k in (2, 3, 4, 6) for a in (0.5, 0.75, 1.0, 1.25)] + [("dual", t, l) for t in (0.5, 1, 2) for l in (0.5, 1, 2)]
        for kind, p1, p2 in cfgs:
            for ur in (False, True):
                res.append(dict(base=base, post=kind, p1=p1, p2=p2, rules=ur,
                                cv_f1=round(float(np.mean([f1_at_k(post(L, ctx, "dev", kind, p1, p2, ur), spl, "dev", cfg.k)["f1"] for L in Ls])), 4)))
    RES = pd.DataFrame(res).sort_values("cv_f1", ascending=False)
    print(RES.head(12).to_string(index=False))
    print("\nreferences:")
    print(RES[(RES.post == "none") & (~RES.rules)][["base", "cv_f1"]].to_string(index=False))
    b = RES.iloc[0]
    print("chosen:", dict(b))
    Xt, topt, _ = feats_v4(ctx, exps, "test")
    nT, nC = len(spl["test"]["qids"]), len(spl["test"]["cids"])

    def test_scores(kind):
        """Fit on all dev and score the test reranking depth.

        Args:
            kind (str): ``lr`` or ``hgb``.

        Returns:
            np.ndarray: Test score matrix with NaN outside the reranking depth.
        """
        f = fit2(kind, Xd, yd, bestC)
        S = np.full((nT, nC), np.nan, np.float32)
        for i in range(nT):
            S[i, topt[i]] = f(Xt[i])
        return S

    TL = {"lr": test_scores("lr")} if b.base == "lr" else {"hgb": test_scores("hgb")} if b.base == "hgb" else {"lr": test_scores("lr"), "hgb": test_scores("hgb")}
    St = gstd(TL["lr"]) + gstd(TL["hgb"]) if b.base == "blend" else TL[b.base]
    St = post(fill_nan(St), ctx, "test", b.post, b.p1, b.p2, bool(b.rules))
    zp = write_submission(f"LTRv4_{b.base}_{b.post}{b.p1}_{b.p2}_r{int(b.rules)}", St, spl, cfg)
    validate_submission(zp, spl, cfg.k)
    return dict(Xd=Xd, topd=topd, yd=yd, names=names, bestC=bestC, OO=OO, table=RES, chosen=b, scores_test=St, zip=zp)
