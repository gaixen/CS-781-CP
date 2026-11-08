"""Learning-to-rank trained on the *train* split's lexical / rule features, then stacked with the dev-fitted LTR.

* :func:`build_ltr_state`  : strict year / citation features and lexical channels for all splits, train-split LR + HGB LTR.
* :func:`run_ltr_v5`       : dev LTR + train-LTR scores + strict year features (LR / HGB blend, CSLS / rules).
* :func:`run_ltr_v6`       : v5 + pseudo-relevance feedback features (similarity to the current top-m).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pcr.data.loader import load_raw_splits
from pcr.rerank.common import fit2, post, stack_top
from pcr.rerank.ltr_dev import feats_v4
from pcr.retrieval.lexical import cc_sim, lex_hash, lex_tok, statute_toks
from pcr.retrieval.metrics import f1_at_k, gold_matrix, label_matrix
from pcr.retrieval.rules import docinfo, extra_feats, rule_feats, year_pair
from pcr.submission import validate_submission, write_submission
from pcr.utils import fill_nan, gstd, zs

LT = ["uni", "bi", "tri", "bm25", "stat", "future", "known", "cite", "fwd", "rev", "soft_year", "ydiff",
      "yc_viol", "yc_gap", "yd_viol", "yd_known", "cand_cites_q"]


def lt_mats(ctx, state, s):
    """Feature matrices of the train-split LTR for a split.

    Args:
        ctx (PCRContext): Pipeline context.
        state (dict): Output of :func:`build_ltr_state` (needs ``LEXT`` and ``YP``).
        s (str): ``train`` | ``dev`` | ``test``.

    Returns:
        Dict[str, np.ndarray]: ``uni, bi, tri, bm25, stat``, rule features, header features and year-pair features.
    """
    L = state["LEXT"][s]
    m = dict(uni=zs(L["uni"]), bi=zs(L["bi"]), tri=zs(L["tri"]), bm25=zs(ctx.lex["bm25"][s]), stat=L["stat"])
    m.update({k: ctx.rf[s][k] for k in ("future", "known", "cite")})
    m.update({k: ctx.ex[s][k] for k in ("fwd", "rev", "soft_year", "ydiff")})
    m.update(state["YP"][s])
    return m


def build_ltr_state(ctx, paths):
    """Compute strict year / citation features and lexical channels for train / dev / test and fit the train-split LTR.

    Side effects: ``ctx.rf['train']`` and ``ctx.ex['train']`` are filled.

    Args:
        ctx (PCRContext): Pipeline context (``ctx.lex2`` filled for dev / test, ``ctx.rf`` / ``ctx.ex`` for dev / test).
        paths (Dict[str, str]): Local data paths (to re-read the RAW train headers).

    Returns:
        dict: ``LEXT, DI, YP, scT, LRT, HGT, TRS`` where ``TRS[split]`` holds the train-LTR ``lr`` / ``hgb`` scores of dev and test.
    """
    cfg, spl, meta = ctx.cfg, ctx.spl, ctx.meta
    raw0 = load_raw_splits(paths, ("train", "train_c"))
    ctx.rf["train"] = rule_feats("train", spl, meta)
    ctx.ex["train"] = extra_feats("train", spl, raw0)
    del raw0
    DI = {s: docinfo(s, spl, meta) for s in ("train", "dev", "test")}
    YP = {s: year_pair(s, spl, meta, DI[s]) for s in DI}
    Gd = gold_matrix("dev", spl)
    for k, M in YP["dev"].items():
        print(f"  dev {k:13s} flagged={int((M != 0).sum()):6d} gold flagged={int((M[Gd] != 0).sum()):4d} gold rate among flagged={Gd[M != 0].mean():.4f}")
    LEXT = {}
    for s in ("train", "dev", "test"):
        LEXT[s] = dict(uni=lex_hash(spl, s, cfg, (1, 1), "english"), bi=lex_hash(spl, s, cfg, (1, 2), "english"),
                       tri=ctx.lex2["tri_q"][s] if s in ctx.lex2["tri_q"] else lex_hash(spl, s, cfg, (1, 3), "english"),
                       stat=ctx.lex2["stat"][s] if s in ctx.lex2["stat"] else lex_tok(spl, s, lambda t: statute_toks(t, cfg.max_chars)))
        print(f"  lexical {s} done")
    state = dict(LEXT=LEXT, DI=DI, YP=YP)
    R_TR = cfg.r_train
    mt = lt_mats(ctx, state, "train")
    Gt = gold_matrix("train", spl)
    rows = [i for i in range(len(spl["train"]["qids"])) if Gt[i].any()]
    topT = np.argsort(-(mt["uni"] + mt["bi"] + mt["tri"]), 1)[:, :R_TR]
    Xtr = np.stack([np.stack([mt[n][i, topT[i]] for n in LT], 1) for i in rows])
    ytr = np.stack([Gt[i, topT[i]] for i in rows]).astype(int)
    Ftr = Xtr.shape[2]
    print("train LTR rows:", Xtr.shape, "| pos rate", round(ytr.mean(), 3), "| oracle recall in top-%d: %.3f" % (R_TR, ytr.sum() / Gt[rows].sum()))
    scT = StandardScaler().fit(Xtr.reshape(-1, Ftr))
    Ztr = scT.transform(Xtr.reshape(-1, Ftr))
    yy = ytr.reshape(-1)
    LRT = LogisticRegression(C=0.1, max_iter=3000, random_state=0).fit(Ztr, yy)
    HGT = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0).fit(Ztr, yy)
    print(pd.Series(LRT.coef_[0], index=LT).sort_values(key=abs, ascending=False).round(3).head(12).to_string())

    def tr_scores(s):
        """Score every (query, candidate) pair of a split with the train-split LTR.

        Args:
            s (str): ``dev`` | ``test``.

        Returns:
            Dict[str, np.ndarray]: ``lr`` and ``hgb`` score matrices (n_queries, n_candidates).
        """
        m = lt_mats(ctx, state, s)
        nq, nc = m["tri"].shape
        Z = scT.transform(np.stack([m[n].reshape(-1) for n in LT], 1))
        return {"lr": LRT.decision_function(Z).reshape(nq, nc).astype(np.float32), "hgb": HGT.decision_function(Z).reshape(nq, nc).astype(np.float32)}

    TRS = {s: tr_scores(s) for s in ("dev", "test")}
    for nm, S_ in (("TRAIN-LTR lr", TRS["dev"]["lr"]), ("TRAIN-LTR hgb", TRS["dev"]["hgb"]), ("TRAIN-LTR blend", gstd(TRS["dev"]["lr"]) + gstd(TRS["dev"]["hgb"]))):
        print(f"{nm:16s} dev raw", {k: round(v, 4) for k, v in f1_at_k(S_, spl, "dev", cfg.k).items()}, "| +CSLS(3,.75)+rules",
              round(f1_at_k(post(S_, ctx, "dev", "csls", 3, 0.75, True), spl, "dev", cfg.k)["f1"], 4))
    state.update(scT=scT, LRT=LRT, HGT=HGT, TRS=TRS)
    return state


def feats_v5(ctx, exps, state, split):
    """Feature tensor of LTR v5: v4 features + train-LTR scores + strict year features.

    Args:
        ctx (PCRContext): Pipeline context.
        exps (List[str]): Dense experiment ids.
        state (dict): Output of :func:`build_ltr_state`.
        split (str): ``dev`` | ``test``.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: ``(X, top, feature_names)``.
    """
    X, top, names = feats_v4(ctx, exps, split)
    nq = X.shape[0]
    add = {"TR_lr": zs(state["TRS"][split]["lr"]), "TR_hgb": zs(state["TRS"][split]["hgb"])}
    for k in ("yc_viol", "yc_gap", "yd_viol", "yd_known", "cand_cites_q"):
        add[k] = state["YP"][split][k]
    an = list(add)
    Xa = stack_top(add, an, top)
    return np.concatenate([X, Xa], 2), top, names + an


def stage_run(ctx, Xd_, topd_, yd_, Xt_, topt_):
    """Fit LR / HGB / blend with out-of-fold evaluation on dev, choose the post-processing and score test.

    Args:
        ctx (PCRContext): Pipeline context.
        Xd_ (np.ndarray): Dev features (n_dev, R, F).
        topd_ (np.ndarray): Dev candidate indices (n_dev, R).
        yd_ (np.ndarray): Dev labels (n_dev, R).
        Xt_ (np.ndarray): Test features (n_test, R, F).
        topt_ (np.ndarray): Test candidate indices (n_test, R).

    Returns:
        dict: ``res`` (table of CV scores), ``best`` (chosen row), ``C``, ``St`` (post-processed test scores) and
        ``DVB`` (post-processed out-of-fold dev scores of the chosen base).
    """
    cfg, spl = ctx.cfg, ctx.spl
    nqd_, ncd, nT, nC = Xd_.shape[0], len(spl["dev"]["cids"]), Xt_.shape[0], len(spl["test"]["cids"])

    def oof(seed, C, kinds=("lr", "hgb")):
        """Out-of-fold scores on dev.

        Args:
            seed (int): KFold random state.
            C (float): LR regularisation.
            kinds (Tuple[str, ...]): Rankers to fit (``lr`` and / or ``hgb``).

        Returns:
            Dict[str, np.ndarray]: One matrix per ranker (plus ``blend`` if both were fitted).
        """
        out = {k: np.full((nqd_, ncd), np.nan, np.float32) for k in kinds}
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(nqd_)):
            for kind in out:
                f = fit2(kind, Xd_[tr], yd_[tr], C)
                for i in va:
                    out[kind][i, topd_[i]] = f(Xd_[i])
        if len(kinds) == 2:
            out["blend"] = gstd(out["lr"]) + gstd(out["hgb"])
        return out

    C = max((0.03, 0.1, 0.3), key=lambda c: f1_at_k(fill_nan(oof(0, c, ("lr",))["lr"]), spl, "dev", cfg.k)["f1"])
    OO_ = [oof(s, C) for s in (0, 1, 2)]
    res = []
    cfgs = [("none", 0, 0)] + [("csls", k, a) for k in (3, 4, 6) for a in (0.5, 0.75, 1.0, 1.25)] + [("dual", 1, 2), ("dual", 2, 2)]
    for base in ("lr", "hgb", "blend"):
        Ls = [fill_nan(o[base]) for o in OO_]
        for kind, p1, p2 in cfgs:
            for ur in ((False, True) if kind == "none" else (True,)):
                res.append(dict(base=base, post=kind, p1=p1, p2=p2, rules=ur,
                                cv_f1=round(float(np.mean([f1_at_k(post(L, ctx, "dev", kind, p1, p2, ur), spl, "dev", cfg.k)["f1"] for L in Ls])), 4)))
    RES = pd.DataFrame(res).sort_values("cv_f1", ascending=False)
    b = RES.iloc[0]

    def tsc(kind):
        """Fit on all dev and score the test reranking depth.

        Args:
            kind (str): ``lr`` or ``hgb``.

        Returns:
            np.ndarray: Test score matrix with NaN outside the reranking depth.
        """
        f = fit2(kind, Xd_, yd_, C)
        S = np.full((nT, nC), np.nan, np.float32)
        for i in range(nT):
            S[i, topt_[i]] = f(Xt_[i])
        return S

    TL = {k: tsc(k) for k in (("lr", "hgb") if b.base == "blend" else (b.base,))}
    St = gstd(TL["lr"]) + gstd(TL["hgb"]) if b.base == "blend" else TL[b.base]
    return dict(res=RES, best=b, C=C, St=post(fill_nan(St), ctx, "test", b.post, b.p1, b.p2, bool(b.rules)),
                DVB=post(fill_nan(OO_[0][b.base]), ctx, "dev", b.post, b.p1, b.p2, bool(b.rules)))


def run_ltr_v5(ctx, exps, state):
    """LTR v5: dev LTR with train-LTR scores and strict year features; writes the ``LTRv5_*`` submission.

    Args:
        ctx (PCRContext): Pipeline context.
        exps (List[str]): Dense experiment ids.
        state (dict): Output of :func:`build_ltr_state`.

    Returns:
        dict: ``{"R2", "X3", "top3", "y3", "Xt3", "topt3", "zip"}`` where ``R2`` is the :func:`stage_run` result
        (reused by v6).
    """
    cfg, spl = ctx.cfg, ctx.spl
    X3, top3, names3 = feats_v5(ctx, exps, state, "dev")
    y3 = label_matrix(top3, spl, "dev")
    Xt3, topt3, _ = feats_v5(ctx, exps, state, "test")
    print(f"features={X3.shape[2]} | oracle recall in top-{cfg.r_ltr}: {y3.sum()/sum(len(spl['dev']['gold'][q]) for q in spl['dev']['qids']):.3f}")
    R2 = stage_run(ctx, X3, top3, y3, Xt3, topt3)
    print(R2["res"].head(12).to_string(index=False))
    print("\nreferences (no post):")
    print(R2["res"][R2["res"].post == "none"][["base", "rules", "cv_f1"]].to_string(index=False))
    print("chosen:", dict(R2["best"]))
    b = R2["best"]
    zp = write_submission(f"LTRv5_{b.base}_{b.post}{b.p1}_{b.p2}_r{int(b.rules)}", R2["St"], spl, cfg)
    validate_submission(zp, spl, cfg.k)
    return dict(R2=R2, X3=X3, top3=top3, y3=y3, Xt3=Xt3, topt3=topt3, zip=zp)


def add_prf(X, top, B, CC, ms=(3, 7)):
    """Append pseudo-relevance-feedback features: mean similarity of each candidate to the current top-m.

    Args:
        X (np.ndarray): Existing features (n_queries, R, F).
        top (np.ndarray): Candidate indices (n_queries, R).
        B (np.ndarray): Current score matrix defining the top-m of each query.
        CC (np.ndarray): Candidate-candidate similarity (n_candidates, n_candidates).
        ms (Tuple[int, ...]): Values of m.

    Returns:
        Tuple[np.ndarray, List[str]]: ``(X_with_prf, new_feature_names)``.
    """
    o = np.argsort(-B, 1)
    pf = {}
    for m in ms:
        P = np.zeros_like(B)
        for i in range(B.shape[0]):
            P[i] = CC[:, o[i, :m]].mean(1)
        pf[f"prf{m}"] = zs(P)
    an = list(pf)
    Xa = stack_top(pf, an, top)
    return np.concatenate([X, Xa], 2), an


def run_ltr_v6(ctx, v5):
    """LTR v6: v5 + PRF features computed from the v5 scores; writes the ``LTRv6_prf_*`` submission.

    Args:
        ctx (PCRContext): Pipeline context.
        v5 (dict): Output of :func:`run_ltr_v5`.

    Returns:
        dict: ``{"R3", "zip"}`` where ``R3`` is the :func:`stage_run` result.
    """
    cfg, spl = ctx.cfg, ctx.spl
    CCd, CCt = cc_sim(spl, "dev", cfg), cc_sim(spl, "test", cfg)
    X4, _ = add_prf(v5["X3"], v5["top3"], v5["R2"]["DVB"], CCd)
    Xt4, _ = add_prf(v5["Xt3"], v5["topt3"], v5["R2"]["St"], CCt)
    R3 = stage_run(ctx, X4, v5["top3"], v5["y3"], Xt4, v5["topt3"])
    print(R3["res"].head(12).to_string(index=False))
    print("\nreferences (no post):")
    print(R3["res"][R3["res"].post == "none"][["base", "rules", "cv_f1"]].to_string(index=False))
    print("stage-2 best CV:", float(v5["R2"]["best"].cv_f1), "| stage-3 (PRF) best CV:", float(R3["best"].cv_f1))
    b = R3["best"]
    zp = write_submission(f"LTRv6_prf_{b.base}_{b.post}{b.p1}_{b.p2}_r{int(b.rules)}", R3["St"], spl, cfg)
    validate_submission(zp, spl, cfg.k)
    return dict(R3=R3, zip=zp)
