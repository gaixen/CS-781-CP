"""Two-stage / three-stage learning-to-rank trained mostly on the *train* split.

* Stage A : lexical + rule LTR (LR + HGB + LambdaRank) with out-of-fold train scores.
* Stage B : learned "column competition" (hubness-aware features of stage-A scores), trained on train-OOF.
* Final   : stage A refit on train+dev, stage B on train-OOF+dev, blended with the dev-LTR v4 (``LTRv7``).
* Stage C : another column-competition stage on stage-B scores with seed bagging (``LTRv8``, used only if it clearly wins).
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pcr.rerank.common import apply_rules, csls, fit2, stack_top
from pcr.rerank.ltr_dev import feats_v4
from pcr.rerank.ltr_train import LT, lt_mats
from pcr.retrieval.lexical import lex_hash
from pcr.retrieval.metrics import f1_at_k, gold_matrix, label_matrix
from pcr.submission import validate_submission, write_submission
from pcr.utils import fill_nan, gstd, zs

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

LT2 = LT + ["bn", "long", "qlen", "dlen"]
BF_COL = ["s", "zrow", "rr", "margin", "cr", "colz", "csls", "dsm", "t1"]
BF_RAW = ["tri", "bn", "future", "cite", "fwd", "rev", "soft_year", "qlen", "dlen"]
FL = -3.0


def fz(S):
    """Replace NaN (unscored) entries by the floor value ``-3``.

    Args:
        S (np.ndarray): Score matrix with NaN for unscored pairs.

    Returns:
        np.ndarray: Matrix without NaN.
    """
    return np.where(np.isnan(S), FL, S)


def fit_multi(X, y, C=0.1, hdepth=4, leaves=15, nest=300, seed=0, lr_rate=0.05, min_child=30, reg_lambda=1.0, hgb_lr=0.06):
    """Fit LR + HGB + LightGBM-LambdaRank on (queries x depth x features) data.

    Args:
        X (np.ndarray): Features (n, R, F).
        y (np.ndarray): Labels (n, R).
        C (float): LR regularisation.
        hdepth (int): HGB max depth.
        leaves (int): LambdaRank number of leaves.
        nest (int): Number of boosting iterations.
        seed (int): Random state for HGB / LambdaRank.
        lr_rate (float): LambdaRank learning rate.
        min_child (int): LambdaRank ``min_child_samples``.
        reg_lambda (float): LambdaRank L2 regularisation.
        hgb_lr (float): HGB learning rate.

    Returns:
        Callable[[np.ndarray], Dict[str, np.ndarray]]: Predictor returning ``{"lr", "hgb", "rk"}`` score arrays (a, b).
    """
    n, R, F_ = X.shape
    sc = StandardScaler().fit(X.reshape(-1, F_))
    Z = sc.transform(X.reshape(-1, F_))
    yy = y.reshape(-1)
    lr = LogisticRegression(C=C, max_iter=3000).fit(Z, yy)
    hg = HistGradientBoostingClassifier(max_depth=hdepth, learning_rate=hgb_lr, max_iter=nest, l2_regularization=1.0, random_state=seed).fit(Z, yy)
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=nest, learning_rate=lr_rate, num_leaves=leaves, min_child_samples=min_child,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=reg_lambda, random_state=seed, verbose=-1).fit(Z, yy, group=[R] * n)

    def pred(Xq):
        """Score a feature tensor with the three rankers.

        Args:
            Xq (np.ndarray): Features (a, b, F).

        Returns:
            Dict[str, np.ndarray]: Scores (a, b) for ``lr``, ``hgb`` and ``rk``.
        """
        a, b, _ = Xq.shape
        Zq = sc.transform(Xq.reshape(-1, F_))
        return {"lr": lr.decision_function(Zq).reshape(a, b), "hgb": hg.decision_function(Zq).reshape(a, b), "rk": rk.predict(Zq).reshape(a, b)}
    return pred


def to_mat(P, top, nq, nc, rows=None):
    """Scatter per-depth predictions back into full (nq, nc) matrices (NaN elsewhere).

    Args:
        P (Dict[str, np.ndarray]): Predictions ``name -> (len(rows), R)``.
        top (np.ndarray): Candidate indices (nq, R).
        nq (int): Number of queries.
        nc (int): Number of candidates.
        rows (Optional[Iterable[int]]): Query rows the predictions belong to (default ``range(nq)``).

    Returns:
        Dict[str, np.ndarray]: Full matrices per ranker.
    """
    out = {}
    for k, v in P.items():
        M = np.full((nq, nc), np.nan, np.float32)
        for a, i in enumerate(rows if rows is not None else range(nq)):
            M[i, top[i]] = v[a]
        out[k] = M
    return out


def blend_a(M):
    """Blend LR / HGB / LambdaRank scores (global standardisation then sum).

    Args:
        M (Dict[str, np.ndarray]): Score matrices with keys ``lr``, ``hgb``, ``rk``.

    Returns:
        np.ndarray: Blended matrix.
    """
    return gstd(sum(gstd(M[k]) for k in ("lr", "hgb", "rk")))


def colfeats(S):
    """Column-competition features of a score matrix (hubness / mutual-ranking cues).

    Args:
        S (np.ndarray): Score matrix (n_queries, n_candidates) without NaN.

    Returns:
        Dict[str, np.ndarray]: ``s, zrow, rr, margin, cr, colz, csls, dsm, t1`` matrices.
    """
    nq, nc = S.shape
    srt = np.sort(S, 0)
    t1, t2, t3 = srt[-1], srt[-2], srt[-3:].mean(0)
    om = np.where(S >= t1[None, :], t2[None, :], t1[None, :])
    cr = np.argsort(np.argsort(-S, 0), 0)
    rr = np.argsort(np.argsort(-S, 1), 1)
    mu, sd = S.mean(1, keepdims=True), S.std(1, keepdims=True) + 1e-6
    return dict(s=S, zrow=(S - mu) / sd, rr=-np.log1p(rr), margin=S - om, cr=-np.log1p(cr), colz=(S - S.mean(0)) / (S.std(0) + 1e-6),
                csls=S - 0.75 * t3[None, :], dsm=(S - logsumexp(S, 1, keepdims=True)) + 2 * (S - logsumexp(S, 0, keepdims=True)),
                t1=np.repeat(t1[None, :], nq, 0))


class TwoStageLTR:
    """Stage A / B / C rerankers trained on the train split.

    Args:
        ctx (PCRContext): Pipeline context.
        exps (List[str]): Dense experiment ids.
        state (dict): Output of ``build_ltr_state``.
        v4 (dict): Output of ``run_ltr_v4`` (provides ``bestC`` and the dev-LTR out-of-fold scores).
    """

    def __init__(self, ctx, exps, state, v4):
        if lgb is None:
            raise ImportError("lightgbm is required for the two-stage reranker (pip install lightgbm)")
        self.ctx, self.exps, self.state, self.v4 = ctx, exps, state, v4
        self.spl = ctx.spl
        self.nT, self.nC = len(self.spl["test"]["qids"]), len(self.spl["test"]["cids"])

    # ------------------------------------------------------------------ matrices
    def lt_mats2(self, s):
        """Train-LTR matrices plus phrase channels and length features.

        Args:
            s (str): Split name.

        Returns:
            Dict[str, np.ndarray]: Matrices keyed by :data:`LT2` names.
        """
        m = lt_mats(self.ctx, self.state, s)
        L = self.state["LEXT"][s]
        m["bn"] = zs(L["bn"])
        m["long"] = zs(L["long"])
        S = self.spl[s]
        ql = np.log1p([len(S["qtext"][q]) for q in S["qids"]]).astype(np.float32)
        dl = np.log1p([len(S["ctext"][c]) for c in S["cids"]]).astype(np.float32)
        m["qlen"] = np.repeat(ql[:, None], len(dl), 1)
        m["dlen"] = np.repeat(dl[None, :], len(ql), 0)
        return m

    # ------------------------------------------------------------------ stage A
    def fit_stage_a(self):
        """Stage A: phrase channels, LR + HGB + LambdaRank on train, out-of-fold train scores, dev / test scores.

        Args:
            None

        Returns:
            None (results are stored on ``self``: ``AOOF, AD, AT, MA, TOPA, XA_all, yA_all, XA_dev, yA_dev, has, Gm``).
        """
        import time
        ctx, cfg, spl, state = self.ctx, self.ctx.cfg, self.spl, self.state
        t0 = time.time()
        for s in ("train", "dev", "test"):
            state["LEXT"][s]["bn"] = ctx.lex2["bi_q_nostop"][s] if s != "train" else lex_hash(spl, "train", cfg, (2, 3), None)
            state["LEXT"][s]["long"] = lex_hash(spl, s, cfg, (4, 5), None, prep=lambda t: t[:60000], min_df=2)
            print(f"  channels {s} ({time.time()-t0:.0f}s)")
        R_A = cfg.r_stage_a
        self.MA = {s: self.lt_mats2(s) for s in ("train", "dev", "test")}
        self.TOPA = {s: np.argsort(-(self.MA[s]["uni"] + self.MA[s]["bi"] + self.MA[s]["tri"] + self.MA[s]["bn"]), 1)[:, :R_A] for s in self.MA}
        self.Gm = {"train": gold_matrix("train", spl), "dev": gold_matrix("dev", spl)}
        self.nqT, self.ncT = self.Gm["train"].shape
        self.nqd, self.ncd = self.Gm["dev"].shape
        self.has = self.Gm["train"].any(1)
        has, MA, TOPA, Gm = self.has, self.MA, self.TOPA, self.Gm
        nqT, ncT, nqd, ncd = self.nqT, self.ncT, self.nqd, self.ncd
        self.XA_all = stack_top(MA["train"], LT2, TOPA["train"])
        self.yA_all = np.stack([Gm["train"][i, TOPA["train"][i]] for i in range(nqT)]).astype(int)
        self.XA_dev = stack_top(MA["dev"], LT2, TOPA["dev"])
        self.yA_dev = np.stack([Gm["dev"][i, TOPA["dev"][i]] for i in range(nqd)]).astype(int)
        print("oracle recall in top-%d: train %.3f | dev %.3f" % (R_A, self.yA_all[has].sum() / Gm["train"][has].sum(), self.yA_dev.sum() / Gm["dev"].sum()))
        self.AOOF = {k: np.full((nqT, ncT), np.nan, np.float32) for k in ("lr", "hgb", "rk")}
        for tr, va in KFold(5, shuffle=True, random_state=0).split(np.arange(nqT)):
            f = fit_multi(self.XA_all[tr[has[tr]]], self.yA_all[tr[has[tr]]])
            P = to_mat(f(self.XA_all[va]), TOPA["train"], nqT, ncT, rows=va)
            for k in self.AOOF:
                self.AOOF[k][va] = P[k][va]
            print(f"  OOF fold done ({time.time()-t0:.0f}s)")
        fA = fit_multi(self.XA_all[has], self.yA_all[has])
        self.AD = to_mat(fA(self.XA_dev), TOPA["dev"], nqd, ncd)
        self.AT = to_mat(fA(stack_top(MA["test"], LT2, TOPA["test"])), TOPA["test"], self.nT, self.nC)
        rows = []
        for nm, M in list(self.AD.items()) + [("blend", blend_a(self.AD))]:
            L = fill_nan(M if nm == "blend" else gstd(M))
            rows.append(dict(model=nm, raw=round(f1_at_k(L, spl, "dev", cfg.k)["f1"], 4),
                             csls3_75_rules=round(f1_at_k(apply_rules(csls(L, 3, 0.75), ctx, "dev"), spl, "dev", cfg.k)["f1"], 4)))
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"done {(time.time()-t0)/60:.1f} min")

    # ------------------------------------------------------------------ stage B
    def B_X(self, split, SAf, rows):
        """Stage-B features of the top-``r_stage_b`` candidates of stage-A scores.

        Args:
            split (str): Split name.
            SAf (np.ndarray): NaN-filled stage-A score matrix.
            rows (Iterable[int]): Query rows to use.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(X, top)``.
        """
        F = {**colfeats(SAf), **{k: self.MA[split][k] for k in BF_RAW}}
        top = np.argsort(-SAf, 1)[:, :self.ctx.cfg.r_stage_b]
        return stack_top(F, BF_COL + BF_RAW, top, rows), top

    def final_score(self, Bz_, Vz_, cfg_, split):
        """Combine stage-B and dev-LTR scores, apply CSLS and rules.

        Args:
            Bz_ (np.ndarray): Stage-B score matrix.
            Vz_ (np.ndarray): Dev-LTR (v4) score matrix.
            cfg_ (dict): Keys ``w`` (dev-LTR weight), ``k`` and ``a`` (CSLS).
            split (str): Split name.

        Returns:
            np.ndarray: Final scores.
        """
        S = fill_nan(Bz_) + (cfg_["w"] * fill_nan(Vz_) if cfg_["w"] > 0 else 0)
        if cfg_["k"] > 0:
            S = csls(S, cfg_["k"], cfg_["a"])
        return apply_rules(S, self.ctx, split)

    def fit_stage_b(self):
        """Stage B: learn column competition on train-OOF stage-A scores; choose the final blend / CSLS config on dev.

        Args:
            None

        Returns:
            dict: The selected ``CFG`` (keys ``w``, ``k``, ``a``); other artefacts are stored on ``self``.
        """
        ctx, cfg, spl = self.ctx, self.ctx.cfg, self.spl
        Gm, nqd, ncd = self.Gm, self.nqd, self.ncd
        self.SAf = {"train": fz(blend_a(self.AOOF)), "dev": fz(blend_a(self.AD)), "test": fz(blend_a(self.AT))}
        self.rows_tr = np.where(self.has)[0]
        self.XB_tr, self.topB_tr = self.B_X("train", self.SAf["train"], self.rows_tr)
        self.yB_tr = np.stack([Gm["train"][i, self.topB_tr[i]] for i in self.rows_tr]).astype(int)
        self.XB_dv, self.topB_dv = self.B_X("dev", self.SAf["dev"], range(nqd))
        self.yB_dv = np.stack([Gm["dev"][i, self.topB_dv[i]] for i in range(nqd)]).astype(int)
        print("stage-B oracle recall in top-%d: train %.3f | dev %.3f" % (cfg.r_stage_b, self.yB_tr.sum() / Gm["train"][self.has].sum(), self.yB_dv.sum() / Gm["dev"].sum()))
        fB = fit_multi(self.XB_tr, self.yB_tr, C=0.3, hdepth=3, leaves=7, nest=250)
        self.BD = to_mat(fB(self.XB_dv), self.topB_dv, nqd, ncd)
        self.Bz = gstd(sum(gstd(self.BD[k]) for k in self.BD))
        self.Vz = gstd(self.v4["OO"][0]["blend"])
        f = lambda S: {k: round(v, 4) for k, v in f1_at_k(S, spl, "dev", cfg.k).items()}
        rows = [dict(sys="A blend (stage A)", raw=round(f1_at_k(fz(blend_a(self.AD)), spl, "dev", cfg.k)["f1"], 4),
                     post=round(f1_at_k(apply_rules(csls(fz(blend_a(self.AD)), 3, 0.75), ctx, "dev"), spl, "dev", cfg.k)["f1"], 4))]
        for k in self.BD:
            L = fill_nan(gstd(self.BD[k]))
            rows.append(dict(sys=f"B {k}", raw=round(f1_at_k(L, spl, "dev", cfg.k)["f1"], 4), post=round(f1_at_k(apply_rules(L, ctx, "dev"), spl, "dev", cfg.k)["f1"], 4)))
        rows.append(dict(sys="B blend", raw=round(f1_at_k(fill_nan(self.Bz), spl, "dev", cfg.k)["f1"], 4), post=round(f1_at_k(apply_rules(fill_nan(self.Bz), ctx, "dev"), spl, "dev", cfg.k)["f1"], 4)))
        rows.append(dict(sys="dev-LTR v4 (CV)", raw=round(f1_at_k(fill_nan(self.Vz), spl, "dev", cfg.k)["f1"], 4),
                         post=round(f1_at_k(apply_rules(csls(fill_nan(self.Vz), 3, 0.75), ctx, "dev"), spl, "dev", cfg.k)["f1"], 4)))
        print(pd.DataFrame(rows).to_string(index=False))
        res = []
        for w in (0, 0.25, 0.5, 1.0):
            for k, a in [(0, 0)] + [(k, a) for k in (3, 4) for a in (0.25, 0.5, 0.75, 1.0)]:
                c = dict(w=w, k=k, a=a)
                res.append(dict(**c, **f(self.final_score(self.Bz, self.Vz, c, "dev"))))
        RES = pd.DataFrame(res).sort_values("f1", ascending=False)
        print(RES.head(10).to_string(index=False))
        self.CFG = {k: float(RES.iloc[0][k]) for k in ("w", "a")}
        self.CFG["k"] = int(RES.iloc[0]["k"])
        print("CFG:", self.CFG)
        return self.CFG

    def final_ab(self):
        """Final A+B submission: refit stage A on train+dev, stage B on train-OOF+dev, blend with dev-LTR v4 (``LTRv7``).

        Args:
            None

        Returns:
            dict: ``{"St", "Bz_t", "Vz_t", "SAf_t", "zip"}``.
        """
        import time
        ctx, cfg, spl = self.ctx, self.ctx.cfg, self.spl
        t0 = time.time()
        nT, nC = self.nT, self.nC
        fA2 = fit_multi(np.concatenate([self.XA_all[self.has], self.XA_dev]), np.concatenate([self.yA_all[self.has], self.yA_dev]))
        AT2 = to_mat(fA2(stack_top(self.MA["test"], LT2, self.TOPA["test"])), self.TOPA["test"], nT, nC)
        self.SAf_t = fz(blend_a(AT2))
        XB_te, topB_te = self.B_X("test", self.SAf_t, range(nT))
        fB2 = fit_multi(np.concatenate([self.XB_tr, self.XB_dv]), np.concatenate([self.yB_tr, self.yB_dv]), C=0.3, hdepth=3, leaves=7, nest=250)
        BT = to_mat(fB2(XB_te), topB_te, nT, nC)
        self.Bz_t = gstd(sum(gstd(BT[k]) for k in BT))
        v4 = self.v4
        Xd2, top2, _ = feats_v4(ctx, self.exps, "dev")
        y2 = label_matrix(top2, spl, "dev")
        Xt2, topt2, _ = feats_v4(ctx, self.exps, "test")
        fl, fh = fit2("lr", Xd2, y2, v4["bestC"]), fit2("hgb", Xd2, y2, v4["bestC"])
        Vl = np.full((nT, nC), np.nan, np.float32)
        Vh = Vl.copy()
        for i in range(nT):
            Vl[i, topt2[i]] = fl(Xt2[i])
            Vh[i, topt2[i]] = fh(Xt2[i])
        self.Vz_t = gstd(gstd(Vl) + gstd(Vh))
        St = self.final_score(self.Bz_t, self.Vz_t, self.CFG, "test")
        zp = write_submission(f"LTRv7_AB_w{self.CFG['w']}_k{self.CFG['k']}_a{self.CFG['a']}", St, spl, cfg)
        validate_submission(zp, spl, cfg.k)
        print(f"({(time.time()-t0)/60:.1f} min)")
        return dict(St=St, Bz_t=self.Bz_t, Vz_t=self.Vz_t, SAf_t=self.SAf_t, zip=zp)

    # ------------------------------------------------------------------ stage C
    def stage_c(self, w_final=0.5):
        """Stage C: column competition on stage-B scores (train-OOF) with seed bagging; submitted only if it beats B by > 0.3 pt.

        Args:
            w_final (float): Weight of the dev-LTR v4 scores added to stage C in the final blend.

        Returns:
            dict: ``{"use_c": bool, "zip": Optional[str], "f1_B": float, "f1_C": float}``.
        """
        import time
        ctx, cfg, spl = self.ctx, self.ctx.cfg, self.spl
        t0 = time.time()
        nqT, ncT, nqd, ncd, Gm = self.nqT, self.ncT, self.nqd, self.ncd, self.Gm
        bag_fit = lambda X, y, seeds=(0, 1, 2): [fit_multi(X, y, C=0.3, hdepth=3, leaves=7, nest=250, seed=s, lr_rate=0.04, min_child=40, reg_lambda=2.0) for s in seeds]
        bag_pred = lambda fs, X: (lambda Ps: {k: np.mean([P[k] for P in Ps], 0) for k in Ps[0]})([f(X) for f in fs])
        BOOF = {k: np.full((nqT, ncT), np.nan, np.float32) for k in ("lr", "hgb", "rk")}
        rows_tr = self.rows_tr
        for tr, va in KFold(5, shuffle=True, random_state=1).split(np.arange(len(rows_tr))):
            P = fit_multi(self.XB_tr[tr], self.yB_tr[tr], C=0.3, hdepth=3, leaves=7, nest=250)(self.XB_tr[va])
            for k in BOOF:
                for a, j in enumerate(va):
                    BOOF[k][rows_tr[j], self.topB_tr[rows_tr[j]]] = P[k][a]
        print(f"stage-B OOF on train done ({time.time()-t0:.0f}s)")
        names = BF_COL + ["sA"] + BF_RAW

        def C_X(split, SBf, SAf_, rows):
            """Stage-C features of the top-``r_stage_c`` candidates of stage-B scores.

            Args:
                split (str): Split name.
                SBf (np.ndarray): NaN-filled stage-B score matrix.
                SAf_ (np.ndarray): NaN-filled stage-A score matrix.
                rows (Iterable[int]): Query rows.

            Returns:
                Tuple[np.ndarray, np.ndarray]: ``(X, top)``.
            """
            F = {**colfeats(SBf), **{k: self.MA[split][k] for k in BF_RAW}}
            F["sA"] = SAf_
            top = np.argsort(-SBf, 1)[:, :cfg.r_stage_c]
            return stack_top(F, names, top, rows), top

        SBf_tr, SBf_dv = fz(gstd(sum(gstd(BOOF[k]) for k in BOOF))), fz(self.Bz)
        XC_tr, topC_tr = C_X("train", SBf_tr, self.SAf["train"], rows_tr)
        yC_tr = np.stack([Gm["train"][i, topC_tr[i]] for i in rows_tr]).astype(int)
        XC_dv, topC_dv = C_X("dev", SBf_dv, self.SAf["dev"], range(nqd))
        yC_dv = np.stack([Gm["dev"][i, topC_dv[i]] for i in range(nqd)]).astype(int)
        fsC = bag_fit(XC_tr, yC_tr)
        CD = to_mat(bag_pred(fsC, XC_dv), topC_dv, nqd, ncd)
        Cz = gstd(sum(gstd(CD[k]) for k in CD))
        rB, rC = f1_at_k(fill_nan(self.Bz), spl, "dev", cfg.k), f1_at_k(fill_nan(Cz), spl, "dev", cfg.k)
        print(pd.DataFrame([dict(sys="stage B", **{k: round(v, 4) for k, v in rB.items()}, with_rules=round(f1_at_k(apply_rules(fill_nan(self.Bz), ctx, "dev"), spl, "dev", cfg.k)["f1"], 4)),
                            dict(sys="stage C (bagged x3)", **{k: round(v, 4) for k, v in rC.items()}, with_rules=round(f1_at_k(apply_rules(fill_nan(Cz), ctx, "dev"), spl, "dev", cfg.k)["f1"], 4))]).to_string(index=False))
        print("recall in C top-%d: %.3f" % (cfg.r_stage_c, yC_dv.sum() / Gm["dev"].sum()))
        for w in (0.0, 0.5):
            print(f"  C + {w}*dev-LTR + rules ->", round(f1_at_k(apply_rules(fill_nan(Cz) + w * fill_nan(self.Vz), ctx, "dev"), spl, "dev", cfg.k)["f1"], 4))
        use_c = rC["f1"] > rB["f1"] + 0.003
        print("USE_C:", use_c)
        zp = None
        if use_c:
            XC_te, topC_te = C_X("test", fz(self.Bz_t), self.SAf_t, range(self.nT))
            fsC2 = bag_fit(np.concatenate([XC_tr, XC_dv]), np.concatenate([yC_tr, yC_dv]))
            CT_ = to_mat(bag_pred(fsC2, XC_te), topC_te, self.nT, self.nC)
            Cz_t = gstd(sum(gstd(CT_[k]) for k in CT_))
            St = apply_rules(fill_nan(Cz_t) + w_final * fill_nan(self.Vz_t), ctx, "test")
            zp = write_submission(f"LTRv8_stageC_w{w_final}", St, spl, cfg)
            validate_submission(zp, spl, cfg.k)
        else:
            print("stage C did not clearly beat B on the dev holdout -> keep LTRv7, do not submit a new file")
        print(f"done {(time.time()-t0)/60:.1f} min")
        return dict(use_c=bool(use_c), zip=zp, f1_B=rB["f1"], f1_C=rC["f1"])
