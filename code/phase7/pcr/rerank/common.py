"""Helpers shared by fusion and learning-to-rank stages.

* loading / normalising the saved score matrices of the dense experiments,
* column-aware post-processing (CSLS, dual-softmax, Sinkhorn) and the hard / soft rules,
* feature stacking over a reranking depth and the LR / HGB pointwise rankers.
"""
import os

import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pcr.training.trainer import exp_dir
from pcr.utils import zs


def available_experiments(ctx, ids=("E1", "E2", "E3", "E4"), need="dev_ms.npy"):
    """List the experiments whose score files exist on disk.

    Args:
        ctx (PCRContext): Pipeline context.
        ids (Iterable[str]): Candidate experiment ids.
        need (str): File that must exist in the experiment folder.

    Returns:
        List[str]: Experiment ids that have ``need`` available.
    """
    return [e for e in ids if os.path.exists(f"{exp_dir(ctx.cfg, e)}/{need}")]


def load_scores(ctx, e, split, suffix="scores", normalise=True):
    """Load the saved score matrix of an experiment.

    Args:
        ctx (PCRContext): Pipeline context.
        e (str): Experiment id.
        split (str): ``dev`` | ``test``.
        suffix (str): ``scores`` (document-mean cosine) or ``ms`` (MaxSim).
        normalise (bool): Row z-score the matrix.

    Returns:
        np.ndarray: Score matrix (float32).
    """
    M = np.load(f"{exp_dir(ctx.cfg, e)}/{split}_{suffix}.npy").astype(np.float32)
    return zs(M) if normalise else M


def signals(ctx, split, exps):
    """Row-normalised signals of a split: dense (cosine + MaxSim) per experiment, BM25 and TF-IDF.

    Args:
        ctx (PCRContext): Pipeline context (``ctx.lex`` must hold ``bm25`` and ``tfidf``).
        split (str): ``dev`` | ``test``.
        exps (List[str]): Dense experiment ids.

    Returns:
        Dict[str, np.ndarray]: ``name -> z-scored matrix`` with keys ``E*``, ``E*_ms``, ``bm25``, ``tfidf``.
    """
    g = {}
    for e in exps:
        g[e] = load_scores(ctx, e, split, "scores")
        g[e + "_ms"] = load_scores(ctx, e, split, "ms")
    g["bm25"] = zs(ctx.lex["bm25"][split])
    g["tfidf"] = zs(ctx.lex["tfidf"][split])
    return g


def stack_top(feats, names, top, rows=None):
    """Gather feature values of the top-R candidates of every query.

    Args:
        feats (Dict[str, np.ndarray]): ``name -> (n_queries, n_candidates)`` matrices.
        names (List[str]): Feature names (order of the last axis).
        top (np.ndarray): Candidate indices (n_queries, R).
        rows (Optional[Iterable[int]]): Query rows to use (default: all rows of ``top``).

    Returns:
        np.ndarray: Tensor (len(rows), R, len(names)).
    """
    rows = range(top.shape[0]) if rows is None else rows
    return np.stack([np.stack([feats[n][i, top[i]] for n in names], 1) for i in rows])


# ---------------- column-aware post-processing ----------------
def csls(L, k, a):
    """CSLS-style hubness correction: subtract ``a`` x mean of the column's top-k scores.

    Args:
        L (np.ndarray): Score matrix (n_queries, n_candidates).
        k (int): Number of top queries averaged per candidate column.
        a (float): Penalty strength.

    Returns:
        np.ndarray: Corrected scores.
    """
    return L - a * np.sort(L, 0)[-k:, :].mean(0, keepdims=True)


def dual(L, tau, lam):
    """Dual-softmax: row log-softmax + ``lam`` x column log-softmax of ``L / tau``.

    Args:
        L (np.ndarray): Score matrix.
        tau (float): Temperature.
        lam (float): Weight of the column term.

    Returns:
        np.ndarray: Transformed scores.
    """
    A = L / tau
    return (A - logsumexp(A, 1, keepdims=True)) + lam * (A - logsumexp(A, 0, keepdims=True))


def sinkhorn(L, G, tau, it=15):
    """Sinkhorn normalisation to the expected row / column gold marginals (used in the audit).

    Args:
        L (np.ndarray): Score matrix (n_queries, n_candidates).
        G (np.ndarray): Bool gold matrix (only its total count is used for the marginals).
        tau (float): Temperature.
        it (int): Number of alternating normalisations.

    Returns:
        np.ndarray: Normalised log-scores.
    """
    nq, nc = L.shape
    A = L / tau
    r = np.log(G.sum() / nq)
    c = np.log(G.sum() / nc)
    for _ in range(it):
        A = A - logsumexp(A, 1, keepdims=True) + r
        A = A - logsumexp(A, 0, keepdims=True) + c
    return A


def apply_rules(L, ctx, split):
    """Apply the citation / chronology rules: +6 for citation hits, -50 for later (non-prior) candidates.

    Args:
        L (np.ndarray): Score matrix.
        ctx (PCRContext): Context providing ``ctx.ex[split]`` (fwd, rev) and ``ctx.rf[split]`` (cite, future).
        split (str): Split name.

    Returns:
        np.ndarray: Scores with the rules applied.
    """
    pos = (ctx.ex[split]["fwd"] + ctx.rf[split]["cite"]) > 0
    neg = (ctx.ex[split]["rev"] + ctx.rf[split]["future"]) > 0
    return L + 6.0 * pos - 50.0 * neg


def post(L, ctx, split, kind, p1, p2, use_r):
    """Optional column-aware correction (``csls`` | ``dual`` | none) followed by optional rules.

    Args:
        L (np.ndarray): Score matrix.
        ctx (PCRContext): Pipeline context.
        split (str): Split name.
        kind (str): ``csls``, ``dual`` or anything else for no correction.
        p1 (float): ``k`` for CSLS or ``tau`` for dual-softmax.
        p2 (float): ``a`` for CSLS or ``lam`` for dual-softmax.
        use_r (bool): Apply :func:`apply_rules` afterwards.

    Returns:
        np.ndarray: Post-processed scores.
    """
    L = csls(L, int(p1), float(p2)) if kind == "csls" else dual(L, p1, p2) if kind == "dual" else L
    return apply_rules(L, ctx, split) if use_r else L


# ---------------- pointwise rankers ----------------
def fit_lr(X, y, C=0.1):
    """Fit a standardised logistic-regression pointwise ranker on (queries x depth x features) data.

    Args:
        X (np.ndarray): Features (n_queries, R, F).
        y (np.ndarray): Binary labels (n_queries, R).
        C (float): Inverse L2 regularisation strength.

    Returns:
        Tuple[StandardScaler, LogisticRegression]: Fitted scaler and classifier.
    """
    F_ = X.shape[2]
    sc = StandardScaler().fit(X.reshape(-1, F_))
    return sc, LogisticRegression(C=C, max_iter=5000, random_state=0).fit(sc.transform(X.reshape(-1, F_)), y.reshape(-1))


def fit2(kind, X, y, C=0.1):
    """Fit an LR or HGB pointwise ranker and return a scoring function.

    Args:
        kind (str): ``lr`` (logistic regression) or anything else for histogram gradient boosting.
        X (np.ndarray): Features (n_queries, R, F).
        y (np.ndarray): Labels (n_queries, R).
        C (float): LR regularisation (ignored by HGB).

    Returns:
        Callable[[np.ndarray], np.ndarray]: Function mapping features (R, F) or (n, R, F) to decision values.
    """
    F_ = X.shape[2]
    sc = StandardScaler().fit(X.reshape(-1, F_))
    Z = sc.transform(X.reshape(-1, F_))
    yy = y.reshape(-1)
    m = LogisticRegression(C=C, max_iter=5000, random_state=0).fit(Z, yy) if kind == "lr" else \
        HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=0).fit(Z, yy)
    return lambda Xq: m.decision_function(sc.transform(Xq))
