"""Evaluation metrics: Micro-F1@K (primary), Precision@K, Recall@K and recall-at-depth curves."""
import numpy as np


def f1_at_k(S, spl, split, k=7):
    """Micro-averaged P / R / F1 at cut-off ``k`` over all queries of a split.

    Args:
        S (np.ndarray): Score matrix (n_queries, n_candidates) aligned with ``spl[split]['qids'/'cids']``.
        spl (dict): Split container.
        split (str): ``train`` | ``dev`` (a split with gold labels).
        k (int): Cut-off.

    Returns:
        Dict[str, float]: ``{"f1": ..., "p": ..., "r": ...}``.
    """
    cids, qids, gold = np.array(spl[split]["cids"]), spl[split]["qids"], spl[split]["gold"]
    top = np.argsort(-S, 1)[:, :k]
    tp = 0
    ng = 0
    for qi, q in enumerate(qids):
        tp += len(set(cids[top[qi]]) & gold[q])
        ng += len(gold[q])
    P, R = tp / (k * len(qids)), tp / max(ng, 1)
    return dict(f1=2 * P * R / max(P + R, 1e-9), p=P, r=R)


def recall_at(S, spl, split, ks=(7, 20, 50, 100)):
    """Mean per-query recall at several depths (re-ranking headroom).

    Args:
        S (np.ndarray): Score matrix (n_queries, n_candidates).
        spl (dict): Split container.
        split (str): Split with gold labels.
        ks (Iterable[int]): Depths to evaluate.

    Returns:
        Dict[int, float]: ``depth -> mean recall``.
    """
    cids, qids, gold = np.array(spl[split]["cids"]), spl[split]["qids"], spl[split]["gold"]
    o = np.argsort(-S, 1)
    return {k: round(float(np.mean([len(set(cids[o[i, :k]]) & gold[q]) / max(len(gold[q]), 1) for i, q in enumerate(qids)])), 4) for k in ks}


def gold_matrix(split, spl):
    """Boolean gold matrix of a split.

    Args:
        split (str): ``train`` | ``dev``.
        spl (dict): Split container.

    Returns:
        np.ndarray: Bool matrix (n_queries, n_candidates), True where the candidate is relevant.
    """
    S = spl[split]
    ci = {c: j for j, c in enumerate(S["cids"])}
    M = np.zeros((len(S["qids"]), len(S["cids"])), bool)
    for i, q in enumerate(S["qids"]):
        for g in S["gold"][q]:
            M[i, ci[g]] = True
    return M


def label_matrix(top, spl, split="dev"):
    """Binary relevance labels of the candidates selected by a reranking depth.

    Args:
        top (np.ndarray): Candidate column indices per query (n_queries, depth).
        spl (dict): Split container.
        split (str): Split with gold labels.

    Returns:
        np.ndarray: Int matrix (n_queries, depth) with 1 for relevant candidates.
    """
    S = spl[split]
    cids = np.array(S["cids"])
    return np.array([[cids[j] in S["gold"][q] for j in top[i]] for i, q in enumerate(S["qids"])]).astype(int)
