"""Metrics for rhetorical roles (sentence-level macro-F1) and small decoding statistics (numpy only)."""
from collections import Counter

import numpy as np


def rr_scores(labels, gold, pred):
    """Per-label and macro scores from flat lists of label names.

    ``macro_f1`` averages labels in gold UNION pred (sklearn-style); ``macro_f1_gold`` only labels with gold support.

    Args:
        labels (List[str]): Label names.
        gold (List[str]): Gold labels (flat).
        pred (List[str]): Predicted labels (flat).

    Returns:
        dict: ``per`` (label -> p, r, f1, support, n_pred), ``macro_f1, macro_p, macro_r, macro_f1_gold, acc``.
    """
    per = {}
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        pr, rc = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        per[l] = dict(p=pr, r=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, support=tp + fn, n_pred=tp + fp)
    u = [l for l in labels if per[l]["support"] or per[l]["n_pred"]]
    g_ = [l for l in labels if per[l]["support"]]
    mean = lambda ls, k: float(np.mean([per[l][k] for l in ls])) if ls else 0.0
    return dict(per=per, macro_f1=mean(u, "f1"), macro_p=mean(u, "p"), macro_r=mean(u, "r"), macro_f1_gold=mean(g_, "f1"), acc=float(np.mean([g == p for g, p in zip(gold, pred)])))


def print_scores(sc):
    """Print a per-label table and the macro scores.

    Args:
        sc (dict): Output of :func:`rr_scores`.

    Returns:
        None
    """
    print(f"{'label':24s} {'P':>6s} {'R':>6s} {'F1':>6s} {'gold':>6s} {'pred':>6s}")
    for l, v in sc["per"].items():
        print(f"{l:24s} {v['p']:6.3f} {v['r']:6.3f} {v['f1']:6.3f} {v['support']:6d} {v['n_pred']:6d}")
    print(f"MACRO P/R/F1 = {sc['macro_p']:.4f}/{sc['macro_r']:.4f}/{sc['macro_f1']:.4f} | macro-F1 (gold labels only) {sc['macro_f1_gold']:.4f} | acc {sc['acc']:.4f}")


def confusions(gold, pred, top=10):
    """Print the most frequent (gold -> predicted) errors.

    Args:
        gold (List[str]): Gold labels.
        pred (List[str]): Predicted labels.
        top (int): Number of pairs shown.

    Returns:
        None
    """
    c = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    print("top confusions (gold -> pred):", [(f"{g}->{p}", n) for (g, p), n in c.most_common(top)])


def fast_macro(y, pred, nl):
    """Macro-F1 over classes present in gold or prediction (integer arrays).

    Args:
        y (np.ndarray): Gold ids.
        pred (np.ndarray): Predicted ids.
        nl (int): Number of classes.

    Returns:
        float: Macro-F1.
    """
    f = []
    for c in range(nl):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        if tp + fn + fp:
            f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def prf(y, pred, nl, beta=1.0):
    """Macro precision, recall and F-beta over classes present in gold or prediction.

    Args:
        y (np.ndarray): Gold ids.
        pred (np.ndarray): Predicted ids.
        nl (int): Number of classes.
        beta (float): F-beta parameter (>1 favours recall).

    Returns:
        Tuple[float, float, float]: ``(P, R, F_beta)``.
    """
    P, R, F = [], [], []
    for c in range(nl):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        if tp + fp + fn == 0:
            continue
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        b2 = beta ** 2
        P.append(p_)
        R.append(r_)
        F.append((1 + b2) * tp / max((1 + b2) * tp + b2 * fn + fp, 1e-9))
    return float(np.mean(P)), float(np.mean(R)), float(np.mean(F))


def macro_prf(y, pred, nl):
    """Macro P / R / F1 over labels that have gold support only.

    Args:
        y (np.ndarray): Gold ids.
        pred (np.ndarray): Predicted ids.
        nl (int): Number of classes.

    Returns:
        Tuple[float, float, float]: ``(P, R, F1)``.
    """
    P = R = F = 0.0
    n = 0
    for c in range(nl):
        g = int(np.sum(y == c))
        if g == 0:
            continue
        tp = int(np.sum((y == c) & (pred == c)))
        fp = int(np.sum((y != c) & (pred == c)))
        P += tp / (tp + fp) if tp + fp else 0.0
        R += tp / g
        F += 2 * tp / max(2 * tp + fp + (g - tp), 1)
        n += 1
    return P / n, R / n, F / n


def stickiness(seqs):
    """Probability that the next sentence has the same label as the current one.

    Args:
        seqs (Iterable[Sequence[int]]): Label sequences (one per document).

    Returns:
        float: Share of equal consecutive pairs.
    """
    same = tot = 0
    for s in seqs:
        s = np.asarray(s)
        same += int(np.sum(s[1:] == s[:-1]))
        tot += max(len(s) - 1, 0)
    return same / max(tot, 1)
