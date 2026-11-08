import numpy as np

GRID = np.round(np.arange(0.05, 0.951, 0.025), 4)
GRID_B = np.round(np.arange(0.05, 0.951, 0.05), 4)
ST2 = ["g0.50", "global", "global+top1", "pl8", "pl30", "bagpl30", "bagpl30+top1"]
THR_SLUG = {
    "g0.50": "g05",
    "global": "glob",
    "global+top1": "globTop1",
    "pl8": "PL8",
    "pl30": "PL30",
    "bagpl30": "bagPL30",
    "bagpl30+top1": "bagPL30top1",
}


def sigmoid(z):
    """Numerically clipped logistic function.

    Args:
        z (np.ndarray): Logits.

    Returns:
        np.ndarray: Probabilities.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _counts(B, Y):
    """Per-label true positives, false positives and false negatives.

    Args:
        B (np.ndarray): Binary predictions (n, L).
        Y (np.ndarray): Binary references (n, L).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(tp, fp, fn)`` each of shape (L,).
    """
    B = B.astype(np.float32)
    return (B * Y).sum(0), (B * (1 - Y)).sum(0), ((1 - B) * Y).sum(0)


def f1_labels(B, Y):
    """Per-label F1.

    Args:
        B (np.ndarray): Predictions (n, L).
        Y (np.ndarray): References (n, L).

    Returns:
        np.ndarray: F1 per label (L,).
    """
    tp, fp, fn = _counts(B, Y)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def macro_f1(B, Y):
    """Macro-F1 over labels with at least one reference positive.

    Args:
        B (np.ndarray): Predictions.
        Y (np.ndarray): References.

    Returns:
        float: Macro-F1.
    """
    sup = Y.sum(0) > 0
    return float(f1_labels(B, Y)[sup].mean())


def micro_f1(B, Y):
    """Micro-F1.

    Args:
        B (np.ndarray): Predictions.
        Y (np.ndarray): References.

    Returns:
        float: Micro-F1.
    """
    tp, fp, fn = _counts(B, Y)
    return float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1))


def subset_acc(B, Y):
    """Exact-match (subset) accuracy.

    Args:
        B (np.ndarray): Predictions.
        Y (np.ndarray): References.

    Returns:
        float: Share of documents whose predicted label set equals the reference.
    """
    return float((B.astype(np.float32) == Y).all(1).mean())


def macro_pr(B, Y):
    """Macro precision and recall over supported labels.

    Args:
        B (np.ndarray): Predictions.
        Y (np.ndarray): References.

    Returns:
        Tuple[float, float]: ``(macro_precision, macro_recall)``.
    """
    tp, fp, fn = _counts(B, Y)
    sup = Y.sum(0) > 0
    return float((tp / np.maximum(tp + fp, 1))[sup].mean()), float(
        (tp / np.maximum(tp + fn, 1))[sup].mean()
    )


def all_metrics(B, Y):
    """Macro / micro F1, subset accuracy, macro P and R.

    Args:
        B (np.ndarray): Predictions.
        Y (np.ndarray): References.

    Returns:
        Dict[str, float]: ``macro_f1, micro_f1, subset_acc, macro_p, macro_r``.
    """
    p, r = macro_pr(B, Y)
    return dict(
        macro_f1=macro_f1(B, Y),
        micro_f1=micro_f1(B, Y),
        subset_acc=subset_acc(B, Y),
        macro_p=p,
        macro_r=r,
    )


def tune_global(P, Y, grid=GRID):
    """Best single threshold for macro-F1.

    Args:
        P (np.ndarray): Probabilities.
        Y (np.ndarray): References.
        grid (np.ndarray): Candidate thresholds.

    Returns:
        Tuple[float, float]: ``(threshold, macro_f1)``.
    """
    sc = [macro_f1(P >= t, Y) for t in grid]
    i = int(np.argmax(sc))
    return float(grid[i]), float(sc[i])


def tune_per_label(P, Y, g, grid=GRID, k=8.0):
    """Per-label thresholds shrunk toward the global one by ``n_pos / (n_pos + k)`` (rare labels -> mostly global).

    Args:
        P (np.ndarray): Probabilities.
        Y (np.ndarray): References.
        g (float): Global threshold.
        grid (np.ndarray): Candidate thresholds.
        k (float): Shrinkage strength.

    Returns:
        np.ndarray: Threshold per label (L,).
    """
    f = (
        np.stack([f1_labels(P >= t, Y) for t in grid])
        - 1e-6 * np.abs(grid - g)[:, None]
    )
    best = grid[f.argmax(0)]
    npos = Y.sum(0)
    w = npos / (npos + k)
    thr = w * best + (1 - w) * g
    thr[npos == 0] = g
    return thr


def decide(P, thr, top1=False):
    """Binarise probabilities; optionally force the arg-max label for documents with no prediction.

    Args:
        P (np.ndarray): Probabilities (n, L).
        thr (Union[float, np.ndarray]): Global or per-label thresholds.
        top1 (bool): Give empty documents their best-margin label.

    Returns:
        np.ndarray: Boolean predictions (n, L).
    """
    B = P >= thr
    if top1:
        e = np.where(~B.any(1))[0]
        if len(e):
            B[
                e,
                (P[e] - thr).argmax(1)
                if np.ndim(thr) == 0
                else (P[e] - thr[None]).argmax(1),
            ] = True
    return B


def tune_bagged(P, Y, g, k=30.0, B=15, seed=0):
    """Per-label thresholds averaged over bootstrap resamples of dev, then shrunk toward the global threshold.

    Args:
        P (np.ndarray): Probabilities.
        Y (np.ndarray): References.
        g (float): Global threshold.
        k (float): Shrinkage strength.
        B (int): Number of bootstrap resamples.
        seed (int): RNG seed.

    Returns:
        np.ndarray: Threshold per label (L,).
    """
    rs = np.random.RandomState(seed)
    N = len(P)
    acc = np.zeros(P.shape[1])
    for _ in range(B):
        ix = rs.randint(0, N, N)
        Pb, Yb = P[ix], Y[ix]
        f = (
            np.stack([f1_labels(Pb >= t, Yb) for t in GRID_B])
            - 1e-6 * np.abs(GRID_B - g)[:, None]
        )
        acc += GRID_B[f.argmax(0)]
    npos = Y.sum(0)
    w = npos / (npos + k)
    thr = w * (acc / B) + (1 - w) * g
    thr[npos == 0] = g
    return thr


def fit_strategy2(name, P, Y):
    """Fit a threshold strategy of :data:`ST2`.

    Args:
        name (str): Strategy name.
        P (np.ndarray): Probabilities.
        Y (np.ndarray): References.

    Returns:
        Tuple[Union[float, np.ndarray], bool]: ``(thresholds, top1_flag)``.
    """
    if name == "g0.50":
        return 0.5, False
    g, _ = tune_global(P, Y)
    top1 = name.endswith("+top1")
    base = name.replace("+top1", "")
    thr = (
        g
        if base == "global"
        else tune_per_label(P, Y, g, k=8.0)
        if base == "pl8"
        else tune_per_label(P, Y, g, k=30.0)
        if base == "pl30"
        else tune_bagged(P, Y, g, k=30.0)
    )
    return thr, top1


def honest(name, P, Y, K=2, R=3):
    """Out-of-fold macro-F1 of a strategy (fixed-seed K-fold inside dev, R repeats).

    Args:
        name (str): Strategy name.
        P (np.ndarray): Probabilities.
        Y (np.ndarray): References.
        K (int): Number of folds.
        R (int): Repeats.

    Returns:
        Tuple[float, float]: ``(mean, std)`` of the out-of-fold macro-F1.
    """
    sc = []
    for r in range(R):
        perm = np.random.RandomState(7 + r).permutation(len(P))
        B = np.zeros(P.shape, bool)
        for f in np.array_split(perm, K):
            tr = np.setdiff1d(perm, f)
            thr, t1 = fit_strategy2(name, P[tr], Y[tr])
            B[f] = decide(P[f], thr, t1)
        sc.append(macro_f1(B, Y))
    return float(np.mean(sc)), float(np.std(sc))


def choose(tab, margin=0.003):
    """Pick the simplest strategy within ``margin`` of the best honest score.

    Args:
        tab (Dict[str, Tuple[float, float]]): ``strategy -> (mean, std)`` from :func:`honest`.
        margin (float): Tolerance.

    Returns:
        str: Selected strategy name.
    """
    mx = max(v[0] for v in tab.values())
    return next(s for s in ST2 if tab[s][0] >= mx - margin)


def greedy_weights(Zs, Y, iters=10):
    """Greedy forward selection with replacement of ensemble members (integer-count weights).

    Args:
        Zs (List[np.ndarray]): Calibrated logits of each model.
        Y (np.ndarray): References.
        iters (int): Number of selections.

    Returns:
        np.ndarray: Normalised weights (a weak model simply gets 0).
    """
    M = len(Zs)
    cnt = np.zeros(M)
    S = np.zeros_like(Zs[0])
    for it in range(iters):
        best, bm = -1.0, 0
        for m in range(M):
            f = tune_global(sigmoid((S + Zs[m]) / (it + 1)), Y)[1]
            if f > best + 1e-9:
                best, bm = f, m
        cnt[bm] += 1
        S += Zs[bm]
    return cnt / cnt.sum()


def fit_temperature(Z, Y):
    """Temperature that minimises the logistic loss.

    Args:
        Z (np.ndarray): Logits.
        Y (np.ndarray): References.

    Returns:
        float: Temperature in [0.5, 5].
    """
    best, bt = 1e9, 1.0
    for T in np.linspace(0.5, 5, 46):
        z = Z / T
        l = (np.logaddexp(0, z) - Y * z).mean()
        if l < best:
            best, bt = l, float(T)
    return bt


def ece_flat(P, Y, bins=15):
    """Expected calibration error over all (document, label) cells.

    Args:
        P (np.ndarray): Probabilities.
        Y (np.ndarray): References.
        bins (int): Number of bins.

    Returns:
        float: ECE.
    """
    p, y = P.ravel(), Y.ravel()
    b = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
