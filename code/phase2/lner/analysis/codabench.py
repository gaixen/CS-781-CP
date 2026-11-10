"""Analysis of the best Codabench submission (strict macro-F1 0.5849): where points are lost and the headroom per label."""

from collections import Counter

from .. import state as S

CODABENCH = {  # label: (P, R, F1, TP, FP, FN)
    "APP": (0.238, 0.158, 0.19, 40, 128, 213),
    "RESP": (0.222, 0.289, 0.251, 39, 137, 96),
    "A.COUNSEL": (0.626, 0.828, 0.713, 77, 46, 16),
    "R.COUNSEL": (0.386, 0.62, 0.476, 44, 70, 27),
    "JUDGE": (0.699, 0.769, 0.732, 93, 40, 28),
    "WIT": (0.433, 0.336, 0.378, 42, 55, 83),
    "AUTH": (0.448, 0.605, 0.514, 205, 253, 134),
    "COURT": (0.812, 0.902, 0.854, 504, 117, 55),
    "STAT": (0.718, 0.857, 0.781, 681, 268, 114),
    "PREC": (0.521, 0.552, 0.536, 343, 315, 278),
    "DATE": (0.959, 0.954, 0.957, 707, 30, 34),
    "CASENO": (0.634, 0.637, 0.636, 218, 126, 124),
}


def codabench_analysis():
    """Print per-label Codabench scores sorted by F1 with the macro-F1 headroom if every label reached 0.60, and the APP:RESP prior-shift hypothesis.

    Args:
        None

    Returns:
        None
    """
    print(
        f"{'label':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'gold':>5s} {'pred':>5s} | headroom if F1->0.60 (macro pts)"
    )
    for l, (p, r, f, tp, fp, fn) in sorted(CODABENCH.items(), key=lambda kv: kv[1][2]):
        print(
            f"{l:10s} {p:6.3f} {r:6.3f} {f:6.3f} {tp + fn:5d} {tp + fp:5d} | {max(0, 0.60 - f) / len(CODABENCH):+.4f}"
        )
    tot_gain = sum(
        max(0, 0.60 - v[2]) for v in CODABENCH.values() if v[2] < 0.60
    ) / len(CODABENCH)
    print(f"macro-F1 if every label below 0.60 reached 0.60: {0.5849 + tot_gain:.3f}")
    if S.GOLD:
        n = Counter(l for d in S.TEXTS for _, _, l in S.GOLD[d])
        A, R = CODABENCH["APP"], CODABENCH["RESP"]
        print(
            f"APP:RESP  train+val gold = {n['APP']}:{n['RESP']} ({n['APP'] / n['RESP']:.2f})  |  test gold = {A[3] + A[5]}:{R[3] + R[5]} ({(A[3] + A[5]) / (R[3] + R[5]):.2f})  |  predicted on test = {A[3] + A[4]}:{R[3] + R[4]}"
        )
    print(
        "local (35-doc) validation is ~0.53 vs 0.585 on test: treat local numbers as +-0.02."
    )
