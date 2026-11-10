"""Per-class confidence thresholds and the decoder-variant comparison (A greedy / B viterbi / C +thr / D +thr+refine).

Macro-F1 is an average of per-label F1s and removing a span only affects its own label's F1, so the optimum
factorises: each label's threshold is tuned independently.
"""
import numpy as np

from .. import state as S
from ..eval.metrics import strict_scores
from .decoders import decode_scored, apply_thresh

THRESH_GRID = np.round(np.arange(0.0, 0.951, 0.05), 2)


def tune_thresholds(ids, scored, cfg):
    """Tune one confidence threshold per label (centre of the F1 plateau, to limit validation over-fit).

    Args:
        ids (list[str]): doc ids used for tuning.
        scored (dict): ``{doc: [(s, e, l, conf)]}``.
        cfg (dict): config (``THRESH_MIN_SUPPORT``, ``THRESH_MIN_GAIN``).

    Returns:
        dict: label -> threshold (0.0 = keep everything).
    """
    min_sup, min_gain = cfg.get("THRESH_MIN_SUPPORT", 8), cfg.get("THRESH_MIN_GAIN", 0.01); thr = {}
    for l in S.LABELS:
        G = {(d, s, e) for d in ids for s, e, lab in S.GOLD[d] if lab == l}
        P = [(d, s, e, c) for d in ids for s, e, lab, c in scored[d] if lab == l]
        if len(G) < min_sup or not P: thr[l] = 0.0; continue          # too little evidence to tune -> keep everything
        f1s = []
        for t in THRESH_GRID:
            sel = {(d, s, e) for d, s, e, c in P if c >= t}; tp = len(sel & G)
            p = tp / len(sel) if sel else 0.0; r = tp / len(G); f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        f1s = np.array(f1s); best = f1s.max()
        plateau = THRESH_GRID[f1s >= best - 0.005]                    # centre of the plateau -> less val-overfit than the argmax
        thr[l] = float(plateau[len(plateau) // 2]) if best - f1s[0] >= min_gain else 0.0
    return thr


def crossfit_f1(ids, scored, cfg):
    """Honest estimate of 'tune thresholds on val': tune on half the docs, apply to the other half (and vice versa).

    Args:
        ids (list[str]): validation doc ids.
        scored (dict): ``{doc: [(s, e, l, conf)]}``.
        cfg (dict): config.

    Returns:
        float: strict macro-F1 of the cross-fitted predictions.
    """
    ids = sorted(ids); halves = [ids[0::2], ids[1::2]]; pred = {}
    for a, b in ((0, 1), (1, 0)):
        thr = tune_thresholds(halves[a], {d: scored[d] for d in halves[a]}, cfg)
        pred.update(apply_thresh({d: scored[d] for d in halves[b]}, thr))
    return strict_scores({d: S.GOLD[d] for d in ids}, pred, S.LABELS)["macro"]["f1"]


def decode_specs(task):
    """Decoder variants to compare.

    Args:
        task (str): ``"bio"`` or ``"span"``.

    Returns:
        list[tuple]: ``(name, DECODE, BOUNDARY_REFINE, use_confidence_thresholds)``.
    """
    if task == "span":
        return [("S    span", None, False, False), ("S+R  span+refine", None, True, False),
                ("S+T  span+thr", None, False, True), ("S+T+R span+thr+refine", None, True, True)]
    return [("A greedy", "greedy", False, False), ("B viterbi", "viterbi", False, False),
            ("B+R viterbi+refine", "viterbi", True, False), ("C viterbi+thr", "viterbi", False, True),
            ("D viterbi+thr+refine", "viterbi", True, True)]


def finalize_decoding(ids, probs, cfg, margin=0.002):
    """Run every decoder variant on the SAME cached probabilities, print a table, return the best settings.

    Variants with tuned thresholds are ranked by their cross-fitted F1, not the in-sample one.

    Args:
        ids (list[str]): validation doc ids.
        probs (list[np.ndarray]): per-document probabilities aligned with ``ids``.
        cfg (dict): config (``AUTO_DECODE``, ``TASK`` ...).
        margin (float): a variant must beat the incumbent by this margin to be selected.

    Returns:
        tuple[dict, list[dict]]: (best settings ``{DECODE, BOUNDARY_REFINE, CONF_THRESH}`` or ``{}``, table rows).
    """
    if not cfg.get("AUTO_DECODE", True): return {}, []
    gold = {d: S.GOLD[d] for d in ids}; rows, best = [], None
    for name, dec, ref, use_thr in decode_specs(cfg.get("TASK", "bio")):
        vcfg = {**cfg, "CONF_THRESH": None, "BOUNDARY_REFINE": ref}
        if dec: vcfg["DECODE"] = dec
        scored = {d: decode_scored(S.DOCS[d], p, vcfg) for d, p in zip(ids, probs)}
        thr = tune_thresholds(ids, scored, cfg) if use_thr else None
        sc = strict_scores(gold, apply_thresh(scored, thr), S.LABELS)["macro"]
        honest = crossfit_f1(ids, scored, cfg) if use_thr else None
        sel = sc["f1"] if honest is None else honest
        rows.append(dict(variant=name, p=sc["p"], r=sc["r"], f1=sc["f1"], f1_crossfit=honest))
        if best is None or sel > best[0] + margin:
            best = (sel, dict(DECODE=vcfg.get("DECODE", cfg.get("DECODE", "viterbi")), BOUNDARY_REFINE=ref, CONF_THRESH=thr), name)
    print(f"{'decoder variant':26s} {'P':>6s} {'R':>6s} {'F1':>7s} {'F1 (cross-fit)':>15s}")
    for r in rows:
        cf = f"{r['f1_crossfit']:.4f}" if r["f1_crossfit"] is not None else "      -"
        print(f"{r['variant']:26s} {r['p']:6.3f} {r['r']:6.3f} {r['f1']:7.4f} {cf:>15s}")
    print(f"-> selected: {best[2]} | refine={best[1]['BOUNDARY_REFINE']} | thresholds={best[1]['CONF_THRESH']}")
    return best[1], rows
