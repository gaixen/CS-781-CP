"""Strict (doc, start, end, label) macro-F1 and false-positive error analysis."""
from collections import Counter, defaultdict

import numpy as np


def strict_scores(gold, pred, labels):
    """Strict macro P/R/F1: a prediction counts only if ``(doc, start, end, label)`` matches exactly.

    The macro average is over labels that have gold support (confirm against the official scorer on Codabench).

    Args:
        gold (dict): ``{doc: [(s, e, label)]}``.
        pred (dict): ``{doc: [(s, e, label)]}``.
        labels (list[str]): all labels.

    Returns:
        dict: ``{"macro": {p, r, f1}, "per_label": {label: {p, r, f1, support, n_pred}}}``.
    """
    per = {}
    for l in labels:
        G = {(d, s, e) for d, ents in gold.items() for s, e, lab in ents if lab == l}
        P = {(d, s, e) for d, ents in pred.items() for s, e, lab in ents if lab == l}
        tp = len(G & P); p = tp / len(P) if P else 0.0; r = tp / len(G) if G else 0.0
        per[l] = dict(p=p, r=r, f1=2 * p * r / (p + r) if p + r else 0.0, support=len(G), n_pred=len(P))
    used = [l for l in labels if per[l]["support"] > 0]
    macro = {k: float(np.mean([per[l][k] for l in used])) for k in ("p", "r", "f1")}
    return dict(macro=macro, per_label=per)


def print_scores(sc):
    """Print a per-label score table.

    Args:
        sc (dict): output of ``strict_scores``.

    Returns:
        None
    """
    print(f"{'label':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'gold':>6s} {'pred':>6s}")
    for l, v in sc["per_label"].items(): print(f"{l:10s} {v['p']:6.3f} {v['r']:6.3f} {v['f1']:6.3f} {v['support']:6d} {v['n_pred']:6d}")
    m = sc["macro"]; print(f"{'MACRO':10s} {m['p']:6.3f} {m['r']:6.3f} {m['f1']:6.3f}")


def error_analysis(gold, pred):
    """Bucket false positives and count misses per label.

    Buckets: ``label_confusion`` (exact span, wrong label), ``boundary_error`` (same label, overlapping),
    ``overlap_other_label``, ``spurious`` (no overlap = boilerplate-type FPs).

    Args:
        gold (dict): ``{doc: [(s, e, label)]}``.
        pred (dict): ``{doc: [(s, e, label)]}``.

    Returns:
        tuple: (category Counter, confusion Counter, per-label Counter dict).
    """
    cats, conf, per = Counter(), Counter(), defaultdict(Counter)
    for d in set(gold) | set(pred):
        G = set(gold.get(d, [])); P = set(pred.get(d, [])); by_span = {(s, e): l for s, e, l in G}
        for s, e, l in P - G:
            if (s, e) in by_span: c = "label_confusion"; conf[(by_span[(s, e)], l)] += 1
            elif any(l == gl and s < ge and gs < e for gs, ge, gl in G): c = "boundary_error"
            elif any(s < ge and gs < e for gs, ge, gl in G): c = "overlap_other_label"
            else: c = "spurious"
            cats[c] += 1; per[l][c] += 1
        for s, e, l in G - P: per[l]["missed"] += 1
    print("false-positive types:", dict(cats))
    print("top label confusions (gold -> pred):", conf.most_common(8))
    for l in sorted(per): print(f"  {l:10s}", dict(per[l]))
    return cats, conf, per
