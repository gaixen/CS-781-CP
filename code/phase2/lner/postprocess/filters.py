"""Label unification inside a document, FP blacklist and the 'refine only where gold-invariant' gate."""
from collections import Counter, defaultdict

from .. import state as S
from ..decoding.boundary import refine_span


def refine_safe_labels(ids, rcfg, max_rate=0.005):
    """Labels for which boundary refinement almost never changes CORRECT gold spans (<= ``max_rate``).

    Args:
        ids (list[str]): doc ids.
        rcfg (dict): ``{"REFINE_K", "REFINE_LAMBDA"}``.
        max_rate (float): maximum gold-change rate.

    Returns:
        set[str]: safe labels.
    """
    ch, tot = Counter(), Counter()
    for d in ids:
        for s, e, l in S.GOLD[d]:
            tot[l] += 1; ch[l] += refine_span(S.TEXTS[d], s, e, l, rcfg) != (s, e)
    print("refine gold-change rate:", {l: f"{ch[l]}/{tot[l]}" for l in S.LABELS if tot[l]})
    return {l for l in S.LABELS if tot[l] and ch[l] / tot[l] <= max_rate}


def unify_labels(text, spans, labels, min_len=4):
    """Within a document one string normally has ONE role: relabel to the strict-majority label.

    Args:
        text (str): document text.
        spans (list[tuple]): ``(s, e, label)``.
        labels (set[str]): labels subject to unification.
        min_len (int): minimum string length.

    Returns:
        list[tuple]: relabelled spans.
    """
    votes = defaultdict(Counter)
    for s, e, l in spans:
        if l in labels and e - s >= min_len: votes[text[s:e]][l] += 1
    out = []
    for s, e, l in spans:
        v = votes.get(text[s:e])
        if l in labels and v and len(v) > 1:
            top = v.most_common(2)
            if top[0][1] > top[1][1]: l = top[0][0]            # strict majority only
        out.append((s, e, l))
    return out


def learn_fp_blacklist(texts, gold, pred, labels=("STAT", "AUTH", "PREC", "CASENO", "WIT", "APP", "RESP", "JUDGE", "COURT"), min_fp=3, max_prec=0.15):
    """(label, string) pairs that were wrong >= ``min_fp`` times and almost never right in out-of-fold predictions.

    Args:
        texts (dict): doc -> text.
        gold (dict): doc -> gold spans.
        pred (dict): doc -> predicted spans.
        labels (tuple[str]): labels considered.
        min_fp (int): minimum false positives.
        max_prec (float): maximum precision of the pair.

    Returns:
        set[tuple[str, str]]: blacklisted ``(label, lowercase string)`` pairs.
    """
    tp, fp = Counter(), Counter()
    for d, v in pred.items():
        g = set(gold.get(d, []))
        for s, e, l in v:
            if l in labels: (tp if (s, e, l) in g else fp)[(l, texts[d][s:e].strip().lower())] += 1
    return {k for k, n in fp.items() if n >= min_fp and tp[k] / (tp[k] + n) <= max_prec}
