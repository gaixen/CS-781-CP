"""Label-specific character-boundary refinement.

Learned ONLY from the training gold: for every label we estimate how its spans START / END (first/last character class,
and whether the boundary falls in the middle of a word). A predicted span is re-snapped to the best boundary within
+-REFINE_K characters, where ``score = log prior(label, char-class) + log prior(label, mid-word?) - REFINE_LAMBDA * |shift|``.
The unshifted boundary has zero penalty, so a boundary only moves when the evidence is strong.
"""
import math
from collections import Counter, defaultdict

from .. import state as S

CHAR_CLASSES = ["sp", "al", "di", "op", "cl", "dot", "pu", "qu", "ot"]


def char_cls(ch):
    """Coarse character class.

    Args:
        ch (str): a single character.

    Returns:
        str: one of ``CHAR_CLASSES``.
    """
    if ch.isspace(): return "sp"
    if ch.isalpha(): return "al"
    if ch.isdigit(): return "di"
    if ch in "([{": return "op"
    if ch in ")]}": return "cl"
    if ch == ".": return "dot"
    if ch in ",;:": return "pu"
    if ch in "\"'`\u201c\u201d\u2018\u2019": return "qu"
    return "ot"


def build_boundary_prior(ids, alpha=0.5):
    """Estimate per-label boundary priors from training gold.

    Args:
        ids (list[str]): training doc ids.
        alpha (float): additive smoothing.

    Returns:
        dict: ``label -> {start, end, mid_s, mid_e}`` log-prior tables.
    """
    cs, ce, ms, me, n = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()
    for d in ids:
        t = S.TEXTS[d]; N = len(t)
        for s, e, l in S.GOLD[d]:
            n[l] += 1
            cs[l][char_cls(t[s])] += 1; ce[l][char_cls(t[e - 1])] += 1
            ms[l][int(s > 0 and t[s - 1].isalnum() and t[s].isalnum())] += 1
            me[l][int(e < N and t[e - 1].isalnum() and t[e].isalnum())] += 1
    pri = {}
    for l in n:
        tot = n[l] + alpha * len(CHAR_CLASSES)
        pri[l] = dict(
            start={c: math.log((cs[l][c] + alpha) / tot) for c in CHAR_CLASSES},
            end={c: math.log((ce[l][c] + alpha) / tot) for c in CHAR_CLASSES},
            mid_s={v: math.log((ms[l][v] + alpha) / (n[l] + 2 * alpha)) for v in (0, 1)},
            mid_e={v: math.log((me[l][v] + alpha) / (n[l] + 2 * alpha)) for v in (0, 1)})
    return pri


def refine_span(text, s, e, label, cfg):
    """Snap a span's boundaries to the most plausible position within +-``REFINE_K`` characters.

    Args:
        text (str): document text.
        s (int): span start.
        e (int): span end (exclusive).
        label (str): span label.
        cfg (dict): config (``REFINE_K``, ``REFINE_LAMBDA``).

    Returns:
        tuple[int, int]: refined ``(s, e)`` (unchanged if no prior exists or the result would be empty).
    """
    pri = S.BOUNDARY_PRIOR.get(label)
    if pri is None: return s, e
    K, lam, N = cfg.get("REFINE_K", 3), cfg.get("REFINE_LAMBDA", 1.5), len(text)
    def s_score(i):
        if not (0 <= i < N): return -1e9
        mid = int(i > 0 and text[i - 1].isalnum() and text[i].isalnum())
        return pri["start"][char_cls(text[i])] + pri["mid_s"][mid]
    def e_score(j):                       # j is the exclusive end
        if not (0 < j <= N): return -1e9
        mid = int(j < N and text[j - 1].isalnum() and text[j].isalnum())
        return pri["end"][char_cls(text[j - 1])] + pri["mid_e"][mid]
    order = sorted(range(-K, K + 1), key=abs)            # unshifted first -> wins ties
    bs = s + max(order, key=lambda d: s_score(s + d) - lam * abs(d))
    be = e + max(order, key=lambda d: e_score(e + d) - lam * abs(d))
    return (bs, be) if be > bs else (s, e)
