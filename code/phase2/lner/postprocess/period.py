"""Abbreviation-aware period attachment ('Ors.', 'Cr.P.C.' keep their final period)."""
import re
from collections import Counter


def learn_period_table(texts, gold, min_n=3, min_rate=0.8):
    """For every (label, last word): does gold keep the following '.' inside the span, or not (sentence end)?

    Args:
        texts (dict): doc -> text.
        gold (dict): doc -> gold spans.
        min_n (int): minimum support.
        min_rate (float): minimum fraction of cases where the period is kept.

    Returns:
        set[tuple[str, str]]: ``(label, lowercase last word)`` keys where the period is kept.
    """
    inc, exc = Counter(), Counter()
    for d, ents in gold.items():
        t = texts[d]
        for s, e, l in ents:
            sp = t[s:e]; m = re.search(r"([A-Za-z]+)\.$", sp)
            if m: inc[(l, m.group(1).lower())] += 1
            elif e < len(t) and t[e] == ".":
                m = re.search(r"([A-Za-z]+)$", sp)
                if m: exc[(l, m.group(1).lower())] += 1
    return {k for k, n in inc.items() if n >= min_n and n / (n + exc[k]) >= min_rate}


def attach_period(text, s, e, label, table):
    """Extend a span by one character when the next character is a period that belongs to an abbreviation.

    Args:
        text (str): document text.
        s (int): span start.
        e (int): span end (exclusive).
        label (str): span label.
        table (set or None): output of ``learn_period_table``.

    Returns:
        tuple[int, int]: possibly extended ``(s, e)``.
    """
    if table and e < len(text) and text[e] == "." and text[e - 1] != ".":
        m = re.search(r"([A-Za-z]+)$", text[s:e])
        if m and (label, m.group(1).lower()) in table: return s, e + 1
    return s, e
