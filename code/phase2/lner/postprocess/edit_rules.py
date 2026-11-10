"""Exact-string propagation and learned boundary edit rules (transformation-based, gold-invariance gated)."""
import re
from collections import Counter, defaultdict

import numpy as np


def propagate(text, spans, labels, min_len=5):
    """Add every exact (word-bounded) re-occurrence of a predicted string that has no prediction on it yet.

    A string predicted with several labels keeps its majority label.

    Args:
        text (str): document text.
        spans (list[tuple]): ``(s, e, label)``.
        labels (set[str]): labels to propagate.
        min_len (int): minimum string length.

    Returns:
        list[tuple]: sorted spans including the propagated ones.
    """
    occ = np.zeros(len(text) + 1, bool)
    for s, e, l in spans: occ[s:e] = True
    votes = defaultdict(Counter)
    for s, e, l in spans:
        if l in labels and e - s >= min_len: votes[text[s:e]][l] += 1
    out = list(spans)
    for st, c in votes.items():
        lab = c.most_common(1)[0][0]
        for m in re.finditer(r"(?<!\w)" + re.escape(st) + r"(?!\w)", text):
            a, b = m.span()
            if not occ[a:b].any(): out.append((a, b, lab)); occ[a:b] = True
    return sorted(out)


def shp(x):
    """Shape of a string (digits -> 0, max 16 chars).

    Args:
        x (str): string.

    Returns:
        str: shape string.
    """
    return re.sub(r"\d", "0", x)[:16]


def _pat(sh):
    """Regex pattern for a shape string.

    Args:
        sh (str): output of ``shp``.

    Returns:
        str: regex.
    """
    return "".join(r"\d" if c == "0" else re.escape(c) for c in sh)


def apply_rule(text, s, e, rule):
    """Apply one learned edit rule ``(label, kind, pattern)`` to a span.

    Args:
        text (str): document text.
        s (int): span start.
        e (int): span end (exclusive).
        rule (tuple): ``(label, kind, pattern)`` with kind in trim_right|trim_left|extend_right|extend_left.

    Returns:
        tuple[int, int]: edited ``(s, e)``.
    """
    kind, pat = rule[1], rule[2]
    if kind == "trim_right":
        m = re.search("(?:" + pat + ")$", text[s:e]);  return (s, s + m.start()) if m and m.start() > 0 else (s, e)
    if kind == "trim_left":
        m = re.match(pat, text[s:e]);                  return (s + m.end(), e) if m and s + m.end() < e else (s, e)
    if kind == "extend_right":
        m = re.compile(pat).match(text, e);            return (s, m.end()) if m else (s, e)
    if kind == "extend_left":
        m = re.search("(?:" + pat + ")$", text[max(0, s - 24):s]); return (s - (len(text[max(0, s - 24):s]) - m.start()), e) if m else (s, e)
    return (s, e)


def learn_edit_rules(texts, gold, pred, labels, min_net=4, max_break_frac=0.1, max_gold_change=0.001, max_rules=12):
    """Greedy rule learner.

    A candidate (label, kind, pattern) is mined from boundary-error pairs, then accepted only if net fixes >= ``min_net``,
    breaks <= ``max_break_frac`` * fixes AND it changes <= 0.1% of the label's correct GOLD spans.

    Args:
        texts (dict): doc -> text.
        gold (dict): doc -> gold spans.
        pred (dict): doc -> predicted spans.
        labels (list[str]): labels to learn rules for.
        min_net (int): minimum net fixes.
        max_break_frac (float): max breaks / fixes.
        max_gold_change (float): max fraction of correct gold spans that may change.
        max_rules (int): max rules per label.

    Returns:
        list[tuple]: accepted rules.
    """
    gset = {d: set(v) for d, v in gold.items()}; P = {d: list(set(v)) for d, v in pred.items()}; rules = []
    for l in labels:
        n_gold = sum(1 for d in gold for g in gold[d] if g[2] == l)
        cand = Counter()
        for d in P:
            for s, e, lab in P[d]:
                if lab != l or (s, e, lab) in gset[d]: continue
                g = next((g for g in gold[d] if g[2] == l and g[0] < e and s < g[1]), None)
                if not g: continue
                t = texts[d]
                if e > g[1]: cand[(l, "trim_right", _pat(shp(t[g[1]:e])))] += 1
                elif e < g[1]: cand[(l, "extend_right", _pat(shp(t[e:g[1]])))] += 1
                if s < g[0]: cand[(l, "trim_left", _pat(shp(t[s:g[0]])))] += 1
                elif s > g[0]: cand[(l, "extend_left", _pat(shp(t[g[0]:s])))] += 1
        for _ in range(max_rules):
            best = None
            for rule, cnt in cand.items():
                if cnt < min_net: continue
                fix = brk = 0
                for d in P:
                    for s, e, lab in P[d]:
                        if lab != l: continue
                        ns, ne = apply_rule(texts[d], s, e, rule)
                        if (ns, ne) == (s, e): continue
                        was, now = (s, e, l) in gset[d], (ns, ne, l) in gset[d]
                        fix += now and not was; brk += was and not now
                gch = sum(apply_rule(texts[d], g[0], g[1], rule) != (g[0], g[1]) for d in gold for g in gold[d] if g[2] == l)
                if fix - brk >= min_net and brk <= max_break_frac * fix and gch <= max_gold_change * max(n_gold, 1):
                    if best is None or fix - brk > best[0]: best = (fix - brk, rule, fix, brk, gch)
            if best is None: break
            _, rule, fix, brk, gch = best; rules.append(rule); cand.pop(rule)
            P = {d: [(*apply_rule(texts[d], s, e, rule), lab) if lab == l else (s, e, lab) for s, e, lab in v] for d, v in P.items()}
            print(f"  rule {rule}: +{fix} fixes, -{brk} breaks, changes {gch} gold spans")
    return rules
