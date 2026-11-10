"""Greedy / Viterbi BIO decoding, span cleaning (trailing punctuation, brackets)."""
from collections import Counter

import numpy as np

from .. import state as S


def decode_spans_idx(tag_ids, offs):
    """Greedy BIO -> ``[char_start, char_end, label, first_token_idx, last_token_idx]``.

    B-X starts; I-X continues the same X, otherwise (orphan I) starts a new span; O closes. Empty tokens are skipped.

    Args:
        tag_ids (array-like): per-token tag ids.
        offs (np.ndarray): (n, 2) token char offsets.

    Returns:
        list[list]: decoded spans with token indices.
    """
    spans, cur = [], None
    for i, (t, (s, e)) in enumerate(zip(tag_ids, offs)):
        if e <= s: continue
        tag = S.TAGS[int(t)]
        if tag == "O":
            if cur: spans.append(cur); cur = None
        elif tag.startswith("I-") and cur is not None and cur[2] == tag[2:]:
            cur[1] = e; cur[4] = i
        else:
            if cur: spans.append(cur)
            cur = [s, e, tag[2:], i, i]
    if cur: spans.append(cur)
    return spans


def decode_spans(tag_ids, offs):
    """Greedy BIO decoding to ``(char_start, char_end, label)``.

    Args:
        tag_ids (array-like): per-token tag ids.
        offs (np.ndarray): token char offsets.

    Returns:
        list[tuple[int, int, str]]: spans.
    """
    return [tuple(x[:3]) for x in decode_spans_idx(tag_ids, offs)]


def _transition_tables():
    """BIO transition legality.

    Args:
        None

    Returns:
        tuple[np.ndarray, np.ndarray]: (allowed[i, j] bool matrix, start_ok bool vector). Forbids O->I-X and B-X->I-Y (Y != X).
    """
    C = len(S.TAGS); allowed = np.ones((C, C), bool)
    for j, tj in enumerate(S.TAGS):
        if tj.startswith("I-"):
            for i, ti in enumerate(S.TAGS): allowed[i, j] = ti in (f"B-{tj[2:]}", f"I-{tj[2:]}")
    return allowed, np.array([not t.startswith("I-") for t in S.TAGS])


def viterbi_decode(logp):
    """BIO-constrained Viterbi over (T, C) log-probabilities.

    Args:
        logp (np.ndarray): (T, C) log-probs.

    Returns:
        np.ndarray: (T,) best legal tag path.
    """
    T, C = logp.shape
    allowed, start_ok = _transition_tables(); trans = np.where(allowed, 0.0, -1e4)
    dp = logp[0] + np.where(start_ok, 0.0, -1e4); back = np.zeros((T, C), dtype=np.int32); ar = np.arange(C)
    for t in range(1, T):
        sc = dp[:, None] + trans; back[t] = sc.argmax(0); dp = sc[back[t], ar] + logp[t]
    path = np.zeros(T, dtype=np.int64); path[-1] = dp.argmax()
    for t in range(T - 1, 0, -1): path[t - 1] = back[t][path[t]]
    return path


def learn_trim_chars(ids, chars=".,;:"):
    """Learn, per label, which trailing punctuation may be trimmed (chars that <5% of gold spans end with).

    Args:
        ids (list[str]): training doc ids.
        chars (str): candidate punctuation characters.

    Returns:
        dict: label -> string of trimmable characters.
    """
    n, c = Counter(), Counter()
    for d in ids:
        for s, e, l in S.GOLD[d]:
            n[l] += 1; c[(l, S.TEXTS[d][e - 1])] += 1
    return {l: "".join(ch for ch in chars if c[(l, ch)] / n[l] < 0.05) for l in n}


def clean_span(text, s, e, trim_chars=""):
    """Strip whitespace (and optional trailing punctuation) from a span.

    Args:
        text (str): document text.
        s (int): span start.
        e (int): span end (exclusive).
        trim_chars (str): trailing characters to trim.

    Returns:
        tuple[int, int]: cleaned ``(s, e)``.
    """
    while s < e and text[s].isspace(): s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in trim_chars): e -= 1
    return s, e


def balance_brackets(text, s, e):
    """Fix unbalanced ``()`` / ``[]`` at the span end (extend to include / cut the dangling bracket).

    Args:
        text (str): document text.
        s (int): span start.
        e (int): span end (exclusive).

    Returns:
        tuple[int, int]: adjusted ``(s, e)``.
    """
    seg = text[s:e]
    for o, c in (("(", ")"), ("[", "]")):
        if seg.count(o) > seg.count(c) and text[e:e + 1] == c: e += 1; seg = text[s:e]
        elif seg.count(c) > seg.count(o) and seg.endswith(c): e -= 1; seg = text[s:e]
    return s, e
