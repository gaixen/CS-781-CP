"""Scored span decoders.

* bio  : (greedy | BIO-constrained Viterbi) path -> spans; confidence = mean over span tokens of P(B-X)+P(I-X)
* span : probs = [type (O + labels) | start | end]; candidate (i, j, type) with confidence = geometric mean of
         P_start(i), P_end(j), mean_k P_type(k); greedy non-overlapping selection.
"""
import math
from collections import defaultdict

import numpy as np

from .. import state as S
from .boundary import refine_span
from .spans import decode_spans_idx, viterbi_decode, clean_span, balance_brackets


def compute_span_maxlen(ids, mult=1.5, add=3):
    """Max span length in tokens per label (with margin), for the span-head decoder.

    Args:
        ids (list[str]): training doc ids.
        mult (float): safety multiplier.
        add (int): safety additive margin.

    Returns:
        dict: label -> max length in tokens.
    """
    mx = defaultdict(int)
    for d in ids:
        doc = S.DOCS[d]; tags = np.where(doc["tags"] < 0, 0, doc["tags"]); keep = doc["offs"][:, 1] > doc["offs"][:, 0]
        for s, e, l, i0, i1 in decode_spans_idx(tags[keep], doc["offs"][keep]): mx[l] = max(mx[l], i1 - i0 + 1)
    return {l: int(mx[l] * mult) + add for l in S.LABELS}


def _bio_candidates(doc, probs, cfg):
    """Decode BIO probabilities into scored spans.

    Args:
        doc (dict): tokenised document.
        probs (np.ndarray): (n_tokens, n_tags) probabilities.
        cfg (dict): config (``DECODE`` greedy|viterbi).

    Returns:
        list[tuple]: ``(start, end, label, confidence)``.
    """
    keep = doc["offs"][:, 1] > doc["offs"][:, 0]          # drop whitespace-only tokens: entities may span them
    offs, p = doc["offs"][keep], probs[keep]
    if len(offs) == 0: return []
    path = viterbi_decode(np.log(p + 1e-9)) if cfg.get("DECODE", "viterbi") == "viterbi" else p.argmax(-1)
    out = []
    for s, e, l, i0, i1 in decode_spans_idx(path, offs):
        pm = p[i0:i1 + 1, S.TAG2ID[f"B-{l}"]] + p[i0:i1 + 1, S.TAG2ID[f"I-{l}"]]
        out.append((s, e, l, float(pm.mean())))
    return out


def _span_candidates(doc, probs, cfg):
    """Decode span-head probabilities into non-overlapping scored spans.

    Args:
        doc (dict): tokenised document.
        probs (np.ndarray): (n_tokens, N_TYPE + 2) probabilities (type | start | end).
        cfg (dict): config (``SPAN_START_MIN, SPAN_END_MIN, SPAN_MIN_CONF``).

    Returns:
        list[tuple]: ``(start, end, label, confidence)``.
    """
    N_TYPE = S.N_TYPE; LABELS = S.LABELS
    keep = doc["offs"][:, 1] > doc["offs"][:, 0]
    offs, p = doc["offs"][keep], probs[keep]
    T = len(offs)
    if T == 0: return []
    pt, ps, pe = p[:, :N_TYPE], p[:, N_TYPE], p[:, N_TYPE + 1]
    cs = np.vstack([np.zeros((1, N_TYPE - 1)), np.cumsum(np.log(pt[:, 1:] + 1e-9), axis=0)])   # prefix sums of log P(type)
    maxlen = np.array([S.SPAN_MAXLEN.get(l, 64) for l in LABELS]); big = int(maxlen.max())
    starts = np.where(ps >= cfg.get("SPAN_START_MIN", 0.10))[0]; ends = np.where(pe >= cfg.get("SPAN_END_MIN", 0.10))[0]
    min_conf = cfg.get("SPAN_MIN_CONF", 0.05); cands = []
    for i in starts:
        for j in ends[np.searchsorted(ends, i):]:
            n = j - i + 1
            if n > big: break
            ml = np.where(maxlen >= n, (cs[j + 1] - cs[i]) / n, -np.inf)             # mean log P(type) over the span
            c = int(ml.argmax())
            if not np.isfinite(ml[c]): continue
            conf = math.exp((math.log(ps[i] + 1e-9) + math.log(pe[j] + 1e-9) + ml[c]) / 3.0)
            if conf >= min_conf: cands.append((conf, int(i), int(j), c))
    cands.sort(reverse=True)
    taken = np.zeros(T, bool); out = []
    for conf, i, j, c in cands:                                                       # greedy non-overlapping selection
        if taken[i:j + 1].any(): continue
        taken[i:j + 1] = True
        out.append((int(offs[i, 0]), int(offs[j, 1]), LABELS[c], float(conf)))
    return out


def decode_scored(doc, probs, cfg):
    """Decode a document to scored spans after trimming / (optional) boundary refinement / bracket balancing (no thresholding).

    Args:
        doc (dict): tokenised document.
        probs (np.ndarray): (n_tokens, n_out) probabilities for the whole document.
        cfg (dict): config (``TASK, DECODE, TRIM_TRAILING_PUNCT, BOUNDARY_REFINE, ...``).

    Returns:
        list[tuple]: ``(start, end, label, confidence)`` with end-exclusive offsets.
    """
    raw = _span_candidates(doc, probs, cfg) if cfg.get("TASK", "bio") == "span" else _bio_candidates(doc, probs, cfg)
    text, out = doc["text"], {}
    for s, e, l, c in raw:
        s, e = clean_span(text, s, e, S.TRIM_CHARS.get(l, "") if cfg.get("TRIM_TRAILING_PUNCT", True) else "")
        if cfg.get("BOUNDARY_REFINE"):
            s, e = refine_span(text, s, e, l, cfg); s, e = clean_span(text, s, e)
        s, e = balance_brackets(text, s, e)
        if e > s and c > out.get((int(s), int(e), l), -1.0): out[(int(s), int(e), l)] = c
    return [(s, e, l, c) for (s, e, l), c in out.items()]


def apply_thresh(scored, thr):
    """Apply per-label confidence thresholds.

    Args:
        scored (dict): ``{doc: [(s, e, l, conf)]}``.
        thr (dict or None): ``{label: min_conf}``.

    Returns:
        dict: ``{doc: [(s, e, l)]}``.
    """
    thr = thr or {}
    return {d: [(s, e, l) for s, e, l, c in v if c >= thr.get(l, 0.0)] for d, v in scored.items()}


def decode_doc(doc, probs, cfg):
    """Decode one document to final spans using ``cfg["CONF_THRESH"]``.

    Args:
        doc (dict): tokenised document.
        probs (np.ndarray): (n_tokens, n_out) probabilities.
        cfg (dict): config.

    Returns:
        list[tuple[int, int, str]]: ``(s, e, label)``, end-exclusive.
    """
    thr = cfg.get("CONF_THRESH") or {}
    return [(s, e, l) for s, e, l, c in decode_scored(doc, probs, cfg) if c >= thr.get(l, 0.0)]
