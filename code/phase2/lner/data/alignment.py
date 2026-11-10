"""Tokenisation <-> character alignment, BIO encoding and sliding windows (GPT-2 tokenizer, same as Phase 1)."""

import time
from collections import Counter

import numpy as np

from .. import state as S


def get_tokenizer():
    """
    BPE keeps the tokenizer a controlled variable across architectures.

    Args:
        None

    Returns:
        PreTrainedTokenizer: the tokenizer.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/kaggle/input/sohamjee/legal_bpe/hf"
    )  
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.model_max_length = int(1e12)
    print("Vocabulary size:", len(tok))
    return tok


def normalize_offsets(text, offs):
    """Trim whitespace from BPE offsets (GPT-2 tokens may carry their leading space) so token spans are tight.

    Whitespace-only tokens become empty (s==e): labelled -100 and skipped in decoding.

    Args:
        text (str): document text.
        offs (list[tuple[int, int]]): raw char offsets per token.

    Returns:
        np.ndarray: (n_tokens, 2) int64 offsets.
    """
    out = []
    for s, e in offs:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        out.append((s, e))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def encode_tags(offs, ents):
    """Convert char-level entities (end-exclusive) into per-token BIO ids.

    Args:
        offs (np.ndarray): (n_tokens, 2) token char offsets.
        ents (list[tuple[int, int, str]]): entities ``(start, end, label)``.

    Returns:
        tuple[np.ndarray, Counter]: (tag ids with -100 for empty tokens, alignment stats ``ok/no_token/overlap_conflict``).
    """
    n = len(offs)
    tags = np.zeros(n, dtype=np.int64)
    ts, te = offs[:, 0], offs[:, 1]
    nonempty = te > ts
    taken = np.zeros(n, bool)
    stats = Counter()
    for s, e, l in sorted(ents, key=lambda x: (x[0], -(x[1] - x[0]))):
        lo = np.searchsorted(te, s, side="right")
        hi = np.searchsorted(ts, e, side="left")
        idx = np.arange(lo, hi)
        idx = idx[nonempty[idx]] if len(idx) else idx
        if len(idx) == 0:
            stats["no_token"] += 1
            continue
        if taken[idx].any():
            stats["overlap_conflict"] += 1
            continue
        tags[idx[0]] = S.TAG2ID[f"B-{l}"]
        tags[idx[1:]] = S.TAG2ID[f"I-{l}"]
        taken[idx] = True
        stats["ok"] += 1
    tags[~nonempty] = -100
    return tags, stats


def make_windows(n, max_len, stride):
    """Sliding windows over ``n`` tokens (last window is right-aligned).

    Args:
        n (int): number of tokens.
        max_len (int): window length.
        stride (int): step between window starts.

    Returns:
        list[tuple[int, int]]: ``(start, end)`` token ranges.
    """
    if n <= max_len:
        return [(0, n)]
    starts = list(range(0, n - max_len + 1, stride))
    if starts[-1] + max_len < n:
        starts.append(n - max_len)
    return [(s, s + max_len) for s in starts]


# ---------- punctuation pre-split ----------
# GPT-2 BPE merges some punctuation pairs into one token ('.[' '].' '),' ').'), so an entity edge can fall INSIDE a token.
# We cut the text between two DIFFERENT adjacent punctuation characters and tokenise the pieces separately.
_P = lambda ch: not ch.isalnum() and not ch.isspace()


def presplit_tokenize(text):
    """Tokenise with punctuation pairs kept in separate tokens.

    Args:
        text (str): document text.

    Returns:
        tuple[list[int], list[tuple[int, int]]]: (token ids, document-level char offsets).
    """
    tok = get_tokenizer()
    cut = (
        [0]
        + [
            i
            for i in range(1, len(text))
            if _P(text[i - 1]) and _P(text[i]) and text[i - 1] != text[i]
        ]
        + [len(text)]
    )
    segs = [text[a:b] for a, b in zip(cut, cut[1:])]
    enc = tok(segs, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = [], []
    for base, seg_ids, seg_offs in zip(cut, enc["input_ids"], enc["offset_mapping"]):
        ids += seg_ids
        offs += [
            (base + a, base + b) for a, b in seg_offs
        ]  # shift segment-relative offsets to document offsets
    return ids, offs


def plain_tokenize(text):
    """Plain GPT-2 BPE tokenisation (old behaviour).

    Args:
        text (str): document text.

    Returns:
        tuple[list[int], list[tuple[int, int]]]: (token ids, char offsets).
    """
    enc = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def build_doc(did, text, ents=None, use_presplit=True):
    """Tokenise one document (and align gold tags if given).

    Args:
        did (str): document id.
        text (str): document text.
        ents (list or None): gold ``(start, end, label)`` (end-exclusive) or None for unlabeled/test docs.
        use_presplit (bool): punctuation pre-split (True) or plain BPE.

    Returns:
        dict: ``id, text, ids, offs`` (+ ``tags, stats`` when ``ents`` is given).
    """
    ids, raw_offs = presplit_tokenize(text) if use_presplit else plain_tokenize(text)
    d = dict(
        id=did,
        text=text,
        ids=np.array(ids, dtype=np.int64),
        offs=normalize_offsets(text, raw_offs),
    )
    if ents is not None:
        d["tags"], d["stats"] = encode_tags(d["offs"], ents)
    return d


def tokenize_all(max_len, use_presplit=True, verbose=True):
    """Tokenise every labelled document once and store it in ``state.DOCS``.

    Args:
        max_len (int): window length (only used for the printed statistics).
        use_presplit (bool): punctuation pre-split.
        verbose (bool): print alignment stats and length statistics.

    Returns:
        None
    """
    S.DOCS, t0, agg = {}, time.time(), Counter()
    for i, d in enumerate(S.TEXTS):
        S.DOCS[d] = build_doc(d, S.TEXTS[d], S.GOLD[d], use_presplit)
        agg.update(S.DOCS[d]["stats"])
        if verbose and (i + 1) % 200 == 0:
            print(f"  tokenised {i + 1}/{len(S.TEXTS)} docs ({time.time() - t0:.0f}s)")
    if verbose:
        lens = np.array([len(x["ids"]) for x in S.DOCS.values()])
        print("USE_PRESPLIT =", use_presplit)
        print("gold->token alignment stats:", dict(agg))
        print(
            f"doc length in tokens: median={np.median(lens):.0f} p90={np.percentile(lens, 90):.0f} max={lens.max()} -> windows of {max_len} needed for {np.mean(lens > max_len) * 100:.0f}% of docs"
        )
