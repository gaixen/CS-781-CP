"""Tokenisers (one per Phase-1 architecture) and one-time, disk-cached pre-tokenisation."""
import os
import time
import pickle
import hashlib

import numpy as np
from transformers import AutoTokenizer

from ..config import (TOKENIZER_PER_ARCH, CACHE_DIR, HEAD_KEEP, TAIL_KEEP, CHAR_HEAD, CHAR_TAIL, SEED)
from .loader import get_raw_docs

_TOK = {}
_EX = {}


def get_tokenizer(arch):
    """Load (and memoise) the tokenizer stored next to a Phase-1 checkpoint.

    Args:
        arch (str): architecture key of ``TOKENIZER_PER_ARCH``.

    Returns:
        PreTrainedTokenizer: tokenizer with ``model_max_length`` effectively unbounded.

    Raises:
        ValueError: if no path is configured for ``arch``.
    """
    path = TOKENIZER_PER_ARCH[arch]
    if path is None: raise ValueError(f"No tokenizer/checkpoint path for '{arch}'")
    if path not in _TOK:
        t = AutoTokenizer.from_pretrained(path, local_files_only=True); t.model_max_length = int(1e12); _TOK[path] = t
    return _TOK[path]


def tokenize_split(arch, split):
    """Pre-tokenise a split ONCE (cached on disk), keeping the first ``HEAD_KEEP`` and last ``TAIL_KEEP`` tokens per case.

    Very long cases are only tokenised on their head/tail character slices (big speed-up).

    Args:
        arch (str): architecture key (selects the tokenizer).
        split (str): ``"train"`` | ``"dev"`` | ``"test"``.

    Returns:
        list[dict]: one record per document with ``id, label, n_chars, whole, head, tail`` (head/tail are int32 arrays).
    """
    tok = get_tokenizer(arch); path = TOKENIZER_PER_ARCH[arch]
    f = f"{CACHE_DIR}/tok_{hashlib.md5(path.encode()).hexdigest()[:8]}_{split}_{HEAD_KEEP}_{TAIL_KEEP}.pkl"
    if os.path.exists(f):
        with open(f, "rb") as fh: return pickle.load(fh)
    docs, out, t0 = get_raw_docs()[split], [], time.time()
    for i in range(0, len(docs), 256):
        batch, pieces, plan = docs[i:i + 256], [], []
        for d in batch:
            t = d["text"]
            if len(t) <= CHAR_HEAD + CHAR_TAIL: plan.append(("full", len(pieces))); pieces.append(t)
            else: plan.append(("split", len(pieces))); pieces += [t[:CHAR_HEAD], t[-CHAR_TAIL:]]
        enc = tok(pieces, add_special_tokens=False)["input_ids"]
        for d, (kind, j) in zip(batch, plan):
            if kind == "full":
                ids = enc[j]; head, tail, whole = ids[:HEAD_KEEP], ids[-TAIL_KEEP:], len(ids) <= TAIL_KEEP
            else:
                head, tail, whole = enc[j][:HEAD_KEEP], enc[j + 1][-TAIL_KEEP:], False
            out.append(dict(id=d["id"], label=d["label"], n_chars=len(d["text"]), whole=whole,
                            head=np.asarray(head, dtype=np.int32), tail=np.asarray(tail, dtype=np.int32)))
    with open(f, "wb") as fh: pickle.dump(out, fh)
    print(f"tokenised {split} ({arch}): {len(out)} docs in {time.time()-t0:.0f}s"); return out


def subset_indices(n, k):
    """Deterministic random subset (for smoke tests).

    Args:
        n (int): population size.
        k (int or None): subset size (None/0/>=n means use everything).

    Returns:
        np.ndarray: sorted indices.
    """
    if not k or k >= n: return np.arange(n)
    return np.sort(np.random.RandomState(SEED).permutation(n)[:k])


def get_examples(arch, split, cfg):
    """Return tokenised examples of a split, honouring ``TRAIN_SUBSET`` / ``DEV_SUBSET``.

    Args:
        arch (str): architecture key.
        split (str): ``"train"`` | ``"dev"`` | ``"test"``.
        cfg (dict): experiment config.

    Returns:
        list[dict]: examples with an extra ``pos`` field (position in this list; aligns teacher outputs for KD).
    """
    key = (TOKENIZER_PER_ARCH[arch], split)
    if key not in _EX: _EX[key] = tokenize_split(arch, split)
    allx = _EX[key]
    idx = subset_indices(len(allx), {"train": cfg["TRAIN_SUBSET"], "dev": cfg["DEV_SUBSET"]}.get(split))
    return [dict(allx[i], pos=j) for j, i in enumerate(idx)]
