"""Training samples (windows), collation, class weights, optional document-head prefix (CTX_PREFIX) and oversampling."""
import random

import numpy as np
import torch

from .. import state as S
from .alignment import make_windows


def with_prefix(doc, a, b, cfg):
    """Window ``[a, b)`` with the first ``CTX_PREFIX`` document tokens prepended (case caption / party block).

    The prefix is only added to windows that do not already contain the document start.

    Args:
        doc (dict): tokenised document.
        a (int): window start token.
        b (int): window end token (exclusive).
        cfg (dict): config (``CTX_PREFIX``).

    Returns:
        tuple[np.ndarray, int]: (token ids of the model input, offset of the window inside that input).
    """
    k = int(cfg.get("CTX_PREFIX", 0) or 0)
    if k and a >= k: return np.concatenate([doc["ids"][:k], np.array([S.PAD_ID]), doc["ids"][a:b]]).astype(np.int64), k + 1
    return doc["ids"][a:b], 0


def build_train_samples(doc_ids, cfg, seed=42):
    """Cut documents into training windows (edge tokens of inner windows ignored; negative windows subsampled).

    Supports ``CTX_PREFIX`` (document-head prefix) and ``OVERSAMPLE`` (duplicate windows containing weak labels).

    Args:
        doc_ids (list[str]): documents to use (ids may include pseudo-labelled docs).
        cfg (dict): config (``OVERLAP, MAX_LEN, NEG_WINDOW_KEEP, CTX_PREFIX, OVERSAMPLE``).
        seed (int): RNG seed for negative-window subsampling.

    Returns:
        list[dict]: samples ``{"ids": np.ndarray, "tags": np.ndarray}``.
    """
    rng, out, n_all = random.Random(seed), [], 0
    k = cfg["OVERLAP"] // 2          # edge tokens of inner windows are not trained on; matches the ramp in predict_probs
    for did in doc_ids:
        d = S.DOCS[did]; n = len(d["ids"])
        wins = make_windows(n, cfg["MAX_LEN"], cfg["MAX_LEN"] - cfg["OVERLAP"])
        for a, b in wins:
            n_all += 1; tg = d["tags"][a:b].copy()
            has_ent = (tg > 0).any()
            if len(wins) > 1:
                if a > 0: tg[:k] = -100
                if b < n: tg[-k:] = -100
            if not has_ent and rng.random() > cfg["NEG_WINDOW_KEEP"]: continue
            ids_w, off = with_prefix(d, a, b, cfg)
            tg_w = np.concatenate([np.full(off, -100, dtype=tg.dtype), tg]) if off else tg
            out.append(dict(ids=ids_w, tags=tg_w))
            ov = cfg.get("OVERSAMPLE")
            if ov and np.isin(tg_w, [S.TAG2ID[f"{p}-{l}"] for l in ov["labels"] for p in "BI"]).any(): out.extend([out[-1]] * (int(ov["x"]) - 1))
    print(f"train windows: kept {len(out)} / {n_all}")
    return out


def collate(batch):
    """Right-pad a list of samples.

    Args:
        batch (list[dict]): samples with ``ids`` and ``tags``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (input ids (B, L), attention mask (B, L), labels (B, L; -100 = ignore)).
    """
    L = max(len(b["ids"]) for b in batch); B = len(batch)
    ids = torch.full((B, L), S.PAD_ID, dtype=torch.long); lab = torch.full((B, L), -100, dtype=torch.long); am = torch.zeros((B, L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"]); ids[i, :n] = torch.from_numpy(b["ids"]); lab[i, :n] = torch.from_numpy(b["tags"]); am[i, :n] = 1
    return ids, am, lab


def _tag_counts(samples):
    """Count tag occurrences over training samples.

    Args:
        samples (list[dict]): training samples.

    Returns:
        np.ndarray: (n_tags,) counts (ignored tokens excluded).
    """
    cnt = np.zeros(len(S.TAGS))
    for s in samples:
        t = s["tags"][s["tags"] >= 0]; cnt += np.bincount(t, minlength=len(S.TAGS))
    return cnt


def class_weights(samples, cfg):
    """Loss class weights. BIO: one per tag (25); span task: one per TYPE (O + 12 labels, B/I merged).

    ``CLASS_WEIGHT_MODE``: ``power`` ((n_O/n_c)^p) | ``effective`` (class-balanced, Cui et al. 2019) | ``off``.
    Explicit per-label multipliers ``LABEL_W`` are applied on top.

    Args:
        samples (list[dict]): training samples.
        cfg (dict): config.

    Returns:
        torch.Tensor: float weights.
    """
    cnt = _tag_counts(samples)
    if cfg.get("TASK", "bio") == "span":
        cnt = np.array([cnt[0]] + [cnt[2 * c - 1] + cnt[2 * c] for c in range(1, len(S.LABELS) + 1)])
    mode = cfg.get("CLASS_WEIGHT_MODE", "power")
    if mode == "off" or (mode == "power" and cfg["CLASS_WEIGHT_POWER"] <= 0): return torch.ones(len(cnt))
    if mode == "effective":
        b = cfg.get("EFF_BETA", 0.9999); w = (1 - b) / (1 - np.power(b, np.maximum(cnt, 1))); w = w / w[0]
    else:
        w = (cnt[0] / np.maximum(cnt, 1)) ** cfg["CLASS_WEIGHT_POWER"]
    w[0] = 1.0
    w = np.clip(w, 1.0, cfg["CLASS_WEIGHT_MAX"])
    for l, f in (cfg.get("LABEL_W") or {}).items():
        if l not in S.LABELS: continue
        if cfg.get("TASK", "bio") == "span": w[S.LABELS.index(l) + 1] *= f
        else: w[S.TAG2ID[f"B-{l}"]] *= f; w[S.TAG2ID[f"I-{l}"]] *= f
    return torch.tensor(w, dtype=torch.float)


def span_pos_weight(samples, cfg):
    """Positive-class weight for the start/end sigmoid heads (starts and ends are rare vs. all tokens).

    Args:
        samples (list[dict]): training samples.
        cfg (dict): config.

    Returns:
        float: positive weight (1.0 if class weighting is off).
    """
    cnt = _tag_counts(samples); n_pos = max(cnt[1::2].sum(), 1.0); n_neg = max(cnt.sum() - n_pos, 1.0)
    if cfg.get("CLASS_WEIGHT_MODE", "power") == "off" or cfg["CLASS_WEIGHT_POWER"] <= 0: return 1.0
    return float(np.clip((n_neg / n_pos) ** cfg["CLASS_WEIGHT_POWER"], 1.0, cfg["CLASS_WEIGHT_MAX"]))
