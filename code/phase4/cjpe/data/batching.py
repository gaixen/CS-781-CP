"""Document representations (which tokens the model sees), padding, collation and length-bucketed batching."""
import numpy as np
import torch

from ..context import device


def select_ids(ex, cfg):
    """Select tokens of one document for the single-sequence representations.

    Args:
        ex (dict): tokenised example (``head, tail, whole``).
        cfg (dict): config (``REPRESENTATION`` in last_n|first_n|head_tail, ``MAX_TOKENS``, ``HEAD_TOKENS``).

    Returns:
        np.ndarray: token ids to feed the model.

    Raises:
        ValueError: for an unknown representation.
    """
    rep, M = cfg["REPRESENTATION"], cfg["MAX_TOKENS"]
    if rep == "last_n": return ex["tail"][-M:]
    if rep == "first_n": return ex["head"][:M]
    if rep == "head_tail":
        if ex["whole"] and len(ex["tail"]) <= M: return ex["tail"]
        h = cfg["HEAD_TOKENS"]; return np.concatenate([ex["head"][:h], ex["tail"][-(M - h):]])
    raise ValueError(rep)


def make_chunks(tail, L, stride, max_chunks):
    """Split a token sequence into overlapping chunks counted back from the end.

    Args:
        tail (np.ndarray): token ids (the cached tail of the case).
        L (int): chunk length.
        stride (int): overlap between consecutive chunks.
        max_chunks (int): maximum number of chunks.

    Returns:
        list[np.ndarray]: chunks; ``chunks[0]`` is the LAST chunk (k = distance from the end of the case).
    """
    step, chunks, end = L - stride, [], len(tail)
    for _ in range(max_chunks):
        if end <= 0: break
        start = max(0, end - L); chunks.append(tail[start:end])
        if start == 0: break
        end -= step
    return chunks


def _pad(seqs, pad_id):
    """Right-pad variable-length id sequences.

    Args:
        seqs (list[np.ndarray]): token id sequences (empty ones become a single pad token).
        pad_id (int): padding token id.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (ids (N, T) long, mask (N, T) bool).
    """
    seqs = [s if len(s) else np.array([pad_id]) for s in seqs]; T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long); mask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.from_numpy(s.astype(np.int64)); mask[i, :len(s)] = True
    return ids, mask


def collate(exs, cfg):
    """Build a model-ready batch from examples.

    Args:
        exs (list[dict]): tokenised examples.
        cfg (dict): config (must contain ``_pad_id``; for ``hier`` also ``CHUNK_TOKENS, CHUNK_STRIDE, MAX_CHUNKS``).

    Returns:
        dict: ``labels, pos, ids, mask`` and, for ``hier``, ``doc_index, chunk_pos, B, C``.
    """
    pad_id = cfg["_pad_id"]
    b = dict(labels=torch.tensor([e["label"] for e in exs], dtype=torch.long), pos=np.array([e["pos"] for e in exs]))
    if cfg["REPRESENTATION"] != "hier":
        b["ids"], b["mask"] = _pad([select_ids(e, cfg) for e in exs], pad_id)
    else:
        seqs, di, cp = [], [], []
        for i, e in enumerate(exs):
            for k, c in enumerate(make_chunks(e["tail"], cfg["CHUNK_TOKENS"], cfg["CHUNK_STRIDE"], cfg["MAX_CHUNKS"])):
                seqs.append(c); di.append(i); cp.append(k)
        b["ids"], b["mask"] = _pad(seqs, pad_id)
        b["doc_index"], b["chunk_pos"] = torch.tensor(di), torch.tensor(cp); b["B"], b["C"] = len(exs), max(cp) + 1
    return b


def to_device(b):
    """Move all tensors of a batch to the active device.

    Args:
        b (dict): batch from ``collate``.

    Returns:
        dict: same keys, tensors on ``device``.
    """
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def make_batches(exs, bs, shuffle, rng):
    """Create index batches. Training uses length-bucketing + shuffled batches; evaluation sorts by length.

    Args:
        exs (list[dict]): examples (uses ``len(e["tail"])``).
        bs (int): batch size.
        shuffle (bool): training mode if True.
        rng (np.random.RandomState or None): RNG (required when ``shuffle``).

    Returns:
        list[np.ndarray]: list of index arrays into ``exs``.
    """
    order = np.arange(len(exs)); lens = np.array([len(e["tail"]) for e in exs])
    if shuffle:
        rng.shuffle(order); mega = bs * 50
        order = np.concatenate([o[np.argsort(lens[o])] for o in np.array_split(order, max(1, len(order) // mega))])   # length-bucketing
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]; rng.shuffle(batches); return batches
    order = np.argsort(lens); return [order[i:i + bs] for i in range(0, len(order), bs)]
