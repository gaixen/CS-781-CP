"""Long-document input construction (``headtail`` / ``select``), collation and batching."""
import math

import numpy as np
import torch


def build_input(ctx, ids, strat, budget=None):
    """Fit a tokenised document into the token budget.

    ``headtail`` keeps the first 70 % and last 30 % of the budget; ``select`` scores 64-token blocks by IDF-weighted overlap with the document centroid
    and keeps the head / tail blocks plus the highest-scoring ones.

    Args:
        ctx (Context): Context (``s.doc_tok``, ``vsz``, ``idf``).
        ids (np.ndarray): Document token ids.
        strat (str): ``headtail`` or ``select``.
        budget (Optional[int]): Token budget (default ``s.doc_tok``).

    Returns:
        np.ndarray: Token ids with at most ``budget`` tokens.
    """
    budget = budget or ctx.s.doc_tok
    n = len(ids)
    if n <= budget:
        return ids
    if strat == "headtail":
        h = int(budget * .7)
        return np.concatenate([ids[:h], ids[-(budget - h):]])
    if strat == "select":
        BL = 64
        nb = math.ceil(n / BL)
        blocks = [ids[i * BL:(i + 1) * BL] for i in range(nb)]
        w = np.bincount(ids, minlength=ctx.vsz).astype(np.float32) * ctx.idf
        sc = np.array([w[b].sum() / len(b) for b in blocks])
        forced = {0, 1, nb - 1, nb - 2}
        k = budget // BL
        rest = [i for i in np.argsort(-sc) if i not in forced][:max(0, k - len(forced))]
        return np.concatenate([blocks[i] for i in sorted(forced | set(rest))])
    raise ValueError(strat)


def inputs(ctx, split, strat, budget=None):
    """Cached model inputs of a split.

    Args:
        ctx (Context): Context (``tr``, ``dev``, ``te``).
        split (str): ``train`` | ``dev`` | ``test``.
        strat (str): Long-document strategy.
        budget (Optional[int]): Token budget.

    Returns:
        List[np.ndarray]: Inputs per document.
    """
    budget = budget or ctx.s.doc_tok
    k = (split, strat, budget)
    if k not in ctx.inp_cache:
        S = {"train": ctx.tr, "dev": ctx.dev, "test": ctx.te}[split]
        ctx.inp_cache[k] = [build_input(ctx, x, strat, budget) for x in S["doc_ids"]]
    return ctx.inp_cache[k]


def collate(ctx, batch):
    """Collate ``(document ids, summary ids)`` pairs: documents LEFT-padded, summaries right-padded, loss on the summary + EOS only.

    Args:
        ctx (Context): Context (``eos``, ``s.sum_tok``, ``s.doc_tok``).
        batch (List[Tuple[np.ndarray, np.ndarray]]): Pairs.

    Returns:
        Tuple[torch.Tensor, ...]: ``(doc, doc_mask, summary_in, summary_mask, labels)``.
    """
    s, EOS = ctx.s, ctx.eos
    B = len(batch)
    Td = max(len(d) for d, _ in batch)
    Ls = min(s.sum_tok, max(len(x) for _, x in batch) + 1)
    doc = torch.full((B, Td), EOS, dtype=torch.long)
    dm = torch.zeros(B, Td, dtype=torch.long)
    sin_ = torch.full((B, Ls), EOS, dtype=torch.long)
    sm = torch.zeros(B, Ls, dtype=torch.long)
    lab = torch.full((B, Ls), -100, dtype=torch.long)
    for b, (d, x) in enumerate(batch):
        x = np.asarray(x[:s.sum_tok - 1], np.int64)
        doc[b, Td - len(d):] = torch.as_tensor(np.asarray(d, np.int64))
        dm[b, Td - len(d):] = 1
        sin_[b, 1:len(x) + 1] = torch.as_tensor(x)
        sm[b, :len(x) + 1] = 1
        lab[b, :len(x)] = torch.as_tensor(x)
        lab[b, len(x)] = EOS
    assert Td <= s.doc_tok, f"Document length {Td} exceeds {s.doc_tok}"
    assert Ls <= s.sum_tok, f"Summary length {Ls} exceeds {s.sum_tok}"
    assert Td + Ls <= 2048, f"Total length {Td + Ls} exceeds RoPE context"
    return doc, dm, sin_, sm, lab


def make_batches(items, bs, rng):
    """Length-bucketed random batches.

    Args:
        items (list): Training items.
        bs (int): Batch size.
        rng (np.random.RandomState): RNG.

    Returns:
        List[list]: Batches in random order.
    """
    idx = rng.permutation(len(items))
    out = []
    for i in range(0, len(idx), bs * 50):
        c = sorted(idx[i:i + bs * 50], key=lambda j: len(items[j][0]))
        out += [[items[j] for j in c[k:k + bs]] for k in range(0, len(c), bs)]
    return [out[i] for i in rng.permutation(len(out))]
