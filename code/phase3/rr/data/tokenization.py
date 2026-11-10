"""GPT-2 tokenisation of sentences, context inputs (target sentence + neighbours) and window utilities."""

import random

import numpy as np
import torch


def setup_tokenizer(ctx):
    """Load the GPT-2 tokenizer (same as Phase 1) and tokenise every sentence of every document once.

    Fills ``ctx.tokenizer``, ``ctx.pad_id`` (EOS, also the separator between context sentences) and ``ctx.dom_ids``; adds ``raw`` / ``y`` to documents.

    Args:
        ctx (Context): Pipeline context (needs ``ctx.data``).

    Returns:
        None
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/kaggle/input/sohamjee/legal_bpe/hf")
    tok.model_max_length = int(1e12)
    ctx.tokenizer, ctx.pad_id = tok, tok.eos_token_id
    ctx.dom_ids = {
        d: tok(f"[{d}] ", add_special_tokens=False)["input_ids"] for d in ("CL", "IT")
    }
    for ds in (ctx.data.train, ctx.data.dev, ctx.data.test):
        tokenize_docs(ctx, ds)
    tr = [len(r) for d in ctx.data.train for r in d["raw"]]
    print(
        f"tokenised; median tokens/sentence = {int(np.median(tr))} | p95 = {int(np.percentile(tr, 95))} (MAX_SENT_TOK={ctx.cfg['MAX_SENT_TOK']})"
    )


def tokenize_docs(ctx, docs):
    """Tokenise the sentences of documents (adds ``raw`` token arrays and gold index array ``y``).

    NOTE: if a Phase-1 checkpoint used a tokenizer other than GPT-2's, token ids would be misaligned with its embedding table.

    Args:
        ctx (Context): Context with the tokenizer.
        docs (List[dict]): Documents (modified in place).

    Returns:
        None
    """
    for d in docs:
        enc = ctx.tokenizer(d["sents"], add_special_tokens=False)["input_ids"]
        d["raw"] = [np.array(x if x else [ctx.pad_id], dtype=np.int64) for x in enc]
        d["y"] = (
            np.array([ctx.data.l2i[l] for l in d["labels"]], dtype=np.int64)
            if d["labels"] is not None
            else None
        )


def make_inputs_list(ctx, items, cfg):
    """Padded inputs for ``[(document, sentence_index)]`` with optional +-k neighbour context.

    Args:
        ctx (Context): Context (pad id, domain prefix ids).
        items (List[Tuple[dict, int]]): ``(doc, i)`` pairs.
        cfg (dict): Experiment config (``NEIGHBORS, NB_TOK, MAX_SENT_TOK, DOMAIN_TOKEN``).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(ids, attention_mask, target_mask)``; pooling uses the target tokens only.
    """
    k, nb, mx, pad = cfg["NEIGHBORS"], cfg["NB_TOK"], cfg["MAX_SENT_TOK"], ctx.pad_id
    seqs, tms = [], []
    for doc, i in items:
        pre = ctx.dom_ids[doc["domain"]] if cfg["DOMAIN_TOKEN"] else []
        tgt = pre + doc["raw"][i][:mx].tolist()
        left, right = [], []
        for j in range(max(0, i - k), i):
            left += doc["raw"][j][-nb:].tolist() + [pad]
        for j in range(i + 1, min(len(doc["raw"]), i + 1 + k)):
            right += [pad] + doc["raw"][j][:nb].tolist()
        seqs.append(left + tgt + right)
        tms.append([0] * len(left) + [1] * len(tgt) + [0] * len(right))
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    tm = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, (s, t) in enumerate(zip(seqs, tms)):
        ids[r, : len(s)] = torch.tensor(s)
        am[r, : len(s)] = 1
        tm[r, : len(s)] = torch.tensor(t)
    return ids, am, tm


def make_windows(n, W, S):
    """Overlapping windows ``(start, end)`` covering ``n`` sentences.

    Args:
        n (int): Number of sentences.
        W (int): Window length.
        S (int): Stride.

    Returns:
        List[Tuple[int, int]]: Windows (the last one is right-aligned).
    """
    if n <= W:
        return [(0, n)]
    st = list(range(0, n - W + 1, S))
    if st[-1] + W < n:
        st.append(n - W)
    return [(a, a + W) for a in st]


def train_windows(n, W, S):
    """Training windows with a random phase each epoch (more diverse BiLSTM contexts).

    Args:
        n (int): Number of sentences.
        W (int): Window length.
        S (int): Stride.

    Returns:
        List[Tuple[int, int]]: Windows.
    """
    if n <= W:
        return [(0, n)]
    st = sorted(set([0, n - W] + list(range(random.randrange(S), n - W + 1, S))))
    return [(a, a + W) for a in st]
