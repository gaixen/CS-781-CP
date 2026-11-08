"""Long-document handling: tokenisation (cached), overlapping chunks, context selection and batching."""
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp
import torch
from transformers import AutoTokenizer

from lsi.config import ARCH_SHORT
from lsi.utils import md5_text


def get_tokenizer(ctx, arch):
    """Load (and cache) the tokenizer of an architecture's Phase-1 checkpoint.

    Args:
        ctx (Context): Pipeline context (``cfg.phase1_ckpt``, tokenizer cache).
        arch (str): Architecture key.

    Returns:
        PreTrainedTokenizer: Tokenizer without length limit.
    """
    p = ctx.cfg.phase1_ckpt[arch]
    if p not in ctx.tokenizers:
        t = AutoTokenizer.from_pretrained(p, local_files_only=True)
        t.model_max_length = int(1e12)
        ctx.tokenizers[p] = t
    return ctx.tokenizers[p]


def tokenize_split(ctx, arch, split, docs):
    """Tokenise a document list (head 75 % + tail 25 % when longer than ``max_doc_tokens``), cached on disk.

    Args:
        ctx (Context): Pipeline context.
        arch (str): Architecture key.
        split (str): Split name (cache key).
        docs (List[dict]): Documents.

    Returns:
        Tuple[List[np.ndarray], int]: ``(token id arrays, pad id)``.
    """
    cfg = ctx.cfg
    tok = get_tokenizer(ctx, arch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    key = md5_text(f"{cfg.phase1_ckpt[arch]}|{split}|{cfg.max_doc_tokens}|{len(docs)}|{docs[0]['id'] if docs else ''}")[:10]
    f = f"{cfg.cache_dir}/tok_{key}.pkl"
    if os.path.exists(f):
        return pickle.load(open(f, "rb")), pad
    cchar, out, t0 = cfg.max_doc_tokens * 8, [], time.time()
    for i in range(0, len(docs), 128):
        txt = []
        for d in docs[i:i + 128]:
            t = d["text"]
            txt.append(t if len(t) <= cchar else t[:int(cchar * .75)] + " " + t[-int(cchar * .25):])
        for ids in tok(txt, add_special_tokens=False)["input_ids"]:
            if len(ids) > cfg.max_doc_tokens:
                h = int(cfg.max_doc_tokens * .75)
                ids = ids[:h] + ids[-(cfg.max_doc_tokens - h):]
            out.append(np.asarray(ids if len(ids) else [pad], np.int32))
    pickle.dump(out, open(f, "wb"))
    print(f"tokenised {split}: {len(out)} docs in {time.time()-t0:.0f}s")
    return out, pad


def chunk_spans(n, chunk_tokens, overlap):
    """Overlapping ``(start, end)`` chunk spans of a document of ``n`` tokens.

    Args:
        n (int): Number of tokens.
        chunk_tokens (int): Chunk length.
        overlap (int): Overlap between consecutive chunks.

    Returns:
        List[Tuple[int, int]]: Chunk spans.
    """
    if n <= chunk_tokens:
        return [(0, max(n, 1))]
    step, out, s = chunk_tokens - overlap, [], 0
    while True:
        e = min(s + chunk_tokens, n)
        out.append((s, e))
        if e == n:
            break
        s += step
    return out


def select_chunks(n, k, mode, rng=None, score=None):
    """Indices (document order) of the chunks that represent a document.

    Args:
        n (int): Number of chunks of the document.
        k (int): Maximum number of chunks kept.
        mode (str): ``full | head_tail | evidence | first | last``.
        rng (Optional[np.random.RandomState]): Given during training (random / jittered selection); None = deterministic.
        score (Optional[np.ndarray]): Per-chunk informativeness (``evidence`` mode).

    Returns:
        List[int]: Sorted chunk indices.
    """
    if mode == "first":
        return list(range(min(n, k)))
    if mode == "last":
        return list(range(max(0, n - k), n))
    if n <= k:
        return list(range(n))
    if mode == "head_tail":
        h = (k + 1) // 2
        return list(range(h)) + list(range(n - (k - h), n))
    if mode == "evidence" and score is not None:
        s = np.asarray(score, float).copy()
        if rng is not None:
            s = s + rng.gumbel(size=n) * 0.25 * (s.std() + 1e-9)
        return sorted(np.argsort(-s)[:k].tolist())
    if rng is not None:
        return sorted(rng.choice(n, k, replace=False).tolist())
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def build_lexicon(toks, Y, V, min_df=5, a=5.0, block=256):
    """Token-id -> label-informativeness in [0, 1] (max over labels of the smoothed lift). Train tokens only.

    Args:
        toks (List[np.ndarray]): Train token arrays.
        Y (np.ndarray): Train label matrix.
        V (int): Vocabulary size.
        min_df (int): Minimum document frequency.
        a (float): Smoothing strength.
        block (int): Label block size (memory).

    Returns:
        np.ndarray: float32 weights of shape (V,).
    """
    N = len(toks)
    rows = np.concatenate([np.full(len(np.unique(t)), i) for i, t in enumerate(toks)])
    cols = np.concatenate([np.unique(t) for t in toks])
    X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, V))
    df = np.asarray(X.sum(0)).ravel()
    keep = np.where(df >= min_df)[0]
    w = np.zeros(V, np.float32)
    if len(keep) == 0:
        return w
    Xk, dfk, prior = X[:, keep].tocsc(), df[keep][:, None], Y.mean(0)[None, :]
    best = np.zeros(len(keep), np.float32)
    for b in range(0, Y.shape[1], block):
        C = np.asarray((Xk.T @ sp.csr_matrix(Y[:, b:b + block])).todense(), np.float32)
        best = np.maximum(best, ((C + a * prior[:, b:b + block]) / (dfk + a) - prior[:, b:b + block]).max(1))
    w[keep] = np.clip(best, 0, None)
    return w


class Store:
    """Tokenised documents with their chunk spans (and optional chunk informativeness scores).

    Args:
        toks (List[np.ndarray]): Token arrays.
        chunk_tokens (int): Chunk length.
        overlap (int): Chunk overlap.
        lex (Optional[np.ndarray]): Token lexicon from :func:`build_lexicon`.
    """

    def __init__(self, toks, chunk_tokens, overlap, lex=None):
        self.tok = toks
        self.spans = [chunk_spans(len(t), chunk_tokens, overlap) for t in toks]
        self.score = None if lex is None else [np.array([lex[t[a:b]].mean() for a, b in sp_]) for t, sp_ in zip(toks, self.spans)]

    def __len__(self):
        """Number of documents.

        Args:
            None

        Returns:
            int: Document count.
        """
        return len(self.tok)


def pad_batch(seqs, pad_id):
    """Right-pad token arrays into id / mask tensors.

    Args:
        seqs (List[np.ndarray]): Token arrays.
        pad_id (int): Padding id.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(ids, mask)``.
    """
    T = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.from_numpy(x.astype(np.int64))
        mask[i, :len(x)] = True
    return ids, mask


def build_batch(store, idx, k, mode, rng, pad_id):
    """Training batch: selected chunks of the documents ``idx``.

    Args:
        store (Store): Token store.
        idx (Sequence[int]): Document indices.
        k (int): Chunks per document.
        mode (str): Chunk selection mode.
        rng (np.random.RandomState): RNG for random selection.
        pad_id (int): Padding id.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, np.ndarray]: ``(ids, mask, owner)`` where ``owner[c]`` is the batch position of chunk ``c``.
    """
    seqs, owner = [], []
    for b, i in enumerate(idx):
        for j in select_chunks(len(store.spans[i]), k, mode, rng, None if store.score is None else store.score[i]):
            a, e = store.spans[i][j]
            seqs.append(store.tok[i][a:e])
            owner.append(b)
    ids, mask = pad_batch(seqs, pad_id)
    return ids, mask, np.array(owner)


def get_stores(ctx, arch):
    """Token stores (train / dev / test + pad id) of an architecture, cached in the context.

    Args:
        ctx (Context): Pipeline context (needs ``ctx.data``).
        arch (str): Architecture key.

    Returns:
        dict: ``{"train": Store, "dev": Store, "test": Store, "pad": int}``.
    """
    if arch not in ctx.stores:
        cfg, d = ctx.cfg, ctx.data
        tr, pad = tokenize_split(ctx, arch, "train", d.train_docs)
        dv, _ = tokenize_split(ctx, arch, "dev", d.dev_docs)
        te, _ = tokenize_split(ctx, arch, "test", d.test_docs)
        lex = build_lexicon(tr, d.ytr, len(get_tokenizer(ctx, arch)))
        ctx.stores[arch] = dict(train=Store(tr, cfg.chunk_tokens, cfg.overlap, lex), dev=Store(dv, cfg.chunk_tokens, cfg.overlap, lex),
                                test=Store(te, cfg.chunk_tokens, cfg.overlap, lex), pad=pad)
        n = np.array([len(x) for x in ctx.stores[arch]["train"].spans])
        print(f"[{ARCH_SHORT[arch]}] chunks/doc train: median={np.median(n):.0f} p90={np.percentile(n,90):.0f} max={n.max()} | docs above inf_chunks={cfg.inf_chunks}: {(n>cfg.inf_chunks).mean():.1%}")
    return ctx.stores[arch]
