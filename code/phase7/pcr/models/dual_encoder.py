"""Phase-1 backbone as a chunked document encoder (bi-encoder) plus scoring utilities.

Each case is split into ``ch_tok``-token chunks; every chunk is pooled (mean / last / max over tokens),
chunk vectors are aggregated (mean / max) and L2-normalised. Queries are scored against their own split's
candidate pool by cosine similarity (and, optionally, by MaxSim over chunks).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from pcr.models.custom_decoder import build_attn_mask
from pcr.models.phase1_backbone import build_backbone, inject_lorma, inject_adapters, prune_layers, set_trainable_top_k


def _cb(blk, x, cos, sin, mask):
    """Call a decoder block (wrapper so it can be passed to ``torch.utils.checkpoint``).

    Args:
        blk (nn.Module): Decoder block.
        x (torch.Tensor): Hidden states.
        cos (torch.Tensor): RoPE cosine table.
        sin (torch.Tensor): RoPE sine table.
        mask (torch.Tensor): Attention mask.

    Returns:
        torch.Tensor: Block output.
    """
    return blk(x, cos, sin, mask)


class DualEnc(nn.Module):
    """Phase-1 custom decoder -> chunk embedding (token pooling); documents = aggregation of chunk embeddings.

    Args:
        bb (Backbone): Phase-1 backbone (``custom`` kind).
        cfg (Config): Pipeline configuration (chunk size, AMP, checkpointing, ...).
        eos (int): Padding / EOS token id.
        device (torch.device): Compute device.
        pool (str): Token pooling inside a chunk (``mean`` | ``last`` | ``max``).
    """

    def __init__(self, bb, cfg, eos, device, pool="mean"):
        super().__init__()
        self.bb = bb
        self.m = bb.model
        self.pool = pool
        self.dim = bb.hidden_dim
        self.cfg, self.eos, self.device = cfg, eos, device

    def autocast(self):
        """Return the autocast context manager (fp16 on CUDA when AMP is enabled).

        Args:
            None

        Returns:
            torch.autocast: Context manager.
        """
        return torch.autocast(device_type=self.device.type, dtype=torch.float16,
                              enabled=bool(self.cfg.amp and self.device.type == "cuda"))

    def chunk_embed(self, ids, mask):
        """Embed a batch of chunks into L2-normalised vectors.

        Args:
            ids (torch.Tensor): Token ids (N, T).
            mask (torch.Tensor): Padding mask (N, T), 1 for real tokens.

        Returns:
            torch.Tensor: Normalised chunk embeddings (N, dim) in fp32.
        """
        m = self.m
        T = ids.size(1)
        with self.autocast():
            x = m.tok_emb(ids)
            mk = build_attn_mask(T, x.device, m.config.window_size, bool(getattr(m.config, "bidirectional", False)), mask)
            cos, sin = m.rope_cos[:T], m.rope_sin[:T]
            for blk in m.blocks:
                if self.cfg.grad_ckpt and self.training and torch.is_grad_enabled():
                    x = torch_ckpt.checkpoint(_cb, blk, x, cos, sin, mk, use_reentrant=False)
                else:
                    x = _cb(blk, x, cos, sin, mk)
            h = m.norm_f(x)
        h = h.float()
        mf = mask.unsqueeze(-1).float()
        if self.pool == "mean":
            e = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
        elif self.pool == "last":
            e = h[torch.arange(h.size(0), device=h.device), mask.sum(1) - 1]
        else:
            e = h.masked_fill(mf == 0, -1e4).max(1).values
        return F.normalize(e, dim=-1)


def make_chunks(ids, strat, nch, ch_tok, rng=None):
    """Split a token sequence into at most ``nch`` chunks of ``ch_tok`` tokens.

    Args:
        ids (np.ndarray): Token ids of one document.
        strat (str): ``head`` (first chunks), ``spread`` (evenly spaced) or anything else = random.
        nch (int): Maximum number of chunks.
        ch_tok (int): Tokens per chunk.
        rng (Optional[np.random.RandomState]): RNG for random selection.

    Returns:
        List[np.ndarray]: Selected chunks.
    """
    starts = list(range(0, len(ids), ch_tok))
    if len(starts) > nch:
        if strat == "head":
            starts = starts[:nch]
        elif strat == "spread":
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, nch).round().astype(int)]
        else:
            starts = sorted((rng or np.random).choice(starts, nch, replace=False).tolist())
    return [ids[s:s + ch_tok] for s in starts]


def aggregate(e, owner, N, agg):
    """Aggregate chunk embeddings into document embeddings.

    Args:
        e (torch.Tensor): Chunk embeddings (n_chunks, dim).
        owner (torch.Tensor): Document index of each chunk (n_chunks,).
        N (int): Number of documents.
        agg (str): ``mean`` or ``max``.

    Returns:
        torch.Tensor: L2-normalised document embeddings (N, dim).
    """
    if agg == "max":
        return F.normalize(torch.zeros(N, e.size(1), device=e.device).scatter_reduce(0, owner[:, None].expand_as(e), e, "amax", include_self=False), dim=-1)
    out = torch.zeros(N, e.size(1), device=e.device).index_add(0, owner, e)
    return F.normalize(out / torch.bincount(owner, minlength=N).clamp(min=1)[:, None].float(), dim=-1)


def pad(chs, eos, device):
    """Right-pad a list of chunks into id / mask tensors.

    Args:
        chs (List[np.ndarray]): Chunks of token ids.
        eos (int): Padding token id.
        device (torch.device): Target device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(ids, mask)`` of shape (len(chs), max_len).
    """
    T = max(len(c) for c in chs)
    ids = torch.full((len(chs), T), eos, dtype=torch.long)
    mk = torch.zeros(len(chs), T, dtype=torch.long)
    for i, c in enumerate(chs):
        ids[i, :len(c)] = torch.as_tensor(np.asarray(c, np.int64))
        mk[i, :len(c)] = 1
    return ids.to(device), mk.to(device)


def encode_batch(model, docs, strat, nch, agg, rng=None):
    """Differentiable encoding of a batch of documents (used in training).

    Args:
        model (DualEnc): Encoder.
        docs (List[np.ndarray]): Token-id arrays.
        strat (str): Chunk selection strategy.
        nch (int): Maximum chunks per document.
        agg (str): Chunk aggregation (``mean`` | ``max``).
        rng (Optional[np.random.RandomState]): RNG for random chunk selection.

    Returns:
        torch.Tensor: Document embeddings (len(docs), dim).
    """
    chs, owner = [], []
    for i, d in enumerate(docs):
        c = make_chunks(d, strat, nch, model.cfg.ch_tok, rng)
        chs += c
        owner += [i] * len(c)
    ids, mk = pad(chs, model.eos, model.device)
    return aggregate(model.chunk_embed(ids, mk), torch.as_tensor(owner, device=model.device), len(docs), agg)


@torch.no_grad()
def chunk_embs(model, docs, strat=None, nch=None):
    """Embed all chunks of a list of documents without gradients.

    Args:
        model (DualEnc): Encoder.
        docs (List[np.ndarray]): Token-id arrays.
        strat (Optional[str]): Chunk strategy (default ``cfg.eval_strat``).
        nch (Optional[int]): Max chunks per document (default ``cfg.ev_ch``).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(chunk_embeddings, owner_index)``.
    """
    cfg = model.cfg
    strat, nch = strat or cfg.eval_strat, nch or cfg.ev_ch
    model.eval()
    chs, owner = [], []
    for i, d in enumerate(docs):
        c = make_chunks(d, strat, nch, cfg.ch_tok)
        chs += c
        owner += [i] * len(c)
    order = np.argsort([len(c) for c in chs])
    E = torch.zeros(len(chs), model.dim, device=model.device)
    for s in range(0, len(order), cfg.eval_bs):
        ix = order[s:s + cfg.eval_bs]
        ids, mk = pad([chs[i] for i in ix], model.eos, model.device)
        E[torch.as_tensor(ix, device=model.device)] = model.chunk_embed(ids, mk)
    return E, torch.as_tensor(owner, device=model.device)


@torch.no_grad()
def embed_docs(model, docs, strat=None, nch=None, agg=None):
    """Embed documents (chunk embeddings aggregated into one vector per document).

    Args:
        model (DualEnc): Encoder.
        docs (List[np.ndarray]): Token-id arrays.
        strat (Optional[str]): Chunk strategy (default ``cfg.eval_strat``).
        nch (Optional[int]): Max chunks per document (default ``cfg.ev_ch``).
        agg (Optional[str]): Aggregation (default ``cfg.agg``).

    Returns:
        torch.Tensor: Document embeddings (len(docs), dim).
    """
    agg = agg or model.cfg.agg
    E, owner = chunk_embs(model, docs, strat, nch)
    return aggregate(E, owner, len(docs), agg)


def score_split(model, spl, split, **kw):
    """Cosine-similarity score matrix of one split (queries x its own candidate pool).

    Args:
        model (DualEnc): Encoder.
        spl (dict): Split container produced by ``build_splits``.
        split (str): ``train`` | ``dev`` | ``test``.
        **kw: Forwarded to :func:`embed_docs` (``strat``, ``nch``, ``agg``).

    Returns:
        np.ndarray: Score matrix (n_queries, n_candidates).
    """
    S = spl[split]
    Q = embed_docs(model, [S["qtok"][i] for i in S["qids"]], **kw)
    C = embed_docs(model, [S["ctok"][i] for i in S["cids"]], **kw)
    return (Q @ C.T).float().cpu().numpy()


@torch.no_grad()
def score_both(model, spl, split):
    """Return (doc-mean cosine, MaxSim) score matrices.

    MaxSim = mean over query chunks of the best-matching candidate chunk.

    Args:
        model (DualEnc): Encoder.
        spl (dict): Split container.
        split (str): ``dev`` | ``test`` (or ``train``).

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(mean_scores, maxsim_scores)`` each (n_queries, n_candidates).
    """
    S = spl[split]
    Eq, oq = chunk_embs(model, [S["qtok"][i] for i in S["qids"]])
    Ec, oc = chunk_embs(model, [S["ctok"][i] for i in S["cids"]])
    nq, nc = len(S["qids"]), len(S["cids"])
    mean_s = (aggregate(Eq, oq, nq, model.cfg.agg) @ aggregate(Ec, oc, nc, model.cfg.agg).T).float().cpu().numpy()
    ms = np.zeros((nq, nc), np.float32)
    for i in range(nq):
        rows = Eq[oq == i]
        sims = rows @ Ec.T
        mx = torch.full((rows.size(0), nc), -1.0, device=model.device).scatter_reduce(1, oc[None].expand(rows.size(0), -1), sims, "amax", include_self=False)
        ms[i] = mx.mean(0).cpu().numpy()
    return mean_s, ms


def build_model(exp_cfg, cfg, eos, device):
    """Build the encoder for an experiment (backbone + adaptation technique + trainable-parameter mask).

    Args:
        exp_cfg (dict): Entry of ``EXPERIMENTS`` (arch, method, hyper-parameters).
        cfg (Config): Pipeline configuration.
        eos (int): Padding / EOS token id.
        device (torch.device): Compute device.

    Returns:
        DualEnc: Encoder on ``device`` with ``requires_grad`` set according to the method.
    """
    bb = build_backbone(exp_cfg["arch"], cfg.phase1_ckpt[exp_cfg["arch"]], cfg.strict_load)
    m = exp_cfg["method"]
    if m == "kd":
        print("  student layers (before, after):", prune_layers(bb, exp_cfg.get("keep_every", 2)))
    if m == "lorma":
        print("  LoRMA+ modules:", inject_lorma(bb, exp_cfg.get("which", "attn_ffn"), exp_cfg["r"], exp_cfg["alpha"]))
    if m == "adapters":
        print("  adapters:", inject_adapters(bb, exp_cfg["r"]))
    if m in ("lorma", "adapters"):
        for n, p in bb.named_parameters():
            p.requires_grad = getattr(p, "_is_peft", False) or n.endswith("norm_f.weight")
    elif m == "partial":
        for p in bb.parameters():
            p.requires_grad = False
        set_trainable_top_k(bb, exp_cfg["k"])
    elif m == "zeroshot":
        for p in bb.parameters():
            p.requires_grad = False
    else:
        for p in bb.parameters():
            p.requires_grad = True
    return DualEnc(bb, cfg, eos, device, exp_cfg.get("pool", cfg.tok_pool)).to(device)


def n_params(model):
    """Count trainable and total parameters of a model.

    Args:
        model (nn.Module): Model to inspect.

    Returns:
        Tuple[int, int]: ``(trainable, total)``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
