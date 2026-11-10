"""CJPE classifier = Phase-1 backbone + (token pooling) + [optional hierarchical doc encoder] + (deep) MLP head.

REPRESENTATION: last_n | first_n | head_tail  -> one sequence per document ("flat")
                hier                          -> overlapping chunks (taken from the END of the document) ->
                                                 backbone+pool per chunk -> doc encoder (mean|attn|transformer|gru)
All new (task) parameters live under ``model.head.*``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TokenPool(nn.Module):
    """Pool right-padded token states (N, T, D) into one vector per sequence.

    Args:
        dim (int): hidden size D.
        mode (str): ``last`` | ``mean`` | ``attn`` | ``mean_last``.
    """
    def __init__(self, dim, mode):
        super().__init__(); self.mode = mode
        assert mode in ("last", "mean", "attn", "mean_last"), mode
        if mode == "attn": self.score = nn.Linear(dim, 1)
        self.out_dim = dim * 2 if mode == "mean_last" else dim

    def forward(self, h, mask):
        """Pool hidden states.

        Args:
            h (torch.Tensor): (N, T, D) hidden states.
            mask (torch.Tensor): (N, T) bool padding mask (True = real token).

        Returns:
            torch.Tensor: (N, out_dim) float32 pooled vectors.
        """
        m = mask.bool(); n = m.sum(1).clamp(min=1); hf = h.float()
        last = hf[torch.arange(hf.size(0), device=hf.device), n - 1]
        if self.mode == "last": return last
        mean = (hf * m[..., None]).sum(1) / n[:, None]
        if self.mode == "mean": return mean
        if self.mode == "mean_last": return torch.cat([mean, last], -1)
        s = self.score(hf).squeeze(-1).float().masked_fill(~m, -1e4)
        return (F.softmax(s, 1)[..., None] * hf).sum(1)


class ResBlock(nn.Module):
    """Pre-norm residual MLP block used in deeper heads.

    Args:
        d (int): feature dimension.
        p (float): dropout probability.
    """
    def __init__(self, d, p):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(p), nn.Linear(d * 2, d), nn.Dropout(p))

    def forward(self, x):
        """Apply the block with a residual connection.

        Args:
            x (torch.Tensor): (B, d) input.

        Returns:
            torch.Tensor: (B, d) output.
        """
        return x + self.net(x)


class MLPHead(nn.Module):
    """Classification head. ``n_layers=1`` -> LN-Linear-GELU-Dropout-Linear (starter head); ``n_layers>1`` adds residual blocks.

    Args:
        in_dim (int): input feature size.
        hidden (int): hidden width.
        n_layers (int): total depth (1 = starter head).
        p (float): dropout probability.
        n_out (int): number of classes.
    """
    def __init__(self, in_dim, hidden, n_layers, p, n_out=2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p))
        self.blocks = nn.ModuleList([ResBlock(hidden, p) for _ in range(n_layers - 1)])
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x):
        """Compute class logits.

        Args:
            x (torch.Tensor): (B, in_dim) features.

        Returns:
            torch.Tensor: (B, n_out) logits.
        """
        x = self.inp(x)
        for b in self.blocks: x = b(x)
        return self.out(x)


class DocEncoder(nn.Module):
    """Aggregate chunk vectors (B, C, d) + chunk mask into a document vector (B, d).

    Chunk index k = distance from the END of the document.

    Args:
        d (int): chunk-vector size.
        cfg (dict): experiment config (uses ``DOC_AGG, MAX_CHUNKS, DROPOUT, DOC_LAYERS``).
    """
    def __init__(self, d, cfg):
        super().__init__(); self.kind = cfg["DOC_AGG"]; assert self.kind in ("mean", "attn", "transformer", "gru")
        self.pos = nn.Embedding(cfg["MAX_CHUNKS"], d) if self.kind in ("transformer", "gru") else None
        if self.kind == "transformer":
            nh = next(h for h in (8, 6, 4, 2, 1) if d % h == 0)
            layer = nn.TransformerEncoderLayer(d, nh, 2 * d, cfg["DROPOUT"], activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, cfg["DOC_LAYERS"], enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        elif self.kind == "gru":
            assert d % 2 == 0
            self.enc = nn.GRU(d, d // 2, cfg["DOC_LAYERS"], batch_first=True, bidirectional=True,
                              dropout=cfg["DROPOUT"] if cfg["DOC_LAYERS"] > 1 else 0.0)
        if self.kind != "mean": self.score = nn.Linear(d, 1)

    def forward(self, x, cmask):
        """Aggregate chunks.

        Args:
            x (torch.Tensor): (B, C, d) chunk vectors (zero-padded).
            cmask (torch.Tensor): (B, C) bool mask, True for real chunks.

        Returns:
            torch.Tensor: (B, d) float32 document vectors.
        """
        B, C, d = x.shape
        if self.pos is not None: x = x + self.pos(torch.arange(C, device=x.device))[None]
        if self.kind == "transformer": x = self.norm(self.enc(x, src_key_padding_mask=~cmask))
        elif self.kind == "gru":
            lens = cmask.sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x.float(), lens, batch_first=True, enforce_sorted=False)
            out, _ = self.enc(packed); x, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=C)
        xf = x.float()
        if self.kind == "mean":
            w = cmask.float(); return (xf * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)
        s = self.score(xf).squeeze(-1).float().masked_fill(~cmask, -1e4)
        return (F.softmax(s, 1)[..., None] * xf).sum(1)


class HeadStack(nn.Module):
    """All new task parameters: token pooling, optional doc encoder, MLP head, optional KD feature projection.

    Args:
        D (int): backbone hidden size.
        cfg (dict): experiment config.
        feat_target_dim (int or None): teacher feature size (enables the KD ``feat_proj`` layer).
    """
    def __init__(self, D, cfg, feat_target_dim=None):
        super().__init__()
        self.pool = TokenPool(D, cfg["POOLING"]); d = self.pool.out_dim
        self.doc = DocEncoder(d, cfg) if cfg["REPRESENTATION"] == "hier" else None
        self.mlp = MLPHead(d, cfg["HEAD_HIDDEN"], cfg["HEAD_LAYERS"], cfg["DROPOUT"])
        self.feat_proj = nn.Linear(d, feat_target_dim) if feat_target_dim else None   # KD: student feat -> teacher feat space
        self.feat_dim = d


class CJPEModel(nn.Module):
    """Full judgment-prediction classifier.

    Args:
        backbone (Backbone): Phase-1 backbone wrapper.
        cfg (dict): experiment config (uses ``REPRESENTATION, GRAD_CKPT, CHUNK_MBS`` and head settings).
        feat_target_dim (int or None): teacher feature size for feature distillation.
    """
    def __init__(self, backbone, cfg, feat_target_dim=None):
        super().__init__()
        self.backbone = backbone; self.head = HeadStack(backbone.hidden_dim, cfg, feat_target_dim)
        self.rep = cfg["REPRESENTATION"]; self.grad_ckpt = cfg["GRAD_CKPT"]; self.chunk_mbs = cfg["CHUNK_MBS"]

    def _encode(self, ids, mask):
        """Backbone + token pooling.

        Args:
            ids (torch.Tensor): (N, T) token ids.
            mask (torch.Tensor): (N, T) bool padding mask.

        Returns:
            torch.Tensor: (N, pool_dim) pooled vectors.
        """
        return self.head.pool(self.backbone(ids, mask.long()), mask)

    def _encode_chunks(self, ids, mask):
        """Encode many chunks in micro-batches (with optional gradient checkpointing).

        Args:
            ids (torch.Tensor): (N, T) chunk token ids.
            mask (torch.Tensor): (N, T) bool padding mask.

        Returns:
            torch.Tensor: (N, pool_dim) chunk vectors.
        """
        outs = []
        for s in range(0, ids.size(0), self.chunk_mbs):
            i, m = ids[s:s + self.chunk_mbs], mask[s:s + self.chunk_mbs]
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                outs.append(checkpoint(self._encode, i, m, use_reentrant=False))
            else: outs.append(self._encode(i, m))
        return torch.cat(outs)

    def forward(self, batch):
        """Predict class logits for a collated batch.

        Args:
            batch (dict): output of ``collate`` moved to the device (``ids, mask`` and, for ``hier``,
                ``doc_index, chunk_pos, B, C``).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (logits (B, 2), pooled document features (B, d)), both float32.
        """
        if self.rep != "hier":
            feats = self._encode(batch["ids"], batch["mask"])
        else:
            vecs = self._encode_chunks(batch["ids"], batch["mask"]); B, C = batch["B"], batch["C"]
            x = torch.zeros(B, C, vecs.size(-1), device=vecs.device, dtype=vecs.dtype)
            x = x.index_put((batch["doc_index"], batch["chunk_pos"]), vecs)
            cmask = torch.zeros(B, C, dtype=torch.bool, device=vecs.device); cmask[batch["doc_index"], batch["chunk_pos"]] = True
            feats = self.head.doc(x, cmask)
        return self.head.mlp(feats).float(), feats.float()
