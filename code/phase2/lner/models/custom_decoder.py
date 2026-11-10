"""Phase-1 custom decoder (same parameter names as Phase-1 ``custom_decoder.py``): RMSNorm + RoPE + GQA + SwiGLU.

The attention has an extra ``bidirectional`` switch: a causal decoder only sees left context, which hurts tagging.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS layer norm in fp32.

    Args:
        dim (int): feature size.
        eps (float): numerical constant.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Normalise the last dimension.

        Args:
            x (torch.Tensor): (..., dim).

        Returns:
            torch.Tensor: normalised tensor in the input dtype.
        """
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """Rotary-embedding cos/sin tables.

    Args:
        seq_len (int): max length.
        head_dim (int): per-head dim.
        base (float): RoPE base.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: cos, sin of shape (seq_len, head_dim/2).
    """
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(seq_len).float(), inv); return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings.

    Args:
        x (torch.Tensor): (B, H, T, head_dim).
        cos (torch.Tensor): (T, head_dim/2).
        sin (torch.Tensor): (T, head_dim/2).

    Returns:
        torch.Tensor: rotated tensor, same shape/dtype as ``x``.
    """
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    cos = cos[None, None].to(x1.device); sin = sin[None, None].to(x1.device)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward.

    Args:
        dim (int): model dim.
        hidden (int): hidden dim.
    """
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False); self.w3 = nn.Linear(dim, hidden, bias=False); self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        """Compute ``w2(silu(w1 x) * w3 x)``.

        Args:
            x (torch.Tensor): (B, T, dim).

        Returns:
            torch.Tensor: (B, T, dim).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GQAAttention(nn.Module):
    """Grouped-query attention with RoPE and optional sliding window / bidirectional mask.

    Args:
        dim (int): model dim.
        n_head (int): query heads.
        n_kv_head (int): key/value heads.
        window_size (int or None): sliding-window size (None = full).
    """
    def __init__(self, dim, n_head, n_kv_head, window_size=None):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = n_head, n_kv_head, dim // n_head
        self.window_size, self.bidirectional = window_size, False
        self.q_proj = nn.Linear(dim, n_head * self.head_dim, bias=False); self.k_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False); self.o_proj = nn.Linear(n_head * self.head_dim, dim, bias=False)

    def _mask(self, T, device, key_padding_mask):
        """Boolean attention mask (True = may attend); every token may attend to itself (no NaN rows on padding).

        Args:
            T (int): sequence length.
            device (torch.device): device.
            key_padding_mask (torch.Tensor or None): (B, T) padding mask.

        Returns:
            torch.Tensor: (B or 1, 1, T, T) bool mask.
        """
        idx = torch.arange(T, device=device); diff = idx[:, None] - idx[None, :]          # query - key
        if self.bidirectional:
            allowed = torch.ones(T, T, dtype=torch.bool, device=device) if self.window_size is None else diff.abs() < self.window_size
        else:
            allowed = diff >= 0
            if self.window_size is not None: allowed = allowed & (diff < self.window_size)
        mask = allowed[None, None]
        if key_padding_mask is not None: mask = mask & key_padding_mask[:, None, None, :].bool()
        return mask | torch.eye(T, dtype=torch.bool, device=device)[None, None]

    def forward(self, x, cos, sin, attention_mask=None):
        """Self-attention.

        Args:
            x (torch.Tensor): (B, T, dim).
            cos (torch.Tensor): RoPE cos table.
            sin (torch.Tensor): RoPE sin table.
            attention_mask (torch.Tensor or None): (B, T) padding mask.

        Returns:
            torch.Tensor: (B, T, dim).
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:
            r = self.n_head // self.n_kv_head; k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(T, x.device, attention_mask))
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class DecoderBlock(nn.Module):
    """Pre-norm block: RMSNorm -> GQA -> RMSNorm -> SwiGLU (with residuals).

    Args:
        dim (int): model dim.
        n_head (int): query heads.
        n_kv_head (int): kv heads.
        d_ff (int): FFN hidden size.
        window_size (int or None): sliding window.
    """
    def __init__(self, dim, n_head, n_kv_head, d_ff, window_size=None):
        super().__init__()
        self.norm1 = RMSNorm(dim); self.attn = GQAAttention(dim, n_head, n_kv_head, window_size)
        self.norm2 = RMSNorm(dim); self.ffn = SwiGLU(dim, d_ff)

    def forward(self, x, cos, sin, attention_mask=None):
        """Apply the block.

        Args:
            x (torch.Tensor): (B, T, dim).
            cos (torch.Tensor): RoPE cos table.
            sin (torch.Tensor): RoPE sin table.
            attention_mask (torch.Tensor or None): padding mask.

        Returns:
            torch.Tensor: (B, T, dim).
        """
        x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
        return x + self.ffn(self.norm2(x))


class CustomDecoderBackbone(nn.Module):
    """Phase-1 decoder stack used as an encoder backbone for token classification.

    Args:
        c (dict): dims ``vocab, dim, n_head, n_kv_head, d_ff, n_layer, window, max_pos``.
    """
    def __init__(self, c):
        super().__init__()
        self.tok_emb = nn.Embedding(c["vocab"], c["dim"])
        self.blocks = nn.ModuleList([DecoderBlock(c["dim"], c["n_head"], c["n_kv_head"], c["d_ff"], c["window"]) for _ in range(c["n_layer"])])
        self.norm_f = RMSNorm(c["dim"]); self.hidden_size = c["dim"]
        cos, sin = build_rope_cache(c["max_pos"], c["dim"] // c["n_head"])
        self.register_buffer("rope_cos", cos, persistent=False); self.register_buffer("rope_sin", sin, persistent=False)

    def set_attention_mode(self, bidirectional, window="keep"):
        """Switch every layer between causal and bidirectional attention (and optionally change the window).

        Args:
            bidirectional (bool): symmetric attention if True.
            window (str, int or None): ``"keep"`` (Phase-1 window), None (full) or an int.

        Returns:
            None
        """
        for b in self.blocks:
            b.attn.bidirectional = bidirectional
            if window != "keep": b.attn.window_size = window

    def num_layers(self):
        """Number of blocks.

        Args:
            None

        Returns:
            int: layer count.
        """
        return len(self.blocks)

    def prune(self, keep):
        """Keep only the listed layers (KD student).

        Args:
            keep (list[int]): layer indices to keep.

        Returns:
            None
        """
        self.blocks = nn.ModuleList([self.blocks[i] for i in keep])

    def forward(self, input_ids, attention_mask=None):
        """Encode tokens.

        Args:
            input_ids (torch.Tensor): (B, T).
            attention_mask (torch.Tensor or None): (B, T) padding mask.

        Returns:
            torch.Tensor: (B, T, dim) hidden states.
        """
        T = input_ids.shape[1]; x = self.tok_emb(input_ids); cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for b in self.blocks: x = b(x, cos, sin, attention_mask)
        return self.norm_f(x)
