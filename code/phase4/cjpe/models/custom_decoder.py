"""Phase-1 custom decoder backbone (RMSNorm + RoPE + GQA + SwiGLU).

Parameter names are IDENTICAL to Phase 1 so checkpoints load unchanged.
Changes vs the starter: (1) optional ``bidirectional`` flag (for the ``*_bidir`` Phase-1 variant),
(2) mask is built once per forward (not once per layer), (3) every query may attend to itself,
which prevents all-masked rows -> NaN with right-padding / bidirectional windows,
(4) RMSNorm computed in fp32 so fp16 autocast on a T4 does not overflow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class RMSNorm(nn.Module):
    """Root-mean-square layer norm computed in fp32.

    Args:
        dim (int): feature dimension.
        eps (float): numerical-stability constant.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Normalise the last dimension.

        Args:
            x (torch.Tensor): input of shape (..., dim).

        Returns:
            torch.Tensor: normalised tensor in the input dtype.
        """
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return out.to(x.dtype)


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """Pre-compute rotary-embedding cos/sin tables.

    Args:
        seq_len (int): maximum sequence length.
        head_dim (int): per-head dimension (must be even).
        base (float): RoPE base frequency.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: cos and sin tables, each (seq_len, head_dim/2).
    """
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float(); freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings.

    Args:
        x (torch.Tensor): (B, H, T, head_dim) queries or keys.
        cos (torch.Tensor): (T, head_dim/2) cosine table.
        sin (torch.Tensor): (T, head_dim/2) sine table.

    Returns:
        torch.Tensor: rotated tensor with the same shape as ``x``.
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :].to(device=x.device, dtype=x.dtype)
    sin = sin[None, None, :, :].to(device=x.device, dtype=x.dtype)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Args:
        dim (int): model dimension.
        h (int): hidden (intermediate) dimension.
    """
    def __init__(self, dim, h):
        super().__init__()
        self.w1 = nn.Linear(dim, h, bias=False); self.w3 = nn.Linear(dim, h, bias=False); self.w2 = nn.Linear(h, dim, bias=False)

    def forward(self, x):
        """Compute ``w2(silu(w1 x) * w3 x)``.

        Args:
            x (torch.Tensor): (B, T, dim) input.

        Returns:
            torch.Tensor: (B, T, dim) output.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def build_attn_mask(T, device, window, bidirectional, key_padding_mask):
    """Build the boolean attention mask (True = may attend).

    Args:
        T (int): sequence length.
        device (torch.device): target device.
        window (int or None): sliding-window size (None = full attention).
        bidirectional (bool): if True the window is symmetric, otherwise causal.
        key_padding_mask (torch.Tensor or None): (B, T) mask, 1/True for real tokens.

    Returns:
        torch.Tensor: bool mask of shape (B or 1, 1, T, T); the diagonal is always True.
    """
    idx = torch.arange(T, device=device); diff = idx[:, None] - idx[None, :]      # i - j
    if bidirectional:
        allowed = torch.ones(T, T, dtype=torch.bool, device=device) if window is None else diff.abs() < window
    else:
        allowed = diff >= 0
        if window is not None: allowed = allowed & (diff < window)
    allowed = allowed[None, None]
    if key_padding_mask is not None: allowed = allowed & key_padding_mask[:, None, None, :].bool()
    return allowed | torch.eye(T, dtype=torch.bool, device=device)[None, None]


class GQAFlashAttention(nn.Module):
    """Grouped-query attention with RoPE, using ``F.scaled_dot_product_attention``.

    Args:
        dim (int): model dimension.
        n_head (int): number of query heads.
        n_kv_head (int): number of key/value heads (divides ``n_head``).
        window_size (int or None): sliding-window size.
        dropout (float): attention dropout.
    """
    def __init__(self, dim, n_head, n_kv_head, window_size=None, dropout=0.0):
        super().__init__()
        self.n_head, self.n_kv_head = n_head, n_kv_head; self.head_dim = dim // n_head
        self.window_size = window_size; self.dropout = dropout
        self.q_proj = nn.Linear(dim, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin, mask):
        """Self-attention forward pass.

        Args:
            x (torch.Tensor): (B, T, dim) input.
            cos (torch.Tensor): RoPE cosine table (T, head_dim/2).
            sin (torch.Tensor): RoPE sine table (T, head_dim/2).
            mask (torch.Tensor): bool attention mask from ``build_attn_mask``.

        Returns:
            torch.Tensor: (B, T, dim) attention output.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:
            r = self.n_head // self.n_kv_head; k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim))


class DecoderBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> GQA attention -> RMSNorm -> SwiGLU (with residuals).

    Args:
        dim (int): model dimension.
        n_head (int): query heads.
        n_kv_head (int): key/value heads.
        d_ff (int): FFN hidden size.
        window_size (int or None): sliding-window size.
        dropout (float): attention dropout.
    """
    def __init__(self, dim, n_head, n_kv_head, d_ff, window_size=None, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim); self.attn = GQAFlashAttention(dim, n_head, n_kv_head, window_size, dropout)
        self.norm2 = RMSNorm(dim); self.ffn = SwiGLU(dim, d_ff)

    def forward(self, x, cos, sin, mask):
        """Apply the block.

        Args:
            x (torch.Tensor): (B, T, dim) input.
            cos (torch.Tensor): RoPE cosine table.
            sin (torch.Tensor): RoPE sine table.
            mask (torch.Tensor): bool attention mask.

        Returns:
            torch.Tensor: (B, T, dim) output.
        """
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        return x + self.ffn(self.norm2(x))


class CustomDecoderConfig(PretrainedConfig):
    """HF config for the Phase-1 custom decoder.

    Args:
        vocab_size (int): vocabulary size.
        dim (int): model dimension.
        n_layer (int): number of blocks.
        n_head (int): query heads.
        n_kv_head (int): key/value heads.
        d_ff (int): FFN hidden size.
        max_position_embeddings (int): RoPE cache length.
        window_size (int or None): sliding-window size.
        dropout (float): attention dropout.
        bidirectional (bool): symmetric attention window instead of causal.
    """
    model_type = "custom_decoder"

    def __init__(self, vocab_size=50257, dim=512, n_layer=10, n_head=8, n_kv_head=4, d_ff=1536,
                 max_position_embeddings=2048, window_size=None, dropout=0.0, bidirectional=False, **kw):
        self.vocab_size = vocab_size; self.dim = dim; self.n_layer = n_layer; self.n_head = n_head
        self.n_kv_head = n_kv_head; self.d_ff = d_ff; self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size; self.dropout = dropout; self.bidirectional = bidirectional
        super().__init__(**kw)


class CustomDecoderForCausalLM(PreTrainedModel):
    """Phase-1 decoder LM (weight-tied ``lm_head``) exposing ``forward_hidden`` for classification.

    Args:
        config (CustomDecoderConfig): model configuration.
    """
    config_class = CustomDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([DecoderBlock(config.dim, config.n_head, config.n_kv_head, config.d_ff,
                                                  config.window_size, config.dropout) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False); self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(config.max_position_embeddings, config.dim // config.n_head)
        self.register_buffer("rope_cos", cos, persistent=False); self.register_buffer("rope_sin", sin, persistent=False)
        self.config.use_cache = False; self.post_init()

    def get_input_embeddings(self):
        """Return the token embedding module.

        Args:
            None

        Returns:
            nn.Embedding: the input embedding layer.
        """
        return self.tok_emb

    def set_input_embeddings(self, v):
        """Replace the token embedding module.

        Args:
            v (nn.Embedding): new embedding layer.

        Returns:
            None
        """
        self.tok_emb = v

    def forward_hidden(self, input_ids, attention_mask=None):
        """Run the decoder stack and return the final normalised hidden states.

        Args:
            input_ids (torch.Tensor): (B, T) token ids.
            attention_mask (torch.Tensor or None): (B, T) padding mask (1 = real token).

        Returns:
            torch.Tensor: (B, T, dim) hidden states.
        """
        B, T = input_ids.shape; x = self.tok_emb(input_ids)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        mask = build_attn_mask(T, x.device, self.config.window_size, getattr(self.config, "bidirectional", False), attention_mask)
        for b in self.blocks: x = b(x, cos, sin, mask)
        return self.norm_f(x)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        """Language-model forward pass (used only for Phase-1 compatibility).

        Args:
            input_ids (torch.Tensor): (B, T) token ids.
            attention_mask (torch.Tensor or None): (B, T) padding mask.
            labels (torch.Tensor or None): (B, T) next-token targets (-100 = ignore).

        Returns:
            CausalLMOutput: ``loss`` (if labels given) and ``logits``.
        """
        x = self.forward_hidden(input_ids, attention_mask); logits = self.lm_head(x); loss = None
        if labels is not None:
            sl = logits[..., :-1, :].contiguous(); sy = labels[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sy.view(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)
