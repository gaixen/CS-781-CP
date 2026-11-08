"""Phase-1 custom decoder backbone (RMSNorm + RoPE + GQA + SwiGLU).

Parameter names are IDENTICAL to Phase 1 so checkpoints load unchanged.
Changes vs the starter: (1) optional ``bidirectional`` flag (for the ``*_bidir`` Phase-1 variant),
(2) the mask is built once per forward (not once per layer), (3) every query may attend to itself,
which prevents all-masked rows -> NaN with right-padding / bidirectional windows,
(4) RMSNorm is computed in fp32 so fp16 autocast on a T4 does not overflow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (computed in fp32).

    Args:
        dim (int): Feature dimension.
        eps (float): Numerical stabiliser.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Normalise the last dimension of ``x``.

        Args:
            x (torch.Tensor): Input of shape (..., dim).

        Returns:
            torch.Tensor: Normalised tensor with the dtype of ``x``.
        """
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return out.to(x.dtype)


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """Pre-compute rotary-embedding cosine / sine tables.

    Args:
        seq_len (int): Maximum sequence length.
        head_dim (int): Attention head dimension.
        base (float): RoPE base frequency.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)`` each of shape (seq_len, head_dim / 2).
    """
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings.

    Args:
        x (torch.Tensor): Tensor of shape (B, heads, T, head_dim).
        cos (torch.Tensor): Cosine table of shape (T, head_dim / 2).
        sin (torch.Tensor): Sine table of shape (T, head_dim / 2).

    Returns:
        torch.Tensor: Rotated tensor with the same shape as ``x``.
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :].to(device=x.device, dtype=x.dtype)
    sin = sin[None, None, :, :].to(device=x.device, dtype=x.dtype)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Args:
        dim (int): Model dimension.
        h (int): Hidden (feed-forward) dimension.
    """

    def __init__(self, dim, h):
        super().__init__()
        self.w1 = nn.Linear(dim, h, bias=False)
        self.w3 = nn.Linear(dim, h, bias=False)
        self.w2 = nn.Linear(h, dim, bias=False)

    def forward(self, x):
        """Apply ``w2(silu(w1 x) * w3 x)``.

        Args:
            x (torch.Tensor): Input of shape (B, T, dim).

        Returns:
            torch.Tensor: Output of shape (B, T, dim).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def build_attn_mask(T, device, window, bidirectional, key_padding_mask):
    """Build the boolean attention mask (True = may attend).

    Args:
        T (int): Sequence length.
        device (torch.device): Target device.
        window (Optional[int]): Sliding-window size, or None for full attention.
        bidirectional (bool): If True tokens attend both directions, else causal.
        key_padding_mask (Optional[torch.Tensor]): (B, T) mask with 1 for real tokens.

    Returns:
        torch.Tensor: Bool tensor of shape (B or 1, 1, T, T).
    """
    idx = torch.arange(T, device=device)
    diff = idx[:, None] - idx[None, :]      # i - j
    if bidirectional:
        allowed = torch.ones(T, T, dtype=torch.bool, device=device) if window is None else diff.abs() < window
    else:
        allowed = diff >= 0
        if window is not None:
            allowed = allowed & (diff < window)
    allowed = allowed[None, None]
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask[:, None, None, :].bool()
    return allowed | torch.eye(T, dtype=torch.bool, device=device)[None, None]


class GQAFlashAttention(nn.Module):
    """Grouped-query attention using ``scaled_dot_product_attention``.

    Args:
        dim (int): Model dimension.
        n_head (int): Number of query heads.
        n_kv_head (int): Number of key / value heads.
        window_size (Optional[int]): Sliding window size (None = full).
        dropout (float): Attention dropout probability.
    """

    def __init__(self, dim, n_head, n_kv_head, window_size=None, dropout=0.0):
        super().__init__()
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.head_dim = dim // n_head
        self.window_size = window_size
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin, mask):
        """Self-attention forward pass.

        Args:
            x (torch.Tensor): Input of shape (B, T, dim).
            cos (torch.Tensor): RoPE cosine table.
            sin (torch.Tensor): RoPE sine table.
            mask (torch.Tensor): Boolean attention mask from :func:`build_attn_mask`.

        Returns:
            torch.Tensor: Output of shape (B, T, dim).
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:
            r = self.n_head // self.n_kv_head
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim))


class DecoderBlock(nn.Module):
    """Pre-norm transformer block (attention + SwiGLU FFN).

    Args:
        dim (int): Model dimension.
        n_head (int): Number of query heads.
        n_kv_head (int): Number of key / value heads.
        d_ff (int): Feed-forward hidden size.
        window_size (Optional[int]): Sliding window size.
        dropout (float): Attention dropout.
    """

    def __init__(self, dim, n_head, n_kv_head, d_ff, window_size=None, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GQAFlashAttention(dim, n_head, n_kv_head, window_size, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff)

    def forward(self, x, cos, sin, mask):
        """Run the block.

        Args:
            x (torch.Tensor): Input of shape (B, T, dim).
            cos (torch.Tensor): RoPE cosine table.
            sin (torch.Tensor): RoPE sine table.
            mask (torch.Tensor): Boolean attention mask.

        Returns:
            torch.Tensor: Output of shape (B, T, dim).
        """
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        return x + self.ffn(self.norm2(x))


class CustomDecoderConfig(PretrainedConfig):
    """HuggingFace config of the Phase-1 custom decoder.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        n_layer (int): Number of blocks.
        n_head (int): Query heads.
        n_kv_head (int): Key / value heads.
        d_ff (int): Feed-forward hidden size.
        max_position_embeddings (int): Maximum positions for the RoPE cache.
        window_size (Optional[int]): Sliding-window size (None = full attention).
        dropout (float): Attention dropout.
        bidirectional (bool): Bidirectional attention flag.
    """
    model_type = "custom_decoder"

    def __init__(self, vocab_size=50257, dim=512, n_layer=10, n_head=8, n_kv_head=4, d_ff=1536,
                 max_position_embeddings=2048, window_size=None, dropout=0.0, bidirectional=False, **kw):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.d_ff = d_ff
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        super().__init__(**kw)


class CustomDecoderForCausalLM(PreTrainedModel):
    """Phase-1 decoder with tied input / output embeddings.

    Args:
        config (CustomDecoderConfig): Model configuration.
    """
    config_class = CustomDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([DecoderBlock(config.dim, config.n_head, config.n_kv_head, config.d_ff,
                                                  config.window_size, config.dropout) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(config.max_position_embeddings, config.dim // config.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.config.use_cache = False
        self.post_init()

    def get_input_embeddings(self):
        """Return the token embedding module.

        Args:
            None

        Returns:
            nn.Embedding: Token embedding.
        """
        return self.tok_emb

    def set_input_embeddings(self, v):
        """Replace the token embedding module.

        Args:
            v (nn.Embedding): New embedding module.

        Returns:
            None
        """
        self.tok_emb = v

    def forward_hidden(self, input_ids, attention_mask=None):
        """Return final-norm hidden states (no LM head).

        Args:
            input_ids (torch.Tensor): Token ids of shape (B, T).
            attention_mask (Optional[torch.Tensor]): (B, T) mask with 1 for real tokens.

        Returns:
            torch.Tensor: Hidden states of shape (B, T, dim).
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        mask = build_attn_mask(T, x.device, self.config.window_size, getattr(self.config, "bidirectional", False), attention_mask)
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        return self.norm_f(x)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        """Causal-LM forward pass.

        Args:
            input_ids (torch.Tensor): Token ids of shape (B, T).
            attention_mask (Optional[torch.Tensor]): (B, T) padding mask.
            labels (Optional[torch.Tensor]): Labels for next-token loss (-100 ignored).
            **kw: Ignored extra keyword arguments.

        Returns:
            CausalLMOutput: Output with ``loss`` (if labels given) and ``logits``.
        """
        x = self.forward_hidden(input_ids, attention_mask)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            sl = logits[..., :-1, :].contiguous()
            sy = labels[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sy.view(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)
