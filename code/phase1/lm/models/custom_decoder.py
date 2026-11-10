"""From-scratch decoder shared by ``modern_decoder`` (full causal attention) and ``lightweight_sliding_window`` (local causal attention).

Both use: RMSNorm (pre-norm) + RoPE + Grouped-Query Attention executed through ``F.scaled_dot_product_attention`` (fused flash / memory-efficient
kernels when available) + SwiGLU FFN. They differ only in the attention *pattern* (``window_size=None`` vs an int) -- the substantial
architectural difference required by the assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    Args:
        dim (int): feature size.
        eps (float): numerical constant.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Normalise the last dimension.

        Args:
            x (torch.Tensor): (..., dim).

        Returns:
            torch.Tensor: normalised tensor.
        """
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """Rotary-embedding cos/sin tables.

    Args:
        seq_len (int): maximum sequence length.
        head_dim (int): per-head dimension.
        base (float): RoPE base.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: cos and sin, each (seq_len, head_dim/2).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings.

    Args:
        x (torch.Tensor): (batch, heads, seq, head_dim).
        cos (torch.Tensor): (seq, head_dim/2).
        sin (torch.Tensor): (seq, head_dim/2).

    Returns:
        torch.Tensor: rotated tensor, same shape as ``x``.
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :].to(dtype=x.dtype, device=x.device)
    sin = sin[None, None, :, :].to(dtype=x.dtype, device=x.device)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Args:
        dim (int): model dimension.
        hidden_dim (int): hidden dimension.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        """Compute ``w2(silu(w1 x) * w3 x)``.

        Args:
            x (torch.Tensor): (B, T, dim).

        Returns:
            torch.Tensor: (B, T, dim).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GQAFlashAttention(nn.Module):
    """Grouped-Query causal self-attention using torch's fused SDPA kernel (optionally sliding-window).

    Args:
        dim (int): model dimension.
        n_head (int): query heads.
        n_kv_head (int): key/value heads (divides ``n_head``).
        window_size (int or None): None = full causal attention, int = local causal window.
        dropout (float): attention dropout.
    """

    def __init__(self, dim, n_head, n_kv_head, window_size=None, dropout=0.0):
        super().__init__()
        assert dim % n_head == 0, "dim NOT divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head NOT div by n_kv_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.window_size = window_size  # None = full causal attention
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, dim, bias=False)

    def _build_mask(self, T, device, key_padding_mask):
        """Boolean attention mask (True = may attend).

        Args:
            T (int): sequence length.
            device (torch.device): device.
            key_padding_mask (torch.Tensor or None): (B, T) padding mask.

        Returns:
            torch.Tensor: (B or 1, 1, T, T) mask.
        """
        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        if self.window_size is not None:
            idx = torch.arange(T, device=device)
            band = idx[None, :] > (idx[:, None] - self.window_size)
            causal = causal & band
        mask = causal[None, None, :, :]  # (1,1,T,T)
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :].bool()  # (B,1,1,T)
            mask = mask & pad
        return mask

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
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_mask = self._build_mask(T, x.device, attention_mask)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(out)


class DecoderBlock(nn.Module):
    """Pre-norm block: RMSNorm -> GQA -> RMSNorm -> SwiGLU (with residuals).

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
        self.norm1 = RMSNorm(dim)
        self.attn = GQAFlashAttention(dim, n_head, n_kv_head, window_size, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff)

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
        x = x + self.ffn(self.norm2(x))
        return x


class CustomDecoderConfig(PretrainedConfig):
    """HF config of the custom decoder.

    Args:
        vocab_size (int): vocabulary size.
        dim (int): model dimension.
        n_layer (int): number of blocks.
        n_head (int): query heads.
        n_kv_head (int): key/value heads.
        d_ff (int): FFN hidden size.
        max_position_embeddings (int): RoPE cache length.
        window_size (int or None): sliding-window size (None = full attention).
        dropout (float): attention dropout.
    """

    model_type = "custom_decoder"

    def __init__(
        self,
        vocab_size=50257,
        dim=512,
        n_layer=10,
        n_head=8,
        n_kv_head=4,
        d_ff=1536,
        max_position_embeddings=2048,
        window_size=None,
        dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.d_ff = d_ff
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        self.dropout = dropout
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)


class CustomDecoderForCausalLM(PreTrainedModel, GenerationMixin):
    """From-scratch decoder-only LM (weight-tied ``lm_head``).

    Generation through ``model.generate`` recomputes the full sequence every step (``use_cache=False``); the KV-cache generator in
    ``lm.generation.generate`` is the fast path.

    Args:
        config (CustomDecoderConfig): model configuration.
    """

    config_class = CustomDecoderConfig
    _tied_weights_keys = {
        "lm_head.weight": "tok_emb.weight"
    }  # tell save_pretrained/safetensors these share storage on purpose

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    config.dim,
                    config.n_head,
                    config.n_kv_head,
                    config.d_ff,
                    config.window_size,
                    config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.norm_f = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        head_dim = config.dim // config.n_head
        cos, sin = build_rope_cache(config.max_position_embeddings, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.config.use_cache = False
        self.post_init()

    def get_input_embeddings(self):
        """Return the token embedding module.

        Args:
            None

        Returns:
            nn.Embedding: input embeddings.
        """
        return self.tok_emb

    def set_input_embeddings(self, value):
        """Replace the token embedding module.

        Args:
            value (nn.Embedding): new embeddings.

        Returns:
            None
        """
        self.tok_emb = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Language-model forward pass.

        Args:
            input_ids (torch.Tensor): (B, T) token ids.
            attention_mask (torch.Tensor or None): (B, T) padding mask.
            labels (torch.Tensor or None): (B, T) targets (-100 = ignore); shifted internally.

        Returns:
            CausalLMOutput: ``loss`` (if labels given) and ``logits``.
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Feed the full sequence back in at each step (no KV cache).

        Args:
            input_ids (torch.Tensor): current ids.
            attention_mask (torch.Tensor or None): padding mask.
            **kwargs: ignored.

        Returns:
            dict: model inputs.
        """
        return {"input_ids": input_ids, "attention_mask": attention_mask}
