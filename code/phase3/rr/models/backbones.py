"""Phase-1 backbones as plain ``nn.Module`` wrappers (custom decoder, GPT-2, T5 encoder) and checkpoint loading.

Parameter names of the custom decoder are identical to Phase 1 so its weights load unchanged.
"""
import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from rr.config import ARCH_INFO


class RMSNorm(nn.Module):
    """RMS layer norm computed in fp32.

    Args:
        dim (int): Feature size.
        eps (float): Stabiliser.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Normalise the last dimension.

        Args:
            x (torch.Tensor): Input (..., dim).

        Returns:
            torch.Tensor: Normalised tensor (dtype of ``x``).
        """
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """Rotary-embedding cos / sin tables.

    Args:
        seq_len (int): Maximum length.
        head_dim (int): Head size.
        base (float): Base frequency.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)`` of shape (seq_len, head_dim/2).
    """
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(seq_len).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    """Apply rotary embeddings.

    Args:
        x (torch.Tensor): (B, heads, T, head_dim).
        cos (torch.Tensor): Cosine table (T, head_dim/2).
        sin (torch.Tensor): Sine table (T, head_dim/2).

    Returns:
        torch.Tensor: Rotated tensor (dtype of ``x``).
    """
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    cos = cos[None, None].to(x1.device)
    sin = sin[None, None].to(x1.device)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Args:
        dim (int): Model size.
        hidden (int): Hidden size.
    """

    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        """Apply ``w2(silu(w1 x) * w3 x)``.

        Args:
            x (torch.Tensor): Input (B, T, dim).

        Returns:
            torch.Tensor: Output (B, T, dim).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GQAAttention(nn.Module):
    """Grouped-query attention with optional sliding window and a switch for bidirectional attention.

    Args:
        dim (int): Model size.
        n_head (int): Query heads.
        n_kv_head (int): Key / value heads.
        window_size (Optional[int]): Sliding window (None = full).
    """

    def __init__(self, dim, n_head, n_kv_head, window_size=None):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = n_head, n_kv_head, dim // n_head
        self.window_size, self.bidirectional = window_size, False
        self.q_proj = nn.Linear(dim, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, dim, bias=False)

    def _mask(self, T, device, key_padding_mask):
        """Boolean attention mask (True = may attend); every token may attend to itself.

        Args:
            T (int): Sequence length.
            device (torch.device): Device.
            key_padding_mask (Optional[torch.Tensor]): (B, T) mask with 1 for real tokens.

        Returns:
            torch.Tensor: Mask (B or 1, 1, T, T).
        """
        idx = torch.arange(T, device=device)
        diff = idx[:, None] - idx[None, :]
        if self.bidirectional:
            allowed = torch.ones(T, T, dtype=torch.bool, device=device) if self.window_size is None else diff.abs() < self.window_size
        else:
            allowed = diff >= 0
            if self.window_size is not None:
                allowed = allowed & (diff < self.window_size)
        mask = allowed[None, None]
        if key_padding_mask is not None:
            mask = mask & key_padding_mask[:, None, None, :].bool()
        return mask | torch.eye(T, dtype=torch.bool, device=device)[None, None]

    def forward(self, x, cos, sin, attention_mask=None):
        """Self-attention.

        Args:
            x (torch.Tensor): Input (B, T, dim).
            cos (torch.Tensor): RoPE cosine table.
            sin (torch.Tensor): RoPE sine table.
            attention_mask (Optional[torch.Tensor]): (B, T) padding mask.

        Returns:
            torch.Tensor: Output (B, T, dim).
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:
            r = self.n_head // self.n_kv_head
            k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(T, x.device, attention_mask))
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class DecoderBlock(nn.Module):
    """Pre-norm block (attention + SwiGLU).

    Args:
        dim (int): Model size.
        n_head (int): Query heads.
        n_kv_head (int): Key / value heads.
        d_ff (int): FFN hidden size.
        window_size (Optional[int]): Sliding window.
    """

    def __init__(self, dim, n_head, n_kv_head, d_ff, window_size=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GQAAttention(dim, n_head, n_kv_head, window_size)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff)

    def forward(self, x, cos, sin, attention_mask=None):
        """Run the block.

        Args:
            x (torch.Tensor): Input (B, T, dim).
            cos (torch.Tensor): RoPE cosine.
            sin (torch.Tensor): RoPE sine.
            attention_mask (Optional[torch.Tensor]): Padding mask.

        Returns:
            torch.Tensor: Output (B, T, dim).
        """
        x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
        return x + self.ffn(self.norm2(x))


class CustomDecoderBackbone(nn.Module):
    """Phase-1 custom decoder returning final hidden states.

    Args:
        c (dict): Config with ``vocab, dim, n_head, n_layer, d_ff, n_kv_head, window, max_pos``.
    """

    def __init__(self, c):
        super().__init__()
        self.tok_emb = nn.Embedding(c["vocab"], c["dim"])
        self.blocks = nn.ModuleList([DecoderBlock(c["dim"], c["n_head"], c["n_kv_head"], c["d_ff"], c["window"]) for _ in range(c["n_layer"])])
        self.norm_f = RMSNorm(c["dim"])
        self.hidden_size = c["dim"]
        cos, sin = build_rope_cache(c["max_pos"], c["dim"] // c["n_head"])
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def set_attention_mode(self, bidirectional, window="keep"):
        """Switch every block between causal and bidirectional attention (and optionally change the window).

        Args:
            bidirectional (bool): Bidirectional attention.
            window (Union[str, int, None]): ``"keep"`` or a new window size.

        Returns:
            None
        """
        for b in self.blocks:
            b.attn.bidirectional = bidirectional
            if window != "keep":
                b.attn.window_size = window

    def num_layers(self):
        """Number of blocks.

        Args:
            None

        Returns:
            int: Layer count.
        """
        return len(self.blocks)

    def prune(self, keep):
        """Keep only the listed layers (KD student).

        Args:
            keep (List[int]): Layer indices to keep.

        Returns:
            None
        """
        self.blocks = nn.ModuleList([self.blocks[i] for i in keep])

    def forward(self, input_ids, attention_mask=None):
        """Hidden states of the final norm.

        Args:
            input_ids (torch.Tensor): (B, T).
            attention_mask (Optional[torch.Tensor]): (B, T).

        Returns:
            torch.Tensor: (B, T, dim).
        """
        T = input_ids.shape[1]
        x = self.tok_emb(input_ids)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin, attention_mask)
        return self.norm_f(x)


class GPT2Backbone(nn.Module):
    """GPT-2 transformer wrapper.

    Args:
        cfg (GPT2Config): HF config.
    """

    def __init__(self, cfg):
        super().__init__()
        from transformers import GPT2Model
        self.m = GPT2Model(cfg)
        self.hidden_size = cfg.n_embd

    def num_layers(self):
        """Number of blocks.

        Args:
            None

        Returns:
            int: Layer count.
        """
        return len(self.m.h)

    def prune(self, keep):
        """Keep only the listed layers.

        Args:
            keep (List[int]): Layer indices.

        Returns:
            None
        """
        self.m.h = nn.ModuleList([self.m.h[i] for i in keep])

    def forward(self, input_ids, attention_mask=None):
        """Hidden states.

        Args:
            input_ids (torch.Tensor): (B, T).
            attention_mask (Optional[torch.Tensor]): (B, T).

        Returns:
            torch.Tensor: (B, T, dim).
        """
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class T5EncBackbone(nn.Module):
    """T5 encoder wrapper.

    Args:
        cfg (T5Config): HF config.
    """

    def __init__(self, cfg):
        super().__init__()
        from transformers import T5EncoderModel
        self.m = T5EncoderModel(cfg)
        self.hidden_size = cfg.d_model

    def num_layers(self):
        """Number of encoder blocks.

        Args:
            None

        Returns:
            int: Layer count.
        """
        return len(self.m.encoder.block)

    def prune(self, keep):
        """Keep only the listed layers.

        Args:
            keep (List[int]): Layer indices.

        Returns:
            None
        """
        self.m.encoder.block = nn.ModuleList([self.m.encoder.block[i] for i in keep])

    def forward(self, input_ids, attention_mask=None):
        """Hidden states.

        Args:
            input_ids (torch.Tensor): (B, T).
            attention_mask (Optional[torch.Tensor]): (B, T).

        Returns:
            torch.Tensor: (B, T, d_model).
        """
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def load_state(path):
    """Load a state dict from .pt / .pth / .bin (``{"model"}``, ``{"state_dict"}``, raw or pickled module), .safetensors or an HF folder.

    Args:
        path (str): File or folder.

    Returns:
        dict: State dict with ``module.`` / ``model.`` prefixes stripped.
    """
    if os.path.isdir(path):
        for f in ("model.safetensors", "pytorch_model.bin"):
            if os.path.exists(os.path.join(path, f)):
                path = os.path.join(path, f)
                break
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            sd = obj.state_dict()
        elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        else:
            sd = obj
    return {re.sub(r"^(module\.|model\.)", "", k): v for k, v in sd.items()}


def _n_layers(sd, pat):
    """Number of layers from key names.

    Args:
        sd (dict): State dict.
        pat (str): Regex with one capture group (layer index).

    Returns:
        int: ``max index + 1``.
    """
    return 1 + max(int(m.group(1)) for k in sd for m in [re.search(pat, k)] if m)


def _load_custom_config(ckpt_path):
    """Read ``config.json`` next to a checkpoint (the only reliable source of ``n_head``).

    Args:
        ckpt_path (str): Checkpoint folder or file.

    Returns:
        Optional[dict]: Parsed config or None.
    """
    d = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
    p = os.path.join(d, "config.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def build_backbone(arch, ckpt):
    """Build a Phase-1 backbone and load its weights (asserts that no weight is missing).

    Args:
        arch (str): Key of :data:`rr.config.ARCH_INFO`.
        ckpt (str): Checkpoint path.

    Returns:
        nn.Module: Backbone exposing ``hidden_size``, ``num_layers()``, ``prune()``.
    """
    info = ARCH_INFO[arch]
    raw_ckpt_path = ckpt
    sd = load_state(ckpt)
    if info["kind"] == "custom":
        V, dim = sd["tok_emb.weight"].shape
        hf_cfg = _load_custom_config(raw_ckpt_path)
        if hf_cfg is not None:
            n_head = hf_cfg["n_head"]
            hd = dim // n_head
            n_kv_head = hf_cfg.get("n_kv_head", sd["blocks.0.attn.k_proj.weight"].shape[0] // hd)
            d_ff = hf_cfg.get("d_ff", sd["blocks.0.ffn.w1.weight"].shape[0])
            n_layer = hf_cfg.get("n_layer", _n_layers(sd, r"^blocks\.(\d+)\."))
            window = hf_cfg.get("window_size", info.get("window"))
            print(f"[{arch}] exact config from config.json: n_head={n_head}, n_kv_head={n_kv_head}, head_dim={hd}")
        else:
            print(f"[{arch}] WARNING: no config.json next to checkpoint -- falling back to the shape heuristic (assumes head_dim==64).")
            n_head = dim // 64
            hd = dim // n_head
            n_kv_head = sd["blocks.0.attn.k_proj.weight"].shape[0] // hd
            d_ff = sd["blocks.0.ffn.w1.weight"].shape[0]
            n_layer = _n_layers(sd, r"^blocks\.(\d+)\.")
            window = info.get("window")
        c = dict(vocab=V, dim=dim, n_head=n_head, n_layer=n_layer, d_ff=d_ff, n_kv_head=n_kv_head, window=window, max_pos=2048)
        bb = CustomDecoderBackbone(c)
        sd = {k: v for k, v in sd.items() if not k.startswith("lm_head")}
    elif info["kind"] == "gpt2":
        from transformers import GPT2Config
        sd = {re.sub(r"^transformer\.", "", k): v for k, v in sd.items() if not k.startswith("lm_head")}
        V, dim = sd["wte.weight"].shape
        c = dict(vocab=V, dim=dim, n_head=dim // 64, n_layer=_n_layers(sd, r"^h\.(\d+)\."), n_pos=sd["wpe.weight"].shape[0])
        bb = GPT2Backbone(GPT2Config(vocab_size=V, n_positions=c["n_pos"], n_ctx=c["n_pos"], n_embd=dim, n_layer=c["n_layer"], n_head=c["n_head"]))
    elif info["kind"] == "t5":
        from transformers import T5Config
        V, dm = sd["shared.weight"].shape
        inner = sd["encoder.block.0.layer.0.SelfAttention.q.weight"].shape[0]
        c = dict(vocab=V, d_model=dm, d_ff=sd["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"].shape[0], heads=inner // 64,
                 n_layer=_n_layers(sd, r"^encoder\.block\.(\d+)\."))
        bb = T5EncBackbone(T5Config(vocab_size=V, d_model=dm, d_kv=64, d_ff=c["d_ff"], num_layers=c["n_layer"], num_decoder_layers=c["n_layer"],
                                    num_heads=c["heads"], feed_forward_proj="gated-gelu", dropout_rate=0.1))
        sd = {k: v for k, v in sd.items() if k.startswith(("shared.", "encoder."))}
    if info["kind"] != "custom":
        sd = {("m." + k): v for k, v in sd.items()}
    res = bb.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if not re.search(r"attn\.bias|masked_bias|rope_", k)]
    print(f"[{arch}] {c} | missing={miss[:5]} unexpected={res.unexpected_keys[:5]}")
    assert not miss, "Phase-1 weights did not load into the backbone -- check ARCH_INFO / checkpoint format"
    return bb
