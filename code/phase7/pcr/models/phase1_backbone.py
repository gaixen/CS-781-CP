"""Phase-1 backbone loading + adaptation techniques (reused from the CJPE phase, unchanged).

Techniques: full FT (optionally layer-wise LR decay), partial top-k / gradual unfreezing,
LoRMA+ (compulsory), BitFit (works on bias-free archs), bottleneck Adapters, layer-pruning (KD student).
"""
import json
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel, T5Config, T5EncoderModel
from transformers.pytorch_utils import Conv1D

from pcr.models.custom_decoder import CustomDecoderConfig, CustomDecoderForCausalLM

ARCH_INFO = {
    "gpt2_decoder":                     {"kind": "gpt2"},
    "modern_decoder":                   {"kind": "custom", "window": None, "bidirectional": False},
    "lightweight_sliding_window":       {"kind": "custom", "window": 256,  "bidirectional": False},
    "lightweight_sliding_window_bidir": {"kind": "custom", "window": 256,  "bidirectional": True},
    "seq2seq_t5":                       {"kind": "t5"},
}
# module names that LoRMA+ wraps, per backbone kind
LORMA_TARGETS = {
    "custom": {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "ffn": ("w1", "w2", "w3")},
    "gpt2":   {"qkv": ("c_attn",),                    "o": (),          "ffn": ("c_fc",)},
    "t5":     {"qkv": ("q", "k", "v"),                "o": ("o",),       "ffn": ("wi", "wi_0", "wi_1", "wo")},
}
_TOLERATED_MISSING = ("lm_head.weight", "attn.bias", "attn.masked_bias", "encoder.embed_tokens.weight", "rope_cos", "rope_sin")


def load_state(ckpt):
    """Load a checkpoint state dict and strip torch.compile / DDP prefixes.

    Args:
        ckpt (str): Checkpoint directory (``model.safetensors`` or ``pytorch_model.bin``) or a weight file.

    Returns:
        dict: State dict with cleaned parameter names.
    """
    if os.path.isdir(ckpt):
        p = os.path.join(ckpt, "model.safetensors")
        sd = load_file(p) if os.path.exists(p) else torch.load(os.path.join(ckpt, "pytorch_model.bin"), map_location="cpu")
    else:
        sd = load_file(ckpt) if ckpt.endswith(".safetensors") else torch.load(ckpt, map_location="cpu")
    out = {}
    for k, v in sd.items():            # strip torch.compile / DDP prefixes
        k = k.replace("_orig_mod.", "")
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def load_cfg_json(ckpt):
    """Read ``config.json`` next to a checkpoint if present.

    Args:
        ckpt (str): Checkpoint directory or file path.

    Returns:
        Optional[dict]: Parsed config or None when no config.json exists.
    """
    d = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
    p = os.path.join(d, "config.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _n_layers(sd, pattern):
    """Infer the number of layers from state-dict key names.

    Args:
        sd (dict): State dict.
        pattern (str): Regex with one capture group for the layer index.

    Returns:
        int: ``max index + 1`` (0 if no key matches).
    """
    idxs = {int(m.group(1)) for k in sd if (m := re.match(pattern, k))}
    return max(idxs) + 1 if idxs else 0


def _check_load(arch, res, strict):
    """Report (and optionally raise on) checkpoint keys missing from the model.

    Args:
        arch (str): Architecture name (for messages).
        res: Result of ``load_state_dict(strict=False)``.
        strict (bool): Raise ``RuntimeError`` if non-tolerated weights are missing.

    Returns:
        None
    """
    missing = [k for k in res.missing_keys if not k.endswith(_TOLERATED_MISSING)]
    print(f"[{arch}] load: missing={len(missing)} (non-tolerated) unexpected={len(res.unexpected_keys)}")
    if missing:
        msg = f"[{arch}] {len(missing)} weights NOT found in checkpoint (would be randomly initialised): {missing[:4]}"
        if strict:
            raise RuntimeError(msg + "  -- check the checkpoint path/arch, or set STRICT_LOAD=False")
        print("WARNING:", msg)


class Backbone(nn.Module):
    """Uniform wrapper: ``forward(ids, mask) -> (B, T, hidden_dim)`` for every Phase-1 architecture.

    Args:
        model (nn.Module): Underlying model.
        kind (str): One of ``custom``, ``gpt2``, ``t5``.
        hidden_dim (int): Hidden size.
        max_len (int): Maximum supported sequence length.
        bidirectional (bool): Whether attention is bidirectional.
    """

    def __init__(self, model, kind, hidden_dim, max_len, bidirectional):
        super().__init__()
        self.model, self.kind, self.hidden_dim, self.max_len, self.bidirectional = model, kind, hidden_dim, max_len, bidirectional

    def forward(self, input_ids, attention_mask):
        """Compute hidden states.

        Args:
            input_ids (torch.Tensor): Token ids (B, T).
            attention_mask (torch.Tensor): Padding mask (B, T).

        Returns:
            torch.Tensor: Hidden states (B, T, hidden_dim).
        """
        if self.kind == "custom":
            return self.model.forward_hidden(input_ids, attention_mask)
        if self.kind == "gpt2":
            return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def get_layers(self):
        """Return the transformer layer list.

        Args:
            None

        Returns:
            nn.ModuleList: The layers of the backbone.
        """
        return {"custom": lambda: self.model.blocks, "gpt2": lambda: self.model.h, "t5": lambda: self.model.encoder.block}[self.kind]()

    def set_layers(self, layers):
        """Replace the layer list (used for pruning).

        Args:
            layers (Iterable[nn.Module]): New layers.

        Returns:
            None
        """
        layers = nn.ModuleList(layers)
        if self.kind == "custom":
            self.model.blocks = layers
            self.model.config.n_layer = len(layers)
        elif self.kind == "gpt2":
            self.model.h = layers
            self.model.config.n_layer = len(layers)
        else:
            self.model.encoder.block = layers
            self.model.config.num_layers = len(layers)

    def get_final_norm(self):
        """Return the final normalisation module.

        Args:
            None

        Returns:
            nn.Module: Final norm layer.
        """
        return {"custom": lambda: self.model.norm_f, "gpt2": lambda: self.model.ln_f, "t5": lambda: self.model.encoder.final_layer_norm}[self.kind]()

    def get_embeddings(self):
        """Return the input embedding module.

        Args:
            None

        Returns:
            nn.Module: Token embedding.
        """
        return self.model.get_input_embeddings()


def build_backbone(arch, ckpt, strict=True):
    """Build a Phase-1 backbone and load its checkpoint.

    Args:
        arch (str): Architecture key of :data:`ARCH_INFO`.
        ckpt (str): Checkpoint directory / file.
        strict (bool): Raise when checkpoint weights are missing.

    Returns:
        Backbone: The loaded backbone wrapper.
    """
    if ckpt is None:
        raise ValueError(f"No Phase-1 checkpoint path set for '{arch}' (PHASE1_CKPT).")
    info = ARCH_INFO[arch]
    sd = load_state(ckpt)
    cfg_json = load_cfg_json(ckpt)
    if info["kind"] == "custom":
        V, dim = sd["tok_emb.weight"].shape
        bidir = info.get("bidirectional", False)
        if cfg_json is not None:
            n_head = cfg_json["n_head"]
            hd = dim // n_head
            n_kv = cfg_json.get("n_kv_head", sd["blocks.0.attn.k_proj.weight"].shape[0] // hd)
            d_ff = cfg_json.get("d_ff", sd["blocks.0.ffn.w1.weight"].shape[0])
            n_layer = cfg_json.get("n_layer", _n_layers(sd, r"^blocks\.(\d+)\."))
            window = cfg_json.get("window_size", info.get("window"))
            bidir = cfg_json.get("bidirectional", cfg_json.get("bidir", bidir))
            max_pos = max(2048, cfg_json.get("max_position_embeddings", 2048))
        else:
            print(f"[{arch}] WARNING: no config.json -- assuming head_dim==64, verify it.")
            n_head = dim // 64
            hd = dim // n_head
            n_kv = sd["blocks.0.attn.k_proj.weight"].shape[0] // hd
            d_ff = sd["blocks.0.ffn.w1.weight"].shape[0]
            n_layer = _n_layers(sd, r"^blocks\.(\d+)\.")
            window = info.get("window")
            max_pos = 2048
        cfg = CustomDecoderConfig(vocab_size=V, dim=dim, n_head=n_head, n_layer=n_layer, d_ff=d_ff, n_kv_head=n_kv,
                                  window_size=window, bidirectional=bool(bidir), max_position_embeddings=max_pos)
        model = CustomDecoderForCausalLM(cfg)
        _check_load(arch, model.load_state_dict(sd, strict=False), strict)
        bb = Backbone(model, "custom", dim, max_pos, bool(bidir))
    elif info["kind"] == "gpt2":
        keep = set(GPT2Config().to_dict())
        cfg = GPT2Config(**{k: v for k, v in cfg_json.items() if k in keep}) if cfg_json else GPT2Config()
        lm = GPT2LMHeadModel(cfg)
        _check_load(arch, lm.load_state_dict(sd, strict=False), strict)
        bb = Backbone(lm.transformer, "gpt2", cfg.n_embd, cfg.n_positions, False)
    else:  # t5: ENCODER only (bidirectional) -- good fit for whole-document classification
        cfg = T5Config(**cfg_json) if cfg_json else T5Config()
        model = T5EncoderModel(cfg)
        _check_load(arch, model.load_state_dict(sd, strict=False), strict)
        bb = Backbone(model, "t5", cfg.d_model, 10 ** 9, True)
    n = sum(p.numel() for p in bb.parameters())
    print(f"[{arch}] {n/1e6:.1f}M params, hidden_dim={bb.hidden_dim}, layers={len(bb.get_layers())}, bidirectional={bb.bidirectional}")
    return bb


def mark_peft(module):
    """Tag every parameter of ``module`` as parameter-efficient (trainable PEFT weight).

    Args:
        module (nn.Module): Module whose parameters are tagged.

    Returns:
        None
    """
    for p in module.parameters():
        p._is_peft = True


# ---------------- LoRMA+ (compulsory) ----------------
class LoRMAPlusLinear(nn.Module):
    """LoRMA+ (Bihany, Patel & Modi, ACL Findings 2025, arXiv:2506.07621).

    ``h = ((alpha/r) B A + I)(W0 x)``; ``B`` is zero-initialised so the layer is the identity at start.

    Args:
        base (nn.Module): Frozen linear (or GPT-2 ``Conv1D``) layer to wrap.
        r (int): Rank of the multiplicative update.
        alpha (float): Scaling numerator.
    """

    def __init__(self, base, r=16, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        out_f = base.out_features if isinstance(base, nn.Linear) else base.nf     # Conv1D (GPT-2) has .nf
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, out_f) * (1.0 / math.sqrt(out_f)))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        mark_peft(self)
        for p in base.parameters():
            p._is_peft = False

    def forward(self, x):
        """Apply the wrapped layer followed by the multiplicative low-rank update.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: ``y + scale * (y A^T) B^T`` where ``y = base(x)``.
        """
        y = self.base(x)
        return y + self.scale * ((y @ self.A.T.to(y.dtype)) @ self.B.T.to(y.dtype))


def inject_lorma(backbone, which="qkv", r=16, alpha=16):
    """Wrap target linear layers of a backbone with :class:`LoRMAPlusLinear`.

    Args:
        backbone (Backbone): Backbone to modify in place.
        which (str): ``qkv`` | ``attn`` | ``attn_ffn`` - which projection groups to wrap.
        r (int): LoRMA+ rank.
        alpha (float): LoRMA+ alpha.

    Returns:
        int: Number of wrapped modules.
    """
    t = LORMA_TARGETS[backbone.kind]
    names = {"qkv": t["qkv"], "attn": t["qkv"] + t["o"], "attn_ffn": t["qkv"] + t["o"] + t["ffn"]}[which]
    n = 0
    for parent in list(backbone.modules()):
        for name, child in list(parent.named_children()):
            if name in names and isinstance(child, (nn.Linear, Conv1D)):
                setattr(parent, name, LoRMAPlusLinear(child, r=r, alpha=alpha))
                n += 1
    if n == 0:
        raise RuntimeError(f"LoRMA+ matched 0 modules for kind={backbone.kind}, targets={names}")
    return n


# ---------------- BitFit (works on bias-free architectures too) ----------------
def enable_bitfit(backbone):
    """Train only bias terms.

    The Phase-1 decoder / T5 have NO bias parameters, so (as in BitFit-for-bias-free-models) zero-initialised
    bias vectors are added to every Linear (function unchanged at init) and only those are trained.

    Args:
        backbone (Backbone): Backbone modified in place.

    Returns:
        Tuple[int, int]: ``(n_existing_bias_params, n_added_bias_params)``.
    """
    for p in backbone.parameters():
        p.requires_grad = False
    n_exist = n_added = 0
    for name, p in backbone.named_parameters():
        if name.endswith(".bias"):
            p.requires_grad = True
            p._is_peft = True
            n_exist += p.numel()
    for name, m in list(backbone.named_modules()):
        if isinstance(m, nn.Linear) and m.bias is None and "lm_head" not in name:
            m.bias = nn.Parameter(torch.zeros(m.out_features, dtype=m.weight.dtype, device=m.weight.device))
            m.bias._is_peft = True
            n_added += m.out_features
    return n_exist, n_added


# ---------------- Adapters (bottleneck, after each FFN) ----------------
class Bottleneck(nn.Module):
    """Residual bottleneck adapter ``x + up(gelu(down(x)))`` with zero-initialised up-projection.

    Args:
        d (int): Hidden size.
        r (int): Bottleneck size.
    """

    def __init__(self, d, r):
        super().__init__()
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        """Apply the adapter.

        Args:
            x (torch.Tensor): Input (..., d).

        Returns:
            torch.Tensor: Output (..., d).
        """
        return x + self.up(F.gelu(self.down(x)))


class AdapterWrap(nn.Module):
    """Wrap a sub-module and apply a :class:`Bottleneck` adapter to its output.

    Args:
        inner (nn.Module): Module to wrap (e.g. the FFN).
        d (int): Hidden size.
        r (int): Adapter bottleneck size.
    """

    def __init__(self, inner, d, r):
        super().__init__()
        self.inner = inner
        self.adapter = Bottleneck(d, r)
        mark_peft(self.adapter)

    def forward(self, *a, **k):
        """Run the wrapped module then the adapter.

        Args:
            *a: Positional arguments for the wrapped module.
            **k: Keyword arguments for the wrapped module.

        Returns:
            torch.Tensor: Adapter output.
        """
        return self.adapter(self.inner(*a, **k))


def inject_adapters(backbone, r=64):
    """Insert a bottleneck adapter after the FFN of every layer.

    Args:
        backbone (Backbone): Backbone modified in place.
        r (int): Adapter bottleneck size.

    Returns:
        int: Number of layers that received an adapter.
    """
    d, n = backbone.hidden_dim, 0
    for layer in backbone.get_layers():
        if backbone.kind == "custom":
            layer.ffn = AdapterWrap(layer.ffn, d, r)
        elif backbone.kind == "gpt2":
            layer.mlp = AdapterWrap(layer.mlp, d, r)
        else:
            layer.layer[-1].DenseReluDense = AdapterWrap(layer.layer[-1].DenseReluDense, d, r)
        n += 1
    return n


# ---------------- layer freezing / unfreezing / pruning ----------------
def set_trainable_top_k(backbone, k):
    """Freeze everything except the top-k layers and the final norm.

    Args:
        backbone (Backbone): Backbone modified in place.
        k (int): Number of top layers left trainable.

    Returns:
        None
    """
    for p in backbone.parameters():
        p.requires_grad = False
    layers = backbone.get_layers()
    for l in layers[max(0, len(layers) - k):]:
        for p in l.parameters():
            p.requires_grad = True
    for p in backbone.get_final_norm().parameters():
        p.requires_grad = True


def layer_depth_map(backbone):
    """Map each parameter id to its depth (for layer-wise LR decay).

    Args:
        backbone (Backbone): Backbone to inspect.

    Returns:
        Tuple[dict, int]: ``(id(param) -> depth, max_depth)`` where 0 = embeddings,
        1..L = layers, L+1 = final norm.
    """
    L = len(backbone.get_layers())
    m = {}
    for p in backbone.parameters():
        m[id(p)] = 0
    for i, l in enumerate(backbone.get_layers()):
        for p in l.parameters():
            m[id(p)] = i + 1
    for p in backbone.get_final_norm().parameters():
        m[id(p)] = L + 1
    return m, L + 1


def prune_layers(backbone, keep_every=2):
    """KD student = Phase-1 model with every ``keep_every``-th layer kept (weights inherited from Phase 1).

    Args:
        backbone (Backbone): Backbone modified in place.
        keep_every (int): Keep layers whose index is a multiple of this value.

    Returns:
        Tuple[int, int]: ``(layers_before, layers_after)``.
    """
    layers = list(backbone.get_layers())
    keep = [l for i, l in enumerate(layers) if i % keep_every == 0]
    backbone.set_layers(keep)
    return len(layers), len(keep)


def count_params(model):
    """Count trainable and total parameters.

    Args:
        model (nn.Module): Model to inspect.

    Returns:
        Tuple[int, int]: ``(trainable, total)``.
    """
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return t, sum(p.numel() for p in model.parameters())
