"""Adaptation techniques: LoRMA+ (compulsory), BitFit, bottleneck Adapters, partial/gradual unfreezing, layer pruning (KD student)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

# module names that LoRMA+ wraps, per backbone kind
LORMA_TARGETS = {
    "custom": {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "ffn": ("w1", "w2", "w3")},
    "gpt2":   {"qkv": ("c_attn",),                    "o": (),          "ffn": ("c_fc",)},
    "t5":     {"qkv": ("q", "k", "v"),                "o": ("o",),       "ffn": ("wi", "wi_0", "wi_1", "wo")},
}


def mark_peft(module):
    """Tag every parameter of ``module`` as a new PEFT parameter (gets ``LR_PEFT`` and no weight decay).

    Args:
        module (nn.Module): module whose parameters are tagged.

    Returns:
        None
    """
    for p in module.parameters(): p._is_peft = True


# ---------------- LoRMA+ (compulsory) ----------------
class LoRMAPlusLinear(nn.Module):
    """LoRMA+ (Bihany, Patel & Modi, ACL Findings 2025, arXiv:2506.07621).

    Computes ``h = ((alpha/r) B A + I)(W0 x)``; ``B`` is zero-initialised so the layer starts as identity.

    Args:
        base (nn.Linear or Conv1D): the frozen pre-trained layer to wrap.
        r (int): rank of the multiplicative update.
        alpha (float): scaling numerator (scale = alpha / r).
    """
    def __init__(self, base, r=16, alpha=16):
        super().__init__(); self.base = base
        for p in base.parameters(): p.requires_grad = False
        out_f = base.out_features if isinstance(base, nn.Linear) else base.nf     # Conv1D (GPT-2) has .nf
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, out_f) * (1.0 / math.sqrt(out_f)))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        mark_peft(self)
        for p in base.parameters(): p._is_peft = False

    def forward(self, x):
        """Apply the frozen layer followed by the multiplicative low-rank update.

        Args:
            x (torch.Tensor): (..., in_features) input.

        Returns:
            torch.Tensor: (..., out_features) output ``y + scale * (y A^T) B^T``.
        """
        y = self.base(x)
        return y + self.scale * ((y @ self.A.T.to(y.dtype)) @ self.B.T.to(y.dtype))


def inject_lorma(backbone, which="qkv", r=16, alpha=16):
    """Wrap target linear layers of the backbone with ``LoRMAPlusLinear``.

    Args:
        backbone (Backbone): backbone wrapper (uses ``backbone.kind``).
        which (str): ``"qkv"`` | ``"attn"`` (qkv + output proj) | ``"attn_ffn"`` (attn + FFN).
        r (int): LoRMA+ rank.
        alpha (float): LoRMA+ alpha.

    Returns:
        int: number of wrapped modules.

    Raises:
        RuntimeError: if no module matched the target names.
    """
    t = LORMA_TARGETS[backbone.kind]
    names = {"qkv": t["qkv"], "attn": t["qkv"] + t["o"], "attn_ffn": t["qkv"] + t["o"] + t["ffn"]}[which]
    n = 0
    for parent in list(backbone.modules()):
        for name, child in list(parent.named_children()):
            if name in names and isinstance(child, (nn.Linear, Conv1D)):
                setattr(parent, name, LoRMAPlusLinear(child, r=r, alpha=alpha)); n += 1
    if n == 0: raise RuntimeError(f"LoRMA+ matched 0 modules for kind={backbone.kind}, targets={names}")
    return n


# ---------------- BitFit (works on bias-free architectures too) ----------------
def enable_bitfit(backbone):
    """Train only bias terms.

    The Phase-1 decoder/T5 have NO bias parameters, so (as in BitFit-for-bias-free-models) zero-initialised
    bias vectors are added to every Linear (function unchanged at init) and only those are trained.

    Args:
        backbone (Backbone): backbone wrapper; all its parameters are frozen first.

    Returns:
        tuple[int, int]: (number of pre-existing bias params made trainable, number of newly added bias params).
    """
    for p in backbone.parameters(): p.requires_grad = False
    n_exist = n_added = 0
    for name, p in backbone.named_parameters():
        if name.endswith(".bias"): p.requires_grad = True; p._is_peft = True; n_exist += p.numel()
    for name, m in list(backbone.named_modules()):
        if isinstance(m, nn.Linear) and m.bias is None and "lm_head" not in name:
            m.bias = nn.Parameter(torch.zeros(m.out_features, dtype=m.weight.dtype, device=m.weight.device))
            m.bias._is_peft = True; n_added += m.out_features
    return n_exist, n_added


# ---------------- Adapters (bottleneck, after each FFN) ----------------
class Bottleneck(nn.Module):
    """Residual bottleneck adapter ``x + up(gelu(down(x)))`` with zero-initialised up-projection.

    Args:
        d (int): model dimension.
        r (int): bottleneck width.
    """
    def __init__(self, d, r):
        super().__init__(); self.down = nn.Linear(d, r); self.up = nn.Linear(r, d)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        """Apply the adapter.

        Args:
            x (torch.Tensor): (..., d) input.

        Returns:
            torch.Tensor: (..., d) output.
        """
        return x + self.up(F.gelu(self.down(x)))


class AdapterWrap(nn.Module):
    """Wrap an FFN module so that its output passes through a bottleneck adapter.

    Args:
        inner (nn.Module): the original FFN.
        d (int): model dimension.
        r (int): bottleneck width.
    """
    def __init__(self, inner, d, r):
        super().__init__(); self.inner = inner; self.adapter = Bottleneck(d, r); mark_peft(self.adapter)

    def forward(self, *a, **k):
        """Run the inner FFN and then the adapter.

        Args:
            *a: positional args forwarded to the inner FFN.
            **k: keyword args forwarded to the inner FFN.

        Returns:
            torch.Tensor: adapted FFN output.
        """
        return self.adapter(self.inner(*a, **k))


def inject_adapters(backbone, r=64):
    """Insert a bottleneck adapter after the FFN of every layer.

    Args:
        backbone (Backbone): backbone wrapper.
        r (int): bottleneck width.

    Returns:
        int: number of layers that received an adapter.
    """
    d, n = backbone.hidden_dim, 0
    for layer in backbone.get_layers():
        if backbone.kind == "custom": layer.ffn = AdapterWrap(layer.ffn, d, r)
        elif backbone.kind == "gpt2": layer.mlp = AdapterWrap(layer.mlp, d, r)
        else: layer.layer[-1].DenseReluDense = AdapterWrap(layer.layer[-1].DenseReluDense, d, r)
        n += 1
    return n


# ---------------- layer freezing / unfreezing / pruning ----------------
def set_trainable_top_k(backbone, k):
    """Freeze everything except the top ``k`` layers and the final norm (partial / gradual unfreezing).

    Args:
        backbone (Backbone): backbone wrapper.
        k (int): number of top layers to leave trainable.

    Returns:
        None
    """
    for p in backbone.parameters(): p.requires_grad = False
    layers = backbone.get_layers()
    for l in layers[max(0, len(layers) - k):]:
        for p in l.parameters(): p.requires_grad = True
    for p in backbone.get_final_norm().parameters(): p.requires_grad = True


def layer_depth_map(backbone):
    """Map every parameter id to its depth, for layer-wise LR decay.

    Args:
        backbone (Backbone): backbone wrapper.

    Returns:
        tuple[dict, int]: (``id(param) -> depth`` with 0 = embeddings, 1..L = layers, L+1 = final norm; max depth L+1).
    """
    L = len(backbone.get_layers()); m = {}
    for p in backbone.parameters(): m[id(p)] = 0
    for i, l in enumerate(backbone.get_layers()):
        for p in l.parameters(): m[id(p)] = i + 1
    for p in backbone.get_final_norm().parameters(): m[id(p)] = L + 1
    return m, L + 1


def prune_layers(backbone, keep_every=2):
    """Build the KD student: keep every ``keep_every``-th layer of the Phase-1 model (weights inherited).

    Args:
        backbone (Backbone): backbone wrapper, modified in place.
        keep_every (int): keep layers with ``index % keep_every == 0``.

    Returns:
        tuple[int, int]: (original layer count, kept layer count).
    """
    layers = list(backbone.get_layers()); keep = [l for i, l in enumerate(layers) if i % keep_every == 0]
    backbone.set_layers(keep); return len(layers), len(keep)


def count_params(model):
    """Count trainable and total parameters.

    Args:
        model (nn.Module): any module.

    Returns:
        tuple[int, int]: (trainable parameters, total parameters).
    """
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return t, sum(p.numel() for p in model.parameters())
