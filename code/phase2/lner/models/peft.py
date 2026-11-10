"""Adaptation techniques (hand-written, no ``peft`` dependency): LoRA, LoRMA (multiplicative), adapters, BitFit, partial, full.

LoRMA here is the multiplicative update of the LoRMA paper (LoRMA+ / additive-inflation form):
``W = (I + (alpha/r) P Q) W0``, implemented as ``y = y0 + (alpha/r) P(Q(y0))`` with ``y0 = W0 x``; ``P`` is zero-initialised.
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..config import ARCH_INFO


class PEFTWrapper(nn.Module):
    """Wrap an ``nn.Linear`` / HF ``Conv1D`` with a parameter-efficient module.

    Args:
        base (nn.Module): frozen base layer.
        mode (str): ``lora`` | ``lorma`` | ``adapter``.
        r (int): rank.
        alpha (float): scaling numerator (scale = alpha / r).
        dropout (float): LoRA dropout.
        bottleneck (int): adapter bottleneck width.
    """
    def __init__(self, base, mode, r=16, alpha=32, dropout=0.0, bottleneck=64):
        super().__init__()
        self.base, self.mode, self.scale = base, mode, alpha / r
        for p in base.parameters(): p.requires_grad = False
        d_in, d_out = (base.in_features, base.out_features) if hasattr(base, "in_features") else (base.weight.shape[0], base.nf)
        if mode == "lora":          # y = W0 x + s * B(A x)
            self.A = nn.Parameter(torch.empty(r, d_in)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = nn.Parameter(torch.zeros(d_out, r)); self.drop = nn.Dropout(dropout)
        elif mode == "lorma":       # W = (I + s * P Q) W0  ->  y = y0 + s * P(Q y0)
            self.Q = nn.Parameter(torch.empty(r, d_out)); nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5))
            self.P = nn.Parameter(torch.zeros(d_out, r))
        elif mode == "adapter":     # y = y0 + Up(GELU(Down(y0)))
            self.down, self.up, self.act = nn.Linear(d_out, bottleneck), nn.Linear(bottleneck, d_out), nn.GELU()
            nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    @property
    def weight(self):
        """Weight of the wrapped layer (some HF modules read ``.weight`` of sub-layers).

        Args:
            None

        Returns:
            torch.Tensor: base weight.
        """
        return self.base.weight

    def forward(self, x):
        """Apply the base layer plus the PEFT update.

        Args:
            x (torch.Tensor): (..., d_in).

        Returns:
            torch.Tensor: (..., d_out).
        """
        y = self.base(x)
        if self.mode == "lora":    return y + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)
        if self.mode == "lorma":   return y + self.scale * F.linear(F.linear(y, self.Q), self.P)
        return y + self.up(self.act(self.down(y)))


DEFAULT_TARGETS = {
    ("custom", "lora"): ("q_proj", "k_proj", "v_proj", "o_proj"), ("custom", "adapter"): ("o_proj", "w2"),
    ("gpt2", "lora"): ("c_attn", "c_proj"),                       ("gpt2", "adapter"): ("c_proj",),
    ("t5", "lora"): ("q", "k", "v", "o"),                         ("t5", "adapter"): ("o", "wo"),
}


def inject_peft(model, mode, targets, cfg):
    """Replace matching ``Linear``/``Conv1D`` children by ``PEFTWrapper``.

    Args:
        model (nn.Module): module to modify in place.
        mode (str): ``lora`` | ``lorma`` | ``adapter``.
        targets (tuple[str]): child module names to wrap.
        cfg (dict): config (``R, ALPHA, LORA_DROPOUT, BOTTLENECK``).

    Returns:
        int: number of wrapped layers.
    """
    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, (nn.Linear, Conv1D)):
                setattr(parent, name, PEFTWrapper(child, mode, cfg["R"], cfg["ALPHA"], cfg["LORA_DROPOUT"], cfg["BOTTLENECK"])); n += 1
    return n


def configure_trainable(model, cfg):
    """Freeze/unfreeze backbone parameters according to ``cfg["METHOD"]`` (head stays trainable).

    Args:
        model (nn.Module): NER model with ``.backbone``.
        cfg (dict): config (``METHOD, PARTIAL_LAYERS, PEFT_TARGETS, ARCH``).

    Returns:
        None

    Raises:
        ValueError: for an unknown method.
    """
    method, bb = cfg["METHOD"], model.backbone
    for p in bb.parameters(): p.requires_grad = False
    if method == "full":
        for p in bb.parameters(): p.requires_grad = True
    elif method == "partial":
        first = bb.num_layers() - cfg["PARTIAL_LAYERS"]
        for n, p in bb.named_parameters():
            m = re.search(r"(?:^|\.)(?:blocks|h|block)\.(\d+)\.", n)
            if (m and int(m.group(1)) >= first) or re.search(r"norm_f|ln_f|final_layer_norm", n): p.requires_grad = True
    elif method == "bitfit":
        for n, p in bb.named_parameters():
            if n.endswith("bias") or re.search(r"norm|ln_", n): p.requires_grad = True
    elif method in ("lora", "lorma", "adapter"):
        kind = ARCH_INFO[cfg["ARCH"]]["kind"]
        targets = cfg["PEFT_TARGETS"] or DEFAULT_TARGETS[(kind, "adapter" if method == "adapter" else "lora")]
        n = inject_peft(bb, method, targets, cfg); assert n > 0, f"no layers matched PEFT targets {targets}"
        print(f"{method}: wrapped {n} layers {targets}")
    else:
        raise ValueError(method)


def count_params(model):
    """Total and trainable parameter counts.

    Args:
        model (nn.Module): any module.

    Returns:
        tuple[int, int]: (total, trainable).
    """
    tot = sum(p.numel() for p in model.parameters()); tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, tr
