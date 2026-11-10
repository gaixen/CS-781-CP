"""Parameter-efficient adaptation (LoRA / rsLoRA / LoRMA / adapters), BitFit, partial and full fine-tuning."""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from rr.config import ARCH_INFO


class PEFTWrapper(nn.Module):
    """Wrap an ``nn.Linear`` / HF ``Conv1D`` with a trainable low-rank or adapter branch.

    Modes: ``lora`` (``y = W0 x + s B(A x)``), ``lorma`` (``y = y0 + s P(Q y0)``, multiplicative), ``adapter`` (``y = y0 + Up(GELU(Down(y0)))``).

    Args:
        base (nn.Module): Frozen base layer.
        mode (str): ``lora | lorma | adapter``.
        r (int): Rank.
        alpha (float): Scaling numerator.
        dropout (float): LoRA dropout.
        bottleneck (int): Adapter bottleneck size.
    """

    def __init__(self, base, mode, r=16, alpha=32, dropout=0.0, bottleneck=64):
        super().__init__()
        self.base, self.mode, self.scale = base, mode, alpha / r
        for p in base.parameters():
            p.requires_grad = False
        d_in, d_out = (base.in_features, base.out_features) if hasattr(base, "in_features") else (base.weight.shape[0], base.nf)
        if mode == "lora":
            self.A = nn.Parameter(torch.empty(r, d_in))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = nn.Parameter(torch.zeros(d_out, r))
            self.drop = nn.Dropout(dropout)
        elif mode == "lorma":
            self.Q = nn.Parameter(torch.empty(r, d_out))
            nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5))
            self.P = nn.Parameter(torch.zeros(d_out, r))
        elif mode == "adapter":
            self.down, self.up, self.act = nn.Linear(d_out, bottleneck), nn.Linear(bottleneck, d_out), nn.GELU()
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    @property
    def weight(self):
        """Weight of the wrapped layer (some HF modules read it).

        Args:
            None

        Returns:
            torch.Tensor: ``base.weight``.
        """
        return self.base.weight

    def forward(self, x):
        """Wrapped layer plus the trainable branch.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        y = self.base(x)
        if self.mode == "lora":
            return y + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)
        if self.mode == "lorma":
            return y + self.scale * F.linear(F.linear(y, self.Q), self.P)
        return y + self.up(self.act(self.down(y)))


DEFAULT_TARGETS = {
    ("custom", "lora"): ("q_proj", "k_proj", "v_proj", "o_proj"), ("custom", "adapter"): ("o_proj", "w2"),
    ("gpt2", "lora"): ("c_attn", "c_proj"), ("gpt2", "adapter"): ("c_proj",),
    ("t5", "lora"): ("q", "k", "v", "o"), ("t5", "adapter"): ("o", "wo"),
}
_ALL_T = {"custom": ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"), "gpt2": ("c_attn", "c_proj", "c_fc"), "t5": ("q", "k", "v", "o", "wi", "wo")}


def inject_peft(model, mode, targets, cfg):
    """Replace every target linear layer by a :class:`PEFTWrapper`.

    Args:
        model (nn.Module): Module to modify in place.
        mode (str): ``lora | lorma | adapter``.
        targets (Tuple[str, ...]): Child-module names to wrap.
        cfg (dict): Config (``R, ALPHA, LORA_DROPOUT, BOTTLENECK``).

    Returns:
        int: Number of wrapped layers.
    """
    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, (nn.Linear, Conv1D)):
                setattr(parent, name, PEFTWrapper(child, mode, cfg["R"], cfg["ALPHA"], cfg["LORA_DROPOUT"], cfg["BOTTLENECK"]))
                n += 1
    return n


def amp_settings(arch, device):
    """Mixed-precision settings (T5 overflows in fp16 -> bf16 or fp32).

    Args:
        arch (str): Architecture key.
        device (torch.device): Device.

    Returns:
        Tuple[bool, torch.dtype]: ``(use_amp, dtype)``.
    """
    if device.type != "cuda":
        return False, torch.float32
    bf16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
    if ARCH_INFO[arch]["kind"] == "t5":
        return bf16, torch.bfloat16
    return True, (torch.bfloat16 if bf16 else torch.float16)


def configure_trainable(model, cfg):
    """Freeze / unfreeze the backbone according to ``cfg['METHOD']`` (architecture-aware PEFT targets).

    Args:
        model (nn.Module): ``RRModel`` (needs ``model.backbone``).
        cfg (dict): Config (``METHOD, ARCH, PEFT_TARGETS, PARTIAL_LAYERS, ...``).

    Returns:
        None
    """
    method, bb = cfg["METHOD"], model.backbone
    for p in bb.parameters():
        p.requires_grad = False
    if method == "full":
        for p in bb.parameters():
            p.requires_grad = True
    elif method == "partial":
        first = bb.num_layers() - cfg["PARTIAL_LAYERS"]
        for n, p in bb.named_parameters():
            m = re.search(r"(?:^|\.)(?:blocks|h|block)\.(\d+)\.", n)
            if (m and int(m.group(1)) >= first) or re.search(r"norm_f|ln_f|final_layer_norm", n):
                p.requires_grad = True
    elif method == "bitfit":
        for n, p in bb.named_parameters():
            if n.endswith("bias") or re.search(r"norm|ln_", n):
                p.requires_grad = True
    elif method in ("lora", "rslora", "lorma", "adapter"):
        kind = ARCH_INFO[cfg["ARCH"]]["kind"]
        targets = cfg["PEFT_TARGETS"] or (DEFAULT_TARGETS[(kind, "adapter")] if method == "adapter" else _ALL_T[kind])
        c2, mode = dict(cfg), method
        if method == "rslora":
            mode, c2["ALPHA"] = "lora", cfg["ALPHA_RS"] * math.sqrt(cfg["R"])       # rsLoRA: scale = alpha / sqrt(r)
        n = inject_peft(bb, mode, targets, c2)
        assert n > 0, f"no layers matched PEFT targets {targets}"
        print(f"{method}: wrapped {n} layers {targets}")
    else:
        raise ValueError(method)


def count_params(model):
    """Total and trainable parameter counts.

    Args:
        model (nn.Module): Model.

    Returns:
        Tuple[int, int]: ``(total, trainable)``.
    """
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
