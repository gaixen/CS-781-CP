"""Phase-1 encoder -> chunk vectors -> label-wise attention -> statute logits, plus losses and optimiser groups.

The bias is initialised to the log-prior of each statute so training starts calibrated.
Losses: ``bce``, ``bcepw`` (sqrt-balanced positive weights), ``focal``, ``asl`` (asymmetric loss); KD uses per-label sigmoid KD.
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from lsi.models.custom_decoder import build_attn_mask
from lsi.models.phase1_backbone import build_backbone, inject_adapters, inject_lorma, layer_depth_map, prune_layers, set_trainable_top_k


def autocast(cfg, device):
    """Autocast context (fp16 on CUDA when AMP is on).

    Args:
        cfg (Config): Configuration (``amp``).
        device (torch.device): Device.

    Returns:
        torch.autocast: Context manager.
    """
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(cfg.amp and device.type == "cuda"))


def pool_hidden(h, mask, mode):
    """Pool token states of each chunk.

    Args:
        h (torch.Tensor): Hidden states (n, T, D).
        mask (torch.Tensor): Token mask (n, T).
        mode (str): ``first | last | mean | mean_last``.

    Returns:
        torch.Tensor: fp32 chunk vectors (n, D) or (n, 2D) for ``mean_last``.
    """
    hf, m = h.float(), mask.bool()
    n = m.sum(1).clamp(min=1)
    last = hf[torch.arange(hf.size(0), device=hf.device), n - 1]
    if mode == "first":
        return hf[:, 0]
    if mode == "last":
        return last
    mean = (hf * m[..., None]).sum(1) / n[:, None]
    return mean if mode == "mean" else torch.cat([mean, last], -1)


def bb_hidden(bb, ids, mask, ckpt):
    """Backbone forward with optional per-block gradient checkpointing.

    Args:
        bb (Backbone): Backbone.
        ids (torch.Tensor): Token ids (n, T).
        mask (torch.Tensor): Token mask (n, T).
        ckpt (bool): Use gradient checkpointing (custom backbones, grad mode only).

    Returns:
        torch.Tensor: Hidden states (n, T, D).
    """
    if bb.kind != "custom" or not ckpt or not torch.is_grad_enabled():
        return bb(ids, mask.long())
    m = bb.model
    T = ids.shape[1]
    x = m.tok_emb(ids)
    am = build_attn_mask(T, x.device, m.config.window_size, getattr(m.config, "bidirectional", False), mask)
    for blk in m.blocks:
        x = torch_ckpt.checkpoint(blk, x, m.rope_cos[:T], m.rope_sin[:T], am, use_reentrant=False)
    return m.norm_f(x)


class LSIModel(nn.Module):
    """Chunk encoder + label-wise attention network (LWAN) or mean/max-pooling linear head.

    Args:
        bb (Backbone): Phase-1 backbone.
        n_labels (int): Number of statutes.
        cfg (Config): Configuration (for AMP).
        device (torch.device): Device.
        head (str): ``lwan`` or ``linear``.
        pool (str): Chunk pooling mode.
        d (int): Head width.
        drop (float): Head dropout.
        prior (Optional[np.ndarray]): Label priors for the bias initialisation.
        ckpt (bool): Gradient checkpointing.
    """

    def __init__(self, bb, n_labels, cfg, device, head="lwan", pool="mean_last", d=256, drop=0.1, prior=None, ckpt=False):
        super().__init__()
        self.bb, self.pool, self.ckpt, self.kind, self.cfg, self.device = bb, pool, ckpt, head, cfg, device
        d_in = bb.hidden_dim * (2 if pool == "mean_last" else 1)
        self.norm, self.proj, self.drop = nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.Dropout(drop)
        if head == "lwan":
            self.Q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
            self.W = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        else:
            self.fc = nn.Linear(2 * d, n_labels, bias=False)
        pr = None if prior is None else torch.as_tensor(np.clip(prior, 1e-4, 0.9), dtype=torch.float32)
        self.b = nn.Parameter(torch.zeros(n_labels) if pr is None else torch.log(pr / (1 - pr)))

    def forward(self, ids, mask, owner, B):
        """Statute logits of ``B`` documents given their chunks.

        Args:
            ids (torch.Tensor): Chunk token ids (n_chunks, T).
            mask (torch.Tensor): Chunk masks (n_chunks, T).
            owner (np.ndarray): Document position of each chunk.
            B (int): Number of documents.

        Returns:
            torch.Tensor: Logits (B, n_labels).
        """
        with autocast(self.cfg, self.device):
            h = bb_hidden(self.bb, ids, mask, self.ckpt)
        v = pool_hidden(h, mask, self.pool)
        ow = torch.as_tensor(owner, device=v.device)
        cnt = torch.bincount(ow, minlength=B)
        C = int(cnt.max())
        pos = torch.arange(len(ow), device=v.device) - (torch.cumsum(cnt, 0) - cnt)[ow]
        X = v.new_zeros(B, C, v.size(1))
        X[ow, pos] = v
        cm = torch.arange(C, device=v.device)[None] < cnt[:, None]
        z = self.drop(F.gelu(self.proj(self.norm(X))))
        if self.kind == "lwan":
            a = torch.einsum("bcd,ld->blc", torch.tanh(z), self.Q).masked_fill(~cm[:, None, :], -1e4).softmax(-1)
            return (torch.einsum("blc,bcd->bld", a, z) * self.W[None]).sum(-1) + self.b
        zm = z * cm[..., None]
        mean = zm.sum(1) / cnt[:, None]
        mx = z.masked_fill(~cm[..., None], -1e4).max(1).values
        return self.fc(torch.cat([mean, mx], -1)) + self.b


def elem_loss(kind, z, y, pw=None):
    """Element-wise multi-label loss.

    Args:
        kind (str): ``bce | bcepw | focal | asl``.
        z (torch.Tensor): Logits (B, L).
        y (torch.Tensor): Targets (B, L).
        pw (Optional[torch.Tensor]): Positive weights (``bcepw``).

    Returns:
        torch.Tensor: Loss per element (B, L).
    """
    z = z.float()
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(z, y, reduction="none")
    if kind == "bcepw":
        return F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="none")
    if kind == "focal":
        bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        p = torch.sigmoid(z)
        pt = y * p + (1 - y) * (1 - p)
        return bce * (1 - pt) ** 2
    if kind == "asl":
        p = torch.sigmoid(z)
        pn = (p - 0.05).clamp(min=0)
        return -(y * torch.log(p.clamp(min=1e-8)) + (1 - y) * torch.log((1 - pn).clamp(min=1e-8)) * pn ** 4)
    raise ValueError(kind)


def count_params(m):
    """Total and trainable parameter counts.

    Args:
        m (nn.Module): Model.

    Returns:
        Tuple[int, int]: ``(total, trainable)``.
    """
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(ctx, exp_cfg):
    """Build backbone + adaptation technique + LSI head for an experiment.

    Args:
        ctx (Context): Pipeline context (config, device, label priors).
        exp_cfg (dict): Experiment definition.

    Returns:
        LSIModel: Model on the device.
    """
    cfg, arch = ctx.cfg, exp_cfg["arch"]
    bb = build_backbone(arch, cfg.phase1_ckpt[arch], cfg.strict_load)
    m = exp_cfg["method"]
    if m == "kd":
        n0, n1 = prune_layers(bb, exp_cfg.get("keep_every", 2))
        print(f"  student: {n0} -> {n1} layers")
    if m in ("full", "kd"):
        for p in bb.parameters():
            p.requires_grad = True
    elif m == "partial":
        set_trainable_top_k(bb, exp_cfg["top_k"])
    elif m == "lorma":
        for p in bb.parameters():
            p.requires_grad = False
        print(f"  LoRMA+ injected into {inject_lorma(bb, exp_cfg['lorma_which'], exp_cfg['lorma_r'], exp_cfg['lorma_alpha'])} modules")
    elif m == "adapters":
        for p in bb.parameters():
            p.requires_grad = False
        print(f"  adapters injected into {inject_adapters(bb, exp_cfg['adapter_r'])} layers")
    else:
        raise ValueError(m)
    model = LSIModel(bb, ctx.data.n_labels, cfg, ctx.device, exp_cfg["head"], exp_cfg["pool"], exp_cfg["head_dim"], cfg.head_dropout, ctx.data.ytr.mean(0), cfg.grad_ckpt).to(ctx.device)
    tot, tr = count_params(model)
    print(f"  params total {tot/1e6:.1f}M | trainable {tr/1e6:.2f}M ({100*tr/tot:.1f}%)")
    return model


def param_groups(model, exp_cfg, ft_lr_head):
    """Optimiser groups: head at ``ft_lr_head``; backbone at ``lr`` (optionally layer-wise LR decay).

    Args:
        model (LSIModel): Model.
        exp_cfg (dict): Experiment definition (``lr``, ``llrd``, ``method``).
        ft_lr_head (float): Head learning rate.

    Returns:
        List[dict]: AdamW parameter groups.
    """
    head = [p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad]
    groups = [dict(params=head, lr=ft_lr_head)]
    bbp = [p for p in model.bb.parameters() if p.requires_grad]
    if exp_cfg.get("llrd") and exp_cfg["method"] in ("full", "kd", "partial"):
        dm, top = layer_depth_map(model.bb)
        by = defaultdict(list)
        for p in bbp:
            by[dm[id(p)]].append(p)
        groups += [dict(params=ps, lr=exp_cfg["lr"] * exp_cfg["llrd"] ** (top - d)) for d, ps in by.items()]
    else:
        groups.append(dict(params=bbp, lr=exp_cfg["lr"]))
    return groups
