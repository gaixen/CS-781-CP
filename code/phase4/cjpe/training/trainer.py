"""Parameter groups, loss (weighted CE + logit/feature KD) and the training loop."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import SEED
from ..context import device
from ..data.batching import make_batches, collate, to_device
from ..eval.predict import predict
from ..models.peft import layer_depth_map, set_trainable_top_k
from ..utils import metrics, best_threshold, class_weights


def build_param_groups(model, cfg):
    """Build optimiser groups with 3 LR tiers.

    Tiers: task head (``LR_HEAD``) | new PEFT params / biases / adapters (``LR_PEFT``) | original Phase-1 weights
    (``LR_BACKBONE``, optional layer-wise decay when ``LLRD_DECAY < 1`` for ``full``/``distill``).

    Args:
        model (CJPEModel): model whose parameters are grouped.
        cfg (dict): experiment config.

    Returns:
        list[dict]: AdamW parameter groups (``params, lr, weight_decay``).
    """
    llrd = cfg["LLRD_DECAY"] < 1 and cfg["method"] in ("full", "distill")
    dmap, maxd = layer_depth_map(model.backbone) if llrd else ({}, 0)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad and not (cfg["method"] == "gradual" and n.startswith("backbone.")): continue
        if n.startswith("head."): lr = cfg["LR_HEAD"]
        elif getattr(p, "_is_peft", False): lr = cfg["LR_PEFT"]
        else: lr = cfg["LR_BACKBONE"] * (cfg["LLRD_DECAY"] ** (maxd - dmap.get(id(p), maxd)) if llrd else 1.0)
        wd = 0.0 if (p.ndim <= 1 or getattr(p, "_is_peft", False)) else cfg["WEIGHT_DECAY"]
        groups.setdefault((round(lr, 12), wd), []).append(p)
    return [dict(params=ps, lr=lr, weight_decay=wd) for (lr, wd), ps in groups.items()]


def compute_loss(model, batch, cw, cfg, kd):
    """Class-weighted cross-entropy, optionally combined with knowledge distillation.

    KD loss = ``(1-a) * CE + a * T^2 * KL(student/T || teacher/T) + beta * (1 - cos(proj(student_feat), teacher_feat))``.

    Args:
        model (CJPEModel): the (student) model.
        batch (dict): collated batch on device (includes ``pos`` for teacher alignment).
        cw (torch.Tensor): class weights (2,).
        cfg (dict): config (``LABEL_SMOOTH, KD_T, KD_ALPHA, KD_BETA``).
        kd (dict or None): cached teacher outputs ``{"logits", "feats", ...}`` or None for plain training.

    Returns:
        torch.Tensor: scalar loss.
    """
    logits, feats = model(batch)
    ce = F.cross_entropy(logits, batch["labels"], weight=cw, label_smoothing=cfg["LABEL_SMOOTH"])
    if kd is None: return ce
    T, a = cfg["KD_T"], cfg["KD_ALPHA"]; pos = batch["pos"]
    t_logits = torch.from_numpy(kd["logits"][pos]).to(device).float()
    soft = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t_logits / T, -1), reduction="batchmean") * T * T
    loss = (1 - a) * ce + a * soft
    if kd["feats"] is not None and model.head.feat_proj is not None and cfg["KD_BETA"] > 0:     # intermediate-representation distillation
        tf = torch.from_numpy(kd["feats"][pos]).to(device).float()
        loss = loss + cfg["KD_BETA"] * (1 - F.cosine_similarity(model.head.feat_proj(feats), tf, dim=-1)).mean()
    return loss


def train_model(model, tr, dv, cfg, best_path, kd=None):
    """Train with AdamW + warmup/cosine schedule, AMP, grad accumulation; keep the best dev checkpoint.

    Args:
        model (CJPEModel): model to train.
        tr (list[dict]): training examples.
        dv (list[dict]): dev examples (model selection; threshold-tuned macro-F1 if ``CALIBRATE``).
        cfg (dict): experiment config.
        best_path (str): where the best ``state_dict`` is saved.
        kd (dict or None): teacher outputs for distillation.

    Returns:
        list[dict]: per-epoch history (``epoch, train_loss, dev_f1_raw, dev_f1_tuned, thr``); the best weights are reloaded into ``model``.
    """
    opt = torch.optim.AdamW(build_param_groups(model, cfg), betas=(0.9, 0.999))
    nb = math.ceil(math.ceil(len(tr) / cfg["BATCH_SIZE"]) / cfg["GRAD_ACCUM"]); total = nb * cfg["EPOCHS"]; warm = max(1, int(cfg["WARMUP_FRAC"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["_amp"] and device.type == "cuda")
    cw = class_weights(np.array([e["label"] for e in tr]), cfg); dv_y = np.array([e["label"] for e in dv])
    rng = np.random.RandomState(SEED); best, hist = -1, []
    nlayers = len(model.backbone.get_layers())
    for ep in range(cfg["EPOCHS"]):
        if cfg["method"] == "gradual":                                    # unfreeze one more layer (from the top) each epoch
            set_trainable_top_k(model.backbone, min(ep + 1, nlayers)); print(f"  gradual: top {min(ep+1, nlayers)}/{nlayers} layers trainable")
        model.train(); t0 = time.time(); losses = []; batches = make_batches(tr, cfg["BATCH_SIZE"], True, rng)
        for bi, idxs in enumerate(batches):
            batch = to_device(collate([tr[i] for i in idxs], cfg))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["_amp"]):
                loss = compute_loss(model, batch, cw, cfg, kd)
            scaler.scale(loss / cfg["GRAD_ACCUM"]).backward(); losses.append(loss.item())
            if (bi + 1) % cfg["GRAD_ACCUM"] == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], cfg["CLIP"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if (bi + 1) % 500 == 0: print(f"    ep{ep+1} batch {bi+1}/{len(batches)} loss={np.mean(losses[-500:]):.4f} ({time.time()-t0:.0f}s)")
        out = predict(model, dv, cfg); raw = metrics(dv_y, (out["prob"] >= 0.5).astype(int))
        thr, tuned = best_threshold(dv_y, out["prob"]); sel = tuned if cfg["CALIBRATE"] else raw["macro_f1"]
        hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(losses)), dev_f1_raw=raw["macro_f1"], dev_f1_tuned=tuned, thr=thr))
        print(f"ep{ep+1}/{cfg['EPOCHS']} loss={np.mean(losses):.4f} dev macroF1@0.5={raw['macro_f1']:.4f} tuned={tuned:.4f}@{thr:.2f} "
              f"(f1_c0={raw['f1_c0']:.3f} f1_c1={raw['f1_c1']:.3f}) [{(time.time()-t0)/60:.1f} min]")
        if sel > best: best = sel; torch.save(model.state_dict(), best_path); print("   saved best ->", best_path)
    model.load_state_dict(torch.load(best_path, map_location=device)); return hist
