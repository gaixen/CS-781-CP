"""Training loop, checkpoint loading, experiment runner, final fit."""
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import state as S
from ..config import CFG, WORK
from ..context import DEVICE, seed_everything
from ..data.samples import build_train_samples, collate, class_weights, span_pos_weight
from ..decoding.decoders import decode_doc
from ..decoding.thresholds import finalize_decoding
from ..eval.metrics import strict_scores, print_scores, error_analysis
from ..models.factory import build_ner_model
from ..models.peft import count_params
from .inference import amp_settings, evaluate, predict_probs
from .losses import ce_loss, span_loss, kd_loss_fn


def train_model(model, train_samples, val_ids, cfg, teacher=None):
    """Train with AdamW (head / backbone LR groups), warm-up + cosine schedule, AMP and gradient accumulation.

    Loss: weighted CE / focal (BIO), CRF NLL (BIO+CRF) or span loss; with a teacher: ``alpha*loss + (1-alpha)*KD``.
    Keeps the best epoch by validation strict macro-F1 (last epoch if ``val_ids`` is empty).

    Args:
        model (nn.Module): NER model.
        train_samples (list[dict]): output of ``build_train_samples``.
        val_ids (list[str]): validation doc ids ([] for a final fit).
        cfg (dict): config.
        teacher (nn.Module or None): frozen teacher for distillation.

    Returns:
        dict: ``history, best_val_f1, best_epoch, train_seconds, peak_mem_mb, best_state`` (trainable weights are reloaded into ``model``).
    """
    model.to(DEVICE); use_amp, dt = amp_settings(cfg["ARCH"]); task = model.task
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dt == torch.float16 and DEVICE.type == "cuda")
    cw = class_weights(train_samples, cfg).to(DEVICE); pos_w = span_pos_weight(train_samples, cfg) if task == "span" else None
    loader = DataLoader(train_samples, batch_size=cfg["BATCH"], shuffle=True, collate_fn=collate, num_workers=0)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    is_head = lambda n: not n.startswith("backbone.")           # head / trunk / start_head / end_head / type_head
    groups = []
    for head in (True, False):
        for decay in (True, False):
            ps = [p for n, p in named if is_head(n) == head and ((p.ndim > 1) == decay)]
            if ps: groups.append(dict(params=ps, lr=cfg["LR_HEAD"] if head else cfg["LR"][cfg["METHOD"]], weight_decay=cfg["WD"] if decay else 0.0))
    opt = torch.optim.AdamW(groups)
    steps = math.ceil(len(loader) / cfg["ACCUM"]) * cfg["EPOCHS"]; warm = max(1, int(cfg["WARMUP"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))))
    trainable = {n for n, _ in named}; best, best_ep, best_state, hist, step = -1.0, 0, None, [], 0
    snap = lambda: {n: p.detach().cpu().clone() for n, p in model.state_dict().items() if n in trainable}
    if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for ep in range(cfg["EPOCHS"]):
        model.train(); run, nb = 0.0, 0
        for it, (ids, am, lab) in enumerate(loader):
            ids, am, lab = ids.to(DEVICE), am.to(DEVICE), lab.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=dt, enabled=use_amp):
                logits = model(ids, am)
                loss = (model.crf_nll(logits, lab, am) if getattr(model, "use_crf", False) else ce_loss(logits, lab, cw, cfg)) if task == "bio" else span_loss(logits, lab, cw, pos_w, model.n_types, cfg)
                if teacher is not None:
                    with torch.no_grad(): t_logits = teacher(ids, am)
                    loss = cfg["KD_ALPHA"] * loss + (1 - cfg["KD_ALPHA"]) * kd_loss_fn(logits, t_logits, lab, cfg["KD_TEMP"], task, getattr(model, "n_types", None))
            scaler.scale(loss / cfg["ACCUM"]).backward(); run += loss.item(); nb += 1
            if (it + 1) % cfg["ACCUM"] == 0 or it + 1 == len(loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for _, p in named], cfg["CLIP"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            if (it + 1) % cfg["LOG_EVERY"] == 0: print(f"  ep{ep+1} it{it+1}/{len(loader)} loss={run/nb:.4f}")
        if val_ids:
            sc, _, _ = evaluate(model, val_ids, cfg); f1 = sc["macro"]["f1"]
            hist.append(dict(epoch=ep + 1, train_loss=run / max(nb, 1), val_f1=f1, val_p=sc["macro"]["p"], val_r=sc["macro"]["r"]))
            print(f"== epoch {ep+1}: train_loss={run/max(nb,1):.4f} | val strict macro P/R/F1 = {sc['macro']['p']:.4f}/{sc['macro']['r']:.4f}/{f1:.4f}")
            if f1 > best: best, best_ep, best_state = f1, ep + 1, snap()
        else:                                                    # final fit on all docs: no validation, keep the last epoch
            hist.append(dict(epoch=ep + 1, train_loss=run / max(nb, 1))); print(f"== epoch {ep+1}: train_loss={run/max(nb,1):.4f}")
            best_ep, best_state = ep + 1, snap()
    model.load_state_dict(best_state, strict=False)
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else 0.0
    return dict(history=hist, best_val_f1=best if val_ids else float("nan"), best_epoch=best_ep, train_seconds=time.time() - t_start, peak_mem_mb=peak, best_state=best_state)


def ckpt_path(name):
    """Checkpoint path of a run.

    Args:
        name (str): experiment name.

    Returns:
        str: ``<WORK>/<name>.pt``.
    """
    return os.path.join(WORK, f"{name}.pt")


def load_trained(path):
    """Load a trained checkpoint (architecture, task, window sizes and decoder settings come from the stored config).

    Args:
        path (str): checkpoint file written by ``run_experiment``.

    Returns:
        tuple[nn.Module, dict]: (model in eval mode on ``DEVICE``, config).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {**{k: v for k, v in CFG.items() if k != "TASK"}, "TASK": "bio", **ck["cfg"]}      # old checkpoints (no TASK key) are BIO
    model = build_ner_model(cfg); model.load_state_dict(ck["trainable"], strict=False)
    return model.to(DEVICE).eval(), cfg


def cfg_from_ckpt(path):
    """Stored config of a checkpoint restricted to known config keys.

    Args:
        path (str): checkpoint file.

    Returns:
        dict: config overrides reproducing that run.
    """
    return {k: v for k, v in torch.load(path, map_location="cpu", weights_only=False)["cfg"].items() if k in CFG}


def run_experiment(name, teacher=None, keep_model=False, train_ids=None, val_ids=None, **overrides):
    """Train, select the best decoder variant on validation, save the checkpoint and append a result record.

    Args:
        name (str): experiment name (checkpoint ``<WORK>/<name>.pt``).
        teacher (nn.Module or None): frozen teacher for distillation.
        keep_model (bool): return the trained model.
        train_ids (list[str] or None): override the training docs (default ``state.TRAIN_IDS``).
        val_ids (list[str] or None): override validation docs ([] = no validation / final fit).
        **overrides: config overrides (unknown keys are reported and ignored).

    Returns:
        tuple: (model or None, final config, result record dict).
    """
    unknown = [k for k in overrides if k not in CFG]
    if unknown: print(f"!!! WARNING: unknown override keys IGNORED: {unknown}  (config keys are case-sensitive, e.g. LOSS not loss)")
    cfg = {**CFG, **overrides}; seed_everything(cfg.get("SEED", 42))
    tr_ids = S.TRAIN_IDS if train_ids is None else train_ids; va_ids = S.VAL_IDS if val_ids is None else val_ids
    print(f"\n######## {name}: arch={cfg['ARCH']} method={cfg['METHOD']} task={cfg['TASK']} attn={cfg['ATTN_MODE']} "
          f"len={cfg['MAX_LEN']}/{cfg['OVERLAP']} loss={cfg['LOSS']} cw={cfg['CLASS_WEIGHT_MODE']} | train docs={len(tr_ids)} val docs={len(va_ids)} ########")
    model = build_ner_model(cfg); samples = build_train_samples(tr_ids, cfg)
    out = train_model(model, samples, va_ids, cfg, teacher=teacher)
    sc, infer_s, table = None, float("nan"), []
    n_tok = sum(len(S.DOCS[d]["ids"]) for d in va_ids)
    if va_ids:
        t0 = time.time(); probs = predict_probs(model, [S.DOCS[d] for d in va_ids], cfg); infer_s = time.time() - t0   # decoded below with every variant
        dec, table = finalize_decoding(va_ids, probs, cfg); cfg = {**cfg, **dec}
        pred = {d: decode_doc(S.DOCS[d], p, cfg) for d, p in zip(va_ids, probs)}
        sc = strict_scores({d: S.GOLD[d] for d in va_ids}, pred, S.LABELS); print_scores(sc); error_analysis({d: S.GOLD[d] for d in va_ids}, pred)
    tot, tr = count_params(model)
    ckpt = ckpt_path(name); torch.save(dict(trainable=out["best_state"], cfg=cfg, best_epoch=out["best_epoch"]), ckpt)
    nan = float("nan")
    rec = dict(name=name, arch=cfg["ARCH"], method=cfg["METHOD"], task=cfg["TASK"], attn=cfg["ATTN_MODE"], decode=cfg["DECODE"], kd=teacher is not None,
               refine=bool(cfg.get("BOUNDARY_REFINE")), thresholds=cfg.get("CONF_THRESH"), best_epoch=out["best_epoch"], decoder_table=table,
               val_f1=sc["macro"]["f1"] if sc else nan, val_p=sc["macro"]["p"] if sc else nan, val_r=sc["macro"]["r"] if sc else nan,
               total_params_M=tot / 1e6, trainable_params_M=tr / 1e6, train_min=out["train_seconds"] / 60, peak_train_mem_MB=out["peak_mem_mb"],
               infer_tok_per_s=n_tok / infer_s if va_ids else nan, ckpt=ckpt, history=out["history"])
    S.RESULTS.append(rec); json.dump(S.RESULTS, open(os.path.join(WORK, "results.json"), "w"), indent=1)
    if keep_model: return model, cfg, rec
    del model; torch.cuda.empty_cache(); return None, cfg, rec


def run_final_fit(name, ckpt_path_, teacher=None, **overrides):
    """Re-train a validated configuration on ALL labelled docs (fold1+fold2) for its best epoch count.

    Decoder settings (refine / thresholds) chosen on validation are reused.

    Args:
        name (str): new experiment name.
        ckpt_path_ (str): validated checkpoint whose config and best epoch are reused.
        teacher (nn.Module or None): optional teacher.
        **overrides: config overrides (e.g. ``SEED=2``).

    Returns:
        tuple: result of ``run_experiment``.
    """
    ck = torch.load(ckpt_path_, map_location="cpu", weights_only=False); cfg0 = dict(ck["cfg"])
    cfg0["EPOCHS"] = ck.get("best_epoch", cfg0["EPOCHS"])
    return run_experiment(name, teacher=teacher, train_ids=list(S.TEXTS), val_ids=[], **{**cfg0, **overrides})


def show_results():
    """Print the results table of all experiments run in this session.

    Args:
        None

    Returns:
        None
    """
    cols = ["name", "arch", "method", "task", "kd", "best_epoch", "val_f1", "val_p", "val_r", "total_params_M", "trainable_params_M", "train_min", "peak_train_mem_MB", "infer_tok_per_s"]
    try:
        import pandas as pd; print(pd.DataFrame(S.RESULTS)[cols].round(3).to_string(index=False))
    except Exception:
        for r in S.RESULTS: print({k: r[k] for k in cols})
