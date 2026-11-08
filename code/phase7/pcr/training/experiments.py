"""Experiment runner: build -> (train) -> score dev / test -> save scores and submission -> ``result.json``.

Runs are resumable: finished experiments are loaded from ``result.json`` (unless ``cfg.force``).
E1 (the KD teacher) must finish before E4.
"""
import gc
import json
import os
import time
import traceback

import numpy as np
import torch

from pcr.models.dual_encoder import build_model, n_params, score_both
from pcr.retrieval.metrics import f1_at_k, recall_at
from pcr.submission import write_submission
from pcr.training.trainer import exp_dir, load_best, train_exp


def run_experiment(eid, ctx):
    """Run (or load from cache) one experiment.

    Args:
        eid (str): Experiment id, key of ``ctx.experiments`` (E0 ... E5).
        ctx (PCRContext): Pipeline context (config, device, tokenizer ids, splits, train data).

    Returns:
        dict: Result with metrics (dev P/R/F1, recall@k), parameter counts, training time, memory, inference
        time per document, the per-epoch history and the path of the submission zip.
    """
    cfg, spl = ctx.cfg, ctx.spl
    exp_cfg, d = ctx.experiments[eid], exp_dir(cfg, eid)
    rj = f"{d}/result.json"
    if os.path.exists(rj) and not cfg.force:
        print(f"[{eid}] cached")
        return json.load(open(rj))
    print(f"\n===== {eid} {exp_cfg['name']} | {exp_cfg['arch']} | {exp_cfg['method']} =====")
    gc.collect()
    torch.cuda.empty_cache()
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = build_model(exp_cfg, cfg, ctx.eos, ctx.device)
    tp, ap = n_params(model)
    print(f"  trainable {tp/1e6:.2f}M / total {ap/1e6:.2f}M")
    teacher, hist, tsec = None, [], 0.0
    if exp_cfg["method"] == "kd":
        teacher = load_best(build_model(ctx.experiments[exp_cfg["teacher"]], cfg, ctx.eos, ctx.device), exp_cfg["teacher"], cfg).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  teacher {exp_cfg['teacher']}: {n_params(teacher)[1]/1e6:.1f}M -> student {ap/1e6:.1f}M ({100*ap/n_params(teacher)[1]:.0f}%)")
    if exp_cfg["method"] != "zeroshot":
        hist, tsec = train_exp(eid, exp_cfg, model, teacher, ctx.train_data, spl, cfg)
        load_best(model, eid, cfg)
    del teacher
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if ctx.device.type == "cuda" else 0.0
    t0 = time.time()
    Sd, Sd_ms = score_both(model, spl, "dev")
    inf = (time.time() - t0) / max(len(spl["dev"]["cids"]) + len(spl["dev"]["qids"]), 1)
    St, St_ms = score_both(model, spl, "test")
    np.save(f"{d}/dev_ms.npy", Sd_ms.astype(np.float16))
    np.save(f"{d}/test_ms.npy", St_ms.astype(np.float16))
    np.save(f"{d}/dev_scores.npy", Sd.astype(np.float16))
    np.save(f"{d}/test_scores.npy", St.astype(np.float16))
    zp = write_submission(f"{eid}-{exp_cfg['name']}_{exp_cfg['arch']}", St, spl, cfg)
    res = dict(exp=eid, name=exp_cfg["name"], method=exp_cfg["method"], arch=exp_cfg["arch"], trainable_M=tp / 1e6, total_M=ap / 1e6,
               fp16_MB=ap * 2 / 2 ** 20, emb_dim=model.dim, train_min=tsec / 60, peak_gpu_GB=peak, infer_s_per_doc=inf,
               best_dev_f1=max([h["f1"] for h in hist], default=None), hist=hist, zip=zp,
               **{f"dev_{k}": v for k, v in f1_at_k(Sd, spl, "dev", cfg.k).items()},
               **{f"dev_rec@{k}": v for k, v in recall_at(Sd, spl, "dev").items()})
    json.dump(res, open(rj, "w"), indent=1)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "hist"})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def run_all(ctx, run_ids):
    """Run a list of experiments; a failing experiment is reported and skipped.

    Args:
        ctx (PCRContext): Pipeline context. ``ctx.results`` is updated in place.
        run_ids (Iterable[str]): Experiment ids to execute in order (E1 before E4).

    Returns:
        Dict[str, dict]: ``eid -> result`` for the experiments that succeeded.
    """
    for e in run_ids:
        try:
            ctx.results[e] = run_experiment(e, ctx)
        except Exception:
            print(f"[{e}] FAILED")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
    return ctx.results
