"""End-to-end experiment runner: config -> model -> (teacher) -> train -> dev eval -> test predictions -> submission."""
import os
import gc
import json
import time
import zipfile

import numpy as np
import torch

from .config import (BASE, EXPERIMENTS, HEAD_KEEP, TAIL_KEEP, OUT_ROOT, RESULTS_PATH, PHASE1_CKPT, STRICT_LOAD, SEED)
from .context import device
from .data.tokenization import get_tokenizer, get_examples
from .eval.predict import predict
from .models.backbone import build_backbone
from .models.classifier import CJPEModel
from .models.peft import (inject_lorma, enable_bitfit, inject_adapters, set_trainable_top_k, prune_layers, count_params)
from .training.trainer import train_model
from .utils import set_seed, metrics, best_threshold


def make_cfg(name, ov=None):
    """Assemble an experiment config: ``BASE`` < ``EXPERIMENTS[name]`` < overrides, plus derived fields.

    Args:
        name (str): experiment name (key of ``EXPERIMENTS``; unknown names fall back to ``BASE``).
        ov (dict or None): override dict (e.g. ``dict(TRAIN_SUBSET=2000, EPOCHS=1)``).

    Returns:
        dict: full config with ``name``, ``_amp`` (fp16 autocast flag; off for T5) and ``_pad_id``.
    """
    cfg = {**BASE, **EXPERIMENTS.get(name, {}), **(ov or {})}; cfg["name"] = name
    cfg["_amp"] = bool(device.type == "cuda" and (cfg["AMP"] if cfg["AMP"] is not None else cfg["arch"] != "seq2seq_t5"))   # T5 overflows in fp16
    assert cfg["MAX_TOKENS"] <= (HEAD_KEEP if cfg["REPRESENTATION"] == "first_n" else TAIL_KEEP), "MAX_TOKENS exceeds cached tokens"
    assert cfg["MAX_CHUNKS"] * (cfg["CHUNK_TOKENS"] - cfg["CHUNK_STRIDE"]) + cfg["CHUNK_STRIDE"] <= TAIL_KEEP, "raise TAIL_KEEP for this many chunks"
    cfg["_pad_id"] = (lambda t: t.pad_token_id if t.pad_token_id is not None else 0)(get_tokenizer(cfg["arch"]))
    return cfg


def build_model(cfg, feat_target_dim=None):
    """Build the CJPE model and apply the adaptation technique (``cfg["method"]``).

    Args:
        cfg (dict): experiment config.
        feat_target_dim (int or None): teacher feature dim (distillation only).

    Returns:
        CJPEModel: model on ``device`` in float32, with ``requires_grad`` set according to the method.

    Raises:
        ValueError: for an unknown method.
    """
    arch, method = cfg["arch"], cfg["method"]
    bb = build_backbone(arch, PHASE1_CKPT[arch], STRICT_LOAD)
    V = bb.get_embeddings().weight.shape[0]; tok = get_tokenizer(arch)
    assert len(tok) <= V, f"tokenizer ({len(tok)}) larger than model vocab ({V})"
    if len(tok) != V: print(f"note: tokenizer={len(tok)} vs embedding rows={V}")
    need = max(cfg["MAX_TOKENS"] if cfg["REPRESENTATION"] != "hier" else cfg["CHUNK_TOKENS"], 1)
    assert need <= bb.max_len, f"{need} tokens > backbone context {bb.max_len}"
    if cfg["KEEP_EVERY"] > 1: print("layer pruning (student): %d -> %d layers" % prune_layers(bb, cfg["KEEP_EVERY"]))
    model = CJPEModel(bb, cfg, feat_target_dim)
    if method in ("full", "distill"): pass
    elif method == "lorma":
        for p in bb.parameters(): p.requires_grad = False
        print(f"LoRMA+ ({cfg['LORMA_TARGETS']}) injected into {inject_lorma(bb, cfg['LORMA_TARGETS'], cfg['PEFT_R'], cfg['PEFT_ALPHA'])} modules")
    elif method == "bitfit":
        e, a = enable_bitfit(bb); print(f"BitFit: {e:,} existing bias params + {a:,} added zero-init biases (arch has none)")
    elif method == "adapters":
        for p in bb.parameters(): p.requires_grad = False
        print(f"Adapters injected into {inject_adapters(bb, cfg['ADAPTER_R'])} blocks")
    elif method == "partial": set_trainable_top_k(bb, cfg["TOP_K"]); print(f"Partial FT: top {cfg['TOP_K']} layers + final norm")
    elif method == "gradual": set_trainable_top_k(bb, 1)
    else: raise ValueError(method)
    model = model.to(device=device, dtype=torch.float32)
    t, tot = count_params(model); print(f"trainable {t:,}/{tot:,} ({100*t/tot:.2f}%)")
    return model


def prepare_teacher(cfg, ov):
    """Load (training first if needed) the teacher experiment and cache its logits/features on the train set.

    Args:
        cfg (dict): student config (uses ``teacher``, ``teacher_weights``).
        ov (dict or None): overrides of the student run; only ``TRAIN_SUBSET, DEV_SUBSET, EPOCHS`` are forwarded.

    Returns:
        dict: ``logits, feats, feat_dim, teacher_params, teacher`` (name).
    """
    tname = cfg["teacher"]; t_ov = {k: v for k, v in (ov or {}).items() if k in ("TRAIN_SUBSET", "DEV_SUBSET", "EPOCHS")}
    tcfg = make_cfg(tname, t_ov); wpath = cfg["teacher_weights"] or f"{OUT_ROOT}/{tname}/best.pt"
    if not os.path.exists(wpath):
        print(f"teacher weights not found at {wpath} -> training teacher '{tname}' first"); run_experiment(tname, **t_ov)
    teacher = build_model(tcfg); teacher.load_state_dict(torch.load(wpath, map_location=device)); n_t = count_params(teacher)[1]
    tr_t = get_examples(tcfg["arch"], "train", tcfg); out = predict(teacher, tr_t, tcfg, want_feats=True)
    acc = np.mean(out["logits"].argmax(1) == np.array([e["label"] for e in tr_t])); print(f"teacher '{tname}': {n_t/1e6:.1f}M params, train acc {acc:.3f}")
    del teacher; gc.collect(); torch.cuda.empty_cache()
    return dict(logits=out["logits"], feats=out["feats"], feat_dim=out["feats"].shape[1], teacher_params=n_t, teacher=tname)


def infer_docs_per_s_fmt(x):
    """Format an inference throughput value.

    Args:
        x (float): documents per second.

    Returns:
        str: e.g. ``"12.3 docs/s"``.
    """
    return f"{x:.1f} docs/s"


def run_experiment(name, **ov):
    """Run one experiment end to end and append a record to ``results.jsonl``.

    Writes ``<OUT_ROOT>/<name>/{best.pt, dev_preds.npz, prediction.json, submission.zip}``.

    Args:
        name (str): experiment name (key of ``EXPERIMENTS``).
        **ov: config overrides, e.g. ``TRAIN_SUBSET=2000, DEV_SUBSET=500, EPOCHS=1`` for a smoke test.

    Returns:
        dict: the result record (dev metrics, params, memory, time, throughput, ...).
    """
    cfg = make_cfg(name, ov); set_seed(SEED); odir = f"{OUT_ROOT}/{name}"; os.makedirs(odir, exist_ok=True)
    print("=" * 100, f"\n{name}: arch={cfg['arch']} method={cfg['method']} rep={cfg['REPRESENTATION']} pool={cfg['POOLING']} "
          f"head_layers={cfg['HEAD_LAYERS']}" + (f" doc={cfg['DOC_AGG']}x{cfg['DOC_LAYERS']}" if cfg["REPRESENTATION"] == "hier" else ""))
    tr, dv, te = (get_examples(cfg["arch"], s, cfg) for s in ("train", "dev", "test"))
    kd = prepare_teacher(cfg, ov) if cfg["method"] == "distill" else None
    model = build_model(cfg, kd["feat_dim"] if kd else None); tot_p = count_params(model)[1]; trn_p = count_params(model)[0]
    if kd and kd["teacher_params"] <= tot_p: print("WARNING: teacher is not larger than the student -> make sure it is at least *stronger* (spec 10.2.2)")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    best_path = f"{odir}/best.pt"; t0 = time.time(); hist = train_model(model, tr, dv, cfg, best_path, kd); train_min = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    # ---- final dev evaluation (timed -> inference cost), threshold calibration ----
    dv_y = np.array([e["label"] for e in dv]); t0 = time.time(); dv_out = predict(model, dv, cfg); infer_dps = len(dv) / (time.time() - t0)
    raw = metrics(dv_y, (dv_out["prob"] >= 0.5).astype(int)); thr, tuned = best_threshold(dv_y, dv_out["prob"]) if cfg["CALIBRATE"] else (0.5, raw["macro_f1"])
    fin = metrics(dv_y, (dv_out["prob"] >= thr).astype(int))
    np.savez(f"{odir}/dev_preds.npz", prob=dv_out["prob"], label=dv_y, n_chars=np.array([e["n_chars"] for e in dv]))
    # ---- test predictions + submission (exact required format) ----
    te_out = predict(model, te, cfg); preds = [{"id": e["id"], "pred": int(p >= thr)} for e, p in zip(te, te_out["prob"])]
    pred_path = f"{odir}/prediction.json"; json.dump(preds, open(pred_path, "w"))
    with zipfile.ZipFile(f"{odir}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z: z.write(pred_path, "prediction.json")
    trainable_mb = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / 2**20
    rec = dict(name=name, arch=cfg["arch"], method=cfg["method"], rep=cfg["REPRESENTATION"], pooling=cfg["POOLING"], head_layers=cfg["HEAD_LAYERS"],
               doc_agg=cfg["DOC_AGG"] if cfg["REPRESENTATION"] == "hier" else "-", max_tokens=cfg["MAX_TOKENS"], class_weighted=cfg["CLASS_WEIGHTED"],
               dev_f1_raw=raw["macro_f1"], dev_f1=fin["macro_f1"], dev_P=fin["macro_p"], dev_R=fin["macro_r"], f1_c0=fin["f1_c0"], f1_c1=fin["f1_c1"], thr=thr,
               params_M=tot_p / 1e6, trainable_M=trn_p / 1e6, trainable_pct=100 * trn_p / tot_p, trainable_MB=trainable_mb, peak_mem_GB=peak,
               train_min=train_min, infer_docs_per_s=infer_dps, test_pos_rate=float(np.mean([p["pred"] for p in preds])),
               teacher=kd["teacher"] if kd else None, teacher_params_M=kd["teacher_params"] / 1e6 if kd else None, epochs=cfg["EPOCHS"],
               subset=cfg["TRAIN_SUBSET"], ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(RESULTS_PATH, "a") as f: f.write(json.dumps(rec) + "\n")
    print(f"==> {name}: dev macro-F1 {fin['macro_f1']:.4f} (P={fin['macro_p']:.3f} R={fin['macro_r']:.3f}, thr={thr:.2f}; raw@0.5 {raw['macro_f1']:.4f}) | "
          f"{tot_p/1e6:.1f}M params, {trn_p/1e6:.2f}M trainable | {infer_docs_per_s_fmt(infer_dps)} | test positives {rec['test_pos_rate']:.2f}")
    del model, kd; gc.collect(); torch.cuda.empty_cache(); return rec
