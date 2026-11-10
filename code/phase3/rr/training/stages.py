"""Experiment stages: A (context ablation), B (architectures x techniques), C (knowledge distillation), domain transfer and recall-oriented training."""
import os

import numpy as np
import pandas as pd
import torch

from rr.config import EXTRA_B, STAGE_A, STAGE_A_ARCHS, STAGE_B
from rr.eval.metrics import rr_scores
from rr.training.trainer import collect_probs, get_window, load_rr, run_once, run_rr, show_results


def run_stage_a(ctx):
    """Stage A: how much does context matter? (LoRMA fixed; only the head / context changes.)

    Args:
        ctx (Context): Pipeline context.

    Returns:
        None
    """
    for arch in STAGE_A_ARCHS:
        for nm, ov in STAGE_A:
            run_once(ctx, nm, ARCH=arch, METHOD="lorma", **ov, EPOCHS=4)
    show_results(ctx)


def run_stage_b(ctx, extra=False):
    """Stage B: Phase-1 architectures with different adaptation techniques (head = ``settings.head_best``).

    Args:
        ctx (Context): Pipeline context.
        extra (bool): Also run the extra full-fine-tuning comparison.

    Returns:
        None
    """
    for nm, ov in STAGE_B + (EXTRA_B if extra else []):
        if nm == "B_lsw_full":
            ov = {**ov, "LR": {**ctx.cfg["LR"], "full": 1e-4}}
        run_once(ctx, nm, **{**ctx.settings.head_best, **ov})
    show_results(ctx)


def student_kw(ctx, tcfg, n_layers):
    """Student configuration derived from a teacher config (LoRA student, optional layer pruning).

    Args:
        ctx (Context): Pipeline context (``settings.student_keep``).
        tcfg (dict): Teacher config.
        n_layers (int): Teacher depth.

    Returns:
        dict: Config overrides for the student run.
    """
    sk = ctx.settings.student_keep
    keep = list(range(0, n_layers, 2)) if sk == "half" else sk
    return dict(ARCH=tcfg["ARCH"], METHOD="lora", ATTN_MODE=tcfg["ATTN_MODE"], POOL=tcfg["POOL"], SEQ=tcfg["SEQ"], NEIGHBORS=tcfg["NEIGHBORS"], USE_CRF=tcfg["USE_CRF"],
                SHIFT_LAMBDA=tcfg["SHIFT_LAMBDA"], KEEP_LAYERS=keep, LR={**ctx.cfg["LR"], "full": 2e-4}, EPOCHS=6)


def run_stage_kd(ctx):
    """Stage C (mandatory KD): distil from a WINDOW of the best models, each individually and as an ensemble teacher.

    Per-sentence label distributions (temperature ``KD_TEMP``) are mixed with the supervised loss (``KD_ALPHA``).
    ``C_student_nokd`` is the identical student without a teacher (optional control).

    Args:
        ctx (Context): Pipeline context.

    Returns:
        None
    """
    s = ctx.settings
    have = lambda nm: os.path.exists(os.path.join(ctx.work, nm + ".pt")) and not s.kd_force
    win = get_window(ctx, s.window_k)
    if not win:
        print("no trained non-KD runs yet -> run stages A/B first")
        return
    print("window of best models:")
    for i, r in enumerate(win):
        print(f"  {i+1}. {r['name']:28s} dev F1 {r['f1']:.4f} (CL {r['f1_CL']:.4f} / IT {r['f1_IT']:.4f})")
    use = s.kd_teachers or [r["name"] for r in win]
    for r in win:
        nm = f"C_kd_from_{r['name']}"
        if r["name"] not in use:
            continue
        if have(nm):
            print(f"[skip] {nm}.pt exists")
            continue
        tm, tc = load_rr(ctx, r["ckpt"])
        for p in tm.parameters():
            p.requires_grad = False
        print(f"\n>>> teacher {r['name']} (dev F1 {r['f1']:.4f})")
        run_rr(ctx, nm, teacher=tm, tcfg=tc, kd_from=r["name"], KD_ALPHA=0.5, KD_TEMP=2.0, **student_kw(ctx, tc, tm.backbone.num_layers()))
        del tm
        torch.cuda.empty_cache()
    T = None
    if s.kd_ensemble and not have(f"C_kd_from_ENS{len(win)}"):
        T = [load_rr(ctx, r["ckpt"]) for r in win]
        for tm, _ in T:
            for p in tm.parameters():
                p.requires_grad = False
        print(f"\n>>> ensemble teacher of {len(win)} models")
        run_rr(ctx, f"C_kd_from_ENS{len(win)}", teacher=T, kd_from="ensemble:" + "+".join(r["name"] for r in win), KD_ALPHA=0.5, KD_TEMP=2.0,
               **student_kw(ctx, T[0][1], T[0][0].backbone.num_layers()))
    if s.kd_control and not have("C_student_nokd"):
        if T is None:
            tm0, tc0 = load_rr(ctx, win[0]["ckpt"])
            nl = tm0.backbone.num_layers()
            del tm0
        else:
            tc0, nl = T[0][1], T[0][0].backbone.num_layers()
        run_rr(ctx, "C_student_nokd", **student_kw(ctx, tc0, nl))
    T = None
    torch.cuda.empty_cache()
    show_results(ctx)


def run_domain_transfer(ctx):
    """Train CL-only / IT-only models and evaluate cross-domain (replicates the paper's domain-transfer table).

    Args:
        ctx (Context): Pipeline context.

    Returns:
        pd.DataFrame: Same-domain vs cross-domain macro-F1 per training domain.
    """
    data = ctx.data
    cl_train = [d for d in data.train if d["domain"] == "CL"]
    cl_dev = [d for d in data.dev if d["domain"] == "CL"]
    it_train = [d for d in data.train if d["domain"] == "IT"]
    it_dev = [d for d in data.dev if d["domain"] == "IT"]
    rows = []
    for name, home, tr, dv, other_dev in [("Dom_cl_only", "CL", cl_train, cl_dev, it_dev), ("Dom_it_only", "IT", it_train, it_dev, cl_dev)]:
        rec = run_rr(ctx, name, **ctx.settings.head_best, train_docs=tr, dev_docs=dv)
        model, cfg = load_rr(ctx, rec["ckpt"])
        other_probs = collect_probs(ctx, model, other_dev, cfg)
        other_y = np.concatenate([d["y"] for d in other_dev])
        sc_cross = rr_scores(data.labels, [data.labels[i] for i in other_y], [data.labels[i] for i in np.concatenate(other_probs).argmax(1)])
        rows.append(dict(trained_on=home, same_domain_f1=rec["f1"], cross_domain_f1=sc_cross["macro_f1"]))
        print(f"[{name}] same-domain dev F1={rec['f1']:.4f} | cross-domain F1={sc_cross['macro_f1']:.4f}")
        del model
        torch.cuda.empty_cache()
    combined = [r for r in ctx.results if r["name"].startswith(("A", "B")) and "f1_CL" in r]
    if combined:
        best = max(combined, key=lambda r: r["f1"])
        rows.append(dict(trained_on="CL+IT (combined)", same_domain_f1=float("nan"), cross_domain_f1=best.get("f1_CL")))
        rows.append(dict(trained_on="CL+IT (combined)", same_domain_f1=float("nan"), cross_domain_f1=best.get("f1_IT")))
    df = pd.DataFrame(rows).round(4)
    print("\n=== Domain transfer summary ===")
    print(df.to_string(index=False))
    print("\nPaper's own finding: combined (IT+CL) training generalized better than either domain alone.")
    return df


def run_recall_training(ctx):
    """Training-time recall experiment R3 (logit adjustment + gentle class weights + rare-window sampling).

    R1 (``LOGIT_ADJ_TAU=1``, no class weights) and R2 (``OVERSAMPLE_POWER=0.5``) are available by editing the call.

    Args:
        ctx (Context): Pipeline context (``settings.force_retrain`` is switched on for this stage).

    Returns:
        None
    """
    ctx.settings.force_retrain = True
    base = {**ctx.settings.head_best, "ARCH": "modern_decoder", "METHOD": "lora"}
    run_once(ctx, "R3_mix", **{**base, "LOGIT_ADJ_TAU": 0.5, "CLASS_WEIGHT_POWER": 0.75, "OVERSAMPLE_POWER": 0.25})
    show_results(ctx)
