"""Named experiment procedures of the notebook (each is a thin wrapper around ``run_experiment``)."""
import os

from . import state as S
from .config import CFG, EXPERIMENTS, EXP_LOSS, ROLES, WORK
from .training.trainer import run_experiment, load_trained, cfg_from_ckpt, ckpt_path, show_results


def run_named(name, **overrides):
    """Run a registered experiment (``config.EXPERIMENTS``) with optional overrides.

    Args:
        name (str): registry key.
        **overrides: config overrides.

    Returns:
        tuple: result of ``run_experiment``.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    if name not in EXPERIMENTS: raise KeyError(f"unknown experiment '{name}'; known: {sorted(EXPERIMENTS)}")
    return run_experiment(name, **{**EXPERIMENTS[name], **overrides})


def run_second_family(base="F_focal", name="F_lsw", arch="lightweight_sliding_window"):
    """Train a second backbone family with the SAME config as the validated base run (only ``ARCH`` changes).

    Args:
        base (str): validated base run name.
        name (str): new run name.
        arch (str): architecture for the new run.

    Returns:
        None
    """
    if os.path.exists(ckpt_path(name)): print(name, "exists -> skip"); return
    ov = {k: v for k, v in cfg_from_ckpt(ckpt_path(base)).items() if k != "ARCH"}
    run_experiment(name, ARCH=arch, **ov)


def run_training_time_enforcement(base="F_focal"):
    """Experiments T1/T2/T3 (ctx prefix, role weights + oversampling, both) and B0/B1 (BIO softmax vs BIO+CRF); one change at a time.

    The 192-token prefix makes windows ~50% longer: same effective batch, less memory.

    Args:
        base (str): validated base run (config reused).

    Returns:
        None
    """
    ov = cfg_from_ckpt(ckpt_path(base))
    half = dict(BATCH=max(4, ov["BATCH"] // 2), ACCUM=ov["ACCUM"] * 2)
    run_experiment("T1_ctx192", **{**ov, **half, "CTX_PREFIX": 192})                  # document context in every window
    run_experiment("T2_roles", **{**ov, **ROLES})                                      # recall for APP / RESP / WIT / AUTH through the loss
    run_experiment("T3_ctx_roles", **{**ov, **half, "CTX_PREFIX": 192, **ROLES})       # both
    show_results()
    bio = {k: v for k, v in ov.items() if k not in ("TASK", "USE_CRF")}
    run_experiment("B0_bio_softmax", TASK="bio", USE_CRF=False, **bio)
    run_experiment("B1_bio_crf", TASK="bio", USE_CRF=True, **bio)                      # CRF loss = plain marginal likelihood
    show_results()


def run_distillation(teacher_name="F_focal", name="E4_kd_student", epochs=6, keep_layers=None):
    """E4: distill the best teacher into a student (works for BIO and span teachers).

    With ``keep_layers=None`` the student keeps all layers (not smaller); use ``run_pruned_kd`` for real compression.

    Args:
        teacher_name (str): fine-tuned teacher run.
        name (str): student run name.
        epochs (int): student epochs.
        keep_layers (list[int] or None): layers to keep.

    Returns:
        None
    """
    path = ckpt_path(teacher_name)
    if not os.path.exists(path): print("Run the teacher experiment first (teacher checkpoint not found)."); return
    teacher, t_cfg = load_trained(path)
    for p in teacher.parameters(): p.requires_grad = False
    run_experiment(name, teacher=teacher, ARCH=t_cfg["ARCH"], METHOD="lora", ATTN_MODE=t_cfg["ATTN_MODE"], TASK=t_cfg["TASK"], MAX_LEN=t_cfg["MAX_LEN"],
                   OVERLAP=t_cfg["OVERLAP"], MERGE=t_cfg["MERGE"], KEEP_LAYERS=keep_layers, KD_ALPHA=0.5, KD_TEMP=2.0, EPOCHS=epochs,
                   CLASS_WEIGHT_MODE="effective", LOSS=EXP_LOSS)
    del teacher
    show_results()


def run_pruned_kd(teacher_name="F_focal"):
    """E4b / E4c: a REAL compression experiment (every 2nd layer): E4b = distilled student, E4c = same student WITHOUT distillation (control).

    Args:
        teacher_name (str): fine-tuned teacher run.

    Returns:
        None
    """
    teacher, t_cfg = load_trained(ckpt_path(teacher_name))
    for p in teacher.parameters(): p.requires_grad = False
    keep = list(range(0, teacher.backbone.num_layers(), 2))
    kw = dict(ARCH=t_cfg["ARCH"], METHOD="full", ATTN_MODE=t_cfg["ATTN_MODE"], TASK=t_cfg["TASK"], MAX_LEN=t_cfg["MAX_LEN"], OVERLAP=t_cfg["OVERLAP"],
              MERGE=t_cfg["MERGE"], KEEP_LAYERS=keep, EPOCHS=8, CLASS_WEIGHT_MODE=t_cfg.get("CLASS_WEIGHT_MODE", "power"), LOSS=EXP_LOSS)
    run_experiment("E4b_pruned_kd", teacher=teacher, KD_ALPHA=0.5, KD_TEMP=2.0, **kw)      # distilled
    run_experiment("E4c_pruned_nokd", teacher=None, **kw)                                    # control
    del teacher
    show_results()
