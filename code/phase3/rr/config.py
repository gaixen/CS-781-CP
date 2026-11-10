"""Configuration: global settings (:class:`Settings`), architecture table, label list and default experiment config ``CFG``."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

OFFICIAL = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent", "PrecedentReliedUpon",
            "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision", "RulingByPresentCourt", "None"]

# kind: which backbone wrapper to use. window: sliding-window size used in Phase 1 (None = full attention)
ARCH_INFO = {
    "lightweight_sliding_window": dict(kind="custom", window=256),
    "modern_decoder": dict(kind="custom", window=None),
    "gpt2_decoder": dict(kind="gpt2"),
    "seq2seq_t5": dict(kind="t5"),
}


def default_phase1_ckpt() -> Dict[str, Optional[str]]:
    """Default Phase-1 checkpoint registry (paths used in the notebook).

    Args:
        None

    Returns:
        Dict[str, Optional[str]]: ``architecture -> checkpoint path`` (.pt, .safetensors or HF folder).
    """
    return {
        "lightweight_sliding_window": "/kaggle/input/datasets/gaixen007/lsw-bidirectional-checkpoints/cs781_phase1/lightweight_sliding_window_bidir/model-final",
        "modern_decoder": "/kaggle/input/datasets/gaixen007/cs781-phase1-modern-decoder-checkpoints/cs781_phase1/modern_decoder/model-final",
        "gpt2_decoder": "/kaggle/input/datasets/sohamjee/cs781-phase1-checkpoints-gpt2-train-2048-150m/model-final",
        "seq2seq_t5": None,
    }


def default_cfg() -> dict:
    """Default experiment configuration (the notebook's ``CFG``); experiments override single keys.

    Args:
        None

    Returns:
        dict: UPPER_CASE hyper-parameters (encoder, head, windows, optimisation, PEFT, KD, recall options).
    """
    return dict(
        ARCH="lightweight_sliding_window", CKPT=None, METHOD="lorma",              # METHOD: full | partial | bitfit | lora | rslora | lorma | adapter
        ATTN_MODE="bidirectional", BIDIR_WINDOW="keep",
        MAX_SENT_TOK=96, NEIGHBORS=3, NB_TOK=32, DOMAIN_TOKEN=True, POOL="mean",
        SEQ="bilstm", HID=256, HEAD_DROPOUT=0.2, USE_CRF=False, SHIFT_LAMBDA=0.0, EMIT_CE_W=0.5,
        WIN=32, WIN_STRIDE=16, INF_WIN=64, INF_STRIDE=32, BATCH_WIN=1,
        ACCUM=8, EPOCHS=4, WARMUP=0.06, WD=0.01, CLIP=1.0,
        LR={"full": 3e-4, "partial": 1e-4, "bitfit": 5e-4, "lora": 6e-4, "rslora": 6e-4, "lorma": 6e-4, "adapter": 5e-4}, LR_HEAD=1e-3,
        CLASS_WEIGHT_POWER=0.5, CLASS_WEIGHT_MAX=10.0, LABEL_SMOOTH=0.0,
        R=64, ALPHA=128, ALPHA_RS=16, LORA_DROPOUT=0.05, BOTTLENECK=64, PEFT_TARGETS=None, PARTIAL_LAYERS=4,
        KEEP_LAYERS=None, KD_ALPHA=0.5, KD_TEMP=2.0, SEED=42, LOG_EVERY=200,
        LOGIT_ADJ_TAU=0.0, OVERSAMPLE_POWER=0.0,
    )


@dataclass
class Settings:
    """Global settings and stage options.

    Args:
        seed (int): Global seed.
        work (str): Working directory (checkpoints ``*.pt``, ``results.json``, submissions).
        hf_dataset_repo (str): HuggingFace dataset repository.
        data_root (Optional[str]): Local folder containing the ``Phase_3_RR`` json files (None = download).
        phase1_ckpt (dict): Phase-1 checkpoints.
        force_retrain (bool): Retrain even when ``<name>.pt`` exists.
        head_best (dict): Head configuration used for the architecture comparison.
        window_k (int): Window of best models used as KD teachers.
        kd_teachers (Optional[List[str]]): Restrict KD teachers to these run names.
        kd_ensemble (bool): Also distil from the ensemble of the window.
        kd_control (bool): Train the no-KD control student.
        kd_force (bool): Retrain KD runs even if their checkpoint exists.
        student_keep (object): ``None`` (same depth) | ``"half"`` | explicit list of layer ids.
        ens_window_k (int): Window size for the all-subsets ensemble search.
        ens_include_kd (bool): Let distilled students join the ensemble window.
        final_seeds (Tuple[int, ...]): Seeds for the train+dev final retrains.
        ens_members (List[str]): Ensemble members tuned in the ensemble stage.
        ens2_teachers (List[str]): Teachers of ``C_kd_from_ENS2`` (cannot be recovered from the checkpoint).
        seed_final (int): Seed of the final train+dev retrains.
        recall_beta (float): F-beta of the recall-oriented bias variant.
        accept_gain (float): Minimum macro-F1 gain for accepting a recall method.
        accept_frac (float): Minimum bootstrap win rate for accepting a recall method.
    """
    seed: int = 42
    work: str = field(default_factory=lambda: "/kaggle/working/cs781_phase3" if os.path.isdir("/kaggle/working") else os.path.abspath("./work/cs781_phase3"))
    hf_dataset_repo: str = "Exploration-Lab/CS-781-Capstone"
    data_root: Optional[str] = None
    phase1_ckpt: Dict[str, Optional[str]] = field(default_factory=default_phase1_ckpt)
    force_retrain: bool = False
    head_best: dict = field(default_factory=lambda: dict(SEQ="bilstm", NEIGHBORS=0, USE_CRF=True, SHIFT_LAMBDA=0.3))
    window_k: int = 2
    kd_teachers: Optional[List[str]] = None
    kd_ensemble: bool = True
    kd_control: bool = False
    kd_force: bool = False
    student_keep: object = None
    ens_window_k: int = 5
    ens_include_kd: bool = True
    final_seeds: Tuple[int, ...] = (1, 2, 3)
    ens_members: List[str] = field(default_factory=lambda: ["A4_bilstm_crf_shift", "C_kd_from_A4_bilstm_crf_shift", "C_kd_from_B_modern_lora", "C_kd_from_A3_bilstm_crf", "C_kd_from_ENS2"])
    ens2_teachers: List[str] = field(default_factory=lambda: ["A4_bilstm_crf_shift", "B_modern_lora"])
    seed_final: int = 1
    recall_beta: float = 2.0
    accept_gain: float = 0.005
    accept_frac: float = 0.70

    def make_dirs(self) -> None:
        """Create the working directory.

        Args:
            None

        Returns:
            None
        """
        os.makedirs(self.work, exist_ok=True)


# Stage A (context ablation) and stage B (architecture x technique) run lists, exactly as in the notebook.
STAGE_A = [("A1_neighbors1", dict(SEQ="none", NEIGHBORS=2)), ("A4_bilstm_crf_shift", dict(SEQ="bilstm", NEIGHBORS=0, USE_CRF=True, SHIFT_LAMBDA=0.3))]
STAGE_A_ARCHS = ["modern_decoder", "lightweight_sliding_window"]
STAGE_B = [("B_lsw_lorma", dict(ARCH="lightweight_sliding_window", METHOD="lorma")), ("B_modern_lora", dict(ARCH="modern_decoder", METHOD="lora"))]
EXTRA_B = [("B_lsw_full", dict(ARCH="lightweight_sliding_window", METHOD="full"))]
