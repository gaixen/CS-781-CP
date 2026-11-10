"""Configuration: global :class:`Settings`, experiment registry (E1-E11) and helpers."""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def default_phase1_ckpt() -> Dict[str, Optional[str]]:
    """Default Phase-1 checkpoint registry (Kaggle paths from the notebook).

    Args:
        None

    Returns:
        Dict[str, Optional[str]]: ``architecture -> checkpoint directory`` (None = unavailable).
    """
    return {
        "lightweight_sliding_window": "/kaggle/input/datasets/elise047/cs-781-phase1-lightweight-sliding-window/cs781_phase1/lightweight_sliding_window/model-final",
        "lightweight_sliding_window_bidir": "/kaggle/input/datasets/gaixen007/lsw-bidirectional-checkpoints/cs781_phase1/lightweight_sliding_window_bidir/model-final",
        "modern_decoder": "/kaggle/input/datasets/gaixen007/cs781-phase1-modern-decoder-checkpoints/cs781_phase1/modern_decoder/model-final",
        "gpt2_decoder": "/kaggle/input/datasets/gaixen047/cs781-phase1-checkpoints-gpt2-train-2048/model-final",
        "seq2seq_t5": "/kaggle/input/datasets/silverwing011/seq2seq-t5/cs781_phase1/seq2seq_t5/model-final",
    }


@dataclass
class Settings:
    """All global settings.

    Args:
        smoke (bool): Tiny subsets + 1 epoch for an end-to-end check.
        seed (int): Global seed.
        work (str): Working directory.
        local_data (Optional[dict]): Optional ``{"train": path, "test": path}`` (bypasses the HF download).
        hf_dataset_repo (str): HuggingFace dataset repository.
        data_files (dict): Files inside the repository.
        dev_n (int): Validation documents carved from train.
        dev_eval_n (int): Dev documents used for generation metrics.
        phase1_ckpt (dict): Phase-1 checkpoints.
        strict_load (bool): Fail when checkpoint weights are missing.
        doc_tok (int): Document tokens fed to the model.
        sum_tok (int): Maximum summary tokens.
        max_raw_tok (int): Tokenised documents longer than this keep head 70 % + tail 30 %.
        strat (str): Default long-document strategy (``headtail | select``).
        bs (int): Documents per micro-batch.
        accum (int): Gradient accumulation steps.
        amp (bool): fp16 autocast on CUDA.
        grad_ckpt (bool): Gradient checkpointing.
        max_min_per_exp (float): Wall-clock cap per experiment (minutes).
        gen (dict): Generation settings (overwritten from the data audit).
        gen_bs (int): Generation batch size.
        amp_t5 (bool): fp16 autocast for T5 (can overflow).
        allow_t5_random_decoder (bool): Continue when the T5 checkpoint has no decoder weights.
        force (bool): Ignore cached ``result.json``.
        ens_bs (int): Ensemble generation batch size.
        ens_select (List[str]): Ensembles from :data:`ENS_RUN` to run.
        kd_seeds (List[int]): Seeds of the student-distillation ensembles.
        kd_epochs (int): Epochs of the student distillation (train+dev).
        kd_lr (float): Student learning rate.
        kd_alpha (float): CE / KD balance.
        kd_temp (float): KD temperature.
        kd_force (bool): Retrain existing student checkpoints.
        student_arch (str): Student architecture.
    """
    smoke: bool = False
    seed: int = 42
    work: str = field(default_factory=lambda: "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("./work"))
    local_data: Optional[Dict[str, str]] = None
    hf_dataset_repo: str = "Exploration-Lab/CS-781-Capstone"
    data_files: Dict[str, str] = field(default_factory=lambda: {"train": "Phase_6_Summ/Train/train.json", "test": "Phase_6_Summ/Test/test.json"})
    dev_n: int = 300
    dev_eval_n: int = 120
    phase1_ckpt: Dict[str, Optional[str]] = field(default_factory=default_phase1_ckpt)
    strict_load: bool = True
    doc_tok: int = 1536
    sum_tok: int = 256
    max_raw_tok: int = 24000
    strat: str = "headtail"
    bs: int = 4
    accum: int = 4
    amp: bool = True
    grad_ckpt: bool = True
    max_min_per_exp: float = 90
    gen: dict = field(default_factory=lambda: dict(max_new=200, min_new=60, nrep=3, rep_pen=1.1))
    gen_bs: int = 8
    amp_t5: bool = False
    allow_t5_random_decoder: bool = False
    force: bool = False
    ens_bs: int = 4
    ens_select: List[str] = field(default_factory=lambda: ["E1+E2"])
    kd_seeds: List[int] = field(default_factory=lambda: [17, 42, 73])
    kd_epochs: int = 1
    kd_lr: float = 2e-4
    kd_alpha: float = 0.5
    kd_temp: float = 2.0
    kd_force: bool = False
    student_arch: str = "lightweight_sliding_window"

    def __post_init__(self):
        """Apply the smoke-test overrides.

        Args:
            None

        Returns:
            None
        """
        if self.smoke:
            self.dev_n, self.dev_eval_n = 16, 8

    @property
    def out(self) -> str:
        """Experiment run folders.

        Args:
            None

        Returns:
            str: ``<work>/p6_runs``.
        """
        return f"{self.work}/p6_runs"

    @property
    def cache(self) -> str:
        """Tokenisation cache folder.

        Args:
            None

        Returns:
            str: ``<work>/p6_cache``.
        """
        return f"{self.work}/p6_cache"

    @property
    def submit(self) -> str:
        """Submission zips folder.

        Args:
            None

        Returns:
            str: ``<work>/p6_submissions``.
        """
        return f"{self.work}/p6_submissions"

    def make_dirs(self) -> None:
        """Create all working folders.

        Args:
            None

        Returns:
            None
        """
        for d in (self.work, self.out, self.cache, self.submit):
            os.makedirs(d, exist_ok=True)

    def have(self, arch: str) -> bool:
        """Whether a Phase-1 checkpoint is configured for an architecture.

        Args:
            arch (str): Architecture key.

        Returns:
            bool: True if a path is set.
        """
        return self.phase1_ckpt.get(arch) is not None


def arch_kind(a: str) -> str:
    """Wrapper family of an architecture.

    Args:
        a (str): Architecture key.

    Returns:
        str: ``gpt2``, ``t5`` or ``custom``.
    """
    return {"gpt2_decoder": "gpt2", "seq2seq_t5": "t5"}.get(a, "custom")


def doc_budget(s: Settings, cfg: dict) -> int:
    """Document tokens fed to a model (GPT-2 has learned absolute positions, so its budget is cut to ``n_positions - sum_tok - 8``).

    Args:
        s (Settings): Settings.
        cfg (dict): Experiment definition.

    Returns:
        int: Document token budget.
    """
    if cfg.get("doc_tok"):
        return cfg["doc_tok"]
    if cfg["arch"] == "gpt2_decoder" and s.have("gpt2_decoder"):
        try:
            npos = int(json.load(open(os.path.join(s.phase1_ckpt["gpt2_decoder"], "config.json"))).get("n_positions", 1024))
        except Exception:
            npos = 1024
        return int(min(s.doc_tok, npos - s.sum_tok - 8))
    return s.doc_tok


def build_experiments(s: Settings) -> Dict[str, dict]:
    """Experiment registry (identical to the notebook).

    Args:
        s (Settings): Settings (``strat``, ``smoke``).

    Returns:
        Dict[str, dict]: ``eid -> definition``.
    """
    st = s.strat
    ex = {
        "E1": dict(name="FullFT", method="full", arch="modern_decoder", lr=2e-4, epochs=3, strat=st),
        "E2": dict(name="LoRMAp", method="lorma", arch="lightweight_sliding_window", lr=1e-3, epochs=4, strat=st, which="attn_ffn", r=128, alpha=128),
        "E3": dict(name="Adapter", method="adapters", arch="lightweight_sliding_window_bidir", lr=1e-3, epochs=4, strat=st, r=128),
        "E4": dict(name="KD", method="kd", arch="lightweight_sliding_window", lr=3e-4, epochs=4, strat=st, teacher="E1", keep_every=2, kd_alpha=0.5, kd_temp=2.0),
        "E5": dict(name="Scratch", method="full", arch="modern_decoder", lr=2e-4, epochs=3, strat=st, scratch=True),
        "E6": dict(name="GPT2-LoRMAp", method="lorma", arch="gpt2_decoder", lr=1e-3, epochs=4, strat=st, which="attn_ffn", r=32, alpha=32),
        "E7": dict(name="GPT2-FullFT", method="full", arch="gpt2_decoder", lr=1e-4, epochs=3, strat=st),
        "E8": dict(name="T5-FullFT", method="full", arch="seq2seq_t5", lr=3e-4, epochs=3, strat=st),
        "E9": dict(name="T5-LoRMAp", method="lorma", arch="seq2seq_t5", lr=1e-3, epochs=4, strat=st, which="attn_ffn", r=32, alpha=32),
        "E10": dict(name="LoRMAp-modern", method="lorma", arch="modern_decoder", lr=1e-3, epochs=4, strat=st, which="attn_ffn", r=128, alpha=128),
        "E11": dict(name="LoRMAp-r64-select", method="lorma", arch="lightweight_sliding_window", lr=1e-3, epochs=4, strat="select", which="attn_ffn", r=128, alpha=128),
    }
    if s.smoke:
        for c in ex.values():
            c["epochs"] = 1
    return ex


DEFAULT_RUN = ["E1", "E2", "E10", "E11"]      # experiments without a Phase-1 checkpoint are skipped automatically; add E4 / E5 if wanted
ENS_RUN = {"E1+E2": ["E1", "E2"], "E1+E2+E10": ["E1", "E2", "E10"], "E1+E2+GPT2": ["E1", "E2", "E6"], "E1+E2+T5": ["E1", "E2", "E8"], "TOP3": "auto:3"}
