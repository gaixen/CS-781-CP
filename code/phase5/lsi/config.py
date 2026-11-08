import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ARCH_SHORT = {
    "lightweight_sliding_window": "lsw",
    "lightweight_sliding_window_bidir": "lswBidir",
    "modern_decoder": "modDec",
    "gpt2_decoder": "gpt2",
    "seq2seq_t5": "t5enc",
}


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
        "gpt2_decoder": None,
        "seq2seq_t5": None,
    }


@dataclass
class Config:
    """All hyper-parameters and paths.

    Args:
        smoke (bool): Tiny subsets + 1 epoch for an end-to-end check.
        seed (int): Global seed.
        work (str): Working directory.
        local_data (Optional[dict]): Optional ``{"train","dev","test"}`` local json paths (bypasses HF download).
        hf_dataset_repo (str): HuggingFace dataset repository.
        data_files (dict): Files inside the repository.
        min_clean_dev (int): If the cleaned dev has fewer docs, hold out part of train as dev.
        holdout_frac (float): Fraction of train held out as dev in that case.
        phase1_ckpt (dict): Phase-1 checkpoints.
        strict_load (bool): Fail when checkpoint weights are missing.
        chunk_tokens (int): Tokens per chunk.
        overlap (int): Token overlap between chunks.
        max_doc_tokens (int): Longer documents keep head 75 % + tail 25 %.
        train_chunks (int): Chunks per document per training step.
        inf_chunks (int): Chunks per document at inference.
        doc_mode (str): ``full | head_tail | evidence | first | last``.
        pool (str): Chunk pooling (``mean | last | first | mean_last``).
        bs_docs (int): Documents per micro-batch.
        accum (int): Gradient accumulation steps.
        inf_chunk_budget (int): Max chunks per inference batch.
        amp (bool): fp16 autocast on CUDA.
        grad_ckpt (bool): Gradient checkpointing.
        head_dim (int): Label-attention head width.
        head_dropout (float): Head dropout.
        ft_lr_head (float): Learning rate of the classification head.
        max_min_per_exp (float): Wall-clock cap per experiment (minutes).
        oversample_pow (float): Rare-label document oversampling exponent (0 = off).
        train_subset (Optional[int]): Smoke-test subset sizes.
        dev_subset (Optional[int]): see ``train_subset``.
        test_subset (Optional[int]): see ``train_subset``.
        run_train (List[str]): Experiments to train (``[]`` = post-processing only, ``["E5"]`` = KD generation 2).
        prev_dataset (Optional[str]): Folder containing a previous ``lsi_runs`` (None = auto-find under /kaggle/input).
        panel_exps (List[str]): Experiments for which every threshold strategy is written.
        kd_seeds (List[int]): Seeds of the E4 seed-ensemble.
        write_all_thr (bool): Write all threshold variants for the seed ensemble.
        lb (dict): Leaderboard macro-F1 of the first batch (used only for printing).
    """

    smoke: bool = False
    seed: int = 42
    work: str = field(
        default_factory=lambda: (
            "/kaggle/working"
            if os.path.isdir("/kaggle/working")
            else os.path.abspath("./work")
        )
    )
    local_data: Optional[Dict[str, str]] = None
    hf_dataset_repo: str = "Exploration-Lab/CS-781-Capstone"
    data_files: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "Phase_5_LSI/Train/train.json",
            "dev": "Phase_5_LSI/Train/dev.json",
            "test": "Phase_5_LSI/Test/test.json",
        }
    )
    min_clean_dev: int = 100
    holdout_frac: float = 0.10
    phase1_ckpt: Dict[str, Optional[str]] = field(default_factory=default_phase1_ckpt)
    strict_load: bool = True
    chunk_tokens: int = 512
    overlap: int = 64
    max_doc_tokens: int = 12288
    train_chunks: int = 4
    inf_chunks: int = 16
    doc_mode: str = "full"
    pool: str = "mean_last"
    bs_docs: int = 4
    accum: int = 4
    inf_chunk_budget: int = 48
    amp: bool = True
    grad_ckpt: bool = True
    head_dim: int = 256
    head_dropout: float = 0.1
    ft_lr_head: float = 1e-3
    max_min_per_exp: float = 150
    oversample_pow: float = 0.3
    train_subset: Optional[int] = None
    dev_subset: Optional[int] = None
    test_subset: Optional[int] = None
    run_train: List[str] = field(default_factory=list)
    prev_dataset: Optional[str] = None
    panel_exps: List[str] = field(default_factory=lambda: ["E4"])
    kd_seeds: List[int] = field(default_factory=lambda: [42, 11])
    write_all_thr: bool = False
    lb: Dict[str, float] = field(
        default_factory=lambda: {"E1": 0.4044, "E2": 0.3605, "E3": 0.3898, "E4": 0.4078}
    )

    def __post_init__(self):
        """Apply the smoke-test overrides.

        Args:
            None

        Returns:
            None
        """
        if self.smoke:
            (
                self.train_subset,
                self.dev_subset,
                self.test_subset,
                self.max_doc_tokens,
                self.inf_chunks,
            ) = 48, 24, 24, 2048, 4

    @property
    def out_dir(self) -> str:
        """Per-experiment run folders.

        Args:
            None

        Returns:
            str: ``<work>/lsi_runs``.
        """
        return f"{self.work}/lsi_runs"

    @property
    def cache_dir(self) -> str:
        """Tokenisation cache folder.

        Args:
            None

        Returns:
            str: ``<work>/lsi_cache``.
        """
        return f"{self.work}/lsi_cache"

    @property
    def submit_dir(self) -> str:
        """Submission zips folder.

        Args:
            None

        Returns:
            str: ``<work>/submissions_fix``.
        """
        return f"{self.work}/submissions_fix"

    def make_dirs(self) -> None:
        """Create all working folders.

        Args:
            None

        Returns:
            None
        """
        for d in (self.work, self.out_dir, self.cache_dir, self.submit_dir):
            os.makedirs(d, exist_ok=True)

    def have(self, arch: str) -> bool:
        """Whether a Phase-1 checkpoint is configured for an architecture.

        Args:
            arch (str): Architecture key.

        Returns:
            bool: True if a checkpoint path is set.
        """
        return self.phase1_ckpt.get(arch) is not None


def build_experiments(cfg: Config) -> Dict[str, dict]:
    """Experiment registry E1-E5 (identical to the notebook).

    Args:
        cfg (Config): Configuration (``doc_mode``, ``pool``, ``head_dim``, ``oversample_pow``, ``smoke``).

    Returns:
        Dict[str, dict]: ``eid -> experiment definition``.
    """
    dm, pl = cfg.doc_mode, cfg.pool
    student = next(
        (a for a in ("gpt2_decoder", "seq2seq_t5") if cfg.have(a)),
        "lightweight_sliding_window_bidir",
    )
    ex = {
        "E1": dict(
            name="FullFT-LLRD",
            arch="lightweight_sliding_window_bidir",
            method="full",
            head="lwan",
            loss="asl",
            lr=5e-5,
            llrd=0.85,
            epochs=4,
            doc_mode=dm,
            pool=pl,
        ),
        "E2": dict(
            name="LoRMAplus-r16",
            arch="modern_decoder",
            method="lorma",
            lorma_r=16,
            lorma_alpha=16,
            lorma_which="attn_ffn",
            head="lwan",
            loss="bcepw",
            lr=5e-4,
            epochs=5,
            doc_mode=dm,
            pool=pl,
        ),
        "E3": dict(
            name="Adapters-r64",
            arch="lightweight_sliding_window",
            method="adapters",
            adapter_r=64,
            head="lwan",
            loss="asl",
            lr=5e-4,
            epochs=5,
            doc_mode=dm,
            pool=pl,
        ),
        "E4": dict(
            name="KD-E1E2E3-T2",
            arch=student,
            method="kd",
            keep_every=2,
            teachers=["E1", "E2", "E3"],
            kd_T=2.0,
            kd_alpha=0.4,
            kd_rare_boost=0.25,
            head="lwan",
            loss="bce",
            lr=1e-4,
            llrd=0.9,
            epochs=5,
            doc_mode=dm,
            pool=pl,
        ),
        "E5": dict(
            name="KD2-from-E1E3E4",
            arch=student,
            method="kd",
            keep_every=2,
            teachers=["E1", "E3", "E4"],
            teacher_w={"E1": 1.0, "E3": 0.5, "E4": 1.0},
            init_from="E4",
            kd_T=2.0,
            kd_alpha=0.3,
            kd_rare_boost=0.25,
            head="lwan",
            loss="bce",
            lr=5e-5,
            llrd=0.9,
            epochs=3,
            doc_mode=dm,
            pool=pl,
        ),
    }
    for e in ex.values():
        e.setdefault("head_dim", cfg.head_dim)
        e.setdefault("wd", 0.01)
        e.setdefault("oversample", cfg.oversample_pow)
        if cfg.smoke:
            e["epochs"] = 1
    return ex
