"""Central configuration for the Phase-7 Prior Case Retrieval (PCR) pipeline.

All hyper-parameters that were module-level constants in the original notebook
live in the :class:`Config` dataclass so that every function receives them
explicitly instead of relying on globals.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def default_phase1_ckpt() -> Dict[str, Optional[str]]:
    """Return the default Phase-1 checkpoint registry (Kaggle paths from the notebook).

    Args:
        None

    Returns:
        Dict[str, Optional[str]]: Mapping ``architecture name -> checkpoint directory``.
        ``None`` means that no checkpoint is available for that architecture.
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
    """Container for every hyper-parameter and path used by the pipeline.

    Args:
        smoke (bool): If True use tiny subsets and 1 epoch to test the pipeline end to end.
        seed (int): Global random seed.
        work (str): Working directory (``/kaggle/working`` on Kaggle, ``./work`` otherwise).
        local_data (Optional[dict]): Optional dict with local json paths
            (keys: train, train_c, dev, dev_c, test, test_c).
        hf_dataset_repo (str): HuggingFace dataset repository holding the data.
        hf_prefix (str): Sub-folder prefix of the PCR files inside the repository.
        k (int): Evaluation cut-off (top-K retrieved per query).
        phase1_ckpt (dict): Phase-1 checkpoint registry.
        strict_load (bool): Raise if any backbone weight is missing from the checkpoint.
        ch_tok (int): Tokens per chunk.
        tr_ch (int): Max chunks per document during training.
        ev_ch (int): Max chunks per document at final evaluation.
        ev_ch_fast (int): Max chunks per document in the per-epoch dev evaluation.
        train_strat (str): Chunk selection strategy in training (random | head | spread).
        eval_strat (str): Chunk selection strategy at evaluation (head | spread).
        tok_pool (str): Token pooling inside a chunk (mean | last | max).
        agg (str): Chunk aggregation (mean | max).
        max_chars (int): Characters kept per document before tokenisation (head 70 % + tail 30 %).
        bs (int): Queries per micro-batch.
        accum (int): Gradient accumulation steps.
        n_hard (int): Hard negatives per query.
        hn_lo (int): Lower BM25 rank (inclusive) for hard-negative sampling.
        hn_hi (int): Upper BM25 rank (exclusive) for hard-negative sampling.
        tau (float): InfoNCE temperature.
        amp (bool): Use fp16 autocast on CUDA.
        grad_ckpt (bool): Use gradient checkpointing.
        max_min_per_exp (float): Wall-clock cap per experiment in minutes.
        eval_bs (int): Chunks per forward pass at inference.
        force (bool): Re-run experiments even if ``result.json`` exists.
        smoke_q (int): Queries per split in smoke mode.
        clean (bool): Strip case-header boilerplate and anonymisation placeholders.
        r_topk (int): Reranker candidate depth for the first LTR (section 10c).
        r_ltr (int): Reranker candidate depth for LTR v3-v6 (``R_`` in the notebook).
        r_train (int): Candidate depth for the train-split lexical LTR.
        r_stage_a (int): Candidate depth of stage A.
        r_stage_b (int): Candidate depth of stage B.
        r_stage_c (int): Candidate depth of stage C.
        kd_alpha (float): Weight of InfoNCE in the KD loss.
        kd_t (float): KD temperature.
        kd_mse (float): Weight of the similarity-MSE term in the KD loss.
    """
    smoke: bool = False
    seed: int = 42
    work: str = field(default_factory=lambda: "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("./work"))
    local_data: Optional[Dict[str, str]] = None
    hf_dataset_repo: str = "Exploration-Lab/CS-781-Capstone"
    hf_prefix: str = "Phase_7_PCR"
    k: int = 7
    phase1_ckpt: Dict[str, Optional[str]] = field(default_factory=default_phase1_ckpt)
    strict_load: bool = True
    ch_tok: int = 512
    tr_ch: int = 2
    ev_ch: int = 8
    ev_ch_fast: int = 4
    train_strat: str = "random"
    eval_strat: str = "spread"
    tok_pool: str = "mean"
    agg: str = "mean"
    max_chars: int = 120000
    bs: int = 6
    accum: int = 2
    n_hard: int = 1
    hn_lo: int = 3
    hn_hi: int = 40
    tau: float = 0.05
    amp: bool = True
    grad_ckpt: bool = True
    max_min_per_exp: float = 80
    eval_bs: int = 48
    force: bool = False
    smoke_q: int = 40
    clean: bool = True
    r_topk: int = 150
    r_ltr: int = 400
    r_train: int = 250
    r_stage_a: int = 300
    r_stage_b: int = 100
    r_stage_c: int = 60
    kd_alpha: float = 0.5
    kd_t: float = 2.0
    kd_mse: float = 5.0

    @property
    def out_dir(self) -> str:
        """Directory holding per-experiment run folders.

        Args:
            None

        Returns:
            str: ``<work>/p7v2_runs``.
        """
        return f"{self.work}/p7v2_runs"

    @property
    def cache_dir(self) -> str:
        """Directory holding tokenisation caches.

        Args:
            None

        Returns:
            str: ``<work>/p7v2_cache``.
        """
        return f"{self.work}/p7v2_cache"

    @property
    def submit_dir(self) -> str:
        """Directory holding submission zips.

        Args:
            None

        Returns:
            str: ``<work>/p7v2_submissions``.
        """
        return f"{self.work}/p7v2_submissions"

    def make_dirs(self) -> None:
        """Create the working, run, cache and submission directories.

        Args:
            None

        Returns:
            None
        """
        for d in (self.work, self.out_dir, self.cache_dir, self.submit_dir):
            os.makedirs(d, exist_ok=True)


# Experiment registry (identical to the notebook, section 7).
EXPERIMENTS: Dict[str, dict] = {
    "E0": dict(name="zeroshot_frozen", arch="modern_decoder", method="zeroshot"),
    "E1": dict(name="full_ft", arch="modern_decoder", method="full", lr=5e-5, epochs=4),
    "E2": dict(name="lorma_plus", arch="lightweight_sliding_window", method="lorma", which="attn_ffn", r=128, alpha=128, lr=3e-4, epochs=6),
    "E3": dict(name="adapters", arch="lightweight_sliding_window_bidir", method="adapters", r=128, lr=3e-4, epochs=6),
    "E4": dict(name="kd_student", arch="lightweight_sliding_window", method="kd", teacher="E1", keep_every=2, lr=1e-4, epochs=5),
    "E5": dict(name="partial_top4", arch="modern_decoder", method="partial", k=4, lr=1e-4, epochs=4),
}

# Experiments executed by default (add "E5" for the optional partial fine-tuning run).
DEFAULT_RUN = ["E0", "E1", "E2", "E3", "E4"]
