"""Shared state container (replaces the notebook's globals)."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from summ.config import Settings, build_experiments


@dataclass
class Context:
    """Everything shared by the stages.

    Args:
        s (Settings): Global settings.
        device (Any): ``torch.device``.
        experiments (dict): Experiment registry.
        tok (Any): GPT-2 tokenizer used in Phase 1.
        eos (Optional[int]): EOS id (also the SEP token between document and summary).
        vsz (int): Padded vocabulary size of the Phase-1 models (50304).
        nv (int): Real vocabulary size (padded logit columns are ignored when decoding).
        idf (Optional[np.ndarray]): Document-frequency based IDF per token id (for ``select``).
        stop (set): Most frequent token ids (ignored in the coverage analysis).
        train_all (list): Raw train rows.
        test_docs (list): Raw test rows.
        tr (Optional[dict]): Training split (``id, doc_ids, sum_ids, doc, ref``).
        dev (Optional[dict]): Dev split.
        te (Optional[dict]): Test split (``id, rid, doc_ids, doc``).
        inp_cache (dict): Cache of built model inputs.
        results (dict): Experiment results.
        paths (Optional[dict]): Local data paths.
        hf_token (Optional[str]): HuggingFace token.
    """
    s: Settings
    device: Any = None
    experiments: Dict[str, dict] = field(default_factory=dict)
    tok: Any = None
    eos: Optional[int] = None
    vsz: int = 50304
    nv: int = 0
    idf: Optional[np.ndarray] = None
    stop: set = field(default_factory=set)
    train_all: list = field(default_factory=list)
    test_docs: list = field(default_factory=list)
    tr: Optional[dict] = None
    dev: Optional[dict] = None
    te: Optional[dict] = None
    inp_cache: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    paths: Optional[dict] = None
    hf_token: Optional[str] = None

    def __post_init__(self):
        """Fill the experiment registry.

        Args:
            None

        Returns:
            None
        """
        if not self.experiments:
            self.experiments = build_experiments(self.s)
