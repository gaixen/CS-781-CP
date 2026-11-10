"""Shared state (replaces the notebook's globals): data, tokenizer, results registry and the log-prior used by logit adjustment."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rr.config import Settings, default_cfg


@dataclass
class RRData:
    """Documents and label vocabulary.

    Args:
        labels (List[str]): Label names (order = class index).
        l2i (Dict[str, int]): Label -> index.
        splits (Dict[tuple, List[dict]]): ``(domain, split) -> documents``.
        train (List[dict]): CL + IT train documents.
        dev (List[dict]): CL + IT dev documents.
        test (List[dict]): CL + IT test documents.
    """
    labels: List[str]
    l2i: Dict[str, int]
    splits: Dict[tuple, List[dict]]
    train: List[dict]
    dev: List[dict]
    test: List[dict]

    @property
    def nl(self) -> int:
        """Number of labels.

        Args:
            None

        Returns:
            int: Size of the label set.
        """
        return len(self.labels)


@dataclass
class Context:
    """Everything shared across stages.

    Args:
        settings (Settings): Global settings.
        device (Any): ``torch.device``.
        cfg (dict): Default experiment config (``CFG``).
        data (Optional[RRData]): Loaded data.
        tokenizer (Any): GPT-2 tokenizer.
        pad_id (Optional[int]): EOS id (also the context separator).
        dom_ids (Optional[dict]): Token ids of the ``[CL]`` / ``[IT]`` domain prefixes.
        results (list): Records of every trained / synced run.
        log_prior (Any): ``log`` class prior (torch tensor) set before training for logit adjustment.
        state (dict): Free-form storage for stage outputs (best names, bias, weights, ...).
    """
    settings: Settings
    device: Any = None
    cfg: dict = field(default_factory=default_cfg)
    data: Optional[RRData] = None
    tokenizer: Any = None
    pad_id: Optional[int] = None
    dom_ids: Optional[dict] = None
    results: List[dict] = field(default_factory=list)
    log_prior: Any = None
    state: Dict[str, Any] = field(default_factory=dict)

    @property
    def work(self) -> str:
        """Working directory.

        Args:
            None

        Returns:
            str: ``settings.work``.
        """
        return self.settings.work
