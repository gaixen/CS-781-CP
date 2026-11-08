"""Shared state container (replaces the notebook's global variables)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from lsi.config import Config, build_experiments


@dataclass
class LSIData:
    """Cleaned documents and multi-hot label matrices.

    Args:
        train_docs (List[dict]): Train documents (``rid, id, text, labels, had_labels``).
        dev_docs (List[dict]): Cleaned dev documents.
        test_docs (List[dict]): Test documents (labels are always empty).
        labels (List[int]): Sorted statute ids.
        l2i (Dict[int, int]): Statute id -> column index.
        ytr (np.ndarray): Train label matrix (n_train, L).
        ydev (np.ndarray): Dev label matrix (n_dev, L).
        train_freq (np.ndarray): Train frequency of each statute.
        test_ids (set): Test document ids (for leak checks).
        paths (Dict[str, str]): Local data file paths.
    """
    train_docs: List[dict]
    dev_docs: List[dict]
    test_docs: List[dict]
    labels: List[int]
    l2i: Dict[int, int]
    ytr: np.ndarray
    ydev: np.ndarray
    train_freq: np.ndarray
    test_ids: set
    paths: Dict[str, str]

    @property
    def n_labels(self) -> int:
        """Number of statutes.

        Args:
            None

        Returns:
            int: Label-space size ``L``.
        """
        return len(self.labels)


@dataclass
class Context:
    """Everything shared by the pipeline stages.

    Args:
        cfg (Config): Configuration.
        device (Any): ``torch.device``.
        data (Optional[LSIData]): Cleaned data.
        experiments (dict): Experiment registry.
        stores (dict): Cached token stores per architecture.
        tokenizers (dict): Cached tokenizers per checkpoint path.
        avail (list): Experiments with usable logits.
        zd (dict): Dev logits per experiment.
        zt (dict): Test logits per experiment.
        temp (dict): Calibration temperature per experiment.
        diag (Any): Diagnostics DataFrame.
        submit_zips (dict): Metadata of every written zip.
        dup_test (dict): Test documents having an exact / near twin in train or dev.
    """
    cfg: Config
    device: Any = None
    data: Optional[LSIData] = None
    experiments: Dict[str, dict] = field(default_factory=dict)
    stores: Dict[str, dict] = field(default_factory=dict)
    tokenizers: Dict[str, Any] = field(default_factory=dict)
    avail: List[str] = field(default_factory=list)
    zd: Dict[str, np.ndarray] = field(default_factory=dict)
    zt: Dict[str, np.ndarray] = field(default_factory=dict)
    temp: Dict[str, float] = field(default_factory=dict)
    diag: Any = None
    submit_zips: Dict[str, dict] = field(default_factory=dict)
    dup_test: Dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        """Fill the experiment registry from the configuration.

        Args:
            None

        Returns:
            None
        """
        if not self.experiments:
            self.experiments = build_experiments(self.cfg)
