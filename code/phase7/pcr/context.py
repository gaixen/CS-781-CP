from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pcr.config import Config, EXPERIMENTS


@dataclass
class PCRContext:
    """Everything the stages of the pipeline need to share.

    Args:
        cfg (Config): Pipeline configuration.
        device (Any): ``torch.device`` used for all computation.
        experiments (dict): Experiment registry (defaults to ``pcr.config.EXPERIMENTS``).
        paths (Optional[dict]): Local paths of the six data files.
        tok (Any): Phase-1 tokenizer.
        eos (Optional[int]): EOS / padding token id.
        raw (Optional[dict]): Preprocessed raw records per split / pool.
        meta (Optional[dict]): Parsed header metadata per split / pool.
        spl (Optional[dict]): Tokenised split containers (queries, pools, gold).
        lex (dict): Lexical score matrices ``lex[name][split]`` (bm25, tfidf).
        lex2 (dict): Lexical suite channels ``lex2[name][split]``.
        rf (dict): Rule features per split (chronology / citation / case-name).
        ex (dict): Extra header features per split (forward / reverse citation, soft year).
        train_data (Any): :class:`pcr.training.trainer.TrainData` with hard negatives.
        results (dict): Experiment results ``eid -> result dict``.
    """

    cfg: Config
    device: Any = None
    experiments: Dict[str, dict] = field(default_factory=lambda: dict(EXPERIMENTS))
    paths: Optional[Dict[str, str]] = None
    tok: Any = None
    eos: Optional[int] = None
    raw: Optional[dict] = None
    meta: Optional[dict] = None
    spl: Optional[dict] = None
    lex: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lex2: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rf: Dict[str, dict] = field(default_factory=dict)
    ex: Dict[str, dict] = field(default_factory=dict)
    train_data: Any = None
    results: Dict[str, dict] = field(default_factory=dict)
