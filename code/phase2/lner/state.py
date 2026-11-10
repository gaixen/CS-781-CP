"""Shared, mutable dataset/runtime state (filled by ``lner.data.dataset.init_dataset``).

All other modules read these attributes dynamically (``from lner import state as S; S.LABELS``), so the notebook's
module-level globals are preserved without circular imports.

Attributes:
    LABELS (list[str]): entity labels (no BIO prefix).
    TAGS (list[str]): ``["O", "B-X", "I-X", ...]``; TAG2ID (dict): tag -> id; N_TYPE (int): 1 + len(LABELS).
    TEXTS (dict): doc_id -> text (fold1 + fold2); GOLD (dict): doc_id -> [(start, end_exclusive, label)].
    GOLD_RAW (dict): gold exactly as stored; DOC_FOLD (dict): doc_id -> "fold1" | "fold2"; END_INCLUSIVE (bool).
    TRAIN_IDS / VAL_IDS (list[str]): train / validation doc ids.
    TRIM_CHARS (dict): label -> punctuation characters that may be trimmed from span ends.
    BOUNDARY_PRIOR (dict): label -> char-class priors for boundary refinement.
    SPAN_MAXLEN (dict): label -> max span length in tokens (span head).
    DOCS (dict): doc_id -> tokenised document (ids, offs, tags, text).
    POST (dict): active post-processing configuration (see ``lner.postprocess.pipeline.postprocess_doc``).
    RESULTS (list[dict]): experiment records. tokenizer / PAD_ID: GPT-2 tokenizer and its padding id.
"""
LABELS, TAGS, TAG2ID, N_TYPE = [], [], {}, 0
TEXTS, GOLD_RAW, DOC_FOLD, GOLD = {}, {}, {}, {}
END_INCLUSIVE = False
TRAIN_IDS, VAL_IDS = [], []
TRIM_CHARS, BOUNDARY_PRIOR, SPAN_MAXLEN = {}, {}, {}
DOCS = {}
POST = {}
RESULTS = []
tokenizer = None
PAD_ID = None
