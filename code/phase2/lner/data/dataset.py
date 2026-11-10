"""Dataset initialisation: labels, folds, offset convention, splits, tokenisation, learned decoder priors."""
import re
from collections import Counter

import numpy as np

from .. import state as S
from ..config import CFG
from ..context import setup_hf_auth
from .io import load_json, locate, parse_gold, detect_end_inclusive, split_docs
from .alignment import get_tokenizer, tokenize_all, make_windows  # noqa: F401
from ..decoding.spans import decode_spans, clean_span, learn_trim_chars
from ..decoding.boundary import build_boundary_prior
from ..decoding.decoders import compute_span_maxlen
from ..eval.metrics import strict_scores


def oracle_check(ids, show=5):
    """ORACLE round-trip (gold -> BIO tags -> char spans): the strict-F1 ceiling imposed by tokenisation + BIO.

    Args:
        ids (list[str]): doc ids to check.
        show (int): number of unrecoverable gold entities to print.

    Returns:
        dict: ``strict_scores`` output.
    """
    pred, misses = {}, []
    for d in ids:
        doc = S.DOCS[d]; tags = np.where(doc["tags"] < 0, 0, doc["tags"]); keep = doc["offs"][:, 1] > doc["offs"][:, 0]
        pred[d] = [(*clean_span(doc["text"], s, e), l) for s, e, l in decode_spans(tags[keep], doc["offs"][keep])]
        miss = set(S.GOLD[d]) - set(pred[d])
        misses += [(d, s, e, l) for s, e, l in sorted(miss)]
    sc = strict_scores({d: S.GOLD[d] for d in ids}, pred, S.LABELS)
    print(f"ORACLE strict macro-F1 = {sc['macro']['f1']:.4f}  | unrecoverable gold entities: {len(misses)}")
    for d, s, e, l in misses[:show]:
        doc = S.DOCS[d]; print(f"  [{l}] gold={doc['text'][s:e]!r}  context={doc['text'][max(0,s-15):e+15]!r}")
    return sc


def init_dataset(cfg=None, use_presplit=True, run_oracle=True, verbose=True, auth=True):
    """Load folds + labels, detect the offset convention, build splits, tokenise, learn trim chars / boundary priors / span max lengths.

    Fills ``lner.state`` (LABELS, TAGS, TEXTS, GOLD, TRAIN_IDS, VAL_IDS, DOCS, TRIM_CHARS, BOUNDARY_PRIOR, SPAN_MAXLEN ...).

    Args:
        cfg (dict or None): config (``VAL_MODE``, ``VAL_FRAC``, ``MAX_LEN``); defaults to ``CFG``.
        use_presplit (bool): punctuation pre-split tokenisation (recommended).
        run_oracle (bool): print the oracle ceiling on the first 300 docs.
        verbose (bool): print statistics.
        auth (bool): log in to the HF Hub first.

    Returns:
        None
    """
    cfg = cfg or CFG
    if auth: setup_hf_auth()
    raw_labels = [l.strip() for l in open(locate("labels.txt"), encoding="utf-8") if l.strip()]
    labels = []
    for l in raw_labels:
        b = re.sub(r"^[BIE]-", "", l)
        if b != "O" and b not in labels: labels.append(b)
    S.LABELS = labels; S.TAGS = ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]
    S.TAG2ID = {t: i for i, t in enumerate(S.TAGS)}; S.N_TYPE = len(labels) + 1       # type head: 0 = O, 1..12 = LABELS[c-1]
    if verbose: print(len(labels), "labels:", labels, "| tags:", len(S.TAGS))

    S.TEXTS, S.GOLD_RAW, S.DOC_FOLD = {}, {}, {}
    for fold in ("fold1", "fold2"):
        t = load_json(locate(f"{fold}-text.json")); g = parse_gold(load_json(locate(f"{fold}-gold.json")), labels)
        assert not (set(t) & set(S.TEXTS)), "doc-id collision between folds"
        S.TEXTS.update({str(k): v for k, v in t.items()}); S.GOLD_RAW.update(g)
        for k in t: S.DOC_FOLD[str(k)] = fold
        if verbose: print(fold, "docs:", len(t), "| entities:", sum(len(v) for k, v in g.items() if k in t))
    missing = [d for d in S.TEXTS if d not in S.GOLD_RAW]
    if missing and verbose: print(f"{len(missing)} docs have no gold entry -> treated as 'no entities'")
    S.END_INCLUSIVE = detect_end_inclusive(S.TEXTS, S.GOLD_RAW, verbose)
    if verbose: print("END_INCLUSIVE =", S.END_INCLUSIVE, "(internally everything is end-EXCLUSIVE; converted back when writing predictions)")
    inc = S.END_INCLUSIVE
    S.GOLD = {d: [(s, e + 1 if inc else e, l) for s, e, l, _ in S.GOLD_RAW.get(d, [])] for d in S.TEXTS}
    bad = [(d, s, e, t, S.TEXTS[d][s:e]) for d, ents in S.GOLD_RAW.items() if d in S.TEXTS
           for s, e, l, t in [(a, b + 1 if inc else b, c, x) for a, b, c, x in ents] if t is not None and S.TEXTS[d][s:e] != t]
    cnt = Counter(l for ents in S.GOLD.values() for _, _, l in ents)
    if verbose: print("entities whose stored text != text[s:e]:", len(bad), bad[:3]); print("label counts:", dict(cnt))

    if cfg["VAL_MODE"] == "fold2":
        S.TRAIN_IDS = [d for d in S.TEXTS if S.DOC_FOLD[d] == "fold1"]; S.VAL_IDS = [d for d in S.TEXTS if S.DOC_FOLD[d] == "fold2"]
    else:
        S.TRAIN_IDS, S.VAL_IDS = split_docs(list(S.TEXTS), cfg["VAL_FRAC"])
    if verbose: print("train docs:", len(S.TRAIN_IDS), "| val docs:", len(S.VAL_IDS))

    S.TRIM_CHARS = learn_trim_chars(S.TRAIN_IDS); S.BOUNDARY_PRIOR = build_boundary_prior(S.TRAIN_IDS)
    if verbose: print("TRIM_CHARS:", S.TRIM_CHARS)
    get_tokenizer(); tokenize_all(cfg["MAX_LEN"], use_presplit, verbose)
    S.SPAN_MAXLEN = compute_span_maxlen(S.TRAIN_IDS)
    if verbose: print("max span length (tokens, with margin):", S.SPAN_MAXLEN)
    if run_oracle: oracle_check(list(S.TEXTS)[:300])      # should be ~1.0; if not, boundaries fall inside BPE tokens
