"""Windowed inference: merge overlapping windows, decode to spans, evaluate (post-processing from ``state.POST`` is applied)."""
import time

import numpy as np
import torch

from .. import state as S
from ..config import ARCH_INFO
from ..context import DEVICE
from ..data.alignment import make_windows
from ..data.samples import with_prefix, collate
from ..decoding.decoders import decode_doc
from ..eval.metrics import strict_scores
from ..postprocess.pipeline import postprocess_doc


def amp_settings(arch):
    """Mixed-precision settings.

    Args:
        arch (str): architecture key (T5 overflows in fp16 -> bf16 or fp32).

    Returns:
        tuple[bool, torch.dtype]: (use autocast, dtype).
    """
    if DEVICE.type != "cuda": return False, torch.float32
    bf16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
    if ARCH_INFO[arch]["kind"] == "t5": return bf16, torch.bfloat16
    return True, (torch.bfloat16 if bf16 else torch.float16)


@torch.no_grad()
def predict_probs(model, docs, cfg, batch_size=16):
    """Per-token probabilities for whole documents, merged over overlapping windows.

    ``MERGE``: ``ramp`` (mean, edge tokens down-weighted) | ``mean`` | ``max`` (most confident window per token).

    Args:
        model (nn.Module): NER model.
        docs (list[dict]): tokenised documents.
        cfg (dict): config (``MAX_LEN, OVERLAP, MERGE, ARCH, CTX_PREFIX``).
        batch_size (int): windows per batch.

    Returns:
        list[np.ndarray]: one ``(n_tokens, model.n_out)`` array per document.
    """
    model.eval(); use_amp, dt = amp_settings(cfg["ARCH"])
    C, merge, k = model.n_out, cfg.get("MERGE", "ramp"), max(1, cfg["OVERLAP"] // 2)
    nconf = len(S.TAGS) if model.task == "bio" else S.N_TYPE                     # columns that form a distribution (for 'max' merging)
    probs = [np.zeros((len(d["ids"]), C), np.float32) for d in docs]; wsum = [np.zeros(len(d["ids"]), np.float32) for d in docs]
    bestc = [np.full(len(d["ids"]), -1.0, np.float32) for d in docs]
    jobs = [(i, a, b) for i, d in enumerate(docs) for a, b in make_windows(len(d["ids"]), cfg["MAX_LEN"], cfg["MAX_LEN"] - cfg["OVERLAP"])]
    jobs.sort(key=lambda j: j[2] - j[1])
    for q in range(0, len(jobs), batch_size):
        chunk = jobs[q:q + batch_size]
        _items = [with_prefix(docs[i], a, b, cfg) for i, a, b in chunk]                    # same prefix as in training
        ids, am, _ = collate([dict(ids=x[0], tags=np.zeros(len(x[0]), dtype=np.int64)) for x in _items])
        with torch.autocast(device_type=DEVICE.type, dtype=dt, enabled=use_amp):
            p = model.to_probs(model(ids.to(DEVICE), am.to(DEVICE))).float().cpu().numpy()
        for (i, a, b), pj, (_, off) in zip(chunk, p, _items):
            L = b - a; pj = pj[off:off + L]
            if merge == "max":
                conf = pj[:, :nconf].max(-1); m = conf > bestc[i][a:b]
                probs[i][a:b][m] = pj[m]; bestc[i][a:b][m] = conf[m]; wsum[i][a:b] = 1.0
            else:
                pos = np.arange(L); w = np.minimum(np.minimum(pos + 1, L - pos), k) / k if merge == "ramp" else np.ones(L)
                probs[i][a:b] += pj * w[:, None]; wsum[i][a:b] += w
    return [p / np.maximum(w[:, None], 1e-6) for p, w in zip(probs, wsum)]


def predict_spans_raw(model, doc_ids, cfg):
    """Model + decoder, NO post-processing.

    Args:
        model (nn.Module): NER model.
        doc_ids (list[str]): documents to predict.
        cfg (dict): config.

    Returns:
        dict: doc -> ``[(s, e, label)]`` (end-exclusive).
    """
    docs = [S.DOCS[d] for d in doc_ids]; pr = predict_probs(model, docs, cfg)
    return {d: decode_doc(doc, p, cfg) for d, doc, p in zip(doc_ids, docs, pr)}


def predict_spans(model, doc_ids, cfg):
    """Model + decoder + post-processing with the active ``state.POST`` (empty POST = no-op).

    Args:
        model (nn.Module): NER model.
        doc_ids (list[str]): documents to predict.
        cfg (dict): config.

    Returns:
        dict: doc -> ``[(s, e, label)]``.
    """
    return {d: postprocess_doc(S.DOCS[d]["text"], v, S.POST) for d, v in predict_spans_raw(model, doc_ids, cfg).items()}


def evaluate(model, doc_ids, cfg):
    """Strict-score a model on labelled documents.

    Args:
        model (nn.Module): NER model.
        doc_ids (list[str]): labelled doc ids.
        cfg (dict): config.

    Returns:
        tuple: (``strict_scores`` dict, predictions, inference seconds).
    """
    t0 = time.time(); pred = predict_spans(model, doc_ids, cfg); sec = time.time() - t0
    return strict_scores({d: S.GOLD[d] for d in doc_ids}, pred, S.LABELS), pred, sec
