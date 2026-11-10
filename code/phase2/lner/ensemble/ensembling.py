"""Average TOKEN PROBABILITIES of several checkpoints (same TASK, same tokenisation) BEFORE decoding."""
import os

import torch

from .. import state as S
from ..decoding.decoders import decode_doc
from ..eval.metrics import strict_scores
from ..postprocess.pipeline import postprocess_doc
from ..training.inference import predict_probs
from ..training.trainer import load_trained


def ens_probs(ckpts, doc_ids, weights=None):
    """Weighted mean of per-token probabilities.

    Args:
        ckpts (list[str]): checkpoint paths.
        doc_ids (list[str]): documents (must be in ``state.DOCS``).
        weights (list[float] or None): model weights (default equal).

    Returns:
        tuple[list[np.ndarray], dict]: (ensembled probabilities per doc, config of the first checkpoint).

    Raises:
        AssertionError: if span and BIO models are mixed.
    """
    weights = weights or [1.0] * len(ckpts); docs = [S.DOCS[d] for d in doc_ids]; acc = cfg0 = None
    for w, path in zip(weights, ckpts):
        m, c = load_trained(path); cfg0 = cfg0 or c; assert c["TASK"] == cfg0["TASK"], "cannot average span and BIO models"
        pr = predict_probs(m, docs, c); acc = [w * p for p in pr] if acc is None else [a + w * p for a, p in zip(acc, pr)]
        del m; torch.cuda.empty_cache(); print(f"  + {os.path.basename(path)} (w={w})")
    return [a / float(sum(weights)) for a in acc], cfg0


def ens_score(ckpts, doc_ids, weights=None, post=None):
    """Strict macro-F1 of an ensemble (per-model confidence thresholds are not used).

    Args:
        ckpts (list[str]): checkpoint paths.
        doc_ids (list[str]): labelled doc ids.
        weights (list[float] or None): model weights.
        post (dict or None): post-processing configuration.

    Returns:
        float: strict macro-F1.
    """
    probs, cfg0 = ens_probs(ckpts, doc_ids, weights); dcfg = {**cfg0, "CONF_THRESH": None, "BOUNDARY_REFINE": False}
    pred = {d: decode_doc(S.DOCS[d], p, dcfg) for d, p in zip(doc_ids, probs)}
    if post: pred = {d: postprocess_doc(S.DOCS[d]["text"], v, post) for d, v in pred.items()}
    return strict_scores({d: S.GOLD[d] for d in doc_ids}, pred, S.LABELS)["macro"]["f1"]


def compare_ensembles(modern_ckpt, lsw_ckpt, post=None):
    """Print fold2 F1 of modern-only / lsw-only / 1:1 / 2:1 ensembles.

    Args:
        modern_ckpt (str): modern-decoder checkpoint.
        lsw_ckpt (str): sliding-window checkpoint.
        post (dict or None): post-processing configuration.

    Returns:
        None
    """
    cks = [modern_ckpt, lsw_ckpt]
    print("fold2 strict macro-F1 (models trained on fold1 only):")
    for nm, c_, w_ in (("modern only", cks[:1], None), ("lsw only", cks[1:], None), ("modern+lsw (1:1)", cks, None), ("modern+lsw (2:1)", cks, [2.0, 1.0])):
        print(f"  {nm:20s} raw {ens_score(c_, S.VAL_IDS, w_):.4f} | +POST {ens_score(c_, S.VAL_IDS, w_, post or {}):.4f}")
