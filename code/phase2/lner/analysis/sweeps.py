"""Inference-only ablations on a trained checkpoint (no retraining): window merging and window length/overlap."""

import torch

from .. import state as S
from ..decoding.decoders import decode_doc
from ..eval.metrics import strict_scores
from ..training.inference import predict_probs, predict_spans_raw
from ..training.trainer import load_trained


def merge_ablation(ckpt):
    """Strict macro-F1 for ``MERGE`` in {ramp, mean, max}.

    Args:
        ckpt (str): trained checkpoint.

    Returns:
        None
    """
    m, c = load_trained(ckpt)
    for mg in ("ramp", "mean", "max"):
        cc = {**c, "MERGE": mg}
        pr = predict_probs(m, [S.DOCS[d] for d in S.VAL_IDS], cc)
        pd_ = {d: decode_doc(S.DOCS[d], p, cc) for d, p in zip(S.VAL_IDS, pr)}
        print(
            f"MERGE={mg:5s} strict macro-F1 = {strict_scores({d: S.GOLD[d] for d in S.VAL_IDS}, pd_, S.LABELS)['macro']['f1']:.4f}"
        )
    del m
    torch.cuda.empty_cache()


def window_sweep(
    ckpt, grid=((256, 128), (384, 192), (512, 256), (768, 384), (1024, 512), (512, 128))
):
    """Does the decoder like longer inference windows? (raw decoding, no POST)

    Args:
        ckpt (str): trained checkpoint.
        grid (tuple[tuple[int, int]]): ``(MAX_LEN, OVERLAP)`` pairs.

    Returns:
        None
    """
    m, c = load_trained(ckpt)
    for L, ov in grid:
        p = predict_spans_raw(m, S.VAL_IDS, {**c, "MAX_LEN": L, "OVERLAP": ov})
        print(
            f"MAX_LEN={L:4d} OVERLAP={ov:3d}  strict macro-F1 = {strict_scores({d: S.GOLD[d] for d in S.VAL_IDS}, p, S.LABELS)['macro']['f1']:.4f}"
        )
    del m
    torch.cuda.empty_cache()
