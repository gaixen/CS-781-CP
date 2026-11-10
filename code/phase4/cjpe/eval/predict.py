"""Batched inference."""
import numpy as np
import torch

from ..context import device
from ..data.batching import make_batches, collate, to_device


@torch.no_grad()
def predict(model, exs, cfg, want_feats=False):
    """Run the model on examples (length-sorted batches, optional fp16 autocast).

    Args:
        model (CJPEModel): model to evaluate (set to eval mode).
        exs (list[dict]): tokenised examples.
        cfg (dict): experiment config (``BATCH_SIZE``, ``_amp`` and collate settings).
        want_feats (bool): also return pooled document features (for KD teachers).

    Returns:
        dict: ``prob`` (N,) P(class 1), ``logits`` (N, 2), ``feats`` ((N, d) float16 or None).
    """
    model.eval(); bs = cfg["BATCH_SIZE"] * 2; N = len(exs)
    prob, logits, feats = np.zeros(N), np.zeros((N, 2), np.float32), None
    for idxs in make_batches(exs, bs, False, None):
        batch = to_device(collate([exs[i] for i in idxs], cfg))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["_amp"]):
            lg, ft = model(batch)
        lg = lg.float().cpu().numpy(); logits[idxs] = lg; prob[idxs] = torch.softmax(torch.from_numpy(lg), -1)[:, 1].numpy()
        if want_feats:
            if feats is None: feats = np.zeros((N, ft.size(-1)), np.float16)
            feats[idxs] = ft.float().cpu().numpy().astype(np.float16)
    return dict(prob=prob, logits=logits, feats=feats)
