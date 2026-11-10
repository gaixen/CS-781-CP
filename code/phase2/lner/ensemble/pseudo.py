"""Rule-distilled pseudo-labels: the post-processed ensemble labels documents WITHOUT gold; the model is retrained on the confident part.

The rules (counsel side, period attachment, caption parties ...) normally exist only at inference; training on their output
moves that behaviour into the weights.
"""
import numpy as np
import torch

from .. import state as S
from ..data.alignment import encode_tags
from ..decoding.decoders import decode_scored
from ..postprocess.pipeline import postprocess_doc
from ..training.trainer import run_experiment
from .ensembling import ens_probs


def make_pseudo_docs(ckpts, post, doc_ids, weights=None, tag="PL", min_conf=0.8, conf_o=0.97):
    """Create pseudo-labelled copies of documents in ``state.DOCS``.

    A span is kept if its decoder confidence >= ``min_conf``; an 'O' token is kept only if the ensemble's P(O) >= ``conf_o``;
    everything else is -100.

    Args:
        ckpts (list[str]): teacher checkpoints.
        post (dict): post-processing configuration applied to the ensemble's spans.
        doc_ids (list[str]): documents to pseudo-label.
        weights (list[float] or None): ensemble weights.
        tag (str): prefix of the new doc ids (``<tag>_<doc_id>``).
        min_conf (float): span confidence threshold.
        conf_o (float): 'O' confidence threshold.

    Returns:
        list[str]: new doc ids.
    """
    probs, cfg0 = ens_probs(ckpts, doc_ids, weights); dcfg = {**cfg0, "CONF_THRESH": None, "BOUNDARY_REFINE": False}; new_ids, n_sp, n_tok, n_keep = [], 0, 0, 0
    for d, p in zip(doc_ids, probs):
        doc = S.DOCS[d]; spans = [(s, e, l) for s, e, l, c in decode_scored(doc, p, dcfg) if c >= min_conf]
        if post: spans = postprocess_doc(doc["text"], spans, post)
        tags, _ = encode_tags(doc["offs"], spans); tags = np.where((tags == 0) & (p[:, 0] < conf_o), -100, tags)      # column 0 = 'O' in both layouts
        S.DOCS[f"{tag}_{d}"] = {**doc, "tags": tags}; new_ids.append(f"{tag}_{d}"); n_sp += len(spans); n_tok += int((tags != -100).sum()); n_keep += len(tags)
    print(f"pseudo-labelled {len(new_ids)} docs: {n_sp} spans, {n_tok}/{n_keep} tokens used as targets ({n_tok / max(n_keep, 1):.0%})")
    return new_ids


def run_pseudo_fit(name, ckpt_path, pseudo_ids, **overrides):
    """Validated config + best epoch, trained on gold (fold1+fold2) PLUS the pseudo-labelled docs.

    Args:
        name (str): experiment name.
        ckpt_path (str): validated checkpoint (config + best epoch reused).
        pseudo_ids (list[str]): pseudo-labelled doc ids.
        **overrides: config overrides.

    Returns:
        tuple: result of ``run_experiment``.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False); cfg0 = dict(ck["cfg"]); cfg0["EPOCHS"] = ck.get("best_epoch", cfg0["EPOCHS"])
    return run_experiment(name, train_ids=list(S.TEXTS) + list(pseudo_ids), val_ids=[], **{**cfg0, **overrides})
