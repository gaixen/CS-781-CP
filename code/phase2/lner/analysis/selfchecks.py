"""Self-checks that need no training: span-target construction, span-decoder oracle, boundary refinement."""

import numpy as np
import torch

from .. import state as S
from ..config import CFG
from ..data.samples import build_train_samples, collate
from ..decoding.boundary import refine_span
from ..decoding.decoders import decode_doc
from ..eval.metrics import strict_scores
from ..training.losses import span_targets


def span_targets_reference(lab):
    """Slow python reference implementation of ``span_targets`` for one sequence.

    Args:
        lab (list[int]): BIO tag ids (-100 = ignore).

    Returns:
        tuple[list, list, list]: type, start, end targets.
    """
    L = len(lab)
    typ, st, en = [-100] * L, [-100] * L, [-100] * L
    for i in range(L):
        if lab[i] < 0:
            continue
        typ[i] = (lab[i] + 1) // 2
        st[i] = int(lab[i] > 0 and lab[i] % 2 == 1)
        j = next((k for k in range(i + 1, L) if lab[k] >= 0), None)
        if j is None:
            continue
        c = typ[i]
        en[i] = int(c > 0 and not (lab[j] == 2 * c))
    return typ, st, en


def oracle_span_probs(doc):
    """Gold-derived near-one-hot span-head probabilities (used to test the span decoder).

    Args:
        doc (dict): labelled tokenised document.

    Returns:
        np.ndarray: (n_tokens, N_TYPE + 2) probabilities.
    """
    N_TYPE = S.N_TYPE
    typ, st, en = [
        x[0].numpy() for x in span_targets(torch.from_numpy(doc["tags"])[None])
    ]
    typ, st, en = (
        np.where(typ < 0, 0, typ),
        np.where(st < 0, 0, st),
        np.where(en < 0, 0, en),
    )
    if typ[-1] > 0:
        en[-1] = 1
    p = np.full((len(typ), N_TYPE + 2), 1e-3)
    p[np.arange(len(typ)), typ] = 0.98
    p[:, N_TYPE] = np.where(st == 1, 0.98, 0.01)
    p[:, N_TYPE + 1] = np.where(en == 1, 0.98, 0.01)
    return p


def run_selfchecks():
    """Run the three self-checks and print a verdict line for each.

    Args:
        None

    Returns:
        dict: ``{"ids": doc ids used for the span oracle, "pred": span-oracle predictions}`` (reused by diagnostics).
    """
    _s = build_train_samples(S.TRAIN_IDS[:6], {**CFG, "NEG_WINDOW_KEEP": 1.0})[:64]
    _, _, _lab = collate(_s)
    _t = [x.tolist() for x in span_targets(_lab)]
    for bi, smp in enumerate(_s):
        ref = span_targets_reference(smp["tags"].tolist())
        n = len(smp["tags"])
        for a, b in zip((x[bi][:n] for x in _t), ref):
            assert a == b, "span_targets mismatch"
    print("1) span_targets == python reference on", len(_s), "windows")

    _ids = list(S.TEXTS)[:100]
    _cfg = {**CFG, "TASK": "span", "BOUNDARY_REFINE": False, "CONF_THRESH": None}
    _pred = {d: decode_doc(S.DOCS[d], oracle_span_probs(S.DOCS[d]), _cfg) for d in _ids}
    print(
        f"2) span-decoder ORACLE strict macro-F1 = {strict_scores({d: S.GOLD[d] for d in _ids}, _pred, S.LABELS)['macro']['f1']:.4f}   (BIO oracle: ~0.986)"
    )

    _rc = {**CFG, "REFINE_K": CFG["REFINE_K"], "REFINE_LAMBDA": CFG["REFINE_LAMBDA"]}
    for nm, ids in (("train", S.TRAIN_IDS), ("val", S.VAL_IDS)):
        ch = sum(
            refine_span(S.TEXTS[d], s, e, l, _rc) != (s, e)
            for d in ids
            for s, e, l in S.GOLD[d]
        )
        tot = sum(len(S.GOLD[d]) for d in ids)
        print(
            f"3a) refine changes {ch}/{tot} ({100 * ch / max(tot, 1):.2f}%) of correct GOLD spans in {nm}  (want ~0%)"
        )

    def _recovery(pert, ids):
        ok = n = 0
        for d in ids:
            t = S.TEXTS[d]
            for s, e, l in S.GOLD[d]:
                p = pert(t, s, e)
                if p is None:
                    continue
                n += 1
                ok += refine_span(t, p[0], p[1], l, _rc) == (s, e)
        return ok, n

    for nm, f in (
        (
            "trailing punctuation included (+1 char)",
            lambda t, s, e: (s, e + 1) if e < len(t) and t[e] in ".,;:" else None,
        ),
        (
            "end cut inside a word (-1 char)",
            lambda t, s, e: (
                (s, e - 1)
                if e - s > 3 and t[e - 1].isalpha() and t[e - 2].isalpha()
                else None
            ),
        ),
        (
            "start cut inside a word (+1 char)",
            lambda t, s, e: (
                (s + 1, e)
                if e - s > 3 and t[s].isalpha() and t[s + 1].isalpha()
                else None
            ),
        ),
    ):
        ok, n = _recovery(f, S.VAL_IDS)
        print(f"3b) repaired {ok}/{n} synthetic errors: {nm}")
    return {"ids": _ids, "pred": _pred}
