"""Combined post-processing (final version of the notebook's ``postprocess_doc``) and its evaluation helper."""
import numpy as np

from .. import state as S
from ..decoding.boundary import refine_span
from ..eval.metrics import strict_scores
from .caption import parse_caption, caption_apply, counsel_party_relabel
from .counsel import counsel_relabel
from .edit_rules import apply_rule, propagate
from .filters import unify_labels
from .period import attach_period
from .roles import roles_relabel


def postprocess_doc(text, spans, post):
    """Apply the enabled post-processing components to one document.

    Args:
        text (str): document text.
        spans (list[tuple]): ``(s, e, label)``.
        post (dict): components: ``rules`` (edit rules), ``period`` (table), ``refine`` (labels), ``counsel`` (True | 'vote'),
            ``counsel_party`` (bool), ``roles`` (['cue','caption']), ``blacklist`` (set), ``capspans`` ('add'|'prop'),
            ``unify`` (labels), ``propagate`` (labels). An empty dict is the identity (up to sorting/dedup).

    Returns:
        list[tuple]: sorted, de-duplicated, non-empty spans.
    """
    out = []; rc = {"REFINE_K": 3, "REFINE_LAMBDA": 1.5}
    for s, e, l in spans:
        for r in post.get("rules", ()):
            if r[0] == l: s, e = apply_rule(text, s, e, r)
        if post.get("period"): s, e = attach_period(text, s, e, l, post["period"])
        if l in post.get("refine", ()): s, e = refine_span(text, s, e, l, rc)
        out.append((s, e, l))
    P = parse_caption(text) if (post.get("counsel_party") or post.get("capspans")) else None
    if post.get("counsel"): out = counsel_relabel(text, out, vote=(post["counsel"] == "vote"))
    if post.get("counsel_party") and P: out = counsel_party_relabel(text, out, P)
    if post.get("roles"): out = roles_relabel(text, out, post["roles"])
    if post.get("blacklist"): out = [x for x in out if (x[2], text[x[0]:x[1]].strip().lower()) not in post["blacklist"]]
    if post.get("capspans") and P: out = caption_apply(text, out, post["capspans"], P)
    if post.get("unify"): out = unify_labels(text, out, set(post["unify"]))
    if post.get("propagate"): out = propagate(text, out, set(post["propagate"]))
    return sorted(set(x for x in out if x[1] > x[0]))


def f1_with(post, pred, ids):
    """Strict macro-F1 of raw predictions after applying ``post``.

    Args:
        post (dict): post-processing configuration.
        pred (dict): doc -> raw predicted spans.
        ids (list[str]): docs to score.

    Returns:
        float: macro-F1 rounded to 4 decimals.
    """
    out = {d: postprocess_doc(S.DOCS[d]["text"], pred[d], post) for d in ids}
    return round(strict_scores({d: S.GOLD[d] for d in ids}, out, S.LABELS)["macro"]["f1"], 4)
