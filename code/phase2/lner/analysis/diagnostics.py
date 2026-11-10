"""Diagnostics on a trained model's predictions and on the gold data (boundary deltas, propagation safety, counsel rule, caption parser, trim/bracket invariance)."""

import re
from collections import Counter, defaultdict

from .. import state as S
from ..eval.metrics import strict_scores
from ..decoding.spans import clean_span, balance_brackets
from ..postprocess.caption import parse_caption
from ..postprocess.counsel import counsel_side
from ..postprocess.edit_rules import shp


def boundary_deltas(
    texts,
    gold,
    pred,
    labels=("PREC", "STAT", "CASENO", "COURT", "JUDGE", "APP", "RESP"),
):
    """HOW are boundaries wrong? Text that the prediction has too much / too little, per label and side (digits shown as 0).

    Args:
        texts (dict): doc -> text.
        gold (dict): doc -> gold spans.
        pred (dict): doc -> predicted spans.
        labels (tuple[str]): labels to analyse.

    Returns:
        None
    """
    C = defaultdict(Counter)
    for d in pred:
        G, t = gold[d], texts[d]
        for s, e, l in set(pred[d]) - set(G):
            if l not in labels:
                continue
            g = next((g for g in G if g[2] == l and g[0] < e and s < g[1]), None)
            if not g:
                continue
            if e > g[1]:
                C[(l, "right: pred has extra")][shp(t[g[1] : e])] += 1
            elif e < g[1]:
                C[(l, "right: pred misses")][shp(t[e : g[1]])] += 1
            if s < g[0]:
                C[(l, "left: pred has extra")][shp(t[s : g[0]])] += 1
            elif s > g[0]:
                C[(l, "left: pred misses")][shp(t[g[0] : s])] += 1
    for k in sorted(C):
        print(k, sum(C[k].values()), C[k].most_common(6))


def propagation_diagnostics(vpred, vgold):
    """Propagation upper bound (missed gold with the same string predicted elsewhere), safety (fraction of annotated occurrences) and counsel-rule accuracy on gold.

    Args:
        vpred (dict): validation predictions.
        vgold (dict): validation gold.

    Returns:
        None
    """
    tot, rec = Counter(), Counter()
    for d in vpred:
        known = {(S.TEXTS[d][s:e], l) for s, e, l in set(vpred[d]) & set(vgold[d])}
        for s, e, l in set(vgold[d]) - set(vpred[d]):
            tot[l] += 1
            rec[l] += (S.TEXTS[d][s:e], l) in known
    for l in S.LABELS:
        print(
            f"{l:10s} missed={tot[l]:4d}  recoverable by exact-string propagation={rec[l]:4d} ({rec[l] / max(tot[l], 1):.0%})"
        )
    for l in S.LABELS:
        ann = n = 0
        for d in S.TEXTS:
            gs = {(s, e) for s, e, x in S.GOLD[d] if x == l}
            for st in {S.TEXTS[d][s:e] for s, e in gs if e - s >= 5}:
                for m in re.finditer(
                    r"(?<!\w)" + re.escape(st) + r"(?!\w)", S.TEXTS[d]
                ):
                    n += 1
                    ann += m.span() in gs
        print(
            f"{l:10s} annotated occurrences: {ann}/{n} = {ann / max(n, 1):.0%}"
        )  # >= ~90% -> propagate that label
    for l in ("A.COUNSEL", "R.COUNSEL"):
        sp = [(d, s, e) for d in S.TEXTS for s, e, x in S.GOLD[d] if x == l]
        pr = [counsel_side(S.TEXTS[d], s, e) for d, s, e in sp]
        fired = [p for p in pr if p]
        print(
            f"{l}: fires on {len(fired)}/{len(sp)}, correct when it fires: {sum(p == l for p in fired) / max(len(fired), 1):.1%}"
        )


def gold_changes(step):
    """How many CORRECT gold spans does a post-step destroy, per label?

    Args:
        step (Callable): ``step(text, s, e, label) -> (s, e)``.

    Returns:
        dict: label -> ``"changed/total"`` for labels with changes.
    """
    ch, tot = Counter(), Counter()
    for d in S.TEXTS:
        for s, e, l in S.GOLD[d]:
            tot[l] += 1
            if tuple(step(S.TEXTS[d], s, e, l)) != (s, e):
                ch[l] += 1
    return {l: f"{ch[l]}/{tot[l]}" for l in S.LABELS if ch[l]}


def trim_diagnostics(model, cfg, predict_raw, span_oracle=None):
    """Gold-invariance of TRIM_CHARS / balance_brackets, trim on/off A/B on the SAME model, span-oracle losses, counsel-rule misses.

    Args:
        model (nn.Module): trained model.
        cfg (dict): its config.
        predict_raw (Callable): ``predict_spans_raw(model, doc_ids, cfg)``.
        span_oracle (dict or None): output of ``run_selfchecks`` (``ids`` and ``pred``) for check 3.

    Returns:
        None
    """
    print(
        "TRIM_CHARS       :",
        gold_changes(lambda t, s, e, l: clean_span(t, s, e, S.TRIM_CHARS.get(l, ""))),
    )
    print(
        "balance_brackets :", gold_changes(lambda t, s, e, l: balance_brackets(t, s, e))
    )
    print("TRIM_CHARS table :", {l: S.TRIM_CHARS.get(l) for l in S.LABELS})
    VGOLD = {d: S.GOLD[d] for d in S.VAL_IDS}
    for trim in (True, False):
        pr = predict_raw(model, S.VAL_IDS, {**cfg, "TRIM_TRAILING_PUNCT": trim})
        sc = strict_scores(VGOLD, pr, S.LABELS)
        print(
            f"TRIM_TRAILING_PUNCT={trim!s:5}  F1={sc['macro']['f1']:.4f}  ",
            {
                l: round(sc["per_label"][l]["f1"], 3)
                for l in ("PREC", "STAT", "APP", "RESP", "JUDGE", "AUTH")
            },
        )
    if span_oracle:
        k = 0
        for d in span_oracle["ids"]:
            for s, e, l in sorted(set(S.GOLD[d]) - set(span_oracle["pred"][d])):
                if l not in ("PREC", "STAT") or k >= 25:
                    continue
                doc = S.DOCS[d]
                n_tok = int(
                    (
                        (doc["offs"][:, 0] >= s)
                        & (doc["offs"][:, 1] <= e)
                        & (doc["offs"][:, 1] > doc["offs"][:, 0])
                    ).sum()
                )
                ov = [
                    (S.TEXTS[d][a:b][:45], pl)
                    for a, b, pl in span_oracle["pred"][d]
                    if a < e and s < b
                ]
                print(
                    f"[{l}] gold={S.TEXTS[d][s:e][:70]!r} tokens={n_tok} maxlen={S.SPAN_MAXLEN.get(l)} -> decoder gave {ov}"
                )
                k += 1
    k = 0
    for d in S.TEXTS:
        for s, e, l in S.GOLD[d]:
            if (
                l in ("A.COUNSEL", "R.COUNSEL")
                and counsel_side(S.TEXTS[d], s, e) is None
                and k < 15
            ):
                print(
                    f"[{l}] ...{S.TEXTS[d][max(0, s - 90) : s]!r} <<{S.TEXTS[d][s:e]}>> {S.TEXTS[d][e : e + 60]!r}"
                )
                k += 1


def caption_diagnostic(ids=None, show=6):
    """How well does the caption parser match YOUR gold APP/RESP spans? (all labelled docs by default)

    Args:
        ids (list[str] or None): docs to check.
        show (int): number of mismatch samples to print.

    Returns:
        None
    """
    gold_head, ext, ok, nofound, samples = Counter(), Counter(), Counter(), 0, []
    for d in ids or list(S.TEXTS):
        P = parse_caption(S.TEXTS[d])
        G = {x for x in S.GOLD[d] if x[2] in ("APP", "RESP")}
        for s, e, l in G:
            gold_head[l] += s < 4500
        if not P:
            nofound += 1
            continue
        for lab, key in (("APP", "app"), ("RESP", "resp")):
            for s, e in P[key]:
                ext[lab] += 1
                if (s, e, lab) in G:
                    ok[lab] += 1
                elif len(samples) < show:
                    samples.append(
                        (
                            lab,
                            S.TEXTS[d][s:e],
                            [
                                S.TEXTS[d][a:b]
                                for a, b, x in G
                                if x == lab and a < e and s < b
                            ][:2],
                        )
                    )
    print(f"docs without a caption marker: {nofound}")
    for l in ("APP", "RESP"):
        print(
            f"{l}: gold spans in the first 4500 chars={gold_head[l]} | extracted={ext[l]} | exact matches={ok[l]} -> precision {ok[l] / max(ext[l], 1):.0%}, recall(head) {ok[l] / max(gold_head[l], 1):.0%}"
        )
    for lab, got, gold in samples:
        print(f"  [{lab}] extracted={got!r}  overlapping gold={gold}")
