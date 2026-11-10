"""Test-set prediction, submission writing/verification (``prediction.json`` in the gold files' own offset convention) and zip packaging."""
import json
import math
import os
import zipfile
from collections import Counter

import torch

from . import state as S
from .config import WORK
from .data.alignment import build_doc
from .data.io import load_json, locate
from .decoding.decoders import decode_doc
from .ensemble.ensembling import ens_probs
from .postprocess.pipeline import postprocess_doc
from .training.inference import predict_spans_raw


def load_test_docs():
    """Load ``fold3-text.json`` and tokenise it into ``state.DOCS``.

    Args:
        None

    Returns:
        dict: doc_id -> test text.
    """
    test_texts = {str(k): v for k, v in load_json(locate("fold3-text.json")).items()}
    for d, t in test_texts.items(): S.DOCS[d] = build_doc(d, t)
    return test_texts


def write_predictions(preds, texts, path, include_text=False):
    """Write predictions in the gold files' offset convention and re-read them to validate.

    Args:
        preds (dict): doc -> ``[(s, e_exclusive, label)]``.
        texts (dict): doc -> text.
        path (str): output ``prediction.json`` path.
        include_text (bool): append the span text to every row.

    Returns:
        None
    """
    out = {}
    for d, ents in preds.items():
        out[d] = []
        for s, e, l in sorted(ents):
            assert 0 <= s < e <= len(texts[d]) and l in S.LABELS, (d, s, e, l)
            row = [int(s), int(e - 1 if S.END_INCLUSIVE else e), l]
            if include_text: row.append(texts[d][s:e])
            out[d].append(row)
    json.dump(out, open(path, "w"), ensure_ascii=False)
    back, ws = load_json(path), 0
    for d, rows in back.items():
        for r in rows:
            s, e = r[0], r[1] + 1 if S.END_INCLUSIVE else r[1]
            assert 0 <= s < e <= len(texts[d]) and r[2] in S.LABELS, (d, r)
            ws += texts[d][s:e].strip() != texts[d][s:e]
    print(f"wrote {path}: {len(back)} docs, {sum(len(v) for v in back.values())} entities | spans with edge whitespace: {ws}")


def verify_submission(pj, test_texts, lo=0.4, hi=2.5):
    """Structure + sanity checks on a prediction.json: all test ids, valid offsets/labels, duplicates, overlaps, per-label density.

    Args:
        pj (str): path of ``prediction.json``.
        test_texts (dict): doc -> test text.
        lo (float): lower bound for the predicted/gold density ratio.
        hi (float): upper bound for the predicted/gold density ratio.

    Returns:
        bool: True if the structure is valid.
    """
    sub = load_json(pj); miss, extra = set(test_texts) - set(sub), set(sub) - set(test_texts)
    print(f"[verify] docs: {len(sub)} (missing {len(miss)}, unknown {len(extra)})")
    bad = dup = ovl = 0; cnt, chars = Counter(), 0
    for d, rows in sub.items():
        t = test_texts.get(d)
        if t is None: continue
        seen, sp = set(), []; chars += len(t)
        for r in rows:
            try: s, e, l = int(r[0]), int(r[1]), r[2]; ee = e + 1 if S.END_INCLUSIVE else e
            except Exception: bad += 1; continue
            if not (0 <= s < ee <= len(t)) or l not in S.LABELS: bad += 1; continue
            if (s, e, l) in seen: dup += 1; continue
            seen.add((s, e, l)); sp.append((s, ee)); cnt[l] += 1
        sp.sort(); ovl += sum(b[0] < a[1] for a, b in zip(sp, sp[1:]))
    print(f"[verify] invalid rows {bad} | duplicates {dup} | overlapping span pairs {ovl} | {'OK' if not (miss or extra or bad) else 'PROBLEM -- do not submit'}")
    g = Counter(l for d in S.TEXTS for _, _, l in S.GOLD[d]); gc = sum(len(t) for t in S.TEXTS.values())
    for l in S.LABELS:
        pd_, gd = 1e5 * cnt[l] / max(chars, 1), 1e5 * g[l] / gc; r = pd_ / max(gd, 1e-9)
        print(f"[verify] {l:10s} pred/100k chars {pd_:7.1f} | train-gold {gd:7.1f} | ratio {r:4.2f} {'<-- check' if not lo <= r <= hi else ''}")
    return not (miss or extra or bad)


def _finish_submission(preds, raw, test_texts, name, include_text):
    """Write ``prediction.json`` + ``submission.zip`` under ``<WORK>/<name>/`` and verify.

    Args:
        preds (dict): final predictions.
        raw (dict or None): predictions before post-processing (for the change report).
        test_texts (dict): doc -> text.
        name (str): submission folder name.
        include_text (bool): include span text in rows.

    Returns:
        dict: ``preds``.
    """
    out_dir = os.path.join(WORK, name); os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "prediction.json"); write_predictions(preds, test_texts, pj, include_text)
    if raw is not None:
        print(f"[post-processing] changed docs: {sum(set(raw[d]) != set(preds[d]) for d in preds)}/{len(preds)} | spans added {sum(len(set(preds[d]) - set(raw[d])) for d in preds)}"
              f" | spans removed/relabelled {sum(len(set(raw[d]) - set(preds[d])) for d in preds)}")
    verify_submission(pj, test_texts)
    zp = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z: z.write(pj, "prediction.json")
    print("submission zip:", zp); return preds


def predict_test(model, cfg, name, include_text=False):
    """Predict the test set with one model (decoder settings from ``cfg``; post-processing from ``state.POST``).

    Args:
        model (nn.Module): trained model.
        cfg (dict): its config.
        name (str): submission name.
        include_text (bool): include span text in rows.

    Returns:
        dict: final predictions.
    """
    test_texts = load_test_docs(); ids = list(test_texts); post = S.POST
    raw = predict_spans_raw(model, ids, cfg)          # model + decoder, no post-processing
    preds = {d: postprocess_doc(S.DOCS[d]["text"], raw[d], post) for d in ids} if post else raw
    return _finish_submission(preds, raw if post else None, test_texts, name, include_text)


def counsel_guard(post, raw, tol=0.3):
    """Label-free guard: drop the counsel components if they move the A:R counsel ratio FURTHER from the training-gold ratio.

    Args:
        post (dict): post-processing configuration.
        raw (dict): raw test predictions.
        tol (float): tolerated increase of |log ratio deviation|.

    Returns:
        dict: ``post`` or a copy with counsel rules disabled.
    """
    if not (post.get("counsel") or post.get("counsel_party")): return post
    cnt = lambda pr, lab: sum(1 for v in pr.values() for x in v if x[2] == lab)
    r_gold = (sum(1 for v in S.GOLD.values() for x in v if x[2] == "A.COUNSEL") + 1) / (sum(1 for v in S.GOLD.values() for x in v if x[2] == "R.COUNSEL") + 1)
    ratio = lambda pr: (cnt(pr, "A.COUNSEL") + 1) / (cnt(pr, "R.COUNSEL") + 1)
    off = {**post, "counsel": False, "counsel_party": False}
    with_c = {d: postprocess_doc(S.DOCS[d]["text"], v, post) for d, v in raw.items()}; no_c = {d: postprocess_doc(S.DOCS[d]["text"], v, off) for d, v in raw.items()}
    dev = lambda pr: abs(math.log(ratio(pr) / r_gold))
    print(f"[counsel guard] A:R train-gold {r_gold:.2f} | without counsel rules {ratio(no_c):.2f} | with {ratio(with_c):.2f}")
    if dev(with_c) > dev(no_c) + tol: print("[counsel guard] counsel relabelling moves A:R away from the gold ratio -> DROPPED for this submission"); return off
    return post


def predict_test_ensemble(ckpts, name, weights=None, include_text=False, guard=True, decode_overrides=None):
    """Average token probabilities of several checkpoints (seeds / architectures), decode once, post-process once, verify and zip.

    Args:
        ckpts (list[str]): checkpoint paths (same TASK).
        name (str): submission name.
        weights (list[float] or None): ensemble weights.
        include_text (bool): include span text in rows.
        guard (bool): apply the label-free counsel guard.
        decode_overrides (dict or None): decoder config overrides.

    Returns:
        dict: final predictions.
    """
    test_texts = load_test_docs(); ids = list(test_texts); probs, cfg0 = ens_probs(ckpts, ids, weights)
    dcfg = {**cfg0, **(decode_overrides or {})}
    raw = {d: decode_doc(S.DOCS[d], p, dcfg) for d, p in zip(ids, probs)}; post = dict(S.POST)
    if post and guard: post = counsel_guard(post, raw)
    preds = {d: postprocess_doc(S.DOCS[d]["text"], raw[d], post) for d in ids} if post else raw
    return _finish_submission(preds, raw if post else None, test_texts, name, include_text)
