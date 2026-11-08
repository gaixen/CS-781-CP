"""Submission zips: naming (distinguishing info first), writing, validation and the per-experiment threshold pipeline."""

import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd

from lsi.config import ARCH_SHORT
from lsi.eval.metrics import (
    ST2,
    THR_SLUG,
    all_metrics,
    choose,
    decide,
    fit_strategy2,
    fit_temperature,
    honest,
    sigmoid,
)

SHORT = {
    "full": "FullFT",
    "lorma": "LoRMAp",
    "adapters": "Adpt",
    "kd": "KD",
    "partial": "Partial",
}


def make_tag2(ctx, key, s):
    """Zip name with experiment, technique and threshold strategy in the first ~35 characters.

    Args:
        ctx (Context): Pipeline context (experiment registry).
        key (str): Experiment id or ensemble name.
        s (str): Threshold strategy name.

    Returns:
        str: Tag such as ``E4-KD_bagPL30top1_lswBidir_LWAN_BCE_doc-full``.
    """
    c = ctx.experiments.get(key)
    if c:
        return f"{key}-{SHORT[c['method']]}_{THR_SLUG[s]}_{ARCH_SHORT[c['arch']]}_{c['head'].upper()}_{c['loss'].upper()}_doc-{c['doc_mode']}"
    return f"{key}_{THR_SLUG[s]}"


def to_pred(ctx, B):
    """Convert a boolean matrix to sorted lists of statute ids.

    Args:
        ctx (Context): Pipeline context (label list).
        B (np.ndarray): Boolean predictions (n, L).

    Returns:
        List[List[int]]: Statute ids per document.
    """
    labels = ctx.data.labels
    return [sorted({int(labels[j]) for j in np.where(r)[0]}) for r in B]


def write_zip(ctx, tag, B_test):
    """Write ``prediction.json`` (``[{"id", "pred"}]`` in test order) into ``<submit_dir>/<tag>.zip``.

    Args:
        ctx (Context): Pipeline context.
        tag (str): Zip name without extension.
        B_test (np.ndarray): Boolean test predictions (n_test, L).

    Returns:
        str: Path of the zip.
    """
    test_docs = ctx.data.test_docs
    pred = [
        {"id": d["rid"], "pred": p} for d, p in zip(test_docs, to_pred(ctx, B_test))
    ]
    assert len(pred) == len(test_docs) and len({str(x["id"]) for x in pred}) == len(
        pred
    )
    d = f"{ctx.cfg.submit_dir}/{tag}"
    os.makedirs(d, exist_ok=True)
    json.dump(pred, open(f"{d}/prediction.json", "w"))
    zp = f"{ctx.cfg.submit_dir}/{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{d}/prediction.json", "prediction.json")
    return zp


def validate_zip(ctx, zp):
    """Check ids / order / integer valid statutes / uniqueness of a submission zip.

    Args:
        ctx (Context): Pipeline context.
        zp (str): Zip path.

    Returns:
        Dict[str, float]: ``n``, ``mean_labels`` and ``empty`` (documents without prediction).
    """
    with zipfile.ZipFile(zp) as z:
        pred = json.loads(z.read("prediction.json"))
    valid = set(ctx.data.labels)
    assert [str(x["id"]) for x in pred] == [d["id"] for d in ctx.data.test_docs], (
        "ids/order differ from test.json"
    )
    assert all(isinstance(v, int) and v in valid for x in pred for v in x["pred"]), (
        "non-int or unknown statute id"
    )
    assert all(len(set(x["pred"])) == len(x["pred"]) for x in pred), "duplicate ids"
    return dict(
        n=len(pred),
        mean_labels=float(np.mean([len(x["pred"]) for x in pred])),
        empty=int(sum(len(x["pred"]) == 0 for x in pred)),
    )


def process(ctx, key, Zd, Zt, tag_fn, write_all=False):
    """Calibrate, select the honest-best threshold strategy and write (and validate) the zip(s).

    Args:
        ctx (Context): Pipeline context (``ctx.submit_zips`` is updated).
        key (str): Experiment / ensemble name.
        Zd (np.ndarray): Dev logits.
        Zt (np.ndarray): Test logits.
        tag_fn (Callable[[str], str]): Strategy name -> zip tag.
        write_all (bool): Write every strategy of :data:`ST2`, not just the selected one.

    Returns:
        None
    """
    YDEV = ctx.data.ydev
    T = fit_temperature(Zd, YDEV)
    Pd, Pt = sigmoid(Zd / T), sigmoid(Zt / T)
    tab = {s: honest(s, Pd, YDEV) for s in ST2}
    sel = choose(tab)
    print(
        f"\n[{key}] T={T:.2f} | honest macro-F1 per strategy: "
        + " | ".join(f"{s}={tab[s][0]:.4f}" for s in ST2)
        + f"\n      selected -> {sel}"
    )
    for s in ST2 if write_all else [sel]:
        thr, t1 = fit_strategy2(s, Pd, YDEV)
        Bd, Bt = decide(Pd, thr, t1), decide(Pt, thr, t1)
        m = all_metrics(Bd, YDEV)
        zp = write_zip(ctx, tag_fn(s), Bt)
        v = validate_zip(ctx, zp)
        ctx.submit_zips[os.path.basename(zp)] = dict(
            key=key,
            strategy=s,
            selected=(s == sel),
            honest_macro_f1=tab[s][0],
            dev_macro_f1_insample=m["macro_f1"],
            dev_micro=m["micro_f1"],
            test_labels_per_doc=v["mean_labels"],
            empty=v["empty"],
        )
        print(
            f"      {'*' if s == sel else ' '} {s:<13} dev(in-sample) macro={m['macro_f1']:.4f} micro={m['micro_f1']:.4f} | test labels/doc={v['mean_labels']:.2f} empty={v['empty']} -> {os.path.basename(zp)}"
        )


def final_check(ctx):
    """Write ``submission_plan.csv`` and re-validate every zip in the submission folder.

    Args:
        ctx (Context): Pipeline context.

    Returns:
        pd.DataFrame: The submission plan.
    """
    plan = pd.DataFrame(ctx.submit_zips).T
    plan.index.name = "zip"
    plan.to_csv(f"{ctx.cfg.submit_dir}/submission_plan.csv")
    print(plan.round(4).to_string())
    zips = sorted(glob.glob(f"{ctx.cfg.submit_dir}/*.zip"))
    for zp in zips:
        validate_zip(ctx, zp)
    print(
        f"\nall {len(zips)} zips validated (ids, order, int, unique, valid statutes). Folder: {ctx.cfg.submit_dir}"
    )
    return plan
