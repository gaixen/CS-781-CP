"""Section 9: dev vs test diagnostics, calibration and predicted-label counts for every finished experiment."""
import os

import numpy as np
import pandas as pd

from lsi.eval.metrics import all_metrics, decide, ece_flat, fit_temperature, honest, sigmoid, tune_global
from lsi.training.engine import exp_dir

KNOWN_KD_TEST = dict(macro_f1=0.4078, micro_f1=0.5056, subset_acc=0.0439)


def load_available(ctx):
    """Load dev / test logits of every finished experiment and fit calibration temperatures.

    Experiments whose logits do not match the current data shapes are excluded.

    Args:
        ctx (Context): Pipeline context; ``avail, zd, zt, temp`` are filled in place.

    Returns:
        List[str]: Usable experiment ids.
    """
    data = ctx.data
    ctx.avail, ctx.zd, ctx.zt, ctx.temp = [], {}, {}, {}
    for e in ctx.experiments:
        d = exp_dir(ctx, e)
        fd, ft = f"{d}/dev_logits.npy", f"{d}/test_logits.npy"
        if not (os.path.exists(fd) and os.path.exists(ft)):
            continue
        zd, zt = np.load(fd), np.load(ft)
        if zd.shape != (len(data.dev_docs), data.n_labels) or zt.shape != (len(data.test_docs), data.n_labels):
            print(f"[{e}] SHAPE MISMATCH logits {zd.shape}/{zt.shape} vs data {(len(data.dev_docs), data.n_labels)}/{(len(data.test_docs), data.n_labels)} -> excluded")
            continue
        ctx.avail.append(e)
        ctx.zd[e], ctx.zt[e] = zd, zt
        ctx.temp[e] = fit_temperature(zd, data.ydev)
    print("usable experiments:", ctx.avail)
    return ctx.avail


def diagnostics(ctx):
    """Calibration (T, ECE), in-sample vs out-of-fold dev macro-F1 and dev-vs-test predicted-label density per experiment.

    Args:
        ctx (Context): Pipeline context after :func:`load_available`.

    Returns:
        pd.DataFrame: Diagnostics table indexed by experiment id (also stored in ``ctx.diag``).
    """
    YDEV = ctx.data.ydev
    rows = []
    for e in ctx.avail:
        Pd, Pt = sigmoid(ctx.zd[e] / ctx.temp[e]), sigmoid(ctx.zt[e] / ctx.temp[e])
        g, f1 = tune_global(Pd, YDEV)
        Bd, Bt = decide(Pd, g), decide(Pt, g)
        m = all_metrics(Bd, YDEV)
        rows.append(dict(exp=e, T=ctx.temp[e], ECE=ece_flat(Pd, YDEV), thr=g, dev_macroF1_insample=f1, dev_macroF1_oof=honest("global", Pd, YDEV)[0], dev_micro=m["micro_f1"],
                         dev_subset=m["subset_acc"], true_lab_per_doc=YDEV.sum(1).mean(), pred_dev=Bd.sum(1).mean(), pred_test=Bt.sum(1).mean(),
                         labels_unpredicted_dev=int((Bd.sum(0) == 0).sum()), labels_unpredicted_test=int((Bt.sum(0) == 0).sum()), test_LB=ctx.cfg.lb.get(e)))
    DIAG = pd.DataFrame(rows).set_index("exp")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(DIAG.round(4).to_string())
    print(f"\nKD test result for reference: {KNOWN_KD_TEST}")
    if "E4" in DIAG.index:
        print(f"E4 dev micro-F1 {DIAG.loc['E4','dev_micro']:.4f} vs test 0.5056 | dev subset-acc {DIAG.loc['E4','dev_subset']:.4f} vs test 0.0439 | dev oof macro {DIAG.loc['E4','dev_macroF1_oof']:.4f} vs test 0.4078")
    for e in ctx.avail:
        r = DIAG.loc[e]
        if r.pred_dev > 0 and abs(r.pred_test - r.pred_dev) / r.pred_dev > 0.15:
            print(f"WARNING [{e}]: predicted labels/doc differ by {100*(r.pred_test-r.pred_dev)/r.pred_dev:+.0f}% between test and dev -> thresholds may not transfer")
    ctx.diag = DIAG
    return DIAG
