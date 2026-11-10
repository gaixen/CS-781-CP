"""Results table, FP/FN error analysis, and size/cost table built from ``results.jsonl`` and saved dev predictions."""
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..config import RESULTS_PATH, OUT_ROOT
from ..utils import metrics

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)


def load_results(latest=True):
    """Load all experiment records.

    Args:
        latest (bool): keep only the most recent record per experiment name.

    Returns:
        pd.DataFrame: records (empty if no run has finished).
    """
    rows = [json.loads(l) for l in open(RESULTS_PATH)] if os.path.exists(RESULTS_PATH) else []
    df = pd.DataFrame(rows)
    return df.drop_duplicates("name", keep="last") if (latest and len(df)) else df


def show_results(prefix=""):
    """Return a dev-set comparison table sorted by macro-F1.

    Args:
        prefix (str): only experiments whose name starts with this prefix (e.g. ``"A_rep"``).

    Returns:
        pd.DataFrame: table of metrics, parameter counts, memory, time and throughput.
    """
    df = load_results()
    if not len(df): print("no results yet -- run some experiments first"); return df
    df = df[df.name.str.startswith(prefix)]
    cols = ["name", "arch", "method", "rep", "pooling", "doc_agg", "head_layers", "dev_f1", "dev_P", "dev_R", "f1_c0", "f1_c1", "params_M", "trainable_M", "trainable_pct",
            "peak_mem_GB", "train_min", "infer_docs_per_s"]
    return df[cols].sort_values("dev_f1", ascending=False).round(4)


def error_analysis(name):
    """Print false-positive / false-negative characteristics on dev (uses saved dev predictions and the calibrated threshold).

    Args:
        name (str): experiment name.

    Returns:
        None
    """
    r = load_results().set_index("name").loc[name]; d = np.load(f"{OUT_ROOT}/{name}/dev_preds.npz"); y, pr, n = d["label"], (d["prob"] >= r.thr).astype(int), d["n_chars"]
    print(f"{name}: threshold={r.thr:.2f}  macro-F1={r.dev_f1:.4f}"); print(pd.DataFrame(confusion_matrix(y, pr), index=["true 0", "true 1"], columns=["pred 0", "pred 1"]))
    grp = np.where((y == 1) & (pr == 1), "TP", np.where((y == 0) & (pr == 0), "TN", np.where((y == 0) & (pr == 1), "FP", "FN")))
    t = pd.DataFrame(dict(group=grp, chars=n, prob=d["prob"])); print(t.groupby("group").agg(count=("chars", "size"), median_chars=("chars", "median"), mean_prob_class1=("prob", "mean")).round(3))
    t["len_bucket"] = pd.qcut(t.chars, 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"]); t["correct"] = (y == pr)
    print("accuracy by document length quartile:"); print(t.groupby("len_bucket", observed=True).correct.mean().round(3))
    print("per-class report:", metrics(y, pr))


def sizes_table():
    """Model size / cost table (parameters, trainable MB, memory, time, throughput, compression vs. teacher).

    Args:
        None

    Returns:
        pd.DataFrame: one row per experiment.
    """
    df = load_results(); df["compression_vs_teacher"] = df.teacher_params_M / df.params_M
    return df[["name", "params_M", "trainable_M", "trainable_MB", "peak_mem_GB", "train_min", "infer_docs_per_s", "teacher_params_M", "compression_vs_teacher"]].round(3)
