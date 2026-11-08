"""Comparison of experiments (section 8) and the pooling / chunking ablation (section 9)."""
import gc
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pcr.models.dual_encoder import build_model, score_split
from pcr.retrieval.metrics import f1_at_k
from pcr.training.trainer import exp_dir, load_best

COLS = ["name", "arch", "method", "dev_f1", "dev_p", "dev_r", "dev_rec@20", "dev_rec@50", "dev_rec@100",
        "trainable_M", "total_M", "fp16_MB", "emb_dim", "train_min", "peak_gpu_GB", "infer_s_per_doc"]


def compare_experiments(ctx, show=True):
    """Tabulate quality and efficiency of all finished experiments and plot them.

    Writes ``comparison.csv`` to the submission folder.

    Args:
        ctx (PCRContext): Pipeline context.
        show (bool): Call ``plt.show()`` after plotting.

    Returns:
        pd.DataFrame: Comparison table indexed by experiment id.
    """
    cfg, spl = ctx.cfg, ctx.spl
    rows = [json.load(open(f"{exp_dir(cfg, e)}/result.json")) for e in ctx.experiments if os.path.exists(f"{exp_dir(cfg, e)}/result.json")]
    CMP = pd.DataFrame(rows).set_index("exp")[COLS]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(CMP.round(4).to_string())
    CMP.to_csv(f"{cfg.submit_dir}/comparison.csv")
    print(f"\nBM25 baseline dev:", {k: round(v, 4) for k, v in f1_at_k(ctx.lex["bm25"]["dev"], spl, "dev", cfg.k).items()},
          "| TF-IDF baseline dev:", {k: round(v, 4) for k, v in f1_at_k(ctx.lex["tfidf"]["dev"], spl, "dev", cfg.k).items()})
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
    CMP[["dev_f1"]].plot.bar(ax=ax[0], title="dev Micro-F1@7")
    CMP[["dev_rec@20", "dev_rec@50", "dev_rec@100"]].plot.bar(ax=ax[1], title="dev recall@k (re-ranking headroom)")
    CMP[["train_min", "infer_s_per_doc"]].plot.bar(ax=ax[2], title="cost", logy=True)
    plt.tight_layout()
    fig.savefig(f"{cfg.submit_dir}/comparison.png", dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    for e in CMP.index:
        h = json.load(open(f"{exp_dir(cfg, e)}/result.json"))["hist"]
        if h:
            print(e, "dev F1 per epoch:", [round(x["f1"], 4) for x in h])
    return CMP


def representation_ablation(ctx, best):
    """Ablate pooling / chunk selection / number of chunks / aggregation on the best experiment (dev).

    Note: pooling / aggregation different from training is a train / test mismatch; if one clearly wins, change the
    config and retrain with ``--force``.

    Args:
        ctx (PCRContext): Pipeline context.
        best (str): Experiment id to ablate (usually ``argmax dev_f1``).

    Returns:
        pd.DataFrame: One row per setting with dev P / R / F1 and runtime.
    """
    cfg, spl = ctx.cfg, ctx.spl
    print("best experiment:", best)
    BM = load_best(build_model(ctx.experiments[best], cfg, ctx.eos, ctx.device), best, cfg).eval()
    AB = []
    for strat, nch, agg, pool in [("head", 1, "mean", "mean"), ("head", 4, "mean", "mean"), ("spread", 4, "mean", "mean"), ("spread", 8, "mean", "mean"),
                                  ("spread", 4, "max", "mean"), ("spread", 4, "mean", "last"), ("spread", 4, "mean", "max")]:
        BM.pool = pool
        t0 = time.time()
        r = f1_at_k(score_split(BM, spl, "dev", strat=strat, nch=nch, agg=agg), spl, "dev", cfg.k)
        AB.append(dict(chunks=strat, n_chunks=nch, chunk_agg=agg, token_pool=pool, sec=round(time.time() - t0), **{k: round(v, 4) for k, v in r.items()}))
    df = pd.DataFrame(AB)
    print(df.to_string(index=False))
    del BM
    gc.collect()
    torch.cuda.empty_cache()
    return df
