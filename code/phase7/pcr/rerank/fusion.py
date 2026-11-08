"""Score-level fusion of the experiments + a lexical signal, weights tuned on dev.

* :func:`fuse_experiments` (section 10): every subset of E1..E5 x lexical weight grid.
* :func:`ensemble_experiments` (section 10b): average of E1-E4 (+ MaxSim, + BM25) with the BM25 weight chosen on dev.

NOTE: in the notebook the "TF-IDF" component of these two sections is BM25 (``TFIDF = LEXS["bm25"]`` at that point);
pass ``ctx.lex["bm25"]`` to reproduce that behaviour.
"""
import itertools
import os

import numpy as np
import pandas as pd

from pcr.rerank.common import load_scores
from pcr.retrieval.metrics import f1_at_k
from pcr.submission import validate_all_submissions, validate_submission, write_submission
from pcr.training.trainer import exp_dir
from pcr.utils import zs

LEX_WEIGHTS = (0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def fuse_experiments(ctx, lexical):
    """Search subset-averages of the fine-tuned experiments plus a weighted lexical component on dev.

    Writes the best fusion as a submission and validates every zip in the submission folder.

    Args:
        ctx (PCRContext): Pipeline context.
        lexical (Dict[str, np.ndarray]): Lexical score matrices per split (``dev`` and ``test`` keys).

    Returns:
        pd.DataFrame: All fusion candidates sorted by dev F1 (best first).
    """
    cfg, spl = ctx.cfg, ctx.spl
    avail = [e for e in ctx.experiments if os.path.exists(f"{exp_dir(cfg, e)}/dev_scores.npy") and e != "E0"]
    DS = {e: load_scores(ctx, e, "dev") for e in avail}
    TS = {e: load_scores(ctx, e, "test") for e in avail}
    TF_D, TF_T = zs(lexical["dev"]), zs(lexical["test"])
    res = []
    for r in range(1, len(avail) + 1):
        for sub in itertools.combinations(avail, r):
            for w in LEX_WEIGHTS:
                S = sum(DS[e] for e in sub) / len(sub) + w * TF_D
                res.append(dict(models="+".join(sub), tfidf_w=w, **{k: round(v, 4) for k, v in f1_at_k(S, spl, "dev", cfg.k).items()}))
    FUS = pd.DataFrame(res).sort_values("f1", ascending=False)
    print(FUS.head(12).to_string(index=False))
    top = FUS.iloc[0]
    sub = top["models"].split("+")
    w = float(top["tfidf_w"])
    write_submission(f"FUSION_{top['models']}_tfidf{w}", sum(TS[e] for e in sub) / len(sub) + w * TF_T, spl, cfg)
    print("wrote fusion submission:", top["models"], "w_tfidf =", w, "| dev F1 =", top["f1"], "(weights tuned on dev -> optimistic)")
    validate_all_submissions(spl, cfg)
    print("Submit one zip per experiment E1-E4 (each: prediction.json, 7 unique pool ids per query) and optionally the fusion zip.")
    return FUS


def ensemble_experiments(ctx, lexical):
    """Ensemble E1-E4 (single seed each), optionally with MaxSim and BM25; the BM25 weight is chosen on dev.

    Args:
        ctx (PCRContext): Pipeline context.
        lexical (Dict[str, np.ndarray]): Lexical score matrices per split (``dev`` and ``test`` keys).

    Returns:
        Tuple[pd.DataFrame, float]: ``(table, chosen_bm25_weight)`` where ``table`` lists every evaluated system on dev.
    """
    cfg, spl = ctx.cfg, ctx.spl
    ENS = [e for e in ("E1", "E2", "E3", "E4")
           if os.path.exists(f"{exp_dir(cfg, e)}/dev_scores.npy") and os.path.exists(f"{exp_dir(cfg, e)}/test_scores.npy")]
    D = {e: load_scores(ctx, e, "dev") for e in ENS}
    T = {e: load_scores(ctx, e, "test") for e in ENS}
    DM = {e: load_scores(ctx, e, "dev", "ms") for e in ENS if os.path.exists(f"{exp_dir(cfg, e)}/dev_ms.npy")}
    lx_d, lx_t = zs(lexical["dev"]), zs(lexical["test"])
    f = lambda S: {k: round(v, 4) for k, v in f1_at_k(S, spl, "dev", cfg.k).items()}
    rows = [dict(system=e, **f(D[e])) for e in ENS] + [dict(system=e + " MaxSim", **f(DM[e])) for e in DM]
    avg_d, avg_t = sum(D.values()) / len(D), sum(T.values()) / len(T)
    rows.append(dict(system="ENS(all)", **f(avg_d)))
    if DM:
        ms_d = sum(DM.values()) / len(DM)
        rows.append(dict(system="ENS(all)+MaxSim", **f(avg_d + ms_d)))
    for w in (0.2, 0.5, 1.0):
        rows.append(dict(system=f"ENS(all)+bm25*{w}", **f(avg_d + w * lx_d)))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    bw = max(LEX_WEIGHTS, key=lambda w: f1_at_k(avg_d + w * lx_d, spl, "dev", cfg.k)["f1"])
    print("chosen bm25 weight:", bw)
    zp = write_submission(f"ENSEMBLE_{'+'.join(ENS)}_bm25{bw}", avg_t + bw * lx_t, spl, cfg)
    validate_submission(zp, spl, cfg.k)
    return table, bw
