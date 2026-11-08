"""Failure analysis of the best single experiment on dev (section 11)."""
import numpy as np
import pandas as pd

from pcr.training.trainer import exp_dir


def failure_analysis(ctx, best):
    """Break down dev errors by gold count, query length and rank; compare dense vs lexical complementarity; show worst queries.

    Args:
        ctx (PCRContext): Pipeline context.
        best (str): Experiment id whose dev scores are analysed.

    Returns:
        pd.DataFrame: Per-query table (``n_gold, tp, f1, median_gold_rank, q_tokens``).
    """
    cfg, spl = ctx.cfg, ctx.spl
    K = cfg.k
    Sd = np.load(f"{exp_dir(cfg, best)}/dev_scores.npy").astype(np.float32)
    cids = np.array(spl["dev"]["cids"])
    qids = spl["dev"]["qids"]
    gold = spl["dev"]["gold"]
    o = np.argsort(-Sd, 1)
    rank_of = lambda i, c: int(np.where(cids[o[i]] == c)[0][0])
    rows = []
    for i, q in enumerate(qids):
        ranks = sorted(rank_of(i, g) for g in gold[q])
        tp = sum(r < K for r in ranks)
        rows.append(dict(q=q, n_gold=len(gold[q]), tp=tp, f1=(2 * tp / (K + len(gold[q]))), median_gold_rank=np.median(ranks) if ranks else np.nan,
                         q_tokens=len(spl["dev"]["qtok"][q])))
    A = pd.DataFrame(rows)
    print(A[["n_gold", "tp", "f1", "median_gold_rank", "q_tokens"]].describe().round(2).to_string())
    print("\nF1 by #gold bucket:")
    print(A.groupby(pd.cut(A.n_gold, [0, 1, 3, 5, 7, 100]))["f1"].agg(["mean", "count"]).round(3).to_string())
    print("\nF1 by query length (tokens) quartile:")
    print(A.groupby(pd.qcut(A.q_tokens, 4, duplicates="drop"))["f1"].agg(["mean", "count"]).round(3).to_string())
    gr = [rank_of(i, g) for i, q in enumerate(qids) for g in gold[q]]
    print("\nrank distribution of gold cands: median", np.median(gr), "| p90", np.percentile(gr, 90), "| % in top7", round(100 * np.mean(np.array(gr) < K), 1),
          "| % in top100", round(100 * np.mean(np.array(gr) < 100), 1))
    lexd = ctx.lex["tfidf"]["dev"]
    tf_o = np.argsort(-lexd, 1)
    both = sum(len(set(cids[o[i, :K]]) & gold[q]) for i, q in enumerate(qids))
    tfo = sum(len(set(cids[tf_o[i, :K]]) & gold[q]) for i, q in enumerate(qids))
    union = sum(len((set(cids[o[i, :K]]) | set(cids[tf_o[i, :K]])) & gold[q]) for i, q in enumerate(qids))
    print(f"TP@7 dense={both} tfidf={tfo} union(oracle)={union} -> complementary signals if union >> max")
    for q in A.sort_values("f1").head(3).q:
        i = qids.index(q)
        print(f"\nWORST query {q}: gold={list(gold[q])[:4]} | top-3 predicted={list(cids[o[i,:3]])}\n   QUERY: {spl['dev']['qtext'][q][:350]!r}\n"
              f"   TOP-1: {spl['dev']['ctext'][cids[o[i,0]]][:350]!r}")
    return A
