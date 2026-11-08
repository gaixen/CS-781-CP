"""Data audit (section 4.1 / 4.2): schema, pools, gold statistics, leak checks and length statistics."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcr.utils import md5_text, percentile_summary


def audit_schema(raw, k=7):
    """Print pool sizes, gold statistics, recall / precision upper bounds and train-test leak checks.

    Args:
        raw (Dict[str, List[dict]]): Normalised records (keys train, train_c, dev, dev_c, test, test_c).
        k (int): Evaluation cut-off used for the upper bounds.

    Returns:
        None
    """
    for s in ("train", "dev", "test"):
        Q, C = raw[s], raw[s + "_c"]
        cids = [d["id"] for d in C]
        cset = set(cids)
        self_in_pool = sum(d["id"] in cset for d in Q)
        print(f"[{s}] queries={len(Q)} pool={len(C)} unique pool ids={len(cset)} | query ids also in pool={self_in_pool} | "
              f"empty q text={sum(not d['text'].strip() for d in Q)} empty cand text={sum(not d['text'].strip() for d in C)}")
        if s != "test":
            ng = np.array([len(d["gold"]) for d in Q])
            inpool = np.array([sum(g in cset for g in d["gold"]) for d in Q])
            print(f"   gold/query: min={ng.min()} median={np.median(ng):.0f} mean={ng.mean():.2f} p90={np.percentile(ng, 90):.0f} max={ng.max()} | "
                  f"share of gold present in pool={inpool.sum()/ng.sum():.3f} | queries with 0 gold in pool={(inpool == 0).sum()}")
            print(f"   gold>{k} for {(ng > k).mean()*100:.1f}% of queries -> recall@{k} upper bound={np.minimum(ng, k).sum()/ng.sum():.3f} ; "
                  f"precision@{k} upper bound={np.minimum(ng, k).sum()/(k*len(Q)):.3f}")
        else:
            print("   test labels present:", sum(d["gold"] is not None for d in Q))
    tr_ids, dv_ids, te_ids = [{d["id"] for d in raw[s]} for s in ("train", "dev", "test")]
    tc, dc, ec = [{d["id"] for d in raw[s + "_c"]} for s in ("train", "dev", "test")]
    print(f"\nLEAK/OVERLAP: query ids train&test={len(tr_ids & te_ids)} train&dev={len(tr_ids & dv_ids)} dev&test={len(dv_ids & te_ids)} | "
          f"pool ids train&test={len(tc & ec)} train&dev={len(tc & dc)} dev&test={len(dc & ec)}")
    hq = {md5_text(d["text"]) for d in raw["test"]}
    print("train queries/candidates whose text equals a test query:", sum(md5_text(d["text"]) in hq for d in raw["train"] + raw["train_c"]))
    print("sentences/doc (q):", pd.Series([d["nsent"] for d in raw["train"] if d["nsent"]]).describe().round(0).to_dict())


def length_statistics(raw, show=True, save_path=None):
    """Print word-length statistics of queries and pools and plot their distributions.

    Args:
        raw (Dict[str, List[dict]]): Normalised records.
        show (bool): Call ``plt.show()`` after plotting.
        save_path (Optional[str]): If given, save the figure to this path.

    Returns:
        pd.DataFrame: Length statistics (rows = split, columns = summary statistic).
    """
    L = pd.DataFrame({f"{s}{'_c' if c else ''} words": percentile_summary([len(d["text"].split()) for d in raw[s + ("_c" if c else "")]])
                      for s in ("train", "dev", "test") for c in (0, 1)})
    print(L.round(0).T.to_string())
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    for s in ("train", "dev", "test"):
        ax[0].hist(np.log10([len(d["text"].split()) + 1 for d in raw[s]]), 40, alpha=.5, label=s)
    ax[0].set_title("log10 words, queries")
    ax[0].legend()
    ax[1].hist([len(d["gold"]) for d in raw["train"]], 40)
    ax[1].set_title("gold per train query")
    ax[2].hist([len(raw["train_c"])], 1)
    ax[2].set_title("pool sizes (train)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return L
