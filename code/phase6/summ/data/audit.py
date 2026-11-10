"""Data audit (sections 4.1-4.4): integrity, length statistics, truncation / positional coverage, abstractiveness and baselines."""
from collections import Counter

import numpy as np
import pandas as pd

from summ.data.loader import md5
from summ.eval.metrics import ngrams, rouge


def pct(a):
    """Percentile summary of an array.

    Args:
        a (array-like): Numeric values.

    Returns:
        pd.Series: min / percentiles / mean / max.
    """
    a = np.asarray(a, float)
    return pd.Series({"min": a.min(), "p5": np.percentile(a, 5), "p25": np.percentile(a, 25), "median": np.median(a), "mean": a.mean(), "p75": np.percentile(a, 75),
                      "p90": np.percentile(a, 90), "p95": np.percentile(a, 95), "p99": np.percentile(a, 99), "max": a.max()})


def audit_integrity(ctx):
    """4.1: schema, duplicates, train/test leak check. Blanks the (ignored) test summaries.

    Args:
        ctx (Context): Context with raw train / test rows.

    Returns:
        None
    """
    train_all, test_docs = ctx.train_all, ctx.test_docs
    is_str = lambda x: isinstance(x, str)
    print("key sets  train:", Counter(d["keys"] for d in train_all).most_common(3))
    print("key sets  test :", Counter(d["keys"] for d in test_docs).most_common(3))
    rep = {}
    for nm, ds, need_sum in (("train", train_all, True), ("test", test_docs, False)):
        r = dict(rows=len(ds), doc_not_str=sum(not is_str(d["document"]) for d in ds), doc_empty=sum((not is_str(d["document"])) or not d["document"].strip() for d in ds),
                 doc_lt_200ch=sum(is_str(d["document"]) and len(d["document"].strip()) < 200 for d in ds), sum_missing=sum(d["summary"] is None for d in ds),
                 sum_empty=sum(is_str(d["summary"]) and not d["summary"].strip() for d in ds), sum_lt_50ch=sum(is_str(d["summary"]) and len(d["summary"].strip()) < 50 for d in ds),
                 dup_ids=len(ds) - len({d["id"] for d in ds}), dup_docs=len(ds) - len({md5(d["document"]) for d in ds}),
                 dup_docs_first300=len(ds) - len({md5((d["document"] or "")[:300]) for d in ds}))
        if need_sum:
            r["dup_summaries"] = len(ds) - len({md5(d["summary"]) for d in ds})
            r["summary_identical_to_doc"] = sum(d["summary"] == d["document"] for d in ds)
        rep[nm] = r
    print(pd.DataFrame(rep).fillna("-").to_string())
    tid, th, tf3 = {d["id"] for d in test_docs}, {md5(d["document"]) for d in test_docs}, {md5((d["document"] or "")[:300]) for d in test_docs}
    print(f"\nLEAK CHECK train vs test: same id={sum(d['id'] in tid for d in train_all)} | identical doc={sum(md5(d['document']) in th for d in train_all)} | "
          f"same first-300-chars={sum(md5((d['document'] or '')[:300]) in tf3 for d in train_all)}")
    print(f"test.json carries a summary field for {sum(d['summary'] is not None for d in test_docs)} docs (ignored).")
    for d in test_docs:
        d["summary"] = None


def audit_lengths(ctx, tr_doc_ids, tr_sum_ids, te_doc_ids, show=False, save_path=None):
    """4.2: length statistics (chars / words / tokens), compression, train-vs-test KS test.

    Args:
        ctx (Context): Context.
        tr_doc_ids (list): Tokenised train documents.
        tr_sum_ids (list): Tokenised train summaries.
        te_doc_ids (list): Tokenised test documents.
        show (bool): Call ``plt.show()``.
        save_path (Optional[str]): Save the figure here.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    train_all, test_docs = ctx.train_all, ctx.test_docs
    wc = lambda t: len(t.split())
    L = pd.DataFrame({"train doc chars": pct([len(d["document"]) for d in train_all]), "train doc words": pct([wc(d["document"]) for d in train_all]), "train doc tok": pct([len(x) for x in tr_doc_ids]),
                      "test doc chars": pct([len(d["document"]) for d in test_docs]), "test doc words": pct([wc(d["document"]) for d in test_docs]), "test doc tok": pct([len(x) for x in te_doc_ids]),
                      "train sum chars": pct([len(d["summary"]) for d in train_all]), "train sum words": pct([wc(d["summary"]) for d in train_all]), "train sum tok": pct([len(x) for x in tr_sum_ids])})
    print(L.round(0).T.to_string())
    cr = np.array([len(s) / max(len(d), 1) for s, d in zip(tr_sum_ids, tr_doc_ids)])
    print("\ncompression (summary tok / doc tok): ", pct(cr).round(4).to_dict())
    ks = stats.ks_2samp([len(x) for x in tr_doc_ids], [len(x) for x in te_doc_ids])
    print(f"KS test doc-length train vs test: stat={ks.statistic:.3f} p={ks.pvalue:.3g}  ({'SHIFT' if ks.pvalue < 0.01 and ks.statistic > 0.1 else 'no strong shift'})")
    print("corr(doc len, summary len) =", round(float(np.corrcoef(np.log([len(x) for x in tr_doc_ids]), np.log([max(len(x), 1) for x in tr_sum_ids]))[0, 1]), 3))
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.2))
    ax[0].hist(np.log10([len(x) for x in tr_doc_ids]), 50, alpha=.6, label="train")
    ax[0].hist(np.log10([len(x) for x in te_doc_ids]), 50, alpha=.6, label="test")
    ax[0].set_title("log10 doc tokens")
    ax[0].legend()
    ax[1].hist([len(x) for x in tr_sum_ids], 50)
    ax[1].set_title("summary tokens")
    ax[2].hist(np.log10(cr), 50)
    ax[2].set_title("log10 compression")
    ax[3].scatter(np.log10([len(x) for x in tr_doc_ids]), [len(x) for x in tr_sum_ids], s=3, alpha=.3)
    ax[3].set_title("doc vs summary len")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def audit_coverage(ctx, tr_doc_ids, tr_sum_ids, te_doc_ids):
    """4.3: truncation loss per token budget and where in the document the reference summary content comes from; builds ``ctx.idf`` / ``ctx.stop``.

    Args:
        ctx (Context): Context (``vsz``).
        tr_doc_ids (list): Tokenised train documents.
        tr_sum_ids (list): Tokenised train summaries.
        te_doc_ids (list): Tokenised test documents.

    Returns:
        np.ndarray: Indices of the sampled train documents (reused by :func:`audit_abstractiveness`).
    """
    dl, tl = np.array([len(x) for x in tr_doc_ids]), np.array([len(x) for x in te_doc_ids])
    print(pd.DataFrame({b: {"train % docs > budget": 100 * (dl > b).mean(), "test % docs > budget": 100 * (tl > b).mean(), "train % tokens kept": 100 * np.minimum(dl, b).sum() / dl.sum()}
                        for b in (512, 1024, 1536, 2048, 4096, 8192)}).round(1).T.to_string())
    df = np.zeros(ctx.vsz)
    for x in tr_doc_ids:
        df[np.unique(x)] += 1
    ctx.idf = np.log((len(tr_doc_ids) + 1) / (df + 1)).astype(np.float32)
    ctx.stop = set(np.argsort(-df)[:300].tolist())
    samp = np.random.RandomState(0).choice(len(tr_doc_ids), min(400, len(tr_doc_ids)), replace=False)
    prof, anyc, n_used = np.zeros(10), [], 0
    for i in samp:
        ids, S = tr_doc_ids[i], set(tr_sum_ids[i].tolist()) - ctx.stop
        if len(ids) < 1000 or not S:
            continue
        n_used += 1
        anyc.append(len(S & set(ids.tolist())) / len(S))
        for k, part in enumerate(np.array_split(ids, 10)):
            prof[k] += len(S & set(part.tolist())) / len(S)
    prof /= max(n_used, 1)
    print(f"\ncontent-token coverage of the REFERENCE summary by document decile (docs>=1000 tok, n={n_used}):")
    print(" ".join(f"d{k+1}:{v:.2f}" for k, v in enumerate(prof)), f"| any-position copy rate={np.mean(anyc):.2f}")
    hd = prof[:5].sum() / prof.sum()
    print(f"share of summary-content mass in the first half of the doc: {hd:.2f} -> " + ("head-heavy: 'headtail' should do well" if hd > .65 else "spread out: 'select'/'hier' are worth testing"))
    return samp


def audit_abstractiveness(ctx, tr_doc_ids, tr_sum_ids, samp):
    """4.4: novel n-gram rate, trivial LEAD / TAIL baselines, noise statistics and outliers.

    Args:
        ctx (Context): Context.
        tr_doc_ids (list): Tokenised train documents.
        tr_sum_ids (list): Tokenised train summaries.
        samp (np.ndarray): Sampled indices from :func:`audit_coverage`.

    Returns:
        None
    """
    import re
    train_all = ctx.train_all
    nov = {n: [] for n in (1, 2, 3)}
    for i in samp[:300]:
        d, s_ = tr_doc_ids[i].tolist(), tr_sum_ids[i].tolist()
        for n in nov:
            sg = ngrams(s_, n)
            if sg:
                nov[n].append(1 - len(sg & ngrams(d, n)) / len(sg))
    print("novel n-gram rate of reference summaries (0 = fully extractive):", {f"{n}-gram": round(float(np.mean(v)), 3) for n, v in nov.items()})
    ii = samp[:200]
    refs = [train_all[i]["summary"] for i in ii]
    medw = int(np.median([len(d["summary"].split()) for d in train_all]))
    base = {f"LEAD-{medw}w": [" ".join(train_all[i]["document"].split()[:medw]) for i in ii], f"TAIL-{medw}w": [" ".join(train_all[i]["document"].split()[-medw:]) for i in ii],
            f"HEAD/TAIL-{medw}w": [" ".join(train_all[i]["document"].split()[:medw // 2] + train_all[i]["document"].split()[-medw // 2:]) for i in ii]}
    print("trivial baselines (ROUGE-1 F, ROUGE-L F):", {k: tuple(round(x, 4) for x in rouge(v, refs)) for k, v in base.items()}, "  <- every model must beat these")
    nz = lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1)
    noise = pd.DataFrame({"doc non-ascii %": pct([100 * nz(d["document"]) for d in train_all[:2000]]),
                          "doc newline density %": pct([100 * d["document"].count("\n") / max(len(d["document"]), 1) for d in train_all[:2000]]),
                          "ref-summary repeated trigram %": pct([100 * (1 - len(ngrams(s_.tolist(), 3)) / max(len(s_) - 2, 1)) for s_ in tr_sum_ids[:2000]])}).round(2)
    print(noise.T[["min", "median", "mean", "p95", "max"]].to_string())
    print("docs containing HTML-like tags:", sum(bool(re.search(r"<[a-zA-Z/][^>]{0,20}>", d["document"])) for d in train_all[:2000]), "/ 2000 checked")
    cr = np.array([len(s_) / max(len(d), 1) for s_, d in zip(tr_sum_ids, tr_doc_ids)])
    bad = (cr > 0.8) | (np.array([len(x) for x in tr_sum_ids]) < 20)
    print("outlier ids (summary >80% of doc, or <20 tokens):", [train_all[i]["id"] for i in np.where(bad)[0]][:10], f"(total {int(bad.sum())})")


def run_audit(ctx, tr_doc_ids, tr_sum_ids, te_doc_ids, save_path=None):
    """Run the length, coverage and abstractiveness audits.

    Args:
        ctx (Context): Context.
        tr_doc_ids (list): Tokenised train documents.
        tr_sum_ids (list): Tokenised train summaries.
        te_doc_ids (list): Tokenised test documents.
        save_path (Optional[str]): Figure path for the length plot.

    Returns:
        None
    """
    audit_lengths(ctx, tr_doc_ids, tr_sum_ids, te_doc_ids, save_path=save_path)
    samp = audit_coverage(ctx, tr_doc_ids, tr_sum_ids, te_doc_ids)
    audit_abstractiveness(ctx, tr_doc_ids, tr_sum_ids, samp)
