"""Summary metrics: ROUGE-1/L F1, BERTScore-F1 (primary), generation diagnostics and post-processing."""
import re

import numpy as np
import torch

_RS = None


def ngrams(a, n):
    """Set of n-grams of a sequence.

    Args:
        a (Sequence): Tokens.
        n (int): N-gram size.

    Returns:
        set: N-gram tuples.
    """
    return set(zip(*[a[i:] for i in range(n)]))


def rouge(preds, refs):
    """Mean ROUGE-1 and ROUGE-L F-measure (stemmed).

    Args:
        preds (List[str]): Predicted summaries.
        refs (List[str]): Reference summaries.

    Returns:
        Tuple[float, float]: ``(rouge1_f, rougeL_f)``.
    """
    global _RS
    if _RS is None:
        from rouge_score import rouge_scorer
        _RS = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rl = [], []
    for p, r in zip(preds, refs):
        s = _RS.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return float(np.mean(r1)), float(np.mean(rl))


def bertscore(preds, refs, device):
    """Mean BERTScore-F1 (NaN if the package or model is unavailable).

    Args:
        preds (List[str]): Predictions.
        refs (List[str]): References.
        device (torch.device): Device for the scorer.

    Returns:
        float: Mean F1.
    """
    try:
        from bert_score import score as bs
        _, _, f = bs(preds, refs, lang="en", batch_size=32, device=str(device))
        torch.cuda.empty_cache()
        return float(f.mean())
    except Exception as e:
        print("BERTScore unavailable:", type(e).__name__, str(e)[:120])
        return float("nan")


def diag(preds, docs):
    """Length, repetition, extractiveness and emptiness diagnostics.

    Args:
        preds (List[str]): Predictions.
        docs (List[str]): Source documents.

    Returns:
        dict: ``words, repeat_trigram, copy_bigram, empty``.
    """
    w = [len(p.split()) for p in preds]
    rep3 = [1 - len(ngrams(p.split(), 3)) / max(len(p.split()) - 2, 1) for p in preds]
    cop = [1 - len(ngrams(p.split(), 2) - ngrams(d.split(), 2)) / max(len(ngrams(p.split(), 2)), 1) for p, d in zip(preds, docs)]
    return dict(words=float(np.mean(w)), repeat_trigram=float(np.mean(rep3)), copy_bigram=float(np.mean(cop)), empty=int(sum(not p.strip() for p in preds)))


def evaluate(preds, refs, docs, device):
    """BERTScore, ROUGE and diagnostics.

    Args:
        preds (List[str]): Predictions.
        refs (List[str]): References.
        docs (List[str]): Source documents.
        device (torch.device): Device for BERTScore.

    Returns:
        dict: ``bertscore_f1, rouge1_f, rougeL_f`` plus :func:`diag` entries.
    """
    r1, rl = rouge(preds, refs)
    return dict(bertscore_f1=bertscore(preds, refs, device), rouge1_f=r1, rougeL_f=rl, **diag(preds, docs))


def postprocess(t):
    """Collapse whitespace, drop repeated sentences and cut a dangling fragment.

    Args:
        t (str): Generated text.

    Returns:
        str: Cleaned summary.
    """
    t = re.sub(r"\s+", " ", t).strip()
    out, seen = [], set()
    for s in re.split(r"(?<=[.!?])\s+", t):
        k = re.sub(r"\W+", " ", s.lower()).strip()
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    t = " ".join(out)
    if t and t[-1] not in ".!?" and "." in t[len(t) // 2:]:
        t = t[:t.rfind(".") + 1]
    return t
