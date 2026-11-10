"""Generation-quality metrics (ROUGE, BLEU, distinct-n, repetition) and the decoding-strategy sweep."""

import time

DECODING_CONFIGS = [
    {"name": "greedy", "temperature": 0.0, "top_p": 1.0},
    {"name": "temp0.7", "temperature": 0.7, "top_p": 1.0},
    {"name": "temp0.7_topp0.9", "temperature": 0.7, "top_p": 0.9},
    {"name": "temp1.0_topp0.95", "temperature": 1.0, "top_p": 0.95},
]


def distinct_n(text, n):
    """Fraction of unique word n-grams.

    Args:
        text (str): text.
        n (int): n-gram size.

    Returns:
        float: distinct-n (0 if the text is shorter than n words).
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def compute_generation_quality(preds, refs):
    """ROUGE-1/2/L, BLEU, distinct-1/2 and repetition rate.

    Args:
        preds (list[str]): generated continuations.
        refs (list[str]): reference continuations.

    Returns:
        dict: metric name -> value (``repetition_rate = 1 - distinct_2``; higher = more degenerate repetition).
    """
    from rouge_score import rouge_scorer
    import sacrebleu

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    tot = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(preds, refs):
        s = sc.score(r, p)
        for k in tot:
            tot[k] += s[k].fmeasure
    n = max(len(preds), 1)
    d2 = sum(distinct_n(p, 2) for p in preds) / n
    return {
        **{k: v / n for k, v in tot.items()},
        "bleu": sacrebleu.corpus_bleu(preds, [refs]).score,
        "distinct_1": sum(distinct_n(p, 1) for p in preds) / n,
        "distinct_2": d2,
        "repetition_rate": 1.0 - d2,
    }


def evaluate_generation(gen_fn, model, tokenizer, device, pairs, gen_kw):
    """Internal generation-quality evaluation on held-out dev prompts (the official references are private).

    Args:
        gen_fn (Callable): generator from ``make_gen_fn``.
        model (nn.Module): model.
        tokenizer (PreTrainedTokenizer): tokenizer.
        device (torch.device): device.
        pairs (list[dict]): ``{"prompt", "reference"}`` pairs.
        gen_kw (dict): generation options (``target_words, max_new_tokens, temperature, top_p``).

    Returns:
        dict: quality metrics plus ``generation_seconds_total`` / ``generation_seconds_per_example`` (empty if no pairs).
    """
    if not pairs:
        print("No internal eval pairs available -- skipping.")
        return {}
    t0 = time.time()
    preds = gen_fn([p["prompt"] for p in pairs], model, tokenizer, device, **gen_kw)
    sec = time.time() - t0
    m = compute_generation_quality(preds, [p["reference"] for p in pairs])
    m["generation_seconds_total"] = sec
    m["generation_seconds_per_example"] = sec / len(pairs)
    print("Internal generation-quality metrics:")
    [print(f"  {k}: {v:.4f}") for k, v in m.items()]
    return m


def decoding_sweep(
    gen_fn, model, tokenizer, device, pairs, n=10, target_words=128, max_new_tokens=384
):
    """Compare decoding strategies on a small subset (assignment section 7.4).

    Args:
        gen_fn (Callable): generator.
        model (nn.Module): model.
        tokenizer (PreTrainedTokenizer): tokenizer.
        device (torch.device): device.
        pairs (list[dict]): eval pairs.
        n (int): number of pairs used.
        target_words (int): words kept.
        max_new_tokens (int): generation budget.

    Returns:
        dict: config name -> metrics.
    """
    sub = pairs[:n]
    res = {}
    for c in DECODING_CONFIGS:
        preds = gen_fn(
            [p["prompt"] for p in sub],
            model,
            tokenizer,
            device,
            temperature=c["temperature"],
            top_p=c["top_p"],
            target_words=target_words,
            max_new_tokens=max_new_tokens,
        )
        m = compute_generation_quality(preds, [p["reference"] for p in sub])
        res[c["name"]] = m
        print(
            f"  {c['name']:>20s} | rougeL={m['rougeL']:.4f} bleu={m['bleu']:.2f} distinct_2={m['distinct_2']:.3f} repetition={m['repetition_rate']:.3f}"
        )
    return res
