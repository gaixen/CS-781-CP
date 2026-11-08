"""Tokenisation with the Phase-1 GPT-2 tokenizer (cached) and construction of the per-split containers."""
import os
import pickle
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from pcr.utils import md5_text


def get_tokenizer(cfg):
    """Load the Phase-1 tokenizer from the first available checkpoint (falls back to ``gpt2``).

    Args:
        cfg (Config): Configuration providing ``phase1_ckpt``.

    Returns:
        Tuple[PreTrainedTokenizer, int]: ``(tokenizer, eos_token_id)``.
    """
    tok = None
    for p in [v for v in cfg.phase1_ckpt.values() if v] + ["gpt2"]:
        try:
            tok = AutoTokenizer.from_pretrained(p, local_files_only=(p != "gpt2"))
            print("tokenizer from:", p)
            break
        except Exception:
            continue
    if tok is None:
        raise RuntimeError("no tokenizer")
    tok.model_max_length = int(1e12)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 50256
    return tok, eos


def tokenize_docs(docs, tag, tok, eos, cfg):
    """Tokenise documents (head 70 % + tail 30 % of ``max_chars``) with an on-disk cache.

    Args:
        docs (List[dict]): Records with ``id`` and ``text``.
        tag (str): Cache tag (e.g. ``dev_q``).
        tok (PreTrainedTokenizer): Tokenizer.
        eos (int): Token id used for empty documents.
        cfg (Config): Configuration (``max_chars``, ``cache_dir``).

    Returns:
        Dict[str, np.ndarray]: ``doc_id -> int32 token ids``.
    """
    key = md5_text(f"{tag}|{len(docs)}|{md5_text(docs[0]['text']) if docs else ''}|{cfg.max_chars}")[:10]
    f = f"{cfg.cache_dir}/tok_{tag}_{key}.pkl"
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    out, t0 = {}, time.time()
    mc = cfg.max_chars
    for i in range(0, len(docs), 64):
        b = docs[i:i + 64]
        txt = [(d["text"] if len(d["text"]) <= mc else d["text"][:int(mc * .7)] + " " + d["text"][-int(mc * .3):]) or "." for d in b]
        for d, ids in zip(b, tok(txt, add_special_tokens=False)["input_ids"]):
            out[d["id"]] = np.asarray(ids if ids else [eos], np.int32)
    pickle.dump(out, open(f, "wb"))
    print(f"tokenised {tag}: {len(out)} docs in {time.time()-t0:.0f}s")
    return out


def build_splits(raw, tok, eos, cfg):
    """Build the per-split container used everywhere downstream.

    Each split uses only its own candidate pool and a query can never retrieve itself. In smoke mode the
    queries are truncated to ``cfg.smoke_q`` and the pool is reduced to the gold documents plus 300 random extras.

    Args:
        raw (Dict[str, List[dict]]): Preprocessed records.
        tok (PreTrainedTokenizer): Tokenizer.
        eos (int): EOS token id.
        cfg (Config): Configuration.

    Returns:
        Dict[str, dict]: ``spl[split]`` with keys ``qids, cids, qtok, ctok, qtext, ctext, gold`` (gold is None for test).
    """
    spl = {}
    rng_s = np.random.RandomState(cfg.seed)
    for s in ("train", "dev", "test"):
        Q, C = raw[s], raw[s + "_c"]
        if cfg.smoke:
            Q = Q[:cfg.smoke_q]
            if s != "test":
                need = {g for d in Q for g in d["gold"]}
                extra = [d["id"] for d in C if d["id"] not in need]
                keep = need | set(rng_s.choice(extra, min(300, len(extra)), replace=False))
                C = [d for d in C if d["id"] in keep]
        qset = {q["id"] for q in Q}
        C = [d for d in C if d["id"] not in qset]          # a query can never retrieve itself
        tq, tcand = tokenize_docs(Q, s + "_q", tok, eos, cfg), tokenize_docs(C, s + "_c", tok, eos, cfg)
        cset = {c["id"] for c in C}
        spl[s] = dict(qids=[d["id"] for d in Q], cids=[d["id"] for d in C], qtok=tq, ctok=tcand,
                      qtext={d["id"]: d["text"] for d in Q}, ctext={d["id"]: d["text"] for d in C},
                      gold=({d["id"]: set(g for g in d["gold"] if g in cset) for d in Q} if s != "test" else None))
        print(f"{s}: queries={len(Q)} pool={len(C)} | tokens/doc: q median={np.median([len(tq[i]) for i in tq]):.0f} "
              f"cand median={np.median([len(tcand[i]) for i in tcand]):.0f}")
    return spl


def truncation_table(spl, budgets=(512, 1024, 2048, 4096, 8192)):
    """Show how much of the dev candidate documents each token budget keeps.

    Args:
        spl (dict): Split container from :func:`build_splits`.
        budgets (Iterable[int]): Token budgets to evaluate.

    Returns:
        pd.DataFrame: ``% dev cand docs > budget`` and ``% tokens kept`` per budget.
    """
    dl = np.array([len(x) for x in spl["dev"]["ctok"].values()])
    tab = pd.DataFrame({b: {"% dev cand docs > budget": 100 * (dl > b).mean(), "% tokens kept": 100 * np.minimum(dl, b).sum() / dl.sum()}
                        for b in budgets}).round(1).T
    print(tab.to_string())
    return tab
