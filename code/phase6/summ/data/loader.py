"""Load Phase-6 data, tokenise with the Phase-1 GPT-2 tokenizer, clean, and split train / dev / test."""
import hashlib
import json
import os
import pickle
import time

import numpy as np

from summ.context import Context


def md5(t):
    """MD5 hex digest of a text (None = empty).

    Args:
        t (Optional[str]): Text.

    Returns:
        str: Digest.
    """
    return hashlib.md5((t or "").encode("utf-8", "ignore")).hexdigest()


def load_json(p):
    """Read a json file.

    Args:
        p (str): Path.

    Returns:
        Any: Parsed content.
    """
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def locate_data(s, token=None):
    """Resolve the train / test json paths.

    Args:
        s (Settings): Settings.
        token (Optional[str]): HuggingFace token.

    Returns:
        Dict[str, str]: ``{"train": path, "test": path}``.
    """
    if s.local_data:
        return s.local_data
    from huggingface_hub import hf_hub_download
    return {k: hf_hub_download(repo_id=s.hf_dataset_repo, filename=v, repo_type="dataset", token=token) for k, v in s.data_files.items()}


def norm(raw):
    """Normalise raw json (dict or list) into ``{id, rid, document, summary, keys}`` records.

    Args:
        raw (Union[dict, list]): Parsed json.

    Returns:
        List[dict]: Records (``summary`` is None when absent).
    """
    if isinstance(raw, dict):
        raw = [({**v, "id": v.get("id", k)} if isinstance(v, dict) else {"id": k, "document": v}) for k, v in raw.items()]
    out = []
    for i, d in enumerate(raw):
        doc = d.get("document", d.get("text", d.get("judgment")))
        summ = d.get("summary", d.get("reference_summary", d.get("abstract")))
        out.append(dict(id=str(d.get("id", i)), rid=d.get("id", i), document=doc, summary=summ, keys=tuple(sorted(d.keys()))))
    return out


def get_tokenizer(ctx):
    """Load the Phase-1 tokenizer (first checkpoint that has one, else ``gpt2``) and set EOS / vocabulary sizes.

    Args:
        ctx (Context): Context; ``tok, eos, nv`` are filled.

    Returns:
        None
    """
    from transformers import AutoTokenizer
    tok = None
    for p in [v for v in ctx.s.phase1_ckpt.values() if v] + ["gpt2"]:
        try:
            tok = AutoTokenizer.from_pretrained(p, local_files_only=(p != "gpt2"))
            print("tokenizer from:", p)
            break
        except Exception:
            continue
    if tok is None:
        raise RuntimeError("no tokenizer found")
    tok.model_max_length = int(1e12)
    ctx.tok, ctx.eos, ctx.nv = tok, (tok.eos_token_id if tok.eos_token_id is not None else 50256), len(tok)


def tok_many(ctx, texts, tag):
    """Tokenise texts (head 70 % + tail 30 % when longer than ``max_raw_tok``), cached on disk.

    Args:
        ctx (Context): Context with the tokenizer.
        texts (List[str]): Texts.
        tag (str): Cache tag.

    Returns:
        List[np.ndarray]: int32 token arrays.
    """
    s = ctx.s
    key = hashlib.md5(f"{tag}|{len(texts)}|{md5(texts[0]) if texts else ''}|{s.max_raw_tok}".encode()).hexdigest()[:10]
    f = f"{s.cache}/tok_{tag}_{key}.pkl"
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    cc, out, t0 = s.max_raw_tok * 7, [], time.time()
    for i in range(0, len(texts), 64):
        b = [(t if len(t) <= cc else t[:int(cc * .7)] + " " + t[-int(cc * .3):]) for t in texts[i:i + 64]]
        for ids in ctx.tok(b, add_special_tokens=False)["input_ids"]:
            if len(ids) > s.max_raw_tok:
                h = int(s.max_raw_tok * .7)
                ids = ids[:h] + ids[-(s.max_raw_tok - h):]
            out.append(np.asarray(ids if ids else [ctx.eos], np.int32))
    pickle.dump(out, open(f, "wb"))
    print(f"tokenised {tag}: {len(out)} in {time.time()-t0:.0f}s")
    return out


def load_raw(ctx):
    """Load the raw train / test rows.

    Args:
        ctx (Context): Context; ``paths, train_all, test_docs`` are filled.

    Returns:
        None
    """
    ctx.paths = locate_data(ctx.s, ctx.hf_token)
    raw_tr, raw_te = load_json(ctx.paths["train"]), load_json(ctx.paths["test"])
    ctx.train_all, ctx.test_docs = norm(raw_tr), norm(raw_te)
    print(f"train rows={len(ctx.train_all)}  test rows={len(ctx.test_docs)}")
    return None


def finish_tokenisation(ctx):
    """Tokenise documents and summaries (after the integrity audit has run).

    Args:
        ctx (Context): Context with ``train_all`` / ``test_docs``.

    Returns:
        Tuple[list, list, list]: ``(train_doc_ids, train_sum_ids, test_doc_ids)``.
    """
    is_str = lambda x: isinstance(x, str)
    for d in ctx.train_all + ctx.test_docs:
        d["document"] = d["document"] if is_str(d["document"]) else ""
    for d in ctx.train_all:
        d["summary"] = d["summary"] if is_str(d["summary"]) else ""
    return (tok_many(ctx, [d["document"] for d in ctx.train_all], "trdoc"), tok_many(ctx, [d["summary"] for d in ctx.train_all], "trsum"),
            tok_many(ctx, [d["document"] for d in ctx.test_docs], "tedoc"))


def clean_and_split(ctx, tr_doc_ids, tr_sum_ids, te_doc_ids):
    """Cleaning, validation split and generation-length settings derived from the data (section 4.5).

    Removes unusable pairs, any train doc that is also in test, and exact duplicate pairs; carves ``dev_n`` documents as dev and sets
    ``ctx.s.gen`` (max_new = p90 of reference length, min_new = half the median). Fills ``ctx.tr / dev / te``.

    Args:
        ctx (Context): Context.
        tr_doc_ids (list): Tokenised train documents.
        tr_sum_ids (list): Tokenised train summaries.
        te_doc_ids (list): Tokenised test documents.

    Returns:
        None
    """
    s, train_all, test_docs = ctx.s, ctx.train_all, ctx.test_docs
    th, tid = {md5(d["document"]) for d in test_docs}, {d["id"] for d in test_docs}
    keep, seen = [], set()
    for i, d in enumerate(train_all):
        h = md5(d["document"])
        if len(d["document"].strip()) < 200 or len(d["summary"].strip()) < 20:
            continue
        if h in th or d["id"] in tid:
            continue
        if (h, md5(d["summary"])) in seen:
            continue
        seen.add((h, md5(d["summary"])))
        keep.append(i)
    print(f"kept {len(keep)}/{len(train_all)} training pairs after cleaning")
    perm = np.random.RandomState(s.seed).permutation(len(keep))
    dev_ix, tr_ix = [keep[i] for i in perm[:s.dev_n]], [keep[i] for i in perm[s.dev_n:]]
    if s.smoke:
        tr_ix = tr_ix[:64]
    mk = lambda ix: dict(id=[train_all[i]["id"] for i in ix], doc_ids=[tr_doc_ids[i] for i in ix], sum_ids=[tr_sum_ids[i] for i in ix],
                         doc=[train_all[i]["document"] for i in ix], ref=[train_all[i]["summary"] for i in ix])
    ctx.tr, ctx.dev = mk(tr_ix), mk(dev_ix)
    ctx.te = dict(id=[d["id"] for d in test_docs], rid=[d["rid"] for d in test_docs], doc_ids=te_doc_ids, doc=[d["document"] for d in test_docs])
    if s.smoke:
        ctx.te = {k: v[:8] for k, v in ctx.te.items()}
    print(f"train={len(ctx.tr['id'])} dev={len(ctx.dev['id'])} (eval subset {s.dev_eval_n}) test={len(ctx.te['id'])}")
    st = np.array([len(x) for x in ctx.tr["sum_ids"]])
    med, p90, p95 = np.median(st), np.percentile(st, 90), np.percentile(st, 95)
    s.gen.update(max_new=int(min(s.sum_tok - 1, max(64, p90))), min_new=int(0.5 * med))
    print(f"reference summary tokens: median={med:.0f} p90={p90:.0f} p95={p95:.0f} | sum_tok={s.sum_tok} truncates {100*(st>=s.sum_tok-1).mean():.1f}% of targets | gen set to {s.gen}")
    if p95 > s.sum_tok:
        print(f"NOTE: p95 summary length {p95:.0f} > sum_tok={s.sum_tok}; raise sum_tok (and lower doc_tok so doc_tok+sum_tok<=2048) to model the long tail.")
    assert s.doc_tok + s.sum_tok <= 2048 + 16, "Phase-1 context is 2048 (RoPE cache is extended slightly, but extrapolation is untested)"
