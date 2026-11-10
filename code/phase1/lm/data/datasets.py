"""Download the corpus, build train/dev Arrow datasets (disk-backed) and the internal word-level eval pairs."""
import random

from .streaming import (make_seq2seq_examples, reservoir_dev_split, shuffled, source_seed, stream_train_texts, take,
                        tokenize_and_block_stream)
from ..config import HF_DATASET_REPO, TEST_FILE, TEST_FILE_ALT


def download_data(cfg, token=None):
    """Download the training sources named in ``cfg["data_sources"]`` and the test prompts.

    Args:
        cfg (dict): run config.
        token (str or None): HF token.

    Returns:
        tuple[dict, str]: (``{source: local path}``, local test-file path).
    """
    from huggingface_hub import hf_hub_download
    paths = {}
    for name, spec in cfg["data_sources"].items():
        paths[name] = hf_hub_download(repo_id=HF_DATASET_REPO, filename=spec["file"], repo_type="dataset", token=token); print(f"[{name}] downloaded -> {paths[name]}")
    try: test_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=TEST_FILE, repo_type="dataset", token=token)
    except Exception as e:
        print(f"Primary TEST_FILE path failed ({e}); trying TEST_FILE_ALT.")
        test_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=TEST_FILE_ALT, repo_type="dataset", token=token)
    print("Test file:", test_path); return paths, test_path


def split_sources(cfg, paths):
    """Document-level dev split per source plus the per-source token budget derived from the mixing weights.

    Args:
        cfg (dict): run config.
        paths (dict): source -> file path.

    Returns:
        tuple[dict, dict, dict]: (dev texts per source, dev indices per source, token budget per source).
    """
    dev_texts, dev_idx = {}, {}
    for name in cfg["data_sources"]:
        total, idx, texts = reservoir_dev_split(paths[name], cfg["dev_doc_fraction"], source_seed(cfg["seed"], name))
        dev_texts[name], dev_idx[name] = texts, idx
        print(f"[{name}] total={total}  train_docs={total - len(idx)}  dev_docs={len(idx)}")
    tw = sum(s["weight"] for s in cfg["data_sources"].values())
    budget = {n: int(cfg["train_tokens"] * (s["weight"] / tw)) for n, s in cfg["data_sources"].items()}
    print("Per-source training token budget:", {k: f"{v:,}" for k, v in budget.items()})
    return dev_texts, dev_idx, budget


def build_datasets(cfg, paths, dev_texts, dev_idx, budget, tokenizer):
    """Build the train / validation datasets (causal: fixed-size blocks; seq2seq: prompt/continuation pairs).

    ``Dataset.from_generator`` writes to an Arrow file and memory-maps it back, so the tokenised corpus is never fully in RAM.

    Args:
        cfg (dict): run config.
        paths (dict): source -> file path.
        dev_texts (dict): source -> dev texts.
        dev_idx (dict): source -> dev doc indices.
        budget (dict): source -> train token budget.
        tokenizer (PreTrainedTokenizer): tokenizer.

    Returns:
        tuple: ``(train_dataset, val_dataset)`` (HF ``Dataset``).
    """
    from datasets import Dataset
    seed, srcs = cfg["seed"], list(cfg["data_sources"])
    if not cfg["is_seq2seq"]:
        def train_block_gen():
            for name in srcs:
                texts = stream_train_texts(paths[name], dev_idx[name])
                for block in tokenize_and_block_stream(texts, budget[name], cfg["train_block_size"], tokenizer, add_eos=cfg["add_eos_between_docs"]):
                    yield {"input_ids": block}
        def val_block_gen():
            per = max(1, cfg["val_tokens"] // len(srcs))
            for name in srcs:
                for block in tokenize_and_block_stream(iter(dev_texts[name]), per, cfg["train_block_size"], tokenizer, add_eos=cfg["add_eos_between_docs"]):
                    yield {"input_ids": block}
        train = Dataset.from_generator(lambda: shuffled(train_block_gen(), buffer_size=20_000, seed=seed))
        val = Dataset.from_generator(lambda: shuffled(val_block_gen(), buffer_size=2_000, seed=seed + 1))
    else:
        rng = random.Random(seed)
        texts = (t for n in srcs for t in take(stream_train_texts(paths[n], dev_idx[n]), cfg["seq2seq_max_train_docs"]))
        tr = make_seq2seq_examples(texts, cfg["seq2seq_src_tokens"], cfg["seq2seq_tgt_tokens"], cfg["seq2seq_pairs_per_doc"], tokenizer, rng)
        va = make_seq2seq_examples([t for n in srcs for t in dev_texts[n]], cfg["seq2seq_src_tokens"], cfg["seq2seq_tgt_tokens"], 1, tokenizer, rng)
        rng.shuffle(tr); train, val = Dataset.from_list(tr), Dataset.from_list(va)
    print(train); print(val); return train, val


def make_word_level_eval_pairs(texts, prompt_words=512, ref_words=128, max_pairs=None, rng=None):
    """(512-word prompt, 128-word reference) pairs from held-out dev docs -- the official test format; stand-in for the private references.

    Args:
        texts (Iterable[str]): dev documents.
        prompt_words (int): prompt length in words.
        ref_words (int): reference length in words.
        max_pairs (int or None): maximum number of pairs.
        rng (random.Random or None): RNG used to shuffle the documents.

    Returns:
        list[dict]: ``{"prompt": str, "reference": str}``.
    """
    pairs, pool, need = [], list(texts), prompt_words + ref_words
    if rng is not None: rng.shuffle(pool)
    for text in pool:
        words = text.split()
        if len(words) < need: continue
        pairs.append({"prompt": " ".join(words[:prompt_words]), "reference": " ".join(words[prompt_words:prompt_words + ref_words])})
        if max_pairs is not None and len(pairs) >= max_pairs: break
    return pairs
