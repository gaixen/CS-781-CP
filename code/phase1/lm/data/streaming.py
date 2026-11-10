"""Bounded-memory streaming of the legal corpus: deterministic document-level train/dev split, token blocking, reservoir shuffling, seq2seq pairs."""
import json
import random
import zlib

import numpy as np


def stream_records(path):
    """Yield each document's text one at a time (never holds the whole JSON array in memory).

    Args:
        path (str): JSON file containing a list of ``{"text": ...}`` records.

    Returns:
        Iterator[str]: document texts.
    """
    import ijson
    with open(path, "r", encoding="utf-8") as f:
        for record in ijson.items(f, "item"):
            if isinstance(record, dict) and record.get("text"):
                yield record["text"]


def reservoir_dev_split(path, dev_fraction, seed):
    """Choose ``dev_fraction`` of the documents as dev (two streaming passes: count, then reservoir-sample by index).

    Splitting on whole documents (not lines/tokens) prevents the dev set from leaking context of documents that were trained on.

    Args:
        path (str): corpus file.
        dev_fraction (float): fraction of documents for dev.
        seed (int): RNG seed.

    Returns:
        tuple[int, set[int], list[str]]: (document count, dev indices, dev texts).
    """
    rng = random.Random(seed)
    total = sum(1 for _ in stream_records(path))
    target_dev = max(1, int(total * dev_fraction))
    reservoir = []  # list of (idx, text)
    for idx, text in enumerate(stream_records(path)):
        if len(reservoir) < target_dev:
            reservoir.append((idx, text))
        else:
            j = rng.randint(0, idx)
            if j < target_dev:
                reservoir[j] = (idx, text)
    return total, {idx for idx, _ in reservoir}, [text for _, text in reservoir]


def source_seed(seed, name):
    """Deterministic per-source seed (``zlib.crc32``; Python's ``hash()`` of a str is randomised per process, which broke the dev split across sessions).

    Args:
        seed (int): base seed.
        name (str): source name.

    Returns:
        int: seed.
    """
    return seed + zlib.crc32(name.encode()) % 10_000


def stream_train_texts(path, dev_idx):
    """Re-stream the file skipping the dev documents.

    Args:
        path (str): corpus file.
        dev_idx (set[int]): dev document indices.

    Returns:
        Iterator[str]: training texts.
    """
    for idx, text in enumerate(stream_records(path)):
        if idx not in dev_idx:
            yield text


def tokenize_and_block_stream(text_iter, token_budget, block_size, tokenizer, add_eos=True):
    """Yield fixed-size token blocks. Inserts EOS between documents; slices the buffer once per document.

    Args:
        text_iter (Iterable[str]): document texts.
        token_budget (int): stop after this many tokens.
        block_size (int): block length.
        tokenizer (PreTrainedTokenizer): tokenizer.
        add_eos (bool): append EOS after every document.

    Returns:
        Iterator[np.ndarray]: int32 blocks of ``block_size`` tokens.
    """
    eos = tokenizer.eos_token_id
    buffer, total_tokens = [], 0
    for text in text_iter:
        if total_tokens >= token_budget:
            break
        ids = tokenizer.encode(text, add_special_tokens=False)
        if add_eos:
            ids.append(eos)
        ids = ids[: token_budget - total_tokens]
        buffer.extend(ids)
        total_tokens += len(ids)
        n_full = len(buffer) // block_size
        for b in range(n_full):
            yield np.asarray(buffer[b * block_size:(b + 1) * block_size], dtype=np.int32)
        buffer = buffer[n_full * block_size:]


def shuffled(source_gen, buffer_size, seed):
    """Bounded-memory ('reservoir') shuffle: swap each new item into a random slot of a fixed buffer and yield the evicted one.

    Args:
        source_gen (Iterable): items.
        buffer_size (int): buffer size.
        seed (int): RNG seed.

    Returns:
        Iterator: approximately shuffled items.
    """
    rng = random.Random(seed)
    buf = []
    for item in source_gen:
        if len(buf) < buffer_size:
            buf.append(item)
        else:
            j = rng.randrange(buffer_size)
            yield buf[j]
            buf[j] = item
    rng.shuffle(buf)
    yield from buf


def make_seq2seq_examples(texts, src_tokens, tgt_tokens, pairs_per_doc, tokenizer, rng):
    """Slide a (src_tokens -> tgt_tokens) window over each document (mirrors the test-time prompt -> continuation shape).

    Args:
        texts (Iterable[str]): documents.
        src_tokens (int): source length.
        tgt_tokens (int): target length.
        pairs_per_doc (int): max windows sampled per document.
        tokenizer (PreTrainedTokenizer): tokenizer.
        rng (random.Random): RNG.

    Returns:
        list[dict]: ``{"input_ids": [...], "labels": [...]}`` examples.
    """
    examples = []
    need = src_tokens + tgt_tokens
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < need:
            continue
        max_start = len(ids) - need
        n_windows = min(pairs_per_doc, max_start // max(1, tgt_tokens) + 1)
        starts = sorted(rng.sample(range(0, max_start + 1), k=min(n_windows, max_start + 1)))
        for start in starts:
            src = ids[start:start + src_tokens]
            tgt = ids[start + src_tokens:start + src_tokens + tgt_tokens]
            examples.append({"input_ids": src, "labels": tgt})
    return examples


def take(it, n):
    """First ``n`` items of an iterator.

    Args:
        it (Iterable): items.
        n (int): maximum count.

    Returns:
        Iterator: at most ``n`` items.
    """
    for i, x in enumerate(it):
        if i >= n:
            break
        yield x
