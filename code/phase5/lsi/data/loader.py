"""Load the LSI data, remove leaks / duplicates from dev and build the multi-hot label matrices."""
import json
from collections import Counter

import numpy as np

from lsi.context import LSIData
from lsi.utils import md5_text


def locate_data(cfg, token=None):
    """Resolve the train / dev / test json paths (local files or HuggingFace download).

    Args:
        cfg (Config): Configuration (``local_data``, ``hf_dataset_repo``, ``data_files``).
        token (Optional[str]): HuggingFace token.

    Returns:
        Dict[str, str]: ``{"train": path, "dev": path, "test": path}``.
    """
    if cfg.local_data:
        return cfg.local_data
    from huggingface_hub import hf_hub_download
    return {k: hf_hub_download(repo_id=cfg.hf_dataset_repo, filename=v, repo_type="dataset", token=token) for k, v in cfg.data_files.items()}


def load_json(p):
    """Read a json file.

    Args:
        p (str): Path.

    Returns:
        Any: Parsed content.
    """
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def norm_docs(raw):
    """Normalise raw json (dict or list) into uniform document records.

    Args:
        raw (Union[dict, list]): Parsed json.

    Returns:
        List[dict]: Records with ``rid, id, text, labels, had_labels``.
    """
    if isinstance(raw, dict):
        raw = [({**v, "id": v.get("id", k)} if isinstance(v, dict) else v) for k, v in raw.items()]
    out = []
    for d in raw:
        lab = d.get("labels", d.get("label", d.get("pred", d.get("statutes"))))
        labs = [] if lab is None else sorted({int(x) for x in lab}) if isinstance(lab, (list, tuple, set)) else [int(lab)]
        rid = d.get("id", d.get("case_id"))
        out.append(dict(rid=rid, id=str(rid), text=d.get("text") or "", labels=labs, had_labels=lab is not None))
    return out


def to_Y(docs, l2i):
    """Multi-hot label matrix of a document list.

    Args:
        docs (List[dict]): Documents with ``labels``.
        l2i (Dict[int, int]): Statute id -> column.

    Returns:
        np.ndarray: float32 matrix (len(docs), L).
    """
    Y = np.zeros((len(docs), len(l2i)), np.float32)
    for i, d in enumerate(docs):
        for l in d["labels"]:
            Y[i, l2i[l]] = 1
    return Y


def load_data(cfg, token=None):
    """Load, clean and index the data.

    Dev rows that also occur in test, in train, or twice in dev are dropped. If fewer than ``cfg.min_clean_dev``
    rows survive, ``cfg.holdout_frac`` of train is held out as dev instead. Test labels are NEVER used.

    Args:
        cfg (Config): Configuration.
        token (Optional[str]): HuggingFace token.

    Returns:
        LSIData: Cleaned documents and label matrices.
    """
    paths = locate_data(cfg, token)
    train_docs, dev_docs, test_docs = (norm_docs(load_json(paths[k])) for k in ("train", "dev", "test"))
    n_test_labeled = sum(d["had_labels"] for d in test_docs)
    for d in test_docs:
        d["labels"] = []
    test_ids, test_h, train_h = {d["id"] for d in test_docs}, {md5_text(d["text"]) for d in test_docs}, {md5_text(d["text"]) for d in train_docs}
    keep, seen, drop = [], set(), Counter()
    for i, d in enumerate(dev_docs):
        h = md5_text(d["text"])
        if d["id"] in test_ids or h in test_h:
            drop["in_test"] += 1
            continue
        if h in train_h:
            drop["in_train"] += 1
            continue
        if d["id"] in seen or h in seen:
            drop["dup_in_dev"] += 1
            continue
        seen.add(d["id"])
        seen.add(h)
        keep.append(i)
    print(f"test.json carried labels for {n_test_labeled} cases (ignored). dev rows={len(dev_docs)} -> clean dev={len(keep)}; dropped: {dict(drop)}")
    if len(keep) >= cfg.min_clean_dev:
        dev_docs = [dev_docs[i] for i in keep]
    else:
        print(f"clean dev too small (<{cfg.min_clean_dev}) -> holding out {cfg.holdout_frac:.0%} of TRAIN as dev (dev.json is unusable)")
        perm = np.random.RandomState(cfg.seed).permutation(len(train_docs))
        nh = int(cfg.holdout_frac * len(train_docs))
        dev_docs = [train_docs[i] for i in perm[:nh]]
        train_docs = [train_docs[i] for i in perm[nh:]]
    if cfg.train_subset:
        train_docs = train_docs[:cfg.train_subset]
    if cfg.dev_subset:
        dev_docs = dev_docs[:cfg.dev_subset]
    if cfg.test_subset:
        test_docs = test_docs[:cfg.test_subset]
    labels = sorted({l for d in train_docs + dev_docs for l in d["labels"]})
    l2i = {l: i for i, l in enumerate(labels)}
    ytr, ydev = to_Y(train_docs, l2i), to_Y(dev_docs, l2i)
    tf = ytr.sum(0)
    print(f"train={len(train_docs)} dev={len(dev_docs)} test={len(test_docs)} | labels={len(labels)} | labels/doc train mean={ytr.sum(1).mean():.2f} max={int(ytr.sum(1).max())}")
    print(f"labels with 0 train docs: {(tf==0).sum()} | <=5: {(tf<=5).sum()} | <=20: {(tf<=20).sum()} | >100: {(tf>100).sum()} | docs with no label: train {(ytr.sum(1)==0).mean():.1%}, dev {(ydev.sum(1)==0).mean():.1%}")
    assert not any(d["id"] in test_ids for d in dev_docs), "dev still overlaps test"
    return LSIData(train_docs, dev_docs, test_docs, labels, l2i, ytr, ydev, tf, test_ids, paths)
