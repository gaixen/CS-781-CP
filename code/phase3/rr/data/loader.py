"""Load the Phase_3_RR data (CL + IT domains) from a local folder or HuggingFace."""
import glob
import json
from collections import Counter

import numpy as np

from rr.config import OFFICIAL
from rr.context import RRData


def load_json(p):
    """Read a json file.

    Args:
        p (str): Path.

    Returns:
        Any: Parsed content.
    """
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def locate(settings, suffix):
    """Find a data file by suffix in ``settings.data_root`` or download it from the HuggingFace dataset repo.

    Args:
        settings (Settings): Settings (``data_root``, ``hf_dataset_repo``).
        suffix (str): File name such as ``CL_train.json``.

    Returns:
        str: Local path.
    """
    if settings.data_root:
        hits = glob.glob(f"{settings.data_root}/**/{suffix}", recursive=True)
        if hits:
            return hits[0]
    from huggingface_hub import hf_hub_download, list_repo_files
    cand = [f for f in list_repo_files(settings.hf_dataset_repo, repo_type="dataset") if f.endswith("/" + suffix) and "Phase_3" in f]
    if not cand:
        raise FileNotFoundError(f"{suffix} not found under Phase_3 (set data_root or check the repo layout)")
    return hf_hub_download(settings.hf_dataset_repo, cand[0], repo_type="dataset")


def norm_vocab(raw):
    """Normalise the label vocabulary json into an ordered list of names.

    Args:
        raw (Union[dict, list]): Parsed ``label_vocab.json``.

    Returns:
        List[str]: Labels in index order.
    """
    if isinstance(raw, dict) and "labels" in raw:
        raw = raw["labels"]
    if isinstance(raw, dict):
        if all(isinstance(v, int) for v in raw.values()):
            return [k for k, _ in sorted(raw.items(), key=lambda kv: kv[1])]
        if all(str(k).isdigit() for k in raw):
            return [v for _, v in sorted(raw.items(), key=lambda kv: int(kv[0]))]
        return list(raw)
    return list(raw)


def norm_docs(raw, domain):
    """Normalise raw json into documents.

    Accepts a list of ``{id, text, labels}`` or a dict ``id -> {text, labels} | [sentences]``.

    Args:
        raw (Union[dict, list]): Parsed json.
        domain (str): ``CL`` or ``IT``.

    Returns:
        List[dict]: Documents with ``id, domain, sents, labels`` (labels None for test).
    """
    items = raw.items() if isinstance(raw, dict) else [(r.get("id", i), r) for i, r in enumerate(raw)]
    out = []
    for did, r in items:
        if isinstance(r, dict):
            sents, labs, did = r.get("text", r.get("sentences")), r.get("labels"), r.get("id", did)
        else:
            sents, labs = r, None
        assert isinstance(sents, list) and (labs is None or len(labs) == len(sents)), f"doc {did}: bad structure / length mismatch"
        out.append(dict(id=str(did), domain=domain, sents=[str(s) for s in sents], labels=list(labs) if labs is not None else None))
    return out


def load_data(settings):
    """Load labels and all CL / IT train / dev / test splits.

    Args:
        settings (Settings): Settings.

    Returns:
        RRData: Documents and label vocabulary.
    """
    labels = norm_vocab(load_json(locate(settings, "label_vocab.json")))
    if set(labels) != set(OFFICIAL):
        print("WARNING: label_vocab differs from the 13 official labels:", set(labels) ^ set(OFFICIAL))
    l2i = {l: i for i, l in enumerate(labels)}
    splits = {(dom, sp): norm_docs(load_json(locate(settings, f"{dom}_{sp}.json")), dom) for dom in ("CL", "IT") for sp in ("train", "dev", "test")}
    train, dev, test = (splits[("CL", s)] + splits[("IT", s)] for s in ("train", "dev", "test"))
    for (dom, sp), ds in splits.items():
        print(f"{dom} {sp:5s}: {len(ds):3d} docs, {sum(len(d['sents']) for d in ds):6d} sentences")
    for d in train + dev:
        assert all(l in l2i for l in d["labels"]), f"unknown label in doc {d['id']}"
    ids = [d["id"] for d in test]
    if len(ids) != len(set(ids)):
        print("!!! WARNING: duplicate doc ids between CL_test and IT_test -> the single prediction.json cannot hold both; ask the TAs how ids are keyed")
    return RRData(labels, l2i, splits, train, dev, test)


def inspect_data(data):
    """Print label distributions, label inertia and length statistics (paper checks).

    Args:
        data (RRData): Loaded data.

    Returns:
        None
    """
    for dom in ("CL", "IT"):
        cnt = Counter(l for d in data.splits[(dom, "train")] for l in d["labels"])
        tot = sum(cnt.values())
        print(f"\n{dom} train label distribution:", {l: f"{cnt[l]} ({100*cnt[l]/tot:.1f}%)" for l in data.labels})
    same = tot_p = 0
    for d in data.train:
        for a, b in zip(d["labels"], d["labels"][1:]):
            same += a == b
            tot_p += 1
    print(f"\nP(label of next sentence == label of this one) = {same/tot_p:.3f}   (paper: 0.88 -> label inertia)")
    lens = np.array([len(s.split()) for d in data.train for s in d["sents"]])
    print("sentence length in words: median", int(np.median(lens)), "| p95", int(np.percentile(lens, 95)), "| max", lens.max())
    print("sentences per doc: median", int(np.median([len(d["sents"]) for d in data.train])), "| max", max(len(d["sents"]) for d in data.train))
