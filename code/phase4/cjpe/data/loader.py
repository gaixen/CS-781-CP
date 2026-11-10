"""Download and normalise the CJPE dataset from the Hugging Face Hub."""
import numpy as np

from ..config import HF_DATASET_REPO, DATA_FILES
from ..context import setup_hf_auth
from ..utils import load_json

_RAW_DOCS = None


def norm_docs(raw):
    """Normalise raw JSON records to ``{"id", "text", "label"}`` dicts.

    Args:
        raw (list or dict): list of records, or a dict whose values are records.
            Label is read from ``label`` (or ``pred``); id from ``id`` (or ``case_id``).

    Returns:
        list[dict]: documents with ``id`` (str), ``text`` (str) and ``label`` (int, -1 if unlabelled).
    """
    if isinstance(raw, dict): raw = list(raw.values())
    out = []
    for d in raw:
        lab = d.get("label", d.get("pred"))
        out.append({"id": str(d.get("id", d.get("case_id"))), "text": d["text"], "label": int(lab) if lab is not None else -1})
    return out


def get_raw_docs(verbose=True):
    """Download (once) and return the train/dev/test documents.

    Args:
        verbose (bool): print split sizes, positive rates and length statistics on first load.

    Returns:
        dict[str, list[dict]]: ``{"train": [...], "dev": [...], "test": [...]}``.
    """
    global _RAW_DOCS
    if _RAW_DOCS is not None:
        return _RAW_DOCS
    from huggingface_hub import hf_hub_download
    token = setup_hf_auth()
    paths = {k: hf_hub_download(repo_id=HF_DATASET_REPO, filename=v, repo_type="dataset", token=token) for k, v in DATA_FILES.items()}
    _RAW_DOCS = {k: norm_docs(load_json(paths[k])) for k in ("train", "dev", "test")}
    if verbose:
        tr, dv, te = _RAW_DOCS["train"], _RAW_DOCS["dev"], _RAW_DOCS["test"]
        print(f"train={len(tr)} dev={len(dv)} test={len(te)}")
        print("positive rate  train: %.3f  dev: %.3f" % (np.mean([d["label"] for d in tr]), np.mean([d["label"] for d in dv])))
        chars = np.array([len(d["text"]) for d in tr])
        print("train chars  median=%d  p90=%d  max=%d" % (np.median(chars), np.percentile(chars, 90), chars.max()))
    return _RAW_DOCS
