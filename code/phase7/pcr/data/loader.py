"""Load the PCR splits (queries + candidate pools) from HuggingFace or local json files."""
import json
from collections import Counter


def locate_data_paths(cfg, hf_token=None):
    """Resolve the six data files (train / dev / test queries and candidate pools).

    Args:
        cfg (Config): Configuration; uses ``local_data``, ``hf_dataset_repo`` and ``hf_prefix``.
        hf_token (Optional[str]): HuggingFace token used to download the files.

    Returns:
        Dict[str, str]: Mapping with keys ``train, train_c, dev, dev_c, test, test_c`` to local file paths.
    """
    if cfg.local_data:
        return cfg.local_data
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(cfg.hf_dataset_repo, repo_type="dataset", token=hf_token)

    def find(name):
        """Download the shortest-path file whose name ends with ``name``.

        Args:
            name (str): File name to look for (e.g. ``train.json``).

        Returns:
            str: Local path of the downloaded file.
        """
        c = [f for f in files if f.endswith("/" + name) and cfg.hf_prefix in f] or [f for f in files if f.endswith(name)]
        assert c, f"{name} not found in {cfg.hf_dataset_repo}: {files[:20]}"
        return hf_hub_download(repo_id=cfg.hf_dataset_repo, filename=sorted(c, key=len)[0], repo_type="dataset", token=hf_token)

    return {"train": find("train.json"), "train_c": find("train_candidates.json"), "dev": find("dev.json"),
            "dev_c": find("dev_candidates.json"), "test": find("test.json"), "test_c": find("test_candidates.json")}


def normalize_records(raw):
    """Convert a raw json object (dict or list) into a list of uniform records.

    Args:
        raw (Union[dict, list]): Parsed json content.

    Returns:
        List[dict]: Records with keys ``id, text, gold, nsent, keys``. ``gold`` is None for test queries.
    """
    items = [({**v, "id": v.get("id", k)} if isinstance(v, dict) else {"id": k, "text": v}) for k, v in raw.items()] if isinstance(raw, dict) else raw
    out = []
    for d in items:
        t = d.get("text", d.get("document", ""))
        ns = len(t) if isinstance(t, list) else None
        t = " ".join(map(str, t)) if isinstance(t, list) else (t or "")
        g = d.get("relevant_candidates")
        g = [str(x) for x in g] if g is not None else None
        out.append(dict(id=str(d["id"]), text=t, gold=g, nsent=ns, keys=tuple(sorted(d.keys()))))
    return out


def load_raw_splits(paths, keys=None):
    """Read and normalise the requested data files.

    Args:
        paths (Dict[str, str]): Output of :func:`locate_data_paths`.
        keys (Optional[Iterable[str]]): Subset of keys to load (default: all keys in ``paths``).

    Returns:
        Dict[str, List[dict]]: Normalised records per key.
    """
    keys = list(keys) if keys is not None else list(paths)
    return {k: normalize_records(json.load(open(paths[k], encoding="utf-8"))) for k in keys}


def describe_raw(raw):
    """Print the number of records and the dominant json key set of every split.

    Args:
        raw (Dict[str, List[dict]]): Output of :func:`load_raw_splits`.

    Returns:
        None
    """
    for k, v in raw.items():
        print(f"{k:8s} n={len(v):6d}  keys={Counter(d['keys'] for d in v).most_common(1)[0][0]}")
