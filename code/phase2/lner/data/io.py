"""Dataset file location, gold parsing and offset-convention detection."""
import glob
import json
import random

from ..config import DATA_ROOT, HF_DATASET_REPO


def load_json(p):
    """Load a UTF-8 JSON file.

    Args:
        p (str): path.

    Returns:
        Any: parsed JSON.
    """
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def locate(suffix):
    """Find a dataset file: first under ``DATA_ROOT``, else download it from the HF dataset repo.

    Args:
        suffix (str): file name suffix, e.g. ``"fold1-text.json"`` or ``"labels.txt"``.

    Returns:
        str: local path of the file.

    Raises:
        FileNotFoundError: if the file is not found anywhere.
    """
    if DATA_ROOT:
        hits = glob.glob(f"{DATA_ROOT}/**/{suffix}", recursive=True)
        if hits: return hits[0]
    from huggingface_hub import hf_hub_download, list_repo_files
    cand = [f for f in list_repo_files(HF_DATASET_REPO, repo_type="dataset") if f.endswith(suffix) and "Phase_2" in f]
    if not cand: raise FileNotFoundError(f"{suffix} not found (set LNER_DATA_ROOT or check the repo layout)")
    return hf_hub_download(HF_DATASET_REPO, cand[0], repo_type="dataset")


def parse_gold(raw, labels):
    """Parse gold annotations exactly as stored (no offset normalisation yet).

    Accepts ``[s, e, label(, text)]`` lists (any order of int/str fields) or dicts with start/end/label/text keys.

    Args:
        raw (dict or list): ``{doc_id: [entities]}`` or a list of ``{"id":..., "entities":[...]}``.
        labels (list[str]): valid labels.

    Returns:
        dict: ``{doc_id: [(start, end, label, text_or_None)]}``.

    Raises:
        ValueError: if an entity has an unknown label.
    """
    if isinstance(raw, list):   # list of {"id":..., "entities":[...]}
        raw = {str(r["id"]): r.get("entities", r.get("annotations", [])) for r in raw}
    lab_set = set(labels); out = {}
    def pick(d, keys):
        for k in keys:
            if k in d: return d[k]
    for did, ents in raw.items():
        lst = []
        for e in ents:
            if isinstance(e, dict):
                s = pick(e, ("start", "start_char", "start_offset", "begin")); en = pick(e, ("end", "end_char", "end_offset"))
                lab = pick(e, ("label", "type", "tag")); txt = pick(e, ("text", "entity_text", "entity", "span_text"))
            else:
                ints = [x for x in e if isinstance(x, int) and not isinstance(x, bool)]
                strs = [x for x in e if isinstance(x, str)]
                s, en = ints[0], ints[1]
                lab = next((x for x in strs if x in lab_set), None)
                txt = next((x for x in strs if x != lab), None)
            if lab not in lab_set: raise ValueError(f"unknown label {lab!r} in doc {did}; labels={labels}")
            lst.append((int(s), int(en), lab, txt))
        out[str(did)] = lst
    return out


def detect_end_inclusive(texts, gold, verbose=True):
    """Decide whether ``end`` offsets are exclusive (``text[s:e]``) or inclusive (``text[s:e+1]``) from stored entity text.

    Args:
        texts (dict): doc_id -> text.
        gold (dict): output of ``parse_gold``.
        verbose (bool): print match counts.

    Returns:
        bool: True if end offsets are inclusive.
    """
    excl = incl = tot = 0
    for d, ents in gold.items():
        for s, e, l, t in ents:
            if t is None or d not in texts: continue
            tot += 1
            excl += texts[d][s:e] == t
            incl += texts[d][s:e + 1] == t
    if verbose: print(f"offset check on {tot} entities: end-exclusive matches={excl}, end-inclusive matches={incl}")
    if tot == 0 or max(excl, incl) == 0:
        print("WARNING: could not verify offset convention from entity text -- inspect a few examples by hand!")
        return False
    return incl > excl


def split_docs(doc_ids, val_frac, seed=42):
    """Deterministic doc-level train/val split.

    Args:
        doc_ids (Iterable[str]): all doc ids.
        val_frac (float): validation fraction (at least one doc).
        seed (int): shuffle seed.

    Returns:
        tuple[list[str], list[str]]: (train ids, val ids).
    """
    ids = sorted(doc_ids); random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]
