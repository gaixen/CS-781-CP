"""Deterministic preprocessing (v2).

Dense models were found to match on *header format* (PETITIONER / CITATION / CITATOR INFO blocks) instead of legal
content. Fixes implemented here: (1) parse year / citation keys / party names from the header **then strip it**,
(2) drop anonymisation placeholders. The parsed metadata later feeds the rule features (chronology, citation match,
case-name match).
"""
import re
from collections import Counter

import pandas as pd

PH = re.compile(r"<[A-Z_]{2,10}>")
MARK = re.compile(r"(?<!OF )\b(ACT\s*:|HEADNOTE\s*:|JUDGMENT\s*:|JUDGEMENT\s*:)")
PSTOP = set("and ors anr others other etc vs v versus the through its by of rep represented".split())
GENERIC = set("state union india government commissioner income tax company limited ltd pvt co corporation board municipal central bank another ors".split())


def normt(s):
    """Normalise text: lower-case, keep ``[a-z0-9 ]`` and collapse whitespace.

    Args:
        s (str): Input text.

    Returns:
        str: Normalised text.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def cite_keys(t):
    """Extract reporter citation keys (AIR / SCC / SCR) from a text.

    Args:
        t (str): Text to scan.

    Returns:
        set: Tuples such as ``("AIR", year, page)`` or ``("SCC", year, volume, page)``.
    """
    k = set()
    for y, p in re.findall(r"\bAIR\s+(\d{4})\s+(?:SC|S\.C\.|SUPREME COURT|SUP\.? ?CT\.?)?\s*(\d{1,4})\b", t):
        k.add(("AIR", y, p))
    for y, p in re.findall(r"\b(\d{4})\s+AIR\s+(\d{1,4})\b", t):
        k.add(("AIR", y, p))
    for y, v, p in re.findall(r"\((\d{4})\)\s*(\d+)\s*S\.?C\.?C\.?\s*(\d+)", t):
        k.add(("SCC", y, v, p))
    for y, v, p in re.findall(r"\b(\d{4})\s*S\.?C\.?C\.?\s*\((\d+)\)\s*(\d+)", t):
        k.add(("SCC", y, v, p))
    for y, v, p in re.findall(r"\b(\d{4})\s*\((\d+)\)\s*S\.?C\.?C\.?\s*(\d+)", t):
        k.add(("SCC", y, v, p))
    for y, v, p in re.findall(r"[\[\(]?(\d{4})[\]\)]?\s*(\d+)\s*S\.?C\.?R\.?\s*(\d+)", t):
        k.add(("SCR", y, v, p))
    for y, v, p in re.findall(r"\b(\d{4})\s*S\.?C\.?R\.?\s*\((\d+)\)\s*(\d+)", t):
        k.add(("SCR", y, v, p))
    return k


def party_key(s):
    """Build a distinctive case-name key from a party string.

    Args:
        s (str): Petitioner or respondent string.

    Returns:
        Optional[str]: First (up to 5) informative words, or None for generic / short names.
    """
    w = [x for x in normt(s).split() if x not in PSTOP][:5]
    if len(w) == 0 or all(x in GENERIC for x in w) or len(" ".join(w)) < 8:
        return None
    return " ".join(w)


def parse_meta(t):
    """Parse year, citation keys and party-name keys from a case header.

    Args:
        t (str): Raw case text.

    Returns:
        Tuple[dict, bool]: ``(meta, had_header)`` where ``meta`` has ``year``, ``keys`` and ``nkey``.
    """
    head = t[:6000]
    m = dict(year=None, keys=set(), nkey=[])
    if not re.match(r"\s*PETITIONER\s*:", t):
        return m, False
    d = re.search(r"DATE OF JUDGMENT\s*:?\s*\d{1,2}\s*/\s*\d{1,2}\s*/\s*(\d{4})", head)
    cit = re.search(r"CITATION\s*:\s*(.*?)(?:CITATOR INFO|ACT\s*:|HEADNOTE|$)", head, re.S)
    m["keys"] = cite_keys(cit.group(1)) if cit else set()
    if d:
        m["year"] = int(d.group(1))
    elif m["keys"]:
        m["year"] = int(sorted(m["keys"])[0][1])
    pm = re.search(r"PETITIONER\s*:\s*(.*?)\s+Vs\.?\s+RESPONDENT\s*:\s*(.*?)\s+DATE OF JUDGMENT", head, re.S | re.I)
    if pm:
        m["nkey"] = [k for k in (party_key(pm.group(1)), party_key(pm.group(2))) if k]
    return m, True


def strip_header(t):
    """Remove the PETITIONER ... CITATOR INFO block that precedes ACT / HEADNOTE / JUDGMENT.

    Args:
        t (str): Raw case text.

    Returns:
        str: Text starting at the first marker (unchanged if no header is present).
    """
    if re.match(r"\s*PETITIONER\s*:", t):
        mm = MARK.search(t, 0, 8000)
        if mm:
            t = t[mm.start():]
    return t


def preprocess_splits(raw, clean=True):
    """Parse metadata and clean the text of every split / pool (in place).

    For every query and candidate the header metadata is stored in ``meta[tag][id]`` and
    ``doc["text"]`` is replaced by the cleaned body. Queries additionally get ``body_keys`` (citation keys
    found in the body) and dev / test queries get ``norm`` (normalised text for case-name matching).

    Args:
        raw (Dict[str, List[dict]]): Records from ``load_raw_splits`` (keys train, train_c, dev, dev_c, test, test_c).
        clean (bool): Strip header boilerplate and ``<NAME>``-style placeholders.

    Returns:
        Tuple[dict, Counter]: ``(meta, stats)`` with ``meta[tag][doc_id] -> dict`` and counters for reporting.
    """
    meta, st = {}, Counter()
    for s in ("train", "dev", "test"):
        for tag, docs in ((s, raw[s]), (s + "_c", raw[s + "_c"])):
            meta[tag] = {}
            for d in docs:
                m, had = parse_meta(d["text"])
                n0 = len(d["text"])
                body = strip_header(d["text"]) if clean else d["text"]
                body = PH.sub(" ", body) if clean else body
                body = re.sub(r"\s+", " ", body).strip() or "."
                if not tag.endswith("_c"):
                    m["body_keys"] = cite_keys(body[:400000])
                    if s != "train":
                        m["norm"] = normt(body[:150000])
                st[tag + "_header"] += had
                st[tag + "_year"] += m["year"] is not None
                st[tag + "_n"] += 1
                st[tag + "_chars_before"] += n0
                st[tag + "_chars_after"] += len(body)
                d["text"] = body
                meta[tag][d["id"]] = m
    return meta, st


def print_preprocess_summary(raw, meta, st):
    """Print the header-parsing / cleaning statistics and one cleaned example.

    Args:
        raw (dict): Records after :func:`preprocess_splits`.
        meta (dict): Metadata returned by :func:`preprocess_splits`.
        st (Counter): Statistics returned by :func:`preprocess_splits`.

    Returns:
        None
    """
    print(pd.DataFrame({t: dict(docs=st[t + "_n"], header_parsed_pct=100 * st[t + "_header"] / st[t + "_n"],
                                year_known_pct=100 * st[t + "_year"] / st[t + "_n"],
                                chars_removed_pct=100 * (1 - st[t + "_chars_after"] / st[t + "_chars_before"])) for t in meta}).round(1).T.to_string())
    ex = next((d for d in raw["dev"] if meta["dev"][d["id"]]["year"]), None)
    if ex is not None:
        print("\nexample meta:", {k: (list(v)[:3] if isinstance(v, set) else v) for k, v in meta["dev"][ex["id"]].items() if k != "norm"},
              "\nexample cleaned start:", ex["text"][:300])
