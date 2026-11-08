"""Deterministic rule features for prior-case retrieval.

* chronology (a prior case cannot be decided after the query), citation match, case-name match (section 4.5),
* forward / reverse citation links parsed from raw ``CITATION`` / ``CITATOR INFO`` headers, soft year rules,
* strict year / citation features computed from header + body citation keys (train / dev / test).
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from pcr.data.preprocessing import cite_keys
from pcr.retrieval.metrics import gold_matrix

YR = re.compile(r"\b(1[6-9]\d\d|20[0-2]\d)\b")


def rule_feats(split, spl, meta):
    """Chronology, citation-match and case-name-match matrices for one split.

    Args:
        split (str): ``train`` | ``dev`` | ``test``.
        spl (dict): Split container.
        meta (dict): Metadata from ``preprocess_splits``.

    Returns:
        Dict[str, np.ndarray]: ``future, gap, known, cite, name`` matrices (n_queries, n_candidates).
    """
    S = spl[split]
    qm, cm = meta[split], meta[split + "_c"]
    qids, cids = S["qids"], S["cids"]
    nq, nc = len(qids), len(cids)
    yq = np.array([qm[q]["year"] or 0 for q in qids])
    yc = np.array([cm[c]["year"] or 0 for c in cids])
    known = (yq[:, None] > 0) & (yc[None, :] > 0)
    future = ((yc[None, :] > yq[:, None]) & known).astype(np.float32)
    gap = np.where(known, np.clip((yq[:, None] - yc[None, :]) / 40.0, -1, 1.5), 0).astype(np.float32)
    inv = defaultdict(list)
    for j, c in enumerate(cids):
        for k in cm[c]["keys"]:
            inv[k].append(j)
    cite = np.zeros((nq, nc), np.float32)
    name = np.zeros((nq, nc), np.float32)
    for i, q in enumerate(qids):
        for k in qm[q]["body_keys"]:
            for j in inv.get(k, ()):
                cite[i, j] = 1
        nt = qm[q].get("norm")
        if nt:
            for j, c in enumerate(cids):
                nk = cm[c]["nkey"]
                if nk and all(x in nt for x in nk):
                    name[i, j] = 1
    return dict(future=future, gap=gap, known=known.astype(np.float32), cite=cite, name=name)


def diagnose_rule_features(rf_dev, spl):
    """Print how well each rule feature separates dev gold from non-gold pairs.

    Args:
        rf_dev (Dict[str, np.ndarray]): Output of :func:`rule_feats` for dev.
        spl (dict): Split container.

    Returns:
        pd.DataFrame: Per-feature diagnostics.
    """
    G = gold_matrix("dev", spl)
    rows = []
    for k, M in rf_dev.items():
        rows.append(dict(feature=k, mean_on_gold=M[G].mean(), mean_on_nongold=M[~G].mean(), pairs_flagged=int((M > 0).sum()),
                         gold_flagged=int((M[G] > 0).sum()), precision_of_flag=float(G[M > 0].mean()) if (M > 0).any() else np.nan))
    df = pd.DataFrame(rows).round(4)
    print(df.to_string(index=False))
    kn = rf_dev["known"] > 0
    print(f"\nchronology: pairs with both years known={kn.mean():.3f} | gold violating (cand year > query year)="
          f"{rf_dev['future'][G].sum()/max(kn[G].sum(), 1):.4f} vs non-gold={rf_dev['future'][~G].sum()/max(kn[~G].sum(), 1):.4f}")
    return df


# ---------------- raw-header features (forward / reverse citation, soft year) ----------------
def hdr(t):
    """Parse a RAW (unstripped) header into the case's own citation keys and its CITATOR INFO keys.

    Args:
        t (str): Raw case text.

    Returns:
        Tuple[set, set]: ``(own_keys, cited_by_keys)``; both empty when the text has no header.
    """
    if not re.match(r"\s*PETITIONER\s*:", t):
        return set(), set()
    h = t[:20000]
    cit = re.search(r"CITATION\s*:\s*(.*?)(?:CITATOR INFO|ACT\s*:|HEADNOTE|$)", h, re.S)
    ctr = re.search(r"CITATOR INFO\s*:\s*(.*?)(?:ACT\s*:|HEADNOTE\s*:|$)", h, re.S)
    own, by = (cite_keys(cit.group(1)) if cit else set()), set()
    if ctr:
        for yy, form, v, p in re.findall(r"\b[A-Z]{1,3}\s+(\d{4})\s+(SC|SCR|SCC)\s*(?:\((\d+)\))?\s*(\d+)", ctr.group(1)):
            by.add(("AIR", yy, p) if form == "SC" else (form, yy, v, p))
    return own, by


def maxyear(t):
    """Largest plausible year (1600-2029) mentioned in the first 300k characters.

    Args:
        t (str): Text to scan.

    Returns:
        int: Maximum year found or 0.
    """
    return max([int(x) for x in YR.findall(t[:300000])] or [0])


def extra_feats(split, spl, raw0):
    """Forward / reverse citation links and soft year rules.

    ``fwd``: the candidate's CITATOR list names the query (query is later -> candidate is a prior case).
    ``rev``: the query's CITATOR list names the candidate (candidate is LATER -> not a prior case).

    Args:
        split (str): Split name.
        spl (dict): Split container.
        raw0 (Dict[str, List[dict]]): RAW (unstripped) records containing ``split`` and ``split + '_c'``.

    Returns:
        Dict[str, np.ndarray]: ``fwd, rev, soft_year, ydiff`` matrices (n_queries, n_candidates).
    """
    S = spl[split]
    qids, cids = S["qids"], S["cids"]
    nq, nc = len(qids), len(cids)
    HQ = {d["id"]: hdr(d["text"]) for d in raw0[split]}
    HC = {d["id"]: hdr(d["text"]) for d in raw0[split + "_c"]}
    inv_by, inv_own = defaultdict(list), defaultdict(list)
    for j, c in enumerate(cids):
        for k_ in HC[c][1]:
            inv_by[k_].append(j)
        for k_ in HC[c][0]:
            inv_own[k_].append(j)
    fwd = np.zeros((nq, nc), np.float32)
    rev = np.zeros((nq, nc), np.float32)
    for i, q in enumerate(qids):
        for k_ in HQ[q][0]:
            for j in inv_by.get(k_, ()):
                fwd[i, j] = 1
        for k_ in HQ[q][1]:
            for j in inv_own.get(k_, ()):
                rev[i, j] = 1
    lq = np.array([maxyear(S["qtext"][q]) for q in qids])
    lc = np.array([maxyear(S["ctext"][c]) for c in cids])
    both = (lq[:, None] > 0) & (lc[None, :] > 0)
    soft = ((lc[None, :] > lq[:, None] + 3) & both).astype(np.float32)
    ydiff = np.where(both, np.clip((lc[None, :] - lq[:, None]) / 20.0, -2, 2), 0).astype(np.float32)
    return dict(fwd=fwd, rev=rev, soft_year=soft, ydiff=ydiff)


# ---------------- strict year / citation features (train, dev, test) ----------------
def docinfo(split, spl, meta):
    """Per-document citation keys and strict years (body-citation year and header-or-body year).

    Args:
        split (str): Split name.
        spl (dict): Split container.
        meta (dict): Metadata from ``preprocess_splits``.

    Returns:
        Dict[str, dict]: ``{"q": {...}, "c": {...}}`` each with ``keys`` (id -> set), ``yc`` and ``yd`` (int32 arrays).
    """
    S = spl[split]
    out = {}
    for tag, ids, txt, mk in (("q", S["qids"], S["qtext"], meta[split]), ("c", S["cids"], S["ctext"], meta[split + "_c"])):
        ks = {}
        yc = np.zeros(len(ids), np.int32)
        yd = np.zeros(len(ids), np.int32)
        for n, i in enumerate(ids):
            k = cite_keys(txt[i][:400000])
            ks[i] = k
            ys = [int(x[1]) for x in k if x[1].isdigit() and 1850 <= int(x[1]) <= 2025]
            yc[n] = max(ys) if ys else 0
            hy = mk[i]["year"]
            yd[n] = hy if hy else yc[n]
        out[tag] = dict(keys=ks, yc=yc, yd=yd)
    return out


def year_pair(split, spl, meta, DI):
    """Pairwise strict-year violation / gap features and the "candidate body cites the query" flag.

    Args:
        split (str): Split name.
        spl (dict): Split container.
        meta (dict): Metadata from ``preprocess_splits``.
        DI (dict): Output of :func:`docinfo` for this split.

    Returns:
        Dict[str, np.ndarray]: ``yc_viol, yc_gap, yd_viol, yd_known, cand_cites_q`` matrices.
    """
    S = spl[split]
    qi, ci = DI["q"], DI["c"]
    nq, nc = len(S["qids"]), len(S["cids"])
    ycq, ycc, ydq, ydc = qi["yc"][:, None], ci["yc"][None, :], qi["yd"][:, None], ci["yd"][None, :]
    bc = (ycq > 0) & (ycc > 0)
    bd = (ydq > 0) & (ydc > 0)
    f = dict(yc_viol=((ycc > ycq) & bc).astype(np.float32), yc_gap=np.where(bc, np.clip((ycc - ycq) / 20., -2, 2), 0).astype(np.float32),
             yd_viol=((ydc > ydq) & bd).astype(np.float32), yd_known=bd.astype(np.float32))
    inv = defaultdict(list)
    for j, c in enumerate(S["cids"]):
        for k in ci["keys"][c]:
            inv[k].append(j)
    cq = np.zeros((nq, nc), np.float32)      # candidate BODY cites the query's own citation -> candidate is later -> not a prior case
    for i, q in enumerate(S["qids"]):
        for k in meta[split][q]["keys"]:
            for j in inv.get(k, ()):
                cq[i, j] = 1
    f["cand_cites_q"] = cq
    return f
