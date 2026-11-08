"""Lexical retrieval signals.

* BM25 (default hard-negative miner + fusion component) and TF-IDF baselines (section 4.4),
* TF-IDF variants used to pick the best lexical baseline,
* the lexical suite of "FIX A": trigram / no-stop phrases / headnote / statutes / co-citation,
* candidate-candidate similarity (pseudo-relevance feedback) and hard-negative mining.
"""
import re
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import (CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer)

from pcr.data.preprocessing import cite_keys
from pcr.retrieval.metrics import f1_at_k, recall_at


def _texts(spl, s, max_chars, prep=lambda t: t):
    """Collect (candidate_texts, query_texts) of a split truncated to ``max_chars``.

    Args:
        spl (dict): Split container.
        s (str): Split name.
        max_chars (int): Character limit per document.
        prep (Callable[[str], str]): Optional text transformation applied after truncation.

    Returns:
        Tuple[List[str], List[str]]: ``(candidate_texts, query_texts)`` in pool / query order.
    """
    S = spl[s]
    return [prep(S["ctext"][i][:max_chars]) for i in S["cids"]], [prep(S["qtext"][i][:max_chars]) for i in S["qids"]]


def tfidf_scores(spl, s, cfg):
    """Unigram TF-IDF cosine scores of a split (baseline).

    Args:
        spl (dict): Split container.
        s (str): Split name.
        cfg (Config): Configuration (uses ``max_chars``).

    Returns:
        np.ndarray: Score matrix (n_queries, n_candidates).
    """
    ct, qt = _texts(spl, s, cfg.max_chars)
    v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.5, max_features=250000, dtype=np.float32)
    Cm = v.fit_transform(ct)
    Qm = v.transform(qt)
    out = np.zeros((len(qt), len(ct)), np.float32)
    for i in range(0, len(qt), 256):
        out[i:i + 256] = (Qm[i:i + 256] @ Cm.T).toarray()
    return out


def bm25_scores(spl, s, cfg, k1=1.2, b=0.75):
    """BM25 scores of a split (documents = candidate pool, queries = split queries).

    Args:
        spl (dict): Split container.
        s (str): Split name.
        cfg (Config): Configuration (uses ``max_chars``).
        k1 (float): Term-frequency saturation.
        b (float): Length-normalisation strength.

    Returns:
        np.ndarray: Score matrix (n_queries, n_candidates).
    """
    ct, qt = _texts(spl, s, cfg.max_chars)
    cv = CountVectorizer(stop_words="english", min_df=2, max_df=0.5, max_features=300000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", dtype=np.float32)
    Cc = cv.fit_transform(ct).tocsr()
    Qc = cv.transform(qt).tocsr()
    N = Cc.shape[0]
    df = np.bincount(Cc.indices, minlength=Cc.shape[1])
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.asarray(Cc.sum(1)).ravel()
    rows = np.repeat(np.arange(N), np.diff(Cc.indptr))
    W = Cc.copy()
    W.data = (Cc.data * (k1 + 1) / (Cc.data + k1 * (1 - b + b * dl[rows] / dl.mean()))).astype(np.float32)
    W = (W @ sp.diags(idf)).tocsr()
    Qc.data = (1 + np.log(Qc.data)).astype(np.float32)
    out = np.zeros((len(qt), N), np.float32)
    for i in range(0, len(qt), 256):
        out[i:i + 256] = (Qc[i:i + 256] @ W.T).toarray()
    return out


def compute_lexical_baselines(spl, cfg, verbose=True):
    """Compute TF-IDF and BM25 score matrices for train / dev / test.

    Args:
        spl (dict): Split container.
        cfg (Config): Configuration.
        verbose (bool): Print dev F1 / recall@k of each baseline.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: ``lex[name][split]`` for ``name`` in ``tfidf``, ``bm25``.
    """
    lex = {"tfidf": {s: tfidf_scores(spl, s, cfg) for s in ("train", "dev", "test")},
           "bm25": {s: bm25_scores(spl, s, cfg) for s in ("train", "dev", "test")}}
    if verbose:
        for nm in lex:
            print(f"{nm:6s} dev:", {k: round(v, 4) for k, v in f1_at_k(lex[nm]["dev"], spl, "dev", cfg.k).items()}, "| recall@k", recall_at(lex[nm]["dev"], spl, "dev"))
    return lex


def mine_hard_negatives(spl, lex_train, cfg):
    """Mine TF-IDF / BM25 hard negatives for every train query (ranks ``[hn_lo, hn_hi)`` excluding gold).

    Args:
        spl (dict): Split container.
        lex_train (np.ndarray): Lexical score matrix of the train split (BM25 by default).
        cfg (Config): Configuration (``hn_lo``, ``hn_hi``).

    Returns:
        Tuple[Dict[str, List[str]], List[str]]: ``(hard, tr_items)`` - hard-negative candidate ids per query and the list of
        trainable queries (at least one gold document in the pool).
    """
    cid_tr = np.array(spl["train"]["cids"])
    order = np.argsort(-lex_train, 1)[:, :cfg.hn_hi + 60]
    gold = spl["train"]["gold"]
    hard = {q: [c for c in cid_tr[order[i]][cfg.hn_lo:] if c not in gold[q]][:cfg.hn_hi - cfg.hn_lo] for i, q in enumerate(spl["train"]["qids"])}
    tr_items = [q for q in spl["train"]["qids"] if gold[q]]
    print("trainable queries (>=1 gold in pool):", len(tr_items), "| hard-neg list size:", int(np.mean([len(v) for v in hard.values()])))
    return hard, tr_items


# ---------------- TF-IDF variants (choose the best lexical baseline on dev) ----------------
TFIDF_VARIANTS = {"base": {}, "df0.3": dict(max_df=0.3), "with_q": dict(with_q=True),
                  "bigram": dict(ngram=(1, 2), maxf=600000, min_df=3), "bigram_q": dict(ngram=(1, 2), maxf=600000, min_df=3, with_q=True)}


def tfidf_variant(spl, s, cfg, ngram=(1, 1), min_df=2, max_df=0.5, maxf=250000, with_q=False):
    """TF-IDF scores with configurable n-gram range / document-frequency limits / vocabulary fitting.

    Args:
        spl (dict): Split container.
        s (str): Split name.
        cfg (Config): Configuration (``max_chars``).
        ngram (Tuple[int, int]): N-gram range.
        min_df (int): Minimum document frequency.
        max_df (float): Maximum document-frequency ratio.
        maxf (int): Maximum vocabulary size.
        with_q (bool): Fit the vocabulary / idf on candidates *and* queries.

    Returns:
        np.ndarray: Score matrix (n_queries, n_candidates).
    """
    ct, qt = _texts(spl, s, cfg.max_chars)
    v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=min_df, max_df=max_df, max_features=maxf, ngram_range=ngram, dtype=np.float32)
    v.fit(ct + qt if with_q else ct)
    Cm, Qm = v.transform(ct), v.transform(qt)
    out = np.zeros((len(qt), len(ct)), np.float32)
    for i in range(0, len(qt), 256):
        out[i:i + 256] = (Qm[i:i + 256] @ Cm.T).toarray()
    return out


def select_best_tfidf(spl, lex, cfg):
    """Evaluate every TF-IDF variant on dev and install the best one as ``lex['tfidf']`` (dev + test).

    Args:
        spl (dict): Split container.
        lex (dict): Lexical score store; ``lex['tfidf']['dev'|'test']`` are replaced in place.
        cfg (Config): Configuration.

    Returns:
        str: Name of the selected variant.
    """
    best_nm, best_f, best_S = None, -1, None
    for nm, kw in TFIDF_VARIANTS.items():
        Sd = tfidf_variant(spl, "dev", cfg, **kw)
        m = f1_at_k(Sd, spl, "dev", cfg.k)
        print(f"{nm:10s}", {k: round(v, 4) for k, v in m.items()}, recall_at(Sd, spl, "dev", (7, 50, 100)))
        if m["f1"] > best_f:
            best_nm, best_f, best_S = nm, m["f1"], Sd
    print("best TF-IDF variant:", best_nm, round(best_f, 4))
    lex["tfidf"]["dev"] = best_S
    lex["tfidf"]["test"] = tfidf_variant(spl, "test", cfg, **TFIDF_VARIANTS[best_nm])
    return best_nm


# ---------------- FIX A: lexical suite ----------------
def _clean(M):
    """Drop explicit zeros of a sparse matrix.

    Args:
        M (sp.spmatrix): Sparse matrix.

    Returns:
        sp.csr_matrix: CSR matrix without stored zeros.
    """
    M = M.tocsr()
    M.eliminate_zeros()
    return M


def _sim(Cc, Qc, min_df, max_df, with_q):
    """Cosine TF-IDF similarity between query and candidate count matrices with df filtering.

    Args:
        Cc (sp.spmatrix): Candidate count matrix.
        Qc (sp.spmatrix): Query count matrix.
        min_df (int): Minimum document frequency (over candidates + queries).
        max_df (float): Maximum document-frequency ratio.
        with_q (bool): Fit the idf on candidates and queries (else candidates only).

    Returns:
        np.ndarray: Score matrix (n_queries, n_candidates).
    """
    M = sp.vstack([Cc, Qc]).tocsr()
    N = M.shape[0]
    df = np.bincount(M.indices, minlength=M.shape[1])
    D = sp.diags(((df >= min_df) & (df <= max_df * N)).astype(np.float32))
    Cc, Qc, M = _clean(Cc @ D), _clean(Qc @ D), _clean(M @ D)
    tt = TfidfTransformer(sublinear_tf=True).fit(M if with_q else Cc)
    Cm, Qm = tt.transform(Cc), tt.transform(Qc)
    out = np.zeros((Qm.shape[0], Cm.shape[0]), np.float32)
    for i in range(0, Qm.shape[0], 256):
        out[i:i + 256] = (Qm[i:i + 256] @ Cm.T).toarray()
    return out


def lex_hash(spl, s, cfg, ngram, stop, prep=lambda t: t, with_q=True, min_df=3, max_df=0.5):
    """Hashed n-gram TF-IDF similarity (2**22 buckets).

    Args:
        spl (dict): Split container.
        s (str): Split name.
        cfg (Config): Configuration (``max_chars``).
        ngram (Tuple[int, int]): N-gram range.
        stop (Optional[str]): Stop-word list name (``english``) or None.
        prep (Callable[[str], str]): Text preprocessing applied after truncation.
        with_q (bool): Fit the idf on queries and candidates.
        min_df (int): Minimum document frequency.
        max_df (float): Maximum document-frequency ratio.

    Returns:
        np.ndarray: Score matrix (n_queries, n_candidates).
    """
    hv = HashingVectorizer(n_features=2 ** 22, ngram_range=ngram, stop_words=stop, alternate_sign=False, norm=None, dtype=np.float32)
    ct, qt = _texts(spl, s, cfg.max_chars, prep)
    return _sim(hv.transform(ct), hv.transform(qt), min_df, max_df, with_q)


SEC = re.compile(r"\b(sections?|ss?|articles?|art|rules?|order)\.?\s+(\d+[a-z]?)\b", re.I)


def statute_toks(t, max_chars=120000):
    """Statute-reference tokens (``sec12``, ``art14`` ...) of a text.

    Args:
        t (str): Text to scan.
        max_chars (int): Character limit scanned.

    Returns:
        List[str]: Tokens (``["_none_"]`` when no reference is found).
    """
    return [f"{a.lower()[:3]}{b.lower()}" for a, b in SEC.findall(t[:max_chars])] or ["_none_"]


def cite_toks(t):
    """Citation-key tokens of a text (for the co-citation signal).

    Args:
        t (str): Text to scan.

    Returns:
        List[str]: ``"|"``-joined citation keys (``["_none_"]`` when none).
    """
    return ["|".join(k) for k in cite_keys(t[:400000])] or ["_none_"]


def lex_tok(spl, s, fn, max_df=0.5):
    """TF-IDF similarity over custom token lists (e.g. statute or citation tokens).

    Args:
        spl (dict): Split container.
        s (str): Split name.
        fn (Callable[[str], List[str]]): Function turning a text into a token list.
        max_df (float): Maximum document-frequency ratio.

    Returns:
        np.ndarray: Score matrix (n_queries, n_candidates).
    """
    S = spl[s]
    ct = [fn(S["ctext"][i]) for i in S["cids"]]
    qt = [fn(S["qtext"][i]) for i in S["qids"]]
    v = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, min_df=1, max_df=max_df, dtype=np.float32).fit(ct + qt)
    Cm, Qm = v.transform(ct), v.transform(qt)
    out = np.zeros((len(qt), len(ct)), np.float32)
    for i in range(0, len(qt), 256):
        out[i:i + 256] = (Qm[i:i + 256] @ Cm.T).toarray()
    return out


def lexical_suite_specs(spl, cfg):
    """Return the callables producing each channel of the lexical suite for a split name.

    Args:
        spl (dict): Split container.
        cfg (Config): Configuration.

    Returns:
        Dict[str, Callable[[str], np.ndarray]]: ``channel -> (split -> score matrix)``.
    """
    return {"bi_q_nostop": lambda s: lex_hash(spl, s, cfg, (2, 3), None),
            "tri_q": lambda s: lex_hash(spl, s, cfg, (1, 3), "english"),
            "head_bi": lambda s: lex_hash(spl, s, cfg, (1, 2), "english", prep=lambda t: t[:6000], min_df=2),
            "stat": lambda s: lex_tok(spl, s, lambda t: statute_toks(t, cfg.max_chars)),
            "cocite": lambda s: lex_tok(spl, s, cite_toks, max_df=0.2)}


def build_lexical_suite(spl, cfg, splits=("dev", "test")):
    """Compute the lexical suite (trigram / phrase / headnote / statute / co-citation) for the given splits.

    Args:
        spl (dict): Split container.
        cfg (Config): Configuration.
        splits (Iterable[str]): Splits to compute.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: ``lex2[channel][split]``.
    """
    lex2 = {}
    for nm, fn in lexical_suite_specs(spl, cfg).items():
        t0 = time.time()
        lex2[nm] = {s: fn(s) for s in splits}
        print(f"{nm:12s} dev", {k: round(v, 4) for k, v in f1_at_k(lex2[nm]["dev"], spl, "dev", cfg.k).items()},
              recall_at(lex2[nm]["dev"], spl, "dev", (7, 50, 100)), f"({time.time()-t0:.0f}s)")
    return lex2


def cc_sim(spl, split, cfg):
    """Candidate-candidate trigram TF-IDF similarity (used for pseudo-relevance-feedback features).

    Args:
        spl (dict): Split container.
        split (str): Split name.
        cfg (Config): Configuration.

    Returns:
        np.ndarray: Matrix (n_candidates, n_candidates).
    """
    S = spl[split]
    hv = HashingVectorizer(n_features=2 ** 22, ngram_range=(1, 3), stop_words="english", alternate_sign=False, norm=None, dtype=np.float32)
    Cc = hv.transform([S["ctext"][i][:cfg.max_chars] for i in S["cids"]]).tocsr()
    return _sim(Cc, Cc, 3, 0.5, False)
