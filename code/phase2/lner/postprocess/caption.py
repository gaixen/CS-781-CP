"""Document-level cue mapping from the case caption ('X ... Appellant  Versus  Y ... Respondent').

* parse_caption     : party names before 'vs/versus' = APP, after = RESP (role words, numbering, '...' leaders, 'through ...' stripped)
* caption_apply     : 'add' = add caption parties the model missed (+ fix APP/RESP side); 'prop' = also tag their exact body mentions
* counsel_by_party  : 'Mr. X, counsel for <party name>' -> counsel side = side of that party
"""
import re

import numpy as np

_CAP_APP = r"appellants?|petitioners?|plaintiffs?|applicants?|complainants?|revisionists?|claimants?"
_CAP_RES = r"respondents?|defendants?|opposite\s+part(?:y|ies)|non-applicants?|caveators?"
_CAP_ROLE = _CAP_APP + "|" + _CAP_RES
_CAP_VS_LINE = re.compile(r"(?im)^[ \t]*(?:versus|vs\.?|v\.?|v/s)[ \t]*$")
_CAP_VS_INL = re.compile(r"(?<![A-Za-z])(?:versus|vs\.?|v/s)(?![A-Za-z])", re.I)
_CAP_STOP = re.compile(r"(?i)\b(?:in the|court of|high court|supreme court|tribunal|jurisdiction|writ petition|civil appeal|criminal appeal|special leave|reportable|"
                       r"judg?ment|digitally|signature|date of|dated)\b|\d{2,}\s*/\s*\d{2,4}|\bno\.?\s*\d")
_CAP_TAIL = re.compile(r"(?i)\s*(?:[\.\u2026\-\u2013\u2014_,:;\(\[]\s*)*(?:" + _CAP_ROLE + r")\b.*$")
_CAP_THRU = re.compile(r"(?i)\s+(?:through|thr\.?|represented\s+by|rep\.?\s+by)\b.*$")
_CAP_NUM = re.compile(r"^\s*(?:\(?\d{1,2}\s*[\.\)]|\([ivxIVX]+\)|[A-Za-z]\))\s*")


def _party_lines(text, a, b):
    """Cleaned party-name candidates found on the lines of ``text[a:b]``.

    Args:
        text (str): document text.
        a (int): region start.
        b (int): region end.

    Returns:
        list[tuple[int, int, bool]]: ``(start, end, has_role_word)``.
    """
    out, pos = [], a
    for line in text[a:b].splitlines(True):
        raw = line.rstrip("\r\n"); ls = pos; pos += len(line)
        if not raw.strip() or _CAP_STOP.search(raw): continue
        i0 = len(raw) - len(raw.lstrip()); m = _CAP_NUM.match(raw[i0:])
        if m: i0 += m.end()
        body = raw[i0:]; tail = _CAP_TAIL.search(body); has_role = bool(tail)
        if tail: body = body[:tail.start()]
        thru = _CAP_THRU.search(body)
        if thru: body = body[:thru.start()]
        body = body.rstrip(" \t.,;:-\u2013\u2014_\u2026"); lead = len(body) - len(body.lstrip(" \t.,;:-\u2013\u2014_\u2026")); body = body.strip(" \t.,;:-\u2013\u2014_\u2026")
        if 3 <= len(body) <= 100 and sum(ch.isalpha() for ch in body) >= 2: out.append((ls + i0 + lead, ls + i0 + lead + len(body), has_role))
    return out


def parse_caption(text, head=4500, left_win=800, right_win=1000):
    """Parse the case caption.

    Args:
        text (str): document text.
        head (int): search the 'vs' marker in the first ``head`` characters.
        left_win (int): characters before 'vs' to scan for appellant lines.
        right_win (int): characters after 'vs' to scan for respondent lines.

    Returns:
        dict or None: ``{"app": [(s, e)], "resp": [(s, e)], "end": offset}`` (absolute, end-exclusive) or None.
    """
    m = _CAP_VS_LINE.search(text, 0, head) or _CAP_VS_INL.search(text, 0, head)
    if not m: return None
    rb_end = min(len(text), m.end() + right_win)
    stop = re.search(r"(?i)\n\s*(?:j\s*u\s*d\s*g\s*m\s*e\s*n\s*t|order|coram|before|present|reserved|pronounced|date of)\b|\n\s*\n\s*\n", text[m.end():rb_end])
    if stop: rb_end = m.end() + stop.start()
    left = _party_lines(text, max(0, m.start() - left_win), m.start()); right = _party_lines(text, m.end(), rb_end)
    def pick(c, last):
        r = [x for x in c if x[2]]
        return [(s, e) for s, e, _ in (r if r else (c[-2:] if last else c[:2]))]
    return dict(app=pick(left, True), resp=pick(right, False), end=rb_end)


def caption_apply(text, spans, mode, P=None):
    """Add / fix APP and RESP spans from the parsed caption.

    Args:
        text (str): document text.
        spans (list[tuple]): ``(s, e, label)``.
        mode (str): ``"add"`` or ``"prop"`` (also tag exact body mentions).
        P (dict or None): precomputed ``parse_caption`` result.

    Returns:
        list[tuple]: updated spans.
    """
    P = P or parse_caption(text)
    if not P: return spans
    out = list(spans); occ = np.zeros(len(text) + 1, bool)
    for s, e, l in out: occ[s:e] = True
    cap = [(s, e, "APP") for s, e in P["app"]] + [(s, e, "RESP") for s, e in P["resp"]]
    for s, e, l in cap:
        over = [i for i, (a, b, x) in enumerate(out) if a < e and s < b]
        if not over: out.append((s, e, l)); occ[s:e] = True
        else:
            for i in over:
                a, b, x = out[i]
                if x in ("APP", "RESP") and x != l: out[i] = (a, b, l)
    if mode == "prop":
        for s, e, l in cap:
            name = text[s:e]
            if len(name) < 6: continue
            for mm in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text[P["end"]:], re.I):      # captions are often upper-case, bodies title-case
                a, b = P["end"] + mm.start(), P["end"] + mm.end()
                if not occ[a:b].any(): out.append((a, b, l)); occ[a:b] = True
    return out


_FOR_PARTY = re.compile(r"(?:for|on\s+behalf\s+of|representing)\s+(?:the\s+)?(?:learned\s+)?", re.I)


def _name_keys(text, ab):
    """Lower-case lookup keys (full name and first two words) for caption parties.

    Args:
        text (str): document text.
        ab (list[tuple[int, int]]): party spans.

    Returns:
        list[str]: keys.
    """
    keys = []
    for s, e in ab:
        n = text[s:e].lower(); keys.append(n)
        w = n.split()
        if len(w) >= 2 and len(" ".join(w[:2])) >= 8: keys.append(" ".join(w[:2]))
    return keys


def counsel_by_party(text, s, e, P):
    """'Mr. X, learned counsel for Ram Kumar' -> side of the caption party named right after 'for' (earliest wins; ambiguous -> None).

    Args:
        text (str): document text.
        s (int): counsel span start.
        e (int): counsel span end (exclusive).
        P (dict): parsed caption.

    Returns:
        str or None: ``"A.COUNSEL"`` / ``"R.COUNSEL"`` or None.
    """
    m = _FOR_PARTY.search(text[e:e + 110])
    if not m: return None
    rest = text[e + m.end():e + m.end() + 70].lower()
    first = lambda keys: min([rest.find(k) for k in keys if k in rest], default=None)
    pa, pr = first(_name_keys(text, P["app"])), first(_name_keys(text, P["resp"]))
    if pa is not None and (pr is None or pa + 25 <= pr): return "A.COUNSEL"
    if pr is not None and (pa is None or pr + 25 <= pa): return "R.COUNSEL"
    return None


def counsel_party_relabel(text, spans, P):
    """Relabel counsel spans by the party they represent.

    Args:
        text (str): document text.
        spans (list[tuple]): ``(s, e, label)``.
        P (dict): parsed caption.

    Returns:
        list[tuple]: relabelled spans.
    """
    out = []
    for s, e, l in spans:
        if l in ("A.COUNSEL", "R.COUNSEL"): l = counsel_by_party(text, s, e, P) or l
        out.append((s, e, l))
    return out
