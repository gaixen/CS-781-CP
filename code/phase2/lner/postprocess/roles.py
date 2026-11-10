"""APP / RESP role rules. They only RELABEL spans the model already tagged APP/RESP.

* cue     : 'X (hereinafter the appellant)' / 'X, the respondent'  -> role word right after the party name
* caption : case caption 'X vs. Y' near the top of the document   -> party BEFORE 'vs' = APP, party AFTER 'vs' = RESP
"""
import re

from .counsel import _APP, _RES

_Q = "[\"'\u201c\u201d\u2018\u2019]*"
_CUE1 = re.compile(r"^[\s,]*[\(\[]\s*(?:hereinafter|hereafter)?\s*(?:referred\s+to\s+as|called)?\s*" + _Q + r"\s*(?:the\s+)?(?:learned\s+)?(" + _APP + "|" + _RES + r")\b", re.I)
_CUE2 = re.compile(r"^[\s,]+(?:the\s+)?(" + _APP + "|" + _RES + r")\b", re.I)
_VS = r"(?<![A-Za-z])(?:vs?\.?|versus)(?![A-Za-z])"
_JUNK = r"[\s,\.\-\u2013:\)\]]*"
_ORS = r"(?:(?:and\s+)?(?:ors|anr|others?|another|etc)\.?)?"


def role_from_cue(text, s, e):
    """Role from a cue right after the party name.

    Args:
        text (str): document text.
        s (int): span start.
        e (int): span end (exclusive).

    Returns:
        str or None: ``"APP"`` / ``"RESP"`` or None.
    """
    m = _CUE1.match(text[e:e + 90]) or _CUE2.match(text[e:e + 50])
    if not m: return None
    return "APP" if re.match(_APP, m.group(1), re.I) else "RESP"


def role_from_caption(text, s, e, max_pos=3000):
    """Role from the position relative to 'vs' in the case caption.

    Args:
        text (str): document text.
        s (int): span start.
        e (int): span end (exclusive).
        max_pos (int): only spans within the first ``max_pos`` characters.

    Returns:
        str or None: ``"APP"`` / ``"RESP"`` or None.
    """
    if s > max_pos: return None
    if re.match(_JUNK + _ORS + _JUNK + _VS, text[e:e + 45], re.I): return "APP"
    if re.search(_VS + _JUNK + r"(?:the\s+)?$", text[max(0, s - 30):s], re.I): return "RESP"
    return None


def roles_relabel(text, spans, rules):
    """Relabel APP/RESP spans with the enabled role rules.

    Args:
        text (str): document text.
        spans (list[tuple]): ``(s, e, label)``.
        rules (Iterable[str]): subset of ``{"cue", "caption"}``.

    Returns:
        list[tuple]: relabelled spans.
    """
    out = []
    for s, e, l in spans:
        if l in ("APP", "RESP"):
            c = role_from_cue(text, s, e) if "cue" in rules else None
            if c is None and "caption" in rules: c = role_from_caption(text, s, e)
            l = c or l
        out.append((s, e, l))
    return out
