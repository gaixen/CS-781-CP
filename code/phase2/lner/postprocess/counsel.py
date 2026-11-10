"""Counsel side (A.COUNSEL / R.COUNSEL) from 'for the <role>' cues, plus per-document name voting."""
import re
from collections import Counter, defaultdict

_APP = r"appellants?|petitioners?|plaintiffs?|applicants?|complainants?|revisionists?|claimants?"
_RES = r"respondents?|defendants?|opposite\s+part(?:y|ies)|non-applicants?|caveators?"
FOR_CUE = re.compile(r"(?:for|on\s+behalf\s+of|representing)\s+(?:the\s+)?(?:learned\s+)?(" + _APP + r"|" + _RES + r")", re.I)
_is_app = re.compile(_APP, re.I)
_HDR = re.compile(r"\s*[:\-\u2013]")


def _side(m):
    """Counsel label from a ``FOR_CUE`` match.

    Args:
        m (re.Match): match of ``FOR_CUE``.

    Returns:
        str: ``"A.COUNSEL"`` or ``"R.COUNSEL"``.
    """
    return "A.COUNSEL" if _is_app.match(m.group(1)) else "R.COUNSEL"


def counsel_side(text, s, e):
    """A.COUNSEL / R.COUNSEL from the sentence pattern around a counsel name, else None.

    (1) name directly after the cue ('counsel for the petitioner Mr. A'); (2) cue right after the name
    ('Mr. A, Advocate for the respondent'); (3) list header before the name ('Counsel for the Appellant: Mr. A').
    Cues followed by ':' belong to the names AFTER them.

    Args:
        text (str): document text.
        s (int): span start.
        e (int): span end (exclusive).

    Returns:
        str or None: counsel label if a cue was found.
    """
    left = text[max(0, s - 200):s]; ms = list(FOR_CUE.finditer(left))
    if ms and re.fullmatch(r"[\s,:\-\u2013]*", left[ms[-1].end():]): return _side(ms[-1])   # (1)
    right = text[e:e + 60]
    for m in FOR_CUE.finditer(right):                                                     # (2)
        if _HDR.match(right[m.end():m.end() + 4]): break
        return _side(m)
    hdr = [m for m in ms if _HDR.match(left[m.end():m.end() + 4])]                        # (3)
    return _side(hdr[-1]) if hdr else None


def counsel_relabel(text, spans, vote=True):
    """Relabel counsel spans by the cue rule; optionally propagate the side by exact-string voting inside the document.

    Args:
        text (str): document text.
        spans (list[tuple]): ``(s, e, label)``.
        vote (bool): spans without a cue take the majority side of the same string elsewhere in the document.

    Returns:
        list[tuple]: relabelled spans.
    """
    side, fired = [], defaultdict(Counter)
    for s, e, l in spans:
        c = counsel_side(text, s, e) if l in ("A.COUNSEL", "R.COUNSEL") else None
        side.append(c)
        if c: fired[text[s:e]][c] += 1
    out = []
    for (s, e, l), c in zip(spans, side):
        if l in ("A.COUNSEL", "R.COUNSEL") and not c and vote and text[s:e] in fired: c = fired[text[s:e]].most_common(1)[0][0]
        out.append((s, e, c or l))
    return out
