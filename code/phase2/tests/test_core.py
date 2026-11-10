"""Light unit tests for pure-python pieces (no GPU, checkpoints or dataset download needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lner import state as S  # noqa: E402
from lner.data.alignment import make_windows, normalize_offsets  # noqa: E402
from lner.data.io import parse_gold, detect_end_inclusive  # noqa: E402
from lner.decoding.spans import viterbi_decode, decode_spans, clean_span  # noqa: E402
from lner.eval.metrics import strict_scores  # noqa: E402
from lner.postprocess.counsel import counsel_side  # noqa: E402
from lner.postprocess.pipeline import postprocess_doc  # noqa: E402


def _init_tags():
    """Initialise a tiny label/tag space in ``state`` for the tests.

    Args:
        None

    Returns:
        None
    """
    S.LABELS = ["APP", "DATE"]; S.TAGS = ["O"] + [f"{p}-{l}" for l in S.LABELS for p in "BI"]
    S.TAG2ID = {t: i for i, t in enumerate(S.TAGS)}; S.N_TYPE = len(S.LABELS) + 1


def test_make_windows_covers_document():
    """Windows cover all tokens and the last one is right-aligned.

    Args:
        None

    Returns:
        None
    """
    w = make_windows(1000, 256, 128)
    assert w[0][0] == 0 and w[-1][1] == 1000 and all(b - a == 256 for a, b in w)
    assert make_windows(100, 256, 128) == [(0, 100)]


def test_normalize_offsets_trims_whitespace():
    """Leading spaces carried by BPE offsets are removed; whitespace-only tokens become empty.

    Args:
        None

    Returns:
        None
    """
    o = normalize_offsets("a  b", [(0, 1), (1, 3), (3, 4)])
    assert o.tolist() == [[0, 1], [3, 3], [3, 4]]


def test_viterbi_forbids_orphan_inside():
    """Viterbi never outputs I-X directly after O.

    Args:
        None

    Returns:
        None
    """
    _init_tags()
    logp = np.log(np.array([[0.9, .02, .02, .03, .03], [0.2, .1, .5, .1, .1]]) + 1e-9)
    path = viterbi_decode(logp)
    assert not (S.TAGS[path[0]] == "O" and S.TAGS[path[1]].startswith("I-"))


def test_decode_and_scores():
    """Greedy BIO decoding gives exact spans and strict scoring matches them.

    Args:
        None

    Returns:
        None
    """
    _init_tags()
    tags = [S.TAG2ID["B-APP"], S.TAG2ID["I-APP"], 0]; offs = np.array([[0, 3], [4, 7], [8, 9]])
    spans = decode_spans(tags, offs); assert spans == [(0, 7, "APP")]
    sc = strict_scores({"d": [(0, 7, "APP")]}, {"d": spans}, S.LABELS)
    assert sc["per_label"]["APP"]["f1"] == 1.0
    assert clean_span("ab. ", 0, 4, ".") == (0, 2)


def test_gold_parsing_and_offset_detection():
    """List-style gold is parsed and the inclusive-end convention detected from entity text.

    Args:
        None

    Returns:
        None
    """
    g = parse_gold({"d": [[0, 2, "APP", "abc"]]}, ["APP"])
    assert g["d"][0][:3] == (0, 2, "APP")
    assert detect_end_inclusive({"d": "abcdef"}, g, verbose=False) is True


def test_counsel_rule_and_empty_post_is_identity():
    """Counsel side from 'for the petitioner'; an empty POST leaves spans unchanged.

    Args:
        None

    Returns:
        None
    """
    t = "Mr. Rao, advocate for the petitioner, argued."
    assert counsel_side(t, 0, 7) == "A.COUNSEL"
    assert postprocess_doc(t, [(0, 7, "A.COUNSEL")], {}) == [(0, 7, "A.COUNSEL")]
