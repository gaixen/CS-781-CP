"""Light unit tests for the pure-python parts (no GPU / checkpoints needed)."""
import pytest

pytest.importorskip("transformers")
from pcr.data.preprocessing import cite_keys, normt, strip_header


def test_normt():
    assert normt("A, B!") == "a b"


def test_cite_keys():
    assert ("AIR", "1975", "200") in cite_keys("AIR 1975 SC 200")


def test_strip_header():
    t = "PETITIONER: X Vs. RESPONDENT: Y DATE OF JUDGMENT: 01/01/1970 ACT: body"
    assert strip_header(t).startswith("ACT:")
