"""Unit tests for the pure-python parts (no checkpoints / data)."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from summ.eval.metrics import ngrams, postprocess
from summ.models.summ_models import summ_mask


def test_postprocess_drops_repeated_sentences():
    assert postprocess("A b c. A b c. D e f.") == "A b c. D e f."


def test_ngrams():
    assert ngrams([1, 2, 3], 2) == {(1, 2), (2, 3)}


def test_summ_mask_summary_sees_all_doc_keys():
    import torch
    keypad = torch.ones(1, 6, dtype=torch.long)
    m = summ_mask(4, 2, window=2, bidir=False, keypad=keypad, dev=torch.device("cpu"))
    assert m[0, 0, 4, :4].all() and not m[0, 0, 4, 5]        # summary row 0 sees every document key, not the future summary token
