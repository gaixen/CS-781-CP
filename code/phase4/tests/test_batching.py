"""Light unit tests for pure-python utilities (no GPU, checkpoints or dataset download needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cjpe.data.batching import make_chunks, select_ids, _pad  # noqa: E402
from cjpe.utils import metrics, best_threshold  # noqa: E402


def test_make_chunks_counts_from_end():
    """Chunk 0 must be the LAST chunk and chunks must overlap by ``stride``.

    Args:
        None

    Returns:
        None
    """
    tail = np.arange(1000)
    ch = make_chunks(tail, 512, 64, 4)
    assert ch[0][-1] == 999 and len(ch[0]) == 512
    assert len(ch) == 4
    assert list(ch[1][-64:]) == list(ch[0][:64])


def test_select_ids_modes():
    """``last_n`` takes the tail, ``first_n`` the head, ``head_tail`` mixes both.

    Args:
        None

    Returns:
        None
    """
    ex = dict(head=np.arange(100), tail=np.arange(1000, 1100), whole=False)
    cfg = dict(MAX_TOKENS=10, HEAD_TOKENS=4)
    assert list(select_ids(ex, {**cfg, "REPRESENTATION": "last_n"})) == list(range(1090, 1100))
    assert list(select_ids(ex, {**cfg, "REPRESENTATION": "first_n"})) == list(range(10))
    assert len(select_ids(ex, {**cfg, "REPRESENTATION": "head_tail"})) == 10


def test_pad_and_metrics():
    """Padding yields a right-padded boolean mask; metrics return the expected keys.

    Args:
        None

    Returns:
        None
    """
    ids, mask = _pad([np.array([1, 2, 3]), np.array([4])], 0)
    assert ids.shape == (2, 3) and mask.sum().item() == 4
    m = metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    assert set(m) == {"macro_f1", "macro_p", "macro_r", "f1_c0", "f1_c1"}
    thr, f = best_threshold(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]))
    assert 0.2 <= thr <= 0.8 and f == 1.0
