"""Unit tests of the pure-numpy metric / threshold code."""
import numpy as np

from lsi.eval.metrics import decide, macro_f1, sigmoid, tune_global


def test_macro_f1_perfect():
    Y = np.array([[1, 0], [0, 1]], np.float32)
    assert macro_f1(Y.astype(bool), Y) == 1.0


def test_decide_top1_fills_empty_rows():
    P = np.array([[0.1, 0.2]])
    assert decide(P, 0.5, top1=True).sum() == 1


def test_tune_global_range():
    rng = np.random.RandomState(0)
    Y = (rng.rand(50, 5) > 0.7).astype(np.float32)
    thr, f = tune_global(sigmoid(rng.randn(50, 5) + 2 * Y), Y)
    assert 0.05 <= thr <= 0.95 and 0 <= f <= 1
