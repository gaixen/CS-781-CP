"""Unit tests that need only torch / numpy (no checkpoints, no data)."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from rr.data.tokenization import make_windows
from rr.eval.metrics import fast_macro, rr_scores
from rr.inference.recall import bigram_viterbi, prior_match_bias
from rr.self_checks import check_crf


def test_crf_matches_bruteforce():
    check_crf("cpu")


def test_windows():
    assert make_windows(10, 4, 2) == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert make_windows(3, 8, 4) == [(0, 3)]


def test_metrics():
    s = rr_scores(["a", "b"], ["a", "a", "b"], ["a", "b", "b"])
    assert abs(s["acc"] - 2 / 3) < 1e-9
    assert fast_macro(np.array([0, 1, 1]), np.array([0, 1, 1]), 2) == 1.0


def test_lambda_zero_is_argmax():
    P = np.random.RandomState(0).dirichlet(np.ones(4), 6)
    A = np.log(np.full((4, 4), 0.25))
    assert (bigram_viterbi(P, A, np.log(np.full(4, 0.25)), 0.0) == P.argmax(1)).all()


def test_prior_match_shapes():
    P = np.random.RandomState(1).dirichlet(np.ones(3), 50)
    assert prior_match_bias(P, [0.5, 0.3, 0.2]).shape == (3,)
