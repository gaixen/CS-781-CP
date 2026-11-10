"""Light unit tests (CPU, no downloads): config, RoPE/attention mask, streaming helpers."""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.config import build_config  # noqa: E402
from lm.data.streaming import shuffled, source_seed, take  # noqa: E402
from lm.generation.metrics import distinct_n  # noqa: E402
from lm.models.custom_decoder import CustomDecoderConfig, CustomDecoderForCausalLM, GQAFlashAttention  # noqa: E402


def test_build_config_presets():
    """Presets differ per architecture and derived paths exist.

    Args:
        None

    Returns:
        None
    """
    m, l = build_config("modern_decoder"), build_config("lightweight_sliding_window")
    assert m["epochs"] == 2 and set(m["data_sources"]) == {"supreme_court", "high_courts"} and l["epochs"] == 1
    assert l["model_cfg"]["window_size"] == 256 and build_config("seq2seq_t5")["is_seq2seq"]


def test_sliding_window_mask_is_causal_and_banded():
    """Window attention only sees the previous ``window`` tokens (and never the future).

    Args:
        None

    Returns:
        None
    """
    a = GQAFlashAttention(64, 4, 2, window_size=3); mk = a._build_mask(6, torch.device("cpu"), None)[0, 0]
    assert mk[5].tolist() == [False, False, False, True, True, True] and not mk[2, 3]


def test_tiny_decoder_forward_and_loss():
    """A tiny custom decoder runs forward and returns a finite loss.

    Args:
        None

    Returns:
        None
    """
    m = CustomDecoderForCausalLM(CustomDecoderConfig(vocab_size=100, dim=64, n_layer=2, n_head=4, n_kv_head=2, d_ff=128, max_position_embeddings=32, window_size=8))
    ids = torch.randint(0, 100, (2, 16)); out = m(input_ids=ids, labels=ids)
    assert out.logits.shape == (2, 16, 100) and torch.isfinite(out.loss)


def test_streaming_helpers():
    """Deterministic source seeds, bounded shuffle keeps all items, ``take`` and ``distinct_n``.

    Args:
        None

    Returns:
        None
    """
    assert source_seed(42, "acts") == source_seed(42, "acts")
    assert sorted(shuffled(range(50), 8, 1)) == list(range(50)) and list(take(range(10), 3)) == [0, 1, 2]
    assert distinct_n("a b a b", 1) == 0.5
