"""Builders for the four Phase-1 architectures (randomly initialised, no pre-trained weights)."""

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    T5Config,
    T5ForConditionalGeneration,
)

from ..utils import count_params
from .custom_decoder import CustomDecoderConfig, CustomDecoderForCausalLM


def build_gpt2_decoder(tokenizer, cfg):
    """(1) Baseline decoder-only Transformer -- standard HF GPT-2, random init.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer (vocab size, special ids).
        cfg (dict): run config (``model_cfg``, ``model_context_length``).

    Returns:
        GPT2LMHeadModel: the model.
    """
    m = cfg["model_cfg"]
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=cfg["model_context_length"],
            n_ctx=cfg["model_context_length"],
            n_embd=m["n_embd"],
            n_layer=m["n_layer"],
            n_head=m["n_head"],
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )


def _custom(tokenizer, cfg, window_size):
    """Shared constructor of the two custom decoders.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer.
        cfg (dict): run config.
        window_size (int or None): attention window.

    Returns:
        CustomDecoderForCausalLM: the model.
    """
    m = cfg["model_cfg"]
    return CustomDecoderForCausalLM(
        CustomDecoderConfig(
            vocab_size=len(tokenizer),
            dim=m["dim"],
            n_layer=m["n_layer"],
            n_head=m["n_head"],
            n_kv_head=m["n_kv_head"],
            d_ff=m["d_ff"],
            dropout=m["dropout"],
            max_position_embeddings=cfg["model_context_length"],
            window_size=window_size,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )


def build_modern_decoder(tokenizer, cfg):
    """(2) From-scratch decoder-only: RMSNorm + RoPE + GQA(SDPA/flash) + SwiGLU, full causal attention.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer.
        cfg (dict): run config.

    Returns:
        CustomDecoderForCausalLM: the model.
    """
    return _custom(tokenizer, cfg, None)


def build_lightweight_sliding_window(tokenizer, cfg):
    """(4) Same decoder with local/sliding-window causal attention (cheaper; legal text has locally clustered dependencies).

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer.
        cfg (dict): run config (``model_cfg["window_size"]``).

    Returns:
        CustomDecoderForCausalLM: the model.
    """
    return _custom(tokenizer, cfg, cfg["model_cfg"]["window_size"])


def build_seq2seq_t5(tokenizer, cfg):
    """(3) Encoder-decoder Transformer (T5 config), random init, framed as prompt -> continuation.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer.
        cfg (dict): run config.

    Returns:
        T5ForConditionalGeneration: the model.
    """
    m = cfg["model_cfg"]
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=len(tokenizer),
            d_model=m["d_model"],
            d_kv=m["d_kv"],
            d_ff=m["d_ff"],
            num_layers=m["num_layers"],
            num_decoder_layers=m["num_decoder_layers"],
            num_heads=m["num_heads"],
            dropout_rate=m["dropout_rate"],
            feed_forward_proj="gated-gelu",
            decoder_start_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )


ARCHITECTURE_BUILDERS = {
    "gpt2_decoder": build_gpt2_decoder,
    "modern_decoder": build_modern_decoder,
    "seq2seq_t5": build_seq2seq_t5,
    "lightweight_sliding_window": build_lightweight_sliding_window,
}
RELOAD_MODEL_CLASS = {
    "gpt2_decoder": GPT2LMHeadModel,
    "modern_decoder": CustomDecoderForCausalLM,
    "lightweight_sliding_window": CustomDecoderForCausalLM,
    "seq2seq_t5": T5ForConditionalGeneration,
}


def build_model(arch, tokenizer, cfg):
    """Build an architecture and check the 50M-300M parameter budget.

    Args:
        arch (str): architecture key.
        tokenizer (PreTrainedTokenizer): tokenizer.
        cfg (dict): run config.

    Returns:
        nn.Module: the model.

    Raises:
        AssertionError: if the parameter count is outside 50M-300M.
    """
    model = ARCHITECTURE_BUILDERS[arch](tokenizer, cfg)
    n = count_params(model)
    print(f"Architecture: {arch} | total parameters: {n:,} ({n / 1e6:.1f}M)")
    assert 50_000_000 <= n <= 300_000_000, (
        f"Parameter count {n:,} outside 50M-300M -- adjust model_cfg for '{arch}'"
    )
    return model


def load_trained(arch, model_dir, device):
    """Reload a saved model + tokenizer (re-tying ``lm_head`` to ``tok_emb`` for the custom decoders).

    Args:
        arch (str): architecture key.
        model_dir (str): directory written by ``save_pretrained``.
        device (torch.device): target device.

    Returns:
        tuple: ``(model, tokenizer)``; tokenizer padding set for generation.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = RELOAD_MODEL_CLASS[arch].from_pretrained(model_dir)
    if hasattr(model, "tok_emb") and hasattr(model, "lm_head"):
        model.lm_head.weight = model.tok_emb.weight  # old checkpoints may not re-tie
    return model.to(device).eval(), tok
