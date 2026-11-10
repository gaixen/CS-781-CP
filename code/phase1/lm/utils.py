"""Small helpers: seeding, Hugging Face login, parameter counting, manual checkpoints, tokenizer."""

import os
import random

import numpy as np
import torch
import torch.nn as nn

TOKENIZER_NAME = "/kaggle/input/sohamjee/legal_bpe/hf"


def set_seed(seed):
    """Seed python, numpy and torch.

    Args:
        seed (int): seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hf_login():
    """Log in to the Hugging Face Hub using ``HF_TOKEN`` (env var) or a Kaggle secret named ``HF_TOKEN``.

    Args:
        None

    Returns:
        str or None: the token used.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = token
        from huggingface_hub import login

        login(token=token)
        print(".......... Hugging Face login successful ..........")
    return token


def count_params(model):
    """Number of trainable parameters.

    Args:
        model (nn.Module): any module.

    Returns:
        int: trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def precision_flags():
    """bf16 on A100/L4/H100, otherwise fp16 on any other CUDA GPU.

    Args:
        None

    Returns:
        tuple[bool, bool]: ``(use_bf16, use_fp16)`` for ``TrainingArguments``.
    """
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    bf16 = any(k in name for k in ("A100", "L4", "H100"))
    return bf16, torch.cuda.is_available() and not bf16


def get_device():
    """CUDA if available, else CPU.

    Args:
        None

    Returns:
        torch.device: the device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer():
    """
    BPE keeps the tokenizer a controlled variable across architectures.

    Args:
        None

    Returns:
        PreTrainedTokenizer: the tokenizer.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.model_max_length = int(1e12)
    print("Vocabulary size:", len(tok))
    return tok


def save_checkpoint(ckpt_dir, tag, model, optimizer=None, epoch=None, history=None):
    """Save a manual ``.pt`` checkpoint.

    Args:
        ckpt_dir (str): directory.
        tag (str): file name stem.
        model (nn.Module): model.
        optimizer (torch.optim.Optimizer or None): optimizer to save too.
        epoch (int or None): epoch.
        history (dict or None): training history.

    Returns:
        str: written path.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {"model": model.state_dict(), "epoch": epoch, "history": history}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    path = os.path.join(ckpt_dir, f"{tag}.pt")
    torch.save(state, path)
    print(f".......... Model checkpoint saved at {path} ..........")
    return path


def load_checkpoint(ckpt_dir, tag, model, optimizer=None, map_location=None):
    """Load a manual ``.pt`` checkpoint in place.

    Args:
        ckpt_dir (str): directory.
        tag (str): file name stem.
        model (nn.Module): model to load into.
        optimizer (torch.optim.Optimizer or None): optimizer to restore.
        map_location (str or torch.device or None): map location.

    Returns:
        tuple: ``(epoch, history)`` or ``(None, None)`` if the checkpoint does not exist.
    """
    path = os.path.join(ckpt_dir, f"{tag}.pt")
    if not os.path.exists(path):
        return None, None
    state = torch.load(path, map_location=map_location or get_device())
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    print(f".......... Model checkpoint loaded from {path} ..........")
    return state.get("epoch"), state.get("history")
