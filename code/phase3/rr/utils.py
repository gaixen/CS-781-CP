"""Shared helpers: seeding, device, HF authentication."""
import os
import random

import numpy as np
import torch


def seed_everything(s: int = 42) -> None:
    """Seed python, numpy and torch (CPU + CUDA).

    Args:
        s (int): Seed.

    Returns:
        None
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device() -> torch.device:
    """CUDA if available else CPU.

    Args:
        None

    Returns:
        torch.device: Selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_hf_auth():
    """Log in to HuggingFace with ``HF_TOKEN`` (env var or Kaggle secret).

    Args:
        None

    Returns:
        Optional[str]: The token if any.
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
    return token
