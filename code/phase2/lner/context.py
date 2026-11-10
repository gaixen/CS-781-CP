"""Runtime context: seeding, device, Hugging Face authentication."""
import os
import random

import numpy as np
import torch


def seed_everything(s=42):
    """Seed python, numpy and torch (CPU + CUDA).

    Args:
        s (int): seed value.

    Returns:
        None
    """
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_hf_auth():
    """Log in to the Hugging Face Hub using ``HF_TOKEN`` (env var) or a Kaggle secret named ``HF_TOKEN``.

    Args:
        None

    Returns:
        str or None: the token used (None if no token was found; downloads of public files may still work).
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


def describe_device():
    """Print torch version, device and GPU count.

    Args:
        None

    Returns:
        None
    """
    print("torch", torch.__version__, "| device", DEVICE, "| gpus", torch.cuda.device_count())
