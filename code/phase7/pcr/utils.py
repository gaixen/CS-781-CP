"""Small shared helpers: seeding, device selection, hashing and score normalisation."""
import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch.

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Pick CUDA when available, otherwise CPU.

    Args:
        None

    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_hf_auth():
    """Log in to HuggingFace using ``HF_TOKEN`` (environment variable or Kaggle secret).

    Args:
        None

    Returns:
        Optional[str]: The token if one was found, else None.
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
        print("HF auth ok")
    return token


def md5_text(text: str) -> str:
    """MD5 hash of a text (used for cache keys and leak checks).

    Args:
        text (str): Input text (None is treated as empty).

    Returns:
        str: Hex digest.
    """
    return hashlib.md5((text or "").encode("utf-8", "ignore")).hexdigest()


def zs(M: np.ndarray) -> np.ndarray:
    """Row-wise z-score of a score matrix.

    Args:
        M (np.ndarray): Score matrix of shape (n_queries, n_candidates).

    Returns:
        np.ndarray: Row-standardised matrix of the same shape.
    """
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)


def gstd(S: np.ndarray) -> np.ndarray:
    """Global standardisation using only finite entries (NaN = not scored).

    Args:
        S (np.ndarray): Score matrix that may contain NaN / inf for unscored pairs.

    Returns:
        np.ndarray: Matrix standardised with the mean / std of its finite entries.
    """
    f = np.isfinite(S)
    mu, sd = S[f].mean(), S[f].std() + 1e-6
    return (S - mu) / sd


def fill_nan(S: np.ndarray) -> np.ndarray:
    """Replace NaN (candidates outside the reranking depth) by ``min - 1``.

    Args:
        S (np.ndarray): Score matrix with NaN for unscored pairs.

    Returns:
        np.ndarray: Matrix where NaN entries are below every scored entry.
    """
    return np.where(np.isnan(S), np.nanmin(S) - 1.0, S)


def percentile_summary(a) -> pd.Series:
    """Summary statistics (min / quartiles / mean / p95 / max) of a 1-D array.

    Args:
        a (array-like): Numeric values.

    Returns:
        pd.Series: Series indexed by statistic name.
    """
    a = np.asarray(a, float)
    return pd.Series({"min": a.min(), "p25": np.percentile(a, 25), "median": np.median(a), "mean": a.mean(),
                      "p75": np.percentile(a, 75), "p95": np.percentile(a, 95), "max": a.max()})
