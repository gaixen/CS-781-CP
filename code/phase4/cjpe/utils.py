"""Generic helpers: seeding, JSON loading, metrics, threshold calibration, class weights."""
import json
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .context import device


def set_seed(s):
    """Seed python, numpy and torch (CPU + CUDA).

    Args:
        s (int): the seed value.

    Returns:
        None
    """
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_json(p):
    """Load a UTF-8 JSON file.

    Args:
        p (str): path to the JSON file.

    Returns:
        Any: the parsed JSON object.
    """
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metrics(y, p):
    """Macro precision/recall/F1 and per-class F1 for a binary task.

    Args:
        y (array-like): gold labels (0/1).
        p (array-like): predicted labels (0/1).

    Returns:
        dict: keys ``macro_f1, macro_p, macro_r, f1_c0, f1_c1``.
    """
    P, R, F, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    _, _, F_pc, _ = precision_recall_fscore_support(y, p, average=None, labels=[0, 1], zero_division=0)
    return dict(macro_f1=float(F), macro_p=float(P), macro_r=float(R), f1_c0=float(F_pc[0]), f1_c1=float(F_pc[1]))


def best_threshold(y, prob):
    """Pick the decision threshold in [0.20, 0.80] that maximises macro-F1.

    Args:
        y (array-like): gold labels (0/1).
        prob (np.ndarray): predicted probability of class 1.

    Returns:
        tuple[float, float]: (best threshold, macro-F1 at that threshold).
    """
    grid = np.arange(0.20, 0.801, 0.01)
    sc = [f1_score(y, (prob >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(sc))]), float(max(sc))


def class_weights(labels, cfg):
    """Inverse-frequency class weights (normalised to mean 1) for the weighted cross-entropy.

    Args:
        labels (np.ndarray): integer training labels.
        cfg (dict): experiment config (uses ``CLASS_WEIGHTED``).

    Returns:
        torch.Tensor: float32 tensor of shape (2,) on ``device``; all-ones if weighting is disabled or a class is missing.
    """
    n = np.bincount(labels, minlength=2).astype(float)
    if not cfg["CLASS_WEIGHTED"] or (n == 0).any():
        return torch.ones(2, device=device)
    w = torch.tensor(1.0 / n, device=device, dtype=torch.float32)
    return w / w.mean()
