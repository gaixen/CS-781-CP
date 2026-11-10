"""Runtime context: torchvision workaround, device selection, HF authentication."""
import os
import sys


def disable_broken_torchvision():
    """Disable ``torchvision`` if it is ABI-incompatible with ``torch``.

    Kaggle's torchvision can break ``import transformers`` ("operator torchvision::nms does not exist").
    We never use torchvision, so it is removed from ``sys.modules`` before transformers is imported.

    Args:
        None

    Returns:
        None
    """
    try:
        import torch, torchvision  # noqa: F401
    except Exception as e:
        print("torchvision unusable ->", type(e).__name__, "- disabling it (not needed for this task)")
        for m in [m for m in sys.modules if m == "torchvision" or m.startswith("torchvision.")]:
            del sys.modules[m]
        sys.modules["torchvision"] = None


disable_broken_torchvision()

import torch  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hf_token():
    """Fetch the Hugging Face token from the ``HF_TOKEN`` env var, falling back to Kaggle secrets.

    Args:
        None

    Returns:
        str or None: the token, or None if no token could be found.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def setup_hf_auth():
    """Resolve the HF token, export it as ``HF_TOKEN`` and log in to the Hub (needed for the gated dataset).

    Args:
        None

    Returns:
        str or None: the HF token that was used (None if unauthenticated).
    """
    token = get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        from huggingface_hub import login
        login(token=token)
    return token


def describe_device():
    """Print CUDA availability and GPU name.

    Args:
        None

    Returns:
        None
    """
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(cpu)")
