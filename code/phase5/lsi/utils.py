import glob, hashlib, os, random, shutil, torch
import numpy as np


def set_seed(s: int) -> None:
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


def md5_text(t: str) -> str:
    """MD5 hex digest of a text.

    Args:
        t (str): Input text.

    Returns:
        str: Hex digest.
    """
    return hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()


def setup_hf_auth():
    """Log in to HuggingFace with ``HF_TOKEN`` (env var or Kaggle secret).

    Args:
        None

    Returns:
        Optional[str]: The token, if any.
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


def restore_runs(cfg, prev=None, need_best=(), copy_tok_cache=False) -> None:
    """Copy a previous run folder (``lsi_runs``) into the working directory.

    Always copies the small files (``*_logits.npy``, ``result.json``, an interrupted run's ``ckpt.pt``) and
    ``best.pt`` only for the experiments in ``need_best``. Existing files of equal size are never overwritten.

    Args:
        cfg (Config): Configuration (``out_dir``, ``cache_dir``).
        prev (Optional[str]): Folder that CONTAINS ``lsi_runs`` (None = auto-find under ``/kaggle/input``).
        need_best (Iterable[str]): Experiment ids whose ``best.pt`` must be copied.
        copy_tok_cache (bool): Also copy tokenisation caches.

    Returns:
        None
    """
    roots = (
        [prev]
        if prev
        else sorted(
            {
                os.path.dirname(p)
                for p in glob.glob("/kaggle/input/**/lsi_runs", recursive=True)
            }
        )
    )
    if not roots:
        print(
            "no lsi_runs folder found under /kaggle/input -> attach the previous output as an input (or set prev_dataset)"
        )
        return
    for root in roots:
        for d in sorted(glob.glob(f"{root}/lsi_runs/*")):
            if not os.path.isdir(d):
                continue
            name = os.path.basename(d)
            eid = name.split("_")[0]
            dst = f"{cfg.out_dir}/{name}"
            os.makedirs(dst, exist_ok=True)
            got = []
            for f in os.listdir(d):
                if f == "best.pt" and eid not in need_best:
                    continue
                s, t = f"{d}/{f}", f"{dst}/{f}"
                if os.path.isfile(s) and not (
                    os.path.exists(t) and os.path.getsize(t) == os.path.getsize(s)
                ):
                    shutil.copy2(s, t)
                    got.append(f)
            print(
                f"  {name}: copied {got if got else 'nothing new'} | now has {sorted(os.listdir(dst))}"
            )
        if copy_tok_cache and os.path.isdir(f"{root}/lsi_cache"):
            for f in glob.glob(f"{root}/lsi_cache/tok_*.pkl"):
                t = f"{cfg.cache_dir}/{os.path.basename(f)}"
                if not os.path.exists(t):
                    shutil.copy2(f, t)
            print("  tokenisation cache restored")
