"""Submission validation and packaging (``prediction.json`` inside ``submission.zip``)."""
import os
import json
import shutil

from ..config import WORK, OUT_ROOT
from ..data.loader import get_raw_docs
from ..analysis.results import load_results


def validate_submission(name):
    """Check a run's ``prediction.json`` against ``test.json`` (ids, order, int 0/1 predictions).

    Args:
        name (str): experiment name (folder under ``OUT_ROOT``).

    Returns:
        bool: True if valid.

    Raises:
        AssertionError: if ids/order differ or predictions are not int 0/1.
    """
    p = json.load(open(f"{OUT_ROOT}/{name}/prediction.json")); ids = [d["id"] for d in get_raw_docs(verbose=False)["test"]]
    assert [x["id"] for x in p] == ids, "ids/order differ from test.json"
    assert all(isinstance(x["pred"], int) and x["pred"] in (0, 1) for x in p), "pred must be int 0/1"
    assert len(p) == len(ids) == len(set(ids)); return True


def package_submissions(names):
    """Validate and collect the chosen submissions under ``<WORK>/submissions/``.

    Should use 4 DIFFERENT Phase-1 models; a warning is printed otherwise.

    Args:
        names (list[str]): experiment names, in submission order.

    Returns:
        None
    """
    out = f"{WORK}/submissions"; os.makedirs(out, exist_ok=True); df = load_results().set_index("name")
    for i, n in enumerate(names, 1):
        validate_submission(n); dst = f"{out}/submission_{i}_{n}.zip"; shutil.copy(f"{OUT_ROOT}/{n}/submission.zip", dst)
        print(f"{i}. {n:28s} arch={df.loc[n,'arch']:34s} method={df.loc[n,'method']:9s} dev macro-F1={df.loc[n,'dev_f1']:.4f} -> {dst}")
    if len({df.loc[n, 'arch'] for n in names}) < len(names): print("WARNING: spec asks for submissions from DIFFERENT Phase-1 models")
