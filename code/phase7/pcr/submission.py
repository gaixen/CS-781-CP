"""Writing and validating ``prediction.json`` submission archives."""
import glob
import json
import os
import zipfile

import numpy as np


def write_submission(tag, S, spl, cfg):
    """Write the top-K test predictions of a score matrix to ``<submit_dir>/<tag>.zip``.

    Args:
        tag (str): Submission name (used for the folder and zip file names).
        S (np.ndarray): Test score matrix (n_test_queries, n_test_candidates).
        spl (dict): Split container (provides test query / candidate ids).
        cfg (Config): Configuration (``k`` and ``submit_dir``).

    Returns:
        str: Path of the created zip archive (containing ``prediction.json``).
    """
    cids = np.array(spl["test"]["cids"])
    top = np.argsort(-S, 1)[:, :cfg.k]
    pred = {q: [str(c) for c in cids[top[i]]] for i, q in enumerate(spl["test"]["qids"])}
    d = f"{cfg.submit_dir}/{tag}"
    os.makedirs(d, exist_ok=True)
    json.dump(pred, open(f"{d}/prediction.json", "w"))
    zp = f"{cfg.submit_dir}/{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{d}/prediction.json", "prediction.json")
    return zp


def validate_submission(zp, spl, k):
    """Assert that a submission zip is well formed.

    Every test query must have exactly ``k`` unique predictions that all belong to the test candidate pool.

    Args:
        zp (str): Path of the zip archive.
        spl (dict): Split container.
        k (int): Required number of predictions per query.

    Returns:
        None
    """
    with zipfile.ZipFile(zp) as zf:
        pr = json.loads(zf.read("prediction.json"))
    pool = set(spl["test"]["cids"])
    assert set(pr) == set(spl["test"]["qids"]) and all(len(v) == k == len(set(v)) and set(v) <= pool for v in pr.values()), zp
    print("OK", os.path.basename(zp))


def validate_all_submissions(spl, cfg):
    """Validate every zip found in ``cfg.submit_dir``.

    Args:
        spl (dict): Split container.
        cfg (Config): Configuration.

    Returns:
        None
    """
    for zp in sorted(glob.glob(f"{cfg.submit_dir}/*.zip")):
        validate_submission(zp, spl, cfg.k)
