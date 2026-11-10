"""CLI: create a verified ``prediction.json`` + ``submission.zip`` from one checkpoint (or an ensemble).

Examples:
    python scripts/predict_test.py --ckpts E4_kd_student --name sub_E4_kd_student_3
    python scripts/predict_test.py --ckpts FINAL_s1 FINAL_s2 FINAL_s3 --name sub_FINAL_ens3 --no-post
"""
import argparse
import os

import _bootstrap  # noqa: F401


def main():
    """Predict the test set and write the submission under ``<LNER_WORK>/<name>/``.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True, help="run names (checkpoints in LNER_WORK) or .pt paths")
    ap.add_argument("--name", required=True); ap.add_argument("--no-post", action="store_true", help="submit raw model output (POST = {})")
    ap.add_argument("--include-text", action="store_true")
    args = ap.parse_args()
    from lner import state as S
    from lner.config import WORK
    from lner.data.dataset import init_dataset
    from lner.postprocess.crossfit import load_post
    from lner.submission import predict_test, predict_test_ensemble
    from lner.training.trainer import load_trained
    paths = [c if c.endswith(".pt") else os.path.join(WORK, f"{c}.pt") for c in args.ckpts]
    init_dataset(run_oracle=False); load_post()
    if args.no_post: S.POST = {}
    if len(paths) == 1:
        model, cfg = load_trained(paths[0]); predict_test(model, cfg, args.name, args.include_text)
    else:
        predict_test_ensemble(paths, args.name, include_text=args.include_text)
    print("upload this:", os.path.join(WORK, args.name, "submission.zip"))


if __name__ == "__main__":
    main()
