"""CLI: cross-fit selection of post-processing components -> ``<LNER_WORK>/POST.pkl``.

Example:
    python scripts/build_postprocessing.py --base F_focal
"""
import argparse
import os

import _bootstrap  # noqa: F401


def main():
    """Train the reverse model if needed, compute out-of-fold predictions and build/save POST.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="F_focal", help="validated fold1-trained run"); ap.add_argument("--margin", type=float, default=0.002)
    args = ap.parse_args()
    from lner.config import WORK
    from lner.data.dataset import init_dataset
    from lner.postprocess.crossfit import out_of_fold_predictions, build_post
    init_dataset(); PA, PB = out_of_fold_predictions(os.path.join(WORK, f"{args.base}.pt")); build_post(PA, PB, args.margin)


if __name__ == "__main__":
    main()
