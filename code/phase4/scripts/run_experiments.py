"""CLI: run one or more registered CJPE experiments.

Examples:
    python scripts/run_experiments.py --names E1_lsw_partial E1a_kd_lsw_student E3_bidir_lorma E4_kd_lsw_student
    python scripts/run_experiments.py --names E3_bidir_lorma --smoke
    python scripts/run_experiments.py --names A_full --override EPOCHS=2 BATCH_SIZE=4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Parse command-line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: ``names`` (list[str]), ``smoke`` (bool), ``override`` (list[str] of KEY=VALUE).
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True, help="experiment names from cjpe.config.EXPERIMENTS")
    ap.add_argument("--smoke", action="store_true", help="smoke test: TRAIN_SUBSET=2000 DEV_SUBSET=500 EPOCHS=1")
    ap.add_argument("--override", nargs="*", default=[], help="config overrides, KEY=VALUE (VALUE parsed as JSON when possible)")
    return ap.parse_args()


def main():
    """Run the requested experiments sequentially.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    from cjpe.context import describe_device
    from cjpe.pipeline import run_experiment
    describe_device()
    ov = dict(TRAIN_SUBSET=2000, DEV_SUBSET=500, EPOCHS=1) if args.smoke else {}
    for kv in args.override:
        k, v = kv.split("=", 1)
        try: ov[k] = json.loads(v)
        except json.JSONDecodeError: ov[k] = v
    for name in args.names:
        run_experiment(name, **ov)


if __name__ == "__main__":
    main()
