#!/usr/bin/env python
"""Command-line entry point for the Phase-3 rhetorical-role pipeline.

Example:
    python scripts/run_pipeline.py --work ./work --data-root /path/to/Phase_3_RR \\
        --ckpt modern_decoder=/path/model-final --stages data a b kd analysis
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rr.config import Settings  # noqa: E402
from rr.pipeline import ALL_STAGES, DEFAULT_STAGES, run_pipeline  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv (Optional[List[str]]): Arguments (default ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="working directory for checkpoints / submissions")
    ap.add_argument("--data-root", default=None, help="folder containing CL_*.json, IT_*.json, label_vocab.json (default: download from HuggingFace)")
    ap.add_argument("--ckpt", action="append", default=[], metavar="ARCH=PATH", help="Phase-1 checkpoint (repeatable)")
    ap.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, choices=ALL_STAGES)
    ap.add_argument("--members", nargs="+", default=None, help="ensemble members for the 'ensemble' / 'final' stages")
    ap.add_argument("--final-names", nargs="+", default=None, help="runs retrained on train+dev with several seeds in 'seed_final'")
    ap.add_argument("--window-k", type=int, default=None, help="size of the KD teacher window")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-retrain", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    """Build the settings from the arguments and run the pipeline.

    Args:
        argv (Optional[List[str]]): Arguments (default ``sys.argv[1:]``).

    Returns:
        None
    """
    a = parse_args(argv)
    s = Settings(seed=a.seed, force_retrain=a.force_retrain, data_root=a.data_root)
    if a.work:
        s.work = os.path.abspath(a.work)
    if a.members:
        s.ens_members = a.members
    if a.window_k:
        s.window_k = a.window_k
    for item in a.ckpt:
        arch, path = item.split("=", 1)
        s.phase1_ckpt[arch] = path
    run_pipeline(s, a.stages, a.final_names)


if __name__ == "__main__":
    main()
