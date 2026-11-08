#!/usr/bin/env python
"""Command-line entry point for the Phase-5 LSI pipeline.

Example (post-processing only, CPU):
    python scripts/run_pipeline.py --work ./work --prev-dataset /path/containing/lsi_runs --local-data paths.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Config  # noqa: E402
from lsi.pipeline import ALL_STAGES, DEFAULT_STAGES, run_pipeline  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv (Optional[List[str]]): Arguments (default ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="working directory (default /kaggle/working or ./work)")
    ap.add_argument("--smoke", action="store_true", help="tiny subsets + 1 epoch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-data", default=None, help='json file {"train": path, "dev": path, "test": path}')
    ap.add_argument("--prev-dataset", default=None, help="folder that CONTAINS a previous lsi_runs directory")
    ap.add_argument("--ckpt", action="append", default=[], metavar="ARCH=PATH", help="Phase-1 checkpoint (repeatable)")
    ap.add_argument("--train", nargs="*", default=[], help="experiments to (re)train, e.g. E5; default none = post-processing only")
    ap.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, choices=ALL_STAGES)
    ap.add_argument("--kd-seeds", nargs="+", type=int, default=None, help="seeds for the optional seed_ensemble stage")
    ap.add_argument("--no-strict-load", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    """Build the configuration from the arguments and run the pipeline.

    Args:
        argv (Optional[List[str]]): Arguments (default ``sys.argv[1:]``).

    Returns:
        None
    """
    a = parse_args(argv)
    cfg = Config(smoke=a.smoke, seed=a.seed, run_train=a.train, prev_dataset=a.prev_dataset, strict_load=not a.no_strict_load)
    if a.work:
        cfg.work = os.path.abspath(a.work)
    if a.local_data:
        cfg.local_data = json.load(open(a.local_data))
    if a.kd_seeds:
        cfg.kd_seeds = a.kd_seeds
    for item in a.ckpt:
        arch, path = item.split("=", 1)
        cfg.phase1_ckpt[arch] = path
    run_pipeline(cfg, a.stages)


if __name__ == "__main__":
    main()
