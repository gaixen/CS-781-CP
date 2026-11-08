#!/usr/bin/env python
"""Command-line entry point for the Phase-7 PCR pipeline.

Example:
    python scripts/run_pipeline.py --work ./work --local-data data_paths.json \\
        --ckpt modern_decoder=/path/to/model-final --stages data experiments
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcr.config import DEFAULT_RUN, Config  # noqa: E402
from pcr.pipeline import ALL_STAGES, run_pipeline  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv (Optional[List[str]]): Argument list (default ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="working directory (default: /kaggle/working or ./work)")
    ap.add_argument("--smoke", action="store_true", help="tiny subsets + 1 epoch to test the pipeline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="re-run experiments even if result.json exists")
    ap.add_argument("--local-data", default=None,
                    help="json file with keys train, train_c, dev, dev_c, test, test_c -> local file paths (default: download from HuggingFace)")
    ap.add_argument("--ckpt", action="append", default=[], metavar="ARCH=PATH",
                    help="Phase-1 checkpoint dir, e.g. modern_decoder=/path/model-final (repeatable)")
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--run", nargs="+", default=DEFAULT_RUN, help="experiment ids to run (E0..E5)")
    ap.add_argument("--no-strict-load", action="store_true", help="do not fail when checkpoint weights are missing")
    return ap.parse_args(argv)


def main(argv=None):
    """Build the configuration from the arguments and run the pipeline.

    Args:
        argv (Optional[List[str]]): Argument list (default ``sys.argv[1:]``).

    Returns:
        None
    """
    a = parse_args(argv)
    cfg = Config(smoke=a.smoke, seed=a.seed, force=a.force, strict_load=not a.no_strict_load)
    if a.work:
        cfg.work = os.path.abspath(a.work)
    if a.local_data:
        cfg.local_data = json.load(open(a.local_data))
    for item in a.ckpt:
        arch, path = item.split("=", 1)
        cfg.phase1_ckpt[arch] = path
    run_pipeline(cfg, a.stages, a.run)


if __name__ == "__main__":
    main()
