#!/usr/bin/env python
"""Command-line entry point for the Phase-6 summarisation pipeline.

Example:
    python scripts/run_pipeline.py --work ./work --local-data paths.json --ckpt modern_decoder=/path/model-final --stages data experiments
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summ.config import DEFAULT_RUN, Settings  # noqa: E402
from summ.pipeline import ALL_STAGES, DEFAULT_STAGES, run_pipeline  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv (Optional[List[str]]): Arguments (default ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="working directory")
    ap.add_argument("--smoke", action="store_true", help="tiny subsets + 1 epoch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-data", default=None, help='json file {"train": path, "test": path}')
    ap.add_argument("--ckpt", action="append", default=[], metavar="ARCH=PATH", help="Phase-1 checkpoint (repeatable)")
    ap.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, choices=ALL_STAGES)
    ap.add_argument("--run", nargs="+", default=DEFAULT_RUN, help="experiment ids (E1..E11)")
    ap.add_argument("--ens", nargs="+", default=None, help="ensembles to run (names from summ.config.ENS_RUN)")
    ap.add_argument("--doc-tok", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-strict-load", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    """Build the settings from the arguments and run the pipeline.

    Args:
        argv (Optional[List[str]]): Arguments (default ``sys.argv[1:]``).

    Returns:
        None
    """
    a = parse_args(argv)
    s = Settings(smoke=a.smoke, seed=a.seed, force=a.force, strict_load=not a.no_strict_load)
    if a.work:
        s.work = os.path.abspath(a.work)
    if a.local_data:
        s.local_data = json.load(open(a.local_data))
    if a.ens:
        s.ens_select = a.ens
    if a.doc_tok:
        s.doc_tok = a.doc_tok
    for item in a.ckpt:
        arch, path = item.split("=", 1)
        s.phase1_ckpt[arch] = path
    run_pipeline(s, a.stages, a.run)


if __name__ == "__main__":
    main()
