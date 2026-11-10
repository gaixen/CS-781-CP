"""CLI: print the results table, size/cost table and (optionally) error analysis.

Examples:
    python scripts/show_results.py
    python scripts/show_results.py --prefix A_hier
    python scripts/show_results.py --errors E3_bidir_lorma E4_kd_lsw_student
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Print result tables for finished experiments.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="", help="only experiments starting with this prefix")
    ap.add_argument("--errors", nargs="*", default=[], help="experiments to run FP/FN error analysis on")
    args = ap.parse_args()
    from cjpe.analysis.results import show_results, error_analysis, sizes_table
    print(show_results(args.prefix).to_string())
    for n in args.errors:
        error_analysis(n)
    print(sizes_table().to_string())


if __name__ == "__main__":
    main()
