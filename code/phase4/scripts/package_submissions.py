"""CLI: validate and collect 4 submissions (ideally from 4 different Phase-1 models).

Example:
    python scripts/package_submissions.py E1_lsw_partial E1a_kd_lsw_student E3_bidir_lorma E4_kd_lsw_student
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Package the experiments named on the command line into ``<CJPE_WORK>/submissions/``.

    Args:
        None

    Returns:
        None
    """
    from cjpe.eval.submission import package_submissions
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    package_submissions(sys.argv[1:])


if __name__ == "__main__":
    main()
