"""CLI: final fits on ALL labelled docs + submissions.

Examples:
    python scripts/final_fit.py --mode seeds --base F_focal                 # 3 seeds, one architecture -> sub_FINAL_ens3
    python scripts/final_fit.py --mode multi --sources modern=F_focal lsw=F_lsw    # seeds x architectures -> sub_FINAL_multiarch
    python scripts/final_fit.py --mode pseudo --base F_focal                # pseudo-label test docs, retrain -> sub_FINALPL_ens3
"""
import argparse
import os

import _bootstrap  # noqa: F401


def main():
    """Run the requested final-fit pipeline (skips checkpoints that already exist).

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["seeds", "multi", "pseudo"], required=True)
    ap.add_argument("--base", default="F_focal"); ap.add_argument("--sources", nargs="*", default=["modern=F_focal", "lsw=F_lsw"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    from lner.config import WORK
    from lner.data.dataset import init_dataset
    from lner.postprocess.crossfit import load_post
    from lner.submission import predict_test_ensemble, load_test_docs
    from lner.training.trainer import run_final_fit
    P = lambda n: os.path.join(WORK, f"{n}.pt")
    init_dataset(); load_post()
    if args.mode == "seeds":
        finals = []
        for sd in args.seeds:
            if not os.path.exists(P(f"FINAL_s{sd}")): run_final_fit(f"FINAL_s{sd}", P(args.base), SEED=sd)
            finals.append(P(f"FINAL_s{sd}"))
        predict_test_ensemble(finals, "sub_FINAL_ens3")
    elif args.mode == "multi":
        finals, weights = [], []
        for kv in args.sources:
            tag, src = kv.split("=")
            if not os.path.exists(P(src)): print("skip", tag, "(no validated checkpoint)"); continue
            for sd in args.seeds:
                nm = f"FINAL_{tag}_s{sd}"
                if not os.path.exists(P(nm)): run_final_fit(nm, P(src), SEED=sd)
                finals.append(P(nm)); weights.append(1.0 / len(args.seeds))
        predict_test_ensemble(finals, "sub_FINAL_multiarch", weights=weights)
    else:
        from lner.ensemble.pseudo import make_pseudo_docs, run_pseudo_fit
        from lner import state as S
        test_ids = list(load_test_docs()); base = [P(f"FINAL_s{s}") for s in args.seeds]
        pl_t = make_pseudo_docs(base, S.POST, test_ids, tag="PLt"); plf = []
        for sd in args.seeds:
            nm = f"FINALPL_s{sd}"
            if not os.path.exists(P(nm)): run_pseudo_fit(nm, P(args.base), pl_t, SEED=sd)
            plf.append(P(nm))
        predict_test_ensemble(plf, "sub_FINALPL_ens3")


if __name__ == "__main__":
    main()
