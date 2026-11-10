"""CLI: self-checks, diagnostics, inference-only sweeps and the Codabench analysis.

Examples:
    python scripts/analyze.py --selfchecks
    python scripts/analyze.py --codabench
    python scripts/analyze.py --diagnostics F_focal --sweep F_focal
    python scripts/analyze.py --ensemble-compare F_focal F_lsw
"""
import argparse
import os

import _bootstrap  # noqa: F401


def main():
    """Run the requested analyses.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfchecks", action="store_true"); ap.add_argument("--codabench", action="store_true")
    ap.add_argument("--diagnostics", metavar="RUN"); ap.add_argument("--sweep", metavar="RUN"); ap.add_argument("--ensemble-compare", nargs=2, metavar=("MODERN", "LSW"))
    args = ap.parse_args()
    from lner.config import WORK
    from lner.data.dataset import init_dataset
    P = lambda n: os.path.join(WORK, f"{n}.pt")
    init_dataset()
    if args.codabench:
        from lner.analysis.codabench import codabench_analysis; codabench_analysis()
    so = None
    if args.selfchecks or args.diagnostics:
        from lner.analysis.selfchecks import run_selfchecks; so = run_selfchecks()
    if args.diagnostics:
        from lner import state as S
        from lner.analysis import diagnostics as D
        from lner.training.inference import predict_spans_raw
        from lner.training.trainer import load_trained
        model, cfg = load_trained(P(args.diagnostics)); vp = predict_spans_raw(model, S.VAL_IDS, cfg); vg = {d: S.GOLD[d] for d in S.VAL_IDS}
        D.boundary_deltas(S.TEXTS, vg, vp); D.propagation_diagnostics(vp, vg); D.trim_diagnostics(model, cfg, predict_spans_raw, so); D.caption_diagnostic()
    if args.sweep:
        from lner.analysis.sweeps import merge_ablation, window_sweep
        merge_ablation(P(args.sweep)); window_sweep(P(args.sweep))
    if args.ensemble_compare:
        from lner.ensemble.ensembling import compare_ensembles
        from lner.postprocess.crossfit import load_post
        compare_ensembles(P(args.ensemble_compare[0]), P(args.ensemble_compare[1]), load_post())


if __name__ == "__main__":
    main()
