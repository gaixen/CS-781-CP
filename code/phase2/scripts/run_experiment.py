"""CLI: run registered experiments, the training-time-enforcement set, or distillation.

Examples:
    python scripts/run_experiment.py --names F_focal
    python scripts/run_experiment.py --names A_lsw_lora --override EPOCHS=3 MAX_LEN=384 OVERLAP=192
    python scripts/run_experiment.py --second-family            # F_lsw from F_focal's config
    python scripts/run_experiment.py --te                       # T1/T2/T3 + B0/B1
    python scripts/run_experiment.py --kd                       # E4 distillation student
    python scripts/run_experiment.py --pruned-kd                # E4b / E4c compression
"""
import argparse
import json

import _bootstrap  # noqa: F401


def parse_overrides(items):
    """Parse ``KEY=VALUE`` strings (VALUE parsed as JSON when possible).

    Args:
        items (list[str]): override strings.

    Returns:
        dict: parsed overrides.
    """
    ov = {}
    for kv in items:
        k, v = kv.split("=", 1)
        try: ov[k] = json.loads(v)
        except json.JSONDecodeError: ov[k] = v
    return ov


def main():
    """Parse arguments, initialise the dataset and run the requested experiments.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="*", default=[], help="registered experiment names (lner.config.EXPERIMENTS)")
    ap.add_argument("--override", nargs="*", default=[], help="KEY=VALUE config overrides")
    ap.add_argument("--second-family", action="store_true"); ap.add_argument("--te", action="store_true")
    ap.add_argument("--kd", action="store_true"); ap.add_argument("--pruned-kd", action="store_true")
    args = ap.parse_args()
    from lner.context import describe_device
    from lner.data.dataset import init_dataset
    from lner import experiments as X
    from lner.training.trainer import show_results
    describe_device(); init_dataset(); ov = parse_overrides(args.override)
    for n in args.names: X.run_named(n, **ov)
    if args.second_family: X.run_second_family()
    if args.te: X.run_training_time_enforcement()
    if args.kd: X.run_distillation()
    if args.pruned_kd: X.run_pruned_kd()
    show_results()


if __name__ == "__main__":
    main()
