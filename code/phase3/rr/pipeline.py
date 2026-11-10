"""Orchestration of the Phase-3 notebook.

Stages (notebook order): ``data`` -> ``a`` -> ``b`` -> ``kd`` -> ``analysis`` -> ``ensemble`` (tune combiner + submissions)
-> ``final`` (retrain members on train+dev) -> ``seed_final`` -> ``recall`` (label-free decoders) -> ``recall_train`` -> ``domain_transfer``.
"""
import os

from rr.config import Settings
from rr.context import Context
from rr.data.loader import inspect_data, load_data
from rr.data.tokenization import setup_tokenizer
from rr.inference.ensemble import analyse_and_ensemble, final_seed_ensemble, retrain_members_final, tune_ensemble
from rr.inference.recall import choose_recall_method, get_dev_probs, inspect_and_refine, rr_audit, submit_recall
from rr.self_checks import run_self_checks
from rr.training.stages import run_domain_transfer, run_recall_training, run_stage_a, run_stage_b, run_stage_kd
from rr.utils import get_device, seed_everything, setup_hf_auth

DEFAULT_STAGES = ["data", "a", "b", "kd", "analysis"]
ALL_STAGES = ["data", "a", "b", "kd", "analysis", "ensemble", "final", "seed_final", "recall", "recall_train", "domain_transfer"]


def run_pipeline(settings, stages=None, final_names=None):
    """Run the requested stages.

    Args:
        settings (Settings): Global settings.
        stages (Optional[List[str]]): Subset of :data:`ALL_STAGES` (default :data:`DEFAULT_STAGES`).
        final_names (Optional[List[str]]): Run names for the ``seed_final`` stage (e.g. ``["C_kd_from_ENS4"]``).

    Returns:
        Context: Context with data, results and stage outputs (``ctx.state``).
    """
    stages = stages or DEFAULT_STAGES
    settings.make_dirs()
    seed_everything(settings.seed)
    ctx = Context(settings=settings, device=get_device())
    ctx.cfg["SEED"] = settings.seed
    if not settings.data_root:
        setup_hf_auth()
    ctx.data = load_data(settings)
    inspect_data(ctx.data)
    setup_tokenizer(ctx)
    if "data" in stages:
        run_self_checks(ctx)
    if "a" in stages:
        run_stage_a(ctx)
    if "b" in stages:
        run_stage_b(ctx)
    if "kd" in stages:
        run_stage_kd(ctx)
    if "analysis" in stages:
        analyse_and_ensemble(ctx)
    if "ensemble" in stages:
        tune_ensemble(ctx)
    if "final" in stages:
        retrain_members_final(ctx)
    if "seed_final" in stages:
        final_seed_ensemble(ctx, final_names or ctx.state.get("best_names") or [])
    if "recall" in stages:
        rr_audit(ctx, get_dev_probs(ctx), ctx.data.dev)
        choose_recall_method(ctx)
        inspect_and_refine(ctx)
        ck = ctx.state.get("finals") or ctx.state.get("best_set") or []
        if ck:
            submit_recall(ctx, ck, "sub_final_recall")
    if "recall_train" in stages:
        run_recall_training(ctx)
    if "domain_transfer" in stages:
        run_domain_transfer(ctx)
    return ctx
