"""Orchestration in notebook order.

Stages: ``data`` (load, audit, tokenise, split) -> ``experiments`` -> ``ensembles`` -> ``analysis`` (comparison + long-document / length sweeps)
-> ``final`` (validation) -> ``distill`` (optional student distillation + seed ensembles).
"""
import os

from summ.analysis.analysis import compare_experiments, long_doc_and_length_sweeps, regenerate_overrides, validate_submissions
from summ.config import DEFAULT_RUN, Settings
from summ.context import Context
from summ.data.audit import audit_integrity, run_audit
from summ.data.loader import clean_and_split, finish_tokenisation, get_tokenizer, load_raw
from summ.training.ensemble import distill_students, run_ensembles
from summ.training.trainer import run_all
from summ.utils import get_device, set_seed, setup_hf_auth

DEFAULT_STAGES = ["data", "experiments", "ensembles", "analysis", "final"]
ALL_STAGES = DEFAULT_STAGES + ["distill"]


def prepare_data(ctx):
    """Load, audit, tokenise, clean and split the data.

    Args:
        ctx (Context): Context; ``tok, eos, tr, dev, te, idf`` are filled.

    Returns:
        None
    """
    load_raw(ctx)
    audit_integrity(ctx)
    get_tokenizer(ctx)
    tr_doc, tr_sum, te_doc = finish_tokenisation(ctx)
    run_audit(ctx, tr_doc, tr_sum, te_doc, save_path=f"{ctx.s.submit}/length_stats.png")
    clean_and_split(ctx, tr_doc, tr_sum, te_doc)


def run_pipeline(s, stages=None, run_ids=None, override=None):
    """Run the requested stages.

    Args:
        s (Settings): Settings.
        stages (Optional[List[str]]): Subset of :data:`ALL_STAGES` (default :data:`DEFAULT_STAGES`).
        run_ids (Optional[List[str]]): Experiments to run (default :data:`summ.config.DEFAULT_RUN`).
        override (Optional[dict]): Test re-generation overrides for the ``final`` stage.

    Returns:
        Context: Context holding data and results.
    """
    stages = stages or DEFAULT_STAGES
    s.make_dirs()
    os.chdir(s.work)
    set_seed(s.seed)
    ctx = Context(s=s, device=get_device())
    ctx.hf_token = None if s.local_data else setup_hf_auth()
    print("device:", ctx.device)
    prepare_data(ctx)
    if "experiments" in stages:
        run_all(ctx, run_ids or DEFAULT_RUN)
    if "ensembles" in stages:
        run_ensembles(ctx)
    cmp_ = None
    if "analysis" in stages:
        cmp_ = compare_experiments(ctx)
        long_doc_and_length_sweeps(ctx, cmp_)
    if "final" in stages:
        if override:
            regenerate_overrides(ctx, override)
        validate_submissions(ctx)
    if "distill" in stages:
        distill_students(ctx)
    return ctx
