import os

from lsi.analysis.audits import audit_citations, audit_data, audit_predictions, audit_zips
from lsi.analysis.diagnostics import diagnostics, load_available
from lsi.config import Config
from lsi.context import Context
from lsi.data.loader import load_data
from lsi.training.engine import run_many
from lsi.training.ensemble import seed_ensemble, weighted_ensemble
from lsi.training.submission import final_check, make_tag2, process
from lsi.utils import get_device, restore_runs, set_seed, setup_hf_auth

DEFAULT_STAGES = ["restore", "train", "diagnostics", "audits", "ensemble", "final"]
ALL_STAGES = DEFAULT_STAGES + ["seed_ensemble"]


def run_pipeline(cfg, stages=None):
    """Run the requested stages.

    Args:
        cfg (Config): Configuration (``run_train = []`` means post-processing only, CPU is enough).
        stages (Optional[List[str]]): Subset of :data:`ALL_STAGES` (default :data:`DEFAULT_STAGES`).

    Returns:
        Context: Context with data, logits, diagnostics and the written submissions.
    """
    stages = stages or DEFAULT_STAGES
    cfg.make_dirs()
    os.chdir(cfg.work)
    set_seed(cfg.seed)
    ctx = Context(cfg=cfg, device=get_device())
    print("device:", ctx.device)
    token = None if cfg.local_data else setup_hf_auth()
    if "restore" in stages:
        need = set()
        for e in cfg.run_train:
            c = ctx.experiments[e]
            need |= {c["init_from"]} if c.get("init_from") else set()
            need |= set(c.get("teachers", []))
        restore_runs(cfg, cfg.prev_dataset, need_best=need, copy_tok_cache=bool(cfg.run_train))
    ctx.data = load_data(cfg, token)
    if "train" in stages and cfg.run_train:
        run_many(ctx, cfg.run_train)
    load_available(ctx)
    pred = None
    if "diagnostics" in stages or "audits" in stages or "ensemble" in stages:
        diagnostics(ctx)
    if "audits" in stages:
        audit_data(ctx)
        pred = audit_predictions(ctx)
        audit_citations(ctx, pred)
    if "ensemble" in stages:
        for e in ctx.avail:
            process(ctx, e, ctx.zd[e], ctx.zt[e], lambda s, e=e: make_tag2(ctx, e, s), write_all=(e in cfg.panel_exps))
        weighted_ensemble(ctx)
    if "final" in stages:
        final_check(ctx)
        audit_zips(ctx)
    if "seed_ensemble" in stages:
        seed_ensemble(ctx)
    return ctx
