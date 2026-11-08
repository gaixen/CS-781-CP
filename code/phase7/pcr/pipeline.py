"""End-to-end orchestration of Phase 7, in the same order as the original notebook.

Stages: ``data`` -> ``experiments`` -> ``analysis`` (comparison / ablation / fusion / ensemble)
-> ``ltr`` (dev-fitted rerankers) -> ``ltr_train`` (train-split rerankers, two-stage) -> ``audits``.
"""
import os

import numpy as np

from pcr.analysis.audits import audit_v3, audit_v4, precision_audit
from pcr.analysis.comparison import compare_experiments, representation_ablation
from pcr.analysis.failure_analysis import failure_analysis
from pcr.config import DEFAULT_RUN, Config
from pcr.context import PCRContext
from pcr.data.audit import audit_schema, length_statistics
from pcr.data.loader import describe_raw, load_raw_splits, locate_data_paths
from pcr.data.preprocessing import preprocess_splits, print_preprocess_summary
from pcr.data.tokenization import build_splits, get_tokenizer, truncation_table
from pcr.rerank.common import available_experiments, signals
from pcr.rerank.fusion import ensemble_experiments, fuse_experiments
from pcr.rerank.ltr_dev import LTRTuner, run_ltr_v1, run_ltr_v3, run_ltr_v4
from pcr.rerank.ltr_train import build_ltr_state, run_ltr_v5, run_ltr_v6
from pcr.rerank.ltr_two_stage import TwoStageLTR
from pcr.retrieval.lexical import build_lexical_suite, compute_lexical_baselines, mine_hard_negatives, select_best_tfidf
from pcr.retrieval.rules import diagnose_rule_features, extra_feats, rule_feats
from pcr.training.experiments import run_all
from pcr.training.trainer import build_train_data
from pcr.utils import get_device, set_seed, setup_hf_auth

ALL_STAGES = ["data", "experiments", "analysis", "ltr", "ltr_train", "audits"]


def prepare_data(ctx):
    """Stages 1-4 of the notebook: load, preprocess, audit, tokenise, lexical baselines, rule features, hard negatives.

    Args:
        ctx (PCRContext): Context; ``paths, raw, meta, tok, eos, spl, lex, rf, train_data`` are filled in place.

    Returns:
        PCRContext: The same context.
    """
    cfg = ctx.cfg
    token = setup_hf_auth() if not cfg.local_data else None
    ctx.paths = locate_data_paths(cfg, token)
    print(ctx.paths)
    ctx.raw = load_raw_splits(ctx.paths)
    describe_raw(ctx.raw)
    ctx.meta, st = preprocess_splits(ctx.raw, cfg.clean)
    print_preprocess_summary(ctx.raw, ctx.meta, st)
    audit_schema(ctx.raw, cfg.k)
    length_statistics(ctx.raw, show=False, save_path=f"{cfg.submit_dir}/length_stats.png")
    ctx.tok, ctx.eos = get_tokenizer(cfg)
    ctx.spl = build_splits(ctx.raw, ctx.tok, ctx.eos, cfg)
    truncation_table(ctx.spl)
    ctx.lex = compute_lexical_baselines(ctx.spl, cfg)
    hard, items = mine_hard_negatives(ctx.spl, ctx.lex["bm25"]["train"], cfg)      # BM25 is the hard-negative miner
    ctx.train_data = build_train_data(ctx.spl, hard, items)
    for s in ("dev", "test"):
        ctx.rf[s] = rule_feats(s, ctx.spl, ctx.meta)
    diagnose_rule_features(ctx.rf["dev"], ctx.spl)
    return ctx


def run_analysis(ctx):
    """Sections 8-10b: comparison table, representation ablation, fusion and ensemble.

    Args:
        ctx (PCRContext): Context with finished experiments.

    Returns:
        str: Id of the best experiment (by dev F1).
    """
    CMP = compare_experiments(ctx, show=False)
    best = CMP["dev_f1"].idxmax()
    representation_ablation(ctx, best)
    fuse_experiments(ctx, ctx.lex["bm25"])
    ensemble_experiments(ctx, ctx.lex["bm25"])
    return best


def run_dev_rerankers(ctx, paths):
    """Sections 10c-10d, v3 and v4 rerankers (fitted on dev, honest CV).

    Args:
        ctx (PCRContext): Context.
        paths (Dict[str, str]): Local data paths (for the RAW headers).

    Returns:
        dict: ``exps, v1, tuner_result, v3, v4, tuner``.
    """
    exps = available_experiments(ctx)
    out = dict(exps=exps)
    out["v1"] = run_ltr_v1(ctx, exps)
    tuner = LTRTuner(ctx, exps)
    out["tuner"] = tuner
    out["tuner_result"] = tuner.run()
    print("best TF-IDF variant chosen on dev:", select_best_tfidf(ctx.spl, ctx.lex, ctx.cfg))
    raw0 = load_raw_splits(paths, ("dev", "dev_c", "test", "test_c"))
    for s in ("dev", "test"):
        ctx.ex[s] = extra_feats(s, ctx.spl, raw0)
    out["v3"] = run_ltr_v3(ctx, exps)
    ctx.lex2 = build_lexical_suite(ctx.spl, ctx.cfg)
    out["v4"] = run_ltr_v4(ctx, exps)
    return out


def run_train_rerankers(ctx, paths, dev_out):
    """v5 / v6 (train-LTR stacking, PRF) and the two-stage A / B / C rerankers.

    Args:
        ctx (PCRContext): Context.
        paths (Dict[str, str]): Local data paths.
        dev_out (dict): Output of :func:`run_dev_rerankers`.

    Returns:
        dict: ``state, v5, v6, two_stage, final_ab, stage_c``.
    """
    exps, v4 = dev_out["exps"], dev_out["v4"]
    out = {}
    out["state"] = state = build_ltr_state(ctx, paths)
    out["v5"] = v5 = run_ltr_v5(ctx, exps, state)
    out["v6"] = run_ltr_v6(ctx, v5)
    ts = TwoStageLTR(ctx, exps, state, v4)
    ts.fit_stage_a()
    ts.fit_stage_b()
    out["two_stage"] = ts
    out["final_ab"] = ts.final_ab()
    out["stage_c"] = ts.stage_c()
    return out


def run_pipeline(cfg, stages=None, run_ids=None):
    """Run the requested stages end to end.

    Args:
        cfg (Config): Pipeline configuration.
        stages (Optional[List[str]]): Subset of :data:`ALL_STAGES` (default: all).
        run_ids (Optional[List[str]]): Experiments to run (default: :data:`pcr.config.DEFAULT_RUN`).

    Returns:
        PCRContext: Context holding all intermediate artefacts (``ctx.outputs`` dict).
    """
    stages = stages or ALL_STAGES
    cfg.make_dirs()
    os.chdir(cfg.work)
    set_seed(cfg.seed)
    ctx = PCRContext(cfg=cfg, device=get_device())
    print("device:", ctx.device)
    prepare_data(ctx)
    ctx.outputs = {}
    if "experiments" in stages:
        run_all(ctx, run_ids or DEFAULT_RUN)
    if "analysis" in stages:
        ctx.outputs["best"] = run_analysis(ctx)
    dev_out = None
    if "ltr" in stages or "ltr_train" in stages or "audits" in stages:
        dev_out = run_dev_rerankers(ctx, ctx.paths)
        ctx.outputs["dev"] = dev_out
    tr_out = None
    if "ltr_train" in stages or "audits" in stages:
        tr_out = run_train_rerankers(ctx, ctx.paths, dev_out)
        ctx.outputs["train"] = tr_out
    if "audits" in stages:
        exps = dev_out["exps"]
        best = ctx.outputs.get("best") or "E1"
        failure_analysis(ctx, best)
        audit_v3(ctx, dev_out["v4"], exps, signals(ctx, "dev", exps))
        audit_v4(ctx, np.where(np.isnan(tr_out["two_stage"].Bz), np.nanmin(tr_out["two_stage"].Bz) - 1.0, tr_out["two_stage"].Bz))
        precision_audit(ctx, dev_out["tuner_result"]["cfg"], dev_out["tuner"], dev_out["tuner_result"]["scores_test"])
    return ctx
