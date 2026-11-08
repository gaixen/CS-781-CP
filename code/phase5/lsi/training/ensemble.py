"""Weighted ensemble (section 10) and the E4 seed ensemble."""

import copy
import gc
import os
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from lsi.eval.metrics import (
    THR_SLUG,
    fit_temperature,
    greedy_weights,
    honest,
    macro_f1,
    sigmoid,
    tune_global,
)
from lsi.config import ARCH_SHORT
from lsi.training.engine import exp_dir, run_experiment
from lsi.training.submission import process
from lsi.utils import restore_runs


def nested_ens(Zc, Y):
    """Nested 2-fold check: fit weights + threshold on one dev half, score the other (both directions).

    Args:
        Zc (Dict[str, np.ndarray]): Calibrated dev logits per model.
        Y (np.ndarray): Dev references.

    Returns:
        Dict[str, float]: Mean held-out macro-F1 of ``greedy``, ``equal`` and each single model.
    """
    names = list(Zc)
    perm = np.random.RandomState(11).permutation(len(Y))
    h = len(Y) // 2
    res = defaultdict(list)
    for a, b in ((perm[:h], perm[h:]), (perm[h:], perm[:h])):
        cands = {
            "greedy": greedy_weights([Zc[n][a] for n in names], Y[a]),
            "equal": np.ones(len(names)) / len(names),
        }
        cands.update({n: np.eye(len(names))[i] for i, n in enumerate(names)})
        for k, w in cands.items():
            za = sum(wi * Zc[n][a] for wi, n in zip(w, names))
            zb = sum(wi * Zc[n][b] for wi, n in zip(w, names))
            t, _ = tune_global(sigmoid(za), Y[a])
            res[k].append(macro_f1(sigmoid(zb) >= t, Y[b]))
    return {k: float(np.mean(v)) for k, v in res.items()}


def weighted_ensemble(ctx):
    """Greedy dev-optimised ensemble of all usable experiments, validated with a nested split; writes its zip.

    Args:
        ctx (Context): Pipeline context (``avail, zd, zt, temp`` filled by ``load_available``).

    Returns:
        Optional[Dict[str, float]]: Greedy weights per experiment (None if fewer than 2 experiments are available).
    """
    avail, YDEV = ctx.avail, ctx.data.ydev
    if len(avail) < 2:
        return None
    ZC = {e: ctx.zd[e] / ctx.temp[e] for e in avail}
    nest = nested_ens(ZC, YDEV)
    print(
        "nested dev macro-F1 (fit on half, score other half):",
        {k: round(v, 4) for k, v in sorted(nest.items(), key=lambda x: -x[1])},
    )
    w = greedy_weights([ZC[e] for e in avail], YDEV)
    ens_w = dict(zip(avail, w))
    print(
        "greedy weights on full dev:", {k: round(float(v), 2) for k, v in ens_w.items()}
    )
    best_single = max(nest[e] for e in avail)
    print(
        "greedy beats best single in the nested check"
        if nest["greedy"] > best_single
        else "greedy does NOT beat the best single model in the nested check -> prefer the single-model zips"
    )
    mem = [e for e in avail if ens_w[e] > 0]
    Zd = sum(ens_w[e] * ctx.zd[e] / ctx.temp[e] for e in mem)
    Zt = sum(ens_w[e] * ctx.zt[e] / ctx.temp[e] for e in mem)
    key = "ENSW-" + "".join(mem)
    process(ctx, key, Zd, Zt, lambda s, k=key: f"{k}_{THR_SLUG[s]}_greedyW")
    return ens_w


def seed_ensemble(ctx):
    """Train / load several E4 (KD) students with different seeds and write an equal-weight seed-ensemble zip.

    Each seed that is not on disk is a full E4 training run. Teachers need ``dev_logits.npy`` and either
    ``train_logits.npy`` or ``best.pt`` (restored from the previous dataset).

    Args:
        ctx (Context): Pipeline context (``cfg.kd_seeds``, ``cfg.write_all_thr``).

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: Dev / test logits of each finished member.
    """
    cfg, YDEV, data = ctx.cfg, ctx.data.ydev, ctx.data
    restore_runs(cfg, cfg.prev_dataset, need_best=set(), copy_tok_cache=True)
    print("\nlsi_runs in work dir:")
    for d in sorted(os.listdir(cfg.out_dir)):
        print("  ", d, sorted(os.listdir(f"{cfg.out_dir}/{d}")))
    teachers, need_best = [], set()
    for t in ctx.experiments["E4"]["teachers"]:
        d = exp_dir(ctx, t)
        has_dev = os.path.exists(f"{d}/result.json") and os.path.exists(
            f"{d}/dev_logits.npy"
        )
        if has_dev and os.path.exists(f"{d}/train_logits.npy"):
            teachers.append(t)
        elif has_dev:
            need_best.add(t)
    if need_best:
        restore_runs(cfg, cfg.prev_dataset, need_best=need_best)
        teachers += [
            t for t in need_best if os.path.exists(f"{exp_dir(ctx, t)}/best.pt")
        ]
    teachers = [t for t in ctx.experiments["E4"]["teachers"] if t in teachers]
    print("\nusable teachers:", teachers)
    assert teachers, (
        "no usable teacher (need dev_logits.npy + result.json and train_logits.npy or best.pt in lsi_runs/E1_*, E2_*, E3_*)"
    )
    members, e4_orig = {}, ctx.experiments["E4"]
    for s in cfg.kd_seeds:
        eid = "E4" if s == cfg.seed else f"E4s{s}"
        ctx.experiments[eid] = {
            **copy.deepcopy(e4_orig),
            "teachers": teachers,
            **({} if s == cfg.seed else {"name": f"KD-E1E2E3-T2-seed{s}"}),
        }
        try:
            run_experiment(ctx, eid, seed=s)
            d = exp_dir(ctx, eid)
            members[f"E4(seed{s})"] = (
                np.load(f"{d}/dev_logits.npy"),
                np.load(f"{d}/test_logits.npy"),
            )
        except Exception as ex:
            print(f"[{eid}] FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            if eid == "E4":
                ctx.experiments["E4"] = e4_orig
            else:
                ctx.experiments.pop(eid, None)
    assert len(members) >= 2, (
        f"only {len(members)} finished member(s); need at least 2 for an ensemble"
    )
    for n, (zd, zt) in members.items():
        assert zd.shape == (len(data.dev_docs), data.n_labels) and zt.shape == (
            len(data.test_docs),
            data.n_labels,
        ), f"{n}: logits do not match the current data"
    Ts = {n: fit_temperature(zd, YDEV) for n, (zd, _) in members.items()}
    Zd = np.mean([members[n][0] / Ts[n] for n in members], 0)
    Zt = np.mean([members[n][1] / Ts[n] for n in members], 0)
    print("\nhonest (out-of-fold) dev macro-F1, global threshold:")
    for n in members:
        print(
            f"  {n:<14} {honest('global', sigmoid(members[n][0] / Ts[n]), YDEV)[0]:.4f}"
        )
    print(
        f"  {'SEED ENSEMBLE':<14} {honest('global', sigmoid(Zd), YDEV)[0]:.4f}   <- submit the zip only if this beats the best single member"
    )
    n, arch = len(members), ARCH_SHORT[ctx.experiments["E4"]["arch"]]
    process(
        ctx,
        f"E4x{n}seeds",
        Zd,
        Zt,
        lambda s: f"E4-KDx{n}seeds_{THR_SLUG[s]}_{arch}_LWAN_BCE_doc-full",
        write_all=cfg.write_all_thr,
    )
    pd.DataFrame(ctx.submit_zips).T.to_csv(f"{cfg.submit_dir}/submission_plan.csv")
    return members
