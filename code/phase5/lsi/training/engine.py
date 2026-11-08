"""Training and inference engine (resumable per epoch, wall-clock capped) and the experiment runner."""

import gc
import json
import math
import os
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lsi.data.chunking import build_batch, get_stores, pad_batch, select_chunks
from lsi.eval.metrics import fit_temperature, sigmoid, tune_global
from lsi.models.lsi_model import build_model, count_params, elem_loss, param_groups
from lsi.utils import set_seed


def exp_dir(ctx, eid):
    """Run folder of an experiment (created on demand).

    Args:
        ctx (Context): Pipeline context.
        eid (str): Experiment id.

    Returns:
        str: ``<out_dir>/<eid>_<name>``.
    """
    d = f"{ctx.cfg.out_dir}/{eid}_{ctx.experiments[eid]['name']}"
    os.makedirs(d, exist_ok=True)
    return d


@torch.no_grad()
def predict_logits(ctx, model, store, pad, k=None, mode=None, budget=None):
    """Statute logits of every document of a store.

    Args:
        ctx (Context): Pipeline context.
        model (LSIModel): Trained model.
        store (Store): Token store.
        pad (int): Pad id.
        k (Optional[int]): Chunks per document (default ``cfg.inf_chunks``).
        mode (Optional[str]): Chunk selection mode (default ``cfg.doc_mode``).
        budget (Optional[int]): Max chunks per batch (default ``cfg.inf_chunk_budget``).

    Returns:
        np.ndarray: Logits (n_docs, L).
    """
    cfg, L = ctx.cfg, ctx.data.n_labels
    model.eval()
    k, mode, budget = (
        k or cfg.inf_chunks,
        mode or cfg.doc_mode,
        budget or cfg.inf_chunk_budget,
    )
    n = len(store)
    out = np.zeros((n, L), np.float32)
    sel = [
        select_chunks(
            len(store.spans[i]),
            k,
            mode,
            None,
            None if store.score is None else store.score[i],
        )
        for i in range(n)
    ]
    order = np.argsort([len(s) for s in sel], kind="stable")
    i = 0
    while i < n:
        j, tot = i, 0
        while j < n and (tot + len(sel[order[j]]) <= budget or j == i):
            tot += len(sel[order[j]])
            j += 1
        idx = order[i:j]
        seqs, owner = [], []
        for b, d in enumerate(idx):
            for c in sel[d]:
                a, e = store.spans[d][c]
                seqs.append(store.tok[d][a:e])
                owner.append(b)
        ids, mask = pad_batch(seqs, pad)
        out[idx] = (
            model(ids.to(ctx.device), mask.to(ctx.device), np.array(owner), len(idx))
            .float()
            .cpu()
            .numpy()
        )
        i = j
    return out


def trainable_state(model):
    """CPU copy of every trainable parameter.

    Args:
        model (nn.Module): Model.

    Returns:
        Dict[str, torch.Tensor]: ``name -> tensor``.
    """
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def train_model(ctx, d, exp_cfg, model, S, tlog=None, seed=None):
    """Train one model (AdamW, warm-up + linear decay, AMP, accumulation, rare-label oversampling, optional KD).

    Resumes from ``<d>/ckpt.pt`` if present; the best epoch (dev macro-F1 at the best global threshold) is kept and saved
    to ``<d>/best.pt``; stops after 2 non-improving epochs or when the wall-clock cap is reached.

    Args:
        ctx (Context): Pipeline context.
        d (str): Run folder.
        exp_cfg (dict): Experiment definition.
        model (LSIModel): Model to train.
        S (dict): Token stores of the architecture (``train``, ``dev``, ``pad``).
        tlog (Optional[np.ndarray]): Teacher logits for the train set (KD).
        seed (Optional[int]): Seed (default ``cfg.seed``).

    Returns:
        dict: ``hist``, ``best_dev_f1``, ``train_min``, ``peak_mem_mb``.
    """
    cfg, dev, data = ctx.cfg, ctx.device, ctx.data
    seed = cfg.seed if seed is None else seed
    set_seed(seed)
    store, dstore, pad = S["train"], S["dev"], S["pad"]
    N = len(store)
    YTR, YDEV = data.ytr, data.ydev
    opt = torch.optim.AdamW(
        param_groups(model, exp_cfg, cfg.ft_lr_head), weight_decay=exp_cfg["wd"]
    )
    total = math.ceil(N / (cfg.bs_docs * cfg.accum)) * exp_cfg["epochs"]
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and dev.type == "cuda"))
    rng = np.random.RandomState(seed)
    npos = YTR.sum(0)
    pw = torch.tensor(
        np.clip(np.sqrt((N - npos) / np.maximum(npos, 1)), 1, 30),
        dtype=torch.float32,
        device=dev,
    )
    lw = torch.tensor(
        (np.maximum(npos, 1) ** -exp_cfg.get("kd_rare_boost", 0.0)),
        dtype=torch.float32,
        device=dev,
    )
    lw = lw / lw.mean()
    Yt = torch.from_numpy(YTR)
    tl = None if tlog is None else torch.from_numpy(tlog)
    ovp = exp_cfg.get("oversample", 0.0)
    lab_w = np.maximum(npos, 1) ** -ovp
    dw = (YTR * lab_w[None]).max(1)
    dw[dw == 0] = lab_w.mean()
    dw = dw / dw.sum()
    ck = f"{d}/ckpt.pt"
    ep0, best, best_state, hist, bad = 0, -1.0, None, [], 0
    if os.path.exists(ck):
        c = torch.load(ck, map_location="cpu")
        model.load_state_dict(c["model"], strict=False)
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"])
        ep0, best, best_state, hist, bad = (
            c["ep"],
            c["best"],
            c["best_state"],
            c["hist"],
            c["bad"],
        )
        print(f"  resuming at epoch {ep0 + 1}")
    t0 = time.time()
    stop = False
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(ep0, exp_cfg["epochs"]):
        model.train()
        _r = np.random.RandomState(seed + ep)
        perm = _r.choice(N, N, p=dw) if ovp > 0 else _r.permutation(N)
        nb = math.ceil(N / cfg.bs_docs)
        losses = []
        for bi in range(nb):
            idx = perm[bi * cfg.bs_docs : (bi + 1) * cfg.bs_docs]
            ids, mask, owner = build_batch(
                store, idx, cfg.train_chunks, exp_cfg["doc_mode"], rng, pad
            )
            y = Yt[idx].to(dev)
            z = model(ids.to(dev), mask.to(dev), owner, len(idx))
            hard = elem_loss(exp_cfg["loss"], z, y, pw).sum(1).mean()
            if tl is not None:
                T = exp_cfg["kd_T"]
                tz = tl[idx].to(dev)
                kd = (
                    (
                        F.binary_cross_entropy_with_logits(
                            z.float() / T, torch.sigmoid(tz / T), reduction="none"
                        )
                        * lw[None]
                    )
                    .sum(1)
                    .mean()
                    * T
                    * T
                )
                loss = exp_cfg["kd_alpha"] * hard + (1 - exp_cfg["kd_alpha"]) * kd
            else:
                loss = hard
            scaler.scale(loss / cfg.accum).backward()
            losses.append(loss.item())
            if (bi + 1) % cfg.accum == 0 or bi == nb - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if (bi + 1) % 200 == 0:
                print(
                    f"    ep{ep + 1} {bi + 1}/{nb} loss={np.mean(losses[-200:]):.3f} ({(time.time() - t0) / 60:.1f} min)"
                )
            if (time.time() - t0) / 60 > cfg.max_min_per_exp:
                print("  wall-clock cap reached -> evaluating and stopping")
                stop = True
                break
        Zd = predict_logits(ctx, model, dstore, pad)
        thr, f1 = tune_global(sigmoid(Zd), YDEV)
        hist.append(
            dict(epoch=ep + 1, loss=float(np.mean(losses)), dev_macro_f1=f1, thr=thr)
        )
        print(
            f"  ep{ep + 1}/{exp_cfg['epochs']} loss={np.mean(losses):.3f} dev macro-F1(global thr {thr:.2f})={f1:.4f} [{(time.time() - t0) / 60:.1f} min]"
        )
        if f1 > best:
            best, best_state, bad = f1, trainable_state(model), 0
        else:
            bad += 1
        torch.save(
            dict(
                model=trainable_state(model),
                opt=opt.state_dict(),
                sched=sched.state_dict(),
                scaler=scaler.state_dict(),
                ep=ep + 1,
                best=best,
                best_state=best_state,
                hist=hist,
                bad=bad,
            ),
            ck + ".tmp",
        )
        os.replace(ck + ".tmp", ck)
        if stop or bad >= 2:
            break
    model.load_state_dict(best_state, strict=False)
    torch.save(best_state, f"{d}/best.pt")
    if os.path.exists(ck):
        os.remove(ck)
    return dict(
        hist=hist,
        best_dev_f1=best,
        train_min=(time.time() - t0) / 60,
        peak_mem_mb=(
            torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else 0.0
        ),
    )


def load_trained(ctx, eid):
    """Rebuild an experiment's model and load its ``best.pt``.

    Args:
        ctx (Context): Pipeline context.
        eid (str): Experiment id.

    Returns:
        LSIModel: Trained model.
    """
    model = build_model(ctx, ctx.experiments[eid])
    r = model.load_state_dict(
        torch.load(f"{exp_dir(ctx, eid)}/best.pt", map_location="cpu"), strict=False
    )
    assert not [
        k
        for k in r.missing_keys
        if any(p.requires_grad for n, p in model.named_parameters() if n == k)
    ], "trained weights missing"
    return model


def ensure_train_logits(ctx, eid):
    """Train-set logits of an experiment (computed from ``best.pt`` once, then cached as ``train_logits.npy``).

    Args:
        ctx (Context): Pipeline context.
        eid (str): Experiment id.

    Returns:
        np.ndarray: Logits (n_train, L).
    """
    f = f"{exp_dir(ctx, eid)}/train_logits.npy"
    if not os.path.exists(f):
        S = get_stores(ctx, ctx.experiments[eid]["arch"])
        m = load_trained(ctx, eid)
        np.save(f, predict_logits(ctx, m, S["train"], S["pad"]))
        del m
        gc.collect()
        torch.cuda.empty_cache()
    return np.load(f)


def run_experiment(ctx, eid, seed=None):
    """Run (or load from cache) one experiment; writes ``dev_logits.npy``, ``test_logits.npy``, ``best.pt`` and ``result.json``.

    KD experiments distil from the temperature-calibrated, weighted mean of their available teachers' train logits.

    Args:
        ctx (Context): Pipeline context.
        eid (str): Experiment id (key of ``ctx.experiments``).
        seed (Optional[int]): Seed (default ``cfg.seed``).

    Returns:
        dict: Result with parameter counts, speed and training history.
    """
    cfg, data = ctx.cfg, ctx.data
    seed = cfg.seed if seed is None else seed
    exp_cfg = ctx.experiments[eid]
    d = exp_dir(ctx, eid)
    rf = f"{d}/result.json"
    if os.path.exists(rf):
        print(f"[{eid}] cached")
        return json.load(open(rf))
    print(
        f"\n===== {eid} {exp_cfg['name']} | {exp_cfg['arch']} | {exp_cfg['method']} | loss={exp_cfg['loss']} head={exp_cfg['head']} ====="
    )
    S = get_stores(ctx, exp_cfg["arch"])
    tlog = None
    if exp_cfg["method"] == "kd":
        Zs, Ws = [], []
        for t in exp_cfg["teachers"]:
            if not os.path.exists(f"{exp_dir(ctx, t)}/result.json"):
                print(f"  teacher {t} not available -> skipped")
                continue
            Zt = ensure_train_logits(ctx, t)
            T_ = fit_temperature(
                np.load(f"{exp_dir(ctx, t)}/dev_logits.npy"), data.ydev
            )
            print(f"  teacher {t}: dev temperature {T_:.2f}")
            Zs.append(Zt / T_)
            Ws.append(exp_cfg.get("teacher_w", {}).get(t, 1.0))
        assert Zs, "no teacher available -> run E1-E3 first"
        tlog = np.tensordot(np.array(Ws) / np.sum(Ws), np.stack(Zs), 1).astype(
            np.float32
        )
        exp_cfg["teachers_used"] = len(Zs)
    set_seed(seed)
    model = build_model(ctx, exp_cfg)
    tot, tr = count_params(model)
    if exp_cfg.get("init_from"):
        r_ = model.load_state_dict(
            torch.load(
                f"{exp_dir(ctx, exp_cfg['init_from'])}/best.pt", map_location="cpu"
            ),
            strict=False,
        )
        print(
            f"  warm start from {exp_cfg['init_from']} best.pt (unexpected keys: {len(r_.unexpected_keys)})"
        )
    info = train_model(ctx, d, exp_cfg, model, S, tlog, seed)
    t0 = time.time()
    Zd = predict_logits(ctx, model, S["dev"], S["pad"])
    dt = time.time() - t0
    Zt_ = predict_logits(ctx, model, S["test"], S["pad"])
    np.save(f"{d}/dev_logits.npy", Zd)
    np.save(f"{d}/test_logits.npy", Zt_)
    res = dict(
        eid=eid,
        name=exp_cfg["name"],
        arch=exp_cfg["arch"],
        method=exp_cfg["method"],
        params_total_M=tot / 1e6,
        params_trainable_M=tr / 1e6,
        trainable_pct=100 * tr / tot,
        weights_fp16_MB=tot * 2 / 1e6,
        best_state_MB=os.path.getsize(f"{d}/best.pt") / 1e6,
        dev_docs_per_s=len(S["dev"]) / dt,
        **info,
    )
    json.dump(res, open(rf, "w"), indent=1)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def run_many(ctx, ids):
    """Run several experiments; failures are reported and skipped.

    Args:
        ctx (Context): Pipeline context.
        ids (Iterable[str]): Experiment ids.

    Returns:
        Dict[str, dict]: Results of the experiments that finished.
    """
    results = {}
    for eid in ids:
        if not ctx.cfg.have(ctx.experiments[eid]["arch"]):
            print(
                f"[{eid}] skipped: no Phase-1 checkpoint for {ctx.experiments[eid]['arch']}"
            )
            continue
        try:
            results[eid] = run_experiment(ctx, eid)
        except Exception as ex:
            print(f"[{eid}] FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
    print("trained this session:", list(results))
    return results
