"""Contrastive training.

Loss: symmetric InfoNCE with in-batch + BM25 hard negatives (other queries' positives that are also relevant
to the current query are masked as false negatives). For the KD student the loss is
``a * InfoNCE + (1 - a) * (T^2 * KL(teacher || student) + KD_MSE * MSE(similarities))``.
"""
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from pcr.models.dual_encoder import encode_batch, score_split
from pcr.retrieval.metrics import f1_at_k


@dataclass
class TrainData:
    """Training-side lookups.

    Args:
        qtok (dict): ``query_id -> token ids`` of the train queries.
        ctok (dict): ``candidate_id -> token ids`` of the train pool.
        gold (dict): ``query_id -> set(gold candidate ids)``.
        hard (Dict[str, List[str]]): Hard-negative candidate ids per query.
        items (List[str]): Trainable query ids (at least one gold document in the pool).
    """
    qtok: dict
    ctok: dict
    gold: dict
    hard: Dict[str, List[str]]
    items: List[str]


def build_train_data(spl, hard, tr_items):
    """Assemble the :class:`TrainData` container.

    Args:
        spl (dict): Split container.
        hard (dict): Hard negatives from ``mine_hard_negatives``.
        tr_items (List[str]): Trainable query ids.

    Returns:
        TrainData: Container used by :func:`step_loss` and :func:`train_exp`.
    """
    return TrainData(qtok=spl["train"]["qtok"], ctok=spl["train"]["ctok"], gold=spl["train"]["gold"], hard=hard, items=tr_items)


def exp_dir(cfg, eid):
    """Return (and create) the run directory of an experiment.

    Args:
        cfg (Config): Configuration (``out_dir``).
        eid (str): Experiment id (e.g. ``E1``).

    Returns:
        str: Directory path.
    """
    d = f"{cfg.out_dir}/{eid}"
    os.makedirs(d, exist_ok=True)
    return d


def step_loss(model, teacher, qids, rng, td, cfg):
    """Loss of one micro-batch of queries.

    Each query brings 1 positive and ``cfg.n_hard`` hard negatives; the teacher (if any) sees exactly the same
    randomly selected chunks as the student because the RNG state is restored before encoding.

    Args:
        model (DualEnc): Student / model being trained.
        teacher (Optional[DualEnc]): Frozen teacher for knowledge distillation (None otherwise).
        qids (List[str]): Query ids of the micro-batch.
        rng (np.random.RandomState): RNG (positives, negatives, chunk selection).
        td (TrainData): Training lookups.
        cfg (Config): Configuration.

    Returns:
        torch.Tensor: Scalar loss.
    """
    B = len(qids)
    pos = [rng.choice(sorted(td.gold[q])) for q in qids]
    hn = [list(rng.choice(td.hard[q], cfg.n_hard, replace=False)) if len(td.hard[q]) >= cfg.n_hard else [] for q in qids]
    dids = pos + [h for hs in hn for h in hs]

    def enc(m):
        """Encode the batch queries and documents with model ``m`` using RNG-derived chunk selection.

        Args:
            m (DualEnc): Encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(query_embeddings, document_embeddings)``.
        """
        rr = np.random.RandomState(rng.randint(1 << 30))
        q = encode_batch(m, [td.qtok[i] for i in qids], cfg.train_strat, cfg.tr_ch, cfg.agg, rr)
        D = encode_batch(m, [td.ctok[i] for i in dids], cfg.train_strat, cfg.tr_ch, cfg.agg, rr)
        return q, D

    state = rng.get_state()
    q, D = enc(model)
    mask = torch.zeros(B, len(dids), dtype=torch.bool)
    for i, qid in enumerate(qids):
        for j, d in enumerate(dids):
            if j != i and d in td.gold[qid]:
                mask[i, j] = True                      # other query's positive that is also relevant here -> not a negative
    mask = mask.to(model.device)
    lg = (q @ D.T / cfg.tau).masked_fill(mask, -1e4)
    tgt = torch.arange(B, device=model.device)
    loss = F.cross_entropy(lg, tgt) + 0.5 * F.cross_entropy(D[:B] @ q.T / cfg.tau, tgt)
    if teacher is not None:
        with torch.no_grad():
            rng.set_state(state)
            tq, tD = enc(teacher)
            tl = (tq @ tD.T / cfg.tau).masked_fill(mask, -1e4)
        kd = F.kl_div(F.log_softmax(lg / cfg.kd_t, -1), F.softmax(tl / cfg.kd_t, -1), reduction="batchmean") * cfg.kd_t ** 2
        loss = cfg.kd_alpha * loss + (1 - cfg.kd_alpha) * (kd + cfg.kd_mse * F.mse_loss(q @ D.T, tq @ tD.T))
    return loss


def train_exp(eid, exp_cfg, model, teacher, td, spl, cfg):
    """Train one experiment and keep the best epoch (by dev F1@K) in ``best.pt``.

    Uses AdamW, linear warm-up (6 %) + cosine decay, gradient accumulation, gradient clipping (1.0),
    fp16 GradScaler and a wall-clock cap (``cfg.max_min_per_exp``).

    Args:
        eid (str): Experiment id.
        exp_cfg (dict): Experiment definition (``lr``, ``epochs`` ...).
        model (DualEnc): Model to train (only ``requires_grad`` parameters are optimised).
        teacher (Optional[DualEnc]): Frozen teacher for KD.
        td (TrainData): Training lookups.
        spl (dict): Split container (dev evaluation after every epoch).
        cfg (Config): Configuration.

    Returns:
        Tuple[List[dict], float]: ``(history, training_seconds)`` where history holds per-epoch loss / dev P, R, F1.
    """
    d = exp_dir(cfg, eid)
    params = [p for p in model.parameters() if p.requires_grad]
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    epochs = 1 if cfg.smoke else exp_cfg["epochs"]
    steps_per_ep = max(1, len(td.items) // (cfg.bs * cfg.accum))
    total = epochs * steps_per_ep
    warm = max(1, int(0.06 * total))
    opt = torch.optim.AdamW(params, lr=exp_cfg["lr"], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler(enabled=bool(cfg.amp and model.device.type == "cuda"))
    rng = np.random.RandomState(cfg.seed)
    hist, best, t0 = [], -1, time.time()
    step = 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(td.items))
        run = []
        for bi in range(steps_per_ep):
            for a in range(cfg.accum):
                s = (bi * cfg.accum + a) * cfg.bs
                qids = [td.items[i] for i in perm[s:s + cfg.bs]]
                if len(qids) < 2:
                    continue
                loss = step_loss(model, teacher, qids, rng, td, cfg) / cfg.accum
                scaler.scale(loss).backward()
                run.append(loss.item() * cfg.accum)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sch.step()
            step += 1
            if bi % 25 == 0:
                print(f"  ep{ep+1} step {bi}/{steps_per_ep} loss {np.mean(run[-50:]):.4f}  ({(time.time()-t0)/60:.1f} min)")
            if (time.time() - t0) / 60 > cfg.max_min_per_exp:
                break
        Sd = score_split(model, spl, "dev", nch=cfg.ev_ch_fast)
        m = f1_at_k(Sd, spl, "dev", cfg.k)
        hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(run)), **m))
        print(f"  == epoch {ep+1}: train loss {np.mean(run):.4f} | dev F1@{cfg.k} {m['f1']:.4f} P {m['p']:.4f} R {m['r']:.4f}")
        if m["f1"] > best:
            best = m["f1"]
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}, f"{d}/best.pt")
        if (time.time() - t0) / 60 > cfg.max_min_per_exp:
            print("  time cap reached")
            break
    return hist, time.time() - t0


def load_best(model, eid, cfg):
    """Load the best-epoch trainable weights of an experiment into ``model`` (if ``best.pt`` exists).

    Args:
        model (DualEnc): Model to update in place.
        eid (str): Experiment id.
        cfg (Config): Configuration.

    Returns:
        DualEnc: The same model.
    """
    p = f"{exp_dir(cfg, eid)}/best.pt"
    if os.path.exists(p):
        model.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
    return model
