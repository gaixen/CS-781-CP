"""Training, inference and the run registry of the RR experiments.

Training uses overlapping windows of consecutive sentences; inference encodes every sentence once, runs the sequence head
on overlapping windows and merges posteriors with an edge-down-weighted mean. Only trainable tensors are checkpointed.
"""
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rr.data.tokenization import make_inputs_list, make_windows, train_windows
from rr.eval.metrics import confusions, print_scores, rr_scores
from rr.models.peft import amp_settings, count_params
from rr.models.rr_model import build_rr_model
from rr.utils import seed_everything


# ---------------------------------------------------------------------------------------------- losses / helpers
def class_weights(ctx, docs, cfg):
    """Class weights ``(max_count / count) ** CLASS_WEIGHT_POWER`` clipped to ``[1, CLASS_WEIGHT_MAX]``.

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Training documents.
        cfg (dict): Config.

    Returns:
        torch.Tensor: Weights (n_labels,).
    """
    nl = ctx.data.nl
    cnt = np.zeros(nl)
    for d in docs:
        cnt += np.bincount(d["y"], minlength=nl)
    w = (cnt.max() / np.maximum(cnt, 1)) ** cfg["CLASS_WEIGHT_POWER"] if cfg["CLASS_WEIGHT_POWER"] > 0 else np.ones(nl)
    return torch.tensor(np.clip(w, 1.0, cfg["CLASS_WEIGHT_MAX"]), dtype=torch.float)


def set_log_prior(ctx, docs):
    """Store ``log`` class prior of the training documents in ``ctx.log_prior`` (used by logit adjustment).

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Training documents.

    Returns:
        None
    """
    cnt = np.bincount(np.concatenate([d["y"] for d in docs]), minlength=ctx.data.nl).astype(float) + 1.0
    ctx.log_prior = torch.log(torch.tensor(cnt / cnt.sum(), dtype=torch.float))


def oversample_windows(ctx, wins, cfg):
    """Rare-label importance sampling of training windows (epoch size unchanged).

    Each window is drawn with probability proportional to ``(1 / count of its rarest label) ** OVERSAMPLE_POWER``.

    Args:
        ctx (Context): Pipeline context.
        wins (List[Tuple[dict, int, int]]): ``(doc, start, end)`` windows.
        cfg (dict): Config.

    Returns:
        List[Tuple[dict, int, int]]: Resampled windows.
    """
    pw = float(cfg.get("OVERSAMPLE_POWER", 0.0) or 0.0)
    if pw <= 0:
        return wins
    nl = ctx.data.nl
    cnt = np.zeros(nl)
    for d in {id(d): d for d, a, b in wins}.values():
        cnt += np.bincount(d["y"], minlength=nl)
    w = np.array([max(1.0 / max(cnt[c], 1.0) for c in set(d["y"][a:b].tolist())) ** pw for d, a, b in wins])
    w = w / w.sum()
    idx = random.choices(range(len(wins)), weights=w.tolist(), k=len(wins))
    print(f"[oversample] power {pw}: {len(set(idx))}/{len(wins)} distinct windows drawn")
    return [wins[i] for i in idx]


def forward_batch(ctx, model, batch, cfg):
    """Encode a batch of windows and run the sequence head.

    Args:
        ctx (Context): Pipeline context.
        model (RRModel): Model.
        batch (List[Tuple[dict, int, int]]): ``(doc, a, b)`` windows.
        cfg (dict): Config.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]: ``(x (B,W,D), mask (B,W), tags (B,W), lens)``.
    """
    dev = ctx.device
    items = [(d, i) for d, a, b in batch for i in range(a, b)]
    ids, am, tm = make_inputs_list(ctx, items, cfg)
    v = model.encode(ids.to(dev), am.to(dev), tm.to(dev))
    lens = [b - a for _, a, b in batch]
    V = nn.utils.rnn.pad_sequence(list(torch.split(v, lens)), batch_first=True)
    x = model.seq(V, lens)
    mask = torch.arange(max(lens), device=dev)[None] < torch.tensor(lens, device=dev)[:, None]
    tags = nn.utils.rnn.pad_sequence([torch.from_numpy(d["y"][a:b]) if d["y"] is not None else torch.zeros(b - a, dtype=torch.long) for d, a, b in batch], batch_first=True).to(dev)
    return x, mask, tags, lens


def rr_loss(ctx, model, x, mask, tags, cw, cfg):
    """Supervised loss: class-weighted CE (or CRF NLL + weighted emission CE) + optional label-shift auxiliary task + logit adjustment.

    Args:
        ctx (Context): Pipeline context (``log_prior``).
        model (RRModel): Model.
        x (torch.Tensor): Features (B, W, D).
        mask (torch.Tensor): Bool (B, W).
        tags (torch.Tensor): Gold tags (B, W).
        cw (torch.Tensor): Class weights.
        cfg (dict): Config.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(loss, emissions)``.
    """
    em = model.emissions(x)
    tau = float(cfg.get("LOGIT_ADJ_TAU", 0.0) or 0.0)
    em_t = em + tau * ctx.log_prior.to(em.device, em.dtype) if tau else em           # logit-adjusted TRAINING (Menon et al. 2021); inference unchanged
    ce = F.cross_entropy(em_t[mask], tags[mask], weight=cw, label_smoothing=cfg["LABEL_SMOOTH"])
    loss = (model.crf.nll(em_t, tags, mask) + cfg["EMIT_CE_W"] * ce) if model.crf is not None else ce
    if model.shift is not None and x.size(1) > 1:                                     # paper's MTL: label shift between consecutive sentences
        xi, xj = x[:, :-1], x[:, 1:]
        lg = model.shift(torch.cat([xi, xj, xi - xj], -1)).squeeze(-1)
        m = mask[:, 1:]
        tgt = (tags[:, :-1] != tags[:, 1:]).float()
        if m.any():
            loss = (1 - cfg["SHIFT_LAMBDA"]) * loss + cfg["SHIFT_LAMBDA"] * F.binary_cross_entropy_with_logits(lg[m], tgt[m])
    return loss, em


def kd_loss_probs(s_em, t_prob, mask, T):
    """KL(teacher distribution || student at temperature T); ``t_prob`` is already a probability distribution.

    Args:
        s_em (torch.Tensor): Student emissions (B, W, n).
        t_prob (torch.Tensor): Teacher probabilities (B, W, n).
        mask (torch.Tensor): Bool (B, W).
        T (float): Temperature.

    Returns:
        torch.Tensor: Scalar KD loss.
    """
    return F.kl_div(F.log_softmax(s_em[mask] / T, -1), t_prob[mask], reduction="batchmean") * T * T


def kd_loss(s_em, t_em, mask, T):
    """KD loss from teacher logits.

    Args:
        s_em (torch.Tensor): Student emissions.
        t_em (torch.Tensor): Teacher emissions.
        mask (torch.Tensor): Bool mask.
        T (float): Temperature.

    Returns:
        torch.Tensor: Scalar KD loss.
    """
    return F.kl_div(F.log_softmax(s_em[mask] / T, -1), F.softmax(t_em[mask] / T, -1), reduction="batchmean") * T * T


# ---------------------------------------------------------------------------------------------- inference
@torch.no_grad()
def doc_probs(ctx, model, doc, cfg, enc_bs=96):
    """Posterior probabilities for every sentence of a document.

    Args:
        ctx (Context): Pipeline context.
        model (RRModel): Trained model.
        doc (dict): Document.
        cfg (dict): Config (``INF_WIN, INF_STRIDE``).
        enc_bs (int): Sentences encoded per forward pass.

    Returns:
        np.ndarray: Probabilities (n_sentences, n_labels).
    """
    dev, nl = ctx.device, ctx.data.nl
    model.eval()
    use_amp, dt = amp_settings(cfg["ARCH"], dev)
    n = len(doc["sents"])
    vs = []
    for a in range(0, n, enc_bs):
        ids, am, tm = make_inputs_list(ctx, [(doc, i) for i in range(a, min(a + enc_bs, n))], cfg)
        with torch.autocast(device_type=dev.type, dtype=dt, enabled=use_amp):
            vs.append(model.encode(ids.to(dev), am.to(dev), tm.to(dev)).float())
    V = torch.cat(vs)
    acc = torch.zeros(n, nl, device=dev)
    ws = torch.zeros(n, device=dev)
    k = max(1, cfg["INF_STRIDE"] // 2)
    for a, b in make_windows(n, cfg["INF_WIN"], cfg["INF_STRIDE"]):
        x = model.seq(V[a:b][None], [b - a])
        em = model.emissions(x)
        mask = torch.ones(1, b - a, dtype=torch.bool, device=dev)
        p = model.posteriors(em, mask)[0]
        pos = torch.arange(b - a, device=dev)
        w = torch.minimum(torch.minimum(pos + 1, (b - a) - pos), torch.tensor(k, device=dev)).float() / k
        acc[a:b] += p * w[:, None]
        ws[a:b] += w
    return (acc / ws.clamp(min=1e-6)[:, None]).cpu().numpy()


def collect_probs(ctx, model, docs, cfg):
    """Posteriors of a list of documents.

    Args:
        ctx (Context): Pipeline context.
        model (RRModel): Model.
        docs (List[dict]): Documents.
        cfg (dict): Config.

    Returns:
        List[np.ndarray]: One (n_sentences, n_labels) array per document.
    """
    return [doc_probs(ctx, model, d, cfg) for d in docs]


def score_docs(ctx, docs, probs):
    """Argmax-decode posteriors and score them.

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Documents with gold labels.
        probs (List[np.ndarray]): Posteriors.

    Returns:
        Tuple[dict, List[str], List[str]]: ``(scores, gold_flat, pred_flat)``.
    """
    labels = ctx.data.labels
    gold, pred = [], []
    for d, p in zip(docs, probs):
        gold += d["labels"]
        pred += [labels[i] for i in p.argmax(1)]
    return rr_scores(labels, gold, pred), gold, pred


def eval_by_domain(ctx, docs, probs):
    """Scores overall and per domain (CL / IT).

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Documents.
        probs (List[np.ndarray]): Posteriors.

    Returns:
        Tuple[dict, List[str], List[str]]: ``({"ALL","CL","IT"} -> scores, gold_flat, pred_flat)``.
    """
    sc, g, p = score_docs(ctx, docs, probs)
    out = {"ALL": sc}
    for dom in ("CL", "IT"):
        sel = [(d, pr) for d, pr in zip(docs, probs) if d["domain"] == dom]
        if sel:
            out[dom] = score_docs(ctx, [s[0] for s in sel], [s[1] for s in sel])[0]
    return out, g, p


# ---------------------------------------------------------------------------------------------- training
def train_rr(ctx, model, train_docs, dev_docs, cfg, teacher=None, tcfg=None):
    """Train a model (AdamW, warm-up + cosine, AMP, accumulation) and keep the best epoch by dev macro-F1.

    Args:
        ctx (Context): Pipeline context.
        model (RRModel): Model to train.
        train_docs (List[dict]): Training documents.
        dev_docs (List[dict]): Dev documents (empty list = final fit, last epoch kept).
        cfg (dict): Config.
        teacher (Optional[Union[RRModel, list]]): One teacher model or a list ``[(model, cfg), ...]`` (ensemble).
        tcfg (Optional[dict]): Teacher config when a single teacher is given.

    Returns:
        dict: ``history, best, best_epoch, seconds, peak_mb, state`` (best trainable tensors).
    """
    dev = ctx.device
    teachers = [] if teacher is None else (list(teacher) if isinstance(teacher, (list, tuple)) else [(teacher, tcfg)])
    model.to(dev)
    use_amp, dt = amp_settings(cfg["ARCH"], dev)
    cw = class_weights(ctx, train_docs, cfg).to(dev)
    set_log_prior(ctx, train_docs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dt == torch.float16 and dev.type == "cuda")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = []
    for head in (True, False):
        for decay in (True, False):
            ps = [p for n, p in named if (not n.startswith("backbone.")) == head and ((p.ndim > 1) == decay)]
            if ps:
                groups.append(dict(params=ps, lr=cfg["LR_HEAD"] if head else cfg["LR"][cfg["METHOD"]], weight_decay=cfg["WD"] if decay else 0.0))
    opt = torch.optim.AdamW(groups)
    W, S, BW = cfg["WIN"], cfg["WIN_STRIDE"], cfg["BATCH_WIN"]
    est = sum(len(train_windows(len(d["sents"]), W, S)) for d in train_docs)
    steps = math.ceil(est / BW / cfg["ACCUM"]) * cfg["EPOCHS"]
    warm = max(1, int(cfg["WARMUP"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))))
    trainable = {n for n, _ in named}
    snap = lambda: {n: p.detach().cpu().clone() for n, p in model.state_dict().items() if n in trainable}
    best, best_ep, best_state, hist = -1.0, 0, None, []
    t_start = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(cfg["EPOCHS"]):
        model.train()
        wins = [(d, a, b) for d in train_docs for a, b in train_windows(len(d["sents"]), W, S)]
        wins = oversample_windows(ctx, wins, cfg)
        random.shuffle(wins)
        run, nb = 0.0, 0
        for it, bi in enumerate(range(0, len(wins), BW)):
            batch = wins[bi:bi + BW]
            with torch.autocast(device_type=dev.type, dtype=dt, enabled=use_amp):
                x, mask, tags, _ = forward_batch(ctx, model, batch, cfg)
                loss, em = rr_loss(ctx, model, x, mask, tags, cw, cfg)
                if teachers:
                    with torch.no_grad():
                        tp = 0
                        for tm_, tc_ in teachers:
                            tx, _, _, _ = forward_batch(ctx, tm_, batch, tc_)
                            tp = tp + F.softmax(tm_.emissions(tx) / cfg["KD_TEMP"], -1)
                        tp = tp / len(teachers)
                    loss = cfg["KD_ALPHA"] * loss + (1 - cfg["KD_ALPHA"]) * kd_loss_probs(em, tp, mask, cfg["KD_TEMP"])
            scaler.scale(loss / cfg["ACCUM"]).backward()
            run += loss.item()
            nb += 1
            if (it + 1) % cfg["ACCUM"] == 0 or bi + BW >= len(wins):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for _, p in named], cfg["CLIP"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if (it + 1) % cfg["LOG_EVERY"] == 0:
                print(f"  ep{ep+1} it{it+1}/{math.ceil(len(wins)/BW)} loss={run/nb:.4f}")
        if dev_docs:
            res, _, _ = eval_by_domain(ctx, dev_docs, collect_probs(ctx, model, dev_docs, cfg))
            f1 = res["ALL"]["macro_f1"]
            hist.append(dict(epoch=ep + 1, train_loss=run / max(nb, 1), dev_f1=f1, **{f"dev_f1_{k}": v["macro_f1"] for k, v in res.items() if k != "ALL"}))
            print(f"== epoch {ep+1}: loss={run/max(nb,1):.4f} | dev macro-F1 ALL={f1:.4f} " + " ".join(f"{k}={v['macro_f1']:.4f}" for k, v in res.items() if k != "ALL"))
            if f1 > best:
                best, best_ep, best_state = f1, ep + 1, snap()
        else:
            hist.append(dict(epoch=ep + 1, train_loss=run / max(nb, 1)))
            print(f"== epoch {ep+1}: loss={run/max(nb,1):.4f}")
            best_ep, best_state = ep + 1, snap()
    model.load_state_dict(best_state, strict=False)
    return dict(history=hist, best=best, best_epoch=best_ep, seconds=time.time() - t_start,
                peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0.0, state=best_state)


def run_rr(ctx, name, teacher=None, tcfg=None, train_docs=None, dev_docs=None, show=True, kd_from=None, **ov):
    """Train and evaluate one configuration, save ``<work>/<name>.pt`` and append a record to ``ctx.results``.

    Args:
        ctx (Context): Pipeline context.
        name (str): Run name.
        teacher (Optional[Union[RRModel, list]]): Teacher model(s) for KD.
        tcfg (Optional[dict]): Config of a single teacher.
        train_docs (Optional[List[dict]]): Training documents (default: all train).
        dev_docs (Optional[List[dict]]): Dev documents (default: all dev; ``[]`` = no validation).
        show (bool): Print the score table.
        kd_from (Optional[str]): Description of the teacher(s) for the record.
        **ov: Config overrides (unknown keys are reported and ignored).

    Returns:
        dict: Result record (scores, params, time, memory, throughput, history, checkpoint path).
    """
    unknown = [k for k in ov if k not in ctx.cfg]
    if unknown:
        print(f"!!! WARNING: unknown override keys IGNORED: {unknown} (config keys are case-sensitive)")
    cfg = {**ctx.cfg, **ov}
    seed_everything(cfg["SEED"])
    tr = ctx.data.train if train_docs is None else train_docs
    dv = ctx.data.dev if dev_docs is None else dev_docs
    print(f"\n######## {name}: arch={cfg['ARCH']} method={cfg['METHOD']} seq={cfg['SEQ']} crf={cfg['USE_CRF']} shift={cfg['SHIFT_LAMBDA']} nb={cfg['NEIGHBORS']} kd={teacher is not None} | train docs={len(tr)} dev docs={len(dv)} ########")
    model = build_rr_model(ctx, cfg)
    out = train_rr(ctx, model, tr, dv, cfg, teacher, tcfg)
    tot, trn = count_params(model)
    res = None
    if dv:
        t0 = time.time()
        probs = collect_probs(ctx, model, dv, cfg)
        inf = time.time() - t0
        res, g, p = eval_by_domain(ctx, dv, probs)
        if show:
            print_scores(res["ALL"])
            confusions(g, p)
    ckpt = os.path.join(ctx.work, f"{name}.pt")
    torch.save(dict(trainable=out["state"], cfg=cfg, best_epoch=out["best_epoch"], labels=ctx.data.labels), ckpt)
    nan = float("nan")
    rec = dict(name=name, arch=cfg["ARCH"], method=cfg["METHOD"], seq=cfg["SEQ"], crf=cfg["USE_CRF"], shift=cfg["SHIFT_LAMBDA"], nb=cfg["NEIGHBORS"], kd=teacher is not None,
               kd_from=kd_from or "", best_epoch=out["best_epoch"], f1=res["ALL"]["macro_f1"] if res else nan,
               f1_CL=res.get("CL", {}).get("macro_f1", nan) if res else nan, f1_IT=res.get("IT", {}).get("macro_f1", nan) if res else nan,
               params_M=tot / 1e6, trainable_M=trn / 1e6, train_min=out["seconds"] / 60, peak_mem_MB=out["peak_mb"],
               sent_per_s=sum(len(d["sents"]) for d in dv) / inf if dv else nan, ckpt=ckpt, history=out["history"])
    ctx.results.append(rec)
    json.dump(ctx.results, open(os.path.join(ctx.work, "results.json"), "w"), indent=1)
    del model
    torch.cuda.empty_cache()
    return rec


def load_rr(ctx, path):
    """Rebuild a model from a checkpoint written by :func:`run_rr`.

    Args:
        ctx (Context): Pipeline context.
        path (str): Checkpoint path.

    Returns:
        Tuple[RRModel, dict]: ``(model in eval mode on the device, config)``.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {**ctx.cfg, **ck["cfg"]}
    assert ck.get("labels", ctx.data.labels) == ctx.data.labels, "label order differs from the checkpoint"
    model = build_rr_model(ctx, cfg)
    own = {n for n, _ in model.named_parameters()}
    bad = [n for n in ck["trainable"] if n not in own]
    if bad:
        raise RuntimeError(f"{path}: {len(bad)} saved tensors missing in rebuilt model, e.g. {bad[:3]}")
    model.load_state_dict(ck["trainable"], strict=False)
    return model.to(ctx.device).eval(), cfg


def run_final_rr(ctx, name, ckpt_path, teacher=None, tcfg=None, **ov):
    """Retrain a validated configuration on TRAIN+DEV (no validation) for its best epoch count.

    Args:
        ctx (Context): Pipeline context.
        name (str): New run name.
        ckpt_path (str): Checkpoint of the validated run (provides config and best epoch).
        teacher (Optional[Union[RRModel, list]]): Teacher(s) for KD students.
        tcfg (Optional[dict]): Config of a single teacher.
        **ov: Config overrides (e.g. ``SEED``) and ``kd_from``.

    Returns:
        dict: Result record.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg0 = dict(ck["cfg"])
    cfg0["EPOCHS"] = ck.get("best_epoch", cfg0["EPOCHS"])
    return run_rr(ctx, name, teacher=teacher, tcfg=tcfg, train_docs=ctx.data.train + ctx.data.dev, dev_docs=[], **{**cfg0, **ov})


# ---------------------------------------------------------------------------------------------- results registry
def show_results(ctx):
    """Print the results table of all runs.

    Args:
        ctx (Context): Pipeline context.

    Returns:
        None
    """
    import pandas as pd
    cols = ["name", "arch", "method", "seq", "crf", "shift", "nb", "kd", "kd_from", "best_epoch", "f1", "f1_CL", "f1_IT", "params_M", "trainable_M", "train_min", "peak_mem_MB", "sent_per_s"]
    df = pd.DataFrame(ctx.results)
    for c_ in cols:
        if c_ not in df:
            df[c_] = ""
    print(df[cols].round(4).to_string(index=False))


def rec_from_ckpt(ctx, p):
    """Re-evaluate a saved checkpoint on dev and build a result record.

    Args:
        ctx (Context): Pipeline context.
        p (str): Checkpoint path.

    Returns:
        dict: Result record (timing fields are NaN).
    """
    name = os.path.basename(p)[:-3]
    m, c = load_rr(ctx, p)
    res, _, _ = eval_by_domain(ctx, ctx.data.dev, collect_probs(ctx, m, ctx.data.dev, c))
    tot, trn = count_params(m)
    nan = float("nan")
    rec = dict(name=name, arch=c["ARCH"], method=c["METHOD"], seq=c["SEQ"], crf=c["USE_CRF"], shift=c["SHIFT_LAMBDA"], nb=c["NEIGHBORS"], kd=("_kd" in name and "nokd" not in name),
               kd_from="", best_epoch=-1, f1=res["ALL"]["macro_f1"], f1_CL=res["CL"]["macro_f1"], f1_IT=res["IT"]["macro_f1"], params_M=tot / 1e6, trainable_M=trn / 1e6,
               train_min=nan, peak_mem_MB=nan, sent_per_s=nan, ckpt=p, history=[])
    del m
    torch.cuda.empty_cache()
    return rec


def run_once(ctx, name, **kw):
    """Run an experiment unless its checkpoint already exists (then only register it in the results).

    Args:
        ctx (Context): Pipeline context (``settings.force_retrain``).
        name (str): Run name.
        **kw: Passed to :func:`run_rr`.

    Returns:
        Optional[dict]: The new record, or None when an existing checkpoint was reused.
    """
    p = os.path.join(ctx.work, name + ".pt")
    if not ctx.settings.force_retrain and os.path.exists(p):
        print(f"[skip] {name}.pt exists")
        if next((x for x in ctx.results if x["name"] == name), None) is None:
            ctx.results.append(rec_from_ckpt(ctx, p))
        return None
    return run_rr(ctx, name, **kw)


def sync_results(ctx):
    """Add every checkpoint in the work folder that is not in the results yet (``FINAL_*`` retrains are skipped: they saw dev data).

    Args:
        ctx (Context): Pipeline context.

    Returns:
        None
    """
    have = {r["name"] for r in ctx.results}
    for p in sorted(glob.glob(os.path.join(ctx.work, "*.pt"))):
        nm = os.path.basename(p)[:-3]
        if nm in have or nm.startswith("FINAL"):
            continue
        try:
            ctx.results.append(rec_from_ckpt(ctx, p))
            print("[sync] loaded", nm)
        except Exception as e:
            print(f"[sync] skipped {nm}: {str(e)[:100]}")


def get_window(ctx, k, include_kd=False):
    """The best ``k`` runs by dev macro-F1 (latest record per name, ``FINAL_*`` excluded).

    Args:
        ctx (Context): Pipeline context.
        k (int): Window size.
        include_kd (bool): Allow distilled students.

    Returns:
        List[dict]: Records sorted by F1 (best first).
    """
    sync_results(ctx)
    d = {}
    for r in ctx.results:
        if r["f1"] == r["f1"] and (include_kd or not r["kd"]) and not r["name"].startswith("FINAL"):
            d[r["name"]] = r
    return sorted(d.values(), key=lambda r: -r["f1"])[:k]
