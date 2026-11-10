"""Training (CE or logit-KD), generation, submission writing and the experiment runner."""
import gc
import json
import math
import os
import time
import traceback
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from summ.config import doc_budget
from summ.data.inputs import build_input, collate, inputs, make_batches
from summ.eval.metrics import evaluate, postprocess
from summ.models.summ_models import build_model, n_params
from summ.utils import set_seed


def exp_dir(ctx, eid):
    """Run folder of an experiment (created on demand).

    Args:
        ctx (Context): Context.
        eid (str): Experiment id.

    Returns:
        str: ``<out>/<eid>_<name>_<arch>``.
    """
    c = ctx.experiments[eid]
    d = f"{ctx.s.out}/{eid}_{c['name']}_{c['arch']}"
    os.makedirs(d, exist_ok=True)
    return d


@torch.no_grad()
def dev_loss(ctx, model, items):
    """Mean summary-token cross-entropy on dev items.

    Args:
        ctx (Context): Context.
        model (nn.Module): Summariser.
        items (list): ``(document ids, summary ids)`` pairs.

    Returns:
        float: Mean loss per supervised token.
    """
    model.eval()
    tot, n = 0.0, 0
    for k in range(0, len(items), 8):
        doc, dm, sin_, sm, lab = [x.to(ctx.device) for x in collate(ctx, items[k:k + 8])]
        lg, tg = model(doc, dm, sin_, sm, lab)
        tot += F.cross_entropy(lg, tg, reduction="sum").item()
        n += len(tg)
    return tot / max(n, 1)


def train_exp(ctx, eid, cfg, model, teacher=None):
    """Train one experiment (AdamW, warm-up + linear decay, AMP, accumulation) keeping the best epoch by dev loss in ``best.pt``.

    Optional logit KD: ``(1 - a) * CE + a * T^2 * KL(teacher || student)``. Stops after 2 non-improving epochs or at the wall-clock cap.

    Args:
        ctx (Context): Context.
        eid (str): Experiment id.
        cfg (dict): Experiment definition.
        model (nn.Module): Model to train.
        teacher (Optional[nn.Module]): Frozen teacher for KD.

    Returns:
        Tuple[List[dict], float]: ``(per-epoch history, training seconds)``.
    """
    s, dev = ctx.s, ctx.device
    d = exp_dir(ctx, eid)
    set_seed(s.seed)
    rng = np.random.RandomState(s.seed)
    bud = doc_budget(s, cfg)
    tr = list(zip(inputs(ctx, "train", cfg["strat"], bud), ctx.tr["sum_ids"]))
    dv = list(zip(inputs(ctx, "dev", cfg["strat"], bud)[:s.dev_eval_n * 2], ctx.dev["sum_ids"][:s.dev_eval_n * 2]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.98))
    spe = math.ceil(math.ceil(len(tr) / s.bs) / s.accum)
    total = spe * cfg["epochs"]
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: (st + 1) / warm if st < warm else max(0.0, (total - st) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(s.amp and dev.type == "cuda"))
    best, bad, hist, t0, step = 1e9, 0, [], time.time(), 0
    for ep in range(cfg["epochs"]):
        model.train()
        run, nb = 0.0, 0
        for i, bt in enumerate(make_batches(tr, s.bs, rng)):
            doc, dm, sin_, sm, lab = [x.to(dev) for x in collate(ctx, bt)]
            lg, tg = model(doc, dm, sin_, sm, lab)
            loss = F.cross_entropy(lg, tg)
            if teacher is not None:
                with torch.no_grad():
                    tl, _ = teacher(doc, dm, sin_, sm, lab)
                v, T = min(lg.size(-1), tl.size(-1)), cfg["kd_temp"]
                kl = F.kl_div(F.log_softmax(lg[:, :v] / T, -1), F.softmax(tl[:, :v] / T, -1), reduction="batchmean") * T * T
                loss = (1 - cfg["kd_alpha"]) * loss + cfg["kd_alpha"] * kl
            scaler.scale(loss / s.accum).backward()
            run += loss.item()
            nb += 1
            if (i + 1) % s.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 20 == 0:
                    print(f"  ep{ep+1} step {step}/{total} loss {run/nb:.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/60:.1f}min")
            if (time.time() - t0) / 60 > s.max_min_per_exp:
                print("  wall-clock cap hit")
                break
        vl = dev_loss(ctx, model, dv)
        hist.append(dict(epoch=ep + 1, train_loss=run / max(nb, 1), dev_loss=vl, dev_ppl=math.exp(min(vl, 20)), min=(time.time() - t0) / 60))
        print("  ", hist[-1])
        if vl < best:
            best, bad = vl, 0
            peft = any(not p.requires_grad for p in model.parameters())
            torch.save({n: p.detach().cpu() for n, p in (model.named_parameters() if peft else model.state_dict().items()) if (not isinstance(p, nn.Parameter)) or p.requires_grad}, f"{d}/best.pt")
        else:
            bad += 1
        if (time.time() - t0) / 60 > s.max_min_per_exp or bad >= 2:
            break
    return hist, time.time() - t0


def load_best(ctx, model, eid):
    """Load the best trainable weights of an experiment into ``model``.

    Args:
        ctx (Context): Context.
        model (nn.Module): Model built for the experiment.
        eid (str): Experiment id.

    Returns:
        nn.Module: The same model.
    """
    model.load_state_dict(torch.load(f"{exp_dir(ctx, eid)}/best.pt", map_location="cpu"), strict=False)
    return model


def pad_docs(ctx, doc_ids, ix):
    """Left-pad the documents ``ix`` into id / mask tensors.

    Args:
        ctx (Context): Context (``eos``).
        doc_ids (list): Token arrays.
        ix (Sequence[int]): Document indices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(doc, mask)`` on the CPU.
    """
    Td = max(len(doc_ids[i]) for i in ix)
    doc, dm = torch.full((len(ix), Td), ctx.eos, dtype=torch.long), torch.zeros(len(ix), Td, dtype=torch.long)
    for b, i in enumerate(ix):
        x = torch.as_tensor(np.asarray(doc_ids[i], np.int64))
        doc[b, Td - len(x):] = x
        dm[b, Td - len(x):] = 1
    return doc, dm


def gen_texts(ctx, model, doc_ids, gen=None, bs=None):
    """Generate summaries for tokenised documents (length-sorted batches, post-processed).

    Args:
        ctx (Context): Context (``s.gen``, ``s.gen_bs``).
        model (nn.Module): Summariser.
        doc_ids (list): Model inputs.
        gen (Optional[dict]): Overrides of the generation settings.
        bs (Optional[int]): Batch size.

    Returns:
        List[str]: Summaries in input order.
    """
    g = {**ctx.s.gen, **(gen or {})}
    bs = bs or ctx.s.gen_bs
    order = np.argsort([len(x) for x in doc_ids])
    res = [None] * len(doc_ids)
    for k in range(0, len(order), bs):
        ix = order[k:k + bs]
        doc, dm = pad_docs(ctx, doc_ids, ix)
        for i, o in zip(ix, model.generate(doc.to(ctx.device), dm.to(ctx.device), **g)):
            res[i] = postprocess(ctx.tok.decode(o, skip_special_tokens=True))
    return res


def hier_inputs(ctx, model, doc_ids, chunk=1000, ov=100, ch_gen=None):
    """Hierarchical inference inputs: summarise each chunk of long documents, concatenate the chunk summaries (they become the new input).

    Documents that fit in ``doc_tok`` are left unchanged.

    Args:
        ctx (Context): Context.
        model (nn.Module): Summariser.
        doc_ids (list): Tokenised documents.
        chunk (int): Chunk length in tokens.
        ov (int): Chunk overlap.
        ch_gen (Optional[dict]): Generation settings of the chunk summaries.

    Returns:
        List[np.ndarray]: New inputs.
    """
    ch_gen = ch_gen or dict(max_new=96, min_new=24)
    pieces, own = [], []
    for k, ids in enumerate(doc_ids):
        if len(ids) <= ctx.s.doc_tok:
            continue
        for st in range(0, len(ids), chunk - ov):
            pieces.append(ids[st:st + chunk])
            own.append(k)
    cs = gen_texts(ctx, model, pieces, ch_gen) if pieces else []
    joined = defaultdict(list)
    for k, t in zip(own, cs):
        joined[k].append(t)
    return [np.asarray(ctx.tok(" ".join(joined[k]), add_special_tokens=False)["input_ids"] or [ctx.eos], np.int32)[:ctx.s.doc_tok] if k in joined else ids for k, ids in enumerate(doc_ids)]


def write_submission(ctx, tag, texts, ids=None):
    """Write ``<submit>/<tag>.zip`` containing ``prediction.json`` = ``[{"id", "summary"}]`` (empty summaries get a fallback sentence).

    Args:
        ctx (Context): Context (``te['rid']``, ``s.submit``).
        tag (str): Zip name without extension.
        texts (List[str]): Summaries in test order.
        ids (Optional[list]): Test ids (default ``te['rid']``).

    Returns:
        str: Zip path.
    """
    ids = ids or ctx.te["rid"]
    assert len(texts) == len(ids) == len(set(map(str, ids)))
    texts = [t if t.strip() else "The court considered the facts and arguments and decided the matter." for t in texts]
    d = f"{ctx.s.submit}/{tag}"
    os.makedirs(d, exist_ok=True)
    json.dump([{"id": i, "summary": t} for i, t in zip(ids, texts)], open(f"{d}/prediction.json", "w"), ensure_ascii=False)
    zp = f"{ctx.s.submit}/{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{d}/prediction.json", "prediction.json")
    return zp


def run_experiment(ctx, eid):
    """Run (or load from cache) one experiment: train, evaluate on the dev subset, generate test, write the zip and ``result.json``.

    Args:
        ctx (Context): Context.
        eid (str): Experiment id.

    Returns:
        dict: Result (metrics, parameter counts, time, memory, history, zip path).
    """
    s = ctx.s
    cfg, d = ctx.experiments[eid], exp_dir(ctx, eid)
    rj = f"{d}/result.json"
    if os.path.exists(rj) and not s.force:
        print(f"[{eid}] cached")
        return json.load(open(rj))
    print(f"\n===== {eid} {cfg['name']} | {cfg['arch']} | {cfg['method']} =====")
    gc.collect()
    torch.cuda.empty_cache()
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = build_model(ctx, cfg)
    tp, ap = n_params(model)
    print(f"  trainable {tp/1e6:.2f}M / total {ap/1e6:.2f}M")
    teacher = None
    if cfg["method"] == "kd":
        teacher = load_best(ctx, build_model(ctx, ctx.experiments[cfg["teacher"]]), cfg["teacher"]).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  teacher {cfg['teacher']}: {n_params(teacher)[1]/1e6:.1f}M params -> student {ap/1e6:.1f}M ({100*ap/n_params(teacher)[1]:.0f}%)")
    hist, tsec = train_exp(ctx, eid, cfg, model, teacher)
    del teacher
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if ctx.device.type == "cuda" else 0.0
    load_best(ctx, model, eid)
    n, bud = s.dev_eval_n, doc_budget(s, cfg)
    t0 = time.time()
    dp = gen_texts(ctx, model, inputs(ctx, "dev", cfg["strat"], bud)[:n])
    inf_s = (time.time() - t0) / max(n, 1)
    m = evaluate(dp, ctx.dev["ref"][:n], ctx.dev["doc"][:n], ctx.device)
    json.dump(dict(preds=dp), open(f"{d}/dev_preds.json", "w"))
    tp_ = gen_texts(ctx, model, inputs(ctx, "test", cfg["strat"], bud))
    zp = write_submission(ctx, f"{eid}-{cfg['name']}_{cfg['arch']}_{cfg['strat']}", tp_)
    json.dump(tp_, open(f"{d}/test_preds.json", "w"))
    res = dict(exp=eid, name=cfg["name"], method=cfg["method"], arch=cfg["arch"], strat=cfg["strat"], trainable_M=tp / 1e6, total_M=ap / 1e6, fp16_MB=ap * 2 / 2 ** 20, train_min=tsec / 60,
               peak_gpu_GB=peak, infer_s_per_doc=inf_s, best_dev_loss=min(h["dev_loss"] for h in hist), hist=hist, zip=zp, **m)
    json.dump(res, open(rj, "w"), indent=1)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "hist"})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def run_all(ctx, run_ids):
    """Run several experiments; those without a Phase-1 checkpoint are skipped and failures are reported.

    Args:
        ctx (Context): Context (``ctx.results`` is updated).
        run_ids (Iterable[str]): Experiment ids.

    Returns:
        Dict[str, dict]: Results of the experiments that finished.
    """
    for eid in run_ids:
        if not ctx.s.have(ctx.experiments[eid]["arch"]):
            print(f"[{eid}] skipped: no Phase-1 checkpoint for {ctx.experiments[eid]['arch']}")
            continue
        try:
            ctx.results[eid] = run_experiment(ctx, eid)
        except Exception as ex:
            print(f"[{eid}] FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
    return ctx.results
