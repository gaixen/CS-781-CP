"""Ensembles of finished experiments (log-probability averaging) and the final student-distillation + seed ensembles."""
import gc
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from summ.config import ENS_RUN, doc_budget
from summ.data.inputs import collate, inputs, make_batches
from summ.eval.metrics import evaluate, postprocess
from summ.models.summ_models import (autocast, block_fwd, build_model, decode_steps, summ_mask)
from summ.training.trainer import exp_dir, gen_texts, load_best, pad_docs, write_submission
from summ.utils import set_seed


@torch.no_grad()
def ens_generate(ctx, members, bs=4, gen=None, weights=None):
    """Greedy decoding from an ensemble: weighted average of log-probabilities; every member reads ITS OWN input tokens.

    Members may mix architectures (custom decoders, GPT-2, T5) and context budgets.

    Args:
        ctx (Context): Context.
        members (List[Tuple[nn.Module, list]]): ``(model, doc_ids of that model)`` pairs.
        bs (int): Batch size.
        gen (Optional[dict]): Generation overrides.
        weights (Optional[Sequence[float]]): Member weights.

    Returns:
        List[str]: Summaries in input order.
    """
    g = {**ctx.s.gen, **(gen or {})}
    g["max_new"] = min(g["max_new"], ctx.s.sum_tok)
    n = len(members[0][1])
    assert all(len(x) == n for _, x in members)
    for m, _ in members:
        m.eval()
    order = np.argsort([len(x) for x in members[0][1]])
    res = [None] * n
    for k in range(0, n, bs):
        ix = order[k:k + bs]
        steps = []
        for m, ids in members:
            doc, dm = pad_docs(ctx, ids, ix)
            steps.append(m.begin(doc.to(ctx.device), dm.to(ctx.device)))
        for i, o in zip(ix, decode_steps(ctx, steps, len(ix), ctx.device, g["max_new"], g["min_new"], g["nrep"], g["rep_pen"], weights)):
            res[i] = postprocess(ctx.tok.decode(o, skip_special_tokens=True))
    return res


def finished(ctx):
    """Experiments with both ``result.json`` and ``best.pt``.

    Args:
        ctx (Context): Context.

    Returns:
        List[str]: Experiment ids.
    """
    return [e for e in ctx.experiments if os.path.exists(f"{exp_dir(ctx, e)}/result.json") and os.path.exists(f"{exp_dir(ctx, e)}/best.pt")]


def members_of(ctx, spec):
    """Resolve an ensemble spec: a list of ids or ``"auto:K"`` (the K best finished experiments by dev BERTScore, ROUGE-1 as fallback).

    Args:
        ctx (Context): Context.
        spec (Union[str, List[str]]): Ensemble spec.

    Returns:
        List[str]: Experiment ids.
    """
    if isinstance(spec, str) and spec.startswith("auto:"):
        k = int(spec.split(":")[1])
        sc = {e: json.load(open(f"{exp_dir(ctx, e)}/result.json")) for e in finished(ctx)}
        return sorted(sc, key=lambda e: -(sc[e].get("bertscore_f1") if sc[e].get("bertscore_f1") == sc[e].get("bertscore_f1") else sc[e]["rouge1_f"]))[:k]
    return list(spec)


def run_ensemble(ctx, name, eids, weights=None):
    """Evaluate an ensemble on the dev subset, generate test and write ``ENS-<name>_<members>_logprob-avg.zip``.

    Args:
        ctx (Context): Context.
        name (str): Ensemble name.
        eids (List[str]): Member experiments (all need ``best.pt``).
        weights (Optional[List[float]]): Relative member weights.

    Returns:
        Optional[dict]: Result (None if a member is not finished).
    """
    s = ctx.s
    d = f"{s.out}/ens_{name.replace('+', '_')}"
    os.makedirs(d, exist_ok=True)
    rj = f"{d}/result.json"
    if os.path.exists(rj) and not s.force:
        print(f"[ENS {name}] cached")
        return json.load(open(rj))
    miss = [e for e in eids if e not in finished(ctx)]
    if miss:
        print(f"[ENS {name}] skipped: not finished -> {miss}")
        return None
    print(f"\n===== ENSEMBLE {name}: {eids} weights={weights} =====")
    gc.collect()
    torch.cuda.empty_cache()
    n = s.dev_eval_n
    models = [load_best(ctx, build_model(ctx, ctx.experiments[e]), e).eval() for e in eids]
    for m in models:
        for p in m.parameters():
            p.requires_grad = False
    mem = lambda split: [(m, inputs(ctx, split, ctx.experiments[e]["strat"], doc_budget(s, ctx.experiments[e]))) for m, e in zip(models, eids)]
    t0 = time.time()
    dp = ens_generate(ctx, [(m, x[:n]) for m, x in mem("dev")], bs=s.ens_bs, weights=weights)
    inf_s = (time.time() - t0) / max(n, 1)
    met = evaluate(dp, ctx.dev["ref"][:n], ctx.dev["doc"][:n], ctx.device)
    json.dump(dict(preds=dp), open(f"{d}/dev_preds.json", "w"))
    print("  dev:", {k: round(v, 4) for k, v in met.items()})
    tp_ = ens_generate(ctx, mem("test"), bs=s.ens_bs, weights=weights)
    json.dump(tp_, open(f"{d}/test_preds.json", "w"))
    wtag = "" if not weights else "_w" + "-".join(f"{w:g}" for w in weights)
    zp = write_submission(ctx, f"ENS-{name}_{'+'.join(eids)}_logprob-avg{wtag}", tp_)
    res = dict(exp=f"ENS:{name}", members=eids, weights=weights, infer_s_per_doc=inf_s, zip=zp, **met)
    json.dump(res, open(rj, "w"), indent=1)
    print("  zip:", zp)
    del models
    gc.collect()
    torch.cuda.empty_cache()
    return res


def run_ensembles(ctx, weights_by_name=None):
    """Run every ensemble in ``settings.ens_select`` and print the comparison with the single models.

    Args:
        ctx (Context): Context.
        weights_by_name (Optional[dict]): ``name -> relative weights``.

    Returns:
        Dict[str, dict]: Ensemble results.
    """
    out = {}
    for nm in ctx.s.ens_select:
        try:
            out[nm] = run_ensemble(ctx, nm, members_of(ctx, ENS_RUN[nm]), (weights_by_name or {}).get(nm))
        except Exception as ex:
            print(f"[ENS {nm}] FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
    rows = [json.load(open(f"{exp_dir(ctx, e)}/result.json")) for e in finished(ctx)] + [r for r in out.values() if r]
    if rows:
        cmp_ = pd.DataFrame(rows).set_index("exp")[["bertscore_f1", "rouge1_f", "rougeL_f", "words", "repeat_trigram", "copy_bigram", "infer_s_per_doc"]]
        print(cmp_.round(4).to_string())
        cmp_.to_csv(f"{ctx.s.submit}/comparison_with_ensembles.csv")
    return out


# ------------------------------------------------------------------------------------------ student distillation
@torch.no_grad()
def ensemble_texts(ctx, models, doc_ids, bs=2):
    """Greedy decoding that averages next-token LOGITS of several custom-decoder models (used for the seed ensembles).

    Args:
        ctx (Context): Context (generation settings).
        models (List[SummLM]): Custom-decoder summarisers.
        doc_ids (list): Model inputs (shared by all members).
        bs (int): Batch size.

    Returns:
        List[str]: Summaries in input order.
    """
    assert len(models) >= 1
    for m in models:
        m.eval()
    g, EOS = ctx.s.gen, ctx.eos
    order = np.argsort([len(x) for x in doc_ids])
    results = [None] * len(doc_ids)
    for start in range(0, len(order), bs):
        ix = order[start:start + bs]
        doc, dm = pad_docs(ctx, doc_ids, ix)
        doc, dm = doc.to(ctx.device), dm.to(ctx.device)
        B, Td, dev = len(ix), doc.size(1), doc.device
        all_pasts = []
        with autocast(ctx):
            for model in models:
                m = model.lm
                assert Td + g["max_new"] <= m.rope_cos.shape[0], "Generation exceeds the RoPE cache."
                x = m.tok_emb(doc)
                mask = summ_mask(Td, 0, model.window, model.bidir, dm, dev)
                cos, sin = m.rope_cos[:Td], m.rope_sin[:Td]
                pasts = []
                for blk in m.blocks:
                    x, pk = block_fwd(blk, x, cos, sin, mask)
                    pasts.append(pk)
                all_pasts.append(pasts)
            tok_next = torch.full((B,), EOS, device=dev, dtype=torch.long)
            outputs, done, ngrams_seen, keymask = [[] for _ in range(B)], [False] * B, [dict() for _ in range(B)], dm.bool()
            for step in range(g["max_new"]):
                keymask = torch.cat([keymask, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
                combined = None
                for mi, model in enumerate(models):
                    m = model.lm
                    x = m.tok_emb(tok_next[:, None])
                    cos, sin = m.rope_cos[Td + step:Td + step + 1], m.rope_sin[Td + step:Td + step + 1]
                    for li, blk in enumerate(m.blocks):
                        x, all_pasts[mi][li] = block_fwd(blk, x, cos, sin, keymask[:, None, None, :], all_pasts[mi][li])
                    lg = m.lm_head(m.norm_f(x))[:, 0].float()
                    combined = lg if combined is None else combined + lg
                logits = combined / len(models)
                if step < g["min_new"]:
                    logits[:, EOS] = -1e9
                for b in range(B):
                    if done[b]:
                        continue
                    if g["rep_pen"] != 1.0 and outputs[b]:
                        ids = torch.tensor(list(set(outputs[b])), device=dev)
                        vals = logits[b, ids]
                        logits[b, ids] = torch.where(vals > 0, vals / g["rep_pen"], vals * g["rep_pen"])
                    nr = g["nrep"]
                    if nr and len(outputs[b]) >= nr - 1:
                        banned = ngrams_seen[b].get(tuple(outputs[b][-(nr - 1):]) if nr > 1 else ())
                        if banned:
                            logits[b, list(banned)] = -1e9
                next_ids = logits.argmax(-1).tolist()
                for b, t in enumerate(next_ids):
                    if done[b]:
                        continue
                    if t == EOS:
                        done[b] = True
                        continue
                    outputs[b].append(t)
                    nr = g["nrep"]
                    if nr and len(outputs[b]) >= nr:
                        ngrams_seen[b].setdefault(tuple(outputs[b][-nr:-1]), set()).add(t)
                if all(done):
                    break
                tok_next = torch.tensor([EOS if done[b] else next_ids[b] for b in range(B)], device=dev)
        for i, o in zip(ix, outputs):
            results[i] = postprocess(ctx.tok.decode(o, skip_special_tokens=True))
    return results


def distill_students(ctx):
    """Final stage: distil the E1+E2 logit-ensemble into students (unpruned = full FT, pruned = every 2nd layer) trained on TRAIN+DEV with several seeds, then write
    the E1+E2 logit-ensemble zip and a seed-ensemble zip per student variant.

    There is NO held-out set in this stage (dev is used for training).

    Args:
        ctx (Context): Context (``s.kd_*``).

    Returns:
        Dict[str, str]: Zip paths (``E1+E2``, ``unpruned``, ``pruned``).
    """
    s = ctx.s
    kd_dir = f"{s.out}/final_distillation"
    os.makedirs(kd_dir, exist_ok=True)
    for e in ("E1", "E2"):
        assert os.path.exists(f"{exp_dir(ctx, e)}/best.pt"), f"Missing {e}/best.pt. E1 and E2 must finish successfully first."
    st = s.strat
    print("\nLoading E1 and E2 teachers...")
    t1 = load_best(ctx, build_model(ctx, ctx.experiments["E1"]), "E1").eval()
    t2 = load_best(ctx, build_model(ctx, ctx.experiments["E2"]), "E2").eval()
    for p in list(t1.parameters()) + list(t2.parameters()):
        p.requires_grad = False
    zips = {}
    zips["E1+E2"] = write_submission(ctx, "E1_E2_logit_ensemble", ensemble_texts(ctx, [t1, t2], inputs(ctx, "test", st), bs=2))
    print("Experiment ensemble:", zips["E1+E2"])
    items = list(zip(inputs(ctx, "train", st), ctx.tr["sum_ids"])) + list(zip(inputs(ctx, "dev", st), ctx.dev["sum_ids"]))
    print(f"\nDistillation data: {len(ctx.tr['sum_ids'])} training + {len(ctx.dev['sum_ids'])} development = {len(items)} examples")
    variants = {"unpruned": dict(method="full", arch=s.student_arch), "pruned": dict(method="kd", arch=s.student_arch, keep_every=2)}
    paths = {v: [] for v in variants}
    for variant, cfg in variants.items():
        print(f"\n{'='*65}\nSTUDENT: {variant.upper()}\n{'='*65}")
        for seed in s.kd_seeds:
            set_seed(seed)
            rng = np.random.RandomState(seed)
            save = f"{kd_dir}/{variant}_seed{seed}.pt"
            paths[variant].append(save)
            if os.path.exists(save) and not s.kd_force:
                print(f"Using existing checkpoint: {save}")
                continue
            print(f"\nTraining {variant} student | seed={seed}")
            student = build_model(ctx, cfg)
            student.train()
            params = [p for p in student.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=s.kd_lr, weight_decay=0.01, betas=(0.9, 0.98))
            batches = make_batches(items, s.bs, rng)
            scaler = torch.amp.GradScaler("cuda", enabled=bool(s.amp and ctx.device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            run, t0 = 0.0, time.time()
            for i, batch in enumerate(batches):
                doc, dm, sin_, sm, lab = [x.to(ctx.device) for x in collate(ctx, batch)]
                lg, tg = student(doc, dm, sin_, sm, lab)
                with torch.no_grad():
                    tl = (t1(doc, dm, sin_, sm, lab)[0] + t2(doc, dm, sin_, sm, lab)[0]) / 2.0
                assert lg.shape == tl.shape, f"Vocabulary/logit mismatch: {lg.shape} vs {tl.shape}"
                T = s.kd_temp
                kl = F.kl_div(F.log_softmax(lg / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * (T * T)
                loss = (1 - s.kd_alpha) * F.cross_entropy(lg, tg) + s.kd_alpha * kl
                gs = (i // s.accum) * s.accum
                scaler.scale(loss / min(s.accum, len(batches) - gs)).backward()      # correct accumulation incl. the final incomplete group
                run += loss.item()
                if (i + 1) % s.accum == 0 or i + 1 == len(batches):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                if (i + 1) % 100 == 0:
                    print(f"  step {i+1}/{len(batches)} loss={run/(i+1):.4f}")
            torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()}, save)
            print(f"Saved {variant} seed={seed} | loss={run/max(len(batches),1):.4f} | time={(time.time()-t0)/60:.1f} min")
            del student, opt, scaler
            gc.collect()
            torch.cuda.empty_cache()
    del t1, t2
    gc.collect()
    torch.cuda.empty_cache()
    for variant, cfg in variants.items():
        print(f"\n{'='*65}\nSEED ENSEMBLE: {variant.upper()}\n{'='*65}")
        seed_models = []
        for seed, path in zip(s.kd_seeds, paths[variant]):
            m = build_model(ctx, cfg)
            m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
            m.eval()
            seed_models.append(m)
            print(f"Loaded seed {seed}")
        zips[variant] = write_submission(ctx, f"student_{variant}_3seed_ensemble", ensemble_texts(ctx, seed_models, inputs(ctx, "test", st), bs=2))
        print("Seed ensemble:", zips[variant])
        del seed_models
        gc.collect()
        torch.cuda.empty_cache()
    print("\nALL ENSEMBLES COMPLETED:", zips)
    return zips
