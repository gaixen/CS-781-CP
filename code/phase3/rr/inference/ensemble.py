"""Ensembling of trained runs, per-class bias tuning (F1- and recall-oriented), final train+dev retraining and submissions.

Dev has very few documents, so every tuned component (ensemble weights, class biases) is only used if a **cross-fit** estimate
(tune on one half of the dev documents, score the other half) improves over the untuned baseline.
"""
import itertools
import json
import os
import zipfile
from collections import Counter

import numpy as np
import torch

from rr.eval.metrics import fast_macro, prf
from rr.training.trainer import collect_probs, eval_by_domain, get_window, load_rr, run_final_rr, show_results
from rr.eval.metrics import confusions, print_scores


# ---------------------------------------------------------------------------------------------- basics
def ens_probs(ctx, ckpts, docs, weights=None):
    """Weighted mean of the posteriors of several checkpoints.

    Args:
        ctx (Context): Pipeline context.
        ckpts (List[str]): Checkpoint paths.
        docs (List[dict]): Documents to score.
        weights (Optional[List[float]]): Member weights (default equal).

    Returns:
        List[np.ndarray]: Posteriors per document (n_sentences, n_labels).
    """
    weights = weights or [1.0] * len(ckpts)
    acc = None
    for w, path in zip(weights, ckpts):
        m, c = load_rr(ctx, path)
        pr = collect_probs(ctx, m, docs, c)
        acc = [w * p for p in pr] if acc is None else [a + w * p for a, p in zip(acc, pr)]
        del m
        torch.cuda.empty_cache()
    return [a / sum(weights) for a in acc]


def tune_bias_obj(P, y, obj, nl, grid=np.linspace(-1.5, 1.5, 13), rounds=2):
    """Per-class additive log-bias maximising an objective (coordinate ascent).

    Args:
        P (np.ndarray): Posteriors (n, nl).
        y (np.ndarray): Gold ids.
        obj (Callable[[np.ndarray, np.ndarray], float]): ``obj(y, pred)`` to maximise.
        nl (int): Number of classes.
        grid (np.ndarray): Candidate bias values.
        rounds (int): Coordinate-ascent rounds.

    Returns:
        np.ndarray: Bias vector (nl,).
    """
    L = np.log(P + 1e-9)
    b = np.zeros(nl)
    for _ in range(rounds):
        for c in range(nl):
            best = (obj(y, (L + b).argmax(1)), b[c])
            for g in grid:
                bb = b.copy()
                bb[c] = g
                s = obj(y, (L + bb).argmax(1))
                if s > best[0] + 1e-9:
                    best = (s, g)
            b[c] = best[1]
    return b


def tune_bias(P, y, nl, grid=np.linspace(-1.5, 1.5, 13), rounds=2):
    """Per-class log-bias that maximises macro-F1.

    Args:
        P (np.ndarray): Posteriors.
        y (np.ndarray): Gold ids.
        nl (int): Number of classes.
        grid (np.ndarray): Candidate values.
        rounds (int): Rounds.

    Returns:
        np.ndarray: Bias (nl,).
    """
    return tune_bias_obj(P, y, lambda yy, pp: fast_macro(yy, pp, nl), nl, grid, rounds)


def crossfit_bias(nl, docs, probs):
    """Honest macro-F1 of bias tuning: tune on half of the dev DOCUMENTS, apply to the other half (both ways).

    Args:
        nl (int): Number of classes.
        docs (List[dict]): Dev documents.
        probs (List[np.ndarray]): Posteriors per document.

    Returns:
        float: Cross-fit macro-F1.
    """
    idx = np.arange(len(docs))
    halves = [idx[0::2], idx[1::2]]
    ys, ps = [], []
    for a, b in ((0, 1), (1, 0)):
        Pa = np.concatenate([probs[i] for i in halves[a]])
        ya = np.concatenate([docs[i]["y"] for i in halves[a]])
        bias = tune_bias(Pa, ya, nl)
        Pb = np.concatenate([probs[i] for i in halves[b]])
        ys.append(np.concatenate([docs[i]["y"] for i in halves[b]]))
        ps.append((np.log(Pb + 1e-9) + bias).argmax(1))
    return fast_macro(np.concatenate(ys), np.concatenate(ps), nl)


def write_submission(ctx, preds, name):
    """Validate and write ``<work>/<name>/prediction.json`` and ``submission.zip``.

    Args:
        ctx (Context): Pipeline context.
        preds (Dict[str, List[str]]): Predicted label names per test document id.
        name (str): Submission folder name.

    Returns:
        str: Path of the zip.
    """
    data = ctx.data
    ids = {d["id"] for d in data.test}
    assert set(preds) == ids, f"id mismatch: missing {len(ids - set(preds))}, unknown {len(set(preds) - ids)}"
    for d in data.test:
        assert len(preds[d["id"]]) == len(d["sents"]) and all(l in data.l2i for l in preds[d["id"]]), f"bad prediction for doc {d['id']}"
    out = os.path.join(ctx.work, name)
    os.makedirs(out, exist_ok=True)
    pj = os.path.join(out, "prediction.json")
    json.dump(preds, open(pj, "w"), ensure_ascii=False)
    zp = os.path.join(out, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    cnt = Counter(l for v in preds.values() for l in v)
    tr = Counter(l for d in data.train + data.dev for l in d["labels"])
    a, b = sum(cnt.values()), sum(tr.values())
    print(f"[submission] {len(preds)} docs, {a} sentences -> {zp}\n[submission] predicted vs train label share: " + ", ".join(f"{l}:{100*cnt[l]/a:.1f}/{100*tr[l]/b:.1f}" for l in data.labels))
    return zp


def predict_test_ens(ctx, ckpts, name, weights=None, bias=None):
    """Ensemble posteriors on TEST, argmax (optionally with a class bias) and write a submission.

    Args:
        ctx (Context): Pipeline context.
        ckpts (List[str]): Checkpoint paths.
        name (str): Submission name.
        weights (Optional[List[float]]): Member weights.
        bias (Optional[np.ndarray]): Per-class log-bias.

    Returns:
        str: Zip path.
    """
    pr = ens_probs(ctx, ckpts, ctx.data.test, weights)
    labels = ctx.data.labels
    preds = {d["id"]: [labels[i] for i in (np.log(p + 1e-9) + (bias if bias is not None else 0)).argmax(1)] for d, p in zip(ctx.data.test, pr)}
    return write_submission(ctx, preds, name)


# ---------------------------------------------------------------------------------------------- analysis: best subset ensemble
def analyse_and_ensemble(ctx):
    """Analyse the best single model, evaluate EVERY subset of the best-K window as an ensemble, tune a class bias (cross-fit gated) and write ``sub_dev_ensemble``.

    Stores ``best_names``, ``best_set``, ``bias`` and ``ens_p`` in ``ctx.state``.

    Args:
        ctx (Context): Pipeline context (``settings.ens_window_k``, ``settings.ens_include_kd``).

    Returns:
        None
    """
    s, data = ctx.settings, ctx.data
    nl, dev = data.nl, data.dev
    win = get_window(ctx, s.ens_window_k, include_kd=s.ens_include_kd)
    show_results(ctx)
    if not win:
        print("no runs yet")
        return
    best = win[0]
    print("\nbest single model:", best["name"])
    m, c = load_rr(ctx, best["ckpt"])
    pr = collect_probs(ctx, m, dev, c)
    res, g, p = eval_by_domain(ctx, dev, pr)
    for k, v in res.items():
        print(f"  {k}: macro-F1 {v['macro_f1']:.4f} | acc {v['acc']:.4f}")
    print_scores(res["ALL"])
    confusions(g, p, 12)
    del m
    DP = {}
    for r in win:
        m, c = load_rr(ctx, r["ckpt"])
        DP[r["name"]] = collect_probs(ctx, m, dev, c)
        del m
        torch.cuda.empty_cache()
    lens = [len(d["y"]) for d in dev]
    Y = np.concatenate([d["y"] for d in dev])
    PM = {n: np.concatenate(v) for n, v in DP.items()}
    names = [r["name"] for r in win]
    combos = []
    for k in range(1, len(names) + 1):
        for sub in itertools.combinations(names, k):
            combos.append((fast_macro(Y, np.mean([PM[n] for n in sub], 0).argmax(1), nl), sub))
    combos.sort(key=lambda x: (-round(x[0], 4), -len(x[1])))
    print("\nsingles:", {n: round(fast_macro(Y, PM[n].argmax(1), nl), 4) for n in names})
    print(f"full window ensemble ({len(names)}): {fast_macro(Y, np.mean([PM[n] for n in names], 0).argmax(1), nl):.4f}")
    print(f"top ensembles out of {len(combos)} subsets (dev has only {len(dev)} docs -> differences of ~0.01 are noise; prefer larger ensembles):")
    for f, sub in combos[:8]:
        print(f"  {f:.4f}  {' + '.join(sub)}")
    _, best_names = combos[0]
    best_set = [next(r["ckpt"] for r in win if r["name"] == n) for n in best_names]
    print("-> using:", best_names)
    Pm = np.mean([PM[n] for n in best_names], 0)
    ens_p = np.split(Pm, np.cumsum(lens)[:-1])
    base = fast_macro(Y, Pm.argmax(1), nl)
    cf = crossfit_bias(nl, dev, ens_p)
    bias = tune_bias(Pm, Y, nl) if cf > base + 0.003 else None
    print(f"class-bias: dev macro-F1 {base:.4f} -> cross-fit {cf:.4f} -> {'USED' if bias is not None else 'not used'}", np.round(bias, 2) if bias is not None else "")
    ctx.state.update(best_names=list(best_names), best_set=best_set, bias=bias, ens_p=ens_p)
    predict_test_ens(ctx, best_set, "sub_dev_ensemble", bias=bias)


# ---------------------------------------------------------------------------------------------- CELL 1: tune the combiner
def tune_weights(PM, y, f1_of, grid=(0.0, 0.5, 1.0, 2.0), rounds=2):
    """Coordinate-ascent ensemble weights maximising macro-F1.

    Args:
        PM (List[np.ndarray]): Member posteriors (each (n, nl)).
        y (np.ndarray): Gold ids.
        f1_of (Callable[[np.ndarray, np.ndarray], float]): ``f1_of(y, pred)``.
        grid (Tuple[float, ...]): Candidate weights.
        rounds (int): Rounds.

    Returns:
        np.ndarray: Weights (len(PM),).
    """
    mix = lambda w: sum(wi * P for wi, P in zip(w, PM)) / np.sum(w)
    w = np.ones(len(PM))
    for _ in range(rounds):
        for i in range(len(PM)):
            best = (f1_of(y, mix(w).argmax(1)), w[i])
            for g in grid:
                ww = w.copy()
                ww[i] = g
                if ww.sum() == 0:
                    continue
                s = f1_of(y, mix(ww).argmax(1))
                if s > best[0] + 1e-9:
                    best = (s, g)
            w[i] = best[1]
    return w


def tune_ensemble(ctx, names=None):
    """Tune ensemble weights and two class-bias variants (F1-oriented and recall-oriented F-beta) on dev, each gated by a cross-fit estimate.

    Stores ``ens_ckpts``, ``ens_weights``, ``bias_f1`` and ``bias_rec`` (None when not accepted) in ``ctx.state``.

    Args:
        ctx (Context): Pipeline context.
        names (Optional[List[str]]): Member run names (default ``settings.ens_members``).

    Returns:
        None
    """
    s, data = ctx.settings, ctx.data
    nl = data.nl
    names = names or s.ens_members
    ck0 = [os.path.join(ctx.work, n + ".pt") for n in names]
    assert all(os.path.exists(p) for p in ck0), [p for p in ck0 if not os.path.exists(p)]

    def member_probs(path):
        """Dev posteriors (flat) of one checkpoint.

        Args:
            path (str): Checkpoint path.

        Returns:
            np.ndarray: (n_sentences, nl).
        """
        m, c = load_rr(ctx, path)
        pr = collect_probs(ctx, m, data.dev, c)
        del m
        torch.cuda.empty_cache()
        return np.concatenate(pr)

    PMS = [member_probs(p) for p in ck0]
    Y = np.concatenate([d["y"] for d in data.dev])
    DOC = np.concatenate([np.full(len(d["y"]), i) for i, d in enumerate(data.dev)])
    f1_of = lambda y, pred: prf(y, pred, nl)[2]
    fb_of = lambda y, pred: prf(y, pred, nl, s.recall_beta)[2]
    mix = lambda w, PM=None: sum(wi * P for wi, P in zip(np.asarray(w, float), PMS if PM is None else PM)) / np.sum(w)

    def crossfit(tune, apply, score):
        """Tune on one half of the dev documents, apply to the other half, both directions.

        Args:
            tune (Callable): ``tune(mask_A)`` -> fitted parameters.
            apply (Callable): ``apply(params, mask_B)`` -> predictions.
            score (Callable): ``score(y, pred)``.

        Returns:
            float: Honest score.
        """
        ys, ps = [], []
        for a, b in ((0, 1), (1, 0)):
            A = (DOC % 2 == a)
            B = ~A
            t = tune(A)
            ys.append(Y[B])
            ps.append(apply(t, B))
        return score(np.concatenate(ys), np.concatenate(ps))

    eq = np.ones(len(PMS))
    cf_w = crossfit(lambda A: tune_weights([P[A] for P in PMS], Y[A], f1_of), lambda w, B: mix(w, [P[B] for P in PMS]).argmax(1), f1_of)
    cf_eq = crossfit(lambda A: eq, lambda w, B: mix(w, [P[B] for P in PMS]).argmax(1), f1_of)
    W = tune_weights(PMS, Y, f1_of) if cf_w > cf_eq + 0.003 else eq
    print(f"weights: equal {cf_eq:.4f} vs tuned {cf_w:.4f} (cross-fit F1) -> {'TUNED ' + str(W) if W is not eq else 'equal weights'}")
    PE = mix(W)
    B_F1, B_REC = tune_bias_obj(PE, Y, f1_of, nl), tune_bias_obj(PE, Y, fb_of, nl)

    def cf_scores(obj):
        """Cross-fit P / R / F1 of a bias tuned with ``obj``.

        Args:
            obj (Callable): Objective used for tuning.

        Returns:
            Tuple[float, float, float]: Macro P, R, F1.
        """
        ys, ps = [], []
        for a in (0, 1):
            A = (DOC % 2 == a)
            B = ~A
            b = tune_bias_obj(PE[A], Y[A], obj, nl)
            ys.append(Y[B])
            ps.append((np.log(PE[B] + 1e-9) + b).argmax(1))
        return prf(np.concatenate(ys), np.concatenate(ps), nl)

    base = prf(Y, PE.argmax(1), nl)
    rows = [("no bias", base, base), ("F1 bias", prf(Y, (np.log(PE + 1e-9) + B_F1).argmax(1), nl), cf_scores(f1_of)),
            (f"RECALL bias (F{s.recall_beta:g})", prf(Y, (np.log(PE + 1e-9) + B_REC).argmax(1), nl), cf_scores(fb_of))]
    print(f"\n{'variant':22s} | dev in-sample  P / R / F1      | cross-fit (honest)  P / R / F1")
    for nm, a, b in rows:
        print(f"{nm:22s} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}        | {b[0]:.3f} / {b[1]:.3f} / {b[2]:.3f}")
    use_f1 = rows[1][2][2] > rows[0][2][2] + 0.003
    use_rec = rows[2][2][1] > rows[0][2][1] + 0.003
    print("F1-bias used:", use_f1, "| recall-bias used:", use_rec)
    ctx.state.update(ens_ckpts=ck0, ens_weights=list(W), bias_f1=B_F1 if use_f1 else None, bias_rec=B_REC if use_rec else None)
    write_two(ctx, ck0, "ens")


def write_two(ctx, ckpts, tag):
    """Write the F1-oriented and recall-oriented ensemble submissions (``sub_<tag>_f1`` / ``sub_<tag>_recall``).

    Args:
        ctx (Context): Pipeline context (needs ``ens_weights`` / ``bias_f1`` / ``bias_rec`` from :func:`tune_ensemble`).
        ckpts (List[str]): Checkpoints of the members (same order as the tuned weights).
        tag (str): Name tag.

    Returns:
        None
    """
    st = ctx.state
    pt = ens_probs(ctx, ckpts, ctx.data.test, st["ens_weights"])
    labels = ctx.data.labels
    for nm, bias in ((f"sub_{tag}_f1", st["bias_f1"] if st["bias_f1"] is not None else 0.0), (f"sub_{tag}_recall", st["bias_rec"] if st["bias_rec"] is not None else 0.0)):
        preds = {d["id"]: [labels[i] for i in (np.log(p + 1e-9) + bias).argmax(1)] for d, p in zip(ctx.data.test, pt)}
        print("->", nm)
        write_submission(ctx, preds, nm)


# ---------------------------------------------------------------------------------------------- CELL 2 / final fits
def teachers_of(ctx, n):
    """Teachers of a KD student run (``C_kd_from_ENS2`` needs ``settings.ens2_teachers``).

    Args:
        ctx (Context): Pipeline context.
        n (str): Run name.

    Returns:
        Optional[List[str]]: Teacher run names, or None for non-KD runs.
    """
    if n == "C_kd_from_ENS2":
        return ctx.settings.ens2_teachers
    return [n[len("C_kd_from_"):]] if n.startswith("C_kd_from_") else None


def retrain_members_final(ctx):
    """Retrain the ensemble members on TRAIN+DEV (plain models first, then KD students from the retrained teachers) and write the final f1 / recall submissions.

    Args:
        ctx (Context): Pipeline context (needs :func:`tune_ensemble` to have run).

    Returns:
        None
    """
    s = ctx.settings
    names = s.ens_members
    need = set(names)
    for n in names:
        need |= set(teachers_of(ctx, n) or [])
    ckp = lambda n: os.path.join(ctx.work, n + ".pt")
    fin = lambda n: os.path.join(ctx.work, f"FINAL_{n}.pt")
    for n in sorted(x for x in need if not teachers_of(ctx, x)):
        if not os.path.exists(fin(n)):
            run_final_rr(ctx, f"FINAL_{n}", ckp(n), SEED=s.seed_final)
    for n in sorted(x for x in need if teachers_of(ctx, x)):
        if os.path.exists(fin(n)):
            continue
        T = [load_rr(ctx, fin(t)) for t in teachers_of(ctx, n)]
        for tm, _ in T:
            for p in tm.parameters():
                p.requires_grad = False
        run_final_rr(ctx, f"FINAL_{n}", ckp(n), teacher=T, kd_from="+".join(teachers_of(ctx, n)), SEED=s.seed_final)
        del T
        torch.cuda.empty_cache()
    write_two(ctx, [fin(n) for n in names], "final_ens")


def final_seed_ensemble(ctx, best_names, run=True):
    """Retrain chosen checkpoints on TRAIN+DEV with several seeds and write ``sub_final_ensemble`` (the notebook's active final cell).

    Args:
        ctx (Context): Pipeline context (``settings.final_seeds``, ``bias`` from the analysis stage).
        best_names (List[str]): Run names to retrain (e.g. ``["C_kd_from_ENS4"]``).
        run (bool): Do nothing when False.

    Returns:
        List[str]: Paths of the final checkpoints (empty if ``run`` is False).
    """
    if not run:
        return []
    finals = []
    for name in best_names:
        ckpt_path = os.path.join(ctx.work, name + ".pt")
        for sd in ctx.settings.final_seeds:
            nm = f"FINAL_{name}_s{sd}"
            if not os.path.exists(os.path.join(ctx.work, nm + ".pt")):
                run_final_rr(ctx, nm, ckpt_path, SEED=sd)
            finals.append(os.path.join(ctx.work, nm + ".pt"))
    print(f"Creating submission from {len(finals)} models...")
    predict_test_ens(ctx, finals, "sub_final_ensemble", bias=ctx.state.get("bias"))
    ctx.state["finals"] = finals
    return finals
