"""Recall-improvement pass: deterministic audit and LABEL-FREE decoding fixes.

Dev has only ~10 documents, so per-class bias tuning (13 free parameters) over-fits. These methods use only posteriors and training statistics
(at most 2 free parameters): prior matching, Saerens-EM prior correction, stickiness-calibrated Viterbi on posteriors and blip smoothing.
Each is judged with a **document bootstrap** of its macro-F1 gain and a share guard checks the submission before it is written.
Metrics here average over labels that have gold support.
"""
import re

import numpy as np
import torch

from rr.eval.metrics import macro_prf, rr_scores, print_scores, stickiness
from rr.inference.ensemble import ens_probs, write_submission
from rr.models.crf import crf_viterbi
from rr.training.trainer import class_weights, collect_probs, forward_batch, get_window, load_rr


# ---------------------------------------------------------------------------------------------- R_A: audit
def rr_audit(ctx, probs, docs, train_docs=None, show=True):
    """Deterministic checks on posteriors. Flags: SINK (predicted > 1.3x gold), RECALL-LIMITED (R < P - 0.15), ABSENT-IN-GOLD.

    Args:
        ctx (Context): Pipeline context.
        probs (List[np.ndarray]): Posteriors per document.
        docs (List[dict]): Documents with gold labels.
        train_docs (Optional[List[dict]]): Training documents for the prior (default: all train).
        show (bool): Print the table.

    Returns:
        Dict[str, str]: ``label -> flag``.
    """
    nl, labels = ctx.data.nl, ctx.data.labels
    train_docs = train_docs or ctx.data.train
    y = np.concatenate([d["y"] for d in docs])
    P = np.concatenate(probs)
    pred = P.argmax(1)
    tr = np.bincount(np.concatenate([d["y"] for d in train_docs]), minlength=nl).astype(float)
    trs = tr / tr.sum()
    gs = np.bincount(y, minlength=nl) / len(y)
    ps = np.bincount(pred, minlength=nl) / len(pred)
    flags = {}
    if show:
        print(f"{'label':24s}{'gold':>6s}{'pred':>6s}{'p/g':>6s}{'P':>7s}{'R':>7s}{'F1':>7s}{'train%':>8s}{'gold%':>7s}{'pred%':>7s}  flag")
    for c in range(nl):
        g = int((y == c).sum())
        n = int((pred == c).sum())
        tp = int(((y == c) & (pred == c)).sum())
        p = tp / n if n else 0.0
        r = tp / g if g else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        fl = "ABSENT-IN-GOLD" if g == 0 else "SINK" if n > 1.3 * g and g >= 10 else "RECALL-LIMITED" if r < p - 0.15 and g >= 10 else ""
        if fl:
            flags[labels[c]] = fl
        if show:
            print(f"{labels[c]:24s}{g:6d}{n:6d}{(n / g if g else float('nan')):6.2f}{p:7.3f}{r:7.3f}{f:7.3f}{100 * trs[c]:8.1f}{100 * gs[c]:7.1f}{100 * ps[c]:7.1f}  {fl}")
    mp, mr, mf = macro_prf(y, pred, nl)
    gold_stick = stickiness([d["y"] for d in docs])
    pred_stick = stickiness(np.split(pred, np.cumsum([len(d["y"]) for d in docs])[:-1]))
    ok = P.argmax(1) == y
    conf = P.max(1)
    print(f"macro P/R/F1 (gold labels only) = {mp:.3f}/{mr:.3f}/{mf:.3f}  -> {'RECALL-LIMITED' if mp - mr > 0.05 else 'balanced'} (P - R = {mp - mr:+.3f})")
    print(f"stickiness P(next == this): gold {gold_stick:.3f} | predicted {pred_stick:.3f} | train {stickiness([d['y'] for d in train_docs]):.3f}")
    print(f"mean confidence: correct {conf[ok].mean():.3f} | wrong {conf[~ok].mean():.3f}   (wrong ~ correct = posteriors are not informative, fix the model not the decoder)")
    for c in np.argsort(-(ps - trs))[:2]:
        print(f"  share drift: {labels[c]} predicted {100 * ps[c]:.1f}% vs train {100 * trs[c]:.1f}%")
    return flags


def get_dev_probs(ctx):
    """Dev posteriors of the ensemble chosen in the analysis stage, else of ``best_set``, else of the best single model.

    Args:
        ctx (Context): Pipeline context.

    Returns:
        List[np.ndarray]: Posteriors per dev document.
    """
    if ctx.state.get("ens_p") is not None:
        return ctx.state["ens_p"]
    ck = ctx.state.get("best_set") or [get_window(ctx, 1, include_kd=True)[0]["ckpt"]]
    return ens_probs(ctx, ck, ctx.data.dev)


# ---------------------------------------------------------------------------------------------- R_B: prior correction + bootstrap
def prior_match_bias(P, target, iters=200, eta=0.5, smooth=1.0):
    """Additive log-bias so that the HARD class shares of ``argmax(log P + b)`` equal ``target`` (needs no labels).

    Args:
        P (np.ndarray): Posteriors (n, K).
        target (np.ndarray): Target class shares.
        iters (int): Maximum iterations.
        eta (float): Step size.
        smooth (float): Count smoothing.

    Returns:
        np.ndarray: Bias (K,).
    """
    L = np.log(P + 1e-9)
    K = P.shape[1]
    b = np.zeros(K)
    t = np.maximum(np.asarray(target, float), 1e-6)
    t = t / t.sum()
    for i in range(iters):
        q = (np.bincount((L + b).argmax(1), minlength=K) + smooth) / (len(P) + smooth * K)
        step = (eta / (1 + i / 40.0)) * (np.log(t) - np.log(q))
        b = b + step - step.mean()
        if np.abs(step).max() < 1e-3:
            break
    return b


def saerens_em(P, train_prior, iters=10, clip=(0.25, 4.0), tol=1e-6):
    """Saerens-Latinne-Decaestecker (2002) EM prior correction; posteriors are rescaled by ``pi_test / pi_train``.

    Few iterations and a clipped ratio act as regularisation. Caveat: EM reads over-prediction of a sink label as a prior shift and
    reinforces it, so it usually loses to prior matching; the bootstrap gate rejects it when it hurts.

    Args:
        P (np.ndarray): Posteriors (n, K).
        train_prior (np.ndarray): Training prior.
        iters (int): EM iterations.
        clip (Tuple[float, float]): Clip range of ``pi / pi_train``.
        tol (float): Convergence tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(adjusted posteriors, estimated prior)``.
    """
    pt = np.maximum(np.asarray(train_prior, float), 1e-9)
    pt = pt / pt.sum()
    pi = pt.copy()
    for _ in range(iters):
        Q = P * np.clip(pi / pt, *clip)
        Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
        done = np.abs(new - pi).max() < tol
        pi = new
        if done:
            break
    Q = P * np.clip(pi / pt, *clip)
    return Q / Q.sum(1, keepdims=True), pi


def train_share(ctx, docs, alpha=1.0):
    """Training label shares (smoothed, optionally tempered by ``alpha``).

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Training documents.
        alpha (float): Exponent (1 = plain shares).

    Returns:
        np.ndarray: Shares (nl,).
    """
    c = np.bincount(np.concatenate([d["y"] for d in docs]), minlength=ctx.data.nl) + 1.0
    s = (c / c.sum()) ** alpha
    return s / s.sum()


def effective_prior(ctx, docs, cfg):
    """Prior implied by class-weighted CE: ``pi_c * w_c`` (what the model was trained to predict).

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Training documents.
        cfg (dict): Config (class-weight settings).

    Returns:
        np.ndarray: Normalised effective prior.
    """
    w = class_weights(ctx, docs, cfg).cpu().numpy()
    s = train_share(ctx, docs) * w
    return s / s.sum()


def split_by_domain(docs):
    """Document indices per domain.

    Args:
        docs (List[dict]): Documents.

    Returns:
        Dict[str, List[int]]: ``{"CL": [...], "IT": [...]}``.
    """
    return {dom: [i for i, d in enumerate(docs) if d["domain"] == dom] for dom in ("CL", "IT")}


def decode_prior_match(ctx, probs, docs, train_docs=None, alpha=1.0, by_domain=True):
    """Per-domain prior matching (CL and IT have different label mixes); domains with < 3 train docs fall back to the pooled share.

    Args:
        ctx (Context): Pipeline context.
        probs (List[np.ndarray]): Posteriors per document.
        docs (List[dict]): Documents.
        train_docs (Optional[List[dict]]): Training documents.
        alpha (float): Share exponent.
        by_domain (bool): Use per-domain training shares.

    Returns:
        List[np.ndarray]: Predicted label ids per document.
    """
    train_docs = train_docs or ctx.data.train
    out = [None] * len(docs)
    for dom, idx in split_by_domain(docs).items():
        if not idx:
            continue
        tdocs = [d for d in train_docs if d["domain"] == dom] if by_domain else train_docs
        tdocs = tdocs if len(tdocs) >= 3 else train_docs
        b = prior_match_bias(np.concatenate([probs[i] for i in idx]), train_share(ctx, tdocs, alpha))
        for i in idx:
            out[i] = (np.log(probs[i] + 1e-9) + b).argmax(1)
    return out


def boot_gain(ctx, docs, base_pred, new_pred, n=300, seed=0):
    """Document-level bootstrap of the macro-F1 gain (gold labels only).

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Documents with gold labels.
        base_pred (List[np.ndarray]): Baseline predictions per document.
        new_pred (List[np.ndarray]): New predictions per document.
        n (int): Number of resamples.
        seed (int): RNG seed.

    Returns:
        Tuple[float, float]: ``(mean gain, share of resamples with gain > 0)``.
    """
    nl = ctx.data.nl
    rng = np.random.default_rng(seed)
    ys = [d["y"] for d in docs]
    g = []
    for _ in range(n):
        ix = rng.integers(0, len(docs), len(docs))
        y = np.concatenate([ys[i] for i in ix])
        g.append(macro_prf(y, np.concatenate([new_pred[i] for i in ix]), nl)[2] - macro_prf(y, np.concatenate([base_pred[i] for i in ix]), nl)[2])
    return float(np.mean(g)), float(np.mean(np.array(g) > 0))


# ---------------------------------------------------------------------------------------------- R_C: calibrated Viterbi on posteriors
def train_bigram(ctx, docs, smooth=0.5):
    """Smoothed label-bigram log-probabilities and start distribution from training documents.

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Training documents.
        smooth (float): Additive smoothing.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(log A (nl, nl), log start (nl,))``.
    """
    nl = ctx.data.nl
    A = np.full((nl, nl), smooth)
    st = np.full(nl, smooth)
    for d in docs:
        y = np.asarray(d["y"])
        np.add.at(A, (y[:-1], y[1:]), 1)
        st[y[0]] += 1
    return np.log(A / A.sum(1, keepdims=True)), np.log(st / st.sum())


def bigram_viterbi(P, logA, logst, lam=0.3, bias=None):
    """``argmax_path sum_t [log p(c_t|x) + bias[c_t]] + lam * (log start + sum_t log A[c_{t-1}, c_t])``; ``lam = 0`` gives plain argmax.

    Args:
        P (np.ndarray): Posteriors (n, K).
        logA (np.ndarray): Log transition matrix.
        logst (np.ndarray): Log start distribution.
        lam (float): Transition weight.
        bias (Optional[np.ndarray]): Per-class log-bias.

    Returns:
        np.ndarray: Label ids (n,).
    """
    E = np.log(P + 1e-9) + (0.0 if bias is None else bias)
    if lam <= 0:
        return E.argmax(1)
    n, K = E.shape
    T = lam * logA
    delta = E[0] + lam * logst
    back = np.zeros((n, K), dtype=np.int32)
    for t in range(1, n):
        cand = delta[:, None] + T
        back[t] = cand.argmax(0)
        delta = cand.max(0) + E[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def calibrate_decoder(ctx, probs, train_docs, alpha=1.0, rounds=10):
    """LABEL-FREE decoder calibration: (1) bias so predicted shares match the training shares, (2) ``lam`` so predicted stickiness
    matches the training stickiness (bisection), (3) a few refinement rounds of the bias with the Viterbi shares.

    Args:
        ctx (Context): Pipeline context.
        probs (List[np.ndarray]): Posteriors of the documents to decode.
        train_docs (List[dict]): Training documents (statistics source).
        alpha (float): Share exponent.
        rounds (int): Refinement rounds.

    Returns:
        dict: ``bias, lam, logA, logst, target, stick_target``.
    """
    nl = ctx.data.nl
    ytr = [d["y"] for d in train_docs]
    target = train_share(ctx, train_docs, alpha)
    stick_t = stickiness(ytr)
    logA, logst = train_bigram(ctx, train_docs)
    b = prior_match_bias(np.concatenate(probs), target)
    dec = lambda lam, bb: [bigram_viterbi(P, logA, logst, lam, bb) for P in probs]
    lo, hi = 0.0, 3.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if stickiness(dec(mid, b)) < stick_t:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    for i in range(rounds):
        pred = np.concatenate(dec(lam, b))
        q = (np.bincount(pred, minlength=nl) + 1.0) / (len(pred) + nl)
        step = (0.5 / (1 + i / 4.0)) * (np.log(target) - np.log(q))
        b = b + step - step.mean()
    return dict(bias=b, lam=lam, logA=logA, logst=logst, target=target, stick_target=stick_t)


def decode_calibrated(ctx, probs, docs, train_docs=None, alpha=1.0, by_domain=True):
    """Calibrated Viterbi decoding, per domain.

    Args:
        ctx (Context): Pipeline context.
        probs (List[np.ndarray]): Posteriors per document.
        docs (List[dict]): Documents.
        train_docs (Optional[List[dict]]): Training documents.
        alpha (float): Share exponent.
        by_domain (bool): Calibrate per domain.

    Returns:
        Tuple[List[np.ndarray], dict]: ``(predictions per document, calibration per domain)``.
    """
    train_docs = train_docs or ctx.data.train
    out = [None] * len(docs)
    cals = {}
    for dom, idx in split_by_domain(docs).items():
        if not idx:
            continue
        tdocs = [d for d in train_docs if d["domain"] == dom] if by_domain else train_docs
        tdocs = tdocs if len(tdocs) >= 3 else train_docs
        cal = calibrate_decoder(ctx, [probs[i] for i in idx], tdocs, alpha)
        cals[dom] = cal
        for i in idx:
            out[i] = bigram_viterbi(probs[i], cal["logA"], cal["logst"], cal["lam"], cal["bias"])
    return out, cals


def evaluate_recall_methods(ctx, docs, probs, cfg=None, train_docs=None):
    """Compare argmax, EM (two priors), prior matching (alpha 1 / .75 / .5) and calibrated Viterbi on dev with a bootstrap of the gain over argmax.

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Dev documents.
        probs (List[np.ndarray]): Posteriors per document.
        cfg (Optional[dict]): Config (default ``ctx.cfg``).
        train_docs (Optional[List[dict]]): Training documents.

    Returns:
        Tuple[dict, dict]: ``(results per method with P, R, F1, gain, p_pos, pred; calibrations)``.
    """
    train_docs = train_docs or ctx.data.train
    cfg = cfg or ctx.cfg
    nl = ctx.data.nl
    y = np.concatenate([d["y"] for d in docs])
    base = [P.argmax(1) for P in probs]
    res = {}

    def add(name, pred):
        """Score one method and register it.

        Args:
            name (str): Method name.
            pred (List[np.ndarray]): Predictions per document.

        Returns:
            None
        """
        mp, mr, mf = macro_prf(y, np.concatenate(pred), nl)
        g, frac = boot_gain(ctx, docs, base, pred) if name != "argmax" else (0.0, 0.0)
        res[name] = dict(P=mp, R=mr, F1=mf, gain=g, p_pos=frac, pred=pred)

    add("argmax", base)
    P_all = np.concatenate(probs)
    lens = np.cumsum([len(P) for P in probs])[:-1]
    for nm, prior in (("EM (train prior)", train_share(ctx, train_docs)), ("EM (weighted prior)", effective_prior(ctx, train_docs, cfg))):
        Q, _ = saerens_em(P_all, prior)
        add(nm, [q.argmax(1) for q in np.split(Q, lens)])
    for a in (1.0, 0.75, 0.5):
        add(f"prior-match a={a}", decode_prior_match(ctx, probs, docs, train_docs, a))
    cal_pred, cals = decode_calibrated(ctx, probs, docs, train_docs)
    add("calibrated Viterbi", cal_pred)
    print(f"{'method':24s}{'P':>7s}{'R':>7s}{'F1':>7s}{'gain':>8s}{'P(gain>0)':>11s}")
    for k, v in res.items():
        print(f"{k:24s}{v['P']:7.3f}{v['R']:7.3f}{v['F1']:7.3f}{v['gain']:+8.3f}{v['p_pos']:11.2f}")
    print("calibrated lam / stickiness target:", {d: (round(c['lam'], 2), round(c['stick_target'], 3)) for d, c in cals.items()})
    return res, cals


# ---------------------------------------------------------------------------------------------- R_D: choose, guard, submit
def share_guard(ctx, pred_lists, train_docs=None, lo=0.5, hi=2.0):
    """Check that every label with a training share >= 1% is predicted within ``[lo, hi]`` x its training share.

    Args:
        ctx (Context): Pipeline context.
        pred_lists (List[np.ndarray]): Predictions per document.
        train_docs (Optional[List[dict]]): Training documents.
        lo (float): Lower factor.
        hi (float): Upper factor.

    Returns:
        bool: True if no label is outside the range.
    """
    train_docs = train_docs or ctx.data.train
    nl, labels = ctx.data.nl, ctx.data.labels
    ts = train_share(ctx, train_docs)
    pred = np.concatenate(pred_lists)
    ps = np.bincount(pred, minlength=nl) / len(pred)
    bad = [f"{labels[c]} {100 * ps[c]:.1f}% vs train {100 * ts[c]:.1f}%" for c in range(nl) if ts[c] >= 0.01 and not (lo * ts[c] <= ps[c] <= hi * ts[c])]
    print("[share guard]", "OK: every label with train share >= 1% is within [0.5x, 2x] of its training share" if not bad else "OUTSIDE [0.5x, 2x]: " + "; ".join(bad))
    return not bad


def choose_recall_method(ctx):
    """Evaluate the label-free methods on dev and accept the best one that gains >= ``accept_gain`` and wins >= ``accept_frac`` of bootstraps.

    Stores ``best_method``, ``recall_res`` and ``recall_cals`` in ``ctx.state``.

    Args:
        ctx (Context): Pipeline context.

    Returns:
        str: Accepted method name (``argmax`` if none qualifies).
    """
    s = ctx.settings
    try:
        p = get_dev_probs(ctx)
        res, cals = evaluate_recall_methods(ctx, ctx.data.dev, p)
        ok = {k: v for k, v in res.items() if k != "argmax" and v["gain"] >= s.accept_gain and v["p_pos"] >= s.accept_frac}
        best = max(ok, key=lambda k: ok[k]["F1"]) if ok else "argmax"
        print(f"-> accepted method: {best}")
        share_guard(ctx, res[best]["pred"])
        ctx.state.update(recall_res=res, recall_cals=cals, dev_probs=p)
    except Exception as e:
        print("recall evaluation skipped:", repr(e)[:200])
        best = "argmax"
    ctx.state["best_method"] = best
    return best


def decode_with(ctx, method, probs, docs, cfg=None, train_docs=None, alpha=None):
    """Apply an accepted label-free method to ANY posteriors (e.g. the test ensemble); label-free, so it is allowed on test.

    Args:
        ctx (Context): Pipeline context.
        method (str): ``argmax``, ``EM (train prior)``, ``EM (weighted prior)``, ``prior-match a=<alpha>`` or ``calibrated Viterbi``.
        probs (List[np.ndarray]): Posteriors per document.
        docs (List[dict]): Documents.
        cfg (Optional[dict]): Config.
        train_docs (Optional[List[dict]]): Training documents.
        alpha (Optional[float]): Share exponent override for prior matching.

    Returns:
        List[np.ndarray]: Predicted label ids per document.
    """
    train_docs = train_docs or ctx.data.train
    cfg = cfg or ctx.cfg
    lens = np.cumsum([len(P) for P in probs])[:-1]
    if method == "argmax":
        return [P.argmax(1) for P in probs]
    if method.startswith("EM"):
        Q, _ = saerens_em(np.concatenate(probs), train_share(ctx, train_docs) if "train" in method else effective_prior(ctx, train_docs, cfg))
        return [q.argmax(1) for q in np.split(Q, lens)]
    if method.startswith("prior-match"):
        return decode_prior_match(ctx, probs, docs, train_docs, alpha or float(method.split("a=")[1]))
    if method.startswith("calibrated"):
        return decode_calibrated(ctx, probs, docs, train_docs)[0]
    raise ValueError(method)


def submit_recall(ctx, ckpts, name, method=None, weights=None):
    """Ensemble posteriors of the checkpoints on TEST -> accepted label-free decoder -> share guard -> write submission.

    Args:
        ctx (Context): Pipeline context.
        ckpts (List[str]): Checkpoints.
        name (str): Submission name.
        method (Optional[str]): Decoder (default ``ctx.state['best_method']``).
        weights (Optional[List[float]]): Member weights.

    Returns:
        str: Zip path.
    """
    method = method or ctx.state.get("best_method", "argmax")
    pr = ens_probs(ctx, ckpts, ctx.data.test, weights)
    lab = decode_with(ctx, method, pr, ctx.data.test)
    share_guard(ctx, lab)
    labels = ctx.data.labels
    preds = {d["id"]: [labels[i] for i in l] for d, l in zip(ctx.data.test, lab)}
    print("decoder:", method)
    return write_submission(ctx, preds, name)


# ---------------------------------------------------------------------------------------------- CRF Viterbi comparison (single CRF model)
def predict_viterbi_for_docs(ctx, model, docs, cfg, bias=None):
    """Viterbi-decoded label sequences per document using the model's own trained CRF transitions (one document-sized window).

    The notebook's unused ``make_windows(..., cfg["WINDOW"], ...)`` line was dropped (that key does not exist).

    Args:
        ctx (Context): Pipeline context.
        model (RRModel): Model with a CRF head.
        docs (List[dict]): Documents.
        cfg (dict): Config.
        bias (Optional[np.ndarray]): Per-class log-bias added to the emissions.

    Returns:
        Dict[str, List[str]]: Label names per document id.
    """
    model.eval()
    preds = {}
    with torch.no_grad():
        for d in docs:
            x, mask, _, _ = forward_batch(ctx, model, [(d, 0, len(d["sents"]))], cfg)
            em = model.emissions(x)
            decoded = crf_viterbi(model.crf, em, mask, bias=bias)[0]
            preds[d["id"]] = [ctx.data.labels[i] for i in decoded]
    return preds


def oversample_weights(docs, power=0.5):
    """Per-document sampling weights that up-weight documents containing rarer labels (mean 1).

    Args:
        docs (List[dict]): Documents with ``labels``.
        power (float): Exponent in (0, 1]; 0.5 is a gentle start.

    Returns:
        np.ndarray: Normalised weights.
    """
    from collections import Counter
    cnt = Counter(l for d in docs for l in d["labels"])
    w = np.array([max((1.0 / cnt[l]) for l in set(d["labels"])) ** power for d in docs])
    return w / w.mean()


def compare_decoders(ctx, ckpt_path, dev_docs):
    """Argmax vs CRF Viterbi vs bias-tuned Viterbi on dev (P / R / F1 side by side).

    The bias is fitted with ``tune_bias`` on the dev posteriors (the notebook passed the float returned by ``crossfit_bias``, which is not a bias vector).
    It is in-sample for dev; the honest estimate is ``crossfit_bias``.

    Args:
        ctx (Context): Pipeline context.
        ckpt_path (str): Checkpoint of a model trained with a CRF.
        dev_docs (List[dict]): Dev documents.

    Returns:
        Tuple[dict, dict, dict]: Scores of argmax, Viterbi and biased Viterbi.
    """
    from rr.inference.ensemble import crossfit_bias, tune_bias
    labels, nl = ctx.data.labels, ctx.data.nl
    model, cfg = load_rr(ctx, ckpt_path)
    probs = collect_probs(ctx, model, dev_docs, cfg)
    y_flat = np.concatenate([d["y"] for d in dev_docs])
    sc_argmax = rr_scores(labels, [labels[i] for i in y_flat], [labels[i] for i in np.concatenate(probs).argmax(1)])
    print(f"honest cross-fit macro-F1 of bias tuning: {crossfit_bias(nl, dev_docs, probs):.4f}")
    bias = tune_bias(np.concatenate(probs), y_flat, nl)
    plain = predict_viterbi_for_docs(ctx, model, dev_docs, cfg, bias=None)
    biased = predict_viterbi_for_docs(ctx, model, dev_docs, cfg, bias=bias)
    gold_flat = [l for d in dev_docs for l in d["labels"]]
    flat = lambda pd_: [l for d in dev_docs for l in pd_[d["id"]]]
    sc_v, sc_vb = rr_scores(labels, gold_flat, flat(plain)), rr_scores(labels, gold_flat, flat(biased))
    print("=== argmax (baseline) ===")
    print_scores(sc_argmax)
    print("\n=== Viterbi (trained CRF transitions, no bias) ===")
    print_scores(sc_v)
    print("\n=== Viterbi + tuned bias ===")
    print_scores(sc_vb)
    print(f"\nmacro-R: argmax={sc_argmax['macro_r']:.4f}  viterbi={sc_v['macro_r']:.4f}  viterbi+bias={sc_vb['macro_r']:.4f}")
    del model
    torch.cuda.empty_cache()
    return sc_argmax, sc_v, sc_vb


# ---------------------------------------------------------------------------------------------- qualitative inspection, R_F, R_G
def show_missing_cases(ctx, docs, pred_concat, top_pairs=8, examples_per_pair=4, context=1):
    """Print the most frequent (gold, predicted) confusions with example sentences and their neighbours.

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Dev documents.
        pred_concat (np.ndarray): Flat predicted ids aligned with ``np.concatenate([d["y"] for d in docs])``.
        top_pairs (int): Number of confusion pairs shown.
        examples_per_pair (int): Examples per pair.
        context (int): Neighbouring sentences on each side.

    Returns:
        List[Tuple[tuple, List[int]]]: Confusion pairs sorted by frequency (with flat indices).
    """
    labels = ctx.data.labels
    y = np.concatenate([d["y"] for d in docs])
    assert len(pred_concat) == len(y)
    idx_map = [(d, j) for d in docs for j in range(len(d["y"]))]
    pair_examples = {}
    for flat_i, (g, p) in enumerate(zip(y, pred_concat)):
        if g != p:
            pair_examples.setdefault((labels[g], labels[p]), []).append(flat_i)
    pair_counts = sorted(pair_examples.items(), key=lambda kv: -len(kv[1]))
    print(f"{'gold -> predicted':45s} {'count':>6s}")
    for pair, idxs in pair_counts[:top_pairs]:
        print(f"{pair[0]:>20s} -> {pair[1]:<20s} {len(idxs):6d}")
    print("\n--- examples ---")
    for pair, idxs in pair_counts[:top_pairs]:
        print(f"\n[{pair[0]} -> {pair[1]}]  ({len(idxs)} total, showing {min(examples_per_pair, len(idxs))})")
        for flat_i in idxs[:examples_per_pair]:
            d, j = idx_map[flat_i]
            for k in range(max(0, j - context), min(len(d["sents"]), j + context + 1)):
                print(f"  {'>> ' if k == j else '   '}[{d['id']}:{k}] {d['sents'][k][:140]}")
            print()
    return pair_counts


def pairwise_bias_nudge(ctx, pair_counts, strength=0.3, top_k=6):
    """For the top-k confusion pairs push the bias toward the missed gold label and away from the confused-into label.

    Args:
        ctx (Context): Pipeline context.
        pair_counts (list): Output of :func:`show_missing_cases`.
        strength (float): Overall strength.
        top_k (int): Number of pairs used.

    Returns:
        np.ndarray: Zero-mean bias (nl,).
    """
    labels = ctx.data.labels
    b = np.zeros(ctx.data.nl)
    total = sum(len(idxs) for _, idxs in pair_counts[:top_k])
    for (gold_lab, pred_lab), idxs in pair_counts[:top_k]:
        w = strength * (len(idxs) / max(total, 1))
        b[labels.index(gold_lab)] += w
        b[labels.index(pred_lab)] -= w
    return b - b.mean()


# Illustrative indicator phrases -- EDIT to match the examples printed by show_missing_cases for your run (not a validated list).
PAIR_PATTERNS = {
    ("Dissent", "RulingByPresentCourt"): [r"\bdissent", r"\bi am unable to agree", r"\bwith respect, i differ"],
    ("ArgumentRespondent", "ArgumentPetitioner"): [r"\bcounsel for the respondent", r"\blearned counsel appearing for the respondent"],
    ("ArgumentPetitioner", "ArgumentRespondent"): [r"\bcounsel for the (appellant|petitioner)", r"\blearned counsel appearing for the (appellant|petitioner)"],
    ("RulingByLowerCourt", "Fact"): [r"\bthe (tribunal|lower court|high court) (held|observed|ruled)"],
}
_PAIR_RE = {pair: [re.compile(p, re.I) for p in pats] for pair, pats in PAIR_PATTERNS.items()}


def boilerplate_override(ctx, docs, pred_concat):
    """Flip a prediction ``pred_lab -> gold_lab`` only when that exact confusion pair is in ``PAIR_PATTERNS`` and the sentence matches one of its regexes.

    Args:
        ctx (Context): Pipeline context.
        docs (List[dict]): Documents.
        pred_concat (np.ndarray): Flat predicted ids.

    Returns:
        np.ndarray: Updated flat predictions.
    """
    labels = ctx.data.labels
    idx_map = [(d, j) for d in docs for j in range(len(d["y"]))]
    out = pred_concat.copy()
    n_flipped = 0
    for flat_i, p in enumerate(pred_concat):
        d, j = idx_map[flat_i]
        sent = d["sents"][j]
        for (gold_lab, pat_pred_lab), regs in _PAIR_RE.items():
            if pat_pred_lab == labels[p] and any(r.search(sent) for r in regs):
                out[flat_i] = labels.index(gold_lab)
                n_flipped += 1
                break
    print(f"boilerplate_override: flipped {n_flipped} / {len(pred_concat)} predictions")
    return out


def smooth_blips(probs, pred_lists, min_margin=0.15):
    """Snap isolated length-1 label flips to their neighbours when both neighbours agree and the blip's confidence margin is small.

    Args:
        probs (List[np.ndarray]): Posteriors per document.
        pred_lists (List[np.ndarray]): Predicted label ids per document.
        min_margin (float): Maximum (top - runner-up) probability margin of a blip that may be merged.

    Returns:
        List[np.ndarray]: Smoothed predictions.
    """
    out = []
    for P, pred in zip(probs, pred_lists):
        pred = np.asarray(pred).copy()
        sp = np.sort(P, axis=1)
        margin = sp[:, -1] - sp[:, -2]
        for t in range(1, len(pred) - 1):
            if pred[t] != pred[t - 1] and pred[t] != pred[t + 1] and pred[t - 1] == pred[t + 1] and margin[t] < min_margin:
                pred[t] = pred[t - 1]
        out.append(pred)
    return out


def inspect_and_refine(ctx):
    """Missing-case inspector, pairwise nudge / boilerplate override gate (R_F) and blip-smoothing gate (R_G) on dev; all gated by the accept rules.

    Args:
        ctx (Context): Pipeline context (needs :func:`choose_recall_method` first).

    Returns:
        None
    """
    s, data = ctx.settings, ctx.data
    nl = data.nl
    p = ctx.state.get("dev_probs") or get_dev_probs(ctx)
    res, _ = evaluate_recall_methods(ctx, data.dev, p)
    method = ctx.state.get("best_method", "argmax")
    method = method if method in res else "argmax"
    pred_best = np.concatenate(res[method]["pred"])
    n_sent = sum(len(d["y"]) for d in data.dev)
    if len(pred_best) != n_sent:
        print(f"[mismatch] pred={len(pred_best)} vs DEV sentences={n_sent}")
        return
    print(f"decoder used for inspection: {method}")
    pair_counts = show_missing_cases(ctx, data.dev, pred_best, top_pairs=8, examples_per_pair=3)
    nudge = pairwise_bias_nudge(ctx, pair_counts)
    print("pairwise nudge (top labels):", sorted(zip(data.labels, nudge), key=lambda kv: -abs(kv[1]))[:6])
    y_dev = np.concatenate([d["y"] for d in data.dev])
    base_f1 = macro_prf(y_dev, pred_best, nl)[2]
    flipped = boilerplate_override(ctx, data.dev, pred_best)
    flipped_f1 = macro_prf(y_dev, flipped, nl)[2]
    print(f"macro-F1: current best={base_f1:.4f}  with boilerplate override={flipped_f1:.4f}")
    print("ACCEPTED: use `flipped` for submission." if flipped_f1 >= base_f1 + s.accept_gain else "REJECTED by the same accept gate as the label-free methods -- edit PAIR_PATTERNS from the printed examples and re-run.")
    cal_pred, _ = decode_calibrated(ctx, p, data.dev)
    base = macro_prf(y_dev, np.concatenate(cal_pred), nl)[2]
    best_f1, best_margin, best_smoothed = base, None, cal_pred
    for margin in (0.10, 0.15, 0.20, 0.25):
        smoothed = smooth_blips(p, cal_pred, min_margin=margin)
        f1 = macro_prf(y_dev, np.concatenate(smoothed), nl)[2]
        print(f"min_margin={margin:.2f}  macro-F1={f1:.4f}  (calibrated Viterbi alone: {base:.4f})")
        if f1 > best_f1:
            best_f1, best_margin, best_smoothed = f1, margin, smoothed
    if best_margin is not None and best_f1 >= base + s.accept_gain:
        _, frac = boot_gain(ctx, data.dev, cal_pred, best_smoothed)
        print(f"ACCEPTED: min_margin={best_margin} gives +{best_f1 - base:.4f} macro-F1, bootstrap win rate {frac:.2f}")
        if frac >= s.accept_frac:
            share_guard(ctx, best_smoothed)
            print(f"-> use smooth_blips(probs, cal_pred, min_margin={best_margin}) ahead of submission")
        else:
            print(f"REJECTED by bootstrap win-rate gate ({frac:.2f} < {s.accept_frac}). Keep calibrated Viterbi alone.")
    else:
        print("REJECTED: no margin threshold beat calibrated Viterbi by accept_gain -- keep it as-is.")
