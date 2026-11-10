"""Honest cross-fit selection of post-processing components.

A reverse model (trained on fold2, predicts fold1) gives out-of-fold predictions on both folds: PA = fold2 preds of the fold1-trained
model, PB = fold1 preds of the reverse model. Tables/rules learned on one fold's GOLD are tested on the OTHER fold's predictions, and a
component is kept only if it improves BOTH folds by more than ``margin``.
"""
import os
import pickle

from .. import state as S
from ..config import CFG, WORK
from ..training.inference import predict_spans_raw
from ..training.trainer import run_experiment, load_trained, cfg_from_ckpt
from .edit_rules import learn_edit_rules
from .filters import learn_fp_blacklist, refine_safe_labels
from .period import learn_period_table
from .pipeline import f1_with, postprocess_doc

RULE_LABELS = ["PREC", "STAT", "CASENO", "COURT", "JUDGE", "APP", "RESP", "AUTH"]


def out_of_fold_predictions(base_ckpt, rev_name="F_rev"):
    """Out-of-fold RAW predictions on both folds (trains the reverse model if its checkpoint is missing).

    Args:
        base_ckpt (str): validated fold1-trained checkpoint (e.g. ``F_focal.pt``).
        rev_name (str): name of the reverse (fold2-trained) run (same config as ``base_ckpt``).

    Returns:
        tuple[dict, dict]: (PA = fold2 predictions, PB = fold1 predictions), each doc -> raw spans.
    """
    rev_path = os.path.join(WORK, f"{rev_name}.pt")
    if not os.path.exists(rev_path):
        run_experiment(rev_name, train_ids=S.VAL_IDS, val_ids=S.TRAIN_IDS, **cfg_from_ckpt(base_ckpt))
    mA, cA = load_trained(base_ckpt); PA = predict_spans_raw(mA, S.VAL_IDS, cA)
    mB, cB = load_trained(rev_path);  PB = predict_spans_raw(mB, S.TRAIN_IDS, cB)
    if any(cA.get(k) != cB.get(k) for k in ("ARCH", "TASK", "METHOD", "MAX_LEN", "OVERLAP")):
        print(f"!!! {rev_name} was trained with a different config than the base run -> delete {rev_name}.pt and re-run")
    return PA, PB


def report_crossfit(PA, PB, rules_A, rules_B):
    """Print the fold-wise effect of edit rules, counsel rule, propagation and period/vote variants.

    Args:
        PA (dict): fold2 raw predictions.
        PB (dict): fold1 raw predictions.
        rules_A (list): rules learned on fold2 gold.
        rules_B (list): rules learned on fold1 gold.

    Returns:
        None
    """
    T, V = S.TRAIN_IDS, S.VAL_IDS
    print("fold1 preds: raw", f1_with({}, PB, T), "| +rules(from fold2)", f1_with(dict(rules=rules_A), PB, T), "| +counsel", f1_with(dict(rules=rules_A, counsel=True), PB, T))
    print("fold2 preds: raw", f1_with({}, PA, V), "| +rules(from fold1)", f1_with(dict(rules=rules_B), PA, V), "| +counsel", f1_with(dict(rules=rules_B, counsel=True), PA, V))
    for labs in (["AUTH"], ["AUTH", "WIT"], ["AUTH", "WIT", "JUDGE", "COURT", "APP", "RESP"]):          # only labels that passed diagnostic (c)
        print("propagate", labs, f1_with(dict(propagate=labs), PB, T), f1_with(dict(propagate=labs), PA, V))
    tab_f1 = learn_period_table(S.TEXTS, {d: S.GOLD[d] for d in T}); tab_f2 = learn_period_table(S.TEXTS, {d: S.GOLD[d] for d in V})
    print("period table sizes:", len(tab_f1), len(tab_f2), "| sample:", sorted(tab_f1)[:8])
    for name, p2, p1 in (("raw", {}, {}), ("counsel", dict(counsel=True), dict(counsel=True)), ("counsel+vote", dict(counsel="vote"), dict(counsel="vote")),
                         ("period", dict(period=tab_f1), dict(period=tab_f2)), ("period+counsel+vote", dict(period=tab_f1, counsel="vote"), dict(period=tab_f2, counsel="vote"))):
        print(f"{name:22s} fold2: {f1_with(p2, PA, V)}   fold1: {f1_with(p1, PB, T)}")


def build_post(PA, PB, margin=0.002, save=True):
    """Accept/reject post-processing components automatically and build the final ``POST``.

    A component is kept only if it improves BOTH folds by > ``margin`` (refine only needs 'no harm' on fold1 because its prior saw fold1 gold).
    The final tables are re-learned on ALL labelled docs.

    Args:
        PA (dict): fold2 raw predictions (model trained on fold1).
        PB (dict): fold1 raw predictions (reverse model).
        margin (float): minimum F1 gain per fold.
        save (bool): pickle the result to ``<WORK>/POST.pkl``.

    Returns:
        tuple[dict, list[str]]: (POST dict, names of accepted components).
    """
    T, V = S.TRAIN_IDS, S.VAL_IDS
    rules_A = learn_edit_rules(S.TEXTS, {d: S.GOLD[d] for d in V}, PA, RULE_LABELS)        # learned on fold2
    rules_B = learn_edit_rules(S.TEXTS, {d: S.GOLD[d] for d in T}, PB, RULE_LABELS)        # learned on fold1
    report_crossfit(PA, PB, rules_A, rules_B)
    tab_f1 = learn_period_table(S.TEXTS, {d: S.GOLD[d] for d in T}); tab_f2 = learn_period_table(S.TEXTS, {d: S.GOLD[d] for d in V})
    bl_f1 = learn_fp_blacklist(S.TEXTS, {d: S.GOLD[d] for d in T}, PB); bl_f2 = learn_fp_blacklist(S.TEXTS, {d: S.GOLD[d] for d in V}, PA)
    SAFE = refine_safe_labels(T, {"REFINE_K": 3, "REFINE_LAMBDA": 1.5}); print("refine allowed for:", sorted(SAFE))
    cands = [("edit rules",      dict(rules=rules_B),                       dict(rules=rules_A)),
             ("counsel+vote",    dict(counsel="vote"),                      dict(counsel="vote")),
             ("period",          dict(period=tab_f1),                       dict(period=tab_f2)),
             ("unify APP/RESP",  dict(unify=["APP", "RESP"]),               dict(unify=["APP", "RESP"])),
             ("unify +WIT/AUTH", dict(unify=["APP", "RESP", "WIT", "AUTH"]), dict(unify=["APP", "RESP", "WIT", "AUTH"])),
             ("roles cue",       dict(roles=["cue"]),                       dict(roles=["cue"])),
             ("roles cue+cap",   dict(roles=["cue", "caption"]),            dict(roles=["cue", "caption"])),
             ("caption add",     dict(capspans="add"),                      dict(capspans="add")),
             ("caption prop",    dict(capspans="prop"),                     dict(capspans="prop")),
             ("counsel party",   dict(counsel_party=True),                  dict(counsel_party=True)),
             ("fp blacklist",    dict(blacklist=bl_f1),                     dict(blacklist=bl_f2)),
             ("refine(safe)",    dict(refine=SAFE),                         dict(refine=SAFE))]
    curA, curB, accepted = {}, {}, []
    baseA, baseB = f1_with({}, PA, V), f1_with({}, PB, T); print(f"raw: fold2={baseA} fold1={baseB}")
    for name, pa, pb in cands:
        ta, tb = {**curA, **pa}, {**curB, **pb}
        fa, fb = f1_with(ta, PA, V), f1_with(tb, PB, T); da, db = fa - baseA, fb - baseB
        ok = da > margin and (db > margin or (name.startswith("refine") and db >= -0.001))
        print(f"{name:18s} fold2 {fa} ({da:+.4f})  fold1 {fb} ({db:+.4f})  -> {'KEEP' if ok else 'drop'}")
        if ok: curA, curB, baseA, baseB = ta, tb, fa, fb; accepted.append(name)
    POST = dict(curA)
    if "rules" in POST: POST["rules"] = rules_A + rules_B
    if "period" in POST: POST["period"] = learn_period_table(S.TEXTS, S.GOLD)
    if "blacklist" in POST: POST["blacklist"] = learn_fp_blacklist(S.TEXTS, S.GOLD, {**PA, **PB})
    print("accepted:", accepted); print("POST =", {k: (v if k not in ("rules", "period") else f"<{len(v)} items>") for k, v in POST.items()})
    if save: pickle.dump(POST, open(os.path.join(WORK, "POST.pkl"), "wb"))
    S.POST = POST
    return POST, accepted


def load_post():
    """Load ``POST.pkl`` (written by ``build_post``) into ``state.POST``.

    Args:
        None

    Returns:
        dict: the loaded POST ({} if the file does not exist).
    """
    p = os.path.join(WORK, "POST.pkl")
    S.POST = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    return S.POST
