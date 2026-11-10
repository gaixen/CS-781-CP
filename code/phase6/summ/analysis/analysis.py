"""Sections 8-10: experiment comparison, long-document strategy / generation-length sweeps, optional re-generation and submission validation."""
import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd

from summ.config import doc_budget
from summ.data.inputs import build_input, inputs
from summ.eval.metrics import bertscore, diag, rouge
from summ.models.summ_models import build_model
from summ.training.trainer import exp_dir, gen_texts, hier_inputs, load_best, write_submission


def compare_experiments(ctx, save=True):
    """Table of quality, efficiency and redundancy of all finished experiments (+ plot).

    Args:
        ctx (Context): Context.
        save (bool): Write ``comparison.csv`` / ``comparison.png`` to the submission folder.

    Returns:
        pd.DataFrame: Comparison indexed by experiment id.
    """
    import matplotlib.pyplot as plt
    rows = [json.load(open(f"{exp_dir(ctx, e)}/result.json")) for e in ctx.experiments if os.path.exists(f"{exp_dir(ctx, e)}/result.json")]
    cmp_ = pd.DataFrame(rows).set_index("exp")[["name", "arch", "method", "bertscore_f1", "rouge1_f", "rougeL_f", "best_dev_loss", "words", "repeat_trigram", "copy_bigram",
                                               "trainable_M", "total_M", "fp16_MB", "train_min", "peak_gpu_GB", "infer_s_per_doc"]]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(cmp_.round(4).to_string())
    print("\nreference words/summary (dev):", round(float(np.mean([len(r.split()) for r in ctx.dev["ref"]])), 1))
    if save:
        cmp_.to_csv(f"{ctx.s.submit}/comparison.csv")
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
        cmp_[["bertscore_f1"]].plot.bar(ax=ax[0], title="BERTScore F1")
        cmp_[["rouge1_f", "rougeL_f"]].plot.bar(ax=ax[1], title="ROUGE F1")
        cmp_[["train_min", "infer_s_per_doc"]].plot.bar(ax=ax[2], title="cost (train min / s per doc)", logy=True)
        plt.tight_layout()
        fig.savefig(f"{ctx.s.submit}/comparison.png", dpi=120)
        plt.close(fig)
    for e in cmp_.index[:4]:
        p = json.load(open(f"{exp_dir(ctx, e)}/dev_preds.json"))["preds"]
        print(f"\n[{e}] {cmp_.loc[e, 'name']}\n  PRED: {p[0][:600]}\n  REF : {ctx.dev['ref'][0][:600]}")
    return cmp_


def long_doc_and_length_sweeps(ctx, cmp_):
    """Section 9: compare ``headtail`` / ``select`` / ``hier`` at inference and sweep generation length / repetition controls on the best experiment.

    Args:
        ctx (Context): Context.
        cmp_ (pd.DataFrame): Output of :func:`compare_experiments`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: ``(long-document strategies, generation sweep)``.
    """
    s = ctx.s
    best = cmp_["bertscore_f1"].fillna(cmp_["rouge1_f"]).idxmax()
    print("best experiment:", best)
    cfg = ctx.experiments[best]
    bm = load_best(ctx, build_model(ctx, cfg), best).eval()
    n, bud = s.dev_eval_n, doc_budget(s, cfg)

    def score_dev(preds):
        """ROUGE, BERTScore and diagnostics on the dev subset.

        Args:
            preds (List[str]): Predictions.

        Returns:
            dict: ``rouge1, rougeL, bertscore`` + diagnostics (without ``empty``).
        """
        r1, rl = rouge(preds, ctx.dev["ref"][:n])
        return dict(rouge1=r1, rougeL=rl, bertscore=bertscore(preds, ctx.dev["ref"][:n], ctx.device), **{k: v for k, v in diag(preds, ctx.dev["doc"][:n]).items() if k != "empty"})

    ld = [dict(strategy=st, **score_dev(gen_texts(ctx, bm, inputs(ctx, "dev", st, bud)[:n]))) for st in ("headtail", "select")]
    ld.append(dict(strategy="hier", **score_dev(gen_texts(ctx, bm, [build_input(ctx, x, "headtail", bud) for x in hier_inputs(ctx, bm, ctx.dev["doc_ids"][:n])]))))
    long_mask = np.array([len(x) > s.doc_tok for x in ctx.dev["doc_ids"][:n]])
    print(f"{long_mask.sum()}/{n} dev docs exceed doc_tok={s.doc_tok} (only those can differ between strategies)")
    LD = pd.DataFrame(ld).set_index("strategy").round(4)
    print(LD.to_string())
    sw = []
    base_in = inputs(ctx, "dev", cfg["strat"], bud)[:n]
    for mx, mn in ((64, 20), (100, 40), (150, 60), (200, 100), (250, 150)):
        if mx > s.sum_tok - 1:
            continue
        sw.append(dict(max_new=mx, min_new=mn, nrep=3, rep_pen=1.1, **score_dev(gen_texts(ctx, bm, base_in, dict(max_new=mx, min_new=mn)))))
    for nr, rp in ((0, 1.0), (3, 1.0), (4, 1.2)):
        sw.append(dict(max_new=s.gen["max_new"], min_new=s.gen["min_new"], nrep=nr, rep_pen=rp, **score_dev(gen_texts(ctx, bm, base_in, dict(nrep=nr, rep_pen=rp)))))
    SW = pd.DataFrame(sw).round(4)
    print(SW.to_string())
    return LD, SW


def regenerate_overrides(ctx, override):
    """Re-generate TEST predictions of experiments with a better strategy / decoding found in the sweeps.

    Args:
        ctx (Context): Context.
        override (dict): ``eid -> {"strat": ..., "gen": {...}}`` (``strat="hier"`` supported).

    Returns:
        List[str]: Written zip paths.
    """
    out = []
    for eid, o in override.items():
        cfg = ctx.experiments[eid]
        m = load_best(ctx, build_model(ctx, cfg), eid).eval()
        st = o.get("strat", cfg["strat"])
        bud = doc_budget(ctx.s, cfg)
        ids = [build_input(ctx, x, "headtail", bud) for x in hier_inputs(ctx, m, ctx.te["doc_ids"])] if st == "hier" else inputs(ctx, "test", st, bud)
        out.append(write_submission(ctx, f"{eid}-{cfg['name']}_{cfg['arch']}_{st}_tuned", gen_texts(ctx, m, ids, o.get("gen"))))
        del m
    return out


def validate_submissions(ctx):
    """Check every zip: ids and order equal the test set, summaries are non-empty strings.

    Args:
        ctx (Context): Context.

    Returns:
        None
    """
    tid_order = [str(i) for i in ctx.te["rid"]]
    for zp in sorted(glob.glob(f"{ctx.s.submit}/*.zip")):
        with zipfile.ZipFile(zp) as z:
            pr = json.loads(z.read("prediction.json"))
        assert [str(x["id"]) for x in pr] == tid_order and all(isinstance(x["summary"], str) and x["summary"].strip() for x in pr), zp
        print("OK", os.path.basename(zp), "| docs:", len(pr), "| mean words:", round(float(np.mean([len(x["summary"].split()) for x in pr])), 1))
    print("Submit the four experiment zips (E1-E4) -- one per different Phase-1 model/technique; each contains prediction.json in list format.")
