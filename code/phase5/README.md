# Phase 5 — Legal Statute Identification (LSI): post-mortem fixes

Multi-label prediction of the **statutes cited by a legal case** from its full text. Primary metric: **Macro-F1** (also Micro-F1, subset accuracy).
This repository is a modular refactor of the Phase-5 post-fix notebook (`p5-postfix`); all algorithms come from that notebook.

> **Public GitHub link (add to the final report):** `<paste-your-repo-URL-here>`

## 1. Repository layout
```
lsi/
  config.py             Config dataclass + experiment registry E1–E5
  context.py            LSIData / Context: shared state (replaces notebook globals)
  utils.py              seeding, device, HF auth, restore_runs (reuse earlier runs)
  pipeline.py           orchestration in notebook order
  models/
    custom_decoder.py   Phase-1 decoder (RMSNorm + RoPE + GQA + SwiGLU)
    phase1_backbone.py  checkpoint loading; LoRMA+, adapters, partial FT, layer pruning
    lsi_model.py        chunk pooling -> label-wise attention (LWAN) head, losses, optimiser groups
  data/
    loader.py           loading, dev leak / duplicate cleaning, label matrices
    chunking.py         tokenisation cache, overlapping chunks, context selection, evidence lexicon
  eval/metrics.py       metrics, calibration, threshold strategies, greedy ensemble weights
  training/
    engine.py           resumable training / inference, KD, experiment runner
    submission.py       zip naming / writing / validation, honest threshold selection
    ensemble.py         nested-validated greedy ensemble, E4 seed ensemble
  analysis/
    diagnostics.py      dev vs test diagnostics, calibration
    audits.py           Audit 1 (data), 2 (predictions), 3 (citations), 4 (zips)
scripts/run_pipeline.py CLI
tests/                  unit tests for the numpy metrics
```

## 2. Model architecture
* **Backbones (Phase 1):** `lightweight_sliding_window_bidir` (E1), `modern_decoder` (E2), `lightweight_sliding_window` (E3); the KD student is a layer-pruned (every 2nd layer) Phase-1 model.
* **Long-document encoder:** the text is tokenised (head 75 % + tail 25 % above 12 288 tokens), split into 512-token chunks with 64 overlap; at most 4 chunks per document in training and 16 at inference (evenly spaced). Each chunk vector = `[mean ; last-token]` of the final hidden states.
* **Head:** label-wise attention network (one learned query per statute attends over the chunks), bias initialised to the log-prior of every statute.

| Exp | Backbone | Technique | Loss |
|---|---|---|---|
| E1 | lsw_bidir | full fine-tuning, layer-wise LR decay | ASL |
| E2 | modern_decoder | **LoRMA+** r=16 (attention + FFN) | BCE (pos-weight) |
| E3 | lsw | bottleneck adapters r=64 | ASL |
| E4 | lsw_bidir (50 % layers) | knowledge distillation from E1+E2+E3 | BCE + KD (T=2) |
| E5 (optional) | same student | KD generation 2, warm-started from E4, teachers E1+E3+E4 | BCE + KD |

## 3. Methodology (what the post-mortem changes)
1. **Rare-label robustness:** per-label thresholds are bootstrap-bagged and shrunk toward the global threshold by `n_pos/(n_pos+k)` (strategies `g0.50, global, global+top1, pl8, pl30, bagpl30, bagpl30+top1`).
2. **Honest selection:** every strategy is scored with a fixed-seed 2-fold split inside dev (3 repeats); the simplest strategy within 0.003 of the best wins.
3. **Weighted ensemble:** greedy forward selection (with replacement) of temperature-calibrated logits, validated with a nested split.
4. **KD generation 2 / seed ensemble:** E5 and the optional multi-seed E4 ensemble.
5. **Leak control:** dev rows that also appear in test, train or twice in dev are dropped; if too few remain, 10 % of train is held out. Test labels are never used. kNN / citation / logit-adjustment branches of the previous notebook were removed.
6. **Audits (read-only):** data structure and duplicates (1), prediction ceilings / per-quartile quality / dev-vs-test shift (2), citation rules (3), submission zips (4).

## 4. Submissions
Zips are written to `<work>/submissions_fix/` with the distinguishing information first, e.g. `E4-KD_bagPL30top1_lswBidir_LWAN_BCE_doc-full.zip`, each containing `prediction.json` (`[{"id", "pred": [statute ids]}]` in test order). For `panel_exps` (default E4) every threshold strategy is written so the leaderboard shows which one transfers; `ENSW-*` is the greedy ensemble; `E4x{n}seeds_*` the seed ensemble; `submission_plan.csv` summarises all zips.

## 5. Setup & run
```bash
pip install -r requirements.txt
# Post-processing only (CPU, minutes): reuse the logits of earlier runs
python scripts/run_pipeline.py --work ./work --prev-dataset /path/containing/lsi_runs \
   --ckpt lightweight_sliding_window_bidir=/path/model-final     # --local-data paths.json to skip the HF download
# Train KD generation 2 (GPU, resumable per epoch)
python scripts/run_pipeline.py --train E5 --prev-dataset ... --ckpt ... --ckpt ...
# Optional multi-seed E4 ensemble (trains seeds not on disk)
python scripts/run_pipeline.py --stages restore seed_ensemble --kd-seeds 42 11
```
Flags: `--smoke`, `--stages restore train diagnostics audits ensemble final seed_ensemble`, `--local-data` (`{"train":..., "dev":..., "test":...}`), `--no-strict-load`.
Data are downloaded from HuggingFace (`Exploration-Lab/CS-781-Capstone`, set `HF_TOKEN`) unless `--local-data` is given. Defaults (batch 4×4 accumulation, AMP, gradient checkpointing, ≤150 min per experiment) are in `lsi/config.py`; the notebook used a single Kaggle T4.

## 6. Notes
* Logits are matched to the current dev / test order by a shape check; mismatching runs are excluded automatically.
* Phase-1 modules are included unchanged. Code of other course phases was not part of the supplied notebook.
