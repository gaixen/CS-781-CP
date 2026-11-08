# Phase 7 — Prior Case Retrieval (PCR) from Phase-1 Legal Language Models

Given a legal case (query), retrieve the **top-7 prior cases** from its split's candidate pool. Primary metric: **Micro-F1@7** (also P@7, R@7).
This repository is a modular refactor of the Phase-7 notebook (`p7-starter`); all algorithms come from that notebook.

> **Public GitHub link (add to the final report):** `<paste-your-repo-URL-here>`

## 1. Repository layout
```
pcr/
  config.py            Config dataclass (all hyper-parameters) + experiment registry E0–E5
  context.py           PCRContext: shared state passed between stages
  utils.py             seeding, device, z-score helpers
  submission.py        write / validate prediction.json zips
  pipeline.py          end-to-end orchestration (same order as the notebook)
  models/
    custom_decoder.py  Phase-1 decoder (RMSNorm + RoPE + GQA + SwiGLU, sliding-window / bidirectional)
    phase1_backbone.py checkpoint loading + LoRMA+, adapters, BitFit, partial FT, layer pruning
    dual_encoder.py    chunked document encoder, pooling, scoring (cosine + MaxSim)
  data/                loader, preprocessing (header stripping), audit, tokenisation
  retrieval/           metrics, BM25/TF-IDF/n-gram lexical suite, rule features
  training/            InfoNCE + hard-negative / KD trainer, experiment runner
  rerank/              fusion, ensemble, LTR (v1, tuned, v3–v6), two-stage A/B/C
  analysis/            comparison, ablation, failure analysis, audits
scripts/run_pipeline.py   CLI
tests/                    basic unit tests
```

## 2. Model architecture
* **Backbones (Phase 1):** `modern_decoder` (full causal attention), `lightweight_sliding_window` (window 256), `lightweight_sliding_window_bidir`. Blocks use RMSNorm, RoPE, grouped-query attention and SwiGLU.
* **Document encoder (bi-encoder):** each case is split into 512-token chunks (up to 2 in training, 8 at final eval, chosen *spread* over the document); each chunk is mean-pooled over tokens, chunk vectors are mean-aggregated and L2-normalised. Score = cosine similarity (also MaxSim over chunks).

| Exp | Backbone | Technique |
|---|---|---|
| E0 | modern_decoder | frozen zero-shot baseline |
| E1 | modern_decoder | full fine-tuning (also KD teacher) |
| E2 | lightweight_sliding_window | **LoRMA+** (compulsory), frozen backbone |
| E3 | lightweight_sliding_window_bidir | bottleneck adapters |
| E4 | lightweight_sliding_window, 50 % layers pruned | knowledge distillation from E1 |
| E5 (opt.) | modern_decoder | partial (top-4 layer) fine-tuning |

## 3. Methodology
1. **Preprocessing (deterministic):** parse year / citation keys / party names from the case header, then strip the PETITIONER…CITATOR INFO boilerplate and `<ORG>`-style placeholders.
2. **Lexical baselines:** BM25 (hard-negative miner) and TF-IDF variants.
3. **Training:** symmetric InfoNCE with in-batch + BM25 hard negatives (ranks 3–40, gold excluded, false negatives masked), AdamW, warm-up + cosine, fp16, gradient checkpointing; best epoch kept by dev F1@7. KD loss = InfoNCE + T²·KL(teacher‖student) + MSE on similarities.
4. **Post-hoc improvements:** score fusion, E1–E4 ensemble, learning-to-rank rerankers (LR / HGB / LambdaRank) over dense, MaxSim, lexical, chronology, citation and case-name features, CSLS / dual-softmax hubness correction and rules.

## 4. Phases / submissions
Each experiment and reranker writes a zip with `prediction.json` (7 unique pool ids per test query) to `<work>/p7v2_submissions/`:

| Submission | Produced by |
|---|---|
| `E0`–`E5` | `pcr.training.experiments` |
| `FUSION_*` | `pcr.rerank.fusion.fuse_experiments` |
| `ENSEMBLE_*` | `pcr.rerank.fusion.ensemble_experiments` |
| `LTR_rerank_*`, `LTR_tuned_*` | `pcr.rerank.ltr_dev` (v1, tuner) |
| `LTRv3_*`, `LTRv4_*` | `pcr.rerank.ltr_dev` |
| `LTRv5_*`, `LTRv6_prf_*` | `pcr.rerank.ltr_train` |
| `LTRv7_AB_*`, `LTRv8_stageC_*` | `pcr.rerank.ltr_two_stage` |

Phase-1 modules (`custom_decoder`, `phase1_backbone`) are included unchanged so Phase-1 checkpoints load directly. Code of other course phases was not part of the supplied notebook; add it under new folders if your submission requires it.

## 5. Setup & run
```bash
pip install -r requirements.txt
# data from HuggingFace (Exploration-Lab/CS-781-Capstone) – set HF_TOKEN – or use --local-data paths.json
python scripts/run_pipeline.py --work ./work \
  --ckpt modern_decoder=/path/to/modern_decoder/model-final \
  --ckpt lightweight_sliding_window=/path/to/lsw/model-final \
  --ckpt lightweight_sliding_window_bidir=/path/to/lsw_bidir/model-final
```
Useful flags: `--smoke` (tiny test run), `--stages data experiments analysis ltr ltr_train audits`, `--run E0 E1 E2 E3 E4 [E5]`, `--force`, `--local-data`.
`paths.json`: `{"train": ..., "train_c": ..., "dev": ..., "dev_c": ..., "test": ..., "test_c": ...}`.
Runs are resumable (finished experiments load from `result.json`); E1 must finish before E4. Default hyper-parameters are in `pcr/config.py` (e.g. τ=0.05, batch 6×2 accumulation, lr 5e-5 for E1, LoRMA+ r=α=128).
Hardware used in the notebook: single Kaggle T4 GPU, ≤80 min per experiment.

## 6. Notes
* Each split is scored only against its own candidate pool; query ids are removed from the pool.
* Reranker hyper-parameters are tuned on dev, so reported dev scores of tuned rerankers are optimistic; CV estimates are printed.
