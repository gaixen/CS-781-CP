# Phase 6 — Abstractive Legal Summarisation with Phase-1 Language Models

End-to-end summarisation of long Indian court judgments using the decoder-style Phase-1 models. Primary metric: **BERTScore-F1** (also ROUGE-1 / ROUGE-L F1). Output: `[{"id", "summary"}]` in `prediction.json`.
This repository is a modular refactor of the Phase-6 notebook (`p6-starter`); all algorithms come from that notebook.

> **Public GitHub link (add to the final report):** `<paste-your-repo-URL-here>`

## 1. Repository layout
```
summ/
  config.py             Settings, experiment registry E1–E11, ensemble specs
  context.py            Context: shared state (replaces notebook globals)
  utils.py              seeding, device, HF auth
  pipeline.py           orchestration in notebook order
  models/
    custom_decoder.py   Phase-1 decoder (RMSNorm + RoPE + GQA + SwiGLU)
    phase1_backbone.py  checkpoint loading; LoRMA+, adapters, layer pruning
    summ_models.py      prefix-LM summarisers: SummLM (KV-cached), SummGPT2, SummT5; ensemble-capable decoding
  data/
    loader.py           loading, GPT-2 tokenisation cache, cleaning, dev split, data-driven generation lengths
    audit.py            sections 4.1–4.4 (integrity, lengths, positional coverage, abstractiveness, baselines)
    inputs.py           long-document strategies (headtail / select), collation, batching
  eval/metrics.py       ROUGE, BERTScore, diagnostics, post-processing
  training/
    trainer.py          CE / KD training, generation, hierarchical inputs, submissions, experiment runner
    ensemble.py         log-prob ensembles, student distillation + seed ensembles
  analysis/analysis.py  comparison, long-document / length sweeps, validation
scripts/run_pipeline.py CLI
tests/test_core.py      unit tests
```

## 2. Model architecture
Every Phase-1 model is a decoder-style LM, so summarisation is a **prefix-LM**: `[document tokens] <SEP=EOS> [summary tokens] <EOS>` with the loss on the summary only. Document tokens keep each architecture's own attention pattern (causal, sliding-window or bidirectional); summary tokens attend to **all** document tokens plus causally to the summary. Documents are left-padded, generation uses a KV cache (greedy, repetition penalty, no-repeat 3-gram, min/max length from the data audit).
GPT-2 (learned absolute positions → document budget `n_positions − sum_tok − 8`) and a true encoder-decoder **T5** (document → encoder, summary → decoder) share the same interface.

| Exp | Phase-1 model | Technique |
|---|---|---|
| E1 | modern_decoder | full fine-tuning (KD teacher) |
| E2 | lightweight_sliding_window | **LoRMA+** (compulsory), frozen backbone |
| E3 | lightweight_sliding_window_bidir | bottleneck adapters |
| E4 | lightweight_sliding_window, 50 % layers | logit knowledge distillation from E1 (T=2) |
| E5 | modern_decoder, random init | ablation: do Phase-1 weights help? |
| E6–E9 | GPT-2 / T5 | LoRMA+ and full fine-tuning |
| E10–E11 | modern_decoder / lsw | LoRMA+ (E11 with `select` long-document strategy) |

## 3. Methodology
* **Data audit** drives the settings: summary-length percentiles set `max_new` (p90) and `min_new` (½ median); positional coverage of the reference summary decides between `headtail`, `select` and `hier`; trivial LEAD / TAIL baselines must be beaten. Train documents that also appear in test and duplicate pairs are removed; 300 train documents are held out as dev.
* **Long documents:** `headtail` (70 % head + 30 % tail of the token budget), `select` (64-token blocks scored by IDF-weighted overlap with the document centroid, head/tail always kept) at train and test time; `hier` (chunk summaries → final summary) at inference.
* **Ensembles:** weighted average of member **log-probabilities** during decoding; members may mix architectures and context budgets.
* **Distillation stage (`distill`):** E1+E2 logit ensemble as teacher, unpruned and pruned students trained on TRAIN+DEV with 3 seeds, then a logit-average seed ensemble per student variant (no held-out set in this stage).

## 4. Submissions
Zips in `<work>/p6_submissions/`: `E*-<name>_<arch>_<strat>.zip`, `ENS-<name>_<members>_logprob-avg.zip`, `E1_E2_logit_ensemble.zip`, `student_{unpruned,pruned}_3seed_ensemble.zip`, each with `prediction.json` (list format); `comparison*.csv/png` summarise quality and cost. `validate_submissions` checks ids, order and non-empty text.

## 5. Setup & run
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --work ./work \
   --ckpt modern_decoder=/path/model-final --ckpt lightweight_sliding_window=/path/model-final \
   --stages data experiments ensembles analysis final --run E1 E2 E10 E11
python scripts/run_pipeline.py --stages data distill        # optional student distillation (needs E1/E2 best.pt)
```
Data come from HuggingFace (`Exploration-Lab/CS-781-Capstone`, set `HF_TOKEN`) unless `--local-data paths.json` (`{"train": ..., "test": ...}`) is given. Runs are resumable (`result.json` caches). Experiments without a Phase-1 checkpoint are skipped; `--smoke` runs a tiny end-to-end check. Defaults (doc 1536 + summary 256 tokens, batch 4×4, ≤ 90 min per experiment, one Kaggle GPU) are in `summ/config.py`.

## 6. Notes
* The T5 checkpoint must contain **decoder** weights; otherwise loading fails unless `allow_t5_random_decoder=True`.
* Phase-1 modules are included unchanged. Code of other course phases was not part of the supplied notebook.
