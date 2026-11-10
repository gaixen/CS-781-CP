# Phase 3 — Rhetorical Role (RR) Prediction in Legal Judgments

Sentence-level labelling of Indian court judgments with **13 rhetorical roles** (Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, None) for two domains: **CL** (Competition Law) and **IT** (Income Tax). Primary metric: **macro-F1** over sentences.
This repository is a modular refactor of the Phase-3 notebook (`p3-rr-recall`); all algorithms come from that notebook.

> **Public GitHub link (add to the final report):** `<paste-your-repo-URL-here>`

## 1. Repository layout
```
rr/
  config.py             Settings, default experiment config CFG, architecture table, stage run lists
  context.py            RRData / Context: shared state (replaces notebook globals)
  utils.py              seeding, device, HF auth
  pipeline.py           orchestration in notebook order
  self_checks.py        CRF vs brute force, windows, metrics, majority baseline
  data/
    loader.py           CL/IT json loading, label vocabulary, inspection (label inertia)
    tokenization.py     GPT-2 tokenisation, +-k neighbour context inputs, windows
  models/
    backbones.py        Phase-1 custom decoder / GPT-2 / T5-encoder wrappers + checkpoint loading
    peft.py             LoRA / rsLoRA / LoRMA / adapters / BitFit / partial / full fine-tuning
    crf.py              linear-chain CRF (NLL, marginals, Viterbi)
    rr_model.py         sentence encoder -> [BiLSTM] -> emissions [-> CRF] [-> label-shift head]
  eval/metrics.py       macro P/R/F1 variants, stickiness
  training/
    trainer.py          windowed training, KD, inference, run registry, checkpoints
    stages.py           stages A (context), B (architectures x techniques), C (KD), domain transfer, recall training
  inference/
    ensemble.py         all-subset ensembles, cross-fit-gated weights / class bias (F1 and recall), final retrains, submissions
    recall.py           label-free recall fixes (prior matching, EM, calibrated Viterbi, blip smoothing), audit, inspector
scripts/run_pipeline.py CLI
tests/test_core.py      unit tests
```

## 2. Model architecture
* **Sentence encoder:** a Phase-1 backbone (`lightweight_sliding_window` made bidirectional, `modern_decoder`, GPT-2 or T5-encoder). Each sentence is encoded with a domain prefix (`[CL]` / `[IT]`) and optionally ±k neighbour sentences; the vector is the mean (or last) over the **target sentence tokens only**.
* **Document head:** projection + LayerNorm, residual **BiLSTM** over a window of sentence vectors, linear emissions, optional **linear-chain CRF**, optional **label-shift** auxiliary head (paper's multi-task setup: does the label change between consecutive sentences?).
* **Windows:** 32-sentence training windows (random phase per epoch); inference encodes every sentence once, runs the head on 64-sentence windows (stride 32) and merges posteriors with an edge-down-weighted mean.
* **Adaptation techniques:** LoRMA (compulsory, default), LoRA, rsLoRA, adapters, BitFit, partial and full fine-tuning; **knowledge distillation** from single and ensemble teachers.

## 3. Methodology and stages
| Stage (`--stages`) | What it does |
|---|---|
| `data` | load, inspect (label inertia P(next==this)≈0.88 in the paper), tokenise, self-checks |
| `a` | context ablation: neighbours-only vs BiLSTM-CRF + shift task, on two backbones |
| `b` | architecture × technique comparison (`B_lsw_lorma`, `B_modern_lora`; optional full FT) |
| `kd` | mandatory KD: window of best models as individual and ensemble teachers (`C_kd_from_*`), optional no-KD control |
| `analysis` | every subset of the best-K window evaluated as an ensemble, cross-fit-gated class bias, `sub_dev_ensemble` |
| `ensemble` | tune weights + F1-oriented / recall-oriented (F-beta) class bias, each gated by cross-fit; writes `sub_ens_f1/recall` |
| `final` / `seed_final` | retrain members on TRAIN+DEV for their best epoch (KD students from retrained teachers; multi-seed) |
| `recall` | audit (sink / recall-limited labels), **label-free** decoders judged by document bootstrap + share guard, qualitative inspector, blip smoothing |
| `recall_train` | training-time recall: logit-adjusted loss, rare-window importance sampling (`R3_mix`) |
| `domain_transfer` | CL-only / IT-only training with cross-domain evaluation |

**Why label-free recall fixes:** dev has ~10 documents, so tuning 13 per-class biases over-fits (cross-fit fell below baseline). Prior matching, Saerens-EM and a stickiness-calibrated Viterbi use only posteriors and training statistics (≤2 free parameters) and are accepted only if the bootstrap gain ≥ 0.5 macro-F1 points in ≥ 70 % of resamples.

## 4. Submissions
Each submission folder under `<work>/` (`sub_dev_ensemble`, `sub_ens_f1`, `sub_ens_recall`, `sub_final_ens_*`, `sub_final_ensemble`, `sub_final_recall`, ...) contains `prediction.json` (`{doc_id: [label per sentence]}`) and `submission.zip`; `write_submission` validates ids, sentence counts and labels, and prints predicted vs training label shares.

## 5. Setup & run
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --work ./work --data-root /path/to/Phase_3_RR \
   --ckpt lightweight_sliding_window=/path/to/lsw/model-final --ckpt modern_decoder=/path/to/modern/model-final \
   --stages data a b kd analysis
# later: tune the combiner / final fits / label-free recall decoding (checkpoints are reused from <work>/*.pt)
python scripts/run_pipeline.py --work ./work --stages ensemble final recall --members A4_bilstm_crf_shift C_kd_from_A4_bilstm_crf_shift
```
Without `--data-root` the files are downloaded from HuggingFace (`Exploration-Lab/CS-781-Capstone`, set `HF_TOKEN`). Existing checkpoints (`<name>.pt`, trainable tensors only) are reused unless `--force-retrain`. Defaults (accumulation 8, 4 epochs, LoRMA r=64, α=128, KD T=2) are in `rr/config.py`; the notebook used one Kaggle GPU.

## 6. Notes
* Phase-1 checkpoints must have been trained with the GPT-2 tokenizer (a vocabulary mismatch would silently misalign embeddings); `config.json` next to a custom-decoder checkpoint is used to recover `n_head` exactly.
* Two notebook bugs were fixed in the refactor: `compare_decoders` passed a float (`crossfit_bias` output) as a bias vector, and `predict_viterbi_for_docs` referenced a non-existent `cfg["WINDOW"]` key (unused line removed).
* `C_kd_from_ENS2`'s teachers cannot be recovered from its checkpoint; set them with `Settings.ens2_teachers`.
* Code of other course phases was not part of the supplied notebook.
