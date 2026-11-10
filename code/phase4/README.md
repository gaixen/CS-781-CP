# Phase 4 -- Court Judgment Prediction (CJPE)

Binary prediction of a court judgment's outcome (label 0 / 1) from the full text of an Indian court case, using the
**Phase-1 pre-trained models** as backbones and the adaptation techniques studied in this phase.
Metric: **macro-F1** (dataset: `Exploration-Lab/CS-781-Capstone`, `Phase_4_CJPE/{Train/train.json, Train/dev.json, Test/test.json}`).

## 1. Model architecture

```
case text ──► tokenizer (Phase-1) ──► token selection / chunking ──► Phase-1 backbone ──► token pooling
                                                                                              │
                                   flat representation ───────────────────────────────────────┤
                                   hier: chunk vectors ──► doc encoder (mean|attn|transformer|gru) ──┤
                                                                                              ▼
                                                                          (deep residual) MLP head ──► 2 logits
```

* **Backbones** (`cjpe/models/backbone.py`, `custom_decoder.py`) -- loaded from Phase-1 checkpoints with identical parameter names:
  `lightweight_sliding_window` (causal, window 256), `lightweight_sliding_window_bidir` (symmetric window 256),
  `modern_decoder` (full causal attention) -- all RMSNorm + RoPE + GQA + SwiGLU; `gpt2_decoder` (GPT-2 stack);
  `seq2seq_t5` (T5 **encoder** only). `Backbone` gives a uniform `forward(ids, mask) -> (B, T, D)`.
* **Token pooling** (`models/classifier.py::TokenPool`): `last | mean | attn | mean_last`.
* **Hierarchical document encoder** (`DocEncoder`): overlapping chunks counted back from the *end* of the case
  (the verdict is at the end), encoded independently, then aggregated by `mean | attn | transformer | gru`.
* **Head** (`MLPHead`): LayerNorm-Linear-GELU-Dropout-Linear; `HEAD_LAYERS>1` adds residual MLP blocks.

## 2. Methodology

| Technique (`method`) | What is trained | Where |
|---|---|---|
| `full` | all weights (optional layer-wise LR decay, `LLRD_DECAY<1`) | `training/trainer.py::build_param_groups` |
| `lorma` (**LoRMA+**, compulsory) | multiplicative low-rank update `h = ((alpha/r)BA + I)(W0 x)`, `B` zero-init; targets `qkv / attn / attn_ffn` | `models/peft.py::LoRMAPlusLinear` |
| `bitfit` | bias terms only (zero-init biases are *added* because the Phase-1 decoder/T5 have none) | `enable_bitfit` |
| `adapters` | bottleneck adapter after every FFN | `inject_adapters` |
| `partial` / `gradual` | top-k layers / one more layer unfrozen per epoch | `set_trainable_top_k` |
| `distill` (**KD**, mandatory) | student = layer-pruned Phase-1 model (`KEEP_EVERY=2`), teacher = a fine-tuned experiment | `prepare_teacher`, `compute_loss` |

**Knowledge distillation loss:** `L = (1-α)·CE + α·T²·KL(softmax(z_s/T) ‖ softmax(z_t/T)) + β·(1 − cos(W·f_s, f_t))`
(logit KD + pooled-feature KD, `KD_ALPHA=0.5, KD_T=2, KD_BETA=0.5`). Teacher logits/features on the train set are computed once and cached.

**Document representation ablations:** `last_n` (default, the end of the judgment), `first_n`, `head_tail`, and `hier`.

**Training recipe:** AdamW with 3 LR tiers (head `1e-3`, new PEFT params `6e-4`, Phase-1 weights `3e-5`), 6 % warm-up + cosine decay,
gradient clipping 1.0, fp16 autocast (off for T5), class-weighted CE for the imbalanced labels, length-bucketed batches,
gradient checkpointing for `hier`, best epoch chosen by dev macro-F1, and **decision-threshold calibration** (threshold in [0.20, 0.80] maximising dev macro-F1).

## 3. Implementation details

* Documents are **tokenised once** and cached to disk (`cjpe/data/tokenization.py`): first 1024 + last 4096 tokens per case
  (very long cases only tokenise their head/tail character slices).
* Fixed seed (`SEED=42`). Every run saves `prediction.json`, `submission.zip` (format: `[{"id": ..., "pred": 0|1}, ...]`), `best.pt`,
  dev predictions, and a row in `results.jsonl` (macro-F1/P/R, #params, trainable %, peak GPU memory, train time, inference docs/s).
* `analysis/results.py` produces the comparison tables, FP/FN error analysis (confusion matrix, accuracy by document length) and size/cost table.
* All public functions/classes carry docstrings in `Args: / Returns:` format.

## 4. Experiments (`cjpe/config.py::EXPERIMENTS`)

* **Required submissions:** `E1_lsw_partial`, `E1a_kd_lsw_student`, `E3_bidir_lorma`, `E4_kd_lsw_student` (+ `E2_modern_lorma_hier`, `E5_t5_bitfit`, `E6_gpt2_partial` when checkpoints/time allow).
* **Ablations (`A_*`):** representation (`first_n`, `head_tail`, 512 tokens), hierarchical aggregators (`mean/attn/transformer/gru`),
  pooling, head depth, class weighting, LoRMA+ targets, and `full / bitfit / adapters / partial / gradual`.

## 5. Setup & how to run

```bash
git clone https://github.com/<your-username>/<your-repo>.git && cd <your-repo>/phase4
pip install -r requirements.txt
export HF_TOKEN=hf_xxx                       # access to the dataset (or add a Kaggle secret named HF_TOKEN)
export CJPE_WORK=./outputs                   # outputs/cache directory (default ./outputs)
```

**Point to your Phase-1 checkpoints** (directories holding `model.safetensors`/`pytorch_model.bin`, `config.json`, tokenizer files) --
either edit `PHASE1_CKPT` in `cjpe/config.py` or provide a JSON file:

```bash
echo '{"lightweight_sliding_window": "/path/to/lsw/model-final", "modern_decoder": "/path/to/modern/model-final"}' > ckpts.json
export PHASE1_CKPT_JSON=ckpts.json
```

```bash
# 1) smoke test (2000 train docs, 500 dev docs, 1 epoch)
python scripts/run_experiments.py --names E3_bidir_lorma --smoke

# 2) the four submissions (KD experiments train their teacher first if best.pt is missing)
python scripts/run_experiments.py --names E1_lsw_partial E1a_kd_lsw_student E3_bidir_lorma E4_kd_lsw_student

# 3) ablations / overrides
python scripts/run_experiments.py --names A_hier_gru A_pool_attn
python scripts/run_experiments.py --names A_full --override EPOCHS=2 BATCH_SIZE=4

# 4) tables + error analysis
python scripts/show_results.py --errors E3_bidir_lorma E4_kd_lsw_student

# 5) validate + zip submissions -> $CJPE_WORK/submissions/submission_<i>_<name>.zip
python scripts/package_submissions.py E1_lsw_partial E1a_kd_lsw_student E3_bidir_lorma E4_kd_lsw_student
```

Use as a library (e.g. in a notebook):

```python
from cjpe.pipeline import run_experiment
from cjpe.analysis.results import show_results, error_analysis, sizes_table
run_experiment("E3_bidir_lorma", TRAIN_SUBSET=2000, DEV_SUBSET=500, EPOCHS=1)
show_results()
```

Unit tests: `pytest tests/`.

Hardware used: a single 16 GB GPU (Kaggle T4); the `HIER` preset (`BATCH_SIZE=2, GRAD_ACCUM=4, GRAD_CKPT=True`) is tuned for it.

## 6. Results

_Fill in from `scripts/show_results.py` (dev macro-F1, params, trainable %, memory, time, throughput) and add your error-analysis findings._
