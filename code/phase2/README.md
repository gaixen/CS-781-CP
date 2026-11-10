# Phase 2 -- Legal Named-Entity Recognition (L-NER)

Character-level span prediction of 12 legal entity types (`APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO`)
in Indian court judgments, using the **Phase-1 pre-trained models** as backbones. Metric: **strict macro-F1** -- a prediction counts only if
`(doc, start, end, label)` matches the gold span exactly (dataset: `Exploration-Lab/CS-781-Capstone`, `Phase_2_LNER`: `fold1`, `fold2` labelled; `fold3` = test).

## 1. Model architecture

```
text ─► GPT-2 tokenizer (punctuation pre-split) ─► sliding windows (MAX_LEN/OVERLAP, optional doc-head prefix)
     ─► Phase-1 backbone ─► head ─► per-token probabilities ─► window merge ─► decoder ─► spans ─► (post-processing) ─► prediction.json
```

* **Backbones** (`lner/models/`): `modern_decoder` (full attention), `lightweight_sliding_window` (window 256) -- RMSNorm + RoPE + GQA + SwiGLU, loaded with
  identical parameter names, sizes inferred from `config.json`; an `ATTN_MODE=bidirectional` switch lets the causally pre-trained decoder see right context.
  `gpt2_decoder` (HF GPT-2) and `seq2seq_t5` (T5 **encoder** only) are supported too.
* **Heads** (`models/heads.py`): `TASK="bio"` -> BiLSTM + residual MLP -> 25 BIO tags (optionally + **linear-chain CRF**, `USE_CRF=True`, hard BIO constraints,
  partial-label training, marginals fed to the normal decoder); `TASK="span"` -> shared trunk + type / start / end heads.
* **Adaptation** (`models/peft.py`, hand-written): `full | partial | bitfit | lora | lorma | adapter`. **LoRMA** is the multiplicative update
  `W = (I + (α/r)·P·Q)·W0` implemented as `y = y0 + (α/r)·P(Q(y0))`, `P` zero-initialised.

## 2. Methodology

| Component | Where | Notes |
|---|---|---|
| Offset convention detection, gold -> BIO alignment | `data/io.py`, `data/alignment.py` | end-inclusive vs exclusive auto-detected from the stored entity text; **oracle round-trip** gives the tokenisation ceiling (~0.986) |
| Punctuation pre-split tokenisation | `data/alignment.py` | stops BPE merging two different punctuation chars so entity edges never fall inside a token |
| Windows + class weights | `data/samples.py` | edge tokens of inner windows ignored, negative windows subsampled, `power`/`effective` weights, `LABEL_W`, `OVERSAMPLE`, `CTX_PREFIX` |
| Losses | `training/losses.py` | weighted CE, focal, span (type CE + start/end BCE), KD (KL at temperature T) |
| Decoding | `decoding/` | greedy / BIO-Viterbi, span-candidate search, label-specific trailing-punctuation trimming, bracket balancing, boundary refinement, per-class confidence thresholds with **cross-fitted** F1 |
| Post-processing (rules) | `postprocess/` | counsel side + voting, period attachment, role (APP/RESP) cue & caption rules, caption parser, label unification, FP blacklist, learned edit rules |
| Cross-fit gate | `postprocess/crossfit.py` | components learned on one fold's gold are tested on the other fold's out-of-fold predictions; **kept only if they help both folds** |
| Training-time enforcement | `experiments.py` (T1-T3, B0/B1) | context prefix, role loss weights/oversampling, CRF -- move the rule behaviour into the weights |
| Ensembling & pseudo-labels | `ensemble/` | probability averaging across seeds/architectures; rule-distilled pseudo-labels on confident tokens |
| Knowledge distillation | `experiments.py::run_distillation / run_pruned_kd` | student (optionally layer-pruned, every 2nd layer) trained with `α·CE + (1-α)·T²·KL(teacher‖student)` |
| Submission | `submission.py` | writes `prediction.json` in the gold convention, verifies structure/density, label-free **counsel guard**, zips `submission.zip` |

## 3. Implementation details

* **Shared state:** the dataset (labels, texts, gold, tokenised docs, splits, learned priors, active `POST`) lives in `lner/state.py` and is filled by
  `lner.data.dataset.init_dataset()`; modules read it as `from lner import state as S`.
* **Validation protocol:** train on fold1, validate on fold2 (`VAL_MODE="fold2"`, contains all 12 labels). Final models retrain on all 70 docs for the best epoch.
* Every run saves `<LNER_WORK>/<name>.pt` (trainable weights + config incl. selected decoder settings), appends to `results.json`
  (F1/P/R, #params, trainable %, train minutes, peak memory, tokens/s), and prints a decoder-variant table (A greedy / B Viterbi / C +thresholds / D +thresholds+refinement).
* All public functions/classes have `Args: / Returns:` docstrings.

## 4. Experiments (`lner/config.py::EXPERIMENTS` + `lner/experiments.py`)

`A_lsw_lora`, `B_lsw_lorma_384/512`, `E_lsw_lorma_span`, **`F_focal`** (modern decoder + LoRMA + span head + effective class weights + focal loss: validated base run),
`E2_modern_full`, `E3_gpt2_adapter`, `X_t5_lora`; derived runs `F_lsw`, `F_rev`, `T1_ctx192`, `T2_roles`, `T3_ctx_roles`, `B0_bio_softmax`, `B1_bio_crf`,
`E4_kd_student`, `E4b_pruned_kd`, `E4c_pruned_nokd`, `FINAL_*`, `FINALPL_*`. Best Codabench score reported in the notebook: **0.6010** (3-seed modern-decoder ensemble on all 70 docs + counsel-vote + period).

## 5. Setup & how to run

```bash
cd phase2 && pip install -r requirements.txt
export HF_TOKEN=hf_xxx                    # dataset access (or a Kaggle secret named HF_TOKEN)
export LNER_WORK=./outputs/cs781_phase2   # checkpoints / results / submissions
echo '{"modern_decoder": "/path/to/modern/model-final", "lightweight_sliding_window": "/path/to/lsw/model-final"}' > ckpts.json
export PHASE1_CKPT_JSON=ckpts.json        # or edit PHASE1_CKPT in lner/config.py
```

```bash
python scripts/analyze.py --selfchecks                         # sanity checks, no training
python scripts/run_experiment.py --names F_focal               # validated base run (fold1 -> fold2)
python scripts/run_experiment.py --second-family               # F_lsw (second backbone family)
python scripts/run_experiment.py --te                          # training-time enforcement: T1/T2/T3, B0/B1
python scripts/run_experiment.py --kd                          # E4 distillation;  --pruned-kd for E4b/E4c
python scripts/build_postprocessing.py --base F_focal          # cross-fit gate -> POST.pkl
python scripts/final_fit.py --mode seeds --base F_focal        # 3 seeds on all 70 docs -> sub_FINAL_ens3
python scripts/final_fit.py --mode multi                       # seeds x architectures
python scripts/final_fit.py --mode pseudo --base F_focal       # rule-distilled pseudo-labels
python scripts/predict_test.py --ckpts E4_kd_student --name sub_E4_kd_student_3
python scripts/analyze.py --diagnostics F_focal --sweep F_focal --codabench
```

Upload `<LNER_WORK>/<name>/submission.zip` (contains `prediction.json`). Unit tests: `pytest tests/`.

## 6. Results

_Fill in from `results.json` / `show_results()` and the Codabench leaderboard; add the error analysis (296 spurious / 175 boundary / 57 overlap / 49 label-confusion on the first model)._
