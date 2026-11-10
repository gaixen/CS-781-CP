# Phase 1 -- Legal-domain language-model pre-training

Pre-train four **from-scratch** language models (50M-300M parameters) on Indian legal text (acts, high courts, supreme court, tribunals) and use them to continue
a 512-word legal prompt with ~128 words. These checkpoints are the backbones of Phases 2-4.
Dataset: `Exploration-Lab/CS-781-Capstone` (`Phase_1_Pretraining/Train/*.json`, test: `Phase_1_Pretraining/Test/test.json`).

## 1. Model architecture

| `--arch` | Description | Default size |
|---|---|---|
| `gpt2_decoder` | baseline decoder-only (HF GPT-2 implementation, random init) | 768 / 12 layers / 12 heads |
| `modern_decoder` | RMSNorm (pre-norm) + RoPE + Grouped-Query Attention (SDPA/flash) + SwiGLU, **full causal attention** | dim 768, 12 layers, 12 heads / 4 KV heads, d_ff 2048 |
| `lightweight_sliding_window` | same blocks with **local (sliding-window) causal attention**, window 256 -- cheaper, matches locally clustered legal dependencies | dim 768, 12 layers, d_ff 2304 (~121M) |
| `seq2seq_t5` | T5 encoder-decoder (gated-GELU), framed as prompt -> continuation | d_model 576, 8+8 layers |

All use GPT-2's BPE tokenizer (controlled variable), tied input/output embeddings, context length 2048. The two custom decoders share `lm/models/custom_decoder.py`;
the attention pattern (`window_size=None` vs `256`) is the architectural difference.

## 2. Methodology

* **Data:** per-source **document-level** dev split (3 % of documents, reservoir-sampled with a deterministic `zlib.crc32` seed so the split is identical in every session),
  streaming tokenisation into 2048-token blocks with **EOS between documents**, bounded-memory reservoir shuffle, Arrow-backed datasets (never fully in RAM).
  Sources are mixed by relative weights under a total token budget (`train_tokens`).
* **Objective:** causal LM cross-entropy (`CausalLMCollator`: `labels = input_ids`, EOS is *not* masked so the model learns to end documents); T5 uses prompt (700 tok) -> continuation (190 tok) pairs.
* **Optimisation:** AdamW, lr 3e-4, cosine schedule, 3 % warm-up, wd 0.01, batch 2 x grad-accum 16, bf16 (A100/L4/H100) or fp16, resumable `checkpoint-*` (validity check + interval re-application on resume).
* **Evaluation:** validation perplexity; internal generation quality on held-out dev (512-word prompt -> 128-word reference): ROUGE-1/2/L, BLEU, distinct-1/2, repetition rate;
  decoding sweep (greedy / temperature / top-p).
* **Generation:** reference `model.generate`, plus a **KV-cache generator** for the custom decoders (prefill once, one token per step, sliding-window layers attend only to the last 256 keys,
  early stop once enough words exist). It is enabled only if it reproduces the reference greedy output on a self-check.

## 3. Implementation details

```
phase1/
├── lm/
│   ├── config.py                 # constants, model sizes, per-architecture presets, build_config()
│   ├── utils.py                  # seeding, HF login, tokenizer, manual checkpoints
│   ├── data/streaming.py         # streaming reader, dev split, token blocking, shuffle, seq2seq pairs
│   ├── data/datasets.py          # download, per-source budgets, Arrow datasets, internal eval pairs
│   ├── models/custom_decoder.py  # RMSNorm, RoPE, SwiGLU, GQA attention (+window), CustomDecoderForCausalLM
│   ├── models/factory.py         # the 4 architectures, 50M-300M assertion, load_trained()
│   ├── training/trainer.py       # Trainer setup, resume, perplexity, loss plot, save
│   ├── generation/generate.py    # reference + KV-cache generators, self-check
│   ├── generation/metrics.py     # ROUGE/BLEU/distinct-n, decoding sweep
│   └── submission.py             # resumable test generation -> prediction.json -> submission.zip
├── scripts/                      # train.py, generate_submission.py, upload_checkpoint_kaggle.py
└── tests/test_core.py
```

Differences between the two notebooks are captured as **presets** (`PRESETS` in `lm/config.py`): the sliding-window run used supreme-court data only for 1 epoch (eval/save every 500 steps);
the modern-decoder run mixed supreme court (w=1) + high courts (w=3) for 2 epochs (eval every 200, save every 100). The EOS-between-documents and non-masking-collator fixes from the later
sliding-window notebook are the default for every architecture (`add_eos_between_docs=True`). All public functions/classes have `Args: / Returns:` docstrings.

## 4. Setup & how to run

```bash
cd phase1 && pip install -r requirements.txt
export HF_TOKEN=hf_xxx            # dataset access (or a Kaggle secret named HF_TOKEN)
export LM_WORK=./outputs          # runs go to $LM_WORK/cs781_phase1/<arch>/
python scripts/train.py --arch lightweight_sliding_window
python scripts/train.py --arch modern_decoder
python scripts/train.py --arch gpt2_decoder
python scripts/train.py --arch seq2seq_t5
python scripts/train.py --arch modern_decoder --resume                          # continue after a session limit
python scripts/train.py --arch gpt2_decoder --override train_tokens=50000000    # any config key via --override
python scripts/generate_submission.py --arch modern_decoder                     # -> $LM_WORK/cs781_phase1/modern_decoder/submission.zip
python scripts/upload_checkpoint_kaggle.py --arch modern_decoder --username <you> --slug <dataset-slug>   # optional: share with Phase 2-4
```

Outputs per run: `model-final/` (weights + `config.json` + tokenizer), `eval_metrics.json`, `training_validation_loss.png`, `prediction.json`, `submission.zip`. Unit tests: `pytest tests/`.
Hardware used: Kaggle GPUs (T4/P100 -> fp16; resumable training across sessions).

## 5. Results

_Fill in from each run's `eval_metrics.json` (perplexity, parameter count, training time, ROUGE/BLEU/distinct-n) and the Codabench leaderboard._
