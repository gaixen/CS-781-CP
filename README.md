# In-House Legal Intelligence System

### A Seven-Phase Pipeline for Indian Legal NLP under Constrained Compute

**CS781 Capstone Project** · Indian Institute of Technology Kanpur
**Author:** Soham Mukherjee (Roll No. 241027) · `smukherjee24@iitk.ac.in`
**Codabench user:** `s_241027`

---

## Abstract

Legal organisations that handle confidential case files require NLP systems that can be deployed in-house, without external APIs or pretrained models. This project develops a seven-phase pipeline over Indian legal text: language-model pretraining, legal named-entity recognition (L-NER), rhetorical-role prediction, court judgment prediction (CJPE), legal statute identification (LSI), abstractive summarisation, and prior case retrieval (PCR).

All models derive from a single from-scratch foundation: a 114M-parameter decoder-only Transformer (grouped-query attention, rotary position embeddings, RMSNorm, SwiGLU), a 58M-parameter sliding-window variant, and a 60k-vocabulary byte-level BPE tokenizer trained on the legal corpus. These checkpoints initialise every downstream task and are adapted through full fine-tuning, parameter-efficient fine-tuning (LoRMA+, LoRA, adapters, BitFit), gradual unfreezing, and knowledge distillation, with task-specific heads (BiLSTM-CRF, label-wise attention, bi-encoder retrieval).

The main findings are:

1. Decoder-only backbones outperform a T5 baseline at matched scale.
2. LoRMA+ matches full fine-tuning within 0.001 BERTScore while training about 10% of the parameters.
3. Domain-calibrated Viterbi decoding gives the largest single improvement in rhetorical-role prediction (+0.05 macro-F1).

---

## 1. Official Results

Scores are the primary Codabench metric of each phase. Ranks were recorded at 14:48, 09 November 2026.

| Phase | Task | Primary metric | Score | Rank | Valid submissions |
|:-----:|------|----------------|:-----:|:----:|:-----------------:|
| 1 | Legal LM pretraining | Generation quality (BERTScore-based) | 0.8302 | 80 | 28 |
| 2 | Legal NER | Strict span-level F1 | 0.6393 | 11 | 90 |
| 3 | Rhetorical role prediction | Macro-F1 | 0.5635 | 21 | 86 |
| 4 | Court judgment prediction | Primary metric | 0.8247 | 9 | 67 |
| 5 | Legal statute identification | Macro-F1 | 0.4511 | 7 | 43 |
| 6 | Abstractive summarisation | BERTScore F1 | 0.8478 | 28 | 39 |
| 7 | Prior case retrieval | F1@7 | 0.5139 | 3 | 32 |

Across all phases there were 385 valid Codabench submissions. Local development scores and leaderboard scores diverge (for example, Phase 2 dev F1 ≈ 0.33 against 0.639 on Codabench). Development figures reported below should therefore be read as relative comparisons, not absolute predictions.

---

## 2. System Overview

```
                     Phase 1: legal pretraining corpus (4 sources, 200M tokens)
                                           │
              60k byte-level BPE tokenizer + decoder-only foundation
          (modern decoder 114M · lightweight sliding window 58M · baselines)
                                           │
        ┌───────────┬───────────┬──────────┼──────────┬───────────┬───────────┐
     Phase 2     Phase 3     Phase 4    Phase 5    Phase 6     Phase 7
     L-NER       Rhetorical  CJPE       LSI        Summar-     PCR
     (BIO +      roles       (hier.     (LWAN +    isation     (bi-encoder +
     Viterbi)    (BiLSTM-    chunking)  sigmoid    (prefix-LM) LTR reranker)
                 CRF)                   heads)
```

Every task model is initialised from the Phase-1 checkpoints (embeddings, transformer layers, attention projections, normalisation) rather than trained independently. Downstream adaptation differs by task and is summarised in Section 4.

### Design constraints

* Phase-1 models are restricted to 50M–300M parameters and are trained from scratch.
* No external pretrained models (including legal-domain BERTs), no external tokenizers or APIs, and no non-legal text.
* Compute is limited to Kaggle-class hardware (2× NVIDIA T4), which motivated FP16 mixed precision, gradient checkpointing, and KV-cached inference.
* Only official course data (`Exploration-Lab/CS-781-Capstone`) were used, and test labels were never read.

---

## 3. Repository Structure

```
cs781-capstone/
├── README.md                 # this document
├── phase1/                   # Legal LM pretraining (lm/ package, scripts/, tests/)
├── phase2/                   # Legal NER (lner/ package, scripts/, tests/)
├── phase3/                   # Rhetorical roles (placeholder in this snapshot)
└── phase4/                   # CJPE (cjpe/ package, scripts/, tests/)
...
```
---

## 4. Methodology

### 4.1 Data

| Phase | Data and splits |
|:-----:|-----------------|
| 1 | Four raw-text sources. 3% of documents per source are held out for perplexity. |
| 2 | 12 entity types, 8,393 entities, two folds of 35 documents plus a 35-document test set. |
| 3 | Corporate Law (CL) and Information Technology (IT) documents, 40/5/5 per domain (10,593 CL and 6,252 IT training sentences). |
| 4 | Binary judgment outcome, with a class-weighted loss and a tuned decision threshold. |
| 5 | 42,750 / 9,999 / 13,019 documents, 100 statutes, strongly skewed label frequencies. |
| 6 | 6,701 / 300 / 100 documents, with a 300-document holdout carved from training. |
| 7 | 827 / 118 / 237 queries, with per-split candidate pools. |

The Phase-1 corpus is mixed with High Courts upweighted ×5 and Acts, Supreme Court and Tribunals at weight 1. This yields 200M training tokens plus 5M held out.

| Source | Documents | Size |
|--------|----------:|-----:|
| Acts | 18,822 | 0.73 GB |
| High Courts | 2,536,738 | 10.7 GB |
| Supreme Court | 55,528 | 1.32 GB |
| Tribunals | 869,132 | 6.48 GB |

**Split repairs.** Phase 5's `dev.json` overlapped other splits, so 182 of 10,181 rows were removed (96 test, 61 train, 25 internal duplicates), leaving 9,999. Phase 6 has no dev file, so 300 training documents were held out and 29 unusable or duplicate pairs removed (7,030 → 7,001).

### 4.2 Tokenizer

A byte-level BPE tokenizer with a 60,000-token vocabulary is trained from scratch on the entire legal corpus.

* Words are whitespace-split. UTF-8 bytes are the base symbols, and a dedicated end-of-word symbol (byte `0x01`) closes every word.
* Merge selection is most-frequent-pair with deterministic tie-breaking. A fast incremental trainer produces exactly the same merges as the naive reference algorithm (verified).
* The tokenizer is exported as a HuggingFace fast tokenizer, so `AutoTokenizer.from_pretrained(<path>/legal_bpe/hf)` works throughout the pipeline.
* Decoded text is single-spaced, because whitespace runs and newlines are not preserved. This is irrelevant to word-level metrics such as ROUGE and BLEU.
* The same tokenizer folder must be used for training, validation and test-time generation. Models trained with it cannot reuse checkpoints trained with another vocabulary.

### 4.3 Phase 1: Foundation Models

| Model | Description |
|-------|-------------|
| **Modern decoder** (primary) | 12 layers, 12 attention heads, 4 KV heads (GQA), hidden size 768, FFN size 2048, RoPE, pre-RMSNorm, manual SwiGLU. About 114.1M parameters; the 50M ≤ n ≤ 300M budget is asserted programmatically. |
| **Lightweight sliding window (LSW)** | Same blocks with local causal attention (window 256). The submitted configuration has about 58M parameters. |
| **GPT-2 baseline** | Hugging Face GPT-2 stack, random initialisation. |
| **T5 baseline** | Encoder-decoder (about 110M parameters). |

* **Objective:** causal language modelling, cross-entropy, 2,048-token context, EOS between documents.
* **Evaluation:** held-out perplexity

$$\mathrm{PPL}=\exp\!\Big(-\tfrac{1}{N}\sum_{i=1}^{N}\log p(x_i\mid x_{<i})\Big),$$

  and generation quality on a prompt → 128-word continuation task.
* **Generation:** a custom KV-cache generator with left-padding for batched causal decoding. It is enabled only if it reproduces the reference greedy output. Raising the temperature from 0.1 to 0.7–0.8 improved the primary metric by about 2%.

### 4.4 Transfer protocol and task heads (Phases 2–7)

| Phase | Input handling | Model and objective | Inference and post-processing |
|:-----:|----------------|--------------------|-------------------------------|
| **2 L-NER** | Overlapping 512-token windows; exact character offsets via offset mapping | Token classification (25-tag BIO scheme), focal loss (γ = 2), LoRMA teacher, KD students (3/6/9 layers) | BIO-constrained Viterbi; seeded ensemble of modern decoder + LSW |
| **3 Rhetorical roles** | Sentence plus up to 3 neighbours per side (96/32-token limits); `[CL]`/`[IT]` domain tag | Sentence encoder → BiLSTM over windows (32 train / 64 inference) → linear-chain CRF; auxiliary label-shift task (λ = 0.3) | Domain-calibrated Viterbi (CL λ = 0.52, IT λ = 0.27) on ensemble posteriors |
| **4 CJPE** | Head/tail truncation or hierarchical 512-token chunks (stride 64) | Pooling (mean / attention / final-token) and a chunk aggregator (mean, attention, BiGRU, Transformer); LoRA, partial FT, KD; layer-wise LR decay | Class-weighted loss and tuned threshold |
| **5 LSI** | 512/64-token chunks, up to 12,288 tokens per document | Label-Wise Attention Network; sigmoid outputs initialised at each statute's log-prior; BCE, focal and asymmetric loss; sigmoid-based KD | Global, per-label (shrunk) and bagged thresholds; top-1 fallback; top-k caps |
| **6 Summarisation** | Prefix-LM `[document] <SEP> [summary] <EOS>`, 1,536 document + 256 summary tokens (head-tail or importance-selected blocks) | LoRMA+ and full fine-tuning; loss on summary tokens only; KD students (full and half depth) | Log-probability ensembling; repetition penalty and 3-gram blocking; fragment trimming |
| **7 PCR** | Header and anonymisation stripping; 70/30 head-tail cap at 120,000 characters; 512-token chunks (2 train, 8 eval) | Shared bi-encoder with mean pooling and L2 normalisation; symmetric InfoNCE with hard negatives | TF-IDF/BM25 + dense scores → logistic-regression LTR reranker over top 150–400 candidates; top-7 output |

---

## 5. Key Findings

**Foundation.**
* The modern decoder and LSW outperform GPT-2.
* Extending LSW context length gave no BERTScore gain. Scaling its parameters halved training loss but did not improve downstream metrics.
* The T5 baseline was unstable in FP16, repeated itself heavily and had worse perplexity.

**Parameter efficiency.**
* On summarisation, LoRMA+ on LSW (11.0M trainable parameters, about 10% of the full model) reaches 0.8395 BERTScore against 0.8402 for full fine-tuning of the modern decoder.
* It also cuts training time by 20% and peak memory by 27% (4.7 → 3.4 GB), at a cost of 0.009 ROUGE-1.

**Sequence decoding.**
* In L-NER, Viterbi decoding lifts mean strict F1 from 0.293 to 0.401 over greedy argmax. Most greedy errors are span fragmentation.
* In rhetorical-role prediction, the label-shift auxiliary task and BiLSTM-CRF reach 0.476 dev macro-F1 (+0.052 over the context-only model).
* Replacing argmax with domain-calibrated Viterbi lifted a leaderboard submission from 0.4991 to 0.5501.

**Distillation.**
* KD improved every L-NER student: +0.073 (3-layer), +0.043 (6-layer) and +0.038 (9-layer) strict macro-F1.
* The 3-layer student tripled throughput (43k → 125k tokens/s) for a 0.024 F1 loss.
* Sigmoid-based KD gave the best statute-identification result (macro-F1 0.4078).

**Ensembling and retrieval.**
* Averaging E1 and E2 logits gives the best summarisation dev result (BERTScore 0.8421, ROUGE-1 0.2871) at about 4× E1's inference cost.
* In PCR, mean token pooling is essential. Switching to max or last-token pooling halves dev F1 from about 0.30 to 0.15.
* Keeping case headers made models overfit to formatting artefacts, which motivated deterministic stripping.

**Negative results.**
* Equal-weight ensembling did not help statute identification.
* Bias tuning and simple prior corrections reduced rhetorical-role performance.
* Later rule-based tweaks for rhetorical roles saturated (ΔF1 < 0.001).

---

## 6. Limitations

* **Context length.** The 512–2,048-token windows force heavy truncation. For summarisation, 91.4% of training documents exceed the 1,536-token budget.
* **Generation length.** Phase-6 outputs average 149–153 words against a reference median of 638. The configured minimum length (387) exceeds the maximum (255), which blocks early stopping.
* **Untested components.** The `select` and `hier` summarisation strategies were implemented but produced no logged results. The distilled Phase-6 students have no held-out score because they were trained on the dev split.
* **Metrics.** ROUGE, BERTScore and strict span-F1 do not measure factual correctness. Hallucinated facts or inverted rulings would pass automated checks. Outputs, especially summaries and predicted judgments, require human legal review.
* **Local–leaderboard gap.** See Section 1.

**Future work.** A multi-task backbone with joint span- and sequence-level objectives, retrieval-augmented generation over statutory databases, sparse-attention architectures for full-length judgments, per-label thresholds learned jointly for rare statutes, and factuality-aware summarisation.

---

## 7. Reproducibility

| Item | Setting |
|------|---------|
| Hardware | 2× NVIDIA T4 (Kaggle-class limits) |
| Seed | 42 (Phase-6 distillation: three seeds per student) |
| Precision | FP16 mixed precision (FP32 where unstable, e.g. T5) |
| Optimiser | AdamW, weight decay 0.01, gradient clipping 1.0 |
| Schedule | Warmup, then cosine or linear decay |
| Checkpointing | Best-validation checkpoint retained |
| Phase-1 budget | 200M training tokens, 5M validation, 2,048-token context |
| Data | Official course data only; no test labels read |

### Setup and execution

Each phase is self-contained.

```bash
# Phase 1: pretraining and test-time generation
cd phase1 && pip install -r requirements.txt
export HF_TOKEN=<token>            # dataset access
export LM_WORK=./outputs
python scripts/train.py --arch modern_decoder
python scripts/train.py --arch lightweight_sliding_window
python scripts/generate_submission.py --arch modern_decoder

# Phase 2: L-NER
cd ../phase2 && pip install -r requirements.txt
python scripts/analyze.py --selfchecks
python scripts/run_experiment.py --names F_focal
python scripts/run_experiment.py --kd
python scripts/final_fit.py --mode seeds --base F_focal

# Phase 4: CJPE
cd ../phase4 && pip install -r requirements.txt
python scripts/run_experiments.py --names E3_bidir_lorma --smoke
python scripts/run_experiments.py --names E1_lsw_partial E1a_kd_lsw_student E3_bidir_lorma E4_kd_lsw_student
python scripts/package_submissions.py E1_lsw_partial E1a_kd_lsw_student E3_bidir_lorma E4_kd_lsw_student
```

Unit tests, where provided, run with `pytest tests/` inside each phase folder. Phase-4 checkpoint paths are set in `phase4/cjpe/config.py` (`PHASE1_CKPT`); tokenizer files are read from the same directories.

---

## 8. Large-Language-Model Usage

LLM assistance (Claude, Gemini) was used for high-level ideation, plotting code, report structuring, and implementation drafting. Architectural decisions, experiments and analysis were the author's own. The full declaration is in the project report.

---

## 9. Selected References

1. Vaswani et al., *Attention is all you need*, NeurIPS 2017.
2. Sennrich, Haddow, Birch, *Neural machine translation of rare words with subword units*, ACL 2016.
3. Ainslie et al., *GQA: Training generalized multi-query transformer models*, EMNLP 2023.
4. Su et al., *RoFormer: Enhanced transformer with rotary position embedding*, 2021.
5. Zhang, Sennrich, *Root mean square layer normalization*, NeurIPS 2019.
6. Shazeer, *GLU variants improve transformer*, 2020.
7. Hinton, Vinyals, Dean, *Distilling the knowledge in a neural network*, 2015.
8. Bihany, Patel, Modi, *LoRMA: Low-rank multiplicative adaptation for LLMs*, Findings of ACL 2025.
9. Hu et al., *LoRA: Low-rank adaptation of large language models*, ICLR 2022.
10. Malik et al., *ILDC for CJPE*, ACL-IJCNLP 2021.
11. Malik et al., *Semantic segmentation of legal documents via rhetorical roles*, NLLP 2022.
12. Kalamkar et al., *Named entity recognition in Indian court judgments*, NLLP 2022.
13. Paul, Goyal, Chakraborty, *LeSICiN*, AAAI 2022.
14. Shukla et al., *Legal case document summarization*, AACL-IJCNLP 2022.
15. Joshi, Sharma, Modi, *U-CREAT: Unsupervised case retrieval using events extrAcTion*, ACL 2023.
16. Mullenbach et al., *Explainable prediction of medical codes from clinical text*, NAACL 2018.
17. Lafferty, McCallum, Pereira, *Conditional random fields*, ICML 2001.
18. van den Oord, Li, Vinyals, *Representation learning with contrastive predictive coding*, 2018.

The complete bibliography is in the project report.

---

## 10. Citation

```bibtex
@misc{mukherjee2026legalintel,
  author = {Mukherjee, Soham},
  title  = {In-House Legal Intelligence System: A Seven-Phase Pipeline for Indian Legal NLP},
  note   = {CS781 Capstone Project, Indian Institute of Technology Kanpur},
  year   = {2026}
}
```

## 11. Acknowledgements

Dataset and evaluation infrastructure: the CS781 course staff and the `Exploration-Lab/CS-781-Capstone` dataset.
