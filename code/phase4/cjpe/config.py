"""Static configuration: paths, Phase-1 checkpoints, dataset files, default hyper-parameters, experiment registry.

Environment variables:
    CJPE_WORK            working directory for outputs/cache (default ``./outputs``).
    PHASE1_CKPT_JSON     optional path to a JSON ``{arch: checkpoint_dir}`` that overrides ``PHASE1_CKPT``.
"""
import os
import json

SEED = 42
WORK = os.environ.get("CJPE_WORK", os.path.abspath("outputs"))
OUT_ROOT, CACHE_DIR = f"{WORK}/phase4_cjpe", f"{WORK}/cache"
RESULTS_PATH = f"{OUT_ROOT}/results.jsonl"
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---- Phase-1 checkpoints (dir containing model.safetensors/pytorch_model.bin + config.json + tokenizer files) ----
PHASE1_CKPT = {
    "lightweight_sliding_window": "/kaggle/input/datasets/elise047/cs-781-phase1-lightweight-sliding-window/cs781_phase1/lightweight_sliding_window/model-final",
    "lightweight_sliding_window_bidir": "/kaggle/input/datasets/gaixen007/lsw-bidirectional-checkpoints/cs781_phase1/lightweight_sliding_window_bidir/model-final",
    "modern_decoder": "/kaggle/input/datasets/gaixen007/cs781-phase1-modern-decoder-checkpoints/cs781_phase1/modern_decoder/model-final",
    "gpt2_decoder": None,     # TODO: set when you have it (spec: all four Phase-1 models should be explored)
    "seq2seq_t5": None,       # TODO: set when you have it
}
if os.environ.get("PHASE1_CKPT_JSON"):
    with open(os.environ["PHASE1_CKPT_JSON"]) as _f:
        PHASE1_CKPT.update(json.load(_f))

TOKENIZER_PER_ARCH = dict(PHASE1_CKPT)     # tokenizer files live next to the checkpoint; override per arch if not
STRICT_LOAD = True                         # raise if checkpoint weights are missing (otherwise they'd silently be random)

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
DATA_FILES = {"train": "Phase_4_CJPE/Train/train.json", "dev": "Phase_4_CJPE/Train/dev.json", "test": "Phase_4_CJPE/Test/test.json"}

# ---- token budget kept per document after pre-tokenisation ----
HEAD_KEEP, TAIL_KEEP = 1024, 4096          # first 1024 / last 4096 tokens are cached (MAX_TOKENS<=HEAD_KEEP for first_n)
CHAR_HEAD, CHAR_TAIL = 8000, 48000         # very long cases: only the head/tail character slices are tokenised

# =============================== default hyper-parameters ===============================
BASE = dict(
    arch="lightweight_sliding_window", method="lorma",   # full | lorma | bitfit | adapters | partial | gradual | distill
    # --- document representation ---
    REPRESENTATION="last_n",        # last_n | first_n | head_tail  (single sequence)   |   hier (chunks -> doc encoder)
    MAX_TOKENS=1024, HEAD_TOKENS=256,                      # head_tail = first HEAD_TOKENS + last (MAX_TOKENS-HEAD_TOKENS)
    CHUNK_TOKENS=512, CHUNK_STRIDE=64, MAX_CHUNKS=8,       # hier: overlapping chunks counted back from the END of the case
    # --- architecture of the classifier ---
    POOLING="mean_last",            # last | mean | attn | mean_last     (token -> vector)
    DOC_AGG="transformer", DOC_LAYERS=2,                   # hier only: mean | attn | transformer | gru  (chunk -> doc)
    HEAD_HIDDEN=256, HEAD_LAYERS=1, DROPOUT=0.2,           # HEAD_LAYERS>1 = deeper residual MLP head
    # --- adaptation ---
    LORMA_TARGETS="qkv", PEFT_R=16, PEFT_ALPHA=16,         # lorma: qkv | attn | attn_ffn
    ADAPTER_R=64, TOP_K=3, LLRD_DECAY=1.0,                 # adapters bottleneck / partial top-k layers / layer-wise LR decay (<1 enables)
    # --- optimisation ---
    LR_BACKBONE=3e-5, LR_PEFT=6e-4, LR_HEAD=1e-3, WEIGHT_DECAY=0.01, WARMUP_FRAC=0.06, CLIP=1.0,
    EPOCHS=3, BATCH_SIZE=8, GRAD_ACCUM=1, GRAD_CKPT=False, CHUNK_MBS=16, AMP=None,   # AMP=None -> fp16 autocast except T5
    CLASS_WEIGHTED=True, LABEL_SMOOTH=0.0, CALIBRATE=True,   # CALIBRATE: pick the decision threshold maximising dev macro-F1
    TRAIN_SUBSET=None, DEV_SUBSET=None,                      # ints for quick smoke tests
    # --- knowledge distillation ---
    teacher=None, teacher_weights=None, KD_ALPHA=0.5, KD_T=2.0, KD_BETA=0.5, KEEP_EVERY=1,   # KEEP_EVERY=2 -> student keeps every 2nd layer
)

HIER = dict(REPRESENTATION="hier", MAX_CHUNKS=4, BATCH_SIZE=2, GRAD_ACCUM=4, GRAD_CKPT=True)   # T4-friendly hierarchical preset
AB = dict(arch="lightweight_sliding_window", method="lorma")                                      # ablation base

EXPERIMENTS = {
    # ===== the 4 required submissions: different Phase-1 backbones x different techniques (edit to taste) =====
    "E1_lsw_partial":         dict(arch="lightweight_sliding_window", method="partial", TOP_K=3),
    "E1a_kd_lsw_student":     dict(arch="lightweight_sliding_window", method="distill", teacher="E1_lsw_partial", KEEP_EVERY=2),
    "E2_modern_lorma_hier":   dict(arch="modern_decoder", method="lorma", LLRD_DECAY=0.9, DOC_AGG="transformer", **HIER),
    "E3_bidir_lorma":         dict(arch="lightweight_sliding_window_bidir", method="lorma", POOLING="mean", HEAD_LAYERS=2),
    "E4_kd_lsw_student":      dict(arch="lightweight_sliding_window_bidir", method="distill", teacher="E3_bidir_lorma", KEEP_EVERY=2),  # mandatory KD
    # ===== when the other two Phase-1 checkpoints are available =====
    "E5_t5_bitfit":           dict(arch="seq2seq_t5", method="bitfit", POOLING="mean"),
    "E6_gpt2_partial":        dict(arch="gpt2_decoder", method="partial", TOP_K=4),
    # ===== ablations on one backbone (representation / sections / pooling / depth / technique / imbalance) =====
    "A_rep_first_n":          dict(**AB, REPRESENTATION="first_n"),
    "A_rep_head_tail":        dict(**AB, REPRESENTATION="head_tail"),
    "A_rep_last_512":         dict(**AB, MAX_TOKENS=512),
    "A_hier_mean":            dict(**AB, DOC_AGG="mean", **HIER),
    "A_hier_attn":            dict(**AB, DOC_AGG="attn", **HIER),
    "A_hier_transformer":     dict(**AB, DOC_AGG="transformer", **HIER),
    "A_hier_gru":             dict(**AB, DOC_AGG="gru", **HIER),
    "A_pool_last":            dict(**AB, POOLING="last"),
    "A_pool_mean":            dict(**AB, POOLING="mean"),
    "A_pool_attn":            dict(**AB, POOLING="attn"),
    "A_head_deep":            dict(**AB, HEAD_LAYERS=3),
    "A_noclassweight":        dict(**AB, CLASS_WEIGHTED=False),
    "A_lorma_attn_ffn":       dict(**AB, LORMA_TARGETS="attn_ffn"),
    "A_full":                 dict(arch="lightweight_sliding_window", method="full"),
    "A_bitfit":               dict(arch="lightweight_sliding_window", method="bitfit"),
    "A_adapters":             dict(arch="lightweight_sliding_window", method="adapters"),
    "A_partial":              dict(arch="lightweight_sliding_window", method="partial", TOP_K=3),
    "A_gradual":              dict(arch="lightweight_sliding_window", method="gradual"),
}
