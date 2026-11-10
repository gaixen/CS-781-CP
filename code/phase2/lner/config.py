"""Static configuration: paths, Phase-1 checkpoints, default hyper-parameters (``CFG``) and the experiment registry.

Environment variables:
    LNER_WORK          output directory (default ``./outputs/cs781_phase2``).
    PHASE1_CKPT_JSON   optional JSON file ``{arch: checkpoint_path}`` overriding ``PHASE1_CKPT``.
    LNER_DATA_ROOT     optional local folder containing the Phase-2 dataset files (else downloaded from the HF repo).
"""
import os
import json

WORK = os.environ.get("LNER_WORK", os.path.abspath("outputs/cs781_phase2"))
os.makedirs(WORK, exist_ok=True)
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
DATA_ROOT = os.environ.get("LNER_DATA_ROOT")    # e.g. "/kaggle/input/<your-dataset>/Phase_2_LNER"; None -> download from the HF repo

# kind: which backbone wrapper to use. window: sliding-window size used in Phase 1 (None = full attention)
ARCH_INFO = {
    "lightweight_sliding_window": dict(kind="custom", window=256),
    "modern_decoder":             dict(kind="custom", window=None),
    "gpt2_decoder":               dict(kind="gpt2"),
    "seq2seq_t5":                 dict(kind="t5"),
}
# Path to each Phase-1 checkpoint (.pt with {"model": state_dict}, raw state_dict .pt, a .safetensors file,
# or an HF folder containing model.safetensors / pytorch_model.bin)
PHASE1_CKPT = {
    "lightweight_sliding_window": "/kaggle/input/datasets/elise047/cs-781-phase1-lightweight-sliding-window/cs781_phase1/lightweight_sliding_window/model-final",
    "modern_decoder":             "/kaggle/input/datasets/gaixen007/cs781-phase1-modern-decoder-checkpoints/cs781_phase1/modern_decoder/model-final",
    "gpt2_decoder":               "/kaggle/input/datasets/sohamjee/cs781-phase1-checkpoints-gpt2-train-2048-150m/model-final",
    "seq2seq_t5":                 "/kaggle/input/YOUR_DATASET/seq2seq_t5.pt",
}
if os.environ.get("PHASE1_CKPT_JSON"):
    with open(os.environ["PHASE1_CKPT_JSON"]) as _f:
        PHASE1_CKPT.update(json.load(_f))

CFG = dict(
    ARCH="lightweight_sliding_window", CKPT=None,         # CKPT=None -> PHASE1_CKPT[ARCH]
    METHOD="lora",              # full | partial | bitfit | lora | lorma | adapter
    ATTN_MODE="bidirectional",   # causal | bidirectional  (custom decoders only; gpt2 is causal, t5 encoder is bidirectional)
    BIDIR_WINDOW="keep",         # "keep" (use Phase-1 window) | None (full) | int  -- only for bidirectional custom decoders
    HEAD="mlp", HEAD_DROPOUT=0.1,                      # linear | mlp
    TASK="bio",                  # bio (token BIO tagging) | span (start / end / type heads)
    # data / windows
    MAX_LEN=256, OVERLAP=128, NEG_WINDOW_KEEP=1.5,        # try 384/192 then 512/256 (Phase-1 context is 2048)
    MERGE="ramp",                # window merging at inference: ramp (edge-down-weighted mean) | mean | max (most confident window)
    VAL_MODE="fold2", VAL_FRAC=0.10,                      # fold2 (train fold1, val fold2: has all 12 labels incl. WIT) | random (doc-level split)
    # optimisation
    BATCH=16, ACCUM=2, EPOCHS=5, WARMUP=0.06, WD=0.01, CLIP=1.0,
    LR={"full": 3e-4, "partial": 1e-4, "bitfit": 5e-4, "lora": 6e-4, "lorma": 6e-4, "adapter": 5e-4}, LR_HEAD=1e-3,
    CLASS_WEIGHT_POWER=0.5, CLASS_WEIGHT_MAX=10.0,        # power-mode weights; POWER=0 disables
    CLASS_WEIGHT_MODE="power",   # power | effective (class-balanced CE, Cui et al.) | off
    EFF_BETA=0.9999,
    LOSS="ce", FOCAL_GAMMA=2.0,  # ce | focal  (focal keeps the class weights)
    SPAN_BOUNDARY_W=1.0,         # span task: weight of the start+end losses relative to the type loss
    # PEFT
    R=64, ALPHA=192, LORA_DROPOUT=0.05, BOTTLENECK=64, PEFT_TARGETS=("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"), PARTIAL_LAYERS=4,
    # decoding (BIO: DECODE greedy|viterbi). AUTO_DECODE=True -> after training, every decoder variant is scored on the cached val
    # probabilities and the best (DECODE, BOUNDARY_REFINE, CONF_THRESH) is stored with the checkpoint.
    DECODE="viterbi",
    TRIM_TRAILING_PUNCT=True,   # label-specific trimming learned from train gold
    BOUNDARY_REFINE=False, REFINE_K=3, REFINE_LAMBDA=1.5,   # char-level boundary snapping (+-K chars, shift penalty LAMBDA nats)
    CONF_THRESH=None,            # {label: min span confidence}; filled automatically by AUTO_DECODE
    AUTO_DECODE=True, THRESH_MIN_SUPPORT=8, THRESH_MIN_GAIN=0.01,
    SPAN_START_MIN=0.10, SPAN_END_MIN=0.10, SPAN_MIN_CONF=0.05,   # span task candidate generation
    # distillation (student side)
    KEEP_LAYERS=None, KD_ALPHA=0.5, KD_TEMP=2.0,
    SEED=42,
    CTX_PREFIX=0, LABEL_W=None, OVERSAMPLE=None, USE_CRF=False, HEAD_LAYERS=2, USE_BILSTM=True,     # training-time enforcement: all OFF by default
    LOG_EVERY=50,
    STRIP_WS=True,
)

EXP_LOSS = "ce"      # "ce" reproduces the old runs; "focal" is a different experiment (old calls silently ignored a lowercase `loss=` key)

# Registry of the notebook's experiments (override any key from the CLI with --override KEY=VALUE).
EXPERIMENTS = {
    "A_lsw_lora":      dict(ARCH="lightweight_sliding_window", METHOD="lora", ATTN_MODE="bidirectional"),
    "B_lsw_lorma_384": dict(ARCH="lightweight_sliding_window", METHOD="lorma", MAX_LEN=384, OVERLAP=192),
    "B_lsw_lorma_512": dict(ARCH="lightweight_sliding_window", METHOD="lorma", MAX_LEN=512, OVERLAP=256),
    "E_lsw_lorma_span": dict(ARCH="lightweight_sliding_window", METHOD="lorma", TASK="span", ATTN_MODE="bidirectional"),
    "F_focal":         dict(ARCH="modern_decoder", METHOD="lorma", TASK="span", CLASS_WEIGHT_MODE="effective", LOSS="focal"),   # validated base run
    "E2_modern_full":  dict(ARCH="modern_decoder", METHOD="full", ATTN_MODE="bidirectional"),
    "E3_gpt2_adapter": dict(ARCH="gpt2_decoder", METHOD="adapter"),
    "X_t5_lora":       dict(ARCH="seq2seq_t5", METHOD="lora"),
}
# Training-time enforcement presets (cell X): extra loss weights + oversampling for the weak role labels
ROLES = dict(LABEL_W={"APP": 2.0, "RESP": 2.0, "WIT": 1.5}, OVERSAMPLE={"labels": ("APP", "RESP", "WIT", "AUTH"), "x": 2})
