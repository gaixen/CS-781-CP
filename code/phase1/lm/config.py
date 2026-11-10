"""Configuration: constants, architecture sizes, per-run presets and ``build_config``.

Environment variables:
    LM_WORK   working/output directory (default ``./outputs``); runs are written to ``<LM_WORK>/cs781_phase1/<arch>``.
    HF_TOKEN  Hugging Face token (dataset access).
"""

import os

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
WORK = os.environ.get("LM_WORK", os.path.abspath("outputs"))

ALL_SOURCES = {
    "acts": {"file": "Phase_1_Pretraining/Train/train_acts.json", "weight": 1.0},
    "high_courts": {
        "file": "Phase_1_Pretraining/Train/train_high_courts.json",
        "weight": 1.0,
    },
    "supreme_court": {
        "file": "Phase_1_Pretraining/Train/train_supreme_court.json",
        "weight": 1.0,
    },
    "tribunals": {
        "file": "Phase_1_Pretraining/Train/train_tribunals.json",
        "weight": 1.0,
    },
}
TEST_FILE = "Phase_1_Pretraining/Test/test.json"
TEST_FILE_ALT = "Phase_1_Pretraining/Test/Phase1- Test.json"

ARCHITECTURES = (
    "gpt2_decoder",
    "modern_decoder",
    "seq2seq_t5",
    "lightweight_sliding_window",
)

# ---- model sizes (all inside the 50M-300M budget; exact counts are printed/asserted when the model is built) ----
GPT2_DECODER_CFG = dict(n_embd=768, n_layer=12, n_head=12)
MODERN_DECODER_CFG = dict(
    dim=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2048, dropout=0.0
)
LIGHTWEIGHT_SW_CFG = dict(
    dim=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2304, window_size=256, dropout=0.0
)  # ~121M params
SEQ2SEQ_T5_CFG = dict(
    d_model=576,
    d_kv=64,
    d_ff=1536,
    num_layers=8,
    num_decoder_layers=8,
    num_heads=9,
    dropout_rate=0.1,
)

BASE = dict(
    arch="lightweight_sliding_window",
    seed=42,
    data_sources={
        "supreme_court": {"file": ALL_SOURCES["supreme_court"]["file"], "weight": 1.0}
    },
    train_tokens=200_000_000,
    val_tokens=5_000_000,
    dev_doc_fraction=0.03,
    train_block_size=2048,
    epochs=1,
    model_context_length=2048,
    add_eos_between_docs=True,
    seq2seq_src_tokens=700,
    seq2seq_tgt_tokens=190,
    seq2seq_pairs_per_doc=2,
    seq2seq_max_train_docs=20000,
    eval_generation_examples=50,
    target_generation_words=128,
    max_generation_tokens=384,
    temperature=0.1,
    top_p=0.95,
    # trainer
    batch_size=2,
    eval_batch_size=2,
    grad_accum=16,
    lr=3e-4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    logging_steps=50,
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    model_cfg=None,  # filled by build_config from the arch presets above
)

# Presets that reproduce the two notebooks (override anything with --override KEY=VALUE).
PRESETS = {
    "lightweight_sliding_window": dict(),  # BASE already = the lightweight-sliding-window notebook (supreme-court only, 1 epoch)
    "modern_decoder": dict(
        data_sources={
            "supreme_court": {
                "file": ALL_SOURCES["supreme_court"]["file"],
                "weight": 1.0,
            },
            "high_courts": {"file": ALL_SOURCES["high_courts"]["file"], "weight": 3.0},
        },
        epochs=2,
        eval_steps=200,
        save_steps=100,
    ),
    "gpt2_decoder": dict(),
    "seq2seq_t5": dict(batch_size=8, eval_batch_size=8, grad_accum=4),
}


def build_config(arch, **overrides):
    """Assemble the run configuration ``BASE < PRESETS[arch] < overrides`` plus derived paths.

    Args:
        arch (str): one of ``ARCHITECTURES``.
        **overrides: any ``BASE`` key (e.g. ``epochs=2``, ``train_tokens=50_000_000``, ``model_cfg={...}``).

    Returns:
        dict: config including ``output_dir``, ``model_output_dir``, ``prediction_path``, ``submission_zip``, ``is_seq2seq``.

    Raises:
        ValueError: for an unknown architecture.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch}; choose from {ARCHITECTURES}")
    cfg = {**BASE, **PRESETS[arch], **overrides, "arch": arch}
    cfg["model_cfg"] = (
        cfg["model_cfg"]
        or {
            "gpt2_decoder": GPT2_DECODER_CFG,
            "modern_decoder": MODERN_DECODER_CFG,
            "lightweight_sliding_window": LIGHTWEIGHT_SW_CFG,
            "seq2seq_t5": SEQ2SEQ_T5_CFG,
        }[arch]
    )
    cfg["is_seq2seq"] = arch == "seq2seq_t5"
    cfg["output_dir"] = os.path.join(WORK, "cs781_phase1", arch)
    cfg["model_output_dir"] = os.path.join(cfg["output_dir"], "model-final")
    cfg["prediction_path"] = os.path.join(cfg["output_dir"], "prediction.json")
    cfg["submission_zip"] = os.path.join(cfg["output_dir"], "submission.zip")
    cfg["loss_plot_path"] = os.path.join(
        cfg["output_dir"], "training_validation_loss.png"
    )
    cfg["eval_metrics_path"] = os.path.join(cfg["output_dir"], "eval_metrics.json")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg
