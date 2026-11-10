"""Model factory: Phase-1 backbone + head + adaptation method."""
from .. import state as S
from ..config import PHASE1_CKPT
from .backbones import build_backbone
from .heads import NERModel, SpanNERModel, CRFNERModel
from .peft import configure_trainable, count_params


def build_ner_model(cfg):
    """Build the NER model described by ``cfg``.

    Steps: load Phase-1 backbone -> optional layer pruning (``KEEP_LAYERS``) -> attention mode -> head
    (span / BIO / BIO+CRF) -> freeze/inject PEFT according to ``METHOD``.

    Args:
        cfg (dict): experiment config.

    Returns:
        nn.Module: ``NERModel`` | ``CRFNERModel`` | ``SpanNERModel``.
    """
    ckpt = cfg["CKPT"] or PHASE1_CKPT[cfg["ARCH"]]
    bb = build_backbone(cfg["ARCH"], ckpt, cfg)
    if cfg["KEEP_LAYERS"]: bb.prune(cfg["KEEP_LAYERS"])
    if hasattr(bb, "set_attention_mode"): bb.set_attention_mode(cfg["ATTN_MODE"] == "bidirectional", cfg["BIDIR_WINDOW"])
    if cfg.get("TASK", "bio") == "span": model = SpanNERModel(bb, S.N_TYPE, cfg["HEAD_DROPOUT"])
    else: model = (CRFNERModel if cfg.get("USE_CRF") else NERModel)(bb, len(S.TAGS), num_layers=cfg.get("HEAD_LAYERS", 2), use_bilstm=cfg.get("USE_BILSTM", True), dropout=cfg["HEAD_DROPOUT"])
    configure_trainable(model, cfg)
    tot, tr = count_params(model); print(f"task={model.task} params total={tot/1e6:.1f}M trainable={tr/1e6:.2f}M ({100*tr/tot:.2f}%)")
    return model
