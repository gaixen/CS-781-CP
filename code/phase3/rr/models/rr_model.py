"""Rhetorical-role model: Phase-1 backbone -> sentence vectors -> [BiLSTM over a window] -> emissions [-> CRF] [-> label-shift head]."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rr.models.backbones import build_backbone
from rr.models.crf import CRF
from rr.models.peft import configure_trainable, count_params


def pool(h, tm, mode):
    """Pool token states of the TARGET sentence.

    Args:
        h (torch.Tensor): Hidden states (N, T, H).
        tm (torch.Tensor): Target-token mask (N, T).
        mode (str): ``mean`` or ``last``.

    Returns:
        torch.Tensor: Sentence vectors (N, H), float32.
    """
    h = h.float()
    if mode == "last":
        idx = (tm * torch.arange(tm.size(1), device=tm.device)).argmax(1)
        return h[torch.arange(h.size(0), device=h.device), idx]
    w = tm.unsqueeze(-1).float()
    return (h * w).sum(1) / w.sum(1).clamp(min=1)


class RRModel(nn.Module):
    """Sentence encoder + document-level head.

    Args:
        backbone (nn.Module): Phase-1 backbone.
        n_lab (int): Number of labels.
        cfg (dict): Config (``HID, SEQ, USE_CRF, SHIFT_LAMBDA, HEAD_DROPOUT, POOL``).
        device (torch.device): Device (autocast control inside the head).
    """

    def __init__(self, backbone, n_lab, cfg, device):
        super().__init__()
        H, D = backbone.hidden_size, 2 * cfg["HID"]
        self.cfg, self.n_lab, self.D, self.device = cfg, n_lab, D, device
        self.backbone = backbone
        self.drop = nn.Dropout(cfg["HEAD_DROPOUT"])
        self.proj = nn.Linear(H, D)
        self.ln = nn.LayerNorm(D)
        self.lstm = nn.LSTM(D, cfg["HID"], batch_first=True, bidirectional=True) if cfg["SEQ"] == "bilstm" else None
        self.emit = nn.Linear(D, n_lab)
        self.crf = CRF(n_lab) if cfg["USE_CRF"] else None
        self.shift = nn.Sequential(nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, 1)) if cfg["SHIFT_LAMBDA"] > 0 else None

    def encode(self, ids, am, tm):
        """Sentence vectors of a batch of (context) inputs.

        Args:
            ids (torch.Tensor): Token ids (N, T).
            am (torch.Tensor): Attention mask (N, T).
            tm (torch.Tensor): Target mask (N, T).

        Returns:
            torch.Tensor: (N, H) float32.
        """
        return pool(self.backbone(ids, am), tm, self.cfg["POOL"])

    def seq(self, V, lens):
        """Document-level sequence head (projection, LayerNorm, residual BiLSTM).

        Args:
            V (torch.Tensor): Padded sentence vectors (B, W, H).
            lens (List[int]): True window lengths.

        Returns:
            torch.Tensor: Contextual features (B, W, D).
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            x = self.ln(self.proj(self.drop(V.float())))
            if self.lstm is not None:
                pk = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(lens), batch_first=True, enforce_sorted=False)
                x = x + self.drop(nn.utils.rnn.pad_packed_sequence(self.lstm(pk)[0], batch_first=True, total_length=x.size(1))[0])
        return x

    def emissions(self, x):
        """Label logits per sentence.

        Args:
            x (torch.Tensor): Features (B, W, D).

        Returns:
            torch.Tensor: Emissions (B, W, n_lab), float32.
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self.emit(self.drop(x)).float()

    def posteriors(self, em, mask):
        """Per-sentence label posteriors (CRF marginals if a CRF is used, else softmax).

        Args:
            em (torch.Tensor): Emissions (B, W, n_lab).
            mask (torch.Tensor): Bool (B, W).

        Returns:
            torch.Tensor: Posteriors (B, W, n_lab).
        """
        return self.crf.marginals(em, mask) if self.crf is not None else F.softmax(em, -1)


def build_rr_model(ctx, cfg):
    """Build backbone, optional layer pruning / attention mode, head and trainable-parameter mask.

    Args:
        ctx (Context): Pipeline context (checkpoint registry, label count, device).
        cfg (dict): Experiment config.

    Returns:
        RRModel: Model (on CPU until training moves it).
    """
    bb = build_backbone(cfg["ARCH"], cfg["CKPT"] or ctx.settings.phase1_ckpt[cfg["ARCH"]])
    if cfg["KEEP_LAYERS"]:
        bb.prune(cfg["KEEP_LAYERS"])
    if hasattr(bb, "set_attention_mode"):
        bb.set_attention_mode(cfg["ATTN_MODE"] == "bidirectional", cfg["BIDIR_WINDOW"])
    model = RRModel(bb, ctx.data.nl, cfg, ctx.device)
    configure_trainable(model, cfg)
    tot, tr = count_params(model)
    print(f"RRModel params total={tot/1e6:.1f}M trainable={tr/1e6:.2f}M ({100*tr/tot:.2f}%)")
    return model
