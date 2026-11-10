"""GPT-2 / T5-encoder wrappers, checkpoint loading and ``build_backbone`` for all four Phase-1 architectures."""
import json
import os
import re

import torch
import torch.nn as nn

from ..config import ARCH_INFO
from .custom_decoder import CustomDecoderBackbone


class GPT2Backbone(nn.Module):
    """GPT-2 stack (random init, Phase-1 weights loaded afterwards).

    Args:
        cfg (GPT2Config): HF config.
    """
    def __init__(self, cfg):
        super().__init__()
        from transformers import GPT2Model
        self.m = GPT2Model(cfg); self.hidden_size = cfg.n_embd

    def num_layers(self):
        """Number of blocks.

        Args:
            None

        Returns:
            int: layer count.
        """
        return len(self.m.h)

    def prune(self, keep):
        """Keep only the listed layers.

        Args:
            keep (list[int]): layer indices.

        Returns:
            None
        """
        self.m.h = nn.ModuleList([self.m.h[i] for i in keep])

    def forward(self, input_ids, attention_mask=None):
        """Encode tokens.

        Args:
            input_ids (torch.Tensor): (B, T).
            attention_mask (torch.Tensor or None): padding mask.

        Returns:
            torch.Tensor: (B, T, dim).
        """
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class T5EncBackbone(nn.Module):
    """T5 encoder only (bidirectional).

    Args:
        cfg (T5Config): HF config.
    """
    def __init__(self, cfg):
        super().__init__()
        from transformers import T5EncoderModel
        self.m = T5EncoderModel(cfg); self.hidden_size = cfg.d_model

    def num_layers(self):
        """Number of blocks.

        Args:
            None

        Returns:
            int: layer count.
        """
        return len(self.m.encoder.block)

    def prune(self, keep):
        """Keep only the listed layers.

        Args:
            keep (list[int]): layer indices.

        Returns:
            None
        """
        self.m.encoder.block = nn.ModuleList([self.m.encoder.block[i] for i in keep])

    def forward(self, input_ids, attention_mask=None):
        """Encode tokens.

        Args:
            input_ids (torch.Tensor): (B, T).
            attention_mask (torch.Tensor or None): padding mask.

        Returns:
            torch.Tensor: (B, T, d_model).
        """
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def load_state(path):
    """Load a state dict from .pt/.pth/.bin (``{"model":sd}`` | ``{"state_dict":sd}`` | raw sd | pickled module), .safetensors, or an HF dir.

    Args:
        path (str): file or directory.

    Returns:
        dict: state dict with ``module.`` / ``model.`` prefixes stripped.
    """
    if os.path.isdir(path):
        for f in ("model.safetensors", "pytorch_model.bin"):
            if os.path.exists(os.path.join(path, f)): path = os.path.join(path, f); break
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file; sd = load_file(path)
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module): sd = obj.state_dict()
        elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict): sd = obj["model"]
        elif isinstance(obj, dict) and "state_dict" in obj: sd = obj["state_dict"]
        else: sd = obj
    return {re.sub(r"^(module\.|model\.)", "", k): v for k, v in sd.items()}


def _n_layers(sd, pat):
    """Infer the layer count from key names.

    Args:
        sd (dict): state dict.
        pat (str): regex with one capture group (layer index).

    Returns:
        int: ``max index + 1``.
    """
    return 1 + max(int(m.group(1)) for k in sd for m in [re.search(pat, k)] if m)


def _load_custom_config(ckpt_path):
    """Read ``config.json`` next to a checkpoint (the only reliable way to recover ``n_head``; ``dim // 64`` breaks for head_dim != 64).

    Args:
        ckpt_path (str): checkpoint directory or file.

    Returns:
        dict or None: parsed config or None.
    """
    d = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
    cfg_path = os.path.join(d, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return None


def build_backbone(arch, ckpt, cfg):
    """Rebuild a Phase-1 architecture (sizes inferred from the checkpoint / config.json) and load its weights.

    Args:
        arch (str): key of ``ARCH_INFO``.
        ckpt (str): checkpoint path.
        cfg (dict): experiment config (unused here, kept for API symmetry).

    Returns:
        nn.Module: backbone with ``hidden_size``, ``num_layers()`` and ``prune()``.

    Raises:
        AssertionError: if Phase-1 weights did not load.
    """
    info = ARCH_INFO[arch]
    raw_ckpt_path = ckpt   # keep the original (possibly a directory) before load_state() rewrites it
    sd = load_state(ckpt)
    if info["kind"] == "custom":
        V, dim = sd["tok_emb.weight"].shape
        hf_cfg = _load_custom_config(raw_ckpt_path)
        if hf_cfg is not None:
            n_head = hf_cfg["n_head"]
            hd = dim // n_head
            n_kv_head = hf_cfg.get("n_kv_head", sd["blocks.0.attn.k_proj.weight"].shape[0] // hd)
            d_ff = hf_cfg.get("d_ff", sd["blocks.0.ffn.w1.weight"].shape[0])
            n_layer = hf_cfg.get("n_layer", _n_layers(sd, r"^blocks\.(\d+)\."))
            window = hf_cfg.get("window_size", info.get("window"))
            print(f"[{arch}] exact config from config.json: n_head={n_head}, n_kv_head={n_kv_head}, head_dim={hd}")
        else:
            print(f"[{arch}] WARNING: no config.json next to checkpoint -- falling back to the "
                  f"shape-inference heuristic, which ASSUMES head_dim==64. Hardcode n_head for this "
                  f"arch in ARCH_INFO instead of trusting this fallback.")
            n_head = dim // 64
            hd = dim // n_head
            n_kv_head = sd["blocks.0.attn.k_proj.weight"].shape[0] // hd
            d_ff = sd["blocks.0.ffn.w1.weight"].shape[0]
            n_layer = _n_layers(sd, r"^blocks\.(\d+)\.")
            window = info.get("window")
        c = dict(vocab=V, dim=dim, n_head=n_head, n_layer=n_layer, d_ff=d_ff,
                 n_kv_head=n_kv_head, window=window, max_pos=2048)
        bb = CustomDecoderBackbone(c)
        sd = {k: v for k, v in sd.items() if not k.startswith("lm_head")}
    elif info["kind"] == "gpt2":
        from transformers import GPT2Config
        sd = {re.sub(r"^transformer\.", "", k): v for k, v in sd.items() if not k.startswith("lm_head")}
        V, dim = sd["wte.weight"].shape
        c = dict(vocab=V, dim=dim, n_head=dim // 64, n_layer=_n_layers(sd, r"^h\.(\d+)\."), n_pos=sd["wpe.weight"].shape[0])
        bb = GPT2Backbone(GPT2Config(vocab_size=V, n_positions=c["n_pos"], n_ctx=c["n_pos"], n_embd=dim, n_layer=c["n_layer"], n_head=c["n_head"]))
    elif info["kind"] == "t5":
        from transformers import T5Config
        V, dm = sd["shared.weight"].shape; inner = sd["encoder.block.0.layer.0.SelfAttention.q.weight"].shape[0]
        c = dict(vocab=V, d_model=dm, d_ff=sd["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"].shape[0], heads=inner // 64,
                 n_layer=_n_layers(sd, r"^encoder\.block\.(\d+)\."))
        bb = T5EncBackbone(T5Config(vocab_size=V, d_model=dm, d_kv=64, d_ff=c["d_ff"], num_layers=c["n_layer"], num_decoder_layers=c["n_layer"],
                                    num_heads=c["heads"], feed_forward_proj="gated-gelu", dropout_rate=0.1))
        sd = {k: v for k, v in sd.items() if k.startswith(("shared.", "encoder."))}
    if info["kind"] != "custom": sd = {("m." + k): v for k, v in sd.items()}
    res = bb.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if not re.search(r"attn\.bias|masked_bias|rope_", k)]
    print(f"[{arch}] {c} | missing={miss[:5]} unexpected={res.unexpected_keys[:5]}")
    assert not miss, "Phase-1 weights did not load into the backbone -- check ARCH_INFO / checkpoint format"
    return bb
