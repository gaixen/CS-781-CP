"""Phase-1 backbone loading: checkpoint reading, architecture reconstruction, uniform ``Backbone`` wrapper."""
import os
import json
import re

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel, T5Config, T5EncoderModel

from .custom_decoder import CustomDecoderConfig, CustomDecoderForCausalLM

ARCH_INFO = {
    "gpt2_decoder":                     {"kind": "gpt2"},
    "modern_decoder":                   {"kind": "custom", "window": None, "bidirectional": False},
    "lightweight_sliding_window":       {"kind": "custom", "window": 256,  "bidirectional": False},
    "lightweight_sliding_window_bidir": {"kind": "custom", "window": 256,  "bidirectional": True},
    "seq2seq_t5":                       {"kind": "t5"},
}
_TOLERATED_MISSING = ("lm_head.weight", "attn.bias", "attn.masked_bias", "encoder.embed_tokens.weight", "rope_cos", "rope_sin")


def load_state(ckpt):
    """Load a state dict from a checkpoint directory or file, stripping ``torch.compile``/DDP prefixes.

    Args:
        ckpt (str): directory (with ``model.safetensors`` or ``pytorch_model.bin``) or a weights file.

    Returns:
        dict: cleaned ``name -> tensor`` state dict.
    """
    if os.path.isdir(ckpt):
        p = os.path.join(ckpt, "model.safetensors")
        sd = load_file(p) if os.path.exists(p) else torch.load(os.path.join(ckpt, "pytorch_model.bin"), map_location="cpu")
    else:
        sd = load_file(ckpt) if ckpt.endswith(".safetensors") else torch.load(ckpt, map_location="cpu")
    out = {}
    for k, v in sd.items():            # strip torch.compile / DDP prefixes
        k = k.replace("_orig_mod.", "")
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def load_cfg_json(ckpt):
    """Read ``config.json`` next to a checkpoint.

    Args:
        ckpt (str): checkpoint directory or file path.

    Returns:
        dict or None: parsed config, or None if no ``config.json`` exists.
    """
    d = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
    p = os.path.join(d, "config.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _n_layers(sd, pattern):
    """Infer the number of layers from state-dict key names.

    Args:
        sd (dict): state dict.
        pattern (str): regex with one capture group for the layer index.

    Returns:
        int: ``max index + 1`` (0 if nothing matched).
    """
    idxs = {int(m.group(1)) for k in sd if (m := re.match(pattern, k))}
    return max(idxs) + 1 if idxs else 0


def _check_load(arch, res, strict):
    """Report (or raise on) checkpoint weights that were missing at load time.

    Args:
        arch (str): architecture name (for messages).
        res: result of ``load_state_dict(strict=False)``.
        strict (bool): raise instead of warning when non-tolerated weights are missing.

    Returns:
        None

    Raises:
        RuntimeError: if ``strict`` and weights are missing.
    """
    missing = [k for k in res.missing_keys if not k.endswith(_TOLERATED_MISSING)]
    print(f"[{arch}] load: missing={len(missing)} (non-tolerated) unexpected={len(res.unexpected_keys)}")
    if missing:
        msg = f"[{arch}] {len(missing)} weights NOT found in checkpoint (would be randomly initialised): {missing[:4]}"
        if strict: raise RuntimeError(msg + "  -- check the checkpoint path/arch, or set STRICT_LOAD=False")
        print("WARNING:", msg)


class Backbone(nn.Module):
    """Uniform wrapper: ``forward(ids, mask) -> (B, T, hidden_dim)`` for every Phase-1 architecture.

    Args:
        model (nn.Module): the underlying HF/custom model.
        kind (str): ``"custom"`` | ``"gpt2"`` | ``"t5"``.
        hidden_dim (int): hidden size of the output states.
        max_len (int): maximum supported context length.
        bidirectional (bool): whether the backbone attends bidirectionally.
    """
    def __init__(self, model, kind, hidden_dim, max_len, bidirectional):
        super().__init__()
        self.model, self.kind, self.hidden_dim, self.max_len, self.bidirectional = model, kind, hidden_dim, max_len, bidirectional

    def forward(self, input_ids, attention_mask):
        """Encode tokens.

        Args:
            input_ids (torch.Tensor): (B, T) token ids.
            attention_mask (torch.Tensor): (B, T) padding mask (1 = real token).

        Returns:
            torch.Tensor: (B, T, hidden_dim) last hidden states.
        """
        if self.kind == "custom": return self.model.forward_hidden(input_ids, attention_mask)
        if self.kind == "gpt2":   return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def get_layers(self):
        """Return the transformer layer list.

        Args:
            None

        Returns:
            nn.ModuleList: the stack of blocks.
        """
        return {"custom": lambda: self.model.blocks, "gpt2": lambda: self.model.h, "t5": lambda: self.model.encoder.block}[self.kind]()

    def set_layers(self, layers):
        """Replace the layer stack (used for layer pruning).

        Args:
            layers (Iterable[nn.Module]): new layers.

        Returns:
            None
        """
        layers = nn.ModuleList(layers)
        if self.kind == "custom": self.model.blocks = layers; self.model.config.n_layer = len(layers)
        elif self.kind == "gpt2": self.model.h = layers; self.model.config.n_layer = len(layers)
        else: self.model.encoder.block = layers; self.model.config.num_layers = len(layers)

    def get_final_norm(self):
        """Return the final normalisation layer.

        Args:
            None

        Returns:
            nn.Module: the last norm of the backbone.
        """
        return {"custom": lambda: self.model.norm_f, "gpt2": lambda: self.model.ln_f, "t5": lambda: self.model.encoder.final_layer_norm}[self.kind]()

    def get_embeddings(self):
        """Return the input embedding layer.

        Args:
            None

        Returns:
            nn.Embedding: token embeddings.
        """
        return self.model.get_input_embeddings()


def build_backbone(arch, ckpt, strict=True):
    """Rebuild a Phase-1 architecture and load its checkpoint.

    Args:
        arch (str): key of ``ARCH_INFO``.
        ckpt (str or None): Phase-1 checkpoint directory/file.
        strict (bool): raise if checkpoint weights are missing (otherwise they would silently stay random).

    Returns:
        Backbone: the loaded backbone wrapper.

    Raises:
        ValueError: if ``ckpt`` is None.
        RuntimeError: if ``strict`` and weights are missing.
    """
    if ckpt is None: raise ValueError(f"No Phase-1 checkpoint path set for '{arch}' (PHASE1_CKPT).")
    info = ARCH_INFO[arch]; sd = load_state(ckpt); cfg_json = load_cfg_json(ckpt)
    if info["kind"] == "custom":
        V, dim = sd["tok_emb.weight"].shape
        bidir = info.get("bidirectional", False)
        if cfg_json is not None:
            n_head = cfg_json["n_head"]; hd = dim // n_head
            n_kv = cfg_json.get("n_kv_head", sd["blocks.0.attn.k_proj.weight"].shape[0] // hd)
            d_ff = cfg_json.get("d_ff", sd["blocks.0.ffn.w1.weight"].shape[0])
            n_layer = cfg_json.get("n_layer", _n_layers(sd, r"^blocks\.(\d+)\."))
            window = cfg_json.get("window_size", info.get("window"))
            bidir = cfg_json.get("bidirectional", cfg_json.get("bidir", bidir))
            max_pos = max(2048, cfg_json.get("max_position_embeddings", 2048))
        else:
            print(f"[{arch}] WARNING: no config.json -- assuming head_dim==64, verify it.")
            n_head = dim // 64; hd = dim // n_head
            n_kv = sd["blocks.0.attn.k_proj.weight"].shape[0] // hd
            d_ff = sd["blocks.0.ffn.w1.weight"].shape[0]; n_layer = _n_layers(sd, r"^blocks\.(\d+)\."); window = info.get("window"); max_pos = 2048
        cfg = CustomDecoderConfig(vocab_size=V, dim=dim, n_head=n_head, n_layer=n_layer, d_ff=d_ff, n_kv_head=n_kv,
                                  window_size=window, bidirectional=bool(bidir), max_position_embeddings=max_pos)
        model = CustomDecoderForCausalLM(cfg); _check_load(arch, model.load_state_dict(sd, strict=False), strict)
        bb = Backbone(model, "custom", dim, max_pos, bool(bidir))
    elif info["kind"] == "gpt2":
        keep = set(GPT2Config().to_dict())
        cfg = GPT2Config(**{k: v for k, v in cfg_json.items() if k in keep}) if cfg_json else GPT2Config()
        lm = GPT2LMHeadModel(cfg); _check_load(arch, lm.load_state_dict(sd, strict=False), strict)
        bb = Backbone(lm.transformer, "gpt2", cfg.n_embd, cfg.n_positions, False)
    else:  # t5: ENCODER only (bidirectional) -- good fit for whole-document classification
        cfg = T5Config(**cfg_json) if cfg_json else T5Config()
        model = T5EncoderModel(cfg); _check_load(arch, model.load_state_dict(sd, strict=False), strict)
        bb = Backbone(model, "t5", cfg.d_model, 10 ** 9, True)
    n = sum(p.numel() for p in bb.parameters())
    print(f"[{arch}] {n/1e6:.1f}M params, hidden_dim={bb.hidden_dim}, layers={len(bb.get_layers())}, bidirectional={bb.bidirectional}")
    return bb
