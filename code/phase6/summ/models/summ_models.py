"""Phase-1 backbones as prefix-LM summarisers: ``[document] <SEP=EOS> [summary] <EOS>`` with loss on the summary only.

* :class:`SummLM`  - custom decoders (own attention pattern on the document, summary sees ALL document keys + causal over the summary), KV-cached generation.
* :class:`SummGPT2` - causal GPT-2 (left-padded documents, positions from the mask).
* :class:`SummT5`  - true encoder-decoder (document -> encoder, summary -> decoder).
All expose ``forward(doc, dmask, sin, smask, lab)``, ``begin(doc, dmask) -> step(prev_token)`` and ``generate(...)``.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from summ.config import arch_kind
from summ.models.custom_decoder import apply_rope, build_rope_cache
from summ.models.phase1_backbone import AdapterWrap, build_backbone, inject_adapters, inject_lorma, load_cfg_json, load_state, prune_layers


def autocast(ctx):
    """fp16 autocast context (CUDA + ``s.amp``).

    Args:
        ctx (Context): Context.

    Returns:
        torch.autocast: Context manager.
    """
    return torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=bool(ctx.s.amp and ctx.device.type == "cuda"))


def autocast_t5(ctx):
    """Autocast for T5 (disabled unless ``s.amp_t5``; T5 FFNs can overflow in fp16).

    Args:
        ctx (Context): Context.

    Returns:
        torch.autocast: Context manager.
    """
    return torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=bool(ctx.s.amp_t5 and ctx.device.type == "cuda"))


def summ_mask(Td, Ts, window, bidir, keypad, dev):
    """Boolean attention mask (B, 1, T, T).

    Document rows follow the architecture's own pattern restricted to document keys; summary rows see all document keys plus a causal summary.

    Args:
        Td (int): Document length.
        Ts (int): Summary length.
        window (Optional[int]): Sliding window.
        bidir (bool): Bidirectional document attention.
        keypad (torch.Tensor): (B, T) key padding mask.
        dev (torch.device): Device.

    Returns:
        torch.Tensor: Mask (True = may attend).
    """
    T = Td + Ts
    idx = torch.arange(T, device=dev)
    i, j = idx[:, None], idx[None, :]
    d = i - j
    if bidir:
        dm = (d.abs() < window) if window else torch.ones(T, T, dtype=torch.bool, device=dev)
    else:
        dm = d >= 0
        if window:
            dm = dm & (d < window)
    dm = dm & (j < Td)
    sm = (j < Td) | (j <= i)
    allowed = torch.where(i < Td, dm, sm)[None, None] & keypad[:, None, None, :].bool()
    return allowed | torch.eye(T, dtype=torch.bool, device=dev)[None, None]


def attn_kv(a, h, cos, sin, mask, past):
    """Grouped-query attention with an optional KV cache.

    Args:
        a (GQAFlashAttention): Attention module.
        h (torch.Tensor): Input (B, T, dim).
        cos (torch.Tensor): RoPE cosine.
        sin (torch.Tensor): RoPE sine.
        mask (torch.Tensor): Boolean mask.
        past (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached keys / values.

    Returns:
        Tuple[torch.Tensor, Tuple]: ``(output, (k, v))`` with the updated cache.
    """
    B, T, _ = h.shape
    q = a.q_proj(h).view(B, T, a.n_head, a.head_dim).transpose(1, 2)
    k = a.k_proj(h).view(B, T, a.n_kv_head, a.head_dim).transpose(1, 2)
    v = a.v_proj(h).view(B, T, a.n_kv_head, a.head_dim).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if past is not None:
        k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
    new = (k, v)
    if a.n_kv_head != a.n_head:
        r = a.n_head // a.n_kv_head
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return a.o_proj(o.transpose(1, 2).reshape(B, T, -1)), new


def block_fwd(blk, x, cos, sin, mask, past=None):
    """Decoder block forward with KV cache.

    Args:
        blk (DecoderBlock): Block.
        x (torch.Tensor): Input (B, T, dim).
        cos (torch.Tensor): RoPE cosine.
        sin (torch.Tensor): RoPE sine.
        mask (torch.Tensor): Boolean mask.
        past (Optional[tuple]): Cached keys / values.

    Returns:
        Tuple[torch.Tensor, tuple]: ``(output, new cache)``.
    """
    a, new = attn_kv(blk.attn, blk.norm1(x), cos, sin, mask, past)
    x = x + a
    return x + blk.ffn(blk.norm2(x)), new


def _cb(blk, x, cos, sin, mask):
    """Block forward without cache (for ``torch.utils.checkpoint``).

    Args:
        blk (DecoderBlock): Block.
        x (torch.Tensor): Input.
        cos (torch.Tensor): RoPE cosine.
        sin (torch.Tensor): RoPE sine.
        mask (torch.Tensor): Boolean mask.

    Returns:
        torch.Tensor: Block output.
    """
    return block_fwd(blk, x, cos, sin, mask)[0]


@torch.no_grad()
def decode_steps(ctx, steps, B, dev, max_new, min_new=0, nrep=3, rep_pen=1.0, weights=None):
    """Greedy decoding from one or several ``step()`` functions; several steps = weighted average of log-probabilities.

    Includes a repetition penalty and no-repeat n-gram blocking.

    Args:
        ctx (Context): Context (``eos``, ``nv``).
        steps (List[Callable]): Step functions returning next-token logits (B, V).
        B (int): Batch size.
        dev (torch.device): Device.
        max_new (int): Maximum new tokens.
        min_new (int): Minimum tokens before EOS is allowed.
        nrep (int): No-repeat n-gram size (0 = off).
        rep_pen (float): Repetition penalty.
        weights (Optional[Sequence[float]]): Member weights.

    Returns:
        List[List[int]]: Generated token ids per sequence (without EOS).
    """
    EOS, NV = ctx.eos, ctx.nv
    w = np.ones(len(steps)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    out = [[] for _ in range(B)]
    done = [False] * B
    ng = [dict() for _ in range(B)]
    tk = torch.full((B,), EOS, dtype=torch.long, device=dev)
    for s in range(max_new):
        lp = None
        for wi, st in zip(w, steps):
            l = F.log_softmax(st(tk)[:, :NV].float(), -1) * float(wi)
            lp = l if lp is None else lp + l
        if s < min_new:
            lp[:, EOS] = -1e9
        for b in range(B):
            if done[b]:
                continue
            if rep_pen != 1.0 and out[b]:
                ix = torch.tensor(list(set(out[b])), device=dev)
                v = lp[b, ix]
                lp[b, ix] = torch.where(v > 0, v / rep_pen, v * rep_pen)
            if nrep and len(out[b]) >= nrep - 1:
                ban = ng[b].get(tuple(out[b][len(out[b]) - (nrep - 1):]) if nrep > 1 else ())
                if ban:
                    lp[b, list(ban)] = -1e9
        nxt = lp.argmax(-1).tolist()
        for b in range(B):
            if done[b]:
                continue
            t = nxt[b]
            if t == EOS:
                done[b] = True
                continue
            out[b].append(t)
            if nrep and len(out[b]) >= nrep:
                ng[b].setdefault(tuple(out[b][-nrep:-1]), set()).add(t)
        if all(done):
            break
        tk = torch.tensor([EOS if done[b] else nxt[b] for b in range(B)], dtype=torch.long, device=dev)
    return out


class SummLM(nn.Module):
    """Custom Phase-1 decoder as a prefix-LM summariser with KV-cached generation.

    Args:
        bb (Backbone): Phase-1 backbone (custom kind).
        ctx (Context): Context (budgets, device, EOS).
    """

    def __init__(self, bb, ctx):
        super().__init__()
        self.ctx, self.lm = ctx, bb.model
        c = self.lm.config
        self.window, self.bidir = c.window_size, bool(getattr(c, "bidirectional", False))
        need = ctx.s.doc_tok + ctx.s.sum_tok + 8
        if self.lm.rope_cos.shape[0] < need:
            cs, sn = build_rope_cache(need, c.dim // c.n_head)
            self.lm.rope_cos, self.lm.rope_sin = cs, sn

    def forward(self, doc, dmask, sin_, smask, lab):
        """Teacher-forced forward pass.

        Args:
            doc (torch.Tensor): Left-padded document ids (B, Td).
            dmask (torch.Tensor): Document mask.
            sin_ (torch.Tensor): Summary input ids (B, Ts).
            smask (torch.Tensor): Summary mask.
            lab (torch.Tensor): Labels (-100 = ignore).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(logits at supervised positions, targets)``.
        """
        m, ctx = self.lm, self.ctx
        Td = doc.size(1)
        ids = torch.cat([doc, sin_], 1)
        km = torch.cat([dmask, smask], 1)
        T = ids.size(1)
        with autocast(ctx):
            x = m.tok_emb(ids)
            mask = summ_mask(Td, T - Td, self.window, self.bidir, km, x.device)
            cos, sin = m.rope_cos[:T], m.rope_sin[:T]
            for blk in m.blocks:
                x = torch_ckpt.checkpoint(_cb, blk, x, cos, sin, mask, use_reentrant=False) if (ctx.s.grad_ckpt and self.training and torch.is_grad_enabled()) else _cb(blk, x, cos, sin, mask)
            h = m.norm_f(x)[:, Td:]
            sel = lab != -100
            logits = m.lm_head(h[sel])
        return logits.float(), lab[sel]

    def begin(self, doc, dmask):
        """Prefill the document KV cache.

        Args:
            doc (torch.Tensor): Left-padded documents (B, Td).
            dmask (torch.Tensor): Document mask.

        Returns:
            Callable[[torch.Tensor], torch.Tensor]: ``step(prev_token (B,)) -> next-token logits (B, V)``.
        """
        m, ctx = self.lm, self.ctx
        B, Td = doc.shape
        dev = doc.device
        st = dict(s=0, keym=dmask.bool())
        with autocast(ctx):
            x = m.tok_emb(doc)
            mask = summ_mask(Td, 0, self.window, self.bidir, dmask, dev)
            cos, sin = m.rope_cos[:Td], m.rope_sin[:Td]
            st["pasts"] = []
            for blk in m.blocks:
                x, pk = block_fwd(blk, x, cos, sin, mask)
                st["pasts"].append(pk)

        def step(tok):
            """One decoding step.

            Args:
                tok (torch.Tensor): Previous tokens (B,).

            Returns:
                torch.Tensor: Next-token logits (B, V) in fp32.
            """
            st["keym"] = torch.cat([st["keym"], torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
            s = st["s"]
            with autocast(ctx):
                x = m.tok_emb(tok[:, None])
                cos, sin = m.rope_cos[Td + s:Td + s + 1], m.rope_sin[Td + s:Td + s + 1]
                for li, blk in enumerate(m.blocks):
                    x, st["pasts"][li] = block_fwd(blk, x, cos, sin, st["keym"][:, None, None, :], st["pasts"][li])
                lg = m.lm_head(m.norm_f(x))[:, 0].float()
            st["s"] += 1
            return lg
        return step

    @torch.no_grad()
    def generate(self, doc, dmask, max_new, min_new=0, nrep=3, rep_pen=1.0):
        """Greedy generation with a KV cache.

        Args:
            doc (torch.Tensor): Left-padded documents.
            dmask (torch.Tensor): Document mask.
            max_new (int): Maximum new tokens.
            min_new (int): Minimum tokens before EOS.
            nrep (int): No-repeat n-gram size.
            rep_pen (float): Repetition penalty.

        Returns:
            List[List[int]]: Token ids per document.
        """
        self.eval()
        return decode_steps(self.ctx, [self.begin(doc, dmask)], doc.size(0), doc.device, max_new, min_new, nrep, rep_pen)


def _enable_gc(m, grad_ckpt):
    """Enable HF gradient checkpointing when requested.

    Args:
        m (nn.Module): HF model.
        grad_ckpt (bool): Whether to enable it.

    Returns:
        None
    """
    if not grad_ckpt:
        return
    try:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        m.gradient_checkpointing_enable()
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()


class SummGPT2(nn.Module):
    """Causal GPT-2 LM summariser: ``[doc | EOS summary]``; documents are left-padded, positions come from the attention mask; tied LM head.

    Args:
        bb (Backbone): GPT-2 backbone.
        ctx (Context): Context.
    """

    def __init__(self, bb, ctx):
        super().__init__()
        self.ctx, self.tr, self.n_pos = ctx, bb.model, int(bb.model.config.n_positions)
        _enable_gc(self.tr, ctx.s.grad_ckpt)

    def logits(self, h):
        """Tied LM head.

        Args:
            h (torch.Tensor): Hidden states.

        Returns:
            torch.Tensor: Logits.
        """
        return F.linear(h, self.tr.wte.weight)

    def forward(self, doc, dmask, sin_, smask, lab):
        """Teacher-forced forward pass.

        Args:
            doc (torch.Tensor): Left-padded document ids.
            dmask (torch.Tensor): Document mask.
            sin_ (torch.Tensor): Summary input ids.
            smask (torch.Tensor): Summary mask.
            lab (torch.Tensor): Labels (-100 = ignore).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(logits, targets)``.
        """
        Td = doc.size(1)
        ids = torch.cat([doc, sin_], 1)
        am = torch.cat([dmask, smask], 1)
        pos = (am.long().cumsum(1) - 1).clamp(min=0)
        assert ids.size(1) <= self.n_pos, f"sequence {ids.size(1)} > GPT-2 n_positions {self.n_pos}"
        with autocast(self.ctx):
            h = self.tr(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False).last_hidden_state[:, Td:]
            sel = lab != -100
            lg = self.logits(h[sel])
        return lg.float(), lab[sel]

    def begin(self, doc, dmask):
        """Prefill the KV cache.

        Args:
            doc (torch.Tensor): Left-padded documents.
            dmask (torch.Tensor): Document mask.

        Returns:
            Callable[[torch.Tensor], torch.Tensor]: Step function.
        """
        B = doc.size(0)
        am = dmask.long()
        pos = (am.cumsum(1) - 1).clamp(min=0)
        with autocast(self.ctx):
            o = self.tr(input_ids=doc, attention_mask=am, position_ids=pos, use_cache=True)
        st = dict(past=o.past_key_values, am=am, nxt=pos[:, -1] + 1)

        def step(tok):
            """One decoding step.

            Args:
                tok (torch.Tensor): Previous tokens (B,).

            Returns:
                torch.Tensor: Next-token logits.
            """
            st["am"] = torch.cat([st["am"], torch.ones(B, 1, dtype=st["am"].dtype, device=st["am"].device)], 1)
            with autocast(self.ctx):
                o = self.tr(input_ids=tok[:, None], attention_mask=st["am"], position_ids=st["nxt"][:, None], past_key_values=st["past"], use_cache=True)
            st["past"] = o.past_key_values
            st["nxt"] = st["nxt"] + 1
            return self.logits(o.last_hidden_state[:, 0]).float()
        return step

    @torch.no_grad()
    def generate(self, doc, dmask, max_new, min_new=0, nrep=3, rep_pen=1.0):
        """Greedy generation (``max_new`` is capped by the learned positions).

        Args:
            doc (torch.Tensor): Left-padded documents.
            dmask (torch.Tensor): Document mask.
            max_new (int): Maximum new tokens.
            min_new (int): Minimum tokens before EOS.
            nrep (int): No-repeat n-gram size.
            rep_pen (float): Repetition penalty.

        Returns:
            List[List[int]]: Token ids.
        """
        self.eval()
        max_new = max(1, min(max_new, self.n_pos - doc.size(1) - 1))
        return decode_steps(self.ctx, [self.begin(doc, dmask)], doc.size(0), doc.device, max_new, min_new, nrep, rep_pen)


class SummT5(nn.Module):
    """True encoder-decoder summariser: document -> encoder, summary -> decoder (teacher forcing); greedy decoding with a decoder KV cache.

    Args:
        model (T5ForConditionalGeneration): T5 model.
        ctx (Context): Context.
    """

    def __init__(self, model, ctx):
        super().__init__()
        self.ctx, self.m, self.cfg = ctx, model, model.config
        self.start = self.cfg.decoder_start_token_id if self.cfg.decoder_start_token_id is not None else ctx.eos
        _enable_gc(model, ctx.s.grad_ckpt)

    def head(self, h):
        """LM head with the tied-embedding rescaling of T5.

        Args:
            h (torch.Tensor): Decoder hidden states.

        Returns:
            torch.Tensor: Logits.
        """
        if self.cfg.tie_word_embeddings:
            h = h * (self.cfg.d_model ** -0.5)
        return self.m.lm_head(h)

    def forward(self, doc, dmask, sin_, smask, lab):
        """Teacher-forced forward pass.

        Args:
            doc (torch.Tensor): Left-padded document ids.
            dmask (torch.Tensor): Document mask.
            sin_ (torch.Tensor): Decoder inputs (first token replaced by the start token).
            smask (torch.Tensor): Decoder mask.
            lab (torch.Tensor): Labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(logits, targets)``.
        """
        dec = sin_.clone()
        dec[:, 0] = self.start
        with autocast_t5(self.ctx):
            enc = self.m.encoder(input_ids=doc, attention_mask=dmask, use_cache=False).last_hidden_state
            h = self.m.decoder(input_ids=dec, attention_mask=smask, encoder_hidden_states=enc, encoder_attention_mask=dmask, use_cache=False).last_hidden_state
            sel = lab != -100
            lg = self.head(h[sel])
        return lg.float(), lab[sel]

    def begin(self, doc, dmask):
        """Encode the document and return a step function with a decoder cache.

        Args:
            doc (torch.Tensor): Left-padded documents.
            dmask (torch.Tensor): Document mask.

        Returns:
            Callable[[torch.Tensor], torch.Tensor]: Step function.
        """
        with autocast_t5(self.ctx):
            enc = self.m.encoder(input_ids=doc, attention_mask=dmask, use_cache=False).last_hidden_state
        st = dict(past=None, s=0)

        def step(tok):
            """One decoding step (the first fed token is the decoder start token).

            Args:
                tok (torch.Tensor): Previous tokens (B,).

            Returns:
                torch.Tensor: Next-token logits.
            """
            t = torch.full_like(tok, self.start) if st["s"] == 0 else tok
            with autocast_t5(self.ctx):
                o = self.m.decoder(input_ids=t[:, None], encoder_hidden_states=enc, encoder_attention_mask=dmask, past_key_values=st["past"], use_cache=True)
                lg = self.head(o.last_hidden_state[:, 0]).float()
            st["past"] = o.past_key_values
            st["s"] += 1
            return lg
        return step

    @torch.no_grad()
    def generate(self, doc, dmask, max_new, min_new=0, nrep=3, rep_pen=1.0):
        """Greedy generation.

        Args:
            doc (torch.Tensor): Left-padded documents.
            dmask (torch.Tensor): Document mask.
            max_new (int): Maximum new tokens.
            min_new (int): Minimum tokens before EOS.
            nrep (int): No-repeat n-gram size.
            rep_pen (float): Repetition penalty.

        Returns:
            List[List[int]]: Token ids.
        """
        self.eval()
        return decode_steps(self.ctx, [self.begin(doc, dmask)], doc.size(0), doc.device, max_new, min_new, nrep, rep_pen)


def _peft_flags(named, peft):
    """Set ``requires_grad``: PEFT parameters (+ final norms) only, or everything for full fine-tuning.

    Args:
        named (Iterable[Tuple[str, nn.Parameter]]): Named parameters.
        peft (bool): True for LoRMA+ / adapters.

    Returns:
        None
    """
    for n, p in named:
        p.requires_grad = (getattr(p, "_is_peft", False) or n.endswith(("norm_f.weight", "ln_f.weight", "ln_f.bias", "final_layer_norm.weight"))) if peft else True


class _Holder(nn.Module):
    """Minimal holder so ``inject_lorma`` can treat a T5 model like a backbone.

    Args:
        model (nn.Module): Wrapped model.
        kind (str): Backbone kind.
    """

    def __init__(self, model, kind):
        super().__init__()
        self.model, self.kind = model, kind


def _build_t5(ctx, cfg):
    """Build the T5 summariser from a Phase-1 T5 checkpoint (needs decoder weights unless ``allow_t5_random_decoder``).

    Args:
        ctx (Context): Context.
        cfg (dict): Experiment definition.

    Returns:
        SummT5: Model on the device.
    """
    from transformers import T5Config, T5ForConditionalGeneration
    s = ctx.s
    ck = s.phase1_ckpt[cfg["arch"]]
    sd, cj = load_state(ck), load_cfg_json(ck)
    c = T5Config(**cj) if cj else T5Config()
    model = T5ForConditionalGeneration(c)
    res = model.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if not k.endswith(("embed_tokens.weight", "lm_head.weight"))]
    dmiss = [k for k in miss if k.startswith("decoder.")]
    print(f"  [t5] missing={len(miss)} (decoder: {len(dmiss)}) unexpected={len(res.unexpected_keys)} | d_model={c.d_model} layers={c.num_layers}/{c.num_decoder_layers} vocab={c.vocab_size} tied={c.tie_word_embeddings}")
    if miss and not s.allow_t5_random_decoder:
        raise RuntimeError(f"T5 Phase-1 checkpoint is missing {len(miss)} weights (e.g. {miss[:3]}). Summarisation needs the DECODER; set allow_t5_random_decoder=True only if you accept a randomly initialised decoder.")
    if cfg.get("scratch"):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)
    holder = _Holder(model, "t5")
    if cfg["method"] == "lorma":
        print("  LoRMA+ modules:", inject_lorma(holder, cfg.get("which", "attn_ffn"), cfg["r"], cfg["alpha"]))
    elif cfg["method"] == "adapters":
        n = 0
        for blk in list(model.encoder.block) + list(model.decoder.block):
            ff = blk.layer[-1]
            ff.DenseReluDense = AdapterWrap(ff.DenseReluDense, c.d_model, cfg["r"])
            n += 1
        print("  adapters:", n)
    elif cfg["method"] == "kd":
        raise ValueError("KD pruning is implemented for the decoder-only archs, not T5")
    _peft_flags(model.named_parameters(), cfg["method"] in ("lorma", "adapters"))
    return SummT5(model, ctx).to(ctx.device)


def build_model(ctx, cfg):
    """Build a summariser for an experiment (backbone + technique + trainable mask).

    Args:
        ctx (Context): Context.
        cfg (dict): Experiment definition (``arch``, ``method`` and technique hyper-parameters).

    Returns:
        nn.Module: :class:`SummLM`, :class:`SummGPT2` or :class:`SummT5` on the device.
    """
    kind = arch_kind(cfg["arch"])
    if kind == "t5":
        return _build_t5(ctx, cfg)
    bb = build_backbone(cfg["arch"], ctx.s.phase1_ckpt[cfg["arch"]], ctx.s.strict_load)
    if cfg.get("scratch"):
        for p in bb.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)
        print("  [ablation] weights re-initialised: NOT using Phase-1 weights")
    if cfg["method"] == "kd":
        print("  student layers:", prune_layers(bb, cfg.get("keep_every", 2)), "(before, after)")
    if cfg["method"] == "lorma":
        print("  LoRMA+ modules:", inject_lorma(bb, cfg.get("which", "attn_ffn"), cfg["r"], cfg["alpha"]))
    if cfg["method"] == "adapters":
        print("  adapters:", inject_adapters(bb, cfg["r"]))
    _peft_flags(bb.named_parameters(), cfg["method"] in ("lorma", "adapters"))
    return (SummGPT2(bb, ctx) if kind == "gpt2" else SummLM(bb, ctx)).to(ctx.device)


def n_params(model):
    """Trainable and total parameter counts.

    Args:
        model (nn.Module): Model.

    Returns:
        Tuple[int, int]: ``(trainable, total)``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
