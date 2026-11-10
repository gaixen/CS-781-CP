"""Continuation generation: reference HF-``generate`` implementation and a fast KV-cache generator for the custom decoders (self-checked against the reference)."""

import torch
import torch.nn.functional as F

from ..models.custom_decoder import apply_rope, build_rope_cache


def generate_continuations(
    prompts,
    model,
    tokenizer,
    device,
    target_words=128,
    max_new_tokens=384,
    temperature=0.1,
    top_p=0.95,
    is_seq2seq=False,
    batch_size=8,
    model_context_length=2048,
    seq2seq_src_tokens=700,
):
    """Reference generator (``model.generate``); outputs are truncated to ``target_words`` words.

    Args:
        prompts (list[str]): prompts.
        model (nn.Module): model.
        tokenizer (PreTrainedTokenizer): tokenizer (left padding).
        device (torch.device): device.
        target_words (int): words kept per continuation.
        max_new_tokens (int): generation budget.
        temperature (float): sampling temperature (0 -> greedy).
        top_p (float): nucleus sampling threshold.
        is_seq2seq (bool): encoder-decoder model.
        batch_size (int): prompts per batch.
        model_context_length (int): decoder context length.
        seq2seq_src_tokens (int): max source tokens for seq2seq.

    Returns:
        list[str]: continuations.
    """
    model.eval()
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq2seq_src_tokens
            if is_seq2seq
            else model_context_length - max_new_tokens,
        ).to(device)
        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(
            out_ids if is_seq2seq else out_ids[:, enc["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )  # strip prompt for causal LMs
        for text in decoded:
            all_outputs.append(" ".join(text.split()[:target_words]))
    return all_outputs


def _sample_next(logits, temperature, top_p, top_k):
    """Sample / argmax the next token with temperature, top-k and top-p.

    Args:
        logits (torch.Tensor): (B, V).
        temperature (float or None): <=0 -> greedy.
        top_p (float or None): nucleus threshold.
        top_k (int or None): top-k cutoff.

    Returns:
        torch.Tensor: (B,) token ids.
    """
    logits = logits.float()
    if temperature is None or temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        cum = sl.softmax(-1).cumsum(-1)
        remove = cum > top_p
        remove[..., 1:] = remove[
            ..., :-1
        ].clone()  # keep the first token that crosses top_p
        remove[..., 0] = False
        sl = sl.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


@torch.inference_mode()
def fast_generate_continuations(
    prompts,
    model,
    tokenizer,
    device,
    target_words=128,
    max_new_tokens=384,
    temperature=0.1,
    top_p=0.95,
    top_k=50,
    is_seq2seq=False,
    batch_size=16,
    use_autocast=True,
    check_every=16,
    **fallback_kw,
):
    """KV-cache generator for ``CustomDecoderForCausalLM`` (prefill once, then one token per step; sliding-window models only attend to the last ``window`` keys).

    Falls back to ``generate_continuations`` for seq2seq / non-custom models. Stops early once every sequence has more than ``target_words`` words.

    Args:
        prompts (list[str]): prompts.
        model (nn.Module): model.
        tokenizer (PreTrainedTokenizer): tokenizer.
        device (torch.device): device.
        target_words (int): words kept per continuation.
        max_new_tokens (int): generation budget.
        temperature (float): sampling temperature.
        top_p (float): nucleus threshold.
        top_k (int): top-k cutoff.
        is_seq2seq (bool): encoder-decoder model (uses the fallback).
        batch_size (int): prompts per batch.
        use_autocast (bool): fp16 autocast on CUDA.
        check_every (int): steps between early-stop checks.
        **fallback_kw: forwarded to ``generate_continuations`` in the fallback path.

    Returns:
        list[str]: continuations.
    """
    if is_seq2seq or not hasattr(model, "blocks"):
        return generate_continuations(
            prompts,
            model,
            tokenizer,
            device,
            target_words=target_words,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            is_seq2seq=is_seq2seq,
            batch_size=8,
            **fallback_kw,
        )
    model.eval()
    cfg = model.config
    head_dim = cfg.dim // cfg.n_head
    max_len = cfg.max_position_embeddings
    cos_all, sin_all = build_rope_cache(max_len, head_dim)
    cos_all, sin_all = cos_all.to(device), sin_all.to(device)
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"  # keep the END of long prompts
    amp = torch.autocast(
        "cuda", dtype=torch.float16, enabled=use_autocast and device.type == "cuda"
    )
    outputs = []

    for b0 in range(0, len(prompts), batch_size):
        enc = tokenizer(
            prompts[b0 : b0 + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len - max_new_tokens,
        )
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device).bool()
        B, T = ids.shape
        L = T + max_new_tokens
        caches = []

        with amp:
            # ---------------- prefill: one full pass over the prompt, fills the KV cache ----------------
            x = model.tok_emb(ids)
            idx = torch.arange(T, device=device)
            causal = idx[None, :] <= idx[:, None]
            eye = torch.eye(T, dtype=torch.bool, device=device)[None, None]
            for blk in model.blocks:
                a = blk.attn
                allowed = (
                    causal
                    if a.window_size is None
                    else causal & (idx[None, :] > idx[:, None] - a.window_size)
                )
                mask = (allowed[None, None] & am[:, None, None, :]) | eye
                h = blk.norm1(x)
                q = a.q_proj(h).view(B, T, a.n_head, head_dim).transpose(1, 2)
                k = a.k_proj(h).view(B, T, a.n_kv_head, head_dim).transpose(1, 2)
                v = a.v_proj(h).view(B, T, a.n_kv_head, head_dim).transpose(1, 2)
                if hasattr(a, "q_norm"):
                    q, k = a.q_norm(q), a.k_norm(k)
                q = apply_rope(q, cos_all[:T], sin_all[:T])
                k = apply_rope(k, cos_all[:T], sin_all[:T])
                kc = torch.empty(
                    B, a.n_kv_head, L, head_dim, dtype=k.dtype, device=device
                )
                vc = torch.empty(
                    B, a.n_kv_head, L, head_dim, dtype=v.dtype, device=device
                )
                kc[:, :, :T], vc[:, :, :T] = k, v
                caches.append((kc, vc))
                rep = a.n_head // a.n_kv_head
                kk = k.repeat_interleave(rep, 1) if rep > 1 else k
                vv = v.repeat_interleave(rep, 1) if rep > 1 else v
                o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
                x = x + a.o_proj(o.transpose(1, 2).contiguous().view(B, T, -1))
                x = x + blk.ffn(blk.norm2(x))
            logits = model.lm_head(
                model.norm_f(x[:, -1])
            )  # only the last position is needed

            key_ok = torch.ones(B, L, dtype=torch.bool, device=device)
            key_ok[:, :T] = am
            gen = torch.full(
                (B, max_new_tokens), pad_id, dtype=torch.long, device=device
            )
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            next_tok = _sample_next(logits, temperature, top_p, top_k)

            # ---------------- decode: one token per step, attends to the cache ----------------
            for step in range(max_new_tokens):
                next_tok = torch.where(
                    finished, torch.full_like(next_tok, pad_id), next_tok
                )
                gen[:, step] = next_tok
                finished |= next_tok == eos_id
                if (
                    step + 1
                ) % check_every == 0:  # lossless early stop: enough words already
                    texts = tokenizer.batch_decode(
                        gen[:, : step + 1], skip_special_tokens=True
                    )
                    finished |= torch.tensor(
                        [len(t.split()) > target_words for t in texts], device=device
                    )
                if finished.all() or step == max_new_tokens - 1:
                    break
                t = T + step  # sequence position of the token we feed now
                ct, st = cos_all[t : t + 1], sin_all[t : t + 1]
                x = model.tok_emb(next_tok[:, None])
                for blk, (kc, vc) in zip(model.blocks, caches):
                    a = blk.attn
                    h = blk.norm1(x)
                    q = a.q_proj(h).view(B, 1, a.n_head, head_dim).transpose(1, 2)
                    k = a.k_proj(h).view(B, 1, a.n_kv_head, head_dim).transpose(1, 2)
                    v = a.v_proj(h).view(B, 1, a.n_kv_head, head_dim).transpose(1, 2)
                    if hasattr(a, "q_norm"):
                        q, k = a.q_norm(q), a.k_norm(k)
                    q = apply_rope(q, ct, st)
                    k = apply_rope(k, ct, st)
                    kc[:, :, t] = k[:, :, 0]
                    vc[:, :, t] = v[:, :, 0]
                    lo = 0 if a.window_size is None else max(0, t + 1 - a.window_size)
                    rep = a.n_head // a.n_kv_head
                    qg = q.reshape(
                        B, a.n_kv_head, rep, head_dim
                    )  # GQA: the `rep` query heads of a kv-group act as rep "positions"
                    o = F.scaled_dot_product_attention(
                        qg,
                        kc[:, :, lo : t + 1],
                        vc[:, :, lo : t + 1],
                        attn_mask=key_ok[:, None, None, lo : t + 1],
                    )
                    x = x + a.o_proj(o.reshape(B, 1, a.n_head * head_dim))
                    x = x + blk.ffn(blk.norm2(x))
                logits = model.lm_head(model.norm_f(x[:, 0]))
                next_tok = _sample_next(logits, temperature, top_p, top_k)

        for text in tokenizer.batch_decode(gen, skip_special_tokens=True):
            outputs.append(" ".join(text.split()[:target_words]))
        del caches

    tokenizer.truncation_side = old_side
    return outputs


def make_gen_fn(model, tokenizer, device, internal_pairs, is_seq2seq=False):
    """Pick the generator: the fast KV-cache path only if it reproduces the reference greedy output on 2 prompts (fp32, 24 tokens).

    Args:
        model (nn.Module): model.
        tokenizer (PreTrainedTokenizer): tokenizer.
        device (torch.device): device.
        internal_pairs (list[dict]): eval pairs (``prompt`` used for the self-check).
        is_seq2seq (bool): encoder-decoder model.

    Returns:
        Callable: ``gen_fn(prompts, model, tokenizer, device, **kw) -> list[str]``.
    """
    fast_ok = False
    if internal_pairs and not is_seq2seq:
        p = [x["prompt"] for x in internal_pairs[:2]]
        slow = generate_continuations(
            p,
            model,
            tokenizer,
            device,
            target_words=10**6,
            max_new_tokens=24,
            temperature=0.0,
            batch_size=2,
        )
        fast = fast_generate_continuations(
            p,
            model,
            tokenizer,
            device,
            target_words=10**6,
            max_new_tokens=24,
            temperature=0.0,
            batch_size=2,
            use_autocast=False,
            check_every=10**9,
        )
        fast_ok = slow == fast
        print(
            "MATCH -- fast generation enabled"
            if fast_ok
            else "MISMATCH -- falling back to the slow generator"
        )
    else:
        print("Using the slow generator (no prompts to verify against / seq2seq).")
    batch = 16 if fast_ok else 8

    def gen_fn(prompts, model, tokenizer, device, **kw):
        """Generate continuations with the selected implementation.

        Args:
            prompts (list[str]): prompts.
            model (nn.Module): model.
            tokenizer (PreTrainedTokenizer): tokenizer.
            device (torch.device): device.
            **kw: generation options.

        Returns:
            list[str]: continuations.
        """
        kw.setdefault("batch_size", batch)
        return (fast_generate_continuations if fast_ok else generate_continuations)(
            prompts, model, tokenizer, device, is_seq2seq=is_seq2seq, **kw
        )

    return gen_fn
