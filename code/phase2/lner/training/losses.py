"""Losses: span-head targets, focal / weighted CE, span loss, knowledge-distillation loss."""
import torch
import torch.nn.functional as F


def span_targets(lab):
    """Convert BIO tag ids into span-head targets.

    A token is an END if it is inside an entity and the NEXT non-ignored token does not continue it (I-same).
    A token with no following non-ignored token (window edge / doc end) gets end = -100 (unknown).

    Args:
        lab (torch.Tensor): (B, L) BIO tag ids (-100 = ignore).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``type`` (0=O, 1..12), ``start`` (0/1), ``end`` (0/1); -100 where ignored.
    """
    B, L = lab.shape; neg = torch.full_like(lab, -100); valid = lab >= 0
    typ = torch.where(valid, torch.div(lab + 1, 2, rounding_mode="floor"), neg)
    start = torch.where(valid, ((lab > 0) & (lab % 2 == 1)).long(), neg)
    BIG = L + 1
    pos = torch.where(valid, torch.arange(L, device=lab.device).expand(B, L), torch.full_like(lab, BIG))
    pos_next = torch.cat([pos[:, 1:], torch.full((B, 1), BIG, dtype=pos.dtype, device=lab.device)], 1)
    nxt = torch.flip(torch.cummin(torch.flip(pos_next, [1]), 1).values, [1])        # index of the next valid token after i
    has_next = nxt < BIG
    nxt_tag = lab.gather(1, nxt.clamp(max=L - 1))
    cont = has_next & (nxt_tag == 2 * typ)
    end = torch.where(valid & has_next, ((valid & (typ > 0)) & ~cont).long(), neg)
    return typ, start, end


def focal_ce(logits, target, weight, gamma):
    """Class-weighted focal cross-entropy.

    Args:
        logits (torch.Tensor): (N, C).
        target (torch.Tensor): (N,) with -100 = ignore.
        weight (torch.Tensor or None): (C,) class weights.
        gamma (float): focusing parameter.

    Returns:
        torch.Tensor: scalar loss.
    """
    m = target != -100
    if not m.any(): return logits.sum() * 0.0
    lg, tg = logits[m], target[m]
    logp = F.log_softmax(lg, -1).gather(1, tg[:, None]).squeeze(1); pt = logp.exp()
    w = weight[tg] if weight is not None else torch.ones_like(logp)
    return -(w * (1 - pt).pow(gamma) * logp).sum() / w.sum()


def ce_loss(logits, target, weight, cfg):
    """Weighted CE or focal loss according to ``cfg["LOSS"]``.

    Args:
        logits (torch.Tensor): (..., C).
        target (torch.Tensor): (...) class ids (-100 = ignore).
        weight (torch.Tensor or None): class weights.
        cfg (dict): config (``LOSS``, ``FOCAL_GAMMA``).

    Returns:
        torch.Tensor: scalar loss.
    """
    C = logits.size(-1)
    if cfg.get("LOSS", "ce") == "focal": return focal_ce(logits.reshape(-1, C), target.reshape(-1), weight, cfg.get("FOCAL_GAMMA", 2.0))
    return F.cross_entropy(logits.reshape(-1, C), target.reshape(-1), weight=weight, ignore_index=-100)


def span_loss(out, lab, cw, pos_w, n_types, cfg):
    """Span-head loss = type CE/focal + ``SPAN_BOUNDARY_W`` * (start BCE + end BCE).

    Args:
        out (torch.Tensor): (B, L, n_types + 2) logits.
        lab (torch.Tensor): (B, L) BIO tag ids.
        cw (torch.Tensor): type class weights.
        pos_w (float): positive weight for start/end BCE.
        n_types (int): number of types.
        cfg (dict): config.

    Returns:
        torch.Tensor: scalar loss.
    """
    typ, st, en = span_targets(lab)
    l_type = ce_loss(out[..., :n_types], typ, cw, cfg)
    def bce(logit, tgt):
        m = tgt >= 0
        if not m.any(): return logit.sum() * 0.0
        t = tgt[m].float()
        return F.binary_cross_entropy_with_logits(logit[m], t, weight=1.0 + (pos_w - 1.0) * t)
    return l_type + cfg.get("SPAN_BOUNDARY_W", 1.0) * (bce(out[..., n_types], st) + bce(out[..., n_types + 1], en))


def kd_loss_fn(s_logits, t_logits, labels, T, task="bio", n_types=None):
    """Distillation loss (temperature-scaled KL; BCE on start/end for the span task).

    Args:
        s_logits (torch.Tensor): student logits (B, L, C).
        t_logits (torch.Tensor): teacher logits.
        labels (torch.Tensor): (B, L) labels (-100 positions are ignored).
        T (float): temperature.
        task (str): ``bio`` | ``span``.
        n_types (int or None): number of types (span task).

    Returns:
        torch.Tensor: scalar loss.
    """
    m = labels != -100
    if task == "bio":
        return F.kl_div(F.log_softmax(s_logits[m] / T, -1), F.softmax(t_logits[m] / T, -1), reduction="batchmean") * (T * T)
    s, t = s_logits[m], t_logits[m]
    kl = F.kl_div(F.log_softmax(s[:, :n_types] / T, -1), F.softmax(t[:, :n_types] / T, -1), reduction="batchmean") * (T * T)
    return kl + F.binary_cross_entropy_with_logits(s[:, n_types:] / T, torch.sigmoid(t[:, n_types:] / T))
