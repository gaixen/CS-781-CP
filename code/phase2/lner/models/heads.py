"""Token-classification heads: BIO (BiLSTM + residual MLP), span (start/end/type), and a linear-chain CRF on top of the BIO head."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoding.spans import _transition_tables


class NERModel(nn.Module):
    """Deep token-classification head with optional BiLSTM context and residual MLPs.

    Args:
        backbone (nn.Module): backbone exposing ``hidden_size``.
        n_tags (int): number of BIO tags.
        num_layers (int): number of MLP layers in the residual block.
        use_bilstm (bool): add a BiLSTM over the backbone states.
        dropout (float): dropout probability.
    """
    task = "bio"

    def __init__(self, backbone, n_tags, num_layers=2, use_bilstm=True, dropout=0.1):
        super().__init__()
        H = backbone.hidden_size
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.n_out = n_tags
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.bilstm = nn.LSTM(H, H // 2, num_layers=1, batch_first=True, bidirectional=True)
        mlp_layers = []
        for _ in range(num_layers):
            mlp_layers.extend([nn.Linear(H, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*mlp_layers)
        self.classifier = nn.Linear(H, n_tags)

    def forward(self, input_ids, attention_mask):
        """Compute tag logits.

        Args:
            input_ids (torch.Tensor): (B, T) token ids.
            attention_mask (torch.Tensor): (B, T) padding mask.

        Returns:
            torch.Tensor: (B, T, n_tags) float32 logits.
        """
        h = self.backbone(input_ids, attention_mask)
        if self.use_bilstm:
            h, _ = self.bilstm(h)
        h = h + self.mlp(h)  # Residual connection
        return self.classifier(self.drop(h)).float()

    def to_probs(self, logits):
        """Softmax over tags.

        Args:
            logits (torch.Tensor): (B, T, n_tags).

        Returns:
            torch.Tensor: probabilities.
        """
        return F.softmax(logits, dim=-1)


class SpanNERModel(nn.Module):
    """Deep span head with dedicated trunks for Start, End and Entity-Type prediction.

    Output columns: ``[type logits (n_types) | start logit | end logit]``.

    Args:
        backbone (nn.Module): backbone exposing ``hidden_size``.
        n_types (int): number of types incl. O.
        dropout (float): dropout probability.
    """
    task = "span"

    def __init__(self, backbone, n_types, dropout=0.1):
        super().__init__()
        H = backbone.hidden_size
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.n_types = n_types
        self.n_out = n_types + 2
        self.trunk = nn.Sequential(nn.Linear(H, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, H), nn.LayerNorm(H), nn.GELU())
        self.type_head = nn.Sequential(nn.Linear(H, H // 2), nn.LayerNorm(H // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(H // 2, n_types))
        self.start_head = nn.Sequential(nn.Linear(H, H // 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(H // 4, 1))
        self.end_head = nn.Sequential(nn.Linear(H, H // 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(H // 4, 1))

    def forward(self, input_ids, attention_mask):
        """Compute type/start/end logits.

        Args:
            input_ids (torch.Tensor): (B, T) token ids.
            attention_mask (torch.Tensor): (B, T) padding mask.

        Returns:
            torch.Tensor: (B, T, n_types + 2) float32 logits.
        """
        h_base = self.backbone(input_ids, attention_mask)
        h = h_base + self.trunk(self.drop(h_base))  # Residual feature refinement
        return torch.cat([self.type_head(h), self.start_head(h), self.end_head(h)], dim=-1).float()

    def to_probs(self, logits):
        """Softmax over types, sigmoid for start/end.

        Args:
            logits (torch.Tensor): (B, T, n_types + 2).

        Returns:
            torch.Tensor: probabilities with the same layout.
        """
        return torch.cat([F.softmax(logits[..., :self.n_types], dim=-1), torch.sigmoid(logits[..., self.n_types:])], dim=-1)


class LinearCRF(nn.Module):
    """Linear-chain CRF with hard BIO constraints and PARTIAL-label training.

    O->I-X and B-X->I-Y are masked (not learned). Tokens with label -100 (whitespace tokens, masked window edges, padding)
    stay in the chain as free positions, so train and inference see the same chain.

    Args:
        n_tags (int): number of tags.
    """
    def __init__(self, n_tags):
        super().__init__()
        allowed, _ = _transition_tables()
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags)); self.start = nn.Parameter(torch.zeros(n_tags)); self.end = nn.Parameter(torch.zeros(n_tags))
        self.register_buffer("tmask", torch.tensor(np.where(allowed, 0.0, -1e4), dtype=torch.float32))

    def _trans(self):
        """Constrained transition matrix.

        Args:
            None

        Returns:
            torch.Tensor: learned transitions + hard mask.
        """
        return self.trans + self.tmask

    def _alphas(self, em, mask):
        """Forward recursion.

        Args:
            em (torch.Tensor): (B, L, C) fp32 emissions.
            mask (torch.Tensor): (B, L) bool, True = real token (padding only on the right).

        Returns:
            list[torch.Tensor]: alpha vectors per position.
        """
        T = self._trans(); alpha = self.start + em[:, 0]; out = [alpha]
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + T.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha); out.append(alpha)
        return out

    def log_partition(self, em, mask):
        """Log partition function.

        Args:
            em (torch.Tensor): (B, L, C) emissions.
            mask (torch.Tensor): (B, L) bool.

        Returns:
            torch.Tensor: (B,) log Z.
        """
        return torch.logsumexp(self._alphas(em, mask)[-1] + self.end, dim=1)

    def nll(self, em, labels, mask):
        """Partial-label negative log-likelihood per labelled token.

        Args:
            em (torch.Tensor): (B, L, C) emissions.
            labels (torch.Tensor): (B, L) tag ids (-100 = free).
            mask (torch.Tensor): (B, L) bool.

        Returns:
            torch.Tensor: scalar loss.
        """
        ok = labels >= 0; gold = labels.clamp(min=0).unsqueeze(-1)
        only_gold = torch.full_like(em, -1e4).scatter(2, gold, em.gather(2, gold))
        em_c = torch.where(ok.unsqueeze(-1), only_gold, em)                 # labelled token: only its gold tag; free token: all tags
        return (self.log_partition(em, mask) - self.log_partition(em_c, mask)).sum() / ok.sum().clamp(min=1)

    @torch.no_grad()
    def marginals(self, em, mask):
        """Per-token marginals via forward-backward.

        Args:
            em (torch.Tensor): (B, L, C) emissions.
            mask (torch.Tensor): (B, L) bool.

        Returns:
            torch.Tensor: (B, L, C) marginal probabilities.
        """
        B, L, C = em.shape; T = self._trans(); alphas = self._alphas(em, mask)
        logZ = torch.logsumexp(alphas[-1] + self.end, dim=1); endv = self.end.expand(B, C); betas = [None] * L; betas[L - 1] = endv
        for t in range(L - 2, -1, -1):
            cand = torch.logsumexp(T.unsqueeze(0) + (em[:, t + 1] + betas[t + 1]).unsqueeze(1), dim=2)
            betas[t] = torch.where(mask[:, t + 1].unsqueeze(1), cand, endv)
        return (torch.stack([alphas[t] + betas[t] for t in range(L)], 1) - logZ.view(B, 1, 1)).exp()


class CRFNERModel(NERModel):
    """``NERModel`` + ``LinearCRF``. ``forward`` returns emissions; training loss = ``crf_nll``; ``to_probs`` = CRF marginals.

    Args:
        backbone (nn.Module): backbone.
        n_tags (int): number of tags.
        **kw: forwarded to ``NERModel``.
    """
    use_crf = True

    def __init__(self, backbone, n_tags, **kw):
        super().__init__(backbone, n_tags, **kw); self.crf = LinearCRF(n_tags); self._mask = None

    def forward(self, input_ids, attention_mask):
        """Emission logits (also remembers the mask for ``to_probs``).

        Args:
            input_ids (torch.Tensor): (B, T).
            attention_mask (torch.Tensor): (B, T).

        Returns:
            torch.Tensor: (B, T, n_tags) emissions.
        """
        self._mask = attention_mask.bool(); return super().forward(input_ids, attention_mask)

    def crf_nll(self, emissions, labels, attention_mask):
        """CRF training loss.

        Args:
            emissions (torch.Tensor): (B, T, C).
            labels (torch.Tensor): (B, T).
            attention_mask (torch.Tensor): (B, T).

        Returns:
            torch.Tensor: scalar NLL.
        """
        return self.crf.nll(emissions.float(), labels, attention_mask.bool())

    def to_probs(self, logits):
        """CRF marginals (so decoding / ensembling work unchanged).

        Args:
            logits (torch.Tensor): emissions (B, T, C).

        Returns:
            torch.Tensor: marginal probabilities.
        """
        return self.crf.marginals(logits.float(), self._mask)
