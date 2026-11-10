"""Linear-chain CRF over sentence labels (paper: BiLSTM-CRF) with marginals and Viterbi decoding."""
import torch
import torch.nn as nn


class CRF(nn.Module):
    """Linear-chain CRF with learned transitions.

    Args:
        n (int): Number of labels.
    """

    def __init__(self, n):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, n))
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def logZ(self, em, mask):
        """Log partition function (forward algorithm).

        Args:
            em (torch.Tensor): Emissions (B, T, n), float32.
            mask (torch.Tensor): Bool (B, T); ``mask[:, 0]`` must be True.

        Returns:
            torch.Tensor: (B,) log Z.
        """
        a = self.start + em[:, 0]
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(a.unsqueeze(2) + self.trans.unsqueeze(0), 1) + em[:, t]
            a = torch.where(mask[:, t].unsqueeze(1), nxt, a)
        return torch.logsumexp(a + self.end, 1)

    def score(self, em, tags, mask):
        """Score of a given tag sequence.

        Args:
            em (torch.Tensor): Emissions (B, T, n).
            tags (torch.Tensor): Gold tags (B, T).
            mask (torch.Tensor): Bool (B, T).

        Returns:
            torch.Tensor: (B,) path scores.
        """
        B, T, _ = em.shape
        ar = torch.arange(B, device=em.device)
        s = self.start[tags[:, 0]] + em[ar, 0, tags[:, 0]]
        for t in range(1, T):
            s = s + (self.trans[tags[:, t - 1], tags[:, t]] + em[ar, t, tags[:, t]]) * mask[:, t].float()
        return s + self.end[tags[ar, mask.long().sum(1) - 1]]

    def nll(self, em, tags, mask):
        """Mean negative log-likelihood.

        Args:
            em (torch.Tensor): Emissions (B, T, n).
            tags (torch.Tensor): Gold tags (B, T).
            mask (torch.Tensor): Bool (B, T).

        Returns:
            torch.Tensor: Scalar loss.
        """
        return (self.logZ(em, mask) - self.score(em, tags, mask)).mean()

    def marginals(self, em, mask):
        """Posterior ``P(y_t = j)`` as ``d logZ / d emission``.

        Args:
            em (torch.Tensor): Emissions (B, T, n).
            mask (torch.Tensor): Bool (B, T).

        Returns:
            torch.Tensor: Marginals (B, T, n).
        """
        with torch.enable_grad():
            e = em.detach().float().clone().requires_grad_(True)
            g, = torch.autograd.grad(self.logZ(e, mask).sum(), e)
        return g


def crf_viterbi(crf, em, mask, bias=None):
    """Viterbi decoding with the CRF's trained transitions.

    Args:
        crf (CRF): Trained CRF.
        em (torch.Tensor): Emissions (B, T, n); ``mask[:, 0]`` must be True.
        mask (torch.Tensor): Bool (B, T).
        bias (Optional[np.ndarray]): Per-class bias (n,) added to every emission (e.g. from ``tune_bias``).

    Returns:
        List[List[int]]: Predicted label ids per sequence (true lengths).
    """
    B, T, n = em.shape
    e = em + (bias if bias is None else torch.as_tensor(bias, dtype=em.dtype, device=em.device))
    score = crf.start + e[:, 0]
    backptrs = []
    for t in range(1, T):
        cand = score.unsqueeze(2) + crf.trans.unsqueeze(0)
        best_score, best_from = cand.max(dim=1)
        best_score = best_score + e[:, t]
        m = mask[:, t].unsqueeze(1)
        score = torch.where(m, best_score, score)
        backptrs.append(best_from)
    score = score + crf.end
    lens = mask.long().sum(1)
    out = []
    for b in range(B):
        L = int(lens[b].item())
        if L == 0:
            out.append([])
            continue
        last = int(score[b].argmax().item())
        path = [last]
        for t in range(L - 1, 0, -1):
            last = int(backptrs[t - 1][b, last].item())
            path.append(last)
        path.reverse()
        out.append(path)
    return out
