"""Self-checks that need no training (CRF vs brute force, context windows, metrics, majority baseline)."""
import itertools
from collections import Counter

import numpy as np
import torch

from rr.data.tokenization import make_inputs_list, make_windows
from rr.eval.metrics import rr_scores
from rr.models.crf import CRF


def check_crf(device="cpu"):
    """Check CRF ``logZ`` and marginals against brute-force enumeration on a tiny random problem.

    Args:
        device (Union[str, torch.device]): Device for the check.

    Returns:
        None
    """
    crf = CRF(3).to(device)
    torch.nn.init.normal_(crf.trans, std=0.7)
    torch.nn.init.normal_(crf.start, std=0.5)
    torch.nn.init.normal_(crf.end, std=0.5)
    em = torch.randn(2, 4, 3, device=device)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool, device=device)

    def brute(b):
        """Brute-force log Z and marginals of sequence ``b``.

        Args:
            b (int): Batch index.

        Returns:
            Tuple[float, np.ndarray]: ``(log Z, marginals (T, 3))``.
        """
        T = int(mask[b].sum())
        tot, marg = 0.0, np.zeros((T, 3))
        for y in itertools.product(range(3), repeat=T):
            s = crf.start[y[0]] + em[b, 0, y[0]] + sum(crf.trans[y[t - 1], y[t]] + em[b, t, y[t]] for t in range(1, T)) + crf.end[y[-1]]
            w = float(torch.exp(s))
            tot += w
            for t in range(T):
                marg[t, y[t]] += w
        return np.log(tot), marg / tot

    lz, mg = crf.logZ(em, mask), crf.marginals(em, mask)
    for b in (0, 1):
        l_, m_ = brute(b)
        T = int(mask[b].sum())
        assert abs(float(lz[b]) - l_) < 1e-3, ("logZ", float(lz[b]), l_)
        assert np.abs(mg[b, :T].cpu().numpy() - m_).max() < 1e-3, "marginals"
    tags = torch.tensor([[0, 1, 2, 1], [2, 2, 0, 0]], device=device)
    assert float(crf.nll(em, tags, mask)) > 0
    print("1) CRF logZ + marginals == brute force; NLL > 0")


def check_windows_and_metrics():
    """Check the window generator and the metric implementation on toy inputs.

    Args:
        None

    Returns:
        None
    """
    assert make_windows(10, 4, 2) == [(0, 4), (2, 6), (4, 8), (6, 10)] and make_windows(3, 8, 4) == [(0, 3)]
    labels = ["Fact", "Issue", "None"]
    s = rr_scores(labels, ["Fact", "Fact", "Issue", "None"], ["Fact", "Issue", "Issue", "None"])
    assert abs(s["per"]["Fact"]["f1"] - 2 / 3) < 1e-9 and abs(s["acc"] - 0.75) < 1e-9
    print("2/3) windows and metrics OK")


def run_self_checks(ctx):
    """Run all self-checks (data-dependent ones need a loaded and tokenised context).

    Args:
        ctx (Context): Pipeline context with data and tokenizer.

    Returns:
        None
    """
    check_crf(ctx.device)
    check_windows_and_metrics()
    d = ctx.data.train[0]
    cfg = {**ctx.cfg, "NEIGHBORS": 1}
    ids, am, tm = make_inputs_list(ctx, [(d, 0), (d, 1)], cfg)
    assert tm.sum(1)[0] == len(ctx.dom_ids[d["domain"]]) + min(len(d["raw"][0]), ctx.cfg["MAX_SENT_TOK"]) and (tm <= am).all()
    print("4) make_inputs OK | example:", repr(ctx.tokenizer.decode(ids[1][am[1] == 1][:40]))[:120])
    cnt = Counter(l for dd in ctx.data.train for l in dd["labels"])
    maj = cnt.most_common(1)[0][0]
    gold = [l for dd in ctx.data.dev for l in dd["labels"]]
    print(f"5) majority-label baseline (always '{maj}') macro-F1 on DEV = {rr_scores(ctx.data.labels, gold, [maj] * len(gold))['macro_f1']:.4f}")
