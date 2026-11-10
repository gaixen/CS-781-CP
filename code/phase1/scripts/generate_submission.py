"""CLI: reload a trained model, generate continuations for the official test prompts and write ``submission.zip``.

Example:
    python scripts/generate_submission.py --arch lightweight_sliding_window
    python scripts/generate_submission.py --arch modern_decoder --model-dir /path/to/model-final --batch-size 8
"""
import argparse

import _bootstrap  # noqa: F401


def main():
    """Generate and package the test-set predictions (resumable).

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True); ap.add_argument("--model-dir", default=None); ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    from lm.config import build_config
    from lm.data.datasets import download_data
    from lm.generation.generate import make_gen_fn
    from lm.models.factory import load_trained
    from lm.submission import generate_test_predictions, load_test_prompts, write_submission
    from lm.utils import get_device, hf_login
    cfg = build_config(a.arch); device = get_device(); token = hf_login()
    model, tok = load_trained(a.arch, a.model_dir or cfg["model_output_dir"], device)
    _, test_path = download_data({**cfg, "data_sources": {}}, token); ids, prompts = load_test_prompts(test_path)
    gen = make_gen_fn(model, tok, device, [{"prompt": p} for p in prompts[:2]], cfg["is_seq2seq"])
    write_submission(generate_test_predictions(gen, model, tok, device, ids, prompts, cfg, a.batch_size), cfg)


if __name__ == "__main__":
    main()
