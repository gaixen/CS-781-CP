"""CLI: train one Phase-1 architecture (data -> model -> Trainer -> perplexity -> internal generation eval -> saved model).

Examples:
    python scripts/train.py --arch lightweight_sliding_window
    python scripts/train.py --arch modern_decoder
    python scripts/train.py --arch gpt2_decoder --override train_tokens=50000000 epochs=1
    python scripts/train.py --arch modern_decoder --resume          # resume from the latest checkpoint-*
"""
import argparse
import json
import random

import _bootstrap  # noqa: F401


def parse_overrides(items):
    """Parse ``KEY=VALUE`` strings (VALUE parsed as JSON when possible).

    Args:
        items (list[str]): override strings.

    Returns:
        dict: overrides.
    """
    ov = {}
    for kv in items:
        k, v = kv.split("=", 1)
        try: ov[k] = json.loads(v)
        except json.JSONDecodeError: ov[k] = v
    return ov


def main():
    """Run the full training pipeline for one architecture.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True); ap.add_argument("--override", nargs="*", default=[]); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-gen-eval", action="store_true", help="skip the internal generation-quality evaluation and decoding sweep")
    a = ap.parse_args()
    from lm.config import build_config
    from lm.data.datasets import build_datasets, download_data, make_word_level_eval_pairs, split_sources
    from lm.generation.generate import make_gen_fn
    from lm.generation.metrics import decoding_sweep, evaluate_generation
    from lm.models.factory import build_model
    from lm.training.trainer import build_trainer, evaluate_perplexity, find_latest_checkpoint, plot_and_save, train
    from lm.utils import count_params, get_device, get_tokenizer, hf_login, set_seed
    cfg = build_config(a.arch, **parse_overrides(a.override)); set_seed(cfg["seed"]); token = hf_login(); device = get_device()
    tok = get_tokenizer(); paths, _ = download_data(cfg, token); dev_texts, dev_idx, budget = split_sources(cfg, paths)
    pairs = make_word_level_eval_pairs([t for n in cfg["data_sources"] for t in dev_texts[n]], cfg["target_generation_words"] * 4, cfg["target_generation_words"],
                                       cfg["eval_generation_examples"], random.Random(cfg["seed"])); print("internal eval pairs:", len(pairs))
    train_ds, val_ds = build_datasets(cfg, paths, dev_texts, dev_idx, budget, tok)
    model = build_model(cfg["arch"], tok, cfg).to(device); n_params = count_params(model)
    trainer = build_trainer(model, tok, cfg, train_ds, val_ds)
    resume = find_latest_checkpoint(cfg["arch"], cfg["output_dir"]) if a.resume else None
    print("Resuming from:" if resume else "Training from scratch.", resume or "")
    _, sec = train(trainer, resume); metrics = evaluate_perplexity(trainer, cfg["arch"], n_params, sec)
    if not a.skip_gen_eval:
        gen = make_gen_fn(model, tok, device, pairs, cfg["is_seq2seq"])
        kw = dict(target_words=cfg["target_generation_words"], max_new_tokens=cfg["max_generation_tokens"], temperature=cfg["temperature"], top_p=cfg["top_p"])
        metrics.update({f"gen_{k}": v for k, v in evaluate_generation(gen, model, tok, device, pairs, kw).items()})
        if pairs: decoding_sweep(gen, model, tok, device, pairs, 10, cfg["target_generation_words"], cfg["max_generation_tokens"])
    plot_and_save(trainer, tok, cfg, metrics)


if __name__ == "__main__":
    main()
