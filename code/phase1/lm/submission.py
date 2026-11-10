"""Official test-set generation and ``submission.zip`` packaging (resumable: partial results saved after every batch)."""

import json
import os
import time
import zipfile

import torch


def load_test_prompts(test_file_path):
    """Load the official test prompts.

    Args:
        test_file_path (str): ``test.json`` path.

    Returns:
        tuple[list[str], list[str]]: (ids, prompts).
    """
    data = json.load(open(test_file_path, "r", encoding="utf-8"))
    ids = list(data.keys())
    print(f"Loaded {len(ids)} official test prompts.")
    return ids, [data[i]["text"] for i in ids]


def generate_test_predictions(
    gen_fn, model, tokenizer, device, ids, prompts, cfg, batch_size=8, use_autocast=True
):
    """Generate continuations for the test prompts, saving ``prediction_partial.json`` after every batch (resume-safe).

    Args:
        gen_fn (Callable): generator.
        model (nn.Module): model.
        tokenizer (PreTrainedTokenizer): tokenizer.
        device (torch.device): device.
        ids (list[str]): test ids.
        prompts (list[str]): test prompts.
        cfg (dict): run config (generation settings, output dir).
        batch_size (int): prompts per batch.
        use_autocast (bool): fp16 autocast during generation.

    Returns:
        dict: ``{id: {"text": continuation}}``.
    """
    partial = os.path.join(cfg["output_dir"], "prediction_partial.json")
    preds = json.load(open(partial)) if os.path.exists(partial) else {}
    todo = [(i, p) for i, p in zip(ids, prompts) if i not in preds]
    t0 = time.time()
    for b in range(0, len(todo), batch_size):
        chunk = todo[b : b + batch_size]
        with torch.autocast(
            "cuda", dtype=torch.float16, enabled=use_autocast and device.type == "cuda"
        ):
            outs = gen_fn(
                [p for _, p in chunk],
                model,
                tokenizer,
                device,
                target_words=cfg["target_generation_words"],
                max_new_tokens=cfg["max_generation_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                batch_size=batch_size,
            )
        for (tid, _), text in zip(chunk, outs):
            preds[tid] = {"text": text}
        json.dump(preds, open(partial, "w"), ensure_ascii=False)
        print(f"{len(preds)}/{len(ids)} done, {time.time() - t0:.0f}s elapsed")
    return preds


def write_submission(predictions, cfg):
    """Write ``prediction.json`` and zip it as ``submission.zip``.

    Args:
        predictions (dict): ``{id: {"text": ...}}``.
        cfg (dict): run config (paths).

    Returns:
        str: path of the zip.
    """
    with open(cfg["prediction_path"], "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(cfg["submission_zip"], "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(cfg["prediction_path"], arcname="prediction.json")
    print("Saved predictions to:", cfg["prediction_path"])
    print("Created submission zip:", cfg["submission_zip"])
    return cfg["submission_zip"]
