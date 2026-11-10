"""HF ``Trainer`` setup (causal LM or seq2seq), resumable checkpoints, perplexity evaluation, loss plot and model saving."""

import glob
import inspect
import json
import math
import os
import time

import torch

from ..utils import count_params, precision_flags


class CausalLMCollator:
    """Collator with ``labels = input_ids`` for fixed-length blocks.

    ``DataCollatorForLanguageModeling(mlm=False)`` masks every position whose id equals the pad id; here pad == EOS, so appended EOS tokens would be
    masked out of the loss and the model could never learn to end a document. Blocks are fixed-length (no padding), so no masking is needed.
    """

    def __call__(self, features):
        """Collate a batch.

        Args:
            features (list[dict]): items with ``input_ids``.

        Returns:
            dict: ``input_ids``, ``attention_mask`` (ones) and ``labels``.
        """
        ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


def filter_supported_kwargs(cls, kwargs):
    """Keep only kwargs accepted by this ``transformers`` version (``eval_strategy`` vs ``evaluation_strategy`` etc.).

    Args:
        cls (type): ``TrainingArguments`` class.
        kwargs (dict): candidate arguments.

    Returns:
        dict: supported subset (dropped keys are printed).
    """
    accepted = set(inspect.signature(cls.__init__).parameters)
    dropped = set(kwargs) - accepted
    if dropped:
        print(
            f"[{cls.__name__}] dropping unsupported args for this transformers version: {sorted(dropped)}"
        )
    return {k: v for k, v in kwargs.items() if k in accepted}


def build_trainer(model, tokenizer, cfg, train_dataset, val_dataset):
    """Create the Trainer (``Seq2SeqTrainer`` for T5).

    Args:
        model (nn.Module): model.
        tokenizer (PreTrainedTokenizer): tokenizer (for the seq2seq collator).
        cfg (dict): run config (``lr, batch_size, grad_accum, eval_steps, save_steps, epochs`` ...).
        train_dataset (Dataset): training data.
        val_dataset (Dataset): validation data.

    Returns:
        transformers.Trainer: configured trainer.
    """
    bf16, fp16 = precision_flags()
    args = dict(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        seed=cfg["seed"],
    )
    if cfg["is_seq2seq"]:
        from transformers import (
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )

        ta = Seq2SeqTrainingArguments(
            **filter_supported_kwargs(
                Seq2SeqTrainingArguments, {**args, "predict_with_generate": False}
            )
        )
        return Seq2SeqTrainer(
            model=model,
            args=ta,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        )
    from transformers import Trainer, TrainingArguments

    ta = TrainingArguments(**filter_supported_kwargs(TrainingArguments, args))
    return Trainer(
        model=model,
        args=ta,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CausalLMCollator(),
    )


def _is_valid_checkpoint(path):
    """Whether a ``checkpoint-*`` directory is complete (config, trainer state, optimizer, weights).

    Args:
        path (str): checkpoint directory.

    Returns:
        bool: True if complete.
    """
    has_w = any(
        os.path.isfile(os.path.join(path, f))
        for f in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
        )
    )
    return has_w and all(
        os.path.isfile(os.path.join(path, f))
        for f in ("config.json", "trainer_state.json", "optimizer.pt")
    )


def _step_of(path):
    """Training step encoded in a checkpoint directory name.

    Args:
        path (str): ``.../checkpoint-<step>``.

    Returns:
        int: step (-1 if unparsable).
    """
    try:
        return int(path.rstrip("/").split("checkpoint-")[-1])
    except ValueError:
        return -1


def find_latest_checkpoint(arch, output_dir):
    """Latest complete checkpoint in the run dir or in attached Kaggle datasets (for resuming after a session limit).

    Args:
        arch (str): architecture key.
        output_dir (str): current run directory.

    Returns:
        str or None: checkpoint path, or None to start from scratch.
    """
    cands = (
        set(glob.glob(f"{output_dir}/checkpoint-*"))
        | set(glob.glob(f"/kaggle/input/*/cs781_phase1/{arch}/checkpoint-*"))
        | set(
            glob.glob(
                f"/kaggle/input/**/cs781_phase1/{arch}/checkpoint-*", recursive=True
            )
        )
    )
    valid = [c for c in cands if _is_valid_checkpoint(c)]
    bad = [c for c in cands if c not in valid]
    if bad:
        print(
            f"Ignoring {len(bad)} incomplete/corrupt checkpoint dir(s): {sorted(bad)}"
        )
    return max(valid, key=_step_of) if valid else None


def make_force_step_callback():
    """Callback that re-applies save/eval/log intervals from ``TrainingArguments`` on resume (Trainer otherwise restores the old ones from the checkpoint).

    Args:
        None

    Returns:
        TrainerCallback: the callback.
    """
    from transformers import TrainerCallback

    class ForceStepIntervals(TrainerCallback):
        """Re-apply the intervals from TrainingArguments when training begins."""

        def on_train_begin(self, args, state, control, **kwargs):
            state.save_steps, state.eval_steps, state.logging_steps = (
                args.save_steps,
                args.eval_steps,
                args.logging_steps,
            )
            print(
                f"Intervals -> save={state.save_steps}, eval={state.eval_steps}, log={state.logging_steps}"
            )

    return ForceStepIntervals()


def train(trainer, resume_from=None):
    """Run training.

    Args:
        trainer (transformers.Trainer): trainer.
        resume_from (str or None): checkpoint path to resume from.

    Returns:
        tuple[TrainOutput, float]: (train result, seconds).
    """
    trainer.add_callback(make_force_step_callback())
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume_from)
    sec = time.time() - t0
    print(f"Training finished in {sec / 60:.1f} minutes")
    print(result.metrics)
    return result, sec


def evaluate_perplexity(trainer, arch, n_params, train_seconds):
    """Validation loss and perplexity.

    Args:
        trainer (transformers.Trainer): trainer.
        arch (str): architecture name.
        n_params (int): parameter count.
        train_seconds (float): training time.

    Returns:
        dict: ``eval_loss, perplexity, architecture, n_params, train_seconds``.
    """
    m = trainer.evaluate()
    loss = m.get("eval_loss")
    m["perplexity"] = math.exp(loss) if loss is not None and loss < 20 else float("inf")
    print(f"Eval loss: {loss:.4f} | Perplexity: {m['perplexity']:.2f}")
    m.update(architecture=arch, n_params=n_params, train_seconds=train_seconds)
    return m


def plot_and_save(trainer, tokenizer, cfg, eval_metrics):
    """Save the loss curve, metrics JSON and the final model + tokenizer.

    Args:
        trainer (transformers.Trainer): trained trainer.
        tokenizer (PreTrainedTokenizer): tokenizer.
        cfg (dict): run config (output paths).
        eval_metrics (dict): metrics to write.

    Returns:
        None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts, tl, es, el = [], [], [], []
    for e in trainer.state.log_history:
        if "loss" in e and "eval_loss" not in e:
            ts.append(e["step"])
            tl.append(e["loss"])
        if "eval_loss" in e:
            es.append(e["step"])
            el.append(e["eval_loss"])
    plt.figure(figsize=(8, 5))
    plt.plot(ts, tl, label="train loss")
    plt.plot(es, el, label="eval loss", marker="o")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"Training/validation loss -- {cfg['arch']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg["loss_plot_path"])
    plt.close()
    json.dump(eval_metrics, open(cfg["eval_metrics_path"], "w"), indent=2, default=str)
    trainer.save_model(cfg["model_output_dir"])
    tokenizer.save_pretrained(cfg["model_output_dir"])
    print("Saved model + tokenizer to:", cfg["model_output_dir"])
    print("Saved metrics to:", cfg["eval_metrics_path"])
