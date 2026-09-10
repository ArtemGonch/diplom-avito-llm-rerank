"""QLoRA fine-tuning for Exp3RT stages with Qwen chat templates."""

from __future__ import annotations

import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import bitsandbytes as bnb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState

from .prompts import build_assistant_text, build_chat_messages, format_chat


os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _find_linear_names(model) -> list[str]:
    cls = bnb.nn.Linear4bit
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            leaf = name.split(".")[-1]
            if leaf != "lm_head":
                names.add(leaf)
    if not names:
        names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    return list(names)


def _tokenize_example(
    data_point: dict[str, Any],
    *,
    stage: str,
    dataset: str,
    tokenizer,
    cutoff_len: int,
) -> dict[str, list[int]]:
    messages = build_chat_messages(stage, data_point, dataset)
    assistant = build_assistant_text(stage, data_point, dataset)
    _, full_text = format_chat(tokenizer, messages, assistant)
    prompt_only, _ = format_chat(tokenizer, messages, None)

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    tokenized_prompt = tokenizer(
        prompt_only,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    labels = tokenized_full["input_ids"].copy()
    prompt_len = len(tokenized_prompt["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized_full["labels"] = labels
    return tokenized_full


class EvalLossEarlyStopping(TrainerCallback):
    def __init__(self, output_dir: str, model, tokenizer, patience: int = 5):
        self.output_dir = output_dir
        self.model = model
        self.tokenizer = tokenizer
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        metrics = metrics or {}
        loss = metrics.get("eval_loss")
        if loss is None:
            return control
        loss_f = float(loss)
        if not math.isfinite(loss_f):
            print(f"Skipping eval checkpoint: invalid eval_loss={loss}")
            return control
        if loss_f < self.best:
            self.best = loss_f
            self.bad_epochs = 0
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"New best eval_loss={loss_f:.4f} -> saved {self.output_dir}")
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                print(f"Early stopping at epoch {state.epoch}, best={self.best:.4f}")
                control.should_training_stop = True
        return control


class RMSEEarlyStopping(TrainerCallback):
    def __init__(self, output_dir: str, model, tokenizer, patience: int = 1):
        self.output_dir = output_dir
        self.model = model
        self.tokenizer = tokenizer
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        metrics = metrics or {}
        rmse = metrics.get("eval_rmse")
        if rmse is None:
            return control
        if rmse < self.best:
            self.best = rmse
            self.bad_epochs = 0
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"New best eval_rmse={rmse:.4f} -> saved {self.output_dir}")
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                print(f"Early stopping (RMSE) at epoch {state.epoch}, best={self.best:.4f}")
                control.should_training_stop = True
        return control


def _extract_rating(text: str) -> int | None:
    if "Predicted User Rating:" not in text:
        return None
    rating_text = text.split("Predicted User Rating:")[-1].strip()
    match = re.search(r"\b(\d)\b", rating_text)
    return int(match.group(1)) if match else None


def train_stage(cfg: dict[str, Any]) -> Path:
    stage = cfg["stage"]
    dataset = cfg.get("dataset", "amazon-book")
    base_model = cfg["base_model"]
    init_model_path = cfg.get("init_model_path")
    load_path = init_model_path or base_model
    train_path = Path(cfg["train_data_path"])
    val_path = Path(cfg["val_data_path"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 425))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = int(cfg.get("batch_size", 32))
    micro_batch_size = int(cfg.get("micro_batch_size", 4))
    grad_accum = max(1, batch_size // micro_batch_size)
    num_epochs = int(cfg.get("num_epochs", 3))
    lr = float(cfg.get("learning_rate", 2e-4))
    cutoff_len = int(cfg.get("cutoff_len", 1200))
    lora_r = int(cfg.get("lora_r", 128))
    lora_alpha = int(cfg.get("lora_alpha", 32))
    lora_dropout = float(cfg.get("lora_dropout", 0.1))
    rmse_patience = int(cfg.get("rmse_patience", 1))
    eval_patience = int(cfg.get("eval_patience", 5))
    max_train_samples = cfg.get("max_train_samples")
    max_eval_samples = cfg.get("max_eval_samples")
    dataloader_num_workers = int(cfg.get("dataloader_num_workers", 4))
    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))

    print(f"Exp3RT train stage={stage} dataset={dataset} base={base_model}")
    if init_model_path:
        print(f"  init_from_merged={init_model_path}")
    print(f"  train={train_path} val={val_path} out={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    # Always QLoRA: merged checkpoints from prior stages are re-quantized on load.
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=_find_linear_names(model),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    def map_fn(row):
        return _tokenize_example(
            row,
            stage=stage,
            dataset=dataset,
            tokenizer=tokenizer,
            cutoff_len=cutoff_len,
        )

    raw_train = load_dataset("json", data_files=str(train_path))["train"].shuffle(seed=seed)
    raw_val = load_dataset("json", data_files=str(val_path))["train"].shuffle(seed=seed)
    if max_train_samples is not None:
        n = min(int(max_train_samples), len(raw_train))
        train_data = raw_train.select(range(n))
        print(f"  using max_train_samples={n:,} / {len(raw_train):,}")
    else:
        train_data = raw_train
    if max_eval_samples is not None:
        n = min(int(max_eval_samples), len(raw_val))
        val_data = raw_val.select(range(n))
        print(f"  using max_eval_samples={n:,} / {len(raw_val):,}")
    else:
        val_data = raw_val
    train_data = train_data.map(map_fn)
    val_data = val_data.map(map_fn)

    callbacks: list[TrainerCallback] = []
    compute_metrics = None
    if stage == "rating":

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            pred_ratings, true_ratings = [], []
            bs = micro_batch_size
            for i in range(0, len(predictions), bs):
                batch_preds = predictions[i : i + bs]
                batch_labels = labels[i : i + bs]
                pred_texts = tokenizer.batch_decode(
                    [[t for t in seq if t >= 0] for seq in batch_preds],
                    skip_special_tokens=True,
                )
                label_texts = tokenizer.batch_decode(
                    [[t for t in seq if t >= 0 and t != -100] for seq in batch_labels],
                    skip_special_tokens=True,
                )
                for pred_text, label_text in zip(pred_texts, label_texts):
                    pr = _extract_rating(pred_text)
                    tr = _extract_rating(label_text)
                    if pr is not None and tr is not None:
                        pred_ratings.append(pr)
                        true_ratings.append(tr)
            if not pred_ratings:
                return {"eval_rmse": float("inf"), "eval_mae": float("inf")}
            pred_arr = np.array(pred_ratings, dtype=np.float64)
            true_arr = np.array(true_ratings, dtype=np.float64)
            rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
            mae = float(np.mean(np.abs(pred_arr - true_arr)))
            return {"rmse": rmse, "mae": mae}

        callbacks.append(RMSEEarlyStopping(str(output_dir), model, tokenizer, patience=rmse_patience))
    else:
        callbacks.append(EvalLossEarlyStopping(str(output_dir), model, tokenizer, patience=eval_patience))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=0.03,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            output_dir=str(output_dir),
            group_by_length=bool(cfg.get("group_by_length", True)),
            gradient_checkpointing=gradient_checkpointing,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=True,
            report_to=[],
            seed=seed,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1) if compute_metrics else logits,
        callbacks=callbacks,
    )
    trainer.train()

    adapter_cfg = output_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved final adapter to {output_dir} (no valid eval checkpoint on disk)")

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")
    return merged_dir
