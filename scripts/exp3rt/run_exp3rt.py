#!/usr/bin/env python3
"""
Exp3RT reproduction CLI (Qwen backend).

Stages:
  prepare   — merge JSON shards
  train     — QLoRA fine-tune one stage (preference|user|item|rating)
  test      — vLLM inference on rating test split
  eval      — RMSE/MAE from predictions
  all       — prepare + train rating + test + eval (minimal reproduction)

Usage:
  python scripts/exp3rt/run_exp3rt.py --config configs/exp3rt/amazon_book_qwen.yaml --stage prepare
  python scripts/exp3rt/run_exp3rt.py --config configs/exp3rt/amazon_book_qwen.yaml --stage train --train-stage rating
  CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/exp3rt/run_exp3rt.py --config configs/exp3rt/amazon_book_qwen.yaml --stage test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.exp3rt.data_prep import count_rows, prepare_amazon_book  # noqa: E402
from models.exp3rt.evaluate import evaluate_rating_prediction  # noqa: E402


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _data_paths(cfg: dict) -> dict[str, Path]:
    data_root = ROOT / cfg["paths"]["data_root"]
    return prepare_amazon_book(data_root)


def _stage_train_params(cfg: dict, stage: str) -> dict:
    tr = cfg["train"]
    overrides = cfg.get("stages", {}).get(stage, {}).get("train_overrides", {}) or {}
    merged = {**tr, **overrides}
    lr = merged["learning_rate"]
    if stage == "preference":
        lr = merged.get("preference_learning_rate", lr)
    elif stage in {"user", "item"}:
        lr = merged.get("profile_learning_rate", lr)
    merged["learning_rate"] = lr
    return merged


def _output_dir(cfg: dict, stage: str) -> Path:
    root = ROOT / cfg["paths"]["output_root"]
    tr = _stage_train_params(cfg, stage)
    name = (
        f"{cfg['project']['dataset']}_{stage}_"
        f"r{tr['lora_r']}_alpha{tr['lora_alpha']}_seed{tr['seed']}"
    )
    return root / name


def _train_cfg(cfg: dict, stage: str, paths: dict[str, Path], init_model: str | None = None) -> dict:
    tr = _stage_train_params(cfg, stage)
    ds = cfg["project"]["dataset"]
    stage_paths = cfg["stages"][stage]

    rel_train = stage_paths["train"]
    rel_valid = stage_paths["valid"]
    data_root = ROOT / cfg["paths"]["data_root"] / ds
    base = init_model or cfg["model"]["base_model"]
    return {
        "stage": stage,
        "dataset": ds,
        "base_model": cfg["model"]["base_model"],
        "init_model_path": base if init_model else None,
        "train_data_path": str(data_root / rel_train),
        "val_data_path": str(data_root / rel_valid),
        "output_dir": str(_output_dir(cfg, stage)),
        "seed": tr["seed"],
        "batch_size": tr["batch_size"],
        "micro_batch_size": tr["micro_batch_size"],
        "num_epochs": tr["num_epochs"],
        "learning_rate": tr["learning_rate"],
        "cutoff_len": tr["cutoff_len"],
        "lora_r": tr["lora_r"],
        "lora_alpha": tr["lora_alpha"],
        "lora_dropout": tr["lora_dropout"],
        "rmse_patience": tr["rmse_patience"],
        "eval_patience": tr["eval_patience"],
        "group_by_length": tr["group_by_length"],
        "max_train_samples": tr.get("max_train_samples"),
        "max_eval_samples": tr.get("max_eval_samples"),
        "dataloader_num_workers": tr.get("dataloader_num_workers", 4),
        "gradient_checkpointing": tr.get("gradient_checkpointing", True),
    }


def stage_prepare(cfg: dict) -> None:
    paths = _data_paths(cfg)
    print("Exp3RT data ready:")
    for k, p in paths.items():
        if p.suffix == ".json":
            print(f"  {k}: {count_rows(p):,} rows -> {p}")


def stage_train(cfg: dict, train_stage_name: str, init_model: str | None = None) -> Path:
    from models.exp3rt.train import train_stage as _train_stage

    paths = _data_paths(cfg)
    _ = paths
    tcfg = _train_cfg(cfg, train_stage_name, paths, init_model=init_model)
    return _train_stage(tcfg)


def stage_test(cfg: dict) -> Path:
    from models.exp3rt.test import run_inference
    rating_dir = _output_dir(cfg, "rating")
    merged = rating_dir / "merged"
    if not merged.exists():
        raise FileNotFoundError(f"Train rating first; missing {merged}")

    ds = cfg["project"]["dataset"]
    data_root = ROOT / cfg["paths"]["data_root"] / ds
    test_file = cfg["stages"]["rating"]["test"]
    out = rating_dir / f"predictions_{cfg['inference']['test_split'].replace('.json', '')}.json"
    icfg = {
        "dataset": ds,
        "test_data_path": str(data_root / test_file),
        "model_path": str(merged),
        "output_path": str(out),
        "seed": cfg["inference"]["seed"],
        "max_tokens": cfg["inference"]["max_tokens"],
        "tensor_parallel_size": cfg["inference"]["tensor_parallel_size"],
    }
    return run_inference(icfg)


def stage_eval(cfg: dict, predictions: Path | None = None) -> dict:
    ds = cfg["project"]["dataset"]
    data_root = ROOT / cfg["paths"]["data_root"] / ds
    test_path = data_root / cfg["stages"]["rating"]["test"]
    if predictions is None:
        predictions = (
            _output_dir(cfg, "rating")
            / f"predictions_{cfg['inference']['test_split'].replace('.json', '')}.json"
        )
    metrics = evaluate_rating_prediction(test_path, predictions)
    out = _output_dir(cfg, "rating") / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {out}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3RT reproduction (Qwen)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp3rt/amazon_book_qwen.yaml")
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "test", "eval", "all"),
        required=True,
    )
    parser.add_argument(
        "--train-stage",
        choices=("preference", "user", "item", "rating"),
        default="rating",
        help="Which fine-tune stage to run when --stage train",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Merged checkpoint from previous stage (sequential paper training)",
    )
    parser.add_argument(
        "--train-stages",
        nargs="+",
        choices=("preference", "user", "item", "rating"),
        default=None,
        help="For --stage all: which stages to train (default: rating only)",
    )
    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    if args.stage == "prepare":
        stage_prepare(cfg)
        return

    if args.stage == "train":
        init = str(args.init_model) if args.init_model else None
        stage_train(cfg, args.train_stage, init_model=init)
        return

    if args.stage == "test":
        stage_test(cfg)
        return

    if args.stage == "eval":
        stage_eval(cfg)
        return

    if args.stage == "all":
        stage_prepare(cfg)
        stages = args.train_stages or ["rating"]
        for st in stages:
            if cfg["stages"][st].get("enabled", True):
                print(f"\n=== Training stage: {st} ===")
                stage_train(cfg, st)
        stage_test(cfg)
        stage_eval(cfg)


if __name__ == "__main__":
    main()
