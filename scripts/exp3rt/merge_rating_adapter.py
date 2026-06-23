#!/usr/bin/env python3
"""Merge Exp3RT LoRA adapters into a vLLM-compatible bf16 checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_chain(base_model: str, adapter_dirs: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.exists() and any(out_dir.iterdir()):
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {base_model}")
    for i, ad in enumerate(adapter_dirs, 1):
        print(f"  stage {i}: {ad}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    for ad in adapter_dirs:
        if not (ad / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter: {ad / 'adapter_config.json'}")
        model = PeftModel.from_pretrained(model, str(ad))
        model = model.merge_and_unload()

    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(out_dir)

    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"Merged bf16 checkpoint -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--adapter-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Ordered LoRA dirs (preference user item rating)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    merge_chain(args.base_model, list(args.adapter_dirs), args.out_dir)


if __name__ == "__main__":
    main()
