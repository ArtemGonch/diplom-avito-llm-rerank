"""CUDA device selection for training scripts."""

from __future__ import annotations

import os

import torch


def resolve_device(cfg: dict) -> torch.device:
    """
    Config keys:
      device: cpu | cuda | cuda:N
      gpu_id: int (used when device is cuda)
    Env CUDA_VISIBLE_DEVICES remaps indices (e.g. only one visible GPU -> use gpu_id: 0).
    """
    want = str(cfg.get("device", "cuda")).strip().lower()
    if want == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        return torch.device("cpu")
    if want.startswith("cuda:"):
        return torch.device(want)
    if want == "cuda":
        gpu_id = int(cfg.get("gpu_id", 0))
        n = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= n:
            print(f"WARNING: gpu_id={gpu_id} invalid (0..{n - 1}), using 0")
            gpu_id = 0
        return torch.device(f"cuda:{gpu_id}")
    return torch.device(want)


def setup_cuda(cfg: dict, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if bool(cfg.get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    if bool(cfg.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    mem_gb = props.total_memory / (1024**3)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    return f"cuda:{idx} {name} ({mem_gb:.1f} GiB), CUDA_VISIBLE_DEVICES={visible}"
