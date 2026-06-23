"""Shared utilities: metrics, device helpers, HF LLM backends."""

from .metrics import evaluate_batch, map_at_k, ndcg_at_k
from .device_util import describe_device, resolve_device, setup_cuda

__all__ = [
    "evaluate_batch",
    "map_at_k",
    "ndcg_at_k",
    "describe_device",
    "resolve_device",
    "setup_cuda",
]
