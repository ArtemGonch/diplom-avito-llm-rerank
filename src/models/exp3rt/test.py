"""Run vLLM inference for Exp3RT rating stage (Qwen merged model)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from .prompts import build_chat_messages, format_chat


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    exp_x = np.exp((x - np.max(x)) / temp)
    return exp_x / exp_x.sum()


def _valid_digit(token: str, dataset: str) -> bool:
    try:
        num = int(token)
    except ValueError:
        return False
    if dataset == "imdb":
        return 0 <= num <= 9
    return 1 <= num <= 5


def build_inference_prompt(data_point: dict[str, Any], dataset: str, tokenizer) -> str:
    messages = build_chat_messages("rating", data_point, dataset)
    return format_chat(tokenizer, messages, None)[0]


def run_inference(cfg: dict[str, Any]) -> Path:
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Exp3RT inference requires vllm. Install: pip install vllm"
        ) from e

    dataset = cfg.get("dataset", "amazon-book")
    test_path = Path(cfg["test_data_path"])
    model_path = Path(cfg["model_path"])
    output_path = Path(cfg["output_path"])
    seed = int(cfg.get("seed", 425))
    max_model_len = int(cfg.get("max_model_len", 2048))
    gpu_memory_utilization = float(cfg.get("gpu_memory_utilization", 0.85))
    tensor_parallel_size = int(cfg.get("tensor_parallel_size", 1))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    test_data = load_dataset("json", data_files=str(test_path))["train"]
    prompts = [build_inference_prompt(row, dataset, tokenizer) for row in test_data]

    print(
        f"vLLM: tp={tensor_parallel_size} max_model_len={max_model_len} "
        f"gpu_mem_util={gpu_memory_utilization} n_prompts={len(prompts)}"
    )
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        seed=seed,
    )
    sampling = SamplingParams(max_tokens=int(cfg.get("max_tokens", 512)), temperature=0.0, seed=seed)
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    followups = []
    for i, out in enumerate(outputs):
        try:
            reasoning = (
                out.outputs[0].text.split("Predicted User Rating: ")[0].strip().split("Reasoning: ")[1].strip()
            )
        except IndexError:
            reasoning = "No reasoning provided."
        followups.append(f"{prompts[i]}\nReasoning: {reasoning}\nPredicted User Rating: ")

    logprob_sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=seed)
    logprob_outputs = llm.generate(followups, logprob_sampling, use_tqdm=True)

    results: dict[str, Any] = {}
    for i, (out, lp_out) in enumerate(zip(outputs, logprob_outputs)):
        digit_logprobs = {}
        if lp_out.outputs[0].logprobs:
            for logprob in lp_out.outputs[0].logprobs[0].values():
                tok = logprob.decoded_token
                if _valid_digit(tok, dataset):
                    digit_logprobs[tok] = logprob.logprob
        if not digit_logprobs:
            expected_rating = -1.0
            max_prob_rating = "-1"
            probabilities = {str(j): 0.0 for j in (range(1, 11) if dataset == "imdb" else range(1, 6))}
        else:
            probs = _softmax(np.array(list(digit_logprobs.values()), dtype=np.float64))
            if dataset == "imdb":
                expected_rating = float(
                    np.sum([(int(k) + 1) * p for k, p in zip(digit_logprobs.keys(), probs)])
                )
                max_prob_rating = str(int(max(digit_logprobs, key=digit_logprobs.get)) + 1)
            else:
                expected_rating = float(np.sum([int(k) * p for k, p in zip(digit_logprobs.keys(), probs)]))
                max_prob_rating = max(digit_logprobs, key=digit_logprobs.get)
            probabilities = {str(j): 0.0 for j in (range(1, 11) if dataset == "imdb" else range(1, 6))}
            for k, p in zip(digit_logprobs.keys(), probs):
                key = str(int(k) + 1) if dataset == "imdb" else str(k)
                probabilities[key] = float(p)

        try:
            reasoning = out.outputs[0].text.split("Predicted User Rating: ")[0].strip().split("Reasoning: ")[1].strip()
        except IndexError:
            reasoning = "No reasoning provided."

        results[str(i)] = {
            "generated_text": out.outputs[0].text,
            "reasoning": reasoning,
            "max_prob_rating": max_prob_rating,
            "expected_rating": expected_rating,
            "probabilities": probabilities,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved inference results to {output_path}")
    return output_path
