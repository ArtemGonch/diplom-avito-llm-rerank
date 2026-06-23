"""Offline HuggingFace instruct-LLM for UR4Rec §3.1 (su/si generation)."""

from __future__ import annotations

import importlib.util
import os
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 7B instruct models that fit one V100 32GB with 4-bit (no Meta Llama gate)
LLM_PRESETS: dict[str, str] = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-chat": "deepseek-ai/deepseek-llm-7b-chat",
}


def resolve_model_id(llm_cfg: dict) -> str:
    if llm_cfg.get("local_model_path") and os.path.isdir(llm_cfg["local_model_path"]):
        return llm_cfg["local_model_path"]
    if llm_cfg.get("model_name"):
        return llm_cfg["model_name"]
    backend = llm_cfg.get("backend", "qwen")
    if backend not in LLM_PRESETS:
        raise ValueError(
            f"Unknown llm.backend '{backend}'. "
            f"Choose from {list(LLM_PRESETS)} or set llm.model_name explicitly."
        )
    return LLM_PRESETS[backend]


def _has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _require_accelerate_for_device_map(device_map: str | dict | None) -> None:
    if device_map is None:
        return
    if not _has_accelerate():
        raise ImportError(
            "Package 'accelerate' is required when llm.device_map is set "
            f"(got {device_map!r}). Install into your env:\n"
            "  pip install 'accelerate>=0.25'"
        )


def _strip_response(text: str) -> str:
    text = text.strip()
    think_end = "\u003c/think\u003e"
    think_start = "\u003cthink\u003e"
    im_end = "\u003c|im_end|\u003e"
    if think_end in text:
        text = text.split(think_end)[-1].strip()
    elif think_start in text:
        text = re.sub(r"\u003cthink\u003e[\s\S]*", "", text).strip()
    for marker in ("[/INST]", "</s>", "<|eot_id|>", im_end):
        if marker in text:
            text = text.split(marker)[0]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class HFChatKnowledgeGenerator:
    """Paper: su = LLM(pu), si = LLM(pi). Backend: Qwen / DeepSeek / Llama2 via HF."""

    def __init__(self, llm_cfg: dict):
        self.backend = llm_cfg.get("backend", "qwen")
        self.model_id = resolve_model_id(llm_cfg)
        self.max_new_tokens = int(llm_cfg.get("max_new_tokens", 512))
        self.batch_size = max(1, int(llm_cfg.get("batch_size", 1)))
        self.temperature = float(llm_cfg.get("temperature", 0.1))
        self.top_p = float(llm_cfg.get("top_p", 0.9))
        self.max_prompt_tokens = int(llm_cfg.get("max_prompt_tokens", 1536))
        self.load_in_4bit = bool(llm_cfg.get("load_in_4bit", True))
        self.device_map = llm_cfg.get("device_map", "auto")
        self.trust_remote_code = bool(llm_cfg.get("trust_remote_code", True))
        _require_accelerate_for_device_map(self.device_map)

        token = llm_cfg.get("hf_token") or os.environ.get("HF_TOKEN")
        print(f"Loading LLM [{self.backend}]: {self.model_id} (4bit={self.load_in_4bit})")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=token,
                use_fast=True,
                trust_remote_code=self.trust_remote_code,
            )
        except OSError as e:
            if "gated" in str(e).lower() or "403" in str(e):
                raise OSError(
                    f"Cannot download {self.model_id} (gated or denied). "
                    "Use llm.backend: qwen or deepseek in config, or set llm.model_name "
                    "to an open model. Original error: " + str(e)
                ) from e
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "token": token,
            "device_map": self.device_map,
            "torch_dtype": torch.float16,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for recommendation systems. "
                    "Follow the user instruction. Be concise but structured."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _prompt_to_text(self, prompt: str) -> str:
        messages = self._messages(prompt)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {prompt}\nAssistant:"

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        device = next(self.model.parameters()).device
        texts = [self._prompt_to_text(p) for p in prompts]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).to(device)
        input_lens = inputs["attention_mask"].sum(dim=1)
        gen_kw = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)

        results: list[str] = []
        for i in range(len(prompts)):
            start = int(input_lens[i].item())
            new_tokens = out[i, start:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(_strip_response(decoded))
        return results

    def _generate_one(self, prompt: str) -> str:
        return self._generate_batch([prompt])[0]

    def generate_user_preference(self, prompt: str) -> str:
        return self._generate_one(prompt)

    def generate_item_knowledge(self, prompt: str) -> str:
        return self._generate_one(prompt)

    def generate_user_preference_batch(self, prompts: list[str]) -> list[str]:
        return self._generate_batch(prompts)

    def generate_item_knowledge_batch(self, prompts: list[str]) -> list[str]:
        return self._generate_batch(prompts)


# Backward-compatible alias
Llama2KnowledgeGenerator = HFChatKnowledgeGenerator
