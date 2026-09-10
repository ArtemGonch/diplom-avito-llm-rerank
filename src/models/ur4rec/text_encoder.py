"""Frozen BERT encoder for LLM-generated text (paper Eq.5)."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FrozenTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
        *,
        use_hashing_encoder: bool = False,
        hashing_hidden_size: int = 768,
    ):
        super().__init__()
        self.use_hashing_encoder = use_hashing_encoder
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        if use_hashing_encoder:
            self.tokenizer = None
            self.model = nn.Identity()
            self.hidden_size = int(hashing_hidden_size)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size
        self.model.eval()
        self.max_length = max_length
        for p in self.model.parameters():
            p.requires_grad = False
        self._embedding_cache: dict[str, torch.Tensor] = {}

    def _hashing_encode(self, text: str) -> torch.Tensor:
        """Deterministic dependency-free embedding used only by CI/smoke."""
        vector = torch.zeros(self.hidden_size, dtype=torch.float32)
        for token in text.lower().split()[: self.max_length]:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.hidden_size
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        return torch.nn.functional.normalize(vector, dim=0)

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        """Mean-pool token embeddings -> [B, D]."""
        if device is None:
            device = self._device_anchor.device
        missing = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        if missing:
            if self.use_hashing_encoder:
                pooled = torch.stack([self._hashing_encode(text) for text in missing])
            else:
                enc = self.tokenizer(
                    missing,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            for text, vector in zip(missing, pooled):
                self._embedding_cache[text] = vector.detach().cpu()
        return torch.stack([self._embedding_cache[text] for text in texts]).to(device)

    def aggregate_preference(
        self,
        user_pref: torch.Tensor,
        item_knowledge: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """Return e_u^aggr as memory tokens [1 + history_len, hidden].

        Concatenation in UR4Rec Eq. 5 forms the key/value memory consumed by
        cross-attention.  Flattening it into one vector would reduce attention
        to a single key, whose softmax weight is always one.
        """
        parts = [user_pref] + list(item_knowledge)
        return torch.stack(parts, dim=0)
