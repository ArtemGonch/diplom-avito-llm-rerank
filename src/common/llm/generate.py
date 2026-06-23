"""Offline LLM knowledge generation (Algorithm 1, §3.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .knowledge_batch import job_belongs_to_shard
from .prompts import (
    build_item_knowledge_prompt_avito,
    build_item_knowledge_prompt_ml1m,
    build_user_preference_prompt_avito,
    build_user_preference_prompt_ml1m,
    template_item_knowledge,
    template_user_preference,
)


class KnowledgeGenerator(Protocol):
    def generate_user_preference(self, prompt: str) -> str: ...
    def generate_item_knowledge(self, prompt: str) -> str: ...


class TemplateKnowledgeGenerator:
    """Fallback for CI/smoke without GPU LLM — NOT paper reproduction."""

    def generate_user_preference(self, prompt: str) -> str:
        return f"[template-only, not Llama2] {prompt[:200]}..."

    def generate_item_knowledge(self, prompt: str) -> str:
        return f"[template-only, not Llama2] {prompt[:200]}..."


def create_knowledge_generator(cfg: dict) -> KnowledgeGenerator:
    llm = cfg.get("llm", {})
    if llm.get("use_template_generator", False):
        print("WARNING: use_template_generator=true — this is NOT paper reproduction.")
        return TemplateKnowledgeGenerator()
    from .hf_chat_generator import HFChatKnowledgeGenerator

    return HFChatKnowledgeGenerator(llm)


class KnowledgeStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.users_path = self.root / "users.json"
        self.items_path = self.root / "items.json"

    def save(
        self,
        users: dict[str, str],
        items: dict[str, str],
        meta: dict | None = None,
    ) -> None:
        self.users_path.write_text(
            json.dumps(users, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        self.items_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        if meta is not None:
            (self.root / "meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

    def meta(self) -> dict:
        p = self.root / "meta.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def load(self) -> tuple[dict[str, str], dict[str, str]]:
        users = (
            json.loads(self.users_path.read_text(encoding="utf-8"))
            if self.users_path.exists()
            else {}
        )
        items = (
            json.loads(self.items_path.read_text(encoding="utf-8"))
            if self.items_path.exists()
            else {}
        )
        return users, items

    def save_partial(self, users: dict[str, str], items: dict[str, str]) -> None:
        """Checkpoint during long Llama2 runs."""
        self.save(users, items)


def shard_knowledge_dir(root: Path, shard_id: int) -> Path:
    return Path(root) / "shards" / f"shard_{shard_id}"


class ShardKnowledgeWriter:
    """Per-GPU checkpoint writer: only shard-owned keys, no races on merged JSON."""

    def __init__(
        self,
        store: KnowledgeStore,
        shard_id: int,
        num_shards: int,
        seed_users: dict[str, str],
        seed_items: dict[str, str],
    ):
        self._job_belongs = job_belongs_to_shard
        self.store = store
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.seed_users = seed_users
        self.seed_items = seed_items
        self.shard_root = shard_knowledge_dir(store.root, shard_id)
        self.shard_root.mkdir(parents=True, exist_ok=True)

    def _shard_slice(self, users: dict[str, str], items: dict[str, str]) -> tuple[dict, dict]:
        su = {
            k: v
            for k, v in users.items()
            if k not in self.seed_users
            and self._job_belongs(k, self.shard_id, self.num_shards)
        }
        si = {
            k: v
            for k, v in items.items()
            if k not in self.seed_items
            and self._job_belongs(k, self.shard_id, self.num_shards)
        }
        return su, si

    def save_partial(self, users: dict[str, str], items: dict[str, str]) -> None:
        su, si = self._shard_slice(users, items)
        self.shard_root.joinpath("users.json").write_text(
            json.dumps(su, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        self.shard_root.joinpath("items.json").write_text(
            json.dumps(si, ensure_ascii=False, indent=0), encoding="utf-8"
        )

    def save_final(self, users: dict[str, str], items: dict[str, str]) -> None:
        self.save_partial(users, items)


def merge_knowledge_shards(
    knowledge_dir: Path,
    num_shards: int,
    meta: dict | None = None,
) -> tuple[int, int]:
    """Merge base + per-shard JSON into knowledge_dir/items.json and users.json."""
    store = KnowledgeStore(Path(knowledge_dir))
    users, items = store.load()
    for shard_id in range(num_shards):
        shard_root = shard_knowledge_dir(store.root, shard_id)
        for name, target in (("users.json", users), ("items.json", items)):
            path = shard_root / name
            if not path.exists():
                continue
            chunk = json.loads(path.read_text(encoding="utf-8"))
            overlap = set(target) & set(chunk)
            if overlap:
                raise ValueError(
                    f"Shard merge conflict in {path}: {len(overlap)} duplicate keys"
                )
            target.update(chunk)
    store.save(users, items, meta=meta)
    return len(items), len(users)


# Helpers for ML1M / Avito prompt building
def ml1m_user_prompt(user_id: int, titles: list[str], genres: list[str]) -> str:
    return build_user_preference_prompt_ml1m(user_id, titles, genres)


def ml1m_item_prompt(title: str, genres: str) -> str:
    return build_item_knowledge_prompt_ml1m(title, genres)


def avito_user_prompt(session_id: int, query_text: str, history: list[str] | None = None) -> str:
    parts = query_text.replace("Avito search: ", "").split(", region or location id ")
    cat = parts[0] if parts else "Transport"
    loc = parts[1] if len(parts) > 1 else ""
    return build_user_preference_prompt_avito(session_id, cat, loc, history)


def avito_item_prompt(title: str, brand: str, attrs: str, price: float = 0) -> str:
    return build_item_knowledge_prompt_avito(title, brand, attrs, price)
