#!/usr/bin/env python3
"""
UR4Rec full reproduction pipeline (Zhang et al., COLING 2025).

Stages (Algorithm 1):
  1. knowledge  — offline LLM text (template or external JSON)
  2. backbone   — train DLCM reranker without augmentation
  3. pretrain   — retriever L_CL + L_CF
  4. joint      — retriever + backbone with e_aug
  5. eval       — MAP/NDCG@K

Usage:
  conda activate diplom_avito
  pip install -r requirements.txt
  python scripts/run_ur4rec.py --config configs/ur4rec_ml1m.yaml --stage all

GPU (configs: device, gpu_id):
  CUDA_VISIBLE_DEVICES=1 python scripts/run_ur4rec.py --config configs/ur4rec_smoke.yaml --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ur4rec.backbone import DLCMReranker
from ur4rec.device_util import describe_device, resolve_device, setup_cuda
from ur4rec.data.amazon_books import AmazonBooks, download_amazon_books
from ur4rec.data.avito import AvitoSERP
from ur4rec.data.ml1m import MovieLens1M, RerankSample, download_movielens_1m
from ur4rec.data.steam import SteamReviews, download_steam
from ur4rec.llm.generate import (
    KnowledgeStore,
    ShardKnowledgeWriter,
    create_knowledge_generator,
    merge_knowledge_shards,
)
from ur4rec.llm.knowledge_batch import (
    cap_knowledge_ids,
    estimate_knowledge_hours,
    job_belongs_to_shard,
    llm_generate_loop,
)
from ur4rec.llm.prompts import (
    build_item_knowledge_prompt_amazon,
    build_item_knowledge_prompt_avito,
    build_item_knowledge_prompt_ml1m,
    build_item_knowledge_prompt_steam,
    build_user_preference_prompt_amazon,
    build_user_preference_prompt_avito,
    build_user_preference_prompt_ml1m,
    build_user_preference_prompt_steam,
    template_item_knowledge,
    template_user_preference,
)
from ur4rec.losses import info_nce_loss, listwise_ce_loss
from ur4rec.metrics import evaluate_batch
from ur4rec.retriever import UR4RecRetriever
from ur4rec.text_encoder import FrozenTextEncoder


class RerankDataset(Dataset):
    def __init__(self, samples: list[RerankSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RerankSample:
        return self.samples[idx]


def collate_samples(batch: list[RerankSample]) -> list[RerankSample]:
    return batch


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


SequentialData = MovieLens1M | AmazonBooks | SteamReviews


def _all_catalog_item_ids(data: SequentialData | MovieLens1M) -> list[int]:
    if isinstance(data, MovieLens1M):
        return sorted(data.movies["movieId"].astype(int).unique().tolist())
    return sorted(int(x) for x in data.items.index.tolist())


def _item_knowledge_prompt(data: SequentialData | MovieLens1M, iid: int) -> str:
    if isinstance(data, MovieLens1M):
        t, g = data.movie_meta(int(iid))
        return build_item_knowledge_prompt_ml1m(t, g)
    if isinstance(data, AmazonBooks):
        t, _ = data.item_meta(int(iid))
        row = data.items.loc[int(iid)]
        return build_item_knowledge_prompt_amazon(
            t, str(row["category"]), str(row["brand"])
        )
    if isinstance(data, SteamReviews):
        t, _ = data.item_meta(int(iid))
        row = data.items.loc[int(iid)]
        return build_item_knowledge_prompt_steam(
            t, str(row["genres"]), str(row["developer"])
        )
    raise TypeError(f"Unsupported dataset type: {type(data)}")


def _user_preference_prompt(data: SequentialData | MovieLens1M, uid: int, hist_len: int) -> str:
    seq = data.user_items[uid]
    titles, attrs = [], []
    for iid in seq[-hist_len:]:
        t, a = data.item_meta(iid) if not isinstance(data, MovieLens1M) else data.movie_meta(iid)
        titles.append(t)
        attrs.append(a)
    if isinstance(data, MovieLens1M):
        return build_user_preference_prompt_ml1m(uid, titles, attrs)
    if isinstance(data, AmazonBooks):
        return build_user_preference_prompt_amazon(uid, titles, attrs)
    return build_user_preference_prompt_steam(uid, titles, attrs)


def build_knowledge(
    cfg: dict,
    data: MovieLens1M | AvitoSERP | AmazonBooks | SteamReviews,
    samples: list[RerankSample] | None = None,
) -> KnowledgeStore:
    llm_cfg = cfg["llm"]
    num_shards = max(1, int(llm_cfg.get("num_shards", 1)))
    shard_id = int(llm_cfg.get("shard_id", 0))
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id={shard_id} out of range for num_shards={num_shards}")

    store = KnowledgeStore(Path(llm_cfg["knowledge_dir"]))
    users, items = store.load()
    seed_users, seed_items = dict(users), dict(items)
    if num_shards > 1:
        shard_root = store.root / "shards" / f"shard_{shard_id}"
        for name, target in (("users.json", users), ("items.json", items)):
            path = shard_root / name
            if path.exists():
                target.update(json.loads(path.read_text(encoding="utf-8")))
    want_gen = (
        "template"
        if llm_cfg.get("use_template_generator")
        else llm_cfg.get("backend", "qwen")
    )
    meta_path = store.root / "meta.json"
    cached_meta = store.meta().get("generator")
    if num_shards == 1 and users and items and cached_meta == want_gen:
        print(f"Knowledge cache hit ({cached_meta}):", store.root)
        return store
    if num_shards == 1 and users and items:
        reason = (
            "legacy cache without meta.json (old templates)"
            if not meta_path.exists()
            else f"generator '{cached_meta}' != '{want_gen}'"
        )
        print(f"Knowledge cache ignored ({reason}). Regenerating -> {store.root}")

    gen = create_knowledge_generator(cfg)
    print(f"Generating offline knowledge with {want_gen} (UR4Rec §3.1)")
    if num_shards > 1:
        print(f"Knowledge shard {shard_id + 1}/{num_shards} -> {store.root / 'shards' / f'shard_{shard_id}'}")
    users, items = users or {}, items or {}
    hist_len = cfg["dataset"]["history_len"]
    from ur4rec.llm.hf_chat_generator import resolve_model_id

    meta = {
        "generator": want_gen,
        "model_name": resolve_model_id(llm_cfg),
    }
    checkpoint = (
        ShardKnowledgeWriter(store, shard_id, num_shards, seed_users, seed_items)
        if num_shards > 1
        else store
    )

    batch_size = max(1, int(llm_cfg.get("batch_size", getattr(gen, "batch_size", 1))))
    sample_item_ids: set | None = None
    sample_user_ids: set | None = None
    if samples:
        sample_item_ids = {
            i
            for s in samples
            for i in s.candidate_item_ids + s.history_item_ids + [s.target_item_id]
        }
        sample_user_ids = {s.user_id for s in samples}

    if isinstance(data, AvitoSERP):
        if samples and cfg["dataset"].get("knowledge_items_only_in_samples", True):
            item_keys = sorted(
                {i for s in samples for i in s.candidate_item_ids + s.history_item_ids}
            )
        else:
            item_keys = sorted(data.idx2item.keys())
        serp_ids = (
            sorted({s.user_id for s in samples})
            if samples and cfg["dataset"].get("knowledge_items_only_in_samples", True)
            else sorted(data.idx2serp.keys())
        )
        item_keys, serp_ids = cap_knowledge_ids(
            item_keys, serp_ids, llm_cfg,
            sample_item_ids=sample_item_ids, sample_user_ids=sample_user_ids,
        )
        n_item_todo = sum(
            1
            for i in item_keys
            if str(i) not in items
            and job_belongs_to_shard(str(i), shard_id, num_shards)
        )
        n_user_todo = sum(
            1
            for s in serp_ids
            if str(s) not in users
            and job_belongs_to_shard(str(s), shard_id, num_shards)
        )
        est_h = estimate_knowledge_hours(
            n_item_todo, n_user_todo, batch_size=batch_size
        )
        print(
            f"Knowledge plan: {n_item_todo} items + {n_user_todo} users "
            f"(batch={batch_size}, max_new_tokens={llm_cfg.get('max_new_tokens', 512)}; "
            f"rough ETA ~{est_h:.1f}h at ~55s/batch"
            f"{f'; shard {shard_id + 1}/{num_shards}' if num_shards > 1 else ''})"
        )
        item_jobs = []
        for iid in item_keys:
            key = str(iid)
            if key in items or not job_belongs_to_shard(key, shard_id, num_shards):
                continue
            title, attrs = data.item_meta(iid)
            row = data.items.loc[data.items["item_id"] == data.idx2item[iid]].iloc[0]
            brand = str(row.get("brand", "") or "unknown")
            price = float(row.get("price", 0) or 0)
            prompt = build_item_knowledge_prompt_avito(title, brand, attrs, price)
            item_jobs.append((key, prompt))
        llm_generate_loop(
            gen,
            item_jobs,
            items,
            batch_method="generate_item_knowledge_batch",
            llm_cfg=llm_cfg,
            users=users,
            items=items,
            store=checkpoint,
            desc=f"llm-items{'' if num_shards <= 1 else f'-s{shard_id}'}",
        )
        user_jobs = []
        for sid in serp_ids:
            key = str(sid)
            if key in users or not job_belongs_to_shard(key, shard_id, num_shards):
                continue
            serp_x = data.idx2serp[sid]
            row = data.items.loc[data.items["serp_x"] == serp_x].iloc[0]
            cat = str(row.get("query_infm_logical_category", "Transport"))
            loc = str(row.get("query_loc", ""))
            prompt = build_user_preference_prompt_avito(sid, cat, loc)
            user_jobs.append((key, prompt))
        llm_generate_loop(
            gen,
            user_jobs,
            users,
            batch_method="generate_user_preference_batch",
            llm_cfg=llm_cfg,
            users=users,
            items=items,
            store=checkpoint,
            desc=f"llm-serps{'' if num_shards <= 1 else f'-s{shard_id}'}",
        )
    else:
        if samples and cfg["dataset"].get("knowledge_items_only_in_samples", True):
            movie_ids = sorted(
                {
                    i
                    for s in samples
                    for i in s.candidate_item_ids + s.history_item_ids + [s.target_item_id]
                }
            )
        else:
            movie_ids = _all_catalog_item_ids(data)
        user_ids = sorted(data.user_items.keys())
        if samples and cfg["dataset"].get("knowledge_items_only_in_samples", True):
            user_ids = sorted({s.user_id for s in samples})
        movie_ids, user_ids = cap_knowledge_ids(
            movie_ids, user_ids, llm_cfg,
            sample_item_ids=sample_item_ids, sample_user_ids=sample_user_ids,
        )
        n_item_todo = sum(
            1
            for i in movie_ids
            if str(int(i)) not in items
            and job_belongs_to_shard(str(int(i)), shard_id, num_shards)
        )
        n_user_todo = sum(
            1
            for u in user_ids
            if str(u) not in users
            and job_belongs_to_shard(str(u), shard_id, num_shards)
        )
        est_h = estimate_knowledge_hours(
            n_item_todo, n_user_todo, batch_size=batch_size
        )
        print(
            f"Knowledge plan: {n_item_todo} items + {n_user_todo} users "
            f"(unique in samples: {len(movie_ids)} items, {len(user_ids)} users; "
            f"batch={batch_size}; rough ETA ~{est_h:.1f}h"
            f"{f'; shard {shard_id + 1}/{num_shards}' if num_shards > 1 else ''})"
        )
        item_jobs = []
        for iid in movie_ids:
            key = str(int(iid))
            if key in items or not job_belongs_to_shard(key, shard_id, num_shards):
                continue
            item_jobs.append((key, _item_knowledge_prompt(data, int(iid))))
        llm_generate_loop(
            gen,
            item_jobs,
            items,
            batch_method="generate_item_knowledge_batch",
            llm_cfg=llm_cfg,
            users=users,
            items=items,
            store=checkpoint,
            desc=f"llm-items{'' if num_shards <= 1 else f'-s{shard_id}'}",
        )
        user_jobs = []
        for uid in user_ids:
            key = str(uid)
            if key in users or not job_belongs_to_shard(key, shard_id, num_shards):
                continue
            user_jobs.append((key, _user_preference_prompt(data, uid, hist_len)))
        llm_generate_loop(
            gen,
            user_jobs,
            users,
            batch_method="generate_user_preference_batch",
            llm_cfg=llm_cfg,
            users=users,
            items=items,
            store=checkpoint,
            desc=f"llm-users{'' if num_shards <= 1 else f'-s{shard_id}'}",
        )

    if num_shards > 1:
        checkpoint.save_final(users, items)
        print(
            f"Shard {shard_id + 1}/{num_shards} checkpoints saved under "
            f"{store.root / 'shards' / f'shard_{shard_id}'}"
        )
    else:
        store.save(users, items, meta=meta)
    return store


def stage_merge_knowledge(cfg: dict) -> None:
    llm_cfg = cfg["llm"]
    num_shards = max(1, int(llm_cfg.get("num_shards", 1)))
    if num_shards <= 1:
        print("merge_knowledge: num_shards=1, nothing to merge")
        return
    want_gen = (
        "template"
        if llm_cfg.get("use_template_generator")
        else llm_cfg.get("backend", "qwen")
    )
    from ur4rec.llm.hf_chat_generator import resolve_model_id

    meta = {
        "generator": want_gen,
        "model_name": resolve_model_id(llm_cfg),
        "num_shards": num_shards,
    }
    knowledge_dir = Path(llm_cfg["knowledge_dir"])
    n_items, n_users = merge_knowledge_shards(knowledge_dir, num_shards, meta=meta)
    print(f"Merged knowledge -> {knowledge_dir}: {n_items} items, {n_users} users")


def load_data_and_samples(cfg: dict, root: Path):
    ds = cfg["dataset"]
    name = ds.get("name", "movielens-1m")
    hl = ds["history_len"]
    nc = ds["num_candidates"]

    if name == "avito":
        data = AvitoSERP.from_parquet(
            root / ds["items_path"],
            root / ds.get("users_path", ""),
            label_field=ds.get("label_field", "contacts"),
            min_serp_size=int(ds.get("min_serp_size", 10)),
        )
        train_u, val_u, test_u = data.train_val_test_split(
            ds["train_ratio"], ds["val_ratio"], cfg["seed"]
        )
        if ds.get("max_train_serps"):
            train_u = train_u[: ds["max_train_serps"]]
        if ds.get("max_val_serps"):
            val_u = val_u[: ds["max_val_serps"]]
        if ds.get("max_test_serps"):
            test_u = test_u[: ds["max_test_serps"]]
        train_s = data.build_rerank_samples(train_u, hl, nc, cfg["seed"])
        val_s = data.build_rerank_samples(val_u, hl, nc, cfg["seed"] + 1)
        test_s = data.build_rerank_samples(test_u, hl, nc, cfg["seed"] + 2)
        return data, train_s, val_s, test_s

    if name == "amazon-books":
        ab_root = download_amazon_books(root / ds["data_dir"])
        data = AmazonBooks(
            ab_root,
            min_rating=float(ds.get("min_rating", 4.0)),
            max_reviews=ds.get("max_reviews"),
        )
    elif name == "steam":
        st_root = download_steam(root / ds["data_dir"])
        data = SteamReviews(
            st_root,
            min_hours=float(ds.get("min_hours", 1.0)),
            max_reviews=ds.get("max_reviews"),
        )
    else:
        ml_root = download_movielens_1m(root / ds["data_dir"])
        k_core = ds["k_core"] if "k_core" in ds else None
        data = MovieLens1M(
            ml_root,
            min_rating=float(ds.get("min_rating", 4.0)),
            k_core=k_core,
        )

    train_u, val_u, test_u = data.train_val_test_split(
        ds["train_ratio"], ds["val_ratio"], cfg["seed"]
    )
    if ds.get("max_train_users"):
        train_u = train_u[: ds["max_train_users"]]
    if ds.get("max_val_users"):
        val_u = val_u[: ds["max_val_users"]]
    if ds.get("max_test_users"):
        test_u = test_u[: ds["max_test_users"]]

    cand_mode = ds.get("candidate_mode", "random")
    if isinstance(data, MovieLens1M) and cand_mode == "mf_topk":
        mf_cfg = ds.get("mf", {})
        mf_device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
        data.fit_mf_candidates(
            train_u,
            dim=int(mf_cfg.get("dim", 64)),
            epochs=int(mf_cfg.get("epochs", 30)),
            seed=cfg["seed"],
            device=mf_device,
        )

    train_s = data.build_rerank_samples(
        train_u, hl, nc, cfg["seed"], candidate_mode=cand_mode
    )
    val_s = data.build_rerank_samples(
        val_u, hl, nc, cfg["seed"] + 1, candidate_mode=cand_mode
    )
    test_s = data.build_rerank_samples(
        test_u, hl, nc, cfg["seed"] + 2, candidate_mode=cand_mode
    )
    return data, train_s, val_s, test_s


def item_knowledge_text(
    data: MovieLens1M | AvitoSERP | AmazonBooks | SteamReviews,
    items: dict[str, str],
    iid: int,
) -> str:
    key = str(int(iid))
    if key in items:
        return items[key]
    if isinstance(data, (MovieLens1M, AmazonBooks, SteamReviews)):
        title, attrs = (
            data.movie_meta(int(iid))
            if isinstance(data, MovieLens1M)
            else data.item_meta(int(iid))
        )
        return template_item_knowledge(title, attrs)
    title, attrs = data.item_meta(int(iid))
    row = data.items.loc[data.items["item_id"] == data.idx2item[iid]].iloc[0]
    brand = str(row.get("brand", "") or "unknown")
    return template_item_knowledge(f"{title} ({brand})", attrs)


def user_preference_text(
    data: MovieLens1M | AvitoSERP | AmazonBooks | SteamReviews,
    users: dict[str, str],
    sample: RerankSample,
    history_len: int,
) -> str:
    key = str(sample.user_id)
    if key in users:
        return users[key]
    if isinstance(data, (MovieLens1M, AmazonBooks, SteamReviews)):
        seq = data.user_items.get(sample.user_id, [])
        titles, genres = [], []
        for mid in seq[-history_len:]:
            if isinstance(data, MovieLens1M):
                t, g = data.movie_meta(mid)
            else:
                t, g = data.item_meta(mid)
            titles.append(t)
            genres.append(g)
        return template_user_preference(sample.user_id, titles, genres)
    serp_x = data.idx2serp[sample.user_id]
    row = data.items.loc[data.items["serp_x"] == serp_x].iloc[0]
    cat = str(row.get("query_infm_logical_category", "Transport"))
    return template_user_preference(sample.user_id, [cat], [cat])


def encode_user_aggr(
    encoder: FrozenTextEncoder,
    store: KnowledgeStore,
    sample: RerankSample,
    device: torch.device,
    history_len: int,
    data: MovieLens1M | AvitoSERP | AmazonBooks | SteamReviews,
) -> torch.Tensor:
    users, items = store.load()
    u_text = user_preference_text(data, users, sample, history_len)
    eu = encoder.encode([u_text], device)[0]
    hist_ids = list(sample.history_item_ids[-history_len:])
    if len(hist_ids) < history_len:
        hist_ids = [0] * (history_len - len(hist_ids)) + hist_ids
    know = []
    for iid in hist_ids:
        if iid == 0:
            know.append(torch.zeros(encoder.hidden_size, device=device))
        else:
            text = item_knowledge_text(data, items, iid)
            know.append(encoder.encode([text], device)[0])
    return encoder.aggregate_preference(eu, know)


def make_dlcm_backbone(
    data: MovieLens1M | AvitoSERP, cfg: dict, device: torch.device, use_aug: bool = False
) -> DLCMReranker:
    model = DLCMReranker(data.num_items, cfg["backbone"]["hidden_dim"], use_aug=use_aug).to(device)
    model.resize_users(data.num_users + 1)
    return model


def load_dlcm_backbone(
    data: MovieLens1M | AvitoSERP, cfg: dict, device: torch.device, path: Path, use_aug: bool = False
) -> DLCMReranker:
    model = make_dlcm_backbone(data, cfg, device, use_aug=use_aug)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def stage_backbone(
    cfg: dict, data: MovieLens1M | AvitoSERP, train_s: list, val_s: list, device: torch.device, out: Path
) -> DLCMReranker:
    model = make_dlcm_backbone(data, cfg, device, use_aug=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["backbone"]["train_lr"])
    best_ndcg, ckpt = 0.0, out / "backbone.pt"

    for epoch in range(cfg["backbone"]["train_epochs"]):
        model.train()
        rng = np.random.default_rng(cfg["seed"] + epoch)
        rng.shuffle(train_s)
        losses = []
        for i in range(0, len(train_s), cfg["backbone"]["train_batch_size"]):
            batch = train_s[i : i + cfg["backbone"]["train_batch_size"]]
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for s in batch:
                item_t = torch.tensor([s.candidate_item_ids], device=device)
                user_t = torch.tensor([s.user_id], device=device)
                scores = model(item_t, user_t)
                lab = torch.tensor([s.labels], device=device, dtype=torch.float32)
                loss = listwise_ce_loss(scores, lab) / len(batch)
                loss.backward()
                batch_loss += loss.item() * len(batch)
            opt.step()
            losses.append(batch_loss / len(batch))
        metrics = evaluate_backbone(model, val_s, device)
        print(f"[backbone] epoch {epoch+1} loss={np.mean(losses):.4f} val_ndcg@10={metrics.get('ndcg@10',0):.4f}")
        if metrics.get("ndcg@10", 0) >= best_ndcg:
            best_ndcg = metrics["ndcg@10"]
            torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model


def stage_pretrain(
    cfg: dict,
    data: MovieLens1M | AvitoSERP,
    store: KnowledgeStore,
    encoder: FrozenTextEncoder,
    backbone: DLCMReranker,
    train_s: list,
    device: torch.device,
    out: Path,
) -> UR4RecRetriever:
    h = cfg["backbone"]["hidden_dim"]
    hist = cfg["dataset"]["history_len"]
    aggr_dim = h * (1 + hist)
    retriever = UR4RecRetriever(
        hidden_size=h,
        num_proxies=cfg["retriever"]["num_proxies"],
        num_layers=cfg["retriever"]["num_layers"],
        aggr_dim=aggr_dim,
    ).to(device)
    opt = torch.optim.AdamW(retriever.parameters(), lr=cfg["retriever"]["pretrain_lr"])
    M = cfg["dataset"]["num_negatives_cl"]
    alpha = cfg["retriever"]["alpha_cf"]
    tau = cfg["retriever"]["temperature"]

    for epoch in range(cfg["retriever"]["pretrain_epochs"]):
        retriever.train()
        rng = np.random.default_rng(cfg["seed"] + epoch)
        rng.shuffle(train_s)
        cl_losses, cf_losses = [], []
        for s in tqdm(train_s, desc=f"pretrain e{epoch+1}"):
            e_aggr = encode_user_aggr(encoder, store, s, device, hist, data).unsqueeze(0)
            proxy_pref = retriever.forward_preference_filter(
                e_aggr, return_proxy_tokens=True
            )

            pos_id = s.target_item_id
            with torch.no_grad():
                h_pos = backbone.item_hidden(torch.tensor([pos_id], device=device))
            e_pos = retriever.forward_item_only(h_pos)

            negs = []
            if isinstance(data, AvitoSERP):
                all_items = data.all_item_indices()
            elif isinstance(data, MovieLens1M):
                all_items = data.movies["movieId"].astype(int).values
            else:
                all_items = data.items.index.values
            seen = set(s.history_item_ids + [pos_id])
            pool = [i for i in all_items if i not in seen]
            rng.shuffle(pool)
            for nid in pool[:M]:
                negs.append(
                    retriever.forward_item_only(
                        backbone.item_hidden(torch.tensor([nid], device=device))
                    )
                )
            neg_t = torch.cat(negs, dim=0).unsqueeze(0)  # [1, M, D]
            l_cl = info_nce_loss(proxy_pref, e_pos, neg_t, tau)

            h_cand = backbone.item_hidden(torch.tensor(s.candidate_item_ids, device=device))
            logits_cf = []
            for j in range(h_cand.size(0)):
                hj = h_cand[j : j + 1].unsqueeze(0)
                logits_cf.append(retriever.forward_joint(hj, e_aggr, mode="pim"))
            logits_cf = torch.cat(logits_cf, dim=0)
            labels_cf = torch.tensor(
                [float(x) for x in s.labels], device=device, dtype=torch.float32
            )
            l_cf = torch.nn.functional.binary_cross_entropy_with_logits(logits_cf, labels_cf)

            loss = l_cl + alpha * l_cf
            opt.zero_grad()
            loss.backward()
            opt.step()
            cl_losses.append(l_cl.item())
            cf_losses.append(l_cf.item())
        print(f"[pretrain] epoch {epoch+1} L_CL={np.mean(cl_losses):.4f} L_CF={np.mean(cf_losses):.4f}")

    torch.save(retriever.state_dict(), out / "retriever_pretrain.pt")
    return retriever


def _retriever_augmentations(
    retriever: UR4RecRetriever,
    h_items: torch.Tensor,
    e_aggr: torch.Tensor,
) -> torch.Tensor:
    """Per-candidate augmentations [B, L, D] for DLCM."""
    aug = retriever.forward_joint(h_items, e_aggr, mode="aug")
    if aug.dim() == 2:
        aug = aug.unsqueeze(1)
    return aug


def _ur4rec_scores_for_sample(
    cfg: dict,
    store: KnowledgeStore,
    encoder: FrozenTextEncoder,
    backbone: DLCMReranker,
    retriever: UR4RecRetriever,
    s,
    device: torch.device,
    data: MovieLens1M | AvitoSERP,
    *,
    blend_alpha: float = 1.0,
) -> np.ndarray:
    """UR4Rec scores with optional convex blend toward base (alpha=0 → pure base)."""
    hist = cfg["dataset"]["history_len"]
    e_aggr = encode_user_aggr(encoder, store, s, device, hist, data).unsqueeze(0)
    item_t = torch.tensor([s.candidate_item_ids], device=device)
    user_t = torch.tensor([s.user_id], device=device)
    h_items = backbone.item_hidden(item_t)
    aug = _retriever_augmentations(retriever, h_items, e_aggr)
    base_sc = backbone(item_t, user_t, aug=torch.zeros_like(aug)).squeeze(0)
    if blend_alpha <= 0.0:
        saved_aug = backbone.use_aug
        backbone.use_aug = False
        sc = backbone(item_t, user_t).squeeze(0)
        backbone.use_aug = saved_aug
    elif blend_alpha >= 1.0:
        sc = backbone(item_t, user_t, aug=aug).squeeze(0)
    else:
        aug_sc = backbone(item_t, user_t, aug=aug).squeeze(0)
        sc = (1.0 - blend_alpha) * base_sc + blend_alpha * aug_sc
    return sc.detach().cpu().numpy()


def tune_ur4rec_blend_alpha(
    cfg: dict,
    store: KnowledgeStore,
    encoder: FrozenTextEncoder,
    backbone: DLCMReranker,
    retriever: UR4RecRetriever,
    val_s: list,
    device: torch.device,
    data: MovieLens1M | AvitoSERP,
) -> float:
    """Pick blend alpha on val; alpha=0 is pure base, so val ndcg never below base."""
    grid = cfg.get("joint", {}).get("blend_alpha_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    best_a, best_ndcg = 0.0, -1.0
    backbone.eval()
    retriever.eval()
    backbone.use_aug = True
    with torch.no_grad():
        for alpha in grid:
            scores_list, labels_list = [], []
            for s in val_s:
                scores_list.append(_ur4rec_scores_for_sample(
                    cfg, store, encoder, backbone, retriever, s, device, data, blend_alpha=alpha
                ))
                labels_list.append(np.array(s.labels))
            ndcg = evaluate_batch(scores_list, labels_list, ks=(10,)).get("ndcg@10", 0.0)
            print(f"[joint] blend_alpha={alpha:.2f} val_ndcg@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_a = float(alpha)
    print(f"[joint] selected blend_alpha={best_a:.2f} (val_ndcg@10={best_ndcg:.4f})")
    return best_a


def stage_joint(
    cfg: dict,
    data: MovieLens1M | AvitoSERP,
    store: KnowledgeStore,
    encoder: FrozenTextEncoder,
    backbone: DLCMReranker,
    retriever: UR4RecRetriever,
    train_s: list,
    val_s: list,
    device: torch.device,
    out: Path,
) -> None:
    backbone.expand_for_augmentation()
    jcfg = cfg["joint"]
    bb_lr = float(jcfg.get("backbone_lr", jcfg["lr"]))
    rt_lr = float(jcfg.get("retriever_lr", jcfg["lr"]))
    opt = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": bb_lr},
            {"params": retriever.parameters(), "lr": rt_lr},
        ]
    )
    hist = cfg["dataset"]["history_len"]
    best_ndcg = -1.0
    bad_epochs = 0
    patience = int(jcfg.get("patience", 3))
    min_delta = float(jcfg.get("min_delta", 0.0))
    base_loss_w = float(jcfg.get("base_loss_weight", 0.1))
    ckpt_path = out / "ur4rec_joint.pt"
    for epoch in range(jcfg["epochs"]):
        backbone.train()
        retriever.train()
        losses = []
        rng = np.random.default_rng(cfg["seed"] + epoch)
        rng.shuffle(train_s)
        for s in tqdm(train_s, desc=f"joint e{epoch+1}"):
            e_aggr = encode_user_aggr(encoder, store, s, device, hist, data).unsqueeze(0)
            item_t = torch.tensor([s.candidate_item_ids], device=device)
            h_items = backbone.item_hidden(item_t)
            aug = _retriever_augmentations(retriever, h_items, e_aggr)
            user_t = torch.tensor([s.user_id], device=device)
            zero_aug = torch.zeros_like(aug)
            base_scores = backbone(item_t, user_t, aug=zero_aug)
            aug_scores = backbone(item_t, user_t, aug=aug)
            lab = torch.tensor([s.labels], device=device, dtype=torch.float32)
            loss = listwise_ce_loss(aug_scores, lab)
            if base_loss_w > 0:
                loss = loss + base_loss_w * listwise_ce_loss(base_scores, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_m = evaluate_ur4rec(cfg, store, encoder, backbone, retriever, val_s, device, data)
        ndcg = val_m.get("ndcg@10", 0.0)
        print(
            f"[joint] epoch {epoch+1} loss={np.mean(losses):.4f} "
            f"val_ndcg@10={ndcg:.4f}"
        )
        if ndcg >= best_ndcg + min_delta:
            best_ndcg = ndcg
            bad_epochs = 0
            torch.save(
                {"backbone": backbone.state_dict(), "retriever": retriever.state_dict()},
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[joint] early stop at epoch {epoch+1}, best val_ndcg@10={best_ndcg:.4f}")
                break
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        backbone.load_state_dict(ckpt["backbone"])
        retriever.load_state_dict(ckpt["retriever"])
    blend_alpha = tune_ur4rec_blend_alpha(
        cfg, store, encoder, backbone, retriever, val_s, device, data
    )
    torch.save({"blend_alpha": blend_alpha}, out / "ur4rec_joint_meta.pt")


def evaluate_backbone(model: DLCMReranker, samples: list[RerankSample], device: torch.device) -> dict:
    model.eval()
    scores_list, labels_list = [], []
    with torch.no_grad():
        for s in samples:
            item_t = torch.tensor([s.candidate_item_ids], device=device)
            user_t = torch.tensor([s.user_id], device=device)
            sc = model(item_t, user_t).cpu().numpy().reshape(-1)
            scores_list.append(sc)
            labels_list.append(np.array(s.labels))
    return evaluate_batch(scores_list, labels_list, ks=(1, 5, 10))


def evaluate_ur4rec(
    cfg: dict,
    store: KnowledgeStore,
    encoder: FrozenTextEncoder,
    backbone: DLCMReranker,
    retriever: UR4RecRetriever,
    samples: list[RerankSample],
    device: torch.device,
    data: MovieLens1M | AvitoSERP,
    blend_alpha: float = 1.0,
) -> dict:
    backbone.eval()
    retriever.eval()
    backbone.use_aug = True
    scores_list, labels_list = [], []
    with torch.no_grad():
        for s in samples:
            scores_list.append(
                _ur4rec_scores_for_sample(
                    cfg, store, encoder, backbone, retriever, s, device, data,
                    blend_alpha=blend_alpha,
                )
            )
            labels_list.append(np.array(s.labels))
    return evaluate_batch(scores_list, labels_list, ks=(1, 5, 10))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ur4rec/ur4rec_ml1m.yaml")
    parser.add_argument(
        "--stage",
        choices=["knowledge", "merge_knowledge", "backbone", "pretrain", "joint", "eval", "all"],
        default="all",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Override config gpu_id (0..N-1); use with CUDA_VISIBLE_DEVICES to pick a physical card",
    )
    parser.add_argument(
        "--knowledge-shard-id",
        type=int,
        default=None,
        help="Knowledge worker shard index (0..num_shards-1); overrides llm.shard_id",
    )
    parser.add_argument(
        "--knowledge-num-shards",
        type=int,
        default=None,
        help="Parallel knowledge GPUs; overrides llm.num_shards",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.knowledge_shard_id is not None:
        cfg["llm"]["shard_id"] = args.knowledge_shard_id
    if args.knowledge_num_shards is not None:
        cfg["llm"]["num_shards"] = args.knowledge_num_shards

    print(f"Config: {args.config.resolve()}")
    print(f"Knowledge dir: {cfg['llm']['knowledge_dir']}")
    print(f"LLM: backend={cfg['llm'].get('backend', 'qwen')}, "
          f"use_template={cfg['llm'].get('use_template_generator', False)}")
    num_shards = max(1, int(cfg["llm"].get("num_shards", 1)))
    if num_shards > 1:
        shard_id = int(cfg["llm"].get("shard_id", 0))
        print(f"Knowledge sharding: shard {shard_id + 1}/{num_shards}")

    if args.stage == "merge_knowledge":
        stage_merge_knowledge(cfg)
        return

    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id
    device = resolve_device(cfg)
    setup_cuda(cfg, device)
    print(f"Device: {describe_device(device)}")
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    data, train_s, val_s, test_s = load_data_and_samples(cfg, ROOT)
    print(
        f"Dataset: {cfg['dataset'].get('name', 'movielens-1m')} | "
        f"samples train/val/test: {len(train_s)}/{len(val_s)}/{len(test_s)}"
    )

    stages = (
        ["knowledge", "backbone", "pretrain", "joint", "eval"]
        if args.stage == "all"
        else [args.stage]
    )

    store = (
        build_knowledge(cfg, data, train_s + val_s + test_s)
        if "knowledge" in stages
        else KnowledgeStore(Path(cfg["llm"]["knowledge_dir"]))
    )

    needs_encoder = bool(set(stages) & {"backbone", "pretrain", "joint", "eval"})
    encoder = None
    if needs_encoder:
        encoder = FrozenTextEncoder(
            cfg["encoder"]["model_name"], cfg["encoder"]["max_length"]
        ).to(device)

    backbone = None
    retriever = None

    if "backbone" in stages or args.stage == "all":
        backbone = stage_backbone(cfg, data, train_s, val_s, device, out)
    elif any(s in stages for s in ["pretrain", "joint", "eval"]):
        backbone = load_dlcm_backbone(data, cfg, device, out / "backbone.pt", use_aug=False)

    if "pretrain" in stages:
        backbone = backbone or load_dlcm_backbone(data, cfg, device, out / "backbone.pt", use_aug=False)
        retriever = stage_pretrain(cfg, data, store, encoder, backbone, train_s, device, out)

    if "joint" in stages:
        backbone = backbone or load_dlcm_backbone(data, cfg, device, out / "backbone.pt", use_aug=False)
        hist = cfg["dataset"]["history_len"]
        aggr_dim = cfg["backbone"]["hidden_dim"] * (1 + hist)
        retriever = retriever or UR4RecRetriever(
            num_proxies=cfg["retriever"]["num_proxies"],
            num_layers=cfg["retriever"]["num_layers"],
            aggr_dim=aggr_dim,
        ).to(device)
        retriever.load_state_dict(torch.load(out / "retriever_pretrain.pt", map_location=device, weights_only=True))
        stage_joint(cfg, data, store, encoder, backbone, retriever, train_s, val_s, device, out)

    if "eval" in stages:
        base_backbone = load_dlcm_backbone(data, cfg, device, out / "backbone.pt", use_aug=False)
        ckpt = torch.load(out / "ur4rec_joint.pt", map_location=device, weights_only=True)
        backbone = make_dlcm_backbone(data, cfg, device, use_aug=True)
        backbone.load_state_dict(ckpt["backbone"])
        hist = cfg["dataset"]["history_len"]
        aggr_dim = cfg["backbone"]["hidden_dim"] * (1 + hist)
        retriever = UR4RecRetriever(
            num_proxies=cfg["retriever"]["num_proxies"],
            num_layers=cfg["retriever"]["num_layers"],
            aggr_dim=aggr_dim,
        ).to(device)
        retriever.load_state_dict(ckpt["retriever"])
        meta_path = out / "ur4rec_joint_meta.pt"
        blend_alpha = 1.0
        if meta_path.exists():
            meta = torch.load(meta_path, map_location=device, weights_only=False)
            blend_alpha = float(meta.get("blend_alpha", 1.0))
        base_m = evaluate_backbone(base_backbone, test_s, device)
        ur_pure = evaluate_ur4rec(
            cfg, store, encoder, backbone, retriever, test_s, device, data, blend_alpha=1.0
        )
        ur_m = evaluate_ur4rec(
            cfg, store, encoder, backbone, retriever, test_s, device, data, blend_alpha=blend_alpha
        )
        print("Base backbone:", json.dumps(base_m, indent=2))
        print(f"UR4Rec (alpha=1.0):", json.dumps(ur_pure, indent=2))
        print(f"UR4Rec (alpha={blend_alpha:.2f}, tuned on val):", json.dumps(ur_m, indent=2))
        (out / "metrics_test.json").write_text(
            json.dumps(
                {
                    "base": base_m,
                    "ur4rec": ur_m,
                    "ur4rec_pure": ur_pure,
                    "blend_alpha": blend_alpha,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
