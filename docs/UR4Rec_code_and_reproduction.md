# UR4Rec: где код и как воспроизводить

**Статья:** Zhang et al., *Enhancing Reranking for Recommendation with LLMs through User Preference Retrieval* (COLING 2025)  
**PDF:** https://aclanthology.org/2025.coling-main.45/

---

## Главный вывод

**Официального публичного репозитория UR4Rec на GitHub / Gitee нет.**

В PDF и на ACL Anthology **нет** ссылки вида «Code available at …». Поиск по GitHub (`UR4Rec`, `user preference retrieval`, авторы RUC) не даёт репозитория с этой статьёй.

**Практический путь:** воспроизведение = **KAR (базовая линия) + свой модуль Retriever** по §3 paper.

---

## Что есть в экосистеме (близкий код)

### 1. KAR — прямой предшественник (главный baseline в Table 2)

UR4Rec явно строится поверх идеи KAR: LLM offline → augment vectors → reranker, но **добавляет retriever** вместо «тащить весь preference».

| | |
|---|---|
| **Paper** | Xi et al., arXiv:2306.10933, RecSys 2023 |
| **GitHub** | https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation |
| **Gitee (MindSpore)** | https://gitee.com/mindspore/models/tree/master/research/recommend/KAR |

**Структура KAR (PyTorch repo):**

```text
preprocess_amz.py
generate_data_and_prompt.py    # промпты для LLM (код генерации knowledge не выложен)
data/knowledge/                # готовые .klg (user + item knowledge)
RS/
  run_ctr.py
  run_rerank.py                  # ← rerank pipeline, ближе всего к UR4Rec stage 3
```

KAR уже даёт: MovieLens-1M / Amazon-Books, offline knowledge, `run_rerank.py`.  
**Не хватает:** proxy embeddings, cross-attention retriever, CL + Preference-Item Matching (§3.2).

---

### 2. Backbone rerankers из paper

| Backbone | Год | Код / данные |
|----------|-----|----------------|
| **PRM** | RecSys 2019 | https://github.com/zyhfjx2019/drr (из README hf4Academic/PRM) |
| **DLCM** | SIGIR 2018 | часто внутри rank2rec / drr |
| **SetRank** | SIGIR 2020 | отдельные реализации на GitHub |
| **SASRec** | ICDM 2018 | https://github.com/kang205/SASRec |
| **GRU4Rec** | ICLR 2016 | много форков |

Датасет rerank (PRM): https://github.com/rank2rec/rerank

---

### 3. Датасеты из UR4Rec (Appendix C)

| Dataset | Ссылка в paper | Скачать в репо |
|---------|----------------|----------------|
| MovieLens-1M | https://grouplens.org/datasets/movielens/1m/ | `data/movielens-1m/` |
| Amazon-Books 5-core | https://nijianmo.github.io/amazon/index.html | `data/amazon-books/` |
| Steam | https://github.com/kang205/SASRec | `data/steam/` |

```bash
python scripts/download_ur4rec_datasets.py --datasets all
# или по одному: movielens-1m amazon-books steam
```

Split: **8:1:1**, k-core, rerank **top-100** candidates.

---

## Архитектура UR4Rec (что писать самим)

По paper, 3 фазы (Algorithm 1, Appendix A):

```text
Phase 1 (offline, no LLM finetune)
  Llama2-Chat → user preference text su
  Llama2-Chat → item knowledge per history item
  Frozen BERT Encoder → eu, eik, e_u^aggr = concat

Phase 2 (pretrain retriever)
  K proxy embeddings P
  Cross-attention: candidate item = query, Z = LLM knowledge
  Loss: L_CL (InfoNCE/SimCSE-style) + α · L_CF (preference-item matching)

Phase 3 (joint train)
  e_aug = Agg(Retriever([P; h_i], e_u^aggr))
  Concat e_aug to backbone reranker input
  L_train = L_RS (MAP/NDCG loss)
```

**Гиперпараметры (§4.4):**

- LLM: Llama2-Chat  
- Retriever: BERT-base 110M, proxies K=8 (DLCM/PRM/SetRank) или 16 (GRU/SASRec)  
- history len: 10 / 150  
- candidates: 100, negatives in pretrain: 10  
- AdamW, lr 1e-4 / 1e-3, batch 32, dim 768  

---

## План воспроизведения для диплома / Avito

### Вариант A — «как в paper» (ML-1M / Amazon)

1. Клонировать **KAR**, прогнать `run_rerank` baseline.  
2. Реализовать `ur4rec/retriever.py` (Transformer + proxies + mask Fig.3c).  
3. Подключить к PRM или SASRec (проще SASRec — один репо).  
4. Сравнить с KAR и +Llama2-aug на NDCG@5/10.

### Вариант B — «UR4Rec-идея на Avito parquet» (ваши данные)

1. **Offline:** LLM summaries по `title` + `description_short` + user history из `users_with_history`.  
2. **Retriever:** тот же BERT + cross-attention, candidate = `item_id` в SERP.  
3. **Backbone:** LightGBM top-20 → UR4Rec-style augment → rerank (или обучить маленький PRM/DLCM на SERP).  
4. Метрики: NDCG@K по `serp_is_positive` / clicks (если есть labels в items).

### Вариант C — запросить код у авторов

- Haobo Zhang: zhanghb@ruc.edu.cn  
- Zhicheng Dou: dou@ruc.edu.cn  
- Qiannan Zhu: zhuqiannan@bnu.edu.cn  

Имеет смысл написать коротко: reproduction for thesis, Avito Auto vertical.

---

## Что положить в репо `DIPLOM/avito` (следующий шаг)

```text
avito/
  external/
    Open-World-Knowledge-Augmented-Recommendation/   # git submodule KAR
    SASRec/                                        # optional
  src/ur4rec/
    retriever.py
    train_pretrain.py
    train_joint.py
  configs/ur4rec_ml1m.yaml
  docs/UR4Rec_code_and_reproduction.md   # этот файл
```

---

## LLM для профилей (§3.1)

### Модели (выбор в конфиге `llm.backend`)

| backend | HF model | Примечание |
|---------|----------|------------|
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | **рекомендуется**, открыта, 7B, V100 4bit |
| `qwen3` | `Qwen/Qwen3-8B` | новее, нужен `trust_remote_code` |
| `deepseek` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | R1-distill 7B, открыта |
| `deepseek-chat` | `deepseek-ai/deepseek-llm-7b-chat` | классический DeepSeek-Chat 7B |
| `llama2` | `meta-llama/Llama-2-7b-chat-hf` | как в статье, **нужен approve на HF** |

Конфиги:

- `configs/ur4rec_smoke_qwen.yaml`
- `configs/ur4rec_smoke_deepseek.yaml`
- `configs/ur4rec_avito_smoke_qwen.yaml`

Промпты те же, что в UR4Rec §3.1; меняется только генератор текста.

## Llama2-Chat для профилей (§3.1, как в статье)

**В статье:** `su = LLM(pu)`, `si = LLM(pi)` через **Llama2-Chat** (не шаблоны).

В репозитории: `src/ur4rec/llm/llama2_generator.py`, промпты — `src/ur4rec/llm/prompts.py` (§3.1.1 / §3.1.2).

### Подготовка

1. Доступ к [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) на HuggingFace.
2. `export HF_TOKEN=...` или `huggingface-cli login`
3. `pip install -r requirements.txt` (добавлены `bitsandbytes`, `accelerate`)
4. Удалить старый кэш-шаблоны, если были:
   ```bash
   rm -rf data/movielens-1m/knowledge data/avito/knowledge
   ```

### Запуск (сначала только knowledge)

```bash
# Qwen smoke (~30–60 min on V100: caps + batch_size=4 + max_new_tokens=128):
CUDA_VISIBLE_DEVICES=0 python scripts/run_ur4rec.py \
  --config configs/ur4rec_smoke_qwen.yaml --stage knowledge

Полный прогон ML-1M (тысячи фильмов из сэмплов — часы): в конфиге убери
`llm.knowledge_max_items` / `knowledge_max_users`, подними `max_new_tokens`.

# DeepSeek R1-Distill:
CUDA_VISIBLE_DEVICES=0 python scripts/run_ur4rec.py \
  --config configs/ur4rec_smoke_deepseek.yaml --stage knowledge
```

Кэш: `data/movielens-1m/knowledge_llama2/` (`users.json`, `items.json`, `meta.json` с `"generator": "llama2"`).

Полный ML-1M: `configs/ur4rec_ml1m.yaml` — **долго** (~6k пользователей + ~4k фильмов).

Отладка без LLM (не для статьи): в yaml `use_template_generator: true`.

---

## GPU на сервере (локальный repro)

На машине разработки: **4× Tesla V100-PCIE-32GB** (`nvidia-smi -L`).

В `configs/ur4rec_*.yaml`:

```yaml
device: cuda
gpu_id: 0          # логический индекс после CUDA_VISIBLE_DEVICES
cudnn_benchmark: true
tf32: true
```

Запуск на второй карте:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_ur4rec.py --config configs/ur4rec_smoke.yaml --stage all
# или
python scripts/run_ur4rec.py --config configs/ur4rec_smoke.yaml --gpu-id 2 --stage backbone
```

В логе должно быть: `Device: cuda:0 Tesla V100-PCIE-32GB ...`

---

## Ссылки одной строкой

| Ресурс | URL |
|--------|-----|
| UR4Rec paper | https://aclanthology.org/2025.coling-main.45/ |
| KAR code | https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation |
| PRM code | https://github.com/zyhfjx2019/drr |
| SASRec | https://github.com/kang205/SASRec |
