# Сравнение: UR4Rec vs Exp3RT vs LLM4Rerank

> Метрики актуализированы 2026-08-25. Legacy Avito NDCG `0.942` содержал target leakage, а прежний ML-1M `beat_base` предшествует UR4Rec correctness fixes. Corrected-v3 сейчас на stage `knowledge`; финальных метрик ещё нет. См. [актуальную точку входа](START_HERE.md) и [полный аудит](task_2026-06-26_artem.md).

Краткая шпаргалка для защиты / отчёта.
Источники: UR4Rec (COLING 2025), Exp3RT (SIGIR 2025), LLM4Rerank (WWW 2025, [arxiv:2406.12433](https://arxiv.org/abs/2406.12433)).

---

## Суть LLM4Rerank (WWW 2025)

**Проблема:** классические rerank-модели оптимизируют в основном **accuracy**; multi-objective (accuracy + diversity + fairness) требуют отдельных архитектур и плохо масштабируются при добавлении новых критериев.

**Идея:** reranking как **граф функций** + **zero-shot LLM** + Chain-of-Thought:

1. Каждый аспект — **узел графа** (Accuracy, Diversity, Fairness, …).
2. LLM на каждом шаге: переранжирует список кандидатов → выбирает **следующий узел** (полный граф, кроме Stop).
3. Вход: user info, candidate list (~20 items), **Goal** — одно предложение от оператора («приоритет — diversity»).
4. История шагов хранится в **Pool**; LLM сам решает маршрут по Goal и контексту.
5. **Без fine-tuning** — prompt-only, backbone Llama-2-13B в paper.

**Отличие от Exp3RT:** Exp3RT учит LLM **профили и рейтинг** (SFT); LLM4Rerank не обучает веса, а **оркестрирует** уже готовый LLM под multi-aspect rerank.

**Отличие от UR4Rec:** UR4Rec — **обучаемый** DLCM + retriever + offline knowledge; LLM4Rerank — **zero-shot** промпт-граф, multi-objective из коробки.

**Для Avito:** conceptually полезен (accuracy + diversity брендов + fairness geo/price), но **latency** (~секунды на LLM-вызов × несколько узлов) — не для online SERP без distillation.

---

## Сравнительная таблица

| Критерий | **UR4Rec** (COLING 2025) | **Exp3RT** (SIGIR 2025) | **LLM4Rerank** (WWW 2025) |
|----------|--------------------------|-------------------------|---------------------------|
| **Задача** | Listwise rerank (ranking) | Rating prediction + explainable profiles | Multi-aspect listwise rerank |
| **Базовый ranker** | DLCM (GRU listwise) | — (generative rating) | Zero-shot LLM permutation |
| **Роль LLM** | Offline: item/user **knowledge** JSON | SFT: preference → user/item profile → rating | Online/offline: **rerank steps** по аспектам |
| **Обучение** | Backbone + retriever + joint | 3–4 stage QLoRA (Llama-3) | **Zero-shot**, без SFT в paper |
| **User preferences** | LLM knowledge + proxy retriever | Текстовые профили из отзывов | User features + Goal sentence |
| **Query / context** | Слабо (static user emb.) | Query в rating prompt | Goal + rerank history (Pool) |
| **Multi-objective** | Accuracy (NDCG/MAP) | Accuracy (RMSE/MAE) + explainability | **Accuracy + Diversity + Fairness** (граф узлов) |
| **Масштаб кандидатов** | 50–100 items, BERT encode | 1 item / rating | ~20 items (paper protocol) |
| **Latency** | BERT + GRU (ms) | vLLM batch (offline test) | **High** (multi-step LLM CoT) |
| **Интерпретируемость** | Knowledge JSON | Natural-language profiles | CoT trace + aspect nodes |
| **Нужны отзывы** | Нет (metadata + ratings) | **Да** (review text) | Нет (tabular user + item text) |
| **Paper datasets** | ML-1M, Amazon, Steam | Amazon-Book, Steam | ML-1M, KuaiRand, Douban |
| **Key metrics** | NDCG@10, MAP@10 | RMSE, MAE | HR, NDCG, α-NDCG, MAD |
| **Наш repro (основной)** | ML-1M corrected-v3 running; финальных метрик нет. Legacy beat_base: 0.285 → 0.300, не использовать как corrected claim | Amazon chained paper-full: expected RMSE **0.562** vs paper 0.651 | Не воспроизводили (conceptual baseline) |
| **Наш repro (Avito)** | Legacy smoke; rerun required | **Exp3RT-style leakage-free**: graded NDCG@10 **0.3413** vs position **0.3126** | План: Goal-based heuristic / будущий C-UR4Rec |

---

## Что говорить на защите про «+1 алгоритм»

> Мы добавили **LLM4Rerank** (WWW 2025) как **концептуальный baseline** для multi-objective rerank: в отличие от UR4Rec (accuracy-first, обучаемый retriever) и Exp3RT (rating + профили из отзывов), LLM4Rerank **не требует fine-tuning** и явно балансирует accuracy/diversity/fairness через граф узлов и пользовательский **Goal**.
> На Avito нет отзывов → full Exp3RT 4-stage нерелевантен; UR4Rec дорог offline и не учитывает query SERP. Поэтому мы прогнали **Exp3RT-style pseudo-profiles** (structured contacts → fit score) и предложили **C-UR4Rec** как обучаемую альтернативу с query-conditioning. LLM4Rerank — аргумент «зачем multi-aspect» и «почему не pure LLM online» (latency).

---

## Связь с Avito Auto

| Метод | Как извлекать преференсы на Avito | Релевантность |
|-------|-----------------------------------|---------------|
| **Exp3RT-style** | top brand/model/price из contacts → text profile | ✅ **Сейчас лучший** (есть прогон) |
| **UR4Rec** | LLM knowledge по items + user history | ⚠️ дорого offline, без query |
| **LLM4Rerank** | user attrs + Goal («точность vs разнообразие брендов») | 💡 идея для product, не для prod latency |
| **C-UR4Rec** (наш) | UR4Rec + SERP query + gating | 🔬 proposed contribution |
