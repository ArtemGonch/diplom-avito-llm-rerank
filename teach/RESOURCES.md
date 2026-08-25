# LLM-reranking Resources

## Knowledge

- [UR4Rec, COLING 2025](https://aclanthology.org/2025.coling-main.45/)
  Первичный источник для retriever blocks, Eq. 4–9, Figure 3 masks и published protocol.
- [MemRerank, 2026](https://arxiv.org/abs/2603.29247)
  Preference memory и downstream reward для personalized product reranking.
- [Amazon-C4 User Purchase History](https://huggingface.co/datasets/zhiyuanpeng/amazon-c4-user-purchase-history)
  Schema, temporal cutoff, splits и связь query–user–positive product.
- [ESCI Shopping Queries](https://github.com/amazon-science/esci-data)
  Первичный benchmark для graded query–product relevance.
- [Beyond Utility](https://arxiv.org/abs/2411.00331)
  Position bias, hallucination и многомерная оценка LLM recommender.
- [Локальный аудит задания 26.06](../docs/task_2026-06-26_artem.md)
  Факты о коде, данных, результатах и выбранном benchmark.

## Wisdom (Communities)

- [ACM RecSys](https://recsys.acm.org/)
  Принятые статьи и доклады для проверки, какие evaluation protocols считаются убедительными сообществом.
- [ACM SIGIR](https://sigir.org/)
  Product search, ranking и counterfactual evaluation; полезно для критики offline SERP экспериментов.

## Gaps

- Нет schema contract от владельца локальных Avito parquet, однозначно описывающего `user_id` и способ агрегации `contacts_daily`.
- Нет публичного датасета, одновременно совпадающего с Avito по объявлениям подержанных авто, фильтровому запросу, long-term history и impression logs.
