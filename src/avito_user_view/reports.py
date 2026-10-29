"""Readable private deliverables, independent from the interactive browser."""

from pathlib import Path


def summary_text(user, period):
    facts = {f["id"]: f for f in user["facts"]}
    parts = [user["label"], "История: " + " — ".join(period), ""]
    parts += (
        ["Краткий обзор"]
        + [facts[i]["text"] for i in user["summary"]["overview"]]
        + [""]
    )
    for section in user["summary"]["sections"]:
        parts.append(section["title"])
        parts.extend(facts[i]["text"] for i in section["fact_ids"])
        if not section["fact_ids"]:
            parts.append("Недостаточно данных для вывода по этому разделу.")
        parts.append("")
    if user.get("research"):
        from .research import followup_text

        parts += [followup_text(user), ""]
    return "\n".join(parts)


def write_reports(payload, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_text = "\n\n" + "\n\n".join(
        summary_text(u, payload["period"]) for u in payload["users"]
    )
    (output / "all_user_summaries.txt").write_text(all_text)
    request = (
        Path(__file__).resolve().parents[2] / "docs/user_view_missing_data_request.txt"
    ).read_text()
    (output / "missing_data_request.txt").write_text(request)
    a = payload["audit"]
    ed = payload.get("editor_audit", {})
    selected = ed.get("validated_users", 0)
    llm_text = (
        "LLM-отбор не применялся; краткий обзор сформирован правилами."
        if not selected
        else (
            f'LLM-отбор применён для {selected}/{a["users"]} профилей; остальные обзоры сформированы правилами.'
            if selected < a["users"]
            else "Краткий обзор каждого профиля отобран LLM-редакторами из допустимых фактов."
        )
    )
    rows = [
        "ЗАДАНИЕ 16.10 — ОТЧЁТ ПО ИМЕЮЩИМСЯ ДАННЫМ",
        "",
        "1. User-view: реализован. Профили, параметры, модели/сценарии, история, поиски, аудит и основания каждого вывода.",
        f"Профилей {a['users']}, с действиями {a['active_users']}; user–item строк {a['user_item_rows']}, исходных событий {a['raw_events']}, отображаемых {a['display_events']}.",
        "",
        "2. Параметры: концентрация по категориальным полям, отдельные явные фильтры, узкие числовые диапазоны с заполненностью и источниками.",
        f"Явные фильтры есть у {a['explicit_filter_users']} пользователей; регион бренда в запросах — у {a['brand_region_filter_users']}; узкие числовые полосы — у {a['narrow_range_users']}.",
        "Страны производства автомобилей в архиве нет; регион бренда запроса не заменяет это поле.",
        "",
        "3. Ключевые и неключевые параметры: реализовано описательное ранжирование сигналов. Явные фильтры, сравнение с остальной выборкой, концентрация и вариативность показаны раздельно.",
        f"Отличительные сигналы относительно фона найдены у {a['distinct_signal_users']} пользователей. Важность как причина выбора и неважность по разнообразию не утверждаются.",
        "",
        "4. Модель + цена и общие свойства: рассчитаны группы brand/model, число объявлений, медиана/диапазон цены и повторяющиеся свойства. Одноимённые модели разных марок не объединяются.",
        "",
        "5. Основной/дополнительные кластеры и случайные просмотры: реализованы воспроизводимые описательные группы, доли, равные крупнейшие группы, небольшие и единичные сочетания, пропуски полей.",
        f"Группы из трёх объявлений доступны у {a['cluster_users']} пользователей. Случайность редкого просмотра по одному поведению не доказуема.",
        "",
        "6. Новые модели: кандидаты из историй остальных пользователей, исключение ранее встречавшихся моделей и item ID, проверка целого сценария и последней сопоставленной конфигурации.",
        f"Новые модели есть у {a['recommendations_users']} пользователей; отсутствующие ограничения явно помечены. Польза и наличие в продаже требуют внешней проверки.",
        "",
        "7. Коммуникации: не выполнена содержательная кластеризация, потому что текстов и транскрипций нет. Никакие темы не придуманы; факты контактов сохранены.",
        "",
        "8. Саммари: полные текстовые разделы для всех профилей, в том числе пустых. Числа рассчитаны кодом. "
        + llm_text,
        f"Проверенный LLM-отбор: {selected}/{a['users']}."
        + (
            " Редактор — агент текущей сессии; это не Qwen-run, не обучение и не ranking benchmark."
            if selected
            else " Обучение и ranking benchmark не выполнялись."
        ),
        "",
        "Проверка: manifest источников, TDD-регрессии, независимое code review; подробности и команды — docs/avito_user_view.md.",
        "Список недостающих данных готов отдельно: missing_data_request.txt.",
        "Все построчные данные и отчёты приватные; интерфейс доступен только локально.",
    ]
    (output / "assignment_report.txt").write_text("\n".join(rows) + "\n")
    from .research import write_research_reports

    write_research_reports(payload, output)
