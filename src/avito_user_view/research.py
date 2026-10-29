"""Exploratory follow-up of 22 October; canonical editorial facts stay frozen.

The median interval inverts the binomial sign test. Its coverage statement
requires an IID sampling model, which is not established by behavioral logs.
No interval is a budget, purchase prediction or uncertainty about the known
finite export. Overview diagnostics are not measurements of usefulness.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import io
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from datetime import datetime

from .analysis import DOMAINS, LABELS, RULE_VERSION, _num

VERSION = "user-summary-v3"
CI_FIELDS = ("price", "mileage_km", "year", "power_hp")
CI_NOTE = (
    "Номинальный 95% ДИ медианы при модели независимых одинаково распределённых "
    "наблюдений (IID). Для истории просмотров эта модель не проверена: "
    "поведение зависит от выдачи, времени и связанных объявлений. "
    "Интервал исследовательский, не оценка бюджета или будущей покупки. "
    "95% относится к каждому интервалу отдельно, не ко всем сразу."
)


def median_confidence_interval(values, confidence=0.95):
    """Central conservative sign-test CI; ranks are one-based and inclusive.

    Select the largest k with 2*P[Bin(n, .5) <= k-1] <= alpha.
    Integer arithmetic prevents underflow even for very long histories.
    Coverage is a theoretical IID bound, never an observed success rate.
    """
    q = _num(confidence)
    if isinstance(confidence, bool) or q is None or not 0 < q < 1:
        raise ValueError("Confidence must be finite and strictly between zero and one")
    alpha = 1 - Fraction(str(confidence))
    ns = sorted(v for x in values if (v := _num(x)) is not None)
    n = len(ns)
    estimate = (
        None
        if not n
        else (ns[n // 2] if n % 2 else ns[n // 2 - 1] / 2 + ns[n // 2] / 2)
    )
    row = dict(
        status="insufficient" if n else "not_available",
        n=n,
        median=estimate,
        low=None,
        high=None,
        rank_low=None,
        rank_high=None,
        nominal_coverage=q,
        iid_coverage_bound=None,
        method="exact_binomial_order_statistics",
        ties=len(set(ns)) < n,
    )
    total = 1 << n
    cumulative = 0
    coefficient = 1
    best = None
    for j in range((n + 1) // 2):
        cumulative += coefficient
        if 2 * cumulative * alpha.denominator <= alpha.numerator * total:
            best = j + 1, cumulative
        else:
            break
        coefficient = coefficient * (n - j) // (j + 1)
    if best is not None:
        k, tail = best
        row.update(
            status="available",
            low=ns[k - 1],
            high=ns[n - k],
            rank_low=k,
            rank_high=n - k + 1,
            iid_coverage_bound=1 - 2 * tail / total,
        )
    return row


def numeric_intervals(user):
    """Intervals for unique analyzed listings; a conflicting duplicate is missing."""
    groups = defaultdict(list)
    for c in user["items"]:
        if c.get("analyzed"):
            groups[str(c["item"])].append(c)
    total = len(groups)
    rows = []
    for field in CI_FIELDS:
        lo, hi = DOMAINS[field]
        known = {}
        conflicting = 0
        for item, cs in sorted(groups.items()):
            if field == "price" and any(c.get("price_flag") for c in cs):
                continue
            values = {_num(c.get(field)) for c in cs}
            values = {
                v
                for v in values
                if v is not None
                and lo <= v <= hi
                and (field != "year" or v.is_integer())
            }
            if len(values) > 1:
                conflicting += 1
            elif values:
                known[item] = next(iter(values))
        row = median_confidence_interval(known.values())
        fraction = len(known) / total if total else 0
        status = (
            "not_available"
            if not known
            else (
                "insufficient"
                if len(known) < 10
                else "low_coverage" if fraction < 0.6 else row["status"]
            )
        )
        row.update(
            status=status,
            field=field,
            label=LABELS[field],
            total=total,
            known_fraction=fraction,
            missing=total - len(known),
            conflicting_items=conflicting,
            evidence=list(known),
            target="median_of_hypothetical_similarly_observed_listings",
            sampling_model_verified=False,
            note=CI_NOTE,
        )
        if status != "available":
            row.update(
                low=None,
                high=None,
                rank_low=None,
                rank_high=None,
                iid_coverage_bound=None,
            )
        rows.append(row)
    return rows


def top_interest(user):
    """Historical evidence priority, not causal/statistical feature significance.

    All supported explicit fields compete on identical state-support units.
    Without explicit evidence only already qualified categorical contrasts
    compete. Numeric descriptive windows are not optimized into significance.
    """
    a = user["analysis"]
    search = a["search"]
    explicit = []
    for field, x in sorted(search["fields"].items()):
        real_states = search.get("states")
        relevant = [
            s
            for s in (real_states or [])
            if s["categorical"].get(field) or field in s["ranges"]
        ]
        options = [
            dict(
                **s["ranges"][field],
                search_refs=list(s["search_refs"]),
                state_id=s["id"],
            )
            for s in relevant
            if field in s["ranges"]
        ]

        def text_number(v):
            if v is None:
                return ""
            return str(int(v)) if float(v).is_integer() else str(v)

        values = (
            sorted({v for s in relevant for v in s["categorical"].get(field, [])})
            if real_states is not None
            else list(x["values"])
        )
        values.extend(
            ("от " + text_number(o["min"]) if o["min"] is not None else "")
            + (" " if o["min"] is not None and o["max"] is not None else "")
            + ("до " + text_number(o["max"]) if o["max"] is not None else "")
            for o in options
        )
        values = sorted(set(values))
        states = len(relevant) if real_states is not None else x["states"]
        total_states = (
            len(real_states) if real_states is not None else x["total_states"]
        )
        days = (
            len({t[:10] for s in relevant for t in s["times"]})
            if real_states is not None
            else x["days"]
        )
        refs = (
            sorted({r for s in relevant for r in s["search_refs"]})
            if real_states is not None
            else list(x["search_refs"])
        )
        if field not in LABELS or not values or states <= 0 or total_states <= 0:
            continue
        explicit.append(
            dict(
                field=field,
                label=LABELS[field],
                source="historical_explicit",
                values=values,
                range_options=options,
                states=states,
                total_states=total_states,
                days=days,
                search_refs=refs,
                evidence=[],
                score=states / total_states,
                score_definition="fraction_of_unique_supported_filter_states_with_this_field",
            )
        )
    contrasts = [
        dict(
            field=x["field"],
            label=x["label"],
            source="history_contrast",
            values=list(x["values"]),
            evidence=list(x["evidence"]),
            search_refs=[],
            count=x["count"],
            known=x["known"],
            total=x["total"],
            share=x["share"],
            reference_share=x["reference_share"],
            reference_known=x["reference_known"],
            score=x["score"],
            score_definition="positive_wilson_lower_minus_background_times_coverage",
        )
        for x in a["importance"]
        if x["status"] == "distinctive" and x["score"] > 0
    ]
    candidates = sorted(explicit or contrasts, key=lambda x: (-x["score"], x["field"]))
    leaders = (
        [
            x
            for x in candidates
            if math.isclose(
                x["score"], candidates[0]["score"], rel_tol=0, abs_tol=1e-12
            )
        ]
        if candidates
        else []
    )
    latest = search.get("latest")
    context = (
        "newer_ambiguous_search"
        if search.get("newer_ambiguous")
        else (
            "latest_needs_review"
            if latest and latest.get("warnings")
            else (
                "latest_unmodeled_filters"
                if latest
                and latest.get("filters")
                and not latest.get("categorical")
                and not latest.get("ranges")
                else (
                    "latest_without_constraints"
                    if latest is not None
                    and not latest.get("categorical")
                    and not latest.get("ranges")
                    else "latest_available" if latest else "latest_not_available"
                )
            )
        )
    )
    return dict(
        status=(
            "tied" if len(leaders) > 1 else "available" if leaders else "not_available"
        ),
        leaders=leaders,
        candidates=candidates,
        unique_top_field=leaders[0]["field"] if len(leaders) == 1 else None,
        scope="historical_period",
        is_current_constraint=False,
        context_status=context,
        method="explicit_state_support_then_qualified_categorical_contrast",
        note="Эвристический приоритет наблюдаемых сигналов, не причинная важность и не статистическая значимость. Исторические фильтры не являются одним текущим запросом.",
    )


def explicit_stability(user):
    """Delete each observed search day; measure attribute-set, not value stability."""
    original = top_interest(user)
    base = dict(
        status="not_applicable",
        days=0,
        leaders=[],
        same_top_set_removals=0,
        original_unique=original["status"] == "available",
        method="leave_one_observed_search_day_out",
        removals=[],
        note="Чувствительность набора ведущих атрибутов к удалению дня поисков, не вероятность, не устойчивость значений предпочтений и не предсказательная точность.",
    )
    if (
        not original["leaders"]
        or original["leaders"][0]["source"] != "historical_explicit"
    ):
        return base
    search = user["analysis"]["search"]

    def day(t):
        try:
            return datetime.fromisoformat(t).date().isoformat()
        except (ValueError, TypeError):
            return None

    times = [t for s in search["states"] for t in s["times"]]
    if any(day(t) is None for t in times):
        return dict(base, status="invalid_timestamps")
    days = sorted({day(t) for t in times})
    base["days"] = len(days)
    if len(days) < 3:
        return dict(base, status="insufficient_days")
    original_fields = {x["field"] for x in original["leaders"]}
    for removed_day in days:
        states = []
        for s in search["states"]:
            keep = [
                (t, r)
                for t, r in zip(s["times"], s["search_refs"])
                if day(t) != removed_day
            ]
            if keep:
                states.append(
                    dict(
                        s, times=[t for t, r in keep], search_refs=[r for t, r in keep]
                    )
                )
        recomputed = top_interest(
            dict(analysis=dict(importance=[], search=dict(search, states=states)))
        )
        leaders = [
            x for x in recomputed["leaders"] if x["source"] == "historical_explicit"
        ]
        fields = {x["field"] for x in leaders}
        base["same_top_set_removals"] += fields == original_fields
        base["removals"].append(
            dict(
                removed_day=removed_day,
                remaining_states=len(states),
                leaders=[{k: x[k] for k in ("field", "values")} for x in leaders],
            )
        )
    for x in original["leaders"]:
        selected = [r["leaders"] for r in base["removals"]]
        base["leaders"].append(
            dict(
                field=x["field"],
                among_leaders_removals=sum(
                    any(y["field"] == x["field"] for y in rs) for rs in selected
                ),
                unique_leader_removals=sum(
                    len(rs) == 1 and rs[0]["field"] == x["field"] for rs in selected
                ),
            )
        )
    return dict(base, status="available")


def _diverse_ids(facts, budget):
    ordered = sorted(facts, key=lambda f: (-f["priority"], f["id"]))
    selected, sections = [], set()
    for f in ordered:
        if f["section"] not in sections and len(selected) < budget:
            selected.append(f["id"])
            sections.add(f["section"])
    selected.extend(f["id"] for f in ordered if f["id"] not in selected)
    return selected[:budget]


def compare_overviews(user):
    if user["summary"]["status"] != "llm_fact_selection":
        return dict(
            status="not_available",
            reason="No validated LLM editorial cache",
            usefulness_evaluated=False,
        )
    from .narrative import validate_selection

    if user["summary"]["fact_hash"] != user["fact_hash"]:
        raise ValueError("Stale editorial summary hash")
    validate_selection(
        dict(
            user_index=1,
            fact_hash=user["summary"]["fact_hash"],
            overview_fact_ids=user["summary"]["overview"],
        ),
        user,
    )
    original = list(user["summary"]["overview"])
    by_id = {f["id"]: f for f in user["facts"]}
    shared = [i for i in ("communications", "country", "temporal") if i in by_id]
    llm = [i for i in original if i not in shared]
    budget = len(llm)
    if not budget:
        return dict(
            status="context_only",
            budget=0,
            original_llm_count=len(original),
            original_llm_ids=original,
            excluded_context_ids=original,
            shared_context_ids=shared,
            usefulness_evaluated=False,
        )
    candidates = [f for f in user["facts"] if f["id"] not in shared]
    ordered = sorted(candidates, key=lambda f: (-f["priority"], f["id"]))
    rules = [f["id"] for f in ordered[:budget]]
    diverse = _diverse_ids(candidates, budget)
    sections = lambda ids: sorted({by_id[i]["section"] for i in ids})
    jaccard = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
    chars = lambda ids: len("\n".join(by_id[i]["text"] for i in ids))
    return dict(
        status="available",
        budget=budget,
        fact_hash=user["fact_hash"],
        original_llm_count=len(original),
        original_llm_ids=original,
        excluded_context_ids=[i for i in original if i in shared],
        llm_ids=llm,
        rule_ids=rules,
        diverse_rule_ids=diverse,
        shared_context_ids=shared,
        jaccard=jaccard(llm, rules),
        diverse_jaccard=jaccard(llm, diverse),
        llm_sections=sections(llm),
        rule_sections=sections(rules),
        diverse_sections=sections(diverse),
        llm_characters=chars(llm),
        rule_characters=chars(rules),
        diverse_characters=chars(diverse),
        same_selection=set(llm) == set(rules),
        same_diverse_selection=set(llm) == set(diverse),
        usefulness_evaluated=False,
        method="Frozen validated LLM selections; priority and section-diverse heuristics; same facts and per-profile budget",
        note="Покрытие разделов и совпадение фактов — диагностические показатели, не доказательство полезности LLM. Нужна слепая экспертная оценка.",
    )


def blind_pair(user, user_index, baseline="priority"):
    if baseline not in ("priority", "diverse"):
        raise ValueError("Unknown baseline")
    row = compare_overviews(user)
    if row["status"] != "available":
        raise ValueError("No LLM selection to compare")
    by_id = {f["id"]: f["text"] for f in user["facts"]}
    swap = (
        int(
            hashlib.sha256(
                f"{user_index}:{baseline}:{user['fact_hash']}".encode()
            ).hexdigest(),
            16,
        )
        % 2
    )
    sides = {"A": "rules" if swap else "llm", "B": "llm" if swap else "rules"}
    selected = {
        "llm": row["llm_ids"],
        "rules": row["rule_ids"] if baseline == "priority" else row["diverse_rule_ids"],
    }
    pair = dict(
        comparison_id=f"{user_index:03d}-{'P' if baseline == 'priority' else 'D'}",
        user_index=user_index,
        A=[by_id[i] for i in selected[sides["A"]]],
        B=[by_id[i] for i in selected[sides["B"]]],
        shared_context=[by_id[i] for i in row["shared_context_ids"]],
        reference_facts=[
            {k: f[k] for k in ("id", "section", "text")} for f in user["facts"]
        ],
    )
    return pair, dict(
        comparison_id=pair["comparison_id"], baseline=baseline, sides=sides
    )


def enrich_research(payload):
    for user in payload["users"]:
        user["research"] = dict(
            top_interest=top_interest(user),
            numeric_ci=numeric_intervals(user),
            explicit_stability=explicit_stability(user),
            cluster_ci=[],
            cluster_ci_status="not_computed_adaptive_groups",
            overview_comparison=compare_overviews(user),
        )
    users = payload["users"]
    comparisons = [
        u["research"]["overview_comparison"]
        for u in users
        if u["research"]["overview_comparison"]["status"] == "available"
    ]
    payload["research_audit"] = dict(
        version=VERSION,
        task_date="2026-10-22",
        users=len(users),
        fact_rule_version=RULE_VERSION,
        top_interest_statuses=dict(
            Counter(u["research"]["top_interest"]["status"] for u in users)
        ),
        ci_available_users={
            f: sum(
                next(r for r in u["research"]["numeric_ci"] if r["field"] == f)[
                    "status"
                ]
                == "available"
                for u in users
            )
            for f in CI_FIELDS
        },
        ci_total_available=sum(
            r["status"] == "available"
            for u in users
            for r in u["research"]["numeric_ci"]
        ),
        compared_users=len(comparisons),
        equal_budget_verified=all(
            len(r["llm_ids"]) == len(r["rule_ids"]) == len(r["diverse_rule_ids"])
            for r in comparisons
        ),
        same_priority_selection_users=sum(r["same_selection"] for r in comparisons),
        same_diverse_selection_users=sum(
            r["same_diverse_selection"] for r in comparisons
        ),
        mean_jaccard_priority=(
            sum(r["jaccard"] for r in comparisons) / len(comparisons)
            if comparisons
            else None
        ),
        mean_jaccard_diverse=(
            sum(r["diverse_jaccard"] for r in comparisons) / len(comparisons)
            if comparisons
            else None
        ),
        total_fact_budget=sum(r["budget"] for r in comparisons),
        total_original_llm_fact_budget=sum(
            u["research"]["overview_comparison"].get("original_llm_count", 0)
            for u in users
        ),
        context_only_users=sum(
            u["research"]["overview_comparison"]["status"] == "context_only"
            for u in users
        ),
        compared_active_users=sum(
            bool(u["counts"]["analyzed_items"])
            and u["research"]["overview_comparison"]["status"] == "available"
            for u in users
        ),
        explicit_stability_users=sum(
            u["research"]["explicit_stability"]["status"] == "available" for u in users
        ),
        unchanged_top_set_on_all_day_removals=sum(
            s["status"] == "available" and s["same_top_set_removals"] == s["days"]
            for u in users
            for s in [u["research"]["explicit_stability"]]
        ),
        sections_total={
            key: sum(len(r[key]) for r in comparisons)
            for key in ("llm_sections", "rule_sections", "diverse_sections")
        },
        characters_total={
            key: sum(r[key] for r in comparisons)
            for key in ("llm_characters", "rule_characters", "diverse_characters")
        },
        usefulness_evaluated=False,
        ranking_evaluated=False,
        sampling_model_verified=False,
    )
    payload["version"] = VERSION
    payload["method"][
        "median_ci"
    ] = "Exact binomial order-statistic CI; exploratory IID model only; n>=10 and known coverage>=60%; per-interval 95%"
    return payload


def followup_text(user):
    r = user.get("research")
    if not r:
        return ""
    rows = [
        "ДОПОЛНЕНИЕ К ЗАДАНИЮ 22.10",
        "",
        "Исторический интерес по значимым атрибутам",
    ]
    interest = r["top_interest"]
    if not interest["leaders"]:
        rows.append(
            "Недостаточно оснований выделить ведущий атрибут; общий для выборки признак не назначается главным автоматически."
        )
    elif interest["status"] == "tied":
        rows.append(
            "Несколько атрибутов имеют равный эвристический приоритет; единственного лидера нет."
        )
    for x in interest["leaders"]:
        text = x["label"] + ": " + " / ".join(x["values"])
        if x["source"] == "historical_explicit":
            text += f". Явные фильтры в {x['states']}/{x['total_states']} конфигураций, {x['days']} дней."
        else:
            text += f". В истории {x['count']}/{x['known']}; в остальной выборке {100*x['reference_share']:.1f}%."
        rows.append(text)
    rows += [interest["note"], "", "95% ДИ медианы · исследовательская IID-модель"]
    for x in r["numeric_ci"]:
        status = {
            "insufficient": "мало объявлений (нужно не менее 10)",
            "not_available": "нет значений",
            "low_coverage": "заполнено менее 60% истории",
        }
        text = (
            f"{x['label']}: медиана {x['median']:g}, интервал [{x['low']:g}; {x['high']:g}], n={x['n']}/{x['total']}."
            if x["status"] == "available"
            else f"{x['label']}: {status[x['status']]}; n={x['n']}/{x['total']}."
        )
        rows.append(text)
    rows += [
        CI_NOTE,
        "Совпадение границ при повторяющихся значениях не означает полной уверенности.",
        "Для адаптивно выделенных по цене групп ДИ не вычисляется: стандартное покрытие после такого отбора не гарантируется.",
    ]
    return "\n".join(rows)


def write_research_reports(payload, output):
    output = Path(output)
    pairs, keys = [], []
    for i, u in enumerate(payload["users"], 1):
        if u["research"]["overview_comparison"]["status"] != "available":
            continue
        for baseline in ("priority", "diverse"):
            p, k = blind_pair(u, i, baseline)
            pairs.append(p)
            keys.append(k)
    for name, data in [
        ("research_audit.json", payload["research_audit"]),
        ("blind_summary_pairs.json", pairs),
        ("blind_summary_key.json", keys),
    ]:
        (output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    with (output / "summary_annotation_template.csv").open("w", newline="") as f:
        fields = [
            "comparison_id",
            "user_index",
            "preference_A_B_tie",
            "accuracy_A_1_5",
            "accuracy_B_1_5",
            "relevance_A_1_5",
            "relevance_B_1_5",
            "actionability_A_1_5",
            "actionability_B_1_5",
            "clarity_A_1_5",
            "clarity_B_1_5",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {"comparison_id": p["comparison_id"], "user_index": p["user_index"]}
            for p in pairs
        )
    instructions = (
        "СЛЕПОЕ СРАВНЕНИЕ САММАРИ\n\n"
        "Откройте index.html локально. Интернет, модель и сервер не нужны. "
        "Для каждой пары доступны A, B, общие ограничения и полный каталог вычисленных фактов. "
        "Оценивайте точность относительно каталога, релевантность, полезность для следующего уточняющего вопроса и ясность. "
        "Каталог не является независимым аудитом первичных событий; покупка и реальная польза рекомендаций неизвестны.\n\n"
        "Архивы priority и diverse передавайте разным экспертам: повторяющийся текст между двумя парами одного пользователя может раскрыть вариант LLM. "
        "Для каждой пары желательно получить две независимые оценки; спорные случаи разобрать отдельно. "
        "Не передавайте экспертам blind_summary_key.json и основной интерфейс, показывающий названия методов. "
        "Не оценивайте по формальному приоритету факта. Одинаковые наборы фактов не являются содержательной победой одного метода.\n\n"
        "Ответы в HTML хранятся в памяти вкладки. Скачайте CSV перед обновлением или закрытием. "
        "1 — низкая оценка, 5 — высокая, пусто — не оценено. Не заменяйте отсутствие оценки нулём. "
        "Одинаковый бюджет фактов не гарантирует одинаковую длину. Статистическую неопределённость будущих парных оценок считать по пользователям, "
        "а не по фактам; текущая выгрузка оценок не содержит.\n"
    )
    for baseline, suffix in (("priority", "-P"), ("diverse", "-D")):
        selected = [p for p in pairs if p["comparison_id"].endswith(suffix)]
        csv_text = io.StringIO(newline="")
        writer = csv.DictWriter(csv_text, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {"comparison_id": p["comparison_id"], "user_index": p["user_index"]}
            for p in selected
        )
        with zipfile.ZipFile(
            output / f"annotation_{baseline}.zip", "w", zipfile.ZIP_DEFLATED
        ) as z:
            z.writestr("index.html", render_blind_html(selected))
            z.writestr("pairs.json", json.dumps(selected, ensure_ascii=False, indent=2))
            z.writestr("annotation_template.csv", csv_text.getvalue())
            z.writestr("README.txt", instructions)
    audit = payload["research_audit"]
    text = [
        "ЗАДАНИЕ 22.10 — РАБОТА НА ИМЕЮЩИХСЯ ДАННЫХ",
        "",
        "Исторический интерес: явные фильтры всех поддержанных типов; при их отсутствии — уже квалифицированные категориальные отличия от фона. Равные лидеры сохранены; общий или слабо заполненный атрибут не назначается главным автоматически.",
        "Статусы ведущих интересов: "
        + json.dumps(audit["top_interest_statuses"], ensure_ascii=False),
        "ДИ медианы: " + json.dumps(audit["ci_available_users"], ensure_ascii=False),
        CI_NOTE,
        f"Сравнение содержательной части саммари: {audit['compared_users']} профилей с валидированным замороженным LLM-отбором; matched budget {audit['total_fact_budget']} фактов у каждого из трёх методов. Исходный LLM-отбор содержал {audit['total_original_llm_fact_budget']} фактов; общие ограничения вынесены из содержательной части и показаны всем вариантам одинаково.",
        "Эвристики: top-k по заданному приоритету и top-k с разнообразием разделов. Число символов отчитается отдельно; одинаковое k не означает одинаковые tokens.",
        "Полезность LLM не измерена. Совпадение наборов и покрытие разделов — диагностические показатели, не метрика качества рекомендаций.",
        "Новые выдачи/клики, коммуникации и фотографии не получены для этого эксперимента; новый NDCG не рассчитывался.",
        "",
    ]
    (output / "followup_report.txt").write_text(
        "\n".join(text)
        + "\n\n"
        + "\n\n".join(u["label"] + "\n" + followup_text(u) for u in payload["users"])
    )


def render_blind_html(pairs):
    template = (
        Path(__file__).resolve().parents[2] / "apps/avito_user_view/annotation.html"
    ).read_text()
    safe = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for a, b in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        safe = safe.replace(a, b)
    return template.replace("__BLIND_PAIRS__", safe)
