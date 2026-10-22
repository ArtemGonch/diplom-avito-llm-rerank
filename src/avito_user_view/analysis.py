"""Descriptive user analysis for every point of the 16 October assignment.

Explicit search configurations, observed concentrations and cohort differences
remain separate. These are hypotheses about a sampled history, not causal
feature importance or a ranking-quality experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

RULE_VERSION = "user-summary-v2"
CAT_PARAMS = {
    "110000": "brand",
    "110001": "model",
    "110005": "generation",
    "110008": "transmission",
    "110006": "engine_type",
    "110007": "drive",
    "152642": "brand_region",
}
RANGE_PARAMS = {
    "164669": "year",
    "2687": "mileage_km",
    "110011": "power_hp",
    "162396": "engine_liters",
}
LABELS = {
    "brand": "Марка",
    "model": "Модель",
    "generation": "Поколение",
    "transmission": "КПП",
    "body_type": "Кузов",
    "engine_type": "Двигатель",
    "drive": "Привод",
    "brand_region": "Регион бренда",
    "price": "Цена",
    "year": "Год",
    "mileage_km": "Пробег",
    "power_hp": "Мощность",
    "engine_liters": "Объём двигателя",
    "doors": "Двери",
    "owners": "Владельцы",
}
CAT_FIELDS = ("brand", "model", "body_type", "transmission", "engine_type", "drive")
DOMAINS = {
    "price": (10_000, 1e10),
    "year": (1886, 2027),
    "mileage_km": (0, 5_000_000),
    "power_hp": (10, 3000),
    "engine_liters": (0, 20),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _num(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        n = float(str(x).replace(" ", "").replace("\u00a0", ""))
        return n if math.isfinite(n) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _values(value):
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return sorted(
        {
            str(v).strip()
            for v in value
            if not isinstance(v, bool)
            and isinstance(v, (str, int, float))
            and str(v).strip()
        }
    )


def _range_param(value, field):
    if not isinstance(value, dict):
        return None, "Ожидался диапазон"
    bounds = {}
    lo, hi = DOMAINS[field]
    for source, target in [("from", "min"), ("to", "max")]:
        x = value.get(source)
        # In this export zero is an unset upper bound for positive quantities.
        if x in (None, "") or _num(x) == 0 and (source == "to" or lo > 0):
            bounds[target] = None
            continue
        n = _num(x)
        if n is None or not lo <= n <= hi:
            return None, "Некорректная граница " + LABELS[field]
        if field == "year" and not n.is_integer():
            return None, "Год должен быть целым"
        bounds[target] = int(n) if n.is_integer() else n
    if (
        bounds["min"] is not None
        and bounds["max"] is not None
        and bounds["min"] > bounds["max"]
    ):
        return None, "Нижняя граница больше верхней: " + LABELS[field]
    return (bounds if any(v is not None for v in bounds.values()) else None), None


def analyze_searches(raw_searches, vocabulary):
    states = {}
    ambiguous = []
    latest = None
    for i, s in enumerate(raw_searches):
        ref = f"search-{i+1}"
        t = str(s.get("t", ""))
        if s.get("payloadStatus") != "matched":
            ambiguous.append(t)
            continue
        params = s.get("params") if isinstance(s.get("params"), dict) else {}
        categorical = {
            field: _values(params[code])
            for code, field in CAT_PARAMS.items()
            if _values(params.get(code))
        }
        ranges = {}
        warnings = []
        body_options = []
        for code, field in RANGE_PARAMS.items():
            if code not in params:
                continue
            r, error = _range_param(params[code], field)
            if error:
                warnings.append(error)
            if r:
                ranges[field] = r
        for raw in _values(params.get("147411")):
            candidates = sorted(
                vocabulary.get("body_type", []), key=lambda v: (-len(v), v)
            )
            body = next(
                (b for b in candidates if raw.casefold().startswith(b.casefold())), None
            )
            doors = re.search(r"([2-5])\s*[- ]?двер", raw, re.I)
            body_options.append(
                dict(body_type=body, doors=int(doors[1]) if doors else None, label=raw)
            )
            if body is None:
                warnings.append("Не сопоставлен кузов: " + raw)
        if body_options:
            categorical["body_type"] = sorted(
                {x["body_type"] for x in body_options if x["body_type"]}
            )
        if _values(params.get("1167")):
            categorical["owners"] = _values(params["1167"])
        price = {}
        labels = sorted(
            {str(f).strip() for f in s.get("filters", []) if isinstance(f, str)}
        )
        for text in labels:
            m = re.fullmatch(r"Цена (от|до) ([0-9\s\u00a0]+) ₽", text)
            if m:
                price["from" if m[1] == "от" else "to"] = m[2]
        if price:
            r, error = _range_param(price, "price")
            if error:
                warnings.append(error)
            if r:
                ranges["price"] = r
        meaningful = bool(categorical or ranges or warnings or body_options)
        signature = digest(
            dict(
                categorical=categorical,
                ranges=ranges,
                body_options=body_options,
                labels=labels,
                warnings=warnings,
            )
        )
        if signature not in states:
            states[signature] = dict(
                id="state-" + signature[:12],
                categorical=categorical,
                ranges=ranges,
                body_options=body_options,
                warnings=warnings,
                filters=labels,
                search_refs=[],
                times=[],
                meaningful=meaningful,
            )
        state = states[signature]
        state["search_refs"].append(ref)
        state["times"].append(t)
        if latest is None or (t, i) > latest[0]:
            latest = ((t, i), signature)
    ordered = sorted(states.values(), key=lambda x: (max(x["times"]), x["id"]))
    for s in ordered:
        s["events"] = len(s["search_refs"])
        s["days"] = len({t[:10] for t in s["times"]})
        s["last_t"] = max(s["times"])
        s["first_t"] = min(s["times"])
    fields = {}
    for field in list(LABELS):
        relevant = [
            s for s in ordered if field in s["categorical"] or field in s["ranges"]
        ]
        if not relevant:
            continue
        fields[field] = dict(
            field=field,
            label=LABELS[field],
            states=len(relevant),
            total_states=len(ordered),
            events=sum(s["events"] for s in relevant),
            days=len({t[:10] for s in relevant for t in s["times"]}),
            values=sorted(
                {v for s in relevant for v in s["categorical"].get(field, [])}
            ),
            state_ids=[s["id"] for s in relevant],
            search_refs=sorted({r for s in relevant for r in s["search_refs"]}),
        )
    last = states[latest[1]] if latest else None
    return dict(
        states=ordered,
        unique_states=len(ordered),
        matched_events=sum(s["events"] for s in ordered),
        ambiguous_events=len(ambiguous),
        fields=fields,
        latest=last,
        newer_ambiguous=bool(last and any(t > last["last_t"] for t in ambiguous)),
        status="available" if ordered else "not_available",
        unit="Unique readable filter states; pagination and repeated identical states do not add support",
    )


def _wilson(k, n):
    if not n:
        return 0
    z = 1.96
    p = k / n
    return (
        p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    ) / (1 + z * z / n)


def _importance(user, reference, search):
    cards = [c for c in user["items"] if c["analyzed"]]
    result = []
    for field in CAT_FIELDS:
        known = [c for c in cards if c.get(field)]
        ct = Counter(c[field] for c in known)
        top = sorted(ct, key=lambda v: (-ct[v], v))
        pref = next((p for p in user["preferences"] if p["field"] == field), None)
        values = pref["values"] if pref else top[:1]
        k = sum(ct[v] for v in values)
        n = len(known)
        share = k / n if n else None
        ref = [c for c in reference if c.get(field)]
        rk = sum(c[field] in values for c in ref)
        bg = rk / len(ref) if ref else None
        eligible = n >= 5 and n >= 0.6 * len(cards)
        status = "insufficient"
        if eligible:
            if (
                pref
                and len(ref) >= 20
                and bg is not None
                and share - bg >= 0.2
                and _wilson(k, n) > bg
            ):
                status = "distinctive"
            elif pref and len(ref) >= 20 and bg is not None and abs(share - bg) < 0.1:
                status = "common_in_sample"
            elif pref:
                status = "concentrated"
            elif len(ct) >= 3 and max(ct.values()) / n < 0.5:
                status = "variable"
            else:
                status = "inconclusive"
        explicit = search["fields"].get(field)
        if explicit:
            status = "explicit"
        score = (
            1 + explicit["states"] / max(1, explicit["total_states"])
            if explicit
            else max(0, _wilson(k, n) - (bg if bg is not None else 0))
            * (n / max(1, len(cards)))
        )
        result.append(
            dict(
                field=field,
                label=LABELS[field],
                status=status,
                values=values,
                count=k,
                known=n,
                total=len(cards),
                share=share,
                reference_count=rk,
                reference_known=len(ref),
                reference_share=bg,
                delta=share - bg if share is not None and bg is not None else None,
                wilson_lower=_wilson(k, n),
                score=score,
                explicit=explicit,
                evidence=[c["item"] for c in known if c[field] in values],
            )
        )
    return sorted(result, key=lambda r: (-r["score"], r["field"]))


def _quantile(ns, q):
    p = (len(ns) - 1) * q
    lo = int(p)
    hi = math.ceil(p)
    return ns[lo] + (ns[hi] - ns[lo]) * (p - lo)


def _numeric(cards, field):
    lo, hi = DOMAINS[field]
    eligible = [
        c
        for c in cards
        if _num(c.get(field)) is not None
        and lo <= c[field] <= hi
        and not (field == "price" and c.get("price_flag"))
    ]
    if not eligible:
        return dict(
            field=field,
            label=LABELS[field],
            count=0,
            total=len(cards),
            narrow=False,
            status="not_available",
            evidence=[],
        )
    ordered = sorted(eligible, key=lambda c: (c[field], c["item"]))
    ns = [c[field] for c in ordered]
    k = math.ceil(0.8 * len(ns))
    fullmed = median(ns)
    start = min(
        range(len(ns) - k + 1),
        key=lambda i: (
            ns[i + k - 1] - ns[i],
            abs((ns[i + k - 1] + ns[i]) / 2 - fullmed),
            i,
        ),
    )
    corelo, corehi = ns[start], ns[start + k - 1]
    core = [c for c in eligible if corelo <= c[field] <= corehi]
    width = corehi - corelo
    rel = width / max(abs(fullmed), 1)
    thresholds = {
        "price": rel <= 0.4,
        "year": width <= 4,
        "mileage_km": width <= 50_000 and rel <= 0.6,
        "power_hp": width <= 40 and rel <= 0.3,
        "engine_liters": width <= 0.5,
    }
    enough = len(eligible) >= 5 and len(eligible) >= 0.6 * len(cards)
    return dict(
        field=field,
        label=LABELS[field],
        count=len(eligible),
        total=len(cards),
        median=fullmed,
        min=ns[0],
        max=ns[-1],
        p10=_quantile(ns, 0.1),
        p90=_quantile(ns, 0.9),
        core_min=corelo,
        core_max=corehi,
        core_count=len(core),
        core_fraction=len(core) / len(eligible),
        relative_width=rel,
        narrow=bool(enough and thresholds[field]),
        status=(
            "narrow"
            if enough and thresholds[field]
            else "broad" if enough else "insufficient"
        ),
        evidence=[c["item"] for c in core],
        source_counts=dict(
            Counter(
                c.get(
                    "priceSource"
                    if field == "price"
                    else (
                        "technicalSource"
                        if field in ("power_hp", "engine_liters")
                        else "attributeSource"
                    )
                )
                or "Не указан"
                for c in eligible
            )
        ),
    )


def _models(cards):
    groups = defaultdict(list)
    for c in cards:
        if c.get("brand") and c.get("model"):
            groups[(c["brand"], c["model"])].append(c)
    rows = []
    for (brand, model), cs in sorted(groups.items(), key=lambda g: (-len(g[1]), g[0])):
        common = []
        for field in ("body_type", "transmission", "engine_type", "drive"):
            known = [c for c in cs if c.get(field)]
            ct = Counter(c[field] for c in known)
            if not ct:
                continue
            v = min(ct, key=lambda v: (-ct[v], v))
            if (
                len(known) >= 3
                and len(known) >= 0.6 * len(cs)
                and ct[v] / len(known) >= 0.7
            ):
                common.append(
                    dict(
                        field=field,
                        label=LABELS[field],
                        value=v,
                        count=ct[v],
                        known=len(known),
                        total=len(cs),
                        evidence=[c["item"] for c in known if c[field] == v],
                    )
                )
        rows.append(
            dict(
                brand=brand,
                model=model,
                count=len(cs),
                total=len(cards),
                evidence=[c["item"] for c in cs],
                price=_numeric(cs, "price"),
                common=common,
            )
        )
    overlap = []
    for field in ("body_type", "transmission", "engine_type", "drive"):
        known = [
            (m, next((c for c in m["common"] if c["field"] == field), None))
            for m in rows
        ]
        known = [(m, c) for m, c in known if c]
        ct = Counter(c["value"] for _, c in known)
        if len(known) < 2:
            continue
        value = min(ct, key=lambda v: (-ct[v], v))
        if ct[value] / len(known) >= 0.7:
            overlap.append(
                dict(
                    field=field,
                    label=LABELS[field],
                    value=value,
                    model_count=ct[value],
                    known_models=len(known),
                    total_models=len(rows),
                    evidence=sorted(
                        {
                            i
                            for m, c in known
                            if c["value"] == value
                            for i in c["evidence"]
                        }
                    ),
                )
            )
    return rows, overlap


def _topology(user):
    cs = user["clusters"]
    cards = [c for c in user["items"] if c["analyzed"]]
    eligible = [
        c
        for c in cards
        if c.get("body_type")
        and c.get("transmission")
        and c.get("price")
        and not c.get("price_flag")
    ]
    if not cs:
        status = "not_available"
        primary = None
    elif len(cs) > 1 and cs[0]["count"] == cs[1]["count"]:
        status = "tied"
        primary = None
    elif cs[0]["count"] >= 5 and cs[0]["count"] / max(1, len(cards)) >= 0.5:
        status = "dominant"
        primary = cs[0]["id"]
    else:
        status = "largest_only"
        primary = cs[0]["id"]
    for c in cs:
        c.update(
            share=c["count"] / max(1, len(cards)),
            eligible_total=len(eligible),
            eligible_share=c["count"] / max(1, len(eligible)),
            role=(
                "primary"
                if c["id"] == primary
                else (
                    "tied_largest"
                    if status == "tied" and c["count"] == cs[0]["count"]
                    else "secondary"
                )
            ),
        )
    assigned = sum(c["count"] for c in cs)
    rare = len(user["outliers"])
    return dict(
        cluster_status=status,
        primary_cluster=primary,
        cluster_eligible=len(eligible),
        cluster_assigned=assigned,
        small_group_items=len(eligible) - assigned - rare,
        cluster_unavailable=len(cards) - len(eligible),
        rare_count=rare,
        rare_status="descriptive_low_confidence",
        cluster_coverage=assigned / max(1, len(cards)),
    )


def _matches(c, scenario):
    unverified = []
    for field, values in scenario["categorical"].items():
        if field in ("brand_region", "owners"):
            unverified.append(field)
            continue
        if field == "body_type" and scenario.get("body_options"):
            continue
        value = c.get(field)
        if value is None:
            unverified.append(field)
        elif str(value) not in {str(v) for v in values}:
            return False, []
    if scenario.get("body_options"):
        options = scenario["body_options"]
        matched = False
        unknown = False
        for option in options:
            if option["body_type"] is None:
                unknown = True
                continue
            if c.get("body_type") != option["body_type"]:
                continue
            if option["doors"] is None or c.get("doors") == option["doors"]:
                matched = True
                break
            if c.get("doors") is None:
                unknown = True
        if not matched and not unknown:
            return False, []
        if not matched:
            unverified.append("body_and_doors")
    for field, bounds in scenario["ranges"].items():
        value = c.get(field)
        if value is None:
            unverified.append(field)
            continue
        if (
            bounds["min"] is not None
            and value < bounds["min"]
            or bounds["max"] is not None
            and value > bounds["max"]
        ):
            return False, []
    return True, sorted(set(unverified))


def _alternatives(user, pool, search, analysis):
    cards = [c for c in user["items"] if c["analyzed"]]
    if len(cards) < 5 or not user["clusters"]:
        return [], "insufficient_history", []
    latest = search["latest"]
    if latest and latest["warnings"]:
        return [], "invalid_latest_constraints", []
    seen = {(c.get("brand"), c.get("model")) for c in user["items"]}
    scenarios = []
    for cluster in user["clusters"][:3]:
        pr = cluster["price"]
        yr = cluster["year"]
        cats = {
            "body_type": [cluster["body_type"]],
            "transmission": [cluster["transmission"]],
        }
        for p in user["preferences"]:
            if (
                p["field"] in ("engine_type", "drive")
                and p["count"] / p["known"] >= 0.8
            ):
                cats[p["field"]] = p["values"]
        ranges = {"price": dict(min=pr["p10"] * 0.8, max=pr["p90"] * 1.2)}
        if yr and yr["count"] >= 3:
            ranges["year"] = dict(min=yr["p10"] - 3, max=yr["p90"] + 3)
        if latest:
            cats.update(latest["categorical"])
            ranges.update(latest["ranges"])
        scenario = dict(
            id=cluster["id"],
            categorical=cats,
            ranges=ranges,
            body_options=latest["body_options"] if latest else [],
            search_state=latest["id"] if latest else None,
            source=(
                "latest_matched_search_plus_observed_group"
                if latest
                else "observed_group"
            ),
            median_price=pr["median"],
        )
        if not any(
            digest(
                {k: v for k, v in scenario.items() if k not in ("id", "median_price")}
            )
            == digest(
                {k: v for k, v in other.items() if k not in ("id", "median_price")}
            )
            for other in scenarios
        ):
            scenarios.append(scenario)
    best = {}
    seen_items = {c["item"] for c in user["items"]}
    for c in pool:
        pair = (c.get("brand"), c.get("model"))
        if (
            not all(pair)
            or pair in seen
            or c["item"] in seen_items
            or not c.get("price")
            or c.get("price_flag")
        ):
            continue
        for scenario in scenarios:
            matched, unverified = _matches(c, scenario)
            if not matched:
                continue
            if search["newer_ambiguous"]:
                unverified.append("latest_query_context")
            score = abs(math.log(c["price"] / scenario["median_price"])) + 0.3 * len(
                unverified
            )
            key = (score, scenario["id"], c["item"], c.get("snapshot", ""))
            if pair not in best or key < best[pair][0]:
                best[pair] = (key, c, scenario, unverified)
    rows = []
    for _, c, scenario, unverified in sorted(best.values(), key=lambda x: x[0])[:6]:
        reasons = [
            c.get("body_type") or "Кузов не указан",
            c.get("transmission") or "КПП не указана",
            "Гипотеза для " + scenario["id"],
        ]
        if latest:
            reasons.append("Проверены доступные поля последней конфигурации поиска")
        rows.append(
            dict(
                brand=c["brand"],
                model=c["model"],
                example_item=c["item"],
                price=c["price"],
                year=c["year"],
                example=dict(c),
                reasons=reasons,
                unverified=sorted(set(unverified)),
                verification="conditional" if unverified else "available_fields_match",
                scenario_id=scenario["id"],
                search_state=scenario["search_state"],
                source="Ретроспективный пул других пользователей; наличие в продаже не проверено",
            )
        )
    return (
        rows,
        (
            ("conditional" if any(r["unverified"] for r in rows) else "candidates")
            if rows
            else "no_matching_candidates"
        ),
        scenarios,
    )


def _fmt(x):
    if x is None:
        return "неизвестно"
    return (
        f"{x:,.0f}".replace(",", " ")
        if float(x).is_integer()
        else f"{x:,.1f}".replace(",", " ")
    )


def _percent(x):
    return _fmt(100 * x) + "%"


def build_fact_catalog(user):
    a = user["analysis"]
    cards = [c for c in user["items"] if c["analyzed"]]
    facts = []

    def add(key, section, text, items=(), searches=(), priority=1, **other):
        facts.append(
            dict(
                id=key,
                section=section,
                text=text,
                item_ids=sorted(set(items)),
                search_refs=sorted(set(searches)),
                rule_version=RULE_VERSION,
                priority=priority,
                **other,
            )
        )

    add(
        "overview",
        "profile",
        f"В анализе {len(cards)} уникальных объявлений; исходных карточек {len(user['items'])}, поисковых событий {user['counts']['searches']}.",
        [c["item"] for c in cards],
        priority=1,
    )
    for field, row in a["search"]["fields"].items():
        values = ", ".join(row["values"][:8])
        suffix = f" Значения: {values}." if values else ""
        add(
            "explicit-" + field,
            "parameters",
            f"{row['label']} явно задан в {row['states']} из {row['total_states']} уникальных конфигураций, в {row['days']} разных днях.{suffix}",
            searches=row["search_refs"],
            priority=3 if field != "brand_region" else 2,
        )
    for row in a["importance"]:
        if row["status"] not in (
            "distinctive",
            "common_in_sample",
            "variable",
            "concentrated",
        ):
            continue
        values = " / ".join(row["values"])
        obs = f"{row['label']}: {values} — {row['count']} из {row['known']} известных значений (заполнено {row['known']}/{row['total']})."
        bg = (
            f" Среди остальных пользователей эта группа значений составляет {_percent(row['reference_share'])}."
            if row["reference_share"] is not None
            else ""
        )
        meaning = {
            "distinctive": " Отличительный описательный сигнал относительно выборки.",
            "common_in_sample": " Такая концентрация характерна и остальной выборке, сама по себе не выделяет личный приоритет.",
            "variable": " Значения разнообразны; это не доказывает неважность параметра.",
            "concentrated": " Наблюдаемая концентрация; данных для отличия от фона недостаточно.",
        }[row["status"]]
        add(
            "importance-" + row["field"],
            "importance",
            obs + bg + meaning,
            row["evidence"],
            priority=3 if row["status"] == "distinctive" else 1,
        )
    for row in a["numeric"]:
        if not row["count"]:
            continue
        unit = {
            "price": " ₽",
            "year": " г.",
            "mileage_km": " км",
            "power_hp": " л.с.",
            "engine_liters": " л",
        }[row["field"]]
        text = f"{row['label']}: кратчайший интервал, включающий не менее 80% известных значений, {_fmt(row['core_min'])}–{_fmt(row['core_max'])}{unit}; в нём {row['core_count']} из {row['count']}, заполнено {row['count']}/{row['total']}."
        text += (
            " Диапазон узкий по фиксированному правилу."
            if row["narrow"]
            else " Недостаточно оснований называть диапазон узким."
        )
        add(
            "numeric-" + row["field"],
            "parameters",
            text,
            row["evidence"],
            priority=3 if row["narrow"] else 1,
        )
    for i, m in enumerate(a["models"][:6]):
        pr = m["price"]
        price = (
            f" Медиана цены {_fmt(pr['median'])} ₽, цена известна у {pr['count']}."
            if pr["count"]
            else " Цена неизвестна."
        )
        add(
            "model-" + str(i + 1),
            "models",
            f"{m['brand']} {m['model']}: {m['count']} из {m['total']} объявлений истории."
            + price,
            m["evidence"],
            priority=2,
        )
    for row in a["model_common"]:
        add(
            "common-" + row["field"],
            "models",
            f"Что объединяет модели: {row['label']} — {row['value']} является преобладающим значением у {row['model_count']} из {row['known_models']} моделей с достаточными данными (всего моделей {row['total_models']}).",
            row["evidence"],
            priority=2,
        )
    topology = {
        "dominant": "Есть доминирующая группа, покрывающая не менее половины анализируемых объявлений.",
        "tied": "У крупнейших групп одинаковый размер; единственного основного сценария нет.",
        "largest_only": "Крупнейшая группа есть, но её недостаточно называть доминирующим интересом.",
        "not_available": "Недостаточно данных для групп из трёх объявлений.",
    }[a["cluster_status"]]
    add(
        "topology",
        "clusters",
        topology
        + [
            f" В группах {a['cluster_assigned']} из {len(cards)} объявлений; в группах из двух — {a['small_group_items']}, без полных полей группировки — {a['cluster_unavailable']}."
        ][0],
        [c["item"] for c in cards],
        priority=3,
    )
    for i, c in enumerate(user["clusters"][:4]):
        add(
            "cluster-" + str(i + 1),
            "clusters",
            f"Группа {i+1}: {c['body_type']}, {c['transmission']}; {c['count']} из {c['total']} объявлений. Центральные 80% цен {_fmt(c['price']['p10'])}–{_fmt(c['price']['p90'])} ₽.",
            c["evidence"],
            priority=2,
        )
    if user["outliers"]:
        add(
            "rare",
            "clusters",
            f"{len(user['outliers'])} объявлений образуют единичные сочетания кузова, КПП и ценовой полосы. Это кандидаты на отдельный сценарий или случайный просмотр, а не доказанные случайные действия.",
            [x["item"] for x in user["outliers"]],
            priority=2,
        )
    for i, r in enumerate(user["recommendations"]):
        warning = (
            " Есть непроверяемые ограничения: "
            + ", ".join(LABELS.get(f, f) for f in r["unverified"])
            + "."
            if r["unverified"]
            else " Доступные ограничения сценария выполнены."
        )
        add(
            "alternative-" + str(i + 1),
            "alternatives",
            f"Новая для истории пара: {r['brand']} {r['model']}. Пример стоит {_fmt(r['price'])} ₽, год {_fmt(r['year'])}; источник — снимок объявления из общего пула."
            + warning,
            priority=2,
            candidate_item_id=r["example_item"],
        )
    if not user["recommendations"]:
        reasons = {
            "insufficient_history": "Недостаточно истории для подбора новых моделей.",
            "invalid_latest_constraints": "Последняя конфигурация содержит некорректные или неподдерживаемые ограничения; автоматический подбор остановлен до уточнения.",
            "no_matching_candidates": "Новых моделей, соответствующих выбранным ограничениям, в доступном пуле не найдено.",
        }
        add(
            "alternatives-empty",
            "alternatives",
            reasons.get(a["recommendation_status"], "Подбор не выполнен."),
            priority=2,
        )
    region = a["search"]["fields"].get("brand_region")
    add(
        "country",
        "limitations",
        ("Регион бренда доступен как явный фильтр поиска. " if region else "")
        + "Проверенной страны производства каждого автомобиля в архиве нет; по марке она не приписывается.",
        priority=1,
    )
    add(
        "communications",
        "communications",
        "Содержание разговоров и переписок отсутствует. Темы вопросов и ключевой параметр из общения определить нельзя.",
        priority=1,
    )
    add(
        "temporal",
        "limitations",
        "Цены и характеристики описывают сохранённые снимки; не для всех объявлений доказано состояние на момент действия. Наличие альтернатив в продаже и их полезность не проверены.",
        priority=1,
    )
    return facts


def enrich_payload(payload, dataset):
    users = payload["users"]
    vocab = {
        field: {c.get(field) for u in users for c in u["items"] if c.get(field)}
        for field in CAT_FIELDS
    }
    for user, raw in zip(users, dataset["users"]):
        # Leave the user out; deduplicate shared listings in the reference pool.
        others = {}
        for other in users:
            if other["id"] == user["id"]:
                continue
            for c in other["items"]:
                if c["analyzed"]:
                    old = others.get(c["item"])

                    def key(x):
                        stamp = (
                            x.get("last", "")
                            if "момент" in x.get("priceSource", "")
                            else x.get("snapshot", "")
                        )
                        return (
                            stamp,
                            x.get("last", ""),
                            digest(
                                {
                                    k: x.get(k)
                                    for k in CAT_FIELDS
                                    + (
                                        "price",
                                        "year",
                                        "mileage_km",
                                        "power_hp",
                                        "engine_liters",
                                        "doors",
                                        "technicalSource",
                                        "technicalSnapshot",
                                    )
                                }
                            ),
                        )

                    if old is None or key(c) > key(old):
                        others[c["item"]] = c
        reference = list(others.values())
        cards = [c for c in user["items"] if c["analyzed"]]
        search = analyze_searches(raw.get("searches", []), vocab)
        models, common = _models(cards)
        analysis = dict(
            version=RULE_VERSION,
            search=search,
            importance=_importance(user, reference, search),
            numeric=[_numeric(cards, f) for f in DOMAINS],
            models=models,
            model_common=common,
            **_topology(user),
        )
        user["analysis"] = analysis
        (
            user["recommendations"],
            analysis["recommendation_status"],
            analysis["recommendation_scenarios"],
        ) = _alternatives(user, reference, search, analysis)
        user["facts"] = build_fact_catalog(user)
        user["fact_hash"] = digest(user["facts"])
        user["summary"] = dict(
            status="evidence_rules",
            fact_hash=user["fact_hash"],
            overview=[
                f["id"]
                for f in sorted(user["facts"], key=lambda f: (-f["priority"], f["id"]))[
                    :5
                ]
            ],
            sections=[
                dict(
                    id=section,
                    title=title,
                    fact_ids=[
                        f["id"] for f in user["facts"] if f["section"] == section
                    ],
                )
                for section, title in [
                    ("profile", "Область анализа"),
                    ("parameters", "Предпочтения и диапазоны"),
                    ("importance", "Ключевые и вариативные параметры"),
                    ("models", "Модели и общие свойства"),
                    ("clusters", "Сценарии и редкие просмотры"),
                    ("alternatives", "Новые модели для проверки"),
                    ("communications", "Общение"),
                    ("limitations", "Границы вывода"),
                ]
            ],
        )
    payload["audit"].update(
        recommendations_users=sum(bool(u["recommendations"]) for u in users),
        distinct_signal_users=sum(
            any(r["status"] == "distinctive" for r in u["analysis"]["importance"])
            for u in users
        ),
        explicit_filter_users=sum(
            bool(u["analysis"]["search"]["fields"]) for u in users
        ),
        brand_region_filter_users=sum(
            "brand_region" in u["analysis"]["search"]["fields"] for u in users
        ),
        narrow_range_users=sum(
            any(r["narrow"] for r in u["analysis"]["numeric"]) for u in users
        ),
        summary_users=len(users),
        unique_search_states=sum(
            u["analysis"]["search"]["unique_states"] for u in users
        ),
    )
    payload["audit"]["recommendation_statuses"] = dict(
        Counter(u["analysis"]["recommendation_status"] for u in users)
    )
    payload["audit"]["cluster_statuses"] = dict(
        Counter(u["analysis"]["cluster_status"] for u in users)
    )
    payload["version"] = RULE_VERSION
    payload["method"].update(
        analysis="Evidence-grounded descriptive analysis; no ranking evaluation",
        importance="Explicit filters and leave-one-user-out sample contrast, not causal importance",
        numeric_interval="Shortest observed interval covering at least 80% of known values",
    )
    return payload
