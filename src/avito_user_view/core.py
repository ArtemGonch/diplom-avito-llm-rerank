"""Versioned descriptive summaries for the September UsedCars export.

Counts use distinct listings with display-layer activity or canonical contacts.
Raw/client views and contact labels are diagnostics, never independent votes.
All preferences are associations in an observational export, not causal intent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import median

VERSION = "user-view-v1"
FIELDS = (
    "brand",
    "model",
    "body_type",
    "transmission",
    "engine_type",
    "drive",
    "generation",
)
LABELS = dict(
    brand="Марка",
    model="Модель",
    body_type="Кузов",
    transmission="КПП",
    engine_type="Двигатель",
    drive="Привод",
    generation="Поколение",
)
NUMERIC = ("price", "year", "mileage_km", "power_hp", "engine_liters", "doors")


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        n = Decimal(str(value).replace("\u00a0", "").replace(" ", ""))
        value = float(n)
        return value if n.is_finite() and n >= 0 and math.isfinite(value) else None
    except InvalidOperation:
        return None


def verify_archive(root: Path):
    """Verify all manifest members before reading any analysis input."""
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Unsafe or missing manifest member")
        if (
            path.stat().st_size != entry["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Manifest hash mismatch: " + entry["path"])
    return manifest


def _events(rows):
    # Preserve source multiplicity. Distinct items, not event counts, vote in profiles.
    return sorted(
        [
            dict(
                t=str(e.get("t", "")),
                kind=str(e.get("kind", "other")),
                eid=e.get("eid"),
            )
            for e in rows
        ],
        key=lambda e: (e["t"], e["kind"], str(e["eid"])),
    )


def _merge_events(left, right):
    # Duplicate card rows must not double the same source events.
    counts = Counter(json.dumps(e, sort_keys=True) for e in left) | Counter(
        json.dumps(e, sort_keys=True) for e in right
    )
    return _events([json.loads(k) for k, n in counts.items() for _ in range(n)])


def reconcile_btc(dataset, rows):
    """Add source BTC events absent from viewer, as a separate diagnostic layer.

    Match timestamps to microsecond precision and preserve event multiplicity.
    Never merge this layer into the original viewer counts or preference votes.
    """
    result = deepcopy(dataset)
    cards = {
        (str(u["id"]), str(c["item"])): c
        for u in result["users"]
        for c in u.get("cards", [])
    }

    def key(uid, item, t, eid):
        stamp = datetime.fromisoformat(str(t)).isoformat(timespec="microseconds")
        return str(uid), str(item), stamp, int(eid or 0)

    existing = Counter(
        key(uid, item, e["t"], e.get("eid"))
        for (uid, item), c in cards.items()
        for e in c.get("events", [])
    )
    affected = set()
    missing = 0
    extra = Counter()
    for row in rows:
        uid, item = str(row["user_id"]), str(row["item_id"])
        k = key(uid, item, row["action_ts"], row["action_eid"])
        if existing[k]:
            existing[k] -= 1
            continue
        if (uid, item) not in cards:
            missing += 1
            continue
        cards[(uid, item)].setdefault("supplementaryEvents", []).append(
            dict(
                t=str(row["action_ts"]),
                kind=str(row.get("action_type", "btc")),
                eid=row["action_eid"],
                source="source/btc_history.jsonl",
            )
        )
        affected.add(uid)
        extra[str(row["action_eid"])] += 1
    report = dict(
        source_rows=len(rows),
        added_events=sum(extra.values()),
        affected_users=len(affected),
        unmatched_items=missing,
        added_by_eid=dict(extra),
    )
    result["btcReconciliation"] = report
    return result, report


def _normalize(user):
    by_id = {}
    for c in user.get("cards", []):
        item = str(c["item"])  # Never round IDs through float/JavaScript Number.
        r = dict(item=item, title=str(c.get("title") or "Без названия")[:250])
        r.update(
            {
                k: str(c[k]).strip() if c.get(k) not in (None, "") else None
                for k in FIELDS
            }
        )
        r.update({k: number(c.get(k)) for k in NUMERIC})
        if r["year"] is not None and not 1886 <= r["year"] <= 2027:
            r["year"] = None
        r.update(
            {
                k: str(c.get(k) or "")[:200]
                for k in (
                    "priceSource",
                    "attributeSource",
                    "snapshot",
                    "last",
                    "displayLast",
                    "technicalSource",
                    "technicalSnapshot",
                    "technicalTemporalStatus",
                )
            }
        )
        r["technicalInterval"] = c.get("technicalInterval")
        r["price_flag"] = (
            "Цена ниже 10 000 ₽; исключена из ценовых сводок"
            if r["price"] is not None and r["price"] < 10_000
            else None
        )
        r["supplementaryEvents"] = _events(c.get("supplementaryEvents", []))
        imv = c.get("imv") or {}
        r["imv"] = {k: number(imv.get(k)) for k in ("point", "low", "high")}
        r["imv"]["day"] = str(imv.get("day") or "")
        r["events"] = _events(c.get("events", []))
        r["displayEvents"] = _events(c.get("displayEvents", []))
        r["contacts"] = [
            dict(
                t=str(x.get("t", "")),
                channel=str(x.get("channel", "")),
                is_target=x.get("is_target"),
                is_common_funnel=x.get("is_common_funnel"),
            )
            for x in c.get("contacts", [])
        ]
        r["analyzed"] = bool(r["displayEvents"] or r["contacts"])
        if item in by_id:
            # Duplicate row: retain one snapshot deterministically, union evidence.
            old = by_id[item]
            winner = max(
                (old, r),
                key=lambda a: (
                    a["snapshot"],
                    json.dumps({k: a[k] for k in FIELDS + NUMERIC}, sort_keys=True),
                ),
            )
            winner = dict(winner)
            winner["events"] = _merge_events(old["events"], r["events"])
            winner["displayEvents"] = _merge_events(
                old["displayEvents"], r["displayEvents"]
            )
            winner["supplementaryEvents"] = _merge_events(
                old["supplementaryEvents"], r["supplementaryEvents"]
            )
            winner["contacts"] = list(
                {
                    json.dumps(x, sort_keys=True): x
                    for x in old["contacts"] + r["contacts"]
                }.values()
            )
            winner["analyzed"] = bool(winner["displayEvents"] or winner["contacts"])
            r = winner
        by_id[item] = r
    return sorted(by_id.values(), key=lambda c: c["item"])


def _range(cards, field):
    ns = sorted(
        c[field]
        for c in cards
        if c[field] is not None and not (field == "price" and c["price_flag"])
    )
    if not ns:
        return None

    def q(frac):
        p = (len(ns) - 1) * frac
        lo = int(p)
        hi = math.ceil(p)
        return ns[lo] + (ns[hi] - ns[lo]) * (p - lo)

    return dict(
        min=ns[0],
        p10=q(0.1),
        median=median(ns),
        p90=q(0.9),
        max=ns[-1],
        count=len(ns),
        total=len(cards),
    )


def _preferences(cards):
    preferences, variable, distributions = [], [], []
    total = len(cards)
    for field in FIELDS:
        known = [c for c in cards if c[field]]
        counts = Counter(c[field] for c in known)
        top = sorted(counts, key=lambda x: (-counts[x], x))
        dist = dict(
            field=field,
            label=LABELS[field],
            known=len(known),
            total=total,
            values=[
                dict(
                    value=v,
                    count=counts[v],
                    evidence=[c["item"] for c in known if c[field] == v],
                )
                for v in top
            ],
        )
        distributions.append(dist)
        if len(known) < 5 or len(known) < 0.6 * total:
            continue
        values = []
        if counts[top[0]] / len(known) >= 0.7:
            values = top[:1]
        elif (
            len(top) > 1
            and (counts[top[0]] + counts[top[1]]) / len(known) >= 0.85
            and counts[top[1]] / len(known) >= 0.2
        ):
            values = top[:2]
        if values:
            preferences.append(
                dict(
                    field=field,
                    label=LABELS[field],
                    values=values,
                    count=sum(counts[v] for v in values),
                    known=len(known),
                    total=total,
                    evidence=[c["item"] for c in known if c[field] in values],
                )
            )
        elif len(top) >= 3 and counts[top[0]] / len(known) < 0.5:
            variable.append(
                dict(
                    field=field,
                    label=LABELS[field],
                    known=len(known),
                    distinct=len(top),
                )
            )
    return preferences, variable, distributions


def _clusters(cards):
    groups = defaultdict(list)
    for c in cards:
        # Interpretable groups. Missing attributes cannot create a "preference".
        if c["body_type"] and c["transmission"] and c["price"] and not c["price_flag"]:
            groups[(c["body_type"], c["transmission"])].append(c)
    parts = []
    for key, group in sorted(groups.items()):
        batches = []
        for c in sorted(group, key=lambda c: (c["price"], c["item"])):
            if not batches or c["price"] / batches[-1][-1]["price"] > 1.8:
                batches.append([])
            batches[-1].append(c)
        for batch in batches:
            parts.append((key, batch))
    parts.sort(
        key=lambda p: (
            -len(p[1]),
            p[0],
            median(c["price"] for c in p[1]),
            [c["item"] for c in p[1]],
        )
    )
    clusters = []
    for key, batch in parts:
        if len(batch) < 3:
            continue
        clusters.append(
            dict(
                id=f"cluster-{len(clusters)+1}",
                body_type=key[0],
                transmission=key[1],
                count=len(batch),
                total=len(cards),
                evidence=[c["item"] for c in batch],
                price=_range(batch, "price"),
                year=_range(batch, "year"),
                mileage_km=_range(batch, "mileage_km"),
            )
        )
    outliers = [
        dict(
            item=batch[0]["item"],
            reason="Единственное сочетание кузова, КПП и ценовой группы в истории",
        )
        for _, batch in parts
        if len(batch) == 1
    ]
    return clusters, outliers


def _recommend(cards, clusters, pool, all_items):
    if len(cards) < 5 or not clusters:
        return []
    main = clusters[0]
    pr = main["price"]
    seen = {(c["brand"], c["model"]) for c in all_items if c["brand"] and c["model"]}
    best = {}
    for c in pool:
        model = (c["brand"], c["model"])
        if not all(model) or model in seen or not c["price"] or c["price_flag"]:
            continue
        if (
            c["body_type"] != main["body_type"]
            or c["transmission"] != main["transmission"]
        ):
            continue
        if not pr["p10"] * 0.8 <= c["price"] <= pr["p90"] * 1.2:
            continue
        yr = main["year"]
        if yr and (
            c["year"] is None or not yr["p10"] - 3 <= c["year"] <= yr["p90"] + 3
        ):
            continue
        score = abs(math.log(c["price"] / pr["median"]))
        key = (score, c["item"])
        if model not in best or key < best[model][0]:
            best[model] = (key, c)
    return [
        dict(
            brand=c["brand"],
            model=c["model"],
            example_item=c["item"],
            price=c["price"],
            year=c["year"],
            reasons=[
                main["body_type"],
                main["transmission"],
                "Цена рядом с основной группой",
                (
                    "Год в диапазоне группы ±3 года"
                    if main["year"]
                    else "Год не ограничен"
                ),
            ],
            source="Истории других пользователей в этом сэмпле; гипотеза для проверки",
            example=dict(c),
        )
        for _, c in sorted(best.values(), key=lambda x: x[0])[:6]
    ]


def _behavior(cards):
    facts = []
    models = defaultdict(list)
    for c in cards:
        if (
            c["brand"]
            and c["model"]
            and any(e["kind"] == "view" for e in c["displayEvents"])
        ):
            models[(c["brand"], c["model"])].append(c)
    returns = []
    for (brand, model), group in sorted(models.items()):
        days = {
            e["t"][:10]
            for c in group
            for e in c["displayEvents"]
            if e["kind"] == "view"
        }
        if len(days) >= 2 and len(group) >= 3:
            returns.append(
                dict(
                    kind="return_model",
                    brand=brand,
                    model=model,
                    days=len(days),
                    count=len(group),
                    evidence=[c["item"] for c in group],
                )
            )
    facts.extend(
        sorted(
            returns, key=lambda x: (-x["days"], -x["count"], x["brand"], x["model"])
        )[:2]
    )
    contacted = [
        c
        for c in cards
        if any(x.get("is_common_funnel") is True for x in c["contacts"])
    ]
    if len(contacted) >= 3:
        known = [c for c in contacted if c["brand"]]
        ct = Counter(c["brand"] for c in known)
        if ct:
            brand = min(ct, key=lambda b: (-ct[b], b))
            if ct[brand] >= 3 and ct[brand] / len(known) >= 0.7:
                facts.append(
                    dict(
                        kind="contact_focus",
                        brand=brand,
                        count=ct[brand],
                        total=len(contacted),
                        known=len(known),
                        baseline=sum(c["brand"] == brand for c in cards),
                        baseline_total=len(cards),
                        evidence=[c["item"] for c in contacted if c["brand"] == brand],
                    )
                )
    favs = [
        c
        for c in cards
        if any(e["kind"] == "favorite" for e in c["displayEvents"])
        and c["price"]
        and not c["price_flag"]
    ]
    if len(favs) >= 3:
        facts.append(
            dict(
                kind="favorite_prices",
                count=len(favs),
                median=median(c["price"] for c in favs),
                evidence=[c["item"] for c in favs],
            )
        )
    return facts


def build_payload(dataset):
    ids = [str(u["id"]) for u in dataset["users"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate users")
    normalized = [_normalize(u) for u in dataset["users"]]
    pool = sorted(
        (c for cs in normalized for c in cs if c["analyzed"]), key=lambda c: c["item"]
    )
    result = []
    for idx, (raw, items) in enumerate(zip(dataset["users"], normalized)):
        cards = [c for c in items if c["analyzed"]]
        prefs, variable, dist = _preferences(cards)
        clusters, outliers = _clusters(cards)
        searches = []
        for search_index, s in enumerate(raw.get("searches", [])):
            matched = s.get("payloadStatus") == "matched"
            searches.append(
                dict(
                    ref=f"search-{search_index+1}",
                    t=str(s.get("t", "")),
                    status=s.get("payloadStatus", "unknown"),
                    query=str(s.get("query") or "")[:250] if matched else "",
                    filters=(
                        [str(f)[:250] for f in s.get("filters", [])][:25]
                        if matched
                        else []
                    ),
                )
            )
        filters = Counter(f for s in searches for f in set(s["filters"]))
        counts = dict(
            items=len(items),
            analyzed_items=len(cards),
            raw_events=sum(len(c.get("events", [])) for c in raw.get("cards", [])),
            projected_raw_events=sum(len(c["events"]) for c in items),
            display_events=sum(len(c["displayEvents"]) for c in items),
            views=sum(e["kind"] == "view" for c in items for e in c["displayEvents"]),
            favorites=sum(
                e["kind"] == "favorite" for c in items for e in c["displayEvents"]
            ),
            contacts=sum(len(c["contacts"]) for c in items)
            + len(raw.get("unmappedContacts", [])),
            stream_contacts=sum(
                e["kind"] == "stream_contact" for c in items for e in c["events"]
            ),
            funnel_contacts=sum(
                x.get("is_common_funnel") is True for c in items for x in c["contacts"]
            )
            + sum(
                x.get("is_common_funnel") is True
                for x in raw.get("unmappedContacts", [])
            ),
            target_funnel_contacts=sum(
                x.get("is_common_funnel") is True and x.get("is_target") is True
                for c in items
                for x in c["contacts"]
            )
            + sum(
                x.get("is_common_funnel") is True and x.get("is_target") is True
                for x in raw.get("unmappedContacts", [])
            ),
            btc_supplement=sum(len(c["supplementaryEvents"]) for c in items),
            searches=len(searches),
            unmapped_contacts=len(raw.get("unmappedContacts", [])),
        )
        reports = [
            dict(
                t=str(e.get("t", "")),
                kind=str(e.get("kind", "")),
                item=str(e["item"]) if e.get("item") else None,
            )
            for e in raw.get("reportEvents", [])
        ]
        counts["report_purchases"] = sum(
            e["kind"] == "report_purchase" for e in reports
        )
        counts["report_uses"] = sum(e["kind"] == "report_use" for e in reports)
        timeline = defaultdict(Counter)
        for c in cards:
            for e in c["displayEvents"]:
                timeline[e["t"][:10]][e["kind"]] += 1
        result.append(
            dict(
                id=ids[idx],
                label=f"Пользователь {idx+1:03}",
                status=(
                    "empty" if not cards else "sparse" if len(cards) < 5 else "ready"
                ),
                counts=counts,
                behavior=_behavior(cards),
                preferences=prefs,
                variable=variable,
                distributions=dist,
                clusters=clusters,
                outliers=outliers,
                country=None,
                report_events=reports,
                items=items,
                searches=sorted(searches, key=lambda s: s["t"], reverse=True),
                repeated_filters=[
                    dict(text=f, count=n)
                    for f, n in sorted(filters.items(), key=lambda x: (-x[1], x[0]))[
                        :12
                    ]
                ],
                timeline=[
                    dict(day=day, counts=dict(ct))
                    for day, ct in sorted(timeline.items())
                ],
                recommendations=_recommend(cards, clusters, pool, items),
                communications=dict(status="not_available", topics=[]),
                price_sources=dict(
                    Counter(c["priceSource"] or "Не указан" for c in cards)
                ),
                attribute_sources=dict(
                    Counter(c["attributeSource"] or "Не указан" for c in cards)
                ),
            )
        )
    audit = dict(
        users=len(result),
        active_users=sum(u["counts"]["raw_events"] > 0 for u in result),
        empty_users=sum(u["status"] == "empty" for u in result),
        raw_events=sum(u["counts"]["raw_events"] for u in result),
        display_events=sum(u["counts"]["display_events"] for u in result),
        user_item_rows=sum(u["counts"]["items"] for u in result),
        analyzed_items=sum(u["counts"]["analyzed_items"] for u in result),
        searches=sum(u["counts"]["searches"] for u in result),
        ambiguous_searches=sum(
            s["status"] == "ambiguous" for u in result for s in u["searches"]
        ),
        contacts=sum(u["counts"]["contacts"] for u in result),
        stream_contacts=sum(u["counts"]["stream_contacts"] for u in result),
        eligible_preferences=sum(bool(u["preferences"]) for u in result),
        recommendations_users=sum(bool(u["recommendations"]) for u in result),
        cluster_users=sum(bool(u["clusters"]) for u in result),
        missing_attrs={
            f: sum(not c[f] for cs in normalized for c in cs) for f in FIELDS
        },
        missing_prices=sum(c["price"] is None for cs in normalized for c in cs),
        flagged_prices=sum(bool(c["price_flag"]) for cs in normalized for c in cs),
        funnel_contacts=sum(u["counts"]["funnel_contacts"] for u in result),
        target_funnel_contacts=sum(
            u["counts"]["target_funnel_contacts"] for u in result
        ),
    )
    payload = dict(
        version=VERSION,
        period=dataset.get("period", []),
        users=result,
        audit=audit,
        source_audit=dataset.get("enrichmentAudit", {}),
        technical_audit=dataset.get("technicalAudit", {}),
        btc_reconciliation=dataset.get("btcReconciliation", {}),
        coverage_note=dataset.get("coverageNote", ""),
        method=dict(
            min_items=5,
            min_coverage=0.6,
            single_share=0.7,
            pair_share=0.85,
            cluster_price_gap=1.8,
            analysis="Descriptive rules, not an LLM run; no ranking evaluation",
        ),
    )

    from .analysis import enrich_payload

    return enrich_payload(payload, dataset)


def render_html(payload):
    template = (
        Path(__file__).resolve().parents[2] / "apps/avito_user_view/template.html"
    ).read_text()
    safe = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return template.replace("__USER_VIEW_DATA__", safe)
