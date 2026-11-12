"""Aggregate-only evaluation of held-out filters and expert A/B annotations."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean

from .analysis import CAT_PARAMS, _values


VERSION = "avito-search-holdout-v1"
CUTOFF = datetime(2026, 9, 13)
# Directly typed search parameters only; no full-history vocabulary lookup.
FIELDS = {
    code: CAT_PARAMS[code]
    for code in ("110000", "110008", "110006", "110007", "152642")
}
METHODS = ("last_explicit", "mode_by_day", "consensus", "cohort_mode")
RUBRIC = ("accuracy", "relevance", "actionability", "clarity")
PREFERENCE = "preference_A_B_tie"
SCORE_COLUMNS = tuple(f"{d}_{side}_1_5" for d in RUBRIC for side in ("A", "B"))
ANSWER_COLUMNS = {"comparison_id", "user_index", PREFERENCE, *SCORE_COLUMNS, "notes"}


def _mean(values):
    values = list(values)
    return mean(values) if values else None


def _search_time(search):
    raw = search.get("t")
    try:
        t = datetime.fromisoformat(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid search timestamp") from exc
    if t.tzinfo is not None or len(raw) < 16:
        raise ValueError("Expected local date and time as stored in the export")
    return t


def _projection(search):
    params = search.get("params")
    if not isinstance(params, dict):
        raise ValueError("Matched search has no typed parameter object")
    return tuple(tuple(_values(params.get(code))) for code in FIELDS)


def _explicit_by_time(history, field):
    by_time = defaultdict(set)
    for t, config in history:
        by_time[t].update(config[field])
    return {t: values for t, values in by_time.items() if values}


def _unique_mode(counts):
    ordered = counts.most_common(2)
    if ordered and (len(ordered) == 1 or ordered[0][1] > ordered[1][1]):
        return ordered[0][0]
    return None


def _predictions(by_time):
    if not by_time:
        return dict.fromkeys(METHODS[:3])
    latest = by_time[max(by_time)]
    last = next(iter(latest)) if len(latest) == 1 else None
    value_days = {
        (next(iter(values)), t.date())
        for t, values in by_time.items()
        if len(values) == 1
    }
    mode = _unique_mode(Counter(v for v, _ in value_days))
    return dict(
        last_explicit=last,
        mode_by_day=mode,
        consensus=last if last == mode else None,
    )


def _filter_scores(slots):
    result = {}
    for method in METHODS:
        predicted = [
            (u, target == predictions[method])
            for u, _, target, predictions in slots
            if predictions[method] is not None
        ]
        by_user = defaultdict(list)
        for u, hit in predicted:
            by_user[u].append(hit)
        result[method] = dict(
            predicted=len(predicted),
            hits=sum(hit for _, hit in predicted),
            users_predicted=len(by_user),
            coverage=len(predicted) / len(slots) if slots else None,
            precision_on_predicted=_mean(hit for _, hit in predicted),
            hit_rate_over_all_slots=(
                sum(hit for _, hit in predicted) / len(slots) if slots else None
            ),
        )
    return result


def _common_scores(slots, methods):
    common = defaultdict(lambda: defaultdict(list))
    for user, _, target, predictions in slots:
        if all(predictions[m] is not None for m in methods):
            for m in methods:
                common[user][m].append(predictions[m] == target)
    return dict(
        users=len(common),
        slots=sum(len(x[methods[0]]) for x in common.values()),
        **{
            f"{m}_user_mean": _mean(mean(x[m]) for x in common.values())
            for m in methods
        },
    )


def temporal_search_report(dataset):
    """Fixed midpoint holdout. The first recorded future timestamp is the target."""
    identities = [u.get("id") for u in dataset["users"]]
    if (
        any(not isinstance(x, str) or not x or x != x.strip() for x in identities)
        or len(set(identities)) != len(identities)
    ):
        raise ValueError("Expected distinct, nonempty user IDs")
    slots = []
    repeated_users = set()
    counts = Counter()
    histories = [
        [(_search_time(row), row) for row in user.get("searches", [])]
        for user in dataset["users"]
    ]
    prefixes = [
        [
            (t, _projection(row))
            for t, row in searches
            if t < CUTOFF and row.get("payloadStatus") == "matched"
        ]
        for searches in histories
    ]
    explicit = [
        [_explicit_by_time(history, field) for field in range(len(FIELDS))]
        for history in prefixes
    ]
    user_values = [
        [
            Counter({next(iter(v)): 1 for v in by_time.values() if len(v) == 1})
            for by_time in fields
        ]
        for fields in explicit
    ]
    popularity = [
        sum((values[field] for values in user_values), Counter())
        for field in range(len(FIELDS))
    ]
    for index, searches in enumerate(histories):
        future = [(t, row) for t, row in searches if t >= CUTOFF]
        if not future:
            counts["users_without_future_search"] += 1
            continue
        first = min(t for t, _ in future)
        group = [row for t, row in future if t == first]
        targets = {
            _projection(row) for row in group if row.get("payloadStatus") == "matched"
        }
        if len(targets) != 1 or any(
            row.get("payloadStatus") != "matched" for row in group
        ):
            counts["users_ambiguous_first_future"] += 1
            continue
        target = targets.pop()
        if not any(len(values) == 1 for values in target):
            counts["users_without_singleton_target"] += 1
            continue
        history = prefixes[index]
        if not history:
            counts["target_users_without_matched_prefix"] += 1
        if any(config == target for _, config in history):
            repeated_users.add(index)
        for field, values in enumerate(target):
            if len(values) == 1:
                predictions = _predictions(explicit[index][field])
                predictions["cohort_mode"] = _unique_mode(
                    popularity[field] - user_values[index][field]
                )
                slots.append((index, field, values[0], predictions))
    return dict(
        version=VERSION,
        cutoff_source_time=CUTOFF.isoformat(),
        cohort_users=len(dataset["users"]),
        users_without_future_search=counts["users_without_future_search"],
        users_ambiguous_first_future=counts["users_ambiguous_first_future"],
        users_without_singleton_target=counts["users_without_singleton_target"],
        target_users_without_matched_prefix=counts["target_users_without_matched_prefix"],
        target_users=len({u for u, *_ in slots}),
        target_slots=len(slots),
        users_with_repeated_projection=len(repeated_users),
        methods=_filter_scores(slots),
        common=_common_scores(slots, METHODS[:2]),
        common_with_cohort=_common_scores(slots, (*METHODS[:2], "cohort_mode")),
        by_field={
            name: _filter_scores([s for s in slots if s[1] == index])
            for index, name in enumerate(FIELDS.values())
        },
        by_projection={
            "repeated": _filter_scores([s for s in slots if s[0] in repeated_users]),
            "unseen": _filter_scores([s for s in slots if s[0] not in repeated_users]),
        },
        ranking_evaluated=False,
        llm_evaluated=False,
    )


def annotation_report(key_rows, answer_files):
    """Decode fixed A/B keys; answer_files are (pseudonymous expert label, CSV)."""
    keys = {}
    expected = Counter()
    for row in key_rows:
        case = row.get("comparison_id", "")
        baseline = row.get("baseline")
        sides = row.get("sides")
        suffix = {"priority": "P", "diverse": "D"}.get(baseline)
        if (
            not suffix
            or not isinstance(case, str)
            or not re.fullmatch(r"[0-9]{3,}-" + suffix, case)
            or int(case.split("-")[0]) < 1
            or case in keys
            or not isinstance(sides, dict)
            or set(sides) != {"A", "B"}
            or (sides["A"], sides["B"]) not in (("llm", "rules"), ("rules", "llm"))
        ):
            raise ValueError("Invalid or duplicate comparison key")
        user_index = int(case.split("-")[0])
        keys[case] = (baseline, user_index, sides)
        expected[baseline] += 1
    if not keys or len({(b, u) for b, u, _ in keys.values()}) != len(keys):
        raise ValueError("Missing keys or repeated user in baseline")

    votes = defaultdict(lambda: defaultdict(list))
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    insufficient = Counter()
    experts = set()
    for input_files, (expert, file) in enumerate(answer_files, 1):
        if (
            not isinstance(expert, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", expert)
            or expert in experts
        ):
            raise ValueError(f"Invalid or repeated expert label in export {input_files}")
        experts.add(expert)
        reader = csv.DictReader(file, strict=True)
        if (
            not reader.fieldnames
            or len(reader.fieldnames) != len(ANSWER_COLUMNS)
            or set(reader.fieldnames) != ANSWER_COLUMNS
        ):
            raise ValueError(f"Unexpected CSV columns in export {input_files}")
        seen = set()
        for row in reader:
            case = row["comparison_id"]
            where = f"export {input_files}, row {reader.line_num}"
            if None in row or None in row.values() or case in seen or case not in keys:
                raise ValueError(f"Invalid or duplicate comparison at {where}")
            seen.add(case)
            baseline, user, sides = keys[case]
            if not re.fullmatch("[0-9]+", row["user_index"]) or int(row["user_index"]) != user:
                raise ValueError(f"Wrong user index at {where}")
            vote = row[PREFERENCE]
            if vote not in ("", "insufficient", "tie", "A", "B"):
                raise ValueError(f"Invalid preference at {where}")
            if vote == "insufficient":
                insufficient[baseline] += 1
            elif vote:
                votes[baseline][user].append(sides.get(vote, vote))
            for col in SCORE_COLUMNS:
                if row[col] not in ("", "1", "2", "3", "4", "5"):
                    raise ValueError(f"Invalid rating at {where}")
            for rubric in RUBRIC:
                ratings = {sides[s]: row[f"{rubric}_{s}_1_5"] for s in ("A", "B")}
                if all(ratings.values()):
                    scores[baseline][rubric][user].append(
                        int(ratings["llm"]) - int(ratings["rules"])
                    )

    baselines = {}
    vote_value = {"llm": 1, "rules": -1, "tie": 0}
    for baseline in sorted(expected):
        by_user = votes[baseline]
        totals = Counter(v for values in by_user.values() for v in values)
        multi = [v for v in by_user.values() if len(v) >= 2]
        baselines[baseline] = dict(
            expected_users=expected[baseline],
            users_with_preference=len(by_user),
            votes={key: totals[key] for key in vote_value},
            insufficient_votes=insufficient[baseline],
            user_mean_net_preference=_mean(
                mean(vote_value[v] for v in x) for x in by_user.values()
            ),
            multi_rater_users=len(multi),
            agreeing_multi_rater_users=sum(len(set(x)) == 1 for x in multi),
            rubric={
                rubric: dict(
                    paired_users=len(scores[baseline][rubric]),
                    paired_ratings=sum(map(len, scores[baseline][rubric].values())),
                    user_mean_llm_minus_rules=_mean(
                        mean(x) for x in scores[baseline][rubric].values()
                    ),
                )
                for rubric in RUBRIC
            },
        )
    return dict(
        version="avito-summary-annotation-report-v1",
        status="has_preference_votes" if any(votes.values()) else "no_decisive_or_tied_votes",
        rating_exports=len(experts),
        bundle_provenance="manual_confirmation_for_legacy_csv",
        baselines=baselines,
        ranking_evaluated=False,
    )
