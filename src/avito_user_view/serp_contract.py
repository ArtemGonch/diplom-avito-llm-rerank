"""Structural preflight for a normalized, pseudonymized ranking export."""

from datetime import datetime


LABELS = ("clicked", "target_contact")


def _time(value):
    try:
        t = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return t if t.utcoffset() is not None else None


def _has_id(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def inspect_serps(rows):
    request_ids = set()
    user_ids = set()
    candidates = 0
    missing_context = 0
    outcomes = {
        label: dict(
            candidates_with_unknown_labels=0,
            requests_with_unknown_labels=0,
            complete_requests_with_positive=0,
            complete_requests_without_positive=0,
        )
        for label in LABELS
    }
    for i, row in enumerate(rows, 1):
        def reject(reason):
            raise ValueError(f"Request row {i}: {reason}")

        if not isinstance(row, dict):
            reject("expected a request object")
        if not all(_has_id(row.get(key)) for key in ("request_id", "user_id")):
            reject("missing or invalid pseudonymous request_id/user_id")
        if row["request_id"] in request_ids:
            reject("duplicate request_id")
        shown = _time(row.get("shown_at"))
        cutoff = _time(row.get("history_cutoff"))
        if shown is None or cutoff is None or cutoff > shown:
            reject("timestamps must have offsets and history_cutoff <= shown_at")
        if not isinstance(row.get("query_context"), dict):
            reject("query_context must be an object; use {} when unavailable")
        pool = row.get("candidates")
        size = row.get("logged_size")
        if type(size) is not int or size < 1 or not isinstance(pool, list) or len(pool) != size:
            reject("candidate count does not match logged_size")
        item_ids = set()
        positions = set()
        for candidate in pool:
            if not isinstance(candidate, dict) or not _has_id(candidate.get("item_id")):
                reject("candidate has no valid pseudonymous item_id")
            if candidate["item_id"] in item_ids:
                reject("duplicate item_id in request")
            pos = candidate.get("position")
            if type(pos) is not int or not 1 <= pos <= size or pos in positions:
                reject("positions must be unique integers from 1 through logged_size")
            for label in LABELS:
                if label not in candidate or (
                    candidate[label] is not None and type(candidate[label]) is not bool
                ):
                    reject(f"{label} must be true, false or null")
            item_ids.add(candidate["item_id"])
            positions.add(pos)
        for label in LABELS:
            values = [c[label] for c in pool]
            unknown = values.count(None)
            audit = outcomes[label]
            if unknown:
                audit["candidates_with_unknown_labels"] += unknown
                audit["requests_with_unknown_labels"] += 1
            elif any(values):
                audit["complete_requests_with_positive"] += 1
            else:
                audit["complete_requests_without_positive"] += 1
        request_ids.add(row["request_id"])
        user_ids.add(row["user_id"])
        candidates += size
        missing_context += not bool(row["query_context"])
    if not request_ids:
        raise ValueError("Empty export")
    return dict(
        version="avito-serp-preflight-v1",
        requests=len(request_ids),
        users=len(user_ids),
        candidates=candidates,
        requests_without_query_context=missing_context,
        click=outcomes["clicked"],
        target_contact=outcomes["target_contact"],
        ranking_evaluated=False,
    )
