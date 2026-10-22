"""Fact-constrained editorial LLM selection; no free factual prose is accepted.

The language model chooses which verified facts deserve the short overview.
The complete report always retains all assignment sections. Every displayed
sentence is resolved from the recalculated canonical fact store, preventing
fabricated metrics, names, causal stories and sensitive profile attributes.
"""

from copy import deepcopy
from .analysis import build_fact_catalog, digest, RULE_VERSION


def validate_selection(selection, user):
    allowed = {"user_index", "fact_hash", "overview_fact_ids"}
    if set(selection) != allowed:
        raise ValueError(
            "Unexpected editor output fields; free factual text is not allowed"
        )
    canonical = build_fact_catalog(user)
    if canonical != user["facts"] or digest(canonical) != user["fact_hash"]:
        raise ValueError("Fact store differs from recalculated evidence")
    if selection["fact_hash"] != user["fact_hash"]:
        raise ValueError("Stale editor input hash")
    ids = selection["overview_fact_ids"]
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= 6
        or any(not isinstance(x, str) for x in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Expected one to six unique fact IDs")
    known = {f["id"] for f in canonical}
    if not set(ids) <= known:
        raise ValueError("Unknown fact ID")
    own = {c["item"] for c in user["items"]}
    searches = {
        r for s in user["analysis"]["search"]["states"] for r in s["search_refs"]
    }
    pool_ids = {r["example_item"] for r in user["recommendations"]}
    for f in canonical:
        if not set(f["item_ids"]) <= own or not set(f["search_refs"]) <= searches:
            raise ValueError("Foreign fact evidence")
        if f.get("candidate_item_id") and f["candidate_item_id"] not in pool_ids:
            raise ValueError("Foreign candidate evidence")
        if f["rule_version"] != RULE_VERSION:
            raise ValueError("Stale rule version")
    return True


def attach_editor_selections(payload, selections, provenance):
    result = deepcopy(payload)
    seen = set()
    if not all(provenance.get(k) for k in ("provider", "model", "model_revision")):
        raise ValueError("Missing LLM provenance")
    for selection in selections:
        idx = selection.get("user_index")
        if (
            not isinstance(idx, int)
            or isinstance(idx, bool)
            or not 1 <= idx <= len(result["users"])
            or idx in seen
        ):
            raise ValueError("Duplicate or invalid user index")
        seen.add(idx)
        user = result["users"][idx - 1]
        validate_selection(selection, user)
        user["summary"].update(
            status="llm_fact_selection",
            overview=list(selection["overview_fact_ids"]),
            provenance=dict(provenance),
        )
    result["editor_audit"] = dict(
        validated_users=len(seen),
        total_users=len(result["users"]),
        status="complete" if len(seen) == len(result["users"]) else "partial",
        method="LLM selects canonical fact IDs; no free factual prose or new values",
        provenance=dict(provenance),
    )
    return result


def editor_requests(payload):
    return [
        dict(
            user_index=i + 1,
            label=u["label"],
            fact_hash=u["fact_hash"],
            facts=[
                {k: f[k] for k in ("id", "section", "text", "priority")}
                for f in u["facts"]
            ],
            instruction="Choose 3–6 fact IDs for a useful short overview. Prioritize explicit filters, distinctive signals, numeric focus, scenario structure and an actionable uncertainty. Do not invent text or numbers. Empty histories need only 1–3 supported facts.",
        )
        for i, u in enumerate(payload["users"])
    ]
