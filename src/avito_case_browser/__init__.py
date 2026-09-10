"""Offline Avito case-browser data preparation and rendering."""

from .core import (
    BROWSER_VERSION,
    INSIGHT_PROMPT_VERSION,
    INSIGHT_VALIDATION_VERSION,
    build_browser_payload,
    build_insight_requests,
    build_item_request_rows,
    deterministic_order,
    extract_json_object,
    load_insight_results,
    load_item_overlay,
    normalize_user_id,
    render_browser_html,
)

__all__ = [
    "BROWSER_VERSION",
    "INSIGHT_PROMPT_VERSION",
    "INSIGHT_VALIDATION_VERSION",
    "build_browser_payload",
    "build_insight_requests",
    "build_item_request_rows",
    "deterministic_order",
    "extract_json_object",
    "load_insight_results",
    "load_item_overlay",
    "normalize_user_id",
    "render_browser_html",
]
