#!/usr/bin/env python3
"""Build the private User View from Roma's verified export, without networking."""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avito_user_view import build_payload, render_html, verify_archive, reconcile_btc
from avito_user_view.source_io import enrich_technical_attributes
from avito_user_view.narrative import editor_requests, attach_editor_selections
from avito_user_view.reports import write_reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports/avito_user_view")
    p.add_argument(
        "--editor-cache",
        type=Path,
        help="Validated LLM fact-selection JSON; no external requests",
    )
    args = p.parse_args()
    manifest = verify_archive(args.input_dir)
    data_path = args.input_dir / "dataset/users.json"
    dataset = json.loads(data_path.read_text())
    btc_path = args.input_dir / "source/btc_history.jsonl"
    rows = [
        json.loads(line) for line in btc_path.read_text().splitlines() if line.strip()
    ]
    dataset, btc = reconcile_btc(dataset, rows)
    dataset, technical_audit = enrich_technical_attributes(dataset, args.input_dir)
    dataset["technicalAudit"] = technical_audit
    payload = build_payload(dataset)
    payload["missing_data_request"] = (
        ROOT / "docs/user_view_missing_data_request.txt"
    ).read_text()
    if args.editor_cache:
        cache = json.loads(args.editor_cache.read_text())
        payload = attach_editor_selections(
            payload, cache["selections"], cache["provenance"]
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requests = editor_requests(payload)
    (output / "editor_requests.json").write_text(
        json.dumps(requests, ensure_ascii=False, indent=2)
    )
    (output / "index.html").write_text(render_html(payload))
    (output / "user_summaries.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    )
    audit = dict(
        version=payload["version"],
        input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        manifest_verified_files=len(manifest["files"]),
        counts=payload["audit"],
        btc_reconciliation=btc,
        source_enrichment=payload["source_audit"],
        technical_enrichment=technical_audit,
        communications=False,
        photos=False,
        analysis="Evidence-grounded descriptive analysis; no ranking evaluation",
        editor=payload.get(
            "editor_audit",
            dict(
                status="not_run", validated_users=0, total_users=len(payload["users"])
            ),
        ),
        followup=payload["research_audit"],
    )
    write_reports(payload, output)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("Private UI:", output / "index.html")


if __name__ == "__main__":
    main()
