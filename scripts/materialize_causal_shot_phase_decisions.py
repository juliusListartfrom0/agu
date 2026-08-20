#!/usr/bin/env python3
"""Materialize audited Codex JSONL decisions into the sealed-review template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.build_causal_shot_phase_review import DECISION_TEMPLATE_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def materialize_decisions(
    *,
    template_path: Path,
    jsonl_path: Path,
) -> dict[str, Any]:
    template = _read_object(template_path)
    if (
        template.get("schema_version") != DECISION_TEMPLATE_SCHEMA
        or template.get("runtime_consumable") is not False
        or template.get("codex_runtime_answer_used") is not False
        or template.get("labels_hidden_from_reviewer") is not True
        or not isinstance(template.get("decisions"), list)
    ):
        raise ValueError("causal phase decision template is invalid")

    expected_ids = [
        str(row.get("phase_review_id") or "")
        for row in template["decisions"]
    ]
    if (
        not expected_ids
        or any(not value for value in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise ValueError("causal phase decision template IDs are invalid")

    rows = _read_jsonl(jsonl_path)
    observed_ids = [
        str(row.get("phase_review_id") or "") for row in rows
    ]
    if (
        len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != set(expected_ids)
    ):
        raise ValueError(
            "causal phase JSONL decisions must exactly cover the template"
        )
    by_id = dict(zip(observed_ids, rows, strict=True))
    return {
        **template,
        "decisions": [by_id[value] for value in expected_ids],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(
                f"expected a JSON object at {path.name}:{line_number}"
            )
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    materialized = materialize_decisions(
        template_path=args.template,
        jsonl_path=args.jsonl,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(materialized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decisions": len(materialized["decisions"]),
                "plan_sha256": materialized["plan_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
