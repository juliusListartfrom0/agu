#!/usr/bin/env python3
"""Seal offline Codex visual-state decisions and derive training corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.pbp_visual_state_review import (  # noqa: E402
    derive_visual_state_label_corrections,
    seal_visual_state_review,
    verify_visual_state_review_plan,
)
from scripts.build_pbp_visual_state_review import (  # noqa: E402
    DECISION_TEMPLATE_SCHEMA,
    SHEET_SCHEMA,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-artifact", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sheet-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--corrections-output", type=Path, required=True)
    return parser.parse_args()


def seal_review(
    *,
    embedding_artifact_path: Path,
    plan_path: Path,
    sheet_manifest_path: Path,
    decisions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    embeddings = _read_json(embedding_artifact_path)
    plan = verify_visual_state_review_plan(_read_json(plan_path))
    sheets = _verify_sheet_manifest(
        _read_json(sheet_manifest_path),
        manifest_path=sheet_manifest_path,
        plan=plan,
    )
    decisions = _read_json(decisions_path)
    if (
        decisions.get("schema_version") != DECISION_TEMPLATE_SCHEMA
        or decisions.get("runtime_consumable") is not False
        or decisions.get("codex_runtime_answer_used") is not False
        or decisions.get("plan_sha256") != plan["artifact_sha256"]
    ):
        raise ValueError("visual-state Codex decisions have invalid provenance")

    planned_ids = {str(row["review_id"]) for row in plan["examples"]}
    sheet_ids = [
        str(review_id)
        for record in sheets["records"]
        for review_id in record["review_ids"]
    ]
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("visual-state Codex decisions require a list")
    decision_ids = [str(row.get("review_id") or "") for row in raw_decisions]
    if (
        len(sheet_ids) != len(set(sheet_ids))
        or len(decision_ids) != len(set(decision_ids))
        or set(sheet_ids) != planned_ids
        or set(decision_ids) != planned_ids
    ):
        raise ValueError("sheets and decisions must exactly cover the review plan")

    review = seal_visual_state_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "review_sheet_artifact_sha256": sheets["artifact_sha256"],
            "reviewer": "codex_offline_visual_review",
            "reviews": raw_decisions,
        },
        plan=plan,
    )
    corrections = derive_visual_state_label_corrections(
        embedding_artifact=embeddings,
        plan=plan,
        review=review,
    )
    return review, corrections


def _verify_sheet_manifest(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if claimed != _canonical_sha256(artifact):
        raise ValueError("visual-state review sheet manifest hash mismatch")
    if (
        artifact.get("schema_version") != SHEET_SCHEMA
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("labels_hidden_from_reviewer") is not True
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
    ):
        raise ValueError("visual-state review sheets have invalid provenance")
    records = artifact.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("visual-state review sheets require records")
    root = manifest_path.parent.resolve()
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("review_ids"), list):
            raise ValueError("invalid visual-state review sheet record")
        sheet_path = (root / str(row.get("sheet") or "")).resolve()
        if not sheet_path.is_relative_to(root) or not sheet_path.is_file():
            raise ValueError("visual-state review sheet is outside the package")
        if _file_sha256(sheet_path) != row.get("sheet_sha256"):
            raise ValueError("visual-state review sheet hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    review, corrections = seal_review(
        embedding_artifact_path=args.embedding_artifact,
        plan_path=args.plan,
        sheet_manifest_path=args.sheet_manifest,
        decisions_path=args.decisions,
    )
    _write_json(args.review_output, review)
    _write_json(args.corrections_output, corrections)
    print(
        json.dumps(
            {
                "review_output": str(args.review_output),
                "review_sha256": review["artifact_sha256"],
                "corrections_output": str(args.corrections_output),
                "corrections_sha256": corrections["artifact_sha256"],
                "summary": corrections["summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
