#!/usr/bin/env python3
"""Verify dense phase sheets and seal offline Codex training annotations."""

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

from app.analysis.causal_shot_phase_review import (  # noqa: E402
    seal_causal_shot_phase_review,
    verify_causal_shot_phase_review_plan,
)
from scripts.build_causal_shot_phase_review import (  # noqa: E402
    DECISION_TEMPLATE_SCHEMA,
    SHEET_SCHEMA,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sheet-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seal_review(
    *,
    plan_path: Path,
    sheet_manifest_path: Path,
    decisions_path: Path,
) -> dict[str, Any]:
    plan = verify_causal_shot_phase_review_plan(_read_json(plan_path))
    sheet_manifest = _verify_sheet_manifest(
        _read_json(sheet_manifest_path),
        manifest_path=sheet_manifest_path,
        plan=plan,
    )
    decisions = _read_json(decisions_path)
    if (
        decisions.get("schema_version") != DECISION_TEMPLATE_SCHEMA
        or decisions.get("runtime_consumable") is not False
        or decisions.get("codex_runtime_answer_used") is not False
        or decisions.get("labels_hidden_from_reviewer") is not True
        or decisions.get("plan_sha256") != plan["artifact_sha256"]
        or not isinstance(decisions.get("decisions"), list)
    ):
        raise ValueError("causal phase decision provenance is invalid")
    payload = {
        "plan_sha256": plan["artifact_sha256"],
        "sheet_manifest_sha256": sheet_manifest["artifact_sha256"],
        "decisions_sha256": _file_sha256(decisions_path),
        "reviewer": "codex_offline_training_annotation",
        "reviews": decisions["decisions"],
    }
    return seal_causal_shot_phase_review(payload, plan=plan)


def _verify_sheet_manifest(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if claimed != _canonical_sha256(artifact):
        raise ValueError("causal phase sheet manifest hash mismatch")
    rendering = artifact.get("rendering")
    if (
        artifact.get("schema_version") != SHEET_SCHEMA
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("labels_hidden_from_reviewer") is not True
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
        or not isinstance(rendering, dict)
        or rendering.get("raw_frames_only") is not True
        or rendering.get("labels_or_predictions_rendered") is not False
    ):
        raise ValueError("causal phase sheet manifest provenance is invalid")
    records = artifact.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("causal phase sheet manifest requires records")
    plan_by_id = {
        str(row["phase_review_id"]): row for row in plan["examples"]
    }
    observed_ids = []
    root = manifest_path.parent.resolve()
    for row in records:
        source_sha = str(row.get("source_video_sha256") or "")
        phase_ids = row.get("phase_review_ids")
        if not isinstance(phase_ids, list) or not phase_ids:
            raise ValueError("causal phase sheet record is invalid")
        for phase_id in phase_ids:
            planned = plan_by_id.get(str(phase_id))
            if planned is None or planned["source_video_sha256"] != source_sha:
                raise ValueError("causal phase sheet-plan binding is invalid")
            observed_ids.append(str(phase_id))
        sheet = root / str(row.get("sheet") or "")
        resolved_sheet = sheet.resolve()
        if not resolved_sheet.is_relative_to(root) or not resolved_sheet.is_file():
            raise ValueError("causal phase sheet path is invalid")
        if _file_sha256(resolved_sheet) != row.get("sheet_sha256"):
            raise ValueError("causal phase sheet hash mismatch")
    if (
        len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != set(plan_by_id)
    ):
        raise ValueError("causal phase sheets must exactly cover the plan")
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


def main() -> int:
    args = parse_args()
    review = seal_review(
        plan_path=args.plan,
        sheet_manifest_path=args.sheet_manifest,
        decisions_path=args.decisions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reviews": len(review["reviews"]),
                "artifact_sha256": review["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
