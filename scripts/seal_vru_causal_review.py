#!/usr/bin/env python3
"""Seal offline VRU ball-hand-rim decisions against a review plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_review import (  # noqa: E402
    seal_vru_causal_review,
    verify_vru_causal_review_plan,
)

DECISION_SCHEMA = "agu.vru-causal-codex-decisions.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--expected-plan-artifact-sha256",
        help="externally recorded plan SHA-256; required for selection-bound plans",
    )
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seal_review(
    *,
    plan_path: Path,
    decisions_path: Path,
    expected_plan_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    plan = verify_vru_causal_review_plan(
        _read_json(plan_path),
        expected_artifact_sha256=expected_plan_artifact_sha256,
    )
    decisions = _read_json(decisions_path)
    if (
        decisions.get("schema_version") != DECISION_SCHEMA
        or decisions.get("runtime_consumable") is not False
        or decisions.get("codex_runtime_answer_used") is not False
        or decisions.get("labels_hidden_from_reviewer") is not True
        or decisions.get("plan_sha256") != plan["artifact_sha256"]
        or not isinstance(decisions.get("reviews"), list)
    ):
        raise ValueError("VRU causal decision provenance is invalid")
    return seal_vru_causal_review(
        {
            "reviewer": str(decisions.get("reviewer") or "codex_offline_training_annotation"),
            "reviews": decisions["reviews"],
        },
        plan=plan,
        expected_plan_artifact_sha256=expected_plan_artifact_sha256,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _require_output_disjoint_from_inputs(
    output_path: Path,
    *,
    input_paths: tuple[Path, ...],
) -> tuple[Path, tuple[Path, ...]]:
    resolved_output = output_path.resolve()
    resolved_inputs = tuple(path.resolve() for path in input_paths)
    if resolved_output in set(resolved_inputs):
        raise ValueError("output path must not alias a plan or decisions input path")
    return resolved_output, resolved_inputs


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    output_path, input_paths = _require_output_disjoint_from_inputs(
        args.output,
        input_paths=(args.plan, args.decisions),
    )
    plan_path, decisions_path = input_paths
    artifact = seal_review(
        plan_path=plan_path,
        decisions_path=decisions_path,
        expected_plan_artifact_sha256=args.expected_plan_artifact_sha256,
    )
    _write_json_atomic(output_path, artifact)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "reviews": len(artifact["reviews"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
