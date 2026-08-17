#!/usr/bin/env python3
"""Evaluate independent VLM predictions after the frozen raw-frame run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import (  # noqa: E402
    evaluate_independent_shot_vlm,
    verify_independent_shot_vlm_annotation_plan,
    verify_independent_shot_vlm_plan,
    verify_independent_shot_vlm_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--expected-plan-sha256",
        help="plan internal SHA-256 frozen independently before evaluation",
    )
    parser.add_argument(
        "--annotation-plan",
        type=Path,
        help="Frozen source plan that owns annotation hashes for a derived plan.",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--expected-prediction-sha256",
        help=("prediction file SHA-256 frozen independently before truth is opened; legacy v1 remains non-promotable"),
    )
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_output_disjoint_from_inputs(
    output_path: Path,
    *,
    input_paths: tuple[Path, ...],
) -> tuple[Path, tuple[Path, ...]]:
    resolved_output = output_path.resolve()
    resolved_inputs = tuple(path.resolve() for path in input_paths)
    for index, input_path in enumerate(resolved_inputs):
        if _paths_alias(resolved_output, input_path):
            raise ValueError("output path must not alias a plan, prediction, or annotation input path")
        if any(_paths_alias(input_path, prior) for prior in resolved_inputs[:index]):
            raise ValueError("input paths contain a duplicate or aliased file")
    return resolved_output, resolved_inputs


def _paths_alias(left: Path, right: Path) -> bool:
    if left == right or str(left).casefold() == str(right).casefold():
        return True
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except OSError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
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
    ordered_inputs = (
        args.plan,
        *((args.annotation_plan,) if args.annotation_plan is not None else ()),
        args.predictions,
        *args.annotation,
    )
    output_path, resolved_inputs = _require_output_disjoint_from_inputs(
        args.output,
        input_paths=ordered_inputs,
    )
    input_index = 0
    plan_path = resolved_inputs[input_index]
    input_index += 1
    annotation_plan_path: Path | None = None
    if args.annotation_plan is not None:
        annotation_plan_path = resolved_inputs[input_index]
        input_index += 1
    predictions_path = resolved_inputs[input_index]
    annotation_paths = resolved_inputs[input_index + 1 :]
    plan = verify_independent_shot_vlm_plan(_read_json_bytes(plan_path))
    if args.expected_plan_sha256 is not None:
        expected_plan_sha256 = _require_sha256(
            args.expected_plan_sha256,
            field="expected plan SHA-256",
        )
        if plan["plan_sha256"] != expected_plan_sha256:
            raise ValueError("expected plan SHA-256 does not match")
    else:
        expected_plan_sha256 = None
    annotation_plan = plan
    if annotation_plan_path is not None:
        annotation_plan = verify_independent_shot_vlm_annotation_plan(
            plan,
            _read_json_bytes(annotation_plan_path),
        )
    prediction_bytes = _read_bytes(predictions_path)
    prediction_file_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    expected_prediction_file_sha256: str | None = None
    if args.expected_prediction_sha256 is not None:
        expected_prediction_file_sha256 = _require_sha256(
            args.expected_prediction_sha256,
            field="expected prediction file SHA-256",
        )
        if prediction_file_sha256 != expected_prediction_file_sha256:
            raise ValueError("expected prediction file SHA-256 does not match")
    prediction_payload = _json_object_from_bytes(prediction_bytes)
    verified_prediction = verify_independent_shot_vlm_predictions(prediction_payload)
    expected_prediction_artifact_sha256: str | None = None
    if args.expected_prediction_sha256 is not None:
        expected_prediction_artifact_sha256 = verified_prediction["artifact_sha256"]
    prediction_payload = verified_prediction
    expected_annotation_sha = set(annotation_plan["training_annotation_sha256"])
    truth: dict[tuple[str, str, str], bool] = {}
    supplied_sha = set()
    for path in annotation_paths:
        annotation_bytes = _read_bytes(path)
        supplied_sha.add(hashlib.sha256(annotation_bytes).hexdigest())
        payload = _json_object_from_bytes(annotation_bytes)
        source_sha = str(payload.get("source_video_sha256") or "")
        candidate_sha = str(payload.get("candidate_bundle_sha256") or "")
        for row in payload.get("examples", []):
            value = row.get("event_present")
            if not isinstance(value, bool):
                raise ValueError("shot-validity truth must be boolean")
            key = (source_sha, candidate_sha, str(row.get("event_id") or ""))
            if key in truth:
                raise ValueError("duplicate truth key across annotation inputs")
            truth[key] = value
    if supplied_sha != expected_annotation_sha:
        raise ValueError("annotations do not exactly match the frozen plan")
    selected_keys = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
        for row in plan["examples"]
    }
    evaluation = evaluate_independent_shot_vlm(
        plan=plan,
        predictions=prediction_payload,
        truth={key: truth[key] for key in selected_keys},
        expected_plan_sha256=expected_plan_sha256,
        expected_prediction_artifact_sha256=expected_prediction_artifact_sha256,
        prediction_file_sha256=prediction_file_sha256,
        expected_prediction_file_sha256=expected_prediction_file_sha256,
    )
    _write_json_atomic(output_path, evaluation)
    print(json.dumps({"output": str(output_path), **evaluation["metrics"]}))
    return 0 if evaluation["metrics"]["promotion_eligible"] else 2


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _json_object_from_bytes(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation input must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("evaluation input must be a JSON object")
    _reject_nonfinite_json_numbers(value)
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested_value in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = nested_value
    return value


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is invalid: {value}")


def _reject_nonfinite_json_numbers(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is invalid")
    if isinstance(value, Mapping):
        for nested_value in value.values():
            _reject_nonfinite_json_numbers(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _reject_nonfinite_json_numbers(nested_value)


def _read_json_bytes(path: Path) -> dict[str, Any]:
    return _json_object_from_bytes(_read_bytes(path))


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
