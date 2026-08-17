#!/usr/bin/env python3
"""Screen frozen DEIM-M on the sealed LAL-BOS detector review set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ball_candidate_review import canonical_sha256, verify_artifact  # noqa: E402
from app.analysis.deim_basketball_detector import (  # noqa: E402
    DEIM_SCREEN_THRESHOLDS,
    DeimBasketballDetector,
    build_deim_external_screen,
)

_MAX_JSON_BYTES = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-artifact-sha256", required=True)
    parser.add_argument("--expected-plan-file-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--expected-review-artifact-sha256", required=True)
    parser.add_argument("--expected-review-file-sha256", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--expected-video-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _paths_alias(left: Path, right: Path) -> bool:
    if left == right or str(left).casefold() == str(right).casefold():
        return True
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except OSError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def _resolve_disjoint_paths(
    *,
    output_path: Path,
    input_paths: Sequence[Path],
) -> tuple[Path, tuple[Path, ...]]:
    resolved_output = output_path.resolve()
    resolved_inputs = tuple(path.resolve() for path in input_paths)
    for index, input_path in enumerate(resolved_inputs):
        if _paths_alias(resolved_output, input_path):
            raise ValueError("output path must not alias any input path")
        if any(_paths_alias(input_path, prior) for prior in resolved_inputs[:index]):
            raise ValueError("input paths must be unique and non-aliased")
    return resolved_output, resolved_inputs


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is forbidden")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    size = path.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise ValueError("JSON input size is outside the allowed range")
    encoded = path.read_bytes()
    if len(encoded) != size:
        raise ValueError("JSON input changed while it was read")
    payload = json.loads(
        encoded,
        object_pairs_hook=_object_no_duplicates,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON input must contain an object")
    _reject_nonfinite(payload)
    return payload, hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> tuple[Any, str]:
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("candidate frame index must be a non-negative integer")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"could not decode frame {frame_index}")
    return frame, hashlib.sha256(frame.tobytes()).hexdigest()


def _screen(
    *,
    model_path: Path,
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
    video_path: Path,
    model_sha256: str,
    video_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    verified_plan = verify_artifact(plan)
    verified_review = verify_artifact(review)
    if verified_review.get("plan_sha256") != verified_plan["artifact_sha256"]:
        raise ValueError("review is not bound to plan")
    candidates = verified_plan.get("candidates")
    decisions = verified_review.get("decisions")
    if not isinstance(candidates, list) or not isinstance(decisions, list):
        raise ValueError("review inputs are malformed")
    candidate_map = {row.get("candidate_id"): row for row in candidates if isinstance(row, dict)}
    decision_map = {row.get("candidate_id"): row.get("decision") for row in decisions if isinstance(row, dict)}
    if len(candidate_map) != len(candidates) or len(decision_map) != len(decisions):
        raise ValueError("review inputs contain duplicate or malformed candidates")
    if set(candidate_map) != set(decision_map):
        raise ValueError("review does not exactly cover plan candidates")

    detector = DeimBasketballDetector(model_path)
    predictions: dict[str, list[dict[str, Any]]] = {}
    pixels_sha: dict[str, str] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video {video_path}")
    try:
        for candidate_id, candidate in candidate_map.items():
            decision = decision_map[candidate_id]
            if decision == "uncertain":
                continue
            frame, decoded_sha = _read_frame(cap, candidate["frame"])
            expected_pixels_sha = candidate.get("frame_pixels_sha256")
            if expected_pixels_sha != decoded_sha:
                raise ValueError(f"decoded frame receipt mismatch for {candidate_id}")
            predictions[candidate_id] = detector.predict(
                frame,
                confidence_floor=DEIM_SCREEN_THRESHOLDS[0],
            )
            pixels_sha[candidate_id] = decoded_sha
    finally:
        cap.release()

    artifact = build_deim_external_screen(
        verified_plan,
        verified_review,
        predictions,
        model_sha256=model_sha256,
        raw_video_sha256=video_sha256,
        iou_threshold=0.25,
    )
    return artifact, pixels_sha


def main() -> int:
    args = parse_args()
    output_path, resolved_inputs = _resolve_disjoint_paths(
        output_path=args.output,
        input_paths=(args.model, args.plan, args.review, args.video),
    )
    model_path, plan_path, review_path, video_path = resolved_inputs
    expected_model = _require_sha256(args.expected_model_sha256, field="expected model SHA-256")
    expected_plan = _require_sha256(
        args.expected_plan_artifact_sha256,
        field="expected plan artifact SHA-256",
    )
    expected_plan_file = _require_sha256(
        args.expected_plan_file_sha256,
        field="expected plan file SHA-256",
    )
    expected_review = _require_sha256(
        args.expected_review_artifact_sha256,
        field="expected review artifact SHA-256",
    )
    expected_review_file = _require_sha256(
        args.expected_review_file_sha256,
        field="expected review file SHA-256",
    )
    expected_video = _require_sha256(args.expected_video_sha256, field="expected video SHA-256")

    plan, plan_file_sha = _read_json_bytes(plan_path)
    review, review_file_sha = _read_json_bytes(review_path)
    if plan_file_sha != expected_plan_file or review_file_sha != expected_review_file:
        raise ValueError("plan or review file receipt mismatch")
    if verify_artifact(plan)["artifact_sha256"] != expected_plan:
        raise ValueError("plan artifact receipt mismatch")
    if verify_artifact(review)["artifact_sha256"] != expected_review:
        raise ValueError("review artifact receipt mismatch")
    model_sha = _file_sha256(model_path)
    video_sha = _file_sha256(video_path)
    if model_sha != expected_model:
        raise ValueError("model file receipt mismatch")
    if video_sha != expected_video:
        raise ValueError("video file receipt mismatch")

    artifact, pixel_receipts = _screen(
        model_path=model_path,
        plan=plan,
        review=review,
        video_path=video_path,
        model_sha256=model_sha,
        video_sha256=video_sha,
    )
    artifact.pop("artifact_sha256")
    artifact["input_file_receipts"] = {
        "plan_sha256": plan_file_sha,
        "review_sha256": review_file_sha,
    }
    artifact["decoded_frame_pixels_sha256"] = pixel_receipts
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    _write_json_atomic(output_path, artifact)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "metrics_by_threshold": artifact["metrics_by_threshold"],
                "meets_followup_gate": artifact["meets_followup_gate"],
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
