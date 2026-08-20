"""Training-only EBQwen visual-state LoRA data contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence, Set
from pathlib import PurePosixPath
from typing import Any

LORA_DATASET_SCHEMA = "agu.ebqwen-visual-state-lora-dataset.v1"
LORA_PROMPT_VERSION = "agu_visual_state_binary_v1"

_STATE_LABELS = {
    "free_throw": "FREE_THROW",
    "field_goal": "LIVE_FIELD_GOAL",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def seal_lora_dataset_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = LORA_DATASET_SCHEMA
    artifact["purpose"] = "offline_ebqwen_visual_state_lora_training"
    artifact["runtime_consumable"] = False
    artifact["truth_used_for_training_only"] = True
    artifact["codex_runtime_answer_used"] = False
    artifact["prompt_version"] = LORA_PROMPT_VERSION
    artifact.pop("artifact_sha256", None)
    _validate_lora_dataset_manifest(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_lora_dataset_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_lora_dataset_manifest(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("EBQwen LoRA dataset hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def contact_sheet_row_bounds(
    *,
    image_height: int,
    row_count: int,
    row_index: int,
) -> tuple[int, int]:
    """Return deterministic pixel bounds for one vertically stacked row."""

    if (
        isinstance(image_height, bool)
        or isinstance(row_count, bool)
        or isinstance(row_index, bool)
        or image_height < row_count
        or row_count < 1
        or row_index < 0
        or row_index >= row_count
    ):
        raise ValueError("invalid contact-sheet row geometry")
    start = round(row_index * image_height / row_count)
    end = round((row_index + 1) * image_height / row_count)
    return start, end


def select_lora_examples(
    *,
    plan_examples: Sequence[Mapping[str, Any]],
    correction_decisions: Sequence[Mapping[str, Any]],
    sheet_records: Sequence[Mapping[str, Any]],
    sealed_blind_video_sha256s: Set[str],
) -> list[dict[str, Any]]:
    """Bind resolved corrections to their label-free contact-sheet rows."""

    plan_by_id = _unique_rows(plan_examples, field="review_id", kind="plan")
    correction_by_id = _unique_rows(
        correction_decisions,
        field="review_id",
        kind="correction",
    )
    if set(correction_by_id) != set(plan_by_id):
        raise ValueError("corrections must exactly cover the review plan")

    sheet_by_id: dict[str, tuple[Mapping[str, Any], int]] = {}
    for record in sheet_records:
        review_ids = record.get("review_ids")
        if not isinstance(review_ids, list) or not review_ids:
            raise ValueError("sheet records require review IDs")
        for row_index, value in enumerate(review_ids):
            review_id = str(value or "")
            if not review_id or review_id in sheet_by_id:
                raise ValueError("duplicate or invalid sheet review ID")
            sheet_by_id[review_id] = (record, row_index)
    if set(sheet_by_id) != set(plan_by_id):
        raise ValueError("sheets must exactly cover the review plan")

    examples = []
    for planned in plan_examples:
        review_id = str(planned["review_id"])
        source_sha = str(planned.get("source_video_sha256") or "")
        if source_sha in sealed_blind_video_sha256s:
            raise ValueError("sealed blind video cannot enter LoRA training")
        correction = correction_by_id[review_id]
        if (
            str(correction.get("source_video_sha256") or "") != source_sha
            or str(correction.get("event_id") or "")
            != str(planned.get("event_id") or "")
        ):
            raise ValueError("correction does not match its planned example")
        corrected_state = correction.get("corrected_state")
        if corrected_state is None:
            continue
        label = _STATE_LABELS.get(str(corrected_state))
        if label is None:
            raise ValueError("unsupported corrected visual state")
        sheet, row_index = sheet_by_id[review_id]
        examples.append(
            {
                "review_id": review_id,
                "source_video_sha256": source_sha,
                "event_id": str(planned["event_id"]),
                "sheet": str(sheet["sheet"]),
                "sheet_sha256": str(sheet["sheet_sha256"]),
                "sheet_row_index": row_index,
                "label": label,
            }
        )
    if not examples:
        raise ValueError("LoRA training requires resolved visual-state examples")
    return examples


def _unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    kind: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(field) or "")
        if not value or value in output:
            raise ValueError(f"duplicate or invalid {kind} {field}")
        output[value] = row
    return output


def _validate_lora_dataset_manifest(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != LORA_DATASET_SCHEMA:
        raise ValueError("unsupported EBQwen LoRA dataset schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("truth_used_for_training_only") is not True
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("prompt_version") != LORA_PROMPT_VERSION
    ):
        raise ValueError("invalid EBQwen LoRA dataset provenance")
    for field in (
        "review_plan_sha256",
        "sheet_manifest_sha256",
        "label_corrections_sha256",
    ):
        _require_sha256(artifact.get(field), field=field)
    blind_values = artifact.get("sealed_blind_video_sha256s")
    if not isinstance(blind_values, list):
        raise ValueError("sealed blind video hashes are required")
    blind = {_require_sha256(value, field="blind video") for value in blind_values}
    if len(blind) != len(blind_values):
        raise ValueError("duplicate sealed blind video hash")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("EBQwen LoRA dataset requires examples")
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("invalid EBQwen LoRA example")
        review_id = str(row.get("review_id") or "")
        source = _require_sha256(
            row.get("source_video_sha256"),
            field="source video",
        )
        image = str(row.get("image") or "")
        image_path = PurePosixPath(image)
        if (
            not review_id
            or review_id in seen_ids
            or source in blind
            or not str(row.get("event_id") or "")
            or image_path.is_absolute()
            or ".." in image_path.parts
            or image in seen_images
            or row.get("label") not in set(_STATE_LABELS.values())
        ):
            raise ValueError("invalid or duplicate EBQwen LoRA example")
        _require_sha256(row.get("image_sha256"), field="training image")
        seen_ids.add(review_id)
        seen_images.add(image)


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
