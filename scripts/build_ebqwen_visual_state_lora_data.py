#!/usr/bin/env python3
"""Materialize hash-bound contact-sheet rows for EBQwen LoRA training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ebqwen_visual_state_lora import (  # noqa: E402
    contact_sheet_row_bounds,
    seal_lora_dataset_manifest,
    select_lora_examples,
)
from app.analysis.pbp_visual_state_review import (  # noqa: E402
    verify_visual_state_review_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-plan", type=Path, required=True)
    parser.add_argument("--sheet-manifest", type=Path, required=True)
    parser.add_argument("--label-corrections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_lora_data(
    *,
    review_plan_path: Path,
    sheet_manifest_path: Path,
    label_corrections_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    plan = verify_visual_state_review_plan(_read_json(review_plan_path))
    sheets = _verify_canonical_artifact(
        _read_json(sheet_manifest_path),
        expected_schema="agu.pbp-visual-state-review-sheets.v1",
    )
    corrections = _verify_canonical_artifact(
        _read_json(label_corrections_path),
        expected_schema="agu.pbp-visual-state-label-corrections.v1",
    )
    if (
        sheets.get("plan_sha256") != plan["artifact_sha256"]
        or corrections.get("plan_sha256") != plan["artifact_sha256"]
        or sheets.get("runtime_consumable") is not False
        or sheets.get("labels_hidden_from_reviewer") is not True
        or corrections.get("runtime_consumable") is not False
        or corrections.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("LoRA sources do not share the verified review plan")
    selected = select_lora_examples(
        plan_examples=plan["examples"],
        correction_decisions=corrections["decisions"],
        sheet_records=sheets["records"],
        sealed_blind_video_sha256s=set(
            plan["sealed_blind_video_sha256s"]
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    sheet_root = sheet_manifest_path.parent.resolve()
    record_by_path = {
        str(record["sheet"]): record for record in sheets["records"]
    }
    cache: dict[str, Any] = {}
    materialized = []
    for row in selected:
        sheet_name = str(row["sheet"])
        record = record_by_path[sheet_name]
        source_path = (sheet_root / sheet_name).resolve()
        if (
            not source_path.is_relative_to(sheet_root)
            or not source_path.is_file()
            or _file_sha256(source_path) != row["sheet_sha256"]
        ):
            raise ValueError("review sheet is absent or hash-mismatched")
        sheet = cache.get(sheet_name)
        if sheet is None:
            sheet = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if sheet is None:
                raise ValueError("cannot decode review sheet")
            cache[sheet_name] = sheet
        start, end = contact_sheet_row_bounds(
            image_height=int(sheet.shape[0]),
            row_count=len(record["review_ids"]),
            row_index=int(row["sheet_row_index"]),
        )
        crop = sheet[start:end].copy()
        relative_image = f"images/{row['review_id']}.jpg"
        image_path = output_dir / relative_image
        if not cv2.imwrite(
            str(image_path),
            crop,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        ):
            raise RuntimeError("failed to write LoRA training image")
        materialized.append(
            {
                "review_id": row["review_id"],
                "source_video_sha256": row["source_video_sha256"],
                "event_id": row["event_id"],
                "image": relative_image,
                "image_sha256": _file_sha256(image_path),
                "label": row["label"],
            }
        )
    manifest = seal_lora_dataset_manifest(
        {
            "review_plan_sha256": plan["artifact_sha256"],
            "sheet_manifest_sha256": sheets["artifact_sha256"],
            "label_corrections_sha256": corrections["artifact_sha256"],
            "sealed_blind_video_sha256s": plan[
                "sealed_blind_video_sha256s"
            ],
            "examples": materialized,
        }
    )
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _verify_canonical_artifact(
    payload: dict[str, Any],
    *,
    expected_schema: str,
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != expected_schema:
        raise ValueError("unexpected source artifact schema")
    if claimed != _canonical_sha256(artifact):
        raise ValueError("source artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    manifest = build_lora_data(
        review_plan_path=args.review_plan,
        sheet_manifest_path=args.sheet_manifest,
        label_corrections_path=args.label_corrections,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "examples": len(manifest["examples"]),
                "artifact_sha256": manifest["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
