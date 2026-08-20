#!/usr/bin/env python3
"""Build dense label-hidden shot-phase sheets for offline Codex annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.causal_shot_phase_review import (  # noqa: E402
    CAUSAL_PHASE_OFFSETS_SECONDS,
    build_causal_shot_phase_review_plan,
    verify_causal_shot_phase_review_plan,
)
from app.analysis.pbp_visual_state_review import (  # noqa: E402
    verify_visual_state_review_plan,
)
from scripts.build_pbp_visual_state_manifest import (  # noqa: E402
    load_sealed_blind_hashes,
)

SHEET_SCHEMA = "agu.causal-shot-phase-review-sheets.v1"
DECISION_TEMPLATE_SCHEMA = "agu.causal-shot-phase-codex-decisions.v1"
GRID_COLUMNS = 6
GRID_ROWS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-state-plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sealed-blind-acquisition", type=Path)
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--examples-per-sheet", type=int, default=2)
    return parser.parse_args()


def build_causal_phase_review_package(
    *,
    visual_state_plan_path: Path,
    video_paths: list[Path],
    output_dir: Path,
    panel_width: int = 320,
    examples_per_sheet: int = 2,
    sealed_blind_acquisition_path: Path | None = None,
) -> dict[str, Any]:
    base = verify_visual_state_review_plan(_read_json(visual_state_plan_path))
    additional_blind = (
        load_sealed_blind_hashes(sealed_blind_acquisition_path)
        if sealed_blind_acquisition_path is not None
        else ()
    )
    plan = build_causal_shot_phase_review_plan(
        base,
        additional_sealed_blind_video_sha256s=additional_blind,
    )
    return render_causal_phase_review_package(
        plan=plan,
        video_paths=video_paths,
        output_dir=output_dir,
        panel_width=panel_width,
        examples_per_sheet=examples_per_sheet,
    )


def render_causal_phase_review_package(
    *,
    plan: dict[str, Any],
    video_paths: list[Path],
    output_dir: Path,
    panel_width: int = 320,
    examples_per_sheet: int = 2,
) -> dict[str, Any]:
    verified_plan = verify_causal_shot_phase_review_plan(plan)
    if panel_width < 128 or examples_per_sheet < 1:
        raise ValueError("invalid causal phase rendering settings")
    videos = {_file_sha256(path): path for path in video_paths}
    if len(videos) != len(video_paths):
        raise ValueError("causal phase review videos must have unique hashes")
    if set(videos) != set(verified_plan["source_video_sha256s"]):
        raise ValueError("causal phase review videos must exactly match the plan")

    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir = output_dir / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plan.json").write_text(
        json.dumps(verified_plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    examples_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in verified_plan["examples"]:
        examples_by_source[str(row["source_video_sha256"])].append(row)
    records = []
    for source_index, source_sha in enumerate(
        sorted(examples_by_source),
        start=1,
    ):
        capture = cv2.VideoCapture(str(videos[source_sha]))
        if not capture.isOpened():
            raise ValueError(
                f"cannot open causal phase video: {videos[source_sha].name}"
            )
        try:
            examples = examples_by_source[source_sha]
            for batch_index in range(0, len(examples), examples_per_sheet):
                batch = examples[
                    batch_index : batch_index + examples_per_sheet
                ]
                rendered = [
                    _render_example_grid(
                        capture,
                        row,
                        panel_width=panel_width,
                    )
                    for row in batch
                ]
                sheet = cv2.vconcat(rendered)
                sheet_name = (
                    f"source-{source_index:02d}-"
                    f"sheet-{batch_index // examples_per_sheet + 1:03d}.jpg"
                )
                sheet_path = sheet_dir / sheet_name
                if not cv2.imwrite(
                    str(sheet_path),
                    sheet,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 94],
                ):
                    raise RuntimeError(
                        f"failed to write causal phase sheet: {sheet_name}"
                    )
                records.append(
                    {
                        "sheet": str(sheet_path.relative_to(output_dir)),
                        "sheet_sha256": _file_sha256(sheet_path),
                        "source_video_sha256": source_sha,
                        "phase_review_ids": [
                            str(row["phase_review_id"]) for row in batch
                        ],
                    }
                )
        finally:
            capture.release()

    manifest: dict[str, Any] = {
        "schema_version": SHEET_SCHEMA,
        "purpose": "codex_offline_causal_shot_phase_training_annotation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "plan_sha256": verified_plan["artifact_sha256"],
        "rendering": {
            "panel_width": panel_width,
            "grid_columns": GRID_COLUMNS,
            "grid_rows_per_example": GRID_ROWS,
            "examples_per_sheet": examples_per_sheet,
            "raw_frames_only": True,
            "labels_or_predictions_rendered": False,
        },
        "records": records,
    }
    manifest["artifact_sha256"] = _canonical_sha256(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_decision_template(output_dir, verified_plan)
    return manifest


def _render_example_grid(
    capture: cv2.VideoCapture,
    example: dict[str, Any],
    *,
    panel_width: int,
) -> np.ndarray:
    frame_indexes = [int(value) for value in example["frame_indexes"]]
    if len(frame_indexes) != len(CAUSAL_PHASE_OFFSETS_SECONDS):
        raise ValueError("causal phase frames do not match the offset protocol")
    panels = []
    for position, (frame_index, offset) in enumerate(
        zip(
            frame_indexes,
            CAUSAL_PHASE_OFFSETS_SECONDS,
            strict=True,
        )
    ):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode causal phase frame {frame_index}")
        panel = _letterbox(frame, width=panel_width)
        label_height = min(32, panel.shape[0])
        cv2.rectangle(
            panel,
            (0, 0),
            (panel_width, label_height),
            (0, 0, 0),
            -1,
        )
        offset_label = f"{offset:+.2f}s p{position:02d} f{frame_index}"
        label = (
            f"{example['phase_review_id']} {offset_label}"
            if position == 0
            else offset_label
        )
        cv2.putText(
            panel,
            label,
            (5, min(22, label_height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    rows = [
        cv2.hconcat(panels[index : index + GRID_COLUMNS])
        for index in range(0, len(panels), GRID_COLUMNS)
    ]
    if len(rows) != GRID_ROWS:
        raise ValueError("causal phase grid protocol is invalid")
    return cv2.vconcat(rows)


def _letterbox(frame: np.ndarray, *, width: int) -> np.ndarray:
    target_height = max(72, round(width * 9 / 16))
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros((target_height, width, 3), dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (target_height - resized_height) // 2
    panel[top : top + resized_height, left : left + resized_width] = resized
    return panel


def _write_decision_template(
    output_dir: Path,
    plan: dict[str, Any],
) -> None:
    payload = {
        "schema_version": DECISION_TEMPLATE_SCHEMA,
        "purpose": "codex_offline_causal_shot_phase_training_annotation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "plan_sha256": plan["artifact_sha256"],
        "anchor_offsets_seconds": list(CAUSAL_PHASE_OFFSETS_SECONDS),
        "allowed_formation_states": [
            "free_throw_setup",
            "live_play",
            "stoppage_other",
            "uncertain",
        ],
        "allowed_broadcast_contexts": [
            "live_action",
            "replay",
            "non_action",
            "uncertain",
        ],
        "allowed_shot_sequences": [
            "complete",
            "partial",
            "not_a_shot",
            "uncertain",
        ],
        "allowed_outcomes": [
            "made",
            "missed",
            "unknown",
            "not_applicable",
        ],
        "allowed_confidence": ["high", "medium", "low"],
        "decisions": [
            {
                "phase_review_id": row["phase_review_id"],
                "formation_state": None,
                "broadcast_context": None,
                "shot_sequence": None,
                "release_position": None,
                "rim_arrival_position": None,
                "outcome": None,
                "confidence": None,
                "notes": "",
            }
            for row in plan["examples"]
        ],
    }
    (output_dir / "decisions.template.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    manifest = build_causal_phase_review_package(
        visual_state_plan_path=args.visual_state_plan,
        video_paths=args.video,
        output_dir=args.output_dir,
        panel_width=args.panel_width,
        examples_per_sheet=args.examples_per_sheet,
        sealed_blind_acquisition_path=args.sealed_blind_acquisition,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "sheets": len(manifest["records"]),
                "examples": sum(
                    len(row["phase_review_ids"])
                    for row in manifest["records"]
                ),
                "artifact_sha256": manifest["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
