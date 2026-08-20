from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.analysis.causal_shot_phase_review import (
    CAUSAL_PHASE_OFFSETS_SECONDS,
    build_causal_shot_phase_review_plan,
)
from app.analysis.pbp_visual_state_frames import (
    DINO_V2_SMALL_BACKBONE,
    PRE_ANCHOR_OFFSETS_SECONDS,
    seal_anchor_state_embedding_artifact,
)
from app.analysis.pbp_visual_state_review import (
    build_visual_state_review_plan,
)
from scripts.build_causal_shot_phase_review import (
    render_causal_phase_review_package,
)
from scripts.materialize_causal_shot_phase_decisions import (
    materialize_decisions,
)
from scripts.seal_causal_shot_phase_review import seal_review


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 36),
    )
    assert writer.isOpened()
    for index in range(100):
        frame = np.full((36, 64, 3), index * 2, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(video_path: Path) -> dict:
    source_sha = _sha256(video_path)
    embeddings = seal_anchor_state_embedding_artifact(
        {
            "purpose": "test",
            "truth_used_for_training_only": True,
            "codex_runtime_answer_used": False,
            "training_manifest_sha256s": ["1" * 64],
            "clock_artifact_sha256s": ["2" * 64],
            "source_video_sha256s": [source_sha],
            "sealed_blind_video_sha256s": ["4" * 64],
            "backbone": DINO_V2_SMALL_BACKBONE,
            "backbone_sha256": "5" * 64,
            "anchor_offsets_seconds": list(PRE_ANCHOR_OFFSETS_SECONDS),
            "examples": [
                {
                    "source_video_sha256": source_sha,
                    "source_video_filename": video_path.name,
                    "candidate_bundle_sha256": "6" * 64,
                    "event_id": "event-1",
                    "state": "field_goal",
                    "anchor_frame": 50,
                    "source_fps": 10.0,
                    "frame_count": 100,
                    "frame_indexes": [10, 20, 30, 40, 50],
                    "embeddings": [
                        [float(index)] * 384 for index in range(5)
                    ],
                }
            ],
        }
    )
    base = build_visual_state_review_plan(embeddings)
    return build_causal_shot_phase_review_plan(base)


def test_render_causal_phase_package_writes_hash_bound_raw_grid(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "development.avi"
    _write_video(video_path)
    plan = _plan(video_path)
    output_dir = tmp_path / "review"

    manifest = render_causal_phase_review_package(
        plan=plan,
        video_paths=[video_path],
        output_dir=output_dir,
        panel_width=128,
        examples_per_sheet=2,
    )

    assert manifest["schema_version"] == (
        "agu.causal-shot-phase-review-sheets.v1"
    )
    assert manifest["runtime_consumable"] is False
    assert manifest["labels_hidden_from_reviewer"] is True
    assert manifest["rendering"] == {
        "panel_width": 128,
        "grid_columns": 6,
        "grid_rows_per_example": 4,
        "examples_per_sheet": 2,
        "raw_frames_only": True,
        "labels_or_predictions_rendered": False,
    }
    record = manifest["records"][0]
    assert record["phase_review_ids"] == ["causal-phase-0001"]
    sheet_path = output_dir / record["sheet"]
    assert record["sheet_sha256"] == _sha256(sheet_path)
    sheet = cv2.imread(str(sheet_path))
    assert sheet.shape[:2] == (4 * 72, 6 * 128)
    saved_plan = json.loads(
        (output_dir / "plan.json").read_text(encoding="utf-8")
    )
    assert saved_plan["artifact_sha256"] == plan["artifact_sha256"]
    template = json.loads(
        (output_dir / "decisions.template.json").read_text(encoding="utf-8")
    )
    assert template["decisions"][0]["release_position"] is None
    assert template["anchor_offsets_seconds"] == list(
        CAUSAL_PHASE_OFFSETS_SECONDS
    )

    decisions = {
        **template,
        "decisions": [
            {
                "phase_review_id": "causal-phase-0001",
                "formation_state": "stoppage_other",
                "broadcast_context": "non_action",
                "shot_sequence": "not_a_shot",
                "release_position": None,
                "rim_arrival_position": None,
                "outcome": "not_applicable",
                "confidence": "high",
                "notes": "No shot is visible in the dense window.",
            }
        ],
    }
    decisions_path = output_dir / "decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    review = seal_review(
        plan_path=output_dir / "plan.json",
        sheet_manifest_path=output_dir / "manifest.json",
        decisions_path=decisions_path,
    )
    assert review["reviews"][0]["release_frame"] is None
    assert review["reviews"][0]["source_video_sha256"] == _sha256(
        video_path
    )

    with sheet_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="sheet hash"):
        seal_review(
            plan_path=output_dir / "plan.json",
            sheet_manifest_path=output_dir / "manifest.json",
            decisions_path=decisions_path,
        )


def test_render_causal_phase_package_requires_exact_video_hashes(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "development.avi"
    other_path = tmp_path / "other.avi"
    _write_video(video_path)
    _write_video(other_path)
    with other_path.open("ab") as handle:
        handle.write(b"different-hash")
    plan = _plan(video_path)

    with pytest.raises(ValueError, match="exactly match"):
        render_causal_phase_review_package(
            plan=plan,
            video_paths=[other_path],
            output_dir=tmp_path / "review",
        )


def test_materialize_causal_phase_decisions_requires_exact_unique_coverage(
    tmp_path: Path,
) -> None:
    template = {
        "schema_version": "agu.causal-shot-phase-codex-decisions.v1",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "plan_sha256": "a" * 64,
        "decisions": [
            {"phase_review_id": "causal-phase-0001"},
            {"phase_review_id": "causal-phase-0002"},
        ],
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    jsonl_path = tmp_path / "decisions.jsonl"
    rows = [
        {
            "phase_review_id": "causal-phase-0002",
            "formation_state": "live_play",
        },
        {
            "phase_review_id": "causal-phase-0001",
            "formation_state": "free_throw_setup",
        },
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    materialized = materialize_decisions(
        template_path=template_path,
        jsonl_path=jsonl_path,
    )

    assert materialized["plan_sha256"] == "a" * 64
    assert [
        row["phase_review_id"] for row in materialized["decisions"]
    ] == ["causal-phase-0001", "causal-phase-0002"]
    assert materialized["decisions"][0]["formation_state"] == (
        "free_throw_setup"
    )

    jsonl_path.write_text(
        "\n".join(json.dumps(rows[0]) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly cover"):
        materialize_decisions(
            template_path=template_path,
            jsonl_path=jsonl_path,
        )
