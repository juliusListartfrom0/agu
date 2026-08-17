from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.analysis.pbp_visual_state_frames import (
    DINO_V2_SMALL_BACKBONE,
    PRE_ANCHOR_OFFSETS_SECONDS,
    seal_anchor_state_embedding_artifact,
)
from app.analysis.pbp_visual_state_review import (
    build_visual_state_followup_review_plan,
    build_visual_state_review_plan,
    derive_visual_state_label_corrections,
    seal_visual_state_review,
)
from scripts.build_pbp_visual_state_review import (
    build_review_package,
    render_review_plan_package,
)
from scripts.seal_pbp_visual_state_review import seal_review


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        1.0,
        (64, 36),
    )
    assert writer.isOpened()
    for index in range(5):
        frame = np.full((36, 64, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_long_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        1.0,
        (64, 36),
    )
    assert writer.isOpened()
    for index in range(20):
        frame = np.full((36, 64, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_review_package_renders_label_hidden_raw_frame_sheet(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "development.avi"
    _write_video(video_path)
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
            "backbone_sha256": "6" * 64,
            "anchor_offsets_seconds": list(PRE_ANCHOR_OFFSETS_SECONDS),
            "examples": [
                {
                    "source_video_sha256": source_sha,
                    "source_video_filename": video_path.name,
                    "candidate_bundle_sha256": "7" * 64,
                    "event_id": "event-1",
                    "state": "free_throw",
                    "anchor_frame": 4,
                    "source_fps": 1.0,
                    "frame_count": 5,
                    "frame_indexes": [0, 1, 2, 3, 4],
                    "embeddings": [[float(index)] * 384 for index in range(5)],
                }
            ],
        }
    )
    artifact_path = tmp_path / "embeddings.json"
    artifact_path.write_text(json.dumps(embeddings), encoding="utf-8")
    output_dir = tmp_path / "review"

    manifest = build_review_package(
        embedding_artifact_path=artifact_path,
        video_paths=[video_path],
        output_dir=output_dir,
        panel_width=160,
        rows_per_sheet=4,
    )

    assert manifest["schema_version"] == "agu.pbp-visual-state-review-sheets.v1"
    assert manifest["runtime_consumable"] is False
    assert manifest["records"][0]["review_ids"] == ["visual-state-0001"]
    assert (output_dir / manifest["records"][0]["sheet"]).is_file()
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    assert "free_throw" not in json.dumps(plan["examples"], sort_keys=True)
    template = json.loads(
        (output_dir / "decisions.template.json").read_text(encoding="utf-8")
    )
    assert template["decisions"][0]["visual_state"] is None

    decisions = {
        **template,
        "decisions": [
            {
                "review_id": "visual-state-0001",
                "visual_state": "live_play",
                "confidence": "high",
                "notes": "Active possession.",
            }
        ],
    }
    decisions_path = output_dir / "decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")

    review, corrections = seal_review(
        embedding_artifact_path=artifact_path,
        plan_path=output_dir / "plan.json",
        sheet_manifest_path=output_dir / "manifest.json",
        decisions_path=decisions_path,
    )

    assert review["reviews"][0]["visual_state"] == "live_play"
    assert corrections["summary"]["changed"] == 1


def test_render_followup_plan_package_uses_wider_unresolved_plan(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "development.avi"
    _write_long_video(video_path)
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
            "backbone_sha256": "6" * 64,
            "anchor_offsets_seconds": list(PRE_ANCHOR_OFFSETS_SECONDS),
            "examples": [
                {
                    "source_video_sha256": source_sha,
                    "source_video_filename": video_path.name,
                    "candidate_bundle_sha256": "7" * 64,
                    "event_id": "event-1",
                    "state": "free_throw",
                    "anchor_frame": 10,
                    "source_fps": 1.0,
                    "frame_count": 20,
                    "frame_indexes": [6, 7, 8, 9, 10],
                    "embeddings": [
                        [float(index)] * 384 for index in range(5)
                    ],
                }
            ],
        }
    )
    base_plan = build_visual_state_review_plan(embeddings)
    base_review = seal_visual_state_review(
        {
            "plan_sha256": base_plan["artifact_sha256"],
            "reviewer": "codex_offline_visual_review",
            "reviews": [
                {
                    "review_id": "visual-state-0001",
                    "visual_state": "stoppage_other",
                    "confidence": "medium",
                    "notes": "Needs wider context.",
                }
            ],
        },
        plan=base_plan,
    )
    corrections = derive_visual_state_label_corrections(
        embedding_artifact=embeddings,
        plan=base_plan,
        review=base_review,
    )
    followup = build_visual_state_followup_review_plan(
        base_plan,
        corrections=corrections,
    )
    plan_path = tmp_path / "followup-plan.json"
    plan_path.write_text(json.dumps(followup), encoding="utf-8")

    manifest = render_review_plan_package(
        plan_path=plan_path,
        video_paths=[video_path],
        output_dir=tmp_path / "followup",
        panel_width=160,
        rows_per_sheet=4,
    )

    assert manifest["plan_sha256"] == followup["artifact_sha256"]
    assert manifest["records"][0]["review_ids"] == ["visual-state-0001"]
    sheet = cv2.imread(
        str(tmp_path / "followup" / manifest["records"][0]["sheet"])
    )
    assert sheet.shape[1] == 160 * 9
