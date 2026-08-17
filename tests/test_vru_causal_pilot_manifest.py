from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from app.analysis.vru_causal_review import (
    build_vru_causal_review_plan,
    seal_vru_causal_review,
)
from scripts.build_vru_causal_pilot_manifest import build_manifests


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_manifests_binds_frame_and_review_artifacts(tmp_path: Path) -> None:
    source_manifest_path = tmp_path / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "agu.vru-basketball-source-manifest.v1",
                "runtime_consumable": False,
                "codex_runtime_answer_used": False,
                "videos": [
                    {
                        "location": "hazen",
                        "clip_id": "clip-a",
                        "path": "clip-a.webm",
                        "sha256": "a" * 64,
                        "fps": 10.0,
                        "frame_count": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_audit_path = tmp_path / "source_audit.json"
    source_audit_path.write_text(json.dumps({"audit_sha256": "b" * 64}), encoding="utf-8")
    plan = build_vru_causal_review_plan(
        source_manifest_sha256=_sha(source_manifest_path),
        examples=[
            {
                "review_id": "hazen-1s-v7",
                "source_video_sha256": "a" * 64,
                "source_video_filename": "clip-a.webm",
                "source_fps": 10.0,
                "frame_count": 100,
                "frame_indexes": [0, 5, 10],
                "frame_sha256s": {"0": "c" * 64, "5": "d" * 64, "10": "e" * 64},
            }
        ],
    )
    plan_path = tmp_path / "review_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    review = seal_vru_causal_review(
        {
            "reviewer": "test",
            "reviews": [
                {
                    "review_id": "hazen-1s-v7",
                    "shot_sequence": "not_a_shot",
                    "release_position": None,
                    "rim_position": None,
                    "outcome": "not_applicable",
                    "confidence": "high",
                    "evidence": {
                        "controlled_ball_before_release": False,
                        "ball_separated_from_hand": False,
                        "ball_progresses_toward_rim": False,
                        "rim_proximity_visible": False,
                        "rim_contact_visible": False,
                    },
                }
            ],
        },
        plan=plan,
    )
    review_spec_path = tmp_path / "review_spec.json"
    review_spec_path.write_text(
        json.dumps(
            {
                "schema_version": "agu.vru-causal-review-spec.v1",
                "runtime_consumable": False,
                "codex_runtime_answer_used": False,
                "examples": [{"review_id": "hazen-1s-v7", "clip_id": "clip-a"}],
            }
        ),
        encoding="utf-8",
    )
    decisions_path = tmp_path / "review_decisions.json"
    decisions_path.write_text(json.dumps({"plan_sha256": plan["artifact_sha256"]}), encoding="utf-8")
    sealed_path = tmp_path / "review_sealed.json"
    sealed_path.write_text(json.dumps(review), encoding="utf-8")
    frame_manifest_path = tmp_path / "raw_frames_manifest.json"
    frame_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "agu.vru-causal-review-frame-manifest.v1",
                "runtime_consumable": False,
                "codex_runtime_answer_used": False,
                "review_plan_sha256": plan["artifact_sha256"],
                "frame_root": "raw_frames",
                "frames": [
                    {
                        "review_id": "hazen-1s-v7",
                        "position": position,
                        "frame_index": index,
                        "relative_path": f"hazen-1s-v7_{position:03d}_{index}.jpg",
                        "raw_frame_sha256": digest,
                        "jpeg_sha256": digest,
                        "jpeg_bytes": 1,
                    }
                    for position, (index, digest) in enumerate(
                        [(0, "c" * 64), (5, "d" * 64), (10, "e" * 64)]
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "pilot_manifest.json"
    retention_path = tmp_path / "retention_manifest.json"
    pilot, retention = build_manifests(
        SimpleNamespace(
            source_manifest=source_manifest_path,
            source_cross_audit=source_audit_path,
            review_spec=review_spec_path,
            review_plan=plan_path,
            review_decisions=decisions_path,
            review_sealed=sealed_path,
            frame_manifest=frame_manifest_path,
            output=output_path,
            retention_output=retention_path,
        )
    )

    assert pilot["frame_count"] == 3
    assert pilot["label_counts"] == {"not_a_shot": 1}
    assert pilot["windows"][0]["source"] == "hazen"
    assert retention["frame_count"] == 3
    assert retention["windows"][0]["frames"][0]["bytes"] == 1
