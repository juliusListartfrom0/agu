from __future__ import annotations

import copy
import hashlib
import json

import pytest

from app.analysis.causal_phase_ball_windows import (
    causal_ball_windows,
    sampled_frame_count,
    verify_causal_ball_perception,
)


def test_causal_ball_windows_merge_only_overlapping_source_events() -> None:
    source = "a" * 64
    examples = [
        {
            "phase_review_id": "phase-1",
            "source_video_sha256": source,
            "frame_indexes": list(range(10, 34)),
        },
        {
            "phase_review_id": "phase-2",
            "source_video_sha256": source,
            "frame_indexes": list(range(36, 60)),
        },
        {
            "phase_review_id": "other",
            "source_video_sha256": "b" * 64,
            "frame_indexes": list(range(100, 124)),
        },
    ]

    windows, review_ids = causal_ball_windows(
        examples,
        source_video_sha256=source,
        merge_gap_frames=3,
    )

    assert windows == [(10, 60)]
    assert review_ids == ["phase-1", "phase-2"]
    assert sampled_frame_count(windows, stride_frames=3) == 17


def test_causal_ball_windows_reject_missing_source_and_bad_frame_contract() -> None:
    with pytest.raises(ValueError, match="no events"):
        causal_ball_windows(
            [],
            source_video_sha256="a" * 64,
            merge_gap_frames=0,
        )


def _signed_ball_payload() -> tuple[dict, dict]:
    source = "a" * 64
    plan = {
        "artifact_sha256": "b" * 64,
        "source_video_sha256s": [source],
        "sealed_blind_video_sha256s": ["c" * 64],
        "examples": [
            {
                "phase_review_id": "phase-1",
                "source_video_sha256": source,
                "frame_indexes": list(range(10, 34)),
            }
        ],
    }
    artifact = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "sha256": source,
            "source_fps": 30.0,
            "frame_count": 100,
        },
        "detector": {
            "model_sha256": "d" * 64,
            "object_type_allowlist": ["basketball"],
        },
        "sampling": {
            "requested_fps": 10.0,
            "stride_frames": 3,
            "start_frame": 10,
            "end_frame": 34,
            "sample_count": 8,
            "decoded_frame_count": 24,
            "windows": [{"start_frame": 10, "end_frame": 34}],
        },
        "causal_source": {
            "review_plan_sha256": plan["artifact_sha256"],
            "phase_review_ids": ["phase-1"],
            "event_count": 1,
            "merge_gap_sec": 0.5,
            "complete_run": True,
        },
        "counts": {"basketball": 1},
        "detections": [
            {
                "frame": 13,
                "object_type": "basketball",
                "confidence": 0.8,
                "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            }
        ],
        "ball_tracks": [],
    }
    encoded = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    artifact["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    return plan, artifact


def test_verify_causal_ball_perception_binds_exact_windows_and_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, artifact = _signed_ball_payload()
    monkeypatch.setattr(
        "app.analysis.causal_phase_ball_windows."
        "verify_causal_shot_phase_review_plan",
        lambda payload: dict(payload),
    )

    verified = verify_causal_ball_perception(artifact, plan=plan)

    assert verified["artifact_sha256"] == artifact["artifact_sha256"]


def test_verify_causal_ball_perception_rejects_incomplete_or_tampered_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, artifact = _signed_ball_payload()
    monkeypatch.setattr(
        "app.analysis.causal_phase_ball_windows."
        "verify_causal_shot_phase_review_plan",
        lambda payload: dict(payload),
    )
    incomplete = copy.deepcopy(artifact)
    incomplete["causal_source"]["complete_run"] = False
    with pytest.raises(ValueError, match="complete"):
        verify_causal_ball_perception(incomplete, plan=plan)

    tampered = copy.deepcopy(artifact)
    tampered["sampling"]["sample_count"] = 7
    with pytest.raises(ValueError, match="sample count|hash"):
        verify_causal_ball_perception(tampered, plan=plan)

    with pytest.raises(ValueError, match="invalid"):
        causal_ball_windows(
            [
                {
                    "phase_review_id": "phase-1",
                    "source_video_sha256": "a" * 64,
                    "frame_indexes": [1, 2],
                }
            ],
            source_video_sha256="a" * 64,
            merge_gap_frames=0,
        )
