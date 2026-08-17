from __future__ import annotations

import copy
import hashlib

import cv2
import numpy as np
import pytest

from app.analysis.muvs_event_state import (
    seal_muvs_event_state_frames,
    seal_muvs_event_state_plan,
    seal_muvs_event_state_review,
)
from app.analysis.muvs_pair_motion import (
    MUVS_PAIR_MOTION_DIMENSION,
    build_muvs_pair_motion_features,
    extract_muvs_pair_motion_features,
    screen_muvs_pair_motion,
    seal_muvs_pair_motion_features,
    verify_muvs_pair_motion_features,
    verify_muvs_pair_motion_screen,
)


def _feature_artifact(*, hash_seed: int = 0) -> dict:
    examples = []
    for event_index in range(4):
        for label_index, state in enumerate(
            ("live_play", "free_throw_setup")
        ):
            sample_index = event_index * 2 + label_index
            value = -2.0 if label_index == 0 else 2.0
            examples.append(
                {
                    "sample_id": f"muvs-state-{sample_index + 1:04d}",
                    "event_id": f"event-{event_index}",
                    "split": "train" if event_index < 3 else "test",
                    "state": state,
                    "frame_before_sha256": hashlib.sha256(
                        f"{hash_seed}:{sample_index}:before".encode()
                    ).hexdigest(),
                    "frame_after_sha256": hashlib.sha256(
                        f"{hash_seed}:{sample_index}:after".encode()
                    ).hexdigest(),
                    "homography_applied": True,
                    "match_count": 40,
                    "inlier_ratio": 0.8,
                    "features": [value] * MUVS_PAIR_MOTION_DIMENSION,
                }
            )
    return seal_muvs_pair_motion_features(
        {
            "frame_artifact_sha256": hashlib.sha256(
                f"{hash_seed}:frames".encode()
            ).hexdigest(),
            "review_artifact_sha256": hashlib.sha256(
                f"{hash_seed}:review".encode()
            ).hexdigest(),
            "examples": examples,
        }
    )


def test_pair_motion_extracts_finite_camera_compensated_features() -> None:
    rng = np.random.default_rng(7)
    before = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    camera_shift = np.float32([[1, 0, 5], [0, 1, 3]])
    after = cv2.warpAffine(
        before,
        camera_shift,
        (before.shape[1], before.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    cv2.rectangle(after, (120, 100), (155, 180), (255, 255, 255), -1)

    result = extract_muvs_pair_motion_features(before, after)

    values = np.asarray(result["features"], dtype=np.float64)
    assert values.shape == (MUVS_PAIR_MOTION_DIMENSION,)
    assert np.isfinite(values).all()
    assert result["homography_applied"] is True
    assert result["match_count"] >= 8
    assert 0.0 <= result["inlier_ratio"] <= 1.0


def test_pair_motion_artifact_is_hash_bound_and_training_only() -> None:
    artifact = _feature_artifact()

    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    assert verify_muvs_pair_motion_features(artifact) == artifact

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["features"][0] = 99.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_muvs_pair_motion_features(tampered)


def test_pair_motion_builder_verifies_plan_frames_and_review() -> None:
    plan = seal_muvs_event_state_plan(
        {
            "schema_version": "agu.muvs-event-state-plan.v1",
            "purpose": "codex_offline_muvs_source_event_state_annotation",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "source": {
                "record_id": 20_708_683,
                "record_url": "https://zenodo.org/records/20708683",
                "record_license": "CC-BY-4.0",
                "package_readme_license": "TO BE ADDED",
                "license_basis": "Zenodo record metadata",
                "media_redistribution": "not_performed",
            },
            "reviewer_visible_fields": [
                "sample_id",
                "frame_before",
                "frame_after",
            ],
            "event_count": 1,
            "samples_per_event": 1,
            "sample_count": 1,
            "samples": [
                {
                    "sample_id": "muvs-state-0001",
                    "split": "train",
                    "event_id": "event-1",
                    "period": 1,
                    "fragment_number": 1,
                    "camera_id": "cam_1",
                    "camera_offset_sec": 0.0,
                    "frame_times_sec": [1.0, 2.0],
                    "video_url": (
                        "https://storage.googleapis.com/dataset-ugv-sports/"
                        "public/RAW_DATA/basketball/event-1/cam_1/"
                        "VIDEOS/P1/video.mp4"
                    ),
                    "annotation_sha256": "a" * 64,
                    "fragments_sha256": "b" * 64,
                }
            ],
        }
    )
    frame_rows = [
        {
            "sample_id": "muvs-state-0001",
            "frame_before": {
                "path": "frames/before.png",
                "sha256": "c" * 64,
                "size_bytes": 100,
                "width": 320,
                "height": 240,
            },
            "frame_after": {
                "path": "frames/after.png",
                "sha256": "d" * 64,
                "size_bytes": 100,
                "width": 320,
                "height": 240,
            },
        }
    ]
    frames = seal_muvs_event_state_frames({"files": frame_rows}, plan=plan)
    review = seal_muvs_event_state_review(
        {
            "reviewer": "codex_offline_source_annotation",
            "decisions": [
                {
                    "sample_id": "muvs-state-0001",
                    "state": "live_play",
                    "notes": "Players are moving in a live possession.",
                }
            ],
        },
        plan=plan,
        frames=frames,
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    artifact = build_muvs_pair_motion_features(
        plan=plan,
        frames=frames,
        review=review,
        frame_loader=lambda _metadata: image,
    )

    assert artifact["frame_artifact_sha256"] == frames["artifact_sha256"]
    assert artifact["review_artifact_sha256"] == review["artifact_sha256"]
    assert artifact["examples"][0]["event_id"] == "event-1"
    assert artifact["examples"][0]["state"] == "live_play"
    assert verify_muvs_pair_motion_features(artifact) == artifact


def test_pair_motion_screen_uses_fixed_leave_one_event_out_protocol() -> None:
    artifact = _feature_artifact()

    screen = screen_muvs_pair_motion(artifact)

    assert screen["protocol"] == {
        "grouping": "leave_one_event_out",
        "classifier": "StandardScaler+balanced_LogisticRegression_C1",
        "threshold": "fixed_zero_logistic_margin",
        "selection_data": "MUVS_source_only",
        "acceptance_metric": "source_oof_balanced_accuracy",
        "acceptance_threshold": 0.85,
        "minimum_positive_event_recall": 0.5,
        "minimum_negative_specificity": 0.85,
    }
    assert screen["metrics"]["balanced_accuracy"] == pytest.approx(1.0)
    assert screen["accepted"] is True
    assert verify_muvs_pair_motion_screen(screen) == screen


def test_pair_motion_screen_deduplicates_and_rejects_conflicts() -> None:
    first = _feature_artifact()
    second = _feature_artifact(hash_seed=1)

    assert screen_muvs_pair_motion([first, second])["resolved_examples"] == 16
    assert screen_muvs_pair_motion([first, first])["resolved_examples"] == 8

    conflict = copy.deepcopy(first)
    conflict["examples"][0]["state"] = "free_throw_setup"
    conflict = seal_muvs_pair_motion_features(conflict)
    with pytest.raises(ValueError, match="duplicate MUVS pair motion disagrees"):
        screen_muvs_pair_motion([first, conflict])
