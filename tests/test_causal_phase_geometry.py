from __future__ import annotations

import copy

import pytest

from app.analysis.causal_phase_geometry import (
    GEOMETRY_FEATURE_NAMES,
    attach_dense_geometry_features,
    extract_dense_geometry_features,
    extract_prepared_dense_geometry_features,
    prepare_dense_geometry_ball_tracks,
    prepare_dense_geometry_perception,
    seal_dense_geometry_artifact,
    verify_dense_geometry_artifact,
)


def _perception(
    object_type: str,
    *,
    stride: int,
    windows: bool = False,
) -> dict:
    detections = []
    for frame in range(0, 72, stride):
        detections.append(
            {
                "frame": frame,
                "object_type": object_type,
                "confidence": 0.9,
                "bbox": {
                    "x1": 10.0 + frame,
                    "y1": 20.0,
                    "x2": 30.0 + frame,
                    "y2": 60.0,
                },
                "team_id": "dark" if frame % 2 else "light",
            }
        )
    sampling = {
        "stride_frames": stride,
        "start_frame": 0,
        "end_frame": 71,
    }
    if windows:
        sampling["windows"] = [{"start_frame": 0, "end_frame": 71}]
    return {
        "schema_version": "agu.official-perception.v1",
        "sampling": sampling,
        "detections": detections,
    }


def _geometry_artifact() -> dict:
    features = extract_dense_geometry_features(
        frame_indexes=list(range(0, 72, 3)),
        frame_width=200,
        frame_height=100,
        player_artifact=_perception("player", stride=6),
        rim_artifact=_perception("rim", stride=6),
        ball_artifact=_perception("basketball", stride=3, windows=True),
    )
    return seal_dense_geometry_artifact(
        {
            "review_plan_sha256": "1" * 64,
            "source_video_sha256s": ["a" * 64],
            "sealed_blind_video_sha256s": ["b" * 64],
            "source_artifact_sha256s": ["2" * 64, "3" * 64, "4" * 64],
            "examples": [
                {
                    "phase_review_id": "phase-1",
                    "source_video_sha256": "a" * 64,
                    "candidate_bundle_sha256": "c" * 64,
                    "event_id": "event-1",
                    "features": features,
                }
            ],
        }
    )


def test_dense_geometry_extracts_normalized_layout_and_motion() -> None:
    artifact = _geometry_artifact()
    features = artifact["examples"][0]["features"]

    assert len(features) == 24
    assert len(features[0]) == len(GEOMETRY_FEATURE_NAMES)
    assert features[0][GEOMETRY_FEATURE_NAMES.index("ball_visible")] == 1.0
    assert features[1][GEOMETRY_FEATURE_NAMES.index("ball_motion_speed")] > 0.0
    assert verify_dense_geometry_artifact(artifact) == artifact


def test_dense_geometry_hash_and_blind_source_are_fail_closed() -> None:
    artifact = _geometry_artifact()
    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["features"][0][0] = 99.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_dense_geometry_artifact(tampered)

    blind = copy.deepcopy(artifact)
    blind.pop("artifact_sha256")
    blind["sealed_blind_video_sha256s"] = ["a" * 64]
    with pytest.raises(ValueError, match="sealed blind"):
        seal_dense_geometry_artifact(blind)


def test_dense_geometry_fusion_requires_exact_event_binding() -> None:
    artifact = _geometry_artifact()
    examples = [
        {
            "phase_review_id": "phase-1",
            "source_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "c" * 64,
            "event_id": "event-1",
            "embeddings": [[0.0] * 8 for _ in range(24)],
        }
    ]

    fused, provenance = attach_dense_geometry_features(
        examples,
        geometry_artifact=artifact,
    )

    assert len(fused[0]["embeddings"][0]) == 8 + len(
        GEOMETRY_FEATURE_NAMES
    )
    assert provenance["geometry_artifact_sha256"] == artifact["artifact_sha256"]

    with pytest.raises(ValueError, match="cover all"):
        attach_dense_geometry_features(
            [{**examples[0], "event_id": "different"}],
            geometry_artifact=artifact,
        )


def test_dense_geometry_track_mode_uses_supported_interpolated_points() -> None:
    ball = _perception("basketball", stride=3, windows=True)
    ball["ball_tracks"] = [
        {
            "track_id": "supported",
            "points": [
                {
                    "frame": 0,
                    "center": {"x": 20.0, "y": 30.0},
                    "confidence": 0.8,
                    "visible": True,
                    "predicted": False,
                },
                {
                    "frame": 1,
                    "center": {"x": 21.0, "y": 31.0},
                    "confidence": 0.5,
                    "visible": False,
                    "predicted": True,
                },
                {
                    "frame": 3,
                    "center": {"x": 23.0, "y": 33.0},
                    "confidence": 0.7,
                    "visible": True,
                    "predicted": False,
                },
            ],
        },
        {
            "track_id": "singleton",
            "points": [
                {
                    "frame": 2,
                    "center": {"x": 99.0, "y": 99.0},
                    "confidence": 1.0,
                    "visible": True,
                    "predicted": False,
                }
            ],
        },
    ]
    features = extract_prepared_dense_geometry_features(
        frame_indexes=list(range(24)),
        frame_width=200,
        frame_height=100,
        player_perception=prepare_dense_geometry_perception(
            _perception("player", stride=1),
            expected_object_type="player",
        ),
        rim_perception=prepare_dense_geometry_perception(
            _perception("rim", stride=1),
            expected_object_type="rim",
        ),
        ball_perception=prepare_dense_geometry_ball_tracks(ball),
    )

    assert features[1][GEOMETRY_FEATURE_NAMES.index("ball_visible")] == 1.0
    assert features[1][GEOMETRY_FEATURE_NAMES.index("ball_center_x")] == pytest.approx(
        21.0 / 200.0
    )
    assert features[2][GEOMETRY_FEATURE_NAMES.index("ball_visible")] == 0.0
