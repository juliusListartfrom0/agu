from __future__ import annotations

import copy

import pytest

from app.analysis.broadcast_clock import (
    BROADCAST_CLOCK_EVIDENCE_SCHEMA,
    seal_broadcast_clock_artifact,
)
from app.analysis.shot_overlay_state import (
    FEATURE_NAMES,
    build_overlay_state_evidence_artifact,
    extract_overlay_state_features,
    parse_shot_clock_from_raw_text,
    seal_overlay_state_evidence_artifact,
    seal_scoreboard_timeline_artifact,
    verify_overlay_state_evidence_artifact,
)
from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    seal_scene_embedding_artifact,
)


def _clock_event(
    event_id: str,
    game_clocks: list[int],
    raw_texts: list[str],
) -> dict[str, object]:
    samples = []
    for index, (offset, seconds, raw_text) in enumerate(
        zip(
            (-6.0, -3.0, 0.0, 3.0, 6.0),
            game_clocks,
            raw_texts,
            strict=True,
        )
    ):
        frame = 820 + index * 90
        samples.append(
            {
                "frame": frame,
                "offset_seconds": offset,
                "read": {
                    "frame": frame,
                    "period": 1,
                    "clock_seconds": seconds,
                    "confidence": 0.9,
                    "raw_text": raw_text,
                    "source": "rapidocr_broadcast_clock_v1",
                },
            }
        )
    return {
        "event_id": event_id,
        "anchor_frame": 1000,
        "samples": samples,
        "pre_anchor_read_count": 2,
        "post_anchor_missing_count": 0,
        "frozen_period": None,
        "frozen_clock_seconds": None,
        "broadcast_state": "unknown",
        "method": "pre_anchor_frozen_clock_disappearance_v1",
    }


def _scene() -> dict[str, object]:
    return seal_scene_embedding_artifact(
        {
            "purpose": "test",
            "producer": "test",
            "training_manifest_sha256": "a" * 64,
            "training_annotation_sha256": ["b" * 64],
            "source_video_sha256": ["d" * 64],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": "e" * 64,
            "backbone_license": "BSD-3-Clause",
            "backbone_weights_url": "https://example.invalid/model",
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "phase_fractions": [0.15, 0.5, 0.85],
            "examples": [
                {
                    "source_video_sha256": "d" * 64,
                    "candidate_bundle_sha256": "c" * 64,
                    "event_id": "event-1",
                    "event_present": True,
                    "phase_embeddings": [
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                    ],
                }
            ],
        }
    )


def _clock_artifact(event: dict[str, object]) -> dict[str, object]:
    return seal_broadcast_clock_artifact(
        {
            "schema_version": BROADCAST_CLOCK_EVIDENCE_SCHEMA,
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": "d" * 64,
            "candidate_bundle_sha256": "c" * 64,
            "events": [event],
        }
    )


def _timeline() -> dict[str, object]:
    return seal_scoreboard_timeline_artifact(
        {
            "reader_method": "rapidocr_broadcast_scoreboard_v3",
            "raw_video_sha256": "d" * 64,
            "team_ids": ["ATL", "CHI"],
            "stride_frames": 60,
            "confidence": 0.75,
            "crop_top_ratio": 0.58,
            "reads": [
                {
                    "frame": frame,
                    "scores": {"ATL": atl, "CHI": 10},
                    "confidence": 0.9,
                    "source": "rapidocr_broadcast_scoreboard_v3",
                }
                for frame, atl in (
                    (430, 8),
                    (490, 8),
                    (550, 8),
                    (610, 8),
                    (1000, 8),
                    (1060, 10),
                    (1120, 10),
                    (1180, 10),
                )
            ],
            "deltas": [],
        }
    )


def test_shot_clock_parser_requires_game_clock_binding_and_plausible_suffix() -> None:
    assert parse_shot_clock_from_raw_text("1ST11:51:17", 711) == 17
    assert parse_shot_clock_from_raw_text("4TH10:5424", 654) == 24
    assert parse_shot_clock_from_raw_text("2ND4:17", 257) is None
    assert parse_shot_clock_from_raw_text("2ND4:17:35", 257) is None
    assert parse_shot_clock_from_raw_text("3RD38.6:24", 384) is None


def test_overlay_features_capture_shot_clock_reset_and_score_change() -> None:
    event = _clock_event(
        "event-1",
        [611, 608, 605, 602, 599],
        [
            "1ST10:11:10",
            "1ST10:08:07",
            "1ST10:05:05",
            "1ST10:02:24",
            "1ST9:59:21",
        ],
    )

    features = extract_overlay_state_features(event, _timeline())

    assert tuple(features) == FEATURE_NAMES
    assert features["shot_clock_reset_flag"] == 1.0
    assert features["maximum_shot_clock_increase"] == 19.0
    assert features["score_change_flag"] == 1.0
    assert features["score_change_points"] == 2.0
    assert features["score_change_delay_seconds"] == pytest.approx(2.0)


def test_overlay_artifact_is_exact_label_free_and_tamper_evident() -> None:
    scene = _scene()
    clock = _clock_artifact(
        _clock_event(
            "event-1",
            [611, 608, 605, 602, 599],
            ["1ST10:11:10", "1ST10:08:07", "1ST10:05:05", "1ST10:02:24", "1ST9:59:21"],
        )
    )
    timeline = _timeline()

    artifact = build_overlay_state_evidence_artifact(
        scene_artifact=scene,
        clock_artifacts=[clock],
        scoreboard_timeline_artifacts=[timeline],
    )

    assert len(artifact["examples"]) == 1
    assert artifact["source_scene_artifact_sha256"] == scene["artifact_sha256"]
    assert "event_present" not in str(artifact)
    assert verify_overlay_state_evidence_artifact(artifact) == artifact

    leaked = copy.deepcopy(artifact)
    leaked["examples"][0]["event_present"] = True
    with pytest.raises(ValueError, match="labels"):
        seal_overlay_state_evidence_artifact(leaked)

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["features"][0] = 99.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_overlay_state_evidence_artifact(tampered)

    with pytest.raises(ValueError, match="exactly cover"):
        build_overlay_state_evidence_artifact(
            scene_artifact=scene,
            clock_artifacts=[clock],
            scoreboard_timeline_artifacts=[],
        )
