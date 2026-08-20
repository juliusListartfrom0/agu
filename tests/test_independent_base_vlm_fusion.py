from __future__ import annotations

import copy

import pytest

from app.analysis.independent_base_vlm_fusion import (
    centered_clip_frame_indexes,
    fuse_base_and_vlm_predictions,
    seal_independent_base_predictions,
    select_candidate_player_observations,
    verify_independent_base_predictions,
)
from app.analysis.independent_shot_vlm import (
    parse_independent_shot_vlm_decision,
    seal_independent_shot_vlm_plan,
    seal_independent_shot_vlm_predictions,
)


def _row(event: str) -> dict[str, object]:
    return {
        "source_video_sha256": "a" * 64,
        "source_video_filename": "game.mp4",
        "candidate_bundle_sha256": "b" * 64,
        "event_id": event,
        "start_frame": 100,
        "end_frame": 199,
        "source_fps": 30.0,
    }


def _player_clip() -> dict[str, object]:
    return {
        "player_id": "player-1",
        "action": "shoot",
        "action_confidence": 0.8,
        "shoot_probability": 0.8,
    }


def test_candidate_player_selection_is_unique_and_ball_ordered() -> None:
    evidence = [
        {
            "details": {
                "candidate_player_observations": [
                    {
                        "player_id": "far",
                        "frame": 120,
                        "bbox": {"x1": 1, "y1": 2, "x2": 11, "y2": 22},
                        "wrist_ball_distance": 0.8,
                        "ball_player_distance": 0.5,
                    },
                    {
                        "player_id": "near",
                        "frame": 121,
                        "bbox": {"x1": 3, "y1": 4, "x2": 13, "y2": 24},
                        "wrist_ball_distance": 0.1,
                        "ball_player_distance": 0.2,
                    },
                ]
            }
        },
        {
            "details": {
                "candidate_player_observations": [
                    {
                        "player_id": "near",
                        "frame": 122,
                        "bbox": {"x1": 3, "y1": 4, "x2": 13, "y2": 24},
                        "wrist_ball_distance": 0.3,
                        "ball_player_distance": 0.2,
                    },
                    {
                        "player_id": "broken",
                        "frame": 122,
                        "bbox": {"x1": 3, "y1": 4, "x2": 3, "y2": 24},
                    },
                ]
            }
        },
    ]

    selected = select_candidate_player_observations(evidence, limit=2)

    assert [row["player_id"] for row in selected] == ["near", "far"]
    assert selected[0]["frame"] == 121


def test_centered_clip_indexes_stay_inside_event() -> None:
    assert centered_clip_frame_indexes(
        center_frame=102,
        start_frame=100,
        end_frame=199,
        clip_frames=16,
    ) == list(range(100, 116))
    assert centered_clip_frame_indexes(
        center_frame=198,
        start_frame=100,
        end_frame=199,
        clip_frames=16,
    ) == list(range(184, 200))
    with pytest.raises(ValueError, match="shorter"):
        centered_clip_frame_indexes(
            center_frame=2,
            start_frame=0,
            end_frame=5,
            clip_frames=16,
        )


def test_base_predictions_are_hash_bound_and_label_free() -> None:
    plan = seal_independent_shot_vlm_plan(
        {"selection": {}, "examples": [_row("event-1")]}
    )
    artifact = seal_independent_base_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {
                "name": "agu-v3-r2plus1d",
                "checkpoint_sha256": "c" * 64,
            },
            "input_contract": {
                "clip_frames": 16,
                "maximum_player_clips": 3,
                "decision": "any_player_argmax_shoot",
            },
            "predictions": [
                {
                    **{
                        key: _row("event-1")[key]
                        for key in (
                            "source_video_sha256",
                            "candidate_bundle_sha256",
                            "event_id",
                        )
                    },
                    "available": True,
                    "field_goal_state": "live_field_goal",
                    "confidence": 0.8,
                    "shoot_probability": 0.8,
                    "player_clips": [_player_clip()],
                }
            ],
        }
    )

    assert verify_independent_base_predictions(copy.deepcopy(artifact)) == artifact
    assert "event_present" not in str(artifact)
    artifact["predictions"][0]["confidence"] = 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_independent_base_predictions(artifact)


def test_base_predictions_fail_closed_on_invalid_player_clips() -> None:
    plan = seal_independent_shot_vlm_plan(
        {"selection": {}, "examples": [_row("event-1")]}
    )
    payload = {
        "plan_sha256": plan["plan_sha256"],
        "model": {
            "name": "agu-v3-r2plus1d",
            "checkpoint_sha256": "c" * 64,
        },
        "input_contract": {
            "clip_frames": 16,
            "maximum_player_clips": 1,
            "decision": "any_player_argmax_shoot",
        },
        "predictions": [
            {
                **{
                    key: _row("event-1")[key]
                    for key in (
                        "source_video_sha256",
                        "candidate_bundle_sha256",
                        "event_id",
                    )
                },
                "available": True,
                "field_goal_state": "live_field_goal",
                "confidence": 0.8,
                "shoot_probability": 0.8,
                "player_clips": [_player_clip(), _player_clip()],
            }
        ],
    }

    with pytest.raises(ValueError, match="player clip limit"):
        seal_independent_base_predictions(payload)

    payload["predictions"][0]["player_clips"] = []
    with pytest.raises(ValueError, match="availability"):
        seal_independent_base_predictions(payload)


def test_fixed_fusion_requires_exact_coverage_and_propagates_unknown() -> None:
    rows = [_row("event-1"), _row("event-2")]
    plan = seal_independent_shot_vlm_plan({"selection": {}, "examples": rows})
    base = seal_independent_base_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {
                "name": "agu-v3-r2plus1d",
                "checkpoint_sha256": "c" * 64,
            },
            "input_contract": {
                "clip_frames": 16,
                "maximum_player_clips": 3,
                "decision": "any_player_argmax_shoot",
            },
            "predictions": [
                {
                    **{
                        key: row[key]
                        for key in (
                            "source_video_sha256",
                            "candidate_bundle_sha256",
                            "event_id",
                        )
                    },
                    "available": index == 0,
                    "field_goal_state": (
                        "live_field_goal" if index == 0 else "unknown"
                    ),
                    "confidence": 0.9 if index == 0 else 0.0,
                    "shoot_probability": 0.9 if index == 0 else 0.0,
                    "player_clips": [_player_clip()] if index == 0 else [],
                }
                for index, row in enumerate(rows)
            ],
        }
    )
    vlm = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {"name": "independent-vlm"},
            "predictions": [
                {
                    **{
                        key: row[key]
                        for key in (
                            "source_video_sha256",
                            "candidate_bundle_sha256",
                            "event_id",
                        )
                    },
                    **parse_independent_shot_vlm_decision(
                        {
                            "field_goal_state": "live_field_goal",
                            "confidence": 0.7,
                        }
                    ),
                }
                for row in rows
            ],
        }
    )

    fused = fuse_base_and_vlm_predictions(
        plan=plan,
        base_predictions=base,
        vlm_predictions=vlm,
        rule="both_confirm",
    )

    assert [row["field_goal_state"] for row in fused["predictions"]] == [
        "live_field_goal",
        "unknown",
    ]
    incomplete = copy.deepcopy(vlm)
    incomplete["predictions"].pop()
    incomplete = seal_independent_shot_vlm_predictions(incomplete)
    with pytest.raises(ValueError, match="exactly cover"):
        fuse_base_and_vlm_predictions(
            plan=plan,
            base_predictions=base,
            vlm_predictions=incomplete,
            rule="both_confirm",
        )
