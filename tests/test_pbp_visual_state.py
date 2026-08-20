from __future__ import annotations

import copy

import pytest

from app.analysis.broadcast_clock import seal_broadcast_clock_artifact
from app.analysis.pbp_visual_state import (
    build_pbp_visual_state_manifest,
    parse_nba_clock_seconds,
    verify_pbp_visual_state_manifest,
)

VIDEO_SHA = "1" * 64
PBP_SHA = "2" * 64


def _clock_artifact() -> dict:
    return seal_broadcast_clock_artifact(
        {
            "schema_version": "agu.broadcast-clock-replay-evidence.v1",
            "purpose": "test",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": VIDEO_SHA,
            "candidate_bundle_sha256": "3" * 64,
            "source": {},
            "configuration": {},
            "events": [
                {
                    "event_id": "event-1",
                    "anchor_frame": 100,
                    "samples": [
                        {
                            "frame": 90,
                            "offset_seconds": -1.0,
                            "read": {
                                "frame": 90,
                                "period": 1,
                                "clock_seconds": 650,
                                "confidence": 0.99,
                            },
                        },
                        {
                            "frame": 100,
                            "offset_seconds": 0.0,
                            "read": {
                                "frame": 100,
                                "period": 1,
                                "clock_seconds": 649,
                                "confidence": 0.90,
                            },
                        },
                        {
                            "frame": 110,
                            "offset_seconds": 1.0,
                            "read": {
                                "frame": 110,
                                "period": 1,
                                "clock_seconds": 648,
                                "confidence": 0.99,
                            },
                        },
                    ],
                    "broadcast_state": "unknown",
                },
                {
                    "event_id": "event-2",
                    "anchor_frame": 200,
                    "samples": [
                        {"frame": 200, "offset_seconds": 0.0, "read": None}
                    ],
                    "broadcast_state": "unknown",
                },
            ],
        }
    )


def _pbp(
    *,
    action_id: int,
    clock: str = "PT10M49.00S",
    action_type: str = "Made Shot",
) -> dict:
    return {
        "game_id": "0041000203",
        "period": 1,
        "clock": clock,
        "actionId": action_id,
        "actionType": action_type,
        "subType": "Jump Shot",
        "playerName": "must not leak",
        "description": "must not leak",
    }


def _build(rows: list[dict], **overrides: object) -> dict:
    arguments = {
        "clock_artifact": _clock_artifact(),
        "pbp_rows": rows,
        "pbp_sha256": PBP_SHA,
        "pbp_game_id": "0041000203",
        "source_slug": "2011-05-06-chi-vs-atl",
        "allowed_clock_delta_seconds": 1,
        "sealed_blind_video_sha256s": (),
    }
    arguments.update(overrides)
    return build_pbp_visual_state_manifest(**arguments)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PT10M49.00S", 649),
        ("PT0M00.00S", 0),
        ("PT12M00.00S", 720),
    ],
)
def test_parse_nba_clock_seconds(raw: str, expected: int) -> None:
    assert parse_nba_clock_seconds(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["10:49", "PT10M49S", "PT12M01.00S", "PT-1M00.00S", ""],
)
def test_parse_nba_clock_seconds_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_nba_clock_seconds(raw)


def test_build_manifest_labels_determinate_state_and_accounts_for_every_event() -> None:
    manifest = _build(
        [
            _pbp(action_id=1, action_type="Made Shot"),
            _pbp(action_id=2, clock="PT10M48.00S", action_type="Foul"),
        ]
    )

    assert manifest["schema_version"] == "agu.pbp-visual-state-training.v1"
    assert manifest["runtime_consumable"] is False
    assert manifest["truth_used_for_training_only"] is True
    assert manifest["codex_runtime_answer_used"] is False
    assert manifest["summary"] == {
        "total_clock_events": 2,
        "determinate": 1,
        "field_goal": 1,
        "free_throw": 0,
        "foul_only": 0,
        "ambiguous": 0,
        "other": 0,
        "unmatched": 0,
        "no_clock_read": 1,
    }
    assert manifest["examples"][0]["state"] == "field_goal"
    assert manifest["examples"][0]["clock_delta_seconds"] == 0
    assert manifest["examples"][0]["pbp_actions"] == [
        {
            "action_id": 1,
            "action_type": "Made Shot",
            "sub_type": "Jump Shot",
        }
    ]
    assert manifest["exclusions"] == [
        {"event_id": "event-2", "anchor_frame": 200, "reason": "no_clock_read"}
    ]
    serialized = repr(manifest)
    assert "must not leak" not in serialized
    verify_pbp_visual_state_manifest(manifest)


def test_build_manifest_fails_closed_on_ambiguous_same_clock() -> None:
    manifest = _build(
        [
            _pbp(action_id=1, action_type="Made Shot"),
            _pbp(action_id=2, action_type="Free Throw"),
        ]
    )

    assert not manifest["examples"]
    assert manifest["exclusions"][0]["reason"] == "ambiguous"
    assert manifest["summary"]["ambiguous"] == 1


def test_build_manifest_treats_blank_legacy_action_type_as_other() -> None:
    manifest = _build([_pbp(action_id=1, action_type="")])

    assert not manifest["examples"]
    assert manifest["exclusions"][0]["reason"] == "other"


def test_build_manifest_prefers_offset_zero_read_over_higher_confidence() -> None:
    manifest = _build([_pbp(action_id=1)])

    example = manifest["examples"][0]
    assert example["matched_clock_seconds"] == 649
    assert example["clock_sample_offset_seconds"] == 0.0


def test_build_manifest_rejects_blind_video_and_invalid_pbp() -> None:
    with pytest.raises(ValueError, match="sealed blind"):
        _build(
            [_pbp(action_id=1)],
            sealed_blind_video_sha256s=(VIDEO_SHA,),
        )
    with pytest.raises(ValueError, match="game"):
        _build([{**_pbp(action_id=1), "game_id": "wrong"}])
    with pytest.raises(ValueError, match="duplicate"):
        _build([_pbp(action_id=1), _pbp(action_id=1)])
    with pytest.raises(ValueError, match="clock"):
        _build([_pbp(action_id=1, clock="bad")])


def test_manifest_hash_tampering_is_rejected() -> None:
    manifest = _build([_pbp(action_id=1)])
    tampered = copy.deepcopy(manifest)
    tampered["examples"][0]["state"] = "free_throw"

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_pbp_visual_state_manifest(tampered)
