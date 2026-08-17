from __future__ import annotations

import copy

import numpy as np
import pytest

from app.analysis.shot_causal_support import screen_shot_causal_support
from app.analysis.shot_reason_evidence import FEATURE_NAMES, seal_reason_evidence_artifact


def _base_artifact() -> dict[str, object]:
    rows = []
    for group_index, group in enumerate(("a", "b", "c", "d"), start=1):
        source = f"{group}" * 64
        for event_index in range(2):
            positive = event_index == 0
            rows.append(
                {
                    "source_video_sha256": source,
                    "event_id": f"event-{group}-{event_index}",
                    "event_present": positive,
                    "probability": 0.8 if positive else 0.2,
                }
            )
    return {
        "schema_version": "agu.shot-broadcast-fusion-screen.v1",
        "runtime_consumable": False,
        "best_variant": {"name": "base+broadcast_raw"},
        "oof_predictions": rows,
    }


def _reason_artifact(base: dict[str, object]) -> dict[str, object]:
    rows = []
    for row in base["oof_predictions"]:  # type: ignore[index]
        positive = bool(row["event_present"])
        values = {name: 0.0 for name in FEATURE_NAMES}
        values.update(
            {
                "sampled_frame_count": 20.0,
                "candidate_observation_count": 5.0,
                "candidate_player_count": 1.0,
                "ball_rim_hit_count": 2.0 if positive else 0.0,
                "min_ball_rim_distance": 0.5 if positive else 4.0,
                "approach_rise_height_ratio": 0.1 if positive else 0.9,
                "candidate_team_count": 1.0 if positive else 0.0,
            }
        )
        rows.append(
            {
                "source_video_sha256": row["source_video_sha256"],
                "candidate_bundle_sha256": "b" * 64,
                "event_id": row["event_id"],
                "features": [values[name] for name in FEATURE_NAMES],
            }
        )
    return seal_reason_evidence_artifact(
        {
            "source_scene_artifact_sha256": "c" * 64,
            "pose_source_sha256s": ["d" * 64],
            "candidate_bundle_sha256s": ["b" * 64],
            "examples": rows,
        }
    )


def test_causal_support_screen_is_offline_and_reports_baseline_delta() -> None:
    base = _base_artifact()
    reason = _reason_artifact(base)

    result = screen_shot_causal_support(
        base_artifact=base,
        reason_evidence_artifact=reason,
        causal_feature_names=(
            "ball_rim_hit_count",
            "min_ball_rim_distance",
            "approach_rise_height_ratio",
        ),
    )

    assert result["schema_version"] == "agu.shot-causal-support-screen.v2"
    assert result["runtime_consumable"] is False
    assert result["accepted"] is False
    assert "baseline" in result["metrics"]
    assert "causal_support" in result["metrics"]
    assert len(result["folds"]) == 4


def test_causal_threshold_selection_uses_inner_game_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Threshold selection must not consume the zero-initialized OOF buffer."""
    import app.analysis.shot_causal_support as module

    observed_scores: list[np.ndarray] = []
    original = module._select_group_held_threshold

    def capture_scores(scores, labels, groups):
        observed_scores.append(np.asarray(scores, dtype=float).copy())
        return original(scores, labels, groups)

    monkeypatch.setattr(module, "_select_group_held_threshold", capture_scores)
    base = _base_artifact()
    screen_shot_causal_support(
        base_artifact=base,
        reason_evidence_artifact=_reason_artifact(base),
        causal_feature_names=(
            "ball_rim_hit_count",
            "min_ball_rim_distance",
            "approach_rise_height_ratio",
        ),
    )

    assert len(observed_scores) == 8
    assert not any(np.allclose(scores, 0.0) for scores in observed_scores)


def test_causal_support_screen_rejects_duplicate_join_keys() -> None:
    base = _base_artifact()
    reason = _reason_artifact(base)
    reason["examples"] = copy.deepcopy(reason["examples"])
    reason["examples"].append(reason["examples"][0])  # type: ignore[index]

    with pytest.raises(ValueError, match="exactly cover"):
        screen_shot_causal_support(
            base_artifact=base,
            reason_evidence_artifact=reason,
            causal_feature_names=("ball_rim_hit_count",),
        )


def test_causal_support_screen_rejects_unknown_feature() -> None:
    with pytest.raises(ValueError, match="unknown causal feature"):
        screen_shot_causal_support(
            base_artifact=_base_artifact(),
            reason_evidence_artifact=_reason_artifact(_base_artifact()),
            causal_feature_names=("does_not_exist",),
        )


def test_causal_support_screen_rejects_malformed_base_row() -> None:
    base = _base_artifact()
    base["oof_predictions"] = [*base["oof_predictions"], None]  # type: ignore[index]

    with pytest.raises(ValueError, match="base OOF predictions must be objects"):
        screen_shot_causal_support(
            base_artifact=base,
            reason_evidence_artifact=_reason_artifact(_base_artifact()),
            causal_feature_names=("ball_rim_hit_count",),
        )
