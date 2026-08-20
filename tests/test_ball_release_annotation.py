from __future__ import annotations

import pytest

from app.analysis.ball_release_annotation import (
    FRAME_FRACTIONS,
    apply_ball_release_label_corrections,
    build_hard_review_plan,
    derive_ball_release_label_corrections,
    seal_ball_release_review,
    verify_ball_release_plan,
    verify_ball_release_review,
    verify_scene_fusion_screen,
)


def _examples() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scene_rows = []
    predictions = []
    for game in ("game-a", "game-b", "game-c"):
        for index, (label, probability) in enumerate(
            ((True, 0.1), (True, 0.4), (True, 0.9), (False, 0.8), (False, 0.6), (False, 0.2))
        ):
            event_id = f"{game}-{index}"
            scene_rows.append(
                {
                    "source_video_sha256": game,
                    "candidate_bundle_sha256": f"bundle-{game}",
                    "event_id": event_id,
                    "event_present": label,
                }
            )
            predictions.append(
                {
                    "source_video_sha256": game,
                    "event_id": event_id,
                    "event_present": label,
                    "probability": probability,
                }
            )
    return scene_rows, predictions


def test_hard_review_plan_is_balanced_but_does_not_expose_labels() -> None:
    scene_rows, predictions = _examples()
    events = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        ): {"start_frame": 100, "end_frame": 200}
        for row in scene_rows
    }

    plan = build_hard_review_plan(
        training_manifest_sha256="manifest",
        scene_artifact_sha256="scene",
        fusion_artifact_sha256="fusion",
        scene_examples=scene_rows,
        oof_predictions=predictions,
        events=events,
        per_game_per_class=1,
    )

    verified = verify_ball_release_plan(plan)
    assert verified["runtime_consumable"] is False
    assert verified["selection_summary"] == {
        "games": 3,
        "examples": 6,
        "positive_hard_examples": 3,
        "negative_hard_examples": 3,
    }
    assert {row["event_id"] for row in verified["examples"]} == {
        "game-a-0",
        "game-a-3",
        "game-b-0",
        "game-b-3",
        "game-c-0",
        "game-c-3",
    }
    assert all("event_present" not in row and "probability" not in row for row in verified["examples"])
    assert verified["examples"][0]["frame_numbers"] == [
        int(round(100 + 100 * fraction)) for fraction in FRAME_FRACTIONS
    ]
    excluded = {
        (row["source_video_sha256"], row["event_id"])
        for row in verified["examples"]
    }
    second = build_hard_review_plan(
        training_manifest_sha256="manifest",
        scene_artifact_sha256="scene",
        fusion_artifact_sha256="fusion",
        scene_examples=scene_rows,
        oof_predictions=predictions,
        events=events,
        per_game_per_class=1,
        excluded_event_keys=excluded,
        excluded_plan_sha256s=[verified["artifact_sha256"]],
    )
    assert not excluded & {
        (row["source_video_sha256"], row["event_id"])
        for row in second["examples"]
    }
    assert second["excluded_examples"] == 6


def test_ball_release_plan_is_hash_bound() -> None:
    scene_rows, predictions = _examples()
    events = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        ): {"start_frame": 10, "end_frame": 20}
        for row in scene_rows
    }
    plan = build_hard_review_plan(
        training_manifest_sha256="manifest",
        scene_artifact_sha256="scene",
        fusion_artifact_sha256="fusion",
        scene_examples=scene_rows,
        oof_predictions=predictions,
        events=events,
        per_game_per_class=1,
    )
    plan["examples"][0]["frame_numbers"][0] += 1

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_ball_release_plan(plan)


def test_scene_fusion_screen_is_hash_bound_and_training_only() -> None:
    import hashlib
    import json

    payload = {
        "schema_version": "agu.shot-validity-scene-fusion-screen.v1",
        "runtime_consumable": False,
        "training_manifest_sha256": "manifest",
        "scene_embedding_artifact_sha256": "scene",
        "best_variant": {"oof_predictions": [{"event_id": "event-1"}]},
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

    assert verify_scene_fusion_screen(payload)["artifact_sha256"]
    payload["runtime_consumable"] = True
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_scene_fusion_screen(payload)


def test_ball_release_review_requires_complete_plan_coverage() -> None:
    scene_rows, predictions = _examples()
    events = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        ): {"start_frame": 10, "end_frame": 20}
        for row in scene_rows
    }
    plan = build_hard_review_plan(
        training_manifest_sha256="manifest",
        scene_artifact_sha256="scene",
        fusion_artifact_sha256="fusion",
        scene_examples=scene_rows,
        oof_predictions=predictions,
        events=events,
        per_game_per_class=1,
    )
    reviews = [
        {
            "source_video_sha256": row["source_video_sha256"],
            "candidate_bundle_sha256": row["candidate_bundle_sha256"],
            "event_id": row["event_id"],
            "release_observed": None,
            "ball_moves_toward_rim": None,
            "rim_arrival_observed": None,
            "free_throw_formation": False,
            "replay_or_stoppage": False,
            "review_confidence": "uncertain",
            "frame_observations": [
                {
                    "frame": frame,
                    "ball_visible": None,
                    "selected_path_is_ball": None,
                    "ball_bbox": None,
                }
                for frame in row["frame_numbers"]
            ],
        }
        for row in plan["examples"]
    ]
    sealed = seal_ball_release_review(
        {"plan_sha256": plan["artifact_sha256"], "reviews": reviews},
        plan=plan,
    )

    assert verify_ball_release_review(sealed, plan=plan)["runtime_consumable"] is False
    reviews.pop()
    with pytest.raises(ValueError, match="exactly cover"):
        seal_ball_release_review(
            {"plan_sha256": plan["artifact_sha256"], "reviews": reviews},
            plan=plan,
        )


def test_conservative_corrections_exclude_replay_and_fix_live_release() -> None:
    scene_rows, predictions = _examples()
    events = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        ): {"start_frame": 10, "end_frame": 20}
        for row in scene_rows
    }
    plan = build_hard_review_plan(
        training_manifest_sha256="manifest",
        scene_artifact_sha256="scene",
        fusion_artifact_sha256="fusion",
        scene_examples=scene_rows,
        oof_predictions=predictions,
        events=events,
        per_game_per_class=1,
    )
    reviews = []
    for index, row in enumerate(plan["examples"]):
        replay = index == 0
        reviews.append(
            {
                "source_video_sha256": row["source_video_sha256"],
                "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                "event_id": row["event_id"],
                "release_observed": True,
                "ball_moves_toward_rim": True,
                "rim_arrival_observed": True,
                "free_throw_formation": False,
                "replay_or_stoppage": replay,
                "review_confidence": "high",
                "frame_observations": [
                    {
                        "frame": frame,
                        "ball_visible": None,
                        "selected_path_is_ball": None,
                        "ball_bbox": None,
                    }
                    for frame in row["frame_numbers"]
                ],
            }
        )
    review = seal_ball_release_review(
        {"plan_sha256": plan["artifact_sha256"], "reviews": reviews},
        plan=plan,
    )
    corrections = derive_ball_release_label_corrections(
        plan=plan,
        review=review,
        scene_artifact_sha256="scene",
        scene_examples=scene_rows,
    )
    relabeled = apply_ball_release_label_corrections(
        scene_rows, corrections=corrections
    )
    original = {
        (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        ): row["event_present"]
        for row in scene_rows
    }
    corrected = {
        (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        ): row["event_present"]
        for row in relabeled
    }
    first_key = (
        plan["examples"][0]["source_video_sha256"],
        plan["examples"][0]["candidate_bundle_sha256"],
        plan["examples"][0]["event_id"],
    )
    assert corrected[first_key] is False
    assert corrected != original
