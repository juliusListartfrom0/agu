from __future__ import annotations

from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from scripts.evaluate_window_causal_evidence import evaluate_causal_evidence


def test_evaluates_only_versioned_window_evidence(tmp_path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"raw")
    events = []
    for index, predicted in enumerate((True, True, False), start=1):
        evidence = []
        if predicted:
            evidence.append(
                EventEvidenceResponse(
                    evidence_id=f"shot-{index}:causal",
                    kind="ball_rim_proximity_cluster",
                    source_video_id="video_001",
                    start_frame=index,
                    end_frame=index,
                    confidence=0.8,
                    details={
                        "causal_evidence_version": "agu_window_ball_rim_pose_v1"
                    },
                )
            )
        events.append(
            GameEventResponse(
                event_id=f"shot-{index}",
                revision=1,
                event_type="field_goal_attempt",
                source_video_id="video_001",
                start_frame=index,
                end_frame=index + 1,
                status="needs_review",
                evidence=evidence,
            )
        )
    bundle = seal_raw_only_predictions(
        game_id="game",
        raw_video_paths=[video],
        events=events,
        config={"fixture": True},
    )
    labels = {
        "schema_version": "agu.shot-validity-labels.v1",
        "runtime_consumable": False,
        "source_video_sha256": bundle.raw_videos[0].sha256,
        "candidate_bundle_sha256": bundle.bundle_sha256,
        "examples": [
            {"event_id": "shot-1", "event_present": True},
            {"event_id": "shot-2", "event_present": False},
            {"event_id": "shot-3", "event_present": True},
        ],
    }

    metrics = evaluate_causal_evidence(bundle.model_dump(mode="json"), labels)

    assert metrics == {
        "labeled_count": 3,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 0,
        "precision": 0.5,
        "recall": 0.5,
    }
