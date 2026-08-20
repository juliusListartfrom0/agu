from __future__ import annotations

from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from scripts.rebind_shot_validity_labels import rebind_labels


def _bundle(tmp_path, *, evidence: bool = False, start_frame: int = 10):
    video = tmp_path / "game.mp4"
    if not video.exists():
        video.write_bytes(b"raw-game")
    event_evidence = []
    if evidence:
        event_evidence.append(
            EventEvidenceResponse(
                evidence_id="shot-1:causal",
                kind="ball_rim_proximity_cluster",
                source_video_id="video_001",
                start_frame=12,
                end_frame=15,
                confidence=0.7,
                details={"minimum_normalized_distance": 1.2},
            )
        )
    return seal_raw_only_predictions(
        game_id="game",
        raw_video_paths=[video],
        events=[
            GameEventResponse(
                event_id="shot-1",
                event_type="field_goal_attempt",
                source_video_id="video_001",
                start_frame=start_frame,
                end_frame=20,
                confidence=0.8,
                revision=1,
                reason="enriched" if evidence else "source",
                evidence=event_evidence,
            )
        ],
        config={"enriched": evidence},
        model_provenance={"backend": "test"},
    )


def test_rebinds_only_evidence_enriched_bundle(tmp_path):
    source = _bundle(tmp_path)
    enriched = _bundle(tmp_path, evidence=True)
    labels = {
        "schema_version": "agu.shot-validity-labels.v1",
        "purpose": "training_annotation_only",
        "runtime_consumable": False,
        "source_video_sha256": source.raw_videos[0].sha256,
        "candidate_bundle_sha256": source.bundle_sha256,
        "examples": [{"event_id": "shot-1", "event_present": True}],
    }

    rebound = rebind_labels(
        source.model_dump(mode="json"),
        enriched.model_dump(mode="json"),
        labels,
    )

    assert rebound["candidate_bundle_sha256"] == enriched.bundle_sha256
    assert (
        rebound["evidence_rebind"]["source_candidate_bundle_sha256"]
        == source.bundle_sha256
    )


def test_rebind_rejects_candidate_timing_change(tmp_path):
    source = _bundle(tmp_path)
    changed = _bundle(tmp_path, evidence=True, start_frame=11)
    labels = {
        "schema_version": "agu.shot-validity-labels.v1",
        "runtime_consumable": False,
        "source_video_sha256": source.raw_videos[0].sha256,
        "candidate_bundle_sha256": source.bundle_sha256,
        "examples": [{"event_id": "shot-1", "event_present": True}],
    }

    try:
        rebind_labels(
            source.model_dump(mode="json"),
            changed.model_dump(mode="json"),
            labels,
        )
    except ValueError as error:
        assert "candidate identity" in str(error)
    else:
        raise AssertionError("timing changes must fail closed")
