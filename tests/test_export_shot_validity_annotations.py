from __future__ import annotations

from pathlib import Path

from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from scripts.export_shot_validity_annotations import export_annotation_template


def test_export_template_is_raw_bound_and_contains_no_answers(tmp_path: Path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"raw")
    event = GameEventResponse(
        event_id="shot-1",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=10,
        end_frame=20,
        confidence=0.7,
        evidence=[
            EventEvidenceResponse(
                evidence_id="candidate",
                kind="ball_rim_proximity_cluster",
                source_video_id="video_001",
                start_frame=10,
                end_frame=20,
                details={"frames": [15]},
            )
        ],
    )
    bundle = seal_raw_only_predictions(
        game_id="game",
        raw_video_paths=[video],
        events=[event],
        config={"fixture": True},
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(bundle.model_dump_json(), encoding="utf-8")

    payload = export_annotation_template(bundle_path)

    assert payload["runtime_consumable"] is False
    assert payload["source_video_sha256"] == bundle.raw_videos[0].sha256
    assert payload["candidate_bundle_sha256"] == bundle.bundle_sha256
    assert payload["examples"][0]["event_present"] is None
    assert payload["examples"][0]["raw_feature_snapshot"]["candidate_confidence"] == 0.7
