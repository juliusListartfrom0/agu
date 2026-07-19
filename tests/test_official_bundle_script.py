import hashlib
import json
from pathlib import Path

import pytest

from app.analysis.official_evaluation import RawOnlyEvaluationError, seal_raw_only_predictions
from app.analysis.official_inference import seal_agu_autonomous_predictions
from app.analysis.schemas import GameEventResponse
from scripts.evaluate_official_bundle import evaluate_bundle


def test_official_bundle_evaluator_binds_alignment_to_bundle_and_truth(tmp_path: Path) -> None:
    raw = tmp_path / "period1.mov"
    raw.write_bytes(b"raw")
    event = GameEventResponse(
        event_id="shot-1",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=98,
        end_frame=102,
        team_id="raw-light",
        primary_player_id="raw-player",
        shot_value=2,
        outcome="made",
        status="codex_confirmed",
        confidence=1,
    )
    bundle = seal_raw_only_predictions(game_id="g1", raw_video_paths=[raw], events=[event], config={})
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(bundle.model_dump_json(), encoding="utf-8")
    truth_path = tmp_path / "truth.csv"
    truth_path.write_text(
        "event_id,start_sec,end_sec,event_type,team_id,player_id,shot_result,shot_value\n"
        "e1,3.2,3.4,2分投篮,truth-team,truth-player,命中,2\n",
        encoding="utf-8",
    )
    alignment = {
        "schema_version": "agu.identity-alignment.v1",
        "prediction_bundle_sha256": bundle.bundle_sha256,
        "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "player_id_map": {"raw-player": "truth-player"},
        "team_id_map": {"raw-light": "truth-team"},
    }
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(json.dumps(alignment), encoding="utf-8")

    report = evaluate_bundle(
        bundle_path=bundle_path,
        truth_path=truth_path,
        fps=30,
        tolerance_frames=5,
        identity_alignment_path=alignment_path,
    )

    assert report["reviewed_identity_aligned_strict"]["metrics"]["f1"] == 1
    alignment["prediction_bundle_sha256"] = "wrong"
    alignment_path.write_text(json.dumps(alignment), encoding="utf-8")
    with pytest.raises(RawOnlyEvaluationError, match="bundle hash mismatch"):
        evaluate_bundle(
            bundle_path=bundle_path,
            truth_path=truth_path,
            fps=30,
            tolerance_frames=5,
            identity_alignment_path=alignment_path,
        )


def test_official_bundle_evaluator_records_explicit_partial_event_scope(tmp_path: Path) -> None:
    raw = tmp_path / "period1.mov"
    raw.write_bytes(b"raw")
    events = [
        GameEventResponse(
            event_id="shot",
            revision=1,
            event_type="field_goal_attempt",
            source_video_id="video_001",
            start_frame=100,
            end_frame=100,
            status="needs_review",
        ),
        GameEventResponse(
            event_id="turnover-outside-partial-truth",
            revision=1,
            event_type="turnover",
            source_video_id="video_001",
            start_frame=100,
            end_frame=100,
            status="needs_review",
        ),
    ]
    bundle = seal_raw_only_predictions(game_id="g1", raw_video_paths=[raw], events=events, config={})
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(bundle.model_dump_json(), encoding="utf-8")
    truth_path = tmp_path / "partial.csv"
    truth_path.write_text(
        "event_id,time_sec,event_type\nshot-truth,3.333333,shot\n",
        encoding="utf-8",
    )

    report = evaluate_bundle(
        bundle_path=bundle_path,
        truth_path=truth_path,
        fps=30,
        tolerance_frames=1,
        event_types=["field_goal_attempt"],
    )

    assert report["evaluation_scope"] == {
        "event_types": ["field_goal_attempt"],
        "explicit_partial_category_scope": True,
    }
    assert report["tiers"]["candidate_localization"]["metrics"]["f1"] == 1


def test_acceptance_mode_rejects_reviewed_bundle_and_scores_agu_only(tmp_path: Path) -> None:
    raw = tmp_path / "period1.mov"
    raw.write_bytes(b"raw")
    truth_path = tmp_path / "truth.csv"
    truth_path.write_text(
        "event_id,time_sec,event_type,team_id,player_id,shot_result,shot_value\n"
        "e1,3.333333,shot,raw-dark,p1,命中,2\n",
        encoding="utf-8",
    )
    codex_event = GameEventResponse(
        event_id="shot",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=100,
        end_frame=100,
        team_id="raw-dark",
        primary_player_id="p1",
        outcome="made",
        shot_value=2,
        status="codex_confirmed",
        reviewer="codex:truth-maker",
    )
    reviewed = seal_raw_only_predictions(
        game_id="g1", raw_video_paths=[raw], events=[codex_event], config={}
    )
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(reviewed.model_dump_json(), encoding="utf-8")
    with pytest.raises(RawOnlyEvaluationError, match="producer=agu"):
        evaluate_bundle(
            bundle_path=reviewed_path,
            truth_path=truth_path,
            fps=30,
            tolerance_frames=1,
            require_agu_autonomous=True,
        )

    automatic = codex_event.model_copy(
        update={"status": "edge_vlm_confirmed", "reviewer": "edge_vlm:ollama/qwen"}
    )
    autonomous = seal_agu_autonomous_predictions(
        game_id="g1",
        raw_video_paths=[raw],
        events=[automatic],
        config={},
        candidate_backend="traditional_cv",
        semantic_backend="ollama",
        semantic_model="qwen-vl",
    )
    autonomous_path = tmp_path / "autonomous.json"
    autonomous_path.write_text(autonomous.model_dump_json(), encoding="utf-8")
    report = evaluate_bundle(
        bundle_path=autonomous_path,
        truth_path=truth_path,
        fps=30,
        tolerance_frames=1,
        require_agu_autonomous=True,
    )

    assert report["acceptance_mode"] == "agu_autonomous"
    assert report["tiers"]["automatic_strict"]["metrics"]["f1"] == 1
