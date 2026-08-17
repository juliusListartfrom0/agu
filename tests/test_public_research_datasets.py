from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.analysis.public_research_datasets import (
    REQUIRED_PLAYER_STAT_FIELDS,
    build_continuous_causal_source_gate_audit,
    build_nba_research_plan,
    build_public_game_pair_handoff,
    build_public_research_catalog,
    evaluate_continuous_causal_source_gate,
    verify_public_research_plan,
)


def test_continuous_causal_source_gate_fails_closed_on_missing_checks() -> None:
    source = {
        "dataset_id": "candidate",
        "source_url": "https://example.test/candidate",
        "source_revision": "rev-1",
        "declared_size_bytes": 1,
        "rights_cleared": True,
        "broadcast_diverse": True,
        "continuous_five_by_five_video": True,
        "subsecond_ball_hand_rim_outcome_labels": True,
        # Deliberately omit exhaustive_non_shot_hard_negatives.
    }

    result = evaluate_continuous_causal_source_gate(source)

    assert result["eligible_for_payload_download"] is False
    assert result["download_decision"] == "deny"
    assert result["checks"]["exhaustive_non_shot_hard_negatives"] is False
    assert result["missing_checks"] == ["exhaustive_non_shot_hard_negatives"]
    assert result["runtime_consumable"] is False


def test_continuous_causal_source_gate_requires_explicit_true_booleans() -> None:
    source = {
        "dataset_id": "candidate",
        "source_url": "https://example.test/candidate",
        "source_revision": "rev-1",
        "rights_cleared": "true",
        "broadcast_diverse": True,
        "continuous_five_by_five_video": True,
        "subsecond_ball_hand_rim_outcome_labels": True,
        "exhaustive_non_shot_hard_negatives": True,
    }

    result = evaluate_continuous_causal_source_gate(source)

    assert result["checks"]["rights_cleared"] is False
    assert result["eligible_for_payload_download"] is False
    assert "rights_cleared" in result["missing_checks"]


def test_continuous_causal_source_gate_audit_is_metadata_only_and_fail_closed() -> None:
    audit = build_continuous_causal_source_gate_audit(
        [
            {
                "dataset_id": "pass",
                "source_url": "https://example.test/pass",
                "source_revision": "rev-1",
                "declared_size_bytes": 1,
                "rights_cleared": True,
                "broadcast_diverse": True,
                "continuous_five_by_five_video": True,
                "subsecond_ball_hand_rim_outcome_labels": True,
                "exhaustive_non_shot_hard_negatives": True,
            },
            {
                "dataset_id": "reject",
                "source_url": "https://example.test/reject",
                "source_revision": "rev-2",
                "rights_cleared": True,
            },
        ],
        generated_on="2026-08-08",
    )

    assert audit["schema_version"] == "agu.continuous-causal-source-gate.v1"
    assert audit["payload_downloads_performed"] == 0
    assert audit["eligible_source_count"] == 1
    assert audit["sources"][0]["download_decision"] == "allow"
    assert audit["sources"][1]["download_decision"] == "deny"
    assert audit["runtime_consumable"] is False


def test_current_continuous_causal_candidates_do_not_authorize_payload_download() -> None:
    from scripts.audit_continuous_causal_source_gate import current_candidates

    audit = build_continuous_causal_source_gate_audit(
        current_candidates(), generated_on="2026-08-08"
    )

    assert audit["payload_downloads_performed"] == 0
    assert audit["eligible_source_count"] == 0
    assert all(item["download_decision"] == "deny" for item in audit["sources"])


def test_current_source_gate_includes_nsva_metadata_only_candidate() -> None:
    from scripts.audit_continuous_causal_source_gate import current_candidates

    candidates = {item["dataset_id"]: item for item in current_candidates()}

    assert candidates["nsva-2022"]["declared_size_bytes"] == 23_167
    assert candidates["nsva-2022"]["rights_cleared"] is False
    assert candidates["nsva-2022"]["continuous_five_by_five_video"] is False


def test_current_source_gate_includes_gcb_metadata_only_candidate() -> None:
    from scripts.audit_continuous_causal_source_gate import current_candidates

    candidates = {item["dataset_id"]: item for item in current_candidates()}

    assert candidates["gamecommbench-basketball"]["declared_size_bytes"] == 21_761_000_000
    assert candidates["gamecommbench-basketball"]["rights_cleared"] is False
    assert candidates["gamecommbench-basketball"]["continuous_five_by_five_video"] is False


def test_current_source_gate_includes_mev_metadata_only_candidate() -> None:
    from scripts.audit_continuous_causal_source_gate import current_candidates

    candidates = {item["dataset_id"]: item for item in current_candidates()}

    assert candidates["multi-event-video-mev"]["declared_size_bytes"] == 76_009_107_602
    assert candidates["multi-event-video-mev"]["rights_cleared"] is False
    assert candidates["multi-event-video-mev"]["continuous_five_by_five_video"] is False


def test_current_source_gate_records_new_online_action_candidates_as_denied() -> None:
    from scripts.audit_continuous_causal_source_gate import current_candidates

    candidates = {item["dataset_id"]: item for item in current_candidates()}

    for dataset_id in ("sportsmot", "spacejam-basketball-actions"):
        assert dataset_id in candidates
        assert candidates[dataset_id]["rights_cleared"] is False
        assert candidates[dataset_id]["continuous_five_by_five_video"] is False
        assert candidates[dataset_id]["subsecond_ball_hand_rim_outcome_labels"] is False
        assert candidates[dataset_id]["exhaustive_non_shot_hard_negatives"] is False


def test_public_research_catalog_records_new_online_action_candidates() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}

    assert sources["sportsmot"]["data_license"] == "CC-BY-NC-4.0"
    assert sources["sportsmot"]["training_media_eligible"] is False
    assert sources["spacejam-basketball-actions"]["data_license"] == "MIT-repository"
    assert sources["spacejam-basketball-actions"]["training_media_eligible"] is False


def _write_game(
    root: Path,
    slug: str,
    *,
    youtube_id: str,
    person_ids: tuple[int, ...],
    omit_stat: str | None = None,
) -> None:
    game_root = root / "games" / slug
    game_root.mkdir(parents=True)
    metadata = {
        "source_game": {
            "id": youtube_id,
            "url": f"https://www.youtube.com/watch?v={youtube_id}",
            "duration": 6000,
        },
        "official_game": {
            "game_id": f"game-{slug}",
            "game_date": "2026-01-01",
            "away_abbr": "AAA",
            "home_abbr": "BBB",
        },
    }
    (game_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    statistics = {
        "minutes": "10:00",
        "fieldGoalsMade": 3,
        "fieldGoalsAttempted": 5,
        "threePointersMade": 1,
        "threePointersAttempted": 2,
        "reboundsOffensive": 1,
        "reboundsDefensive": 2,
        "assists": 1,
        "steals": 1,
        "blocks": 1,
    }
    if omit_stat:
        statistics.pop(omit_stat)
    box_rows = [
        {
            "row_type": "player",
            "person_id": person_id,
            "first_name": f"Player{person_id}",
            "family_name": "Test",
            "team_tricode": "AAA" if index % 2 == 0 else "BBB",
            "statistics": statistics,
        }
        for index, person_id in enumerate(person_ids)
    ]
    (game_root / "box-score.jsonl").write_text(
        "\n".join(json.dumps(row) for row in box_rows) + "\n",
        encoding="utf-8",
    )
    actions = [
        ("Made Shot", "Player 3PT Jump Shot (Player 1 AST)"),
        ("Missed Shot", "Player MISS 2PT Shot"),
        ("Rebound", "Player REBOUND"),
        ("Turnover", "Player Turnover"),
        ("", "Player STEAL"),
        ("", "Player BLOCK"),
    ]
    (game_root / "play-by-play.jsonl").write_text(
        "\n".join(
            json.dumps({"actionType": action_type, "description": description}) for action_type, description in actions
        )
        + "\n",
        encoding="utf-8",
    )


def test_public_research_catalog_keeps_dataset_roles_and_rights_separate() -> None:
    catalog = build_public_research_catalog()
    sources = {item["dataset_id"]: item for item in catalog["sources"]}

    assert catalog["runtime_consumable"] is False
    assert sources["trackid3x3"]["data_license"] == "CC-BY-4.0"
    assert sources["trackid3x3"]["adapter_boundary"] == ("offline_bbox_track_and_pose_import")
    assert sources["nba-games-metadata"]["media_included"] is False
    assert sources["bard-2026"]["redistribution_policy"] == "external_media_rights_not_granted"
    assert sources["bard-2026"]["media_included"] is True
    assert sources["bard-2026"]["adapter_boundary"] == ("offline_embedded_validation_video_and_annotation_import")
    assert sources["e-bard-detection"]["data_license"] == "CC-BY-4.0"
    assert sources["e-bard-detection"]["adapter_boundary"] == "offline_yolo_detection_import"
    assert sources["e-bard-object-classification"]["data_license"] == "CC-BY-4.0"
    assert sources["e-bard-object-classification"]["role"] == (
        "basketball_object_role_vlm_pretraining"
    )
    assert sources["e-bard-object-classification"]["adapter_boundary"] == (
        "offline_object_role_crop_classification_vlm_finetune_only"
    )
    assert sources["e-bard-team-attribution"]["data_license"] == "CC-BY-4.0"
    assert sources["e-bard-team-attribution"]["role"] == (
        "basketball_team_color_auxiliary_pretraining"
    )
    assert sources["e-bard-team-attribution"]["runtime_consumable"] is False
    assert sources["e-bard-team-attribution"]["training_media_eligible"] is False
    assert sources["e-bard-team-attribution"]["adapter_boundary"] == (
        "offline_team_color_crop_import_only"
    )
    assert sources["open-images-v7-ball-subset"]["data_license"] == (
        "CC-BY-4.0-annotations-per-image-image-license"
    )
    assert sources["open-images-v7-ball-subset"]["adapter_boundary"] == (
        "offline_hash_bound_ball_verifier_training_only"
    )
    assert sources["youtube-boundingboxes"]["data_license"] == "CC-BY-4.0"
    assert sources["youtube-boundingboxes"]["media_included"] is False
    assert sources["youtube-boundingboxes"]["redistribution_policy"] == (
        "reject_no_basketball_ball_class_or_event_labels"
    )
    assert sources["youtube-boundingboxes"]["adapter_boundary"] == (
        "research_reference_only_no_agu_training_import"
    )
    assert sources["leharris3-basketball-shot-test"]["data_license"] == "MIT-tagged-gated"
    assert sources["leharris3-basketball-shot-test"]["media_included"] is True
    assert sources["leharris3-basketball-shot-test"]["redistribution_policy"] == (
        "reject_gated_empty_card_and_unverified_broadcast_provenance"
    )
    assert sources["leharris3-basketball-shot-test"]["adapter_boundary"] == (
        "research_reference_only_no_media_or_annotation_import"
    )
    assert sources["nus-basketball-detection-cs5260"]["source_revision"] == (
        "3ee1a0decfde199117b3a99d79bcf136c902ebc5"
    )
    assert sources["nus-basketball-detection-cs5260"]["data_license"] is None
    assert sources["nus-basketball-detection-cs5260"]["media_included"] is True
    assert sources["nus-basketball-detection-cs5260"]["runtime_consumable"] is False
    assert sources["nus-basketball-detection-cs5260"]["training_media_eligible"] is False
    assert sources["nus-basketball-detection-cs5260"]["redistribution_policy"] == (
        "reject_missing_license_and_broadcast_provenance_no_continuous_causal_labels"
    )
    assert sources["nus-basketball-detection-cs5260"]["audit_artifact"] == (
        "analysis_outputs/public_research/nus_basketball_detection_audit_v1.json"
    )
    assert sources["pl-nba-v1"]["source_revision"] == (
        "0b3d5013b5828004062ed27b10c9eddfca313711"
    )
    assert sources["pl-nba-v1"]["data_license"] == "CC-BY-NC-4.0-declared-upstream"
    assert sources["pl-nba-v1"]["media_included"] is True
    assert sources["pl-nba-v1"]["runtime_consumable"] is False
    assert sources["pl-nba-v1"]["training_media_eligible"] is False
    assert sources["pl-nba-v1"]["redistribution_policy"] == (
        "noncommercial_annotation_only_no_underlying_nba_media_rights_or_runtime"
    )
    assert sources["pl-nba-v1"]["audit_artifact"] == (
        "analysis_outputs/public_research/pl_nba_annotation_audit_v1.json"
    )
    assert sources["pl-nba-v1"]["bounded_annotation_manifest"] == (
        "dataset/public_sources/pl_nba_v1/manifest.json"
    )
    assert sources["tal3x3-v1"]["source_revision"] == (
        "f1093d62818c3e8930561fda5939db19c0981d95"
    )
    assert sources["tal3x3-v1"]["data_license"] is None
    assert sources["tal3x3-v1"]["media_included"] is False
    assert sources["tal3x3-v1"]["runtime_consumable"] is False
    assert sources["tal3x3-v1"]["training_media_eligible"] is False
    assert sources["tal3x3-v1"]["redistribution_policy"] == (
        "retain_annotation_only_until_dataset_license_and_source_media_rights_verified"
    )
    assert sources["tal3x3-v1"]["audit_artifact"] == (
        "analysis_outputs/public_research/tal3x3_annotation_audit_v1.json"
    )
    assert sources["tal3x3-v1"]["bounded_annotation_manifest"] == (
        "dataset/public_sources/tal3x3_v1/manifest.json"
    )
    assert sources["muvy-basketball-v1"]["source_revision"] == (
        "10.5281/zenodo.13883315@2024-10-03"
    )
    assert sources["muvy-basketball-v1"]["data_license"] == "CC-BY-4.0"
    assert sources["muvy-basketball-v1"]["media_included"] is True
    assert sources["muvy-basketball-v1"]["runtime_consumable"] is False
    assert sources["muvy-basketball-v1"]["training_media_eligible"] is False
    assert sources["muvy-basketball-v1"]["redistribution_policy"] == (
        "attribution_required_bounded_multiview_media_only_no_outcome_labels"
    )
    assert sources["muvy-basketball-v1"]["audit_artifact"] == (
        "analysis_outputs/public_research/muvy_basketball_audit_v1.json"
    )
    assert sources["muvy-basketball-v1"]["bounded_annotation_manifest"] == (
        "dataset/public_sources/muvy_v1/metadata_manifest.json"
    )
    assert sources["muvy-basketball-v1"]["bounded_media_manifest"] == (
        "dataset/public_sources/muvy_v1/media_manifest.json"
    )
    assert sources["basketball-51"]["role"] == ("shot_type_and_outcome_pretraining")
    assert sources["basketball-51"]["adapter_boundary"] == ("offline_game_grouped_clip_import")
    assert sources["f-16-nba-shot-test"]["data_license"] == (
        "Apache-2.0-as-declared-by-Hugging-Face-card"
    )
    assert sources["f-16-nba-shot-test"]["media_included"] is True
    assert sources["f-16-nba-shot-test"]["statistics_included"] is True
    assert sources["f-16-nba-shot-test"]["redistribution_policy"] == (
        "do_not_redistribute_underlying_nba_video_without_rights_review"
    )
    assert sources["f-16-nba-shot-test"]["adapter_boundary"] == (
        "offline_shot_outcome_screen_only_no_ball_hand_rim_timing"
    )
    assert sources["yerx-bb-free-throw"]["data_license"] is None
    assert sources["yerx-bb-free-throw"]["redistribution_policy"] == (
        "reject_download_no_dataset_card_license_or_media_provenance"
    )
    assert sources["basketball-detection-github"]["data_license"] is None
    assert sources["basketball-detection-github"]["media_included"] is True
    assert sources["basketball-detection-github"]["statistics_included"] is False
    assert sources["basketball-detection-github"]["runtime_consumable"] is False
    assert sources["basketball-detection-github"]["training_media_eligible"] is False
    assert sources["basketball-detection-github"]["redistribution_policy"] == (
        "reject_download_no_repository_license_or_temporal_ball_hand_rim_labels"
    )
    assert sources["basketball-detection-github"]["adapter_boundary"] == (
        "research_reference_only_no_media_or_annotation_import"
    )
    assert sources["baskethar"]["data_license"] == "Apache-2.0"
    assert sources["baskethar"]["role"] == "wearable_signal_research_reference"
    assert sources["baskethar"]["media_included"] is False
    assert sources["baskethar"]["adapter_boundary"] == ("research_reference_only_no_visual_import")
    assert sources["baskethar"]["redistribution_policy"] == ("visual_pretraining_rejected_video_link_removed_upstream")
    assert sources["basketevent-playnet"]["data_license"] is None
    assert sources["basketevent-playnet"]["code_license"] is None
    assert sources["basketevent-playnet"]["media_included"] is False
    assert sources["basketevent-playnet"]["redistribution_policy"] == (
        "do_not_download_or_train_until_upstream_adds_compatible_terms_and_media"
    )
    assert sources["basketevent-playnet"]["adapter_boundary"] == (
        "research_evidence_only_no_code_weight_or_annotation_import"
    )
    assert sources["nsva-2022"]["data_license"] == "fair-noncommercial-research-license"
    assert sources["nsva-2022"]["redistribution_policy"] == (
        "reject_for_open_commercial_training_noncommercial_license"
    )
    assert sources["vc-nba-2022"]["data_license"] is None
    assert sources["vc-nba-2022"]["media_included"] is True
    assert sources["vc-nba-2022"]["statistics_included"] is True
    assert sources["vc-nba-2022"]["redistribution_policy"] == (
        "reject_download_paper_only_commercial_source_no_public_data_license"
    )
    assert sources["vc-nba-2022"]["adapter_boundary"] == (
        "paper_reference_only_no_video_or_annotation_import"
    )
    assert sources["ncaa-basketball-attention"]["data_license"] is None
    assert sources["ncaa-basketball-attention"]["media_included"] is False
    assert sources["ncaa-basketball-attention"]["redistribution_policy"] == (
        "reject_download_dead_official_host_and_no_declared_dataset_license"
    )
    assert sources["poseshot-free-throw"]["adapter_boundary"] == (
        "paper_reference_only_request_required_no_public_dataset"
    )
    assert sources["nba-pbp-video-dataset"]["code_license"] is None
    assert sources["nba-pbp-video-dataset"]["redistribution_policy"] == (
        "reject_download_no_declared_license_and_multi_terabyte_external_media"
    )
    assert sources["fineaction"]["adapter_boundary"] == ("reject_for_current_basketball_event_gap_generic_taxonomy")
    assert sources["unidata-basketball-tracking"]["data_license"] == ("CC-BY-NC-ND-4.0")
    assert sources["unidata-basketball-tracking"]["adapter_boundary"] == (
        "reject_noncommercial_no_derivatives_sample_only"
    )
    assert sources["basket-skill-estimation"]["data_license"] == "Apache-2.0"
    assert sources["basket-skill-estimation"]["media_included"] is True
    assert sources["basket-skill-estimation"]["adapter_boundary"] == (
        "research_reference_only_no_ball_track_or_event_timing_labels"
    )
    assert sources["svi-bench-2026"]["data_license"] == "CC-BY-NC-4.0"
    assert sources["svi-bench-2026"]["redistribution_policy"] == (
        "reject_download_noncommercial_gated_edu_only_no_redistribution"
    )
    assert sources["svi-bench-2026"]["adapter_boundary"] == ("paper_and_taxonomy_reference_only_no_import")
    assert sources["koppolusameer-court-keypoints-2026"]["data_license"] == (
        "CC-BY-4.0-Roboflow-project-HF-card-conflicts-MIT"
    )
    assert sources["koppolusameer-court-keypoints-2026"]["code_license"] == ("AGPL-3.0-model")
    assert (
        sources["koppolusameer-court-keypoints-2026"]["redistribution_policy"]
        == "reject_runtime_unverified_nba_media_rights_missing_keypoint_semantics"
    )
    assert sources["koppolusameer-court-keypoints-2026"]["adapter_boundary"] == (
        "research_probe_only_weight_deleted_after_cross_game_failure"
    )
    assert sources["kalicalib-deepsport"]["code_license"] == "CeCILL-2.1"
    assert sources["kalicalib-deepsport"]["data_license"] == ("CC-BY-NC-ND-4.0-upstream-DeepSport")
    assert sources["kalicalib-deepsport"]["adapter_boundary"] == (
        "reject_noncommercial_upstream_data_and_trained_weights"
    )
    assert sources["multisports"]["data_license"] == "CC-BY-NC-4.0"
    assert sources["multisports"]["code_license"] == "MIT"
    assert sources["multisports"]["adapter_boundary"] == (
        "paper_and_taxonomy_reference_only_no_media_or_annotation_import"
    )
    assert sources["finesports"]["data_license"] == ("custom-research-only-release-agreement")
    assert sources["finesports"]["code_license"] is None
    assert sources["finesports"]["redistribution_policy"] == (
        "reject_open_commercial_training_signed_research_only_agreement"
    )
    assert sources["shot7m2"]["data_license"] == "ECL-2.0-as-declared-by-Hub"
    assert sources["shot7m2"]["media_included"] is False
    assert sources["shot7m2"]["adapter_boundary"] == (
        "research_reference_only_no_import_single_agent_synthetic_pose_mismatch"
    )
    assert sources["cvballtracking"]["data_license"] is None
    assert sources["cvballtracking"]["redistribution_policy"] == (
        "reject_code_and_checkpoint_import_no_repository_license"
    )
    assert sources["cvballtracking"]["adapter_boundary"] == (
        "paper_reference_only_generic_coco_yolov3_not_basketball_weight"
    )
    assert sources["hardik-ai-basketball"]["data_license"] is None
    assert sources["hardik-ai-basketball"]["code_license"] == ("OpenPose-noncommercial-research-only")
    assert sources["hardik-ai-basketball"]["adapter_boundary"] == (
        "reject_code_weight_and_private_training_data_import"
    )
    assert sources["roboflow-basketball-players-v17"]["data_license"] == ("CC-BY-4.0")
    assert (
        sources["roboflow-basketball-players-v17"]["redistribution_policy"]
        == "attribution_required_download_requires_user_supplied_api_key"
    )
    assert sources["roboflow-basketball-players-v17"]["adapter_boundary"] == (
        "offline_yolo_detection_training_after_versioned_download_and_sha_seal"
    )
    assert sources["roboflow-basketball-ball-tracking-v2"]["data_license"] == "CC-BY-4.0"
    assert sources["roboflow-basketball-ball-tracking-v2"]["role"] == (
        "basketball_ball_goal_detection_pretraining_candidate"
    )
    assert sources["roboflow-basketball-ball-tracking-v2"]["redistribution_policy"] == (
        "attribution_required_cloudflare_export_blocked_no_auth_bypass"
    )
    assert sources["roboflow-basketball-ball-tracking-v2"]["adapter_boundary"] == (
        "candidate_only_authenticated_export_not_available"
    )
    assert sources["emirsahin-basketball-ball-v1"]["data_license"] == (
        "MIT-card-conflicts-upstream-CC-BY-4.0"
    )
    assert sources["emirsahin-basketball-ball-v1"]["redistribution_policy"] == (
        "reject_missing_labels_rights_conflict_single_shoot_augmented_sample"
    )
    assert sources["emirsahin-basketball-ball-v1"]["adapter_boundary"] == (
        "research_reference_only_no_training_import"
    )
    assert sources["infactory-ball-tracking-v1"]["data_license"] == "CC-BY-NC-4.0"
    assert sources["infactory-ball-tracking-v1"]["role"] == (
        "broadcast_tiny_ball_detector_auxiliary_pretraining"
    )
    assert sources["infactory-ball-tracking-v1"]["redistribution_policy"] == (
        "research_only_noncommercial_subset_hash_sealed_no_runtime_promotion"
    )
    assert sources["infactory-ball-tracking-v1"]["adapter_boundary"] == (
        "offline_yolo_tiny_ball_auxiliary_pretraining_only_no_runtime_promotion"
    )
    assert sources["basketball-coco-20260416"]["data_license"] == "CC-BY-4.0"
    assert sources["basketball-coco-20260416"]["role"] == (
        "basketball_ball_rim_causal_detection_pretraining_candidate"
    )
    assert sources["basketball-coco-20260416"]["redistribution_policy"] == (
        "attribution_required_source_revision_hash_sealed_no_runtime_promotion"
    )
    assert sources["basketball-coco-20260416"]["adapter_boundary"] == (
        "offline_coco_ball_rim_detector_and_frame_evidence_import_only"
    )
    assert sources["basketball-coco-2025-04-02"]["source_revision"] == (
        "d0b72a61490311431c00774db5277cdca737dbe5"
    )
    assert sources["basketball-coco-2025-04-02"]["data_license"] == "CC-BY-4.0"
    assert sources["basketball-coco-2025-04-02"]["redistribution_policy"] == (
        "attribution_required_card_mit_conflicts_readme_cc_by4_offline_only_no_runtime"
    )
    assert sources["basketball-coco-2025-04-02"]["adapter_boundary"] == (
        "offline_deduplicated_coco_ball_rim_train_only_import"
    )
    assert sources["basketball-coco-2025-04-02"]["runtime_consumable"] is False
    assert sources["muvy-2024"]["data_license"] == "CC-BY-4.0"
    assert sources["muvy-2024"]["adapter_boundary"] == ("offline_hash_bound_manual_ball_review_training_only")
    assert sources["muvs-2026"]["data_license"] == ("CC-BY-4.0-as-declared-by-Zenodo-record")
    assert sources["muvs-2026"]["redistribution_policy"] == (
        "attribution_required_record_license_readme_placeholder_verify_before_release"
    )
    assert sources["muvs-2026"]["adapter_boundary"] == (
        "offline_hash_bound_manual_event_state_annotation_training_only"
    )
    assert sources["play-by-play-standardized-space"]["data_license"] == "CC-BY-NC-4.0"
    assert sources["play-by-play-standardized-space"]["redistribution_policy"] == (
        "noncommercial_annotation_only_no_raw_video_no_runtime_promotion"
    )
    assert sources["play-by-play-standardized-space"]["media_included"] is False
    assert sources["play-by-play-standardized-space"]["adapter_boundary"] == (
        "offline_standardized_ball_player_trajectory_auxiliary_calibration_only"
    )
    assert sources["play-by-play-standardized-space"]["audit_artifact"] == (
        "analysis_outputs/public_research/play_by_play_basketball_audit_v1.json"
    )
    assert sources["play-by-play-standardized-space"]["runtime_consumable"] is False
    assert sources["apidis-ball-metadata"]["data_license"] == (
        "non-commercial-research-video-signal-processing-only"
    )
    assert sources["apidis-ball-metadata"]["statistics_included"] is True
    assert sources["apidis-ball-metadata"]["adapter_boundary"] == (
        "offline_hash_bound_ball_video_and_event_import_no_runtime"
    )
    assert sources["vru-basketball"]["data_license"] == "CC-BY-4.0"
    assert sources["vru-basketball"]["source_revision"] == (
        "d256fa0b5fab474663f595abe4f386ac4d5adcc6"
    )
    assert sources["vru-basketball"]["role"] == (
        "continuous_basketball_scene_and_ball_hard_negative_candidate"
    )
    assert sources["vru-basketball"]["adapter_boundary"] == (
        "offline_continuous_scene_probe_and_codex_ball_candidate_review_only"
    )
    assert sources["vru-basketball"]["audit_artifact"] == (
        "analysis_outputs/public_research/vru_basketball_audit_v1.json"
    )
    assert sources["vru-basketball"]["runtime_consumable"] is False
    assert sources["vstat-visual-state-2026"]["media_included"] is True
    assert sources["vstat-visual-state-2026"]["redistribution_policy"] == (
        "reject_general_benchmark_no_basketball_event_or_ball_track_labels"
    )
    assert sources["exact-basketball-skill-feedback"]["source_revision"] == (
        "1bd51bfdbd228f850f69cf81d3b4919c71608c04"
    )
    assert sources["exact-basketball-skill-feedback"]["data_license"] == "Apache-2.0-card"
    assert sources["exact-basketball-skill-feedback"]["media_included"] is True
    assert sources["exact-basketball-skill-feedback"]["redistribution_policy"] == (
        "metadata_only_no_full_game_event_labels_no_runtime_unverified_clip_provenance"
    )
    assert sources["exact-basketball-skill-feedback"]["adapter_boundary"] == (
        "offline_skill_feedback_auxiliary_only_no_shot_outcome_or_player_stats_import"
    )
    assert sources["exact-basketball-skill-feedback"]["audit_artifact"] == (
        "analysis_outputs/public_research/exact_basketball_skill_audit_2026-08-09.json"
    )
    assert sources["exact-basketball-skill-feedback"]["runtime_consumable"] is False
    assert sources["exact-basketball-skill-feedback"]["training_media_eligible"] is False
    assert sources["basketball-events-research-only"]["statistics_included"] is True
    assert sources["basketball-events-research-only"]["data_license"] == "research-only"
    assert sources["basketball-events-research-only"]["source_revision"] == (
        "26d3775286f542b41daf4a94a53190440d426111"
    )
    assert sources["basketball-events-research-only"]["redistribution_policy"] == (
        "retain_bounded_annotation_and_twenty_sample_clips_research_only_no_runtime"
    )
    assert sources["basketball-events-research-only"]["adapter_boundary"] == (
        "offline_hash_bound_event_semantics_audit_no_frame_causal_or_runtime_import"
    )
    assert sources["basketball-events-research-only"]["audit_artifact"] == (
        "dataset/public_sources/basketball_events_v1/manifest.json"
    )
    assert sources["basketball-events-research-only"]["bounded_media_manifest"] == (
        "dataset/public_sources/basketball_events_v1/shot_subset_v1/manifest.json"
    )
    assert sources["basketball-events-research-only"]["latest_offline_screen"] == (
        "analysis_outputs/public_research/basketball_events_shot_vlm_v1/"
        "screen_summary_v1.json"
    )
    assert sources["basketball-events-research-only"]["re_audit_artifact"] == (
        "analysis_outputs/public_research/basketball_events_reaudit_v1.json"
    )
    assert sources["basketball-events-research-only"]["runtime_consumable"] is False
    assert sources["basketball-events-research-only"]["training_media_eligible"] is False
    assert sources["nsva-2022"]["source_revision"] == (
        "97c211f85beec54209b44faea2bff73544a16125"
    )
    assert sources["nsva-2022"]["data_license"] == "fair-noncommercial-research-license"
    assert sources["nsva-2022"]["media_included"] is False
    assert sources["nsva-2022"]["statistics_included"] is True
    assert sources["nsva-2022"]["runtime_consumable"] is False
    assert sources["nsva-2022"]["training_media_eligible"] is False
    assert sources["nsva-2022"]["bounded_annotation_manifest"] == (
        "dataset/public_sources/nsva_subset_v1/source-manifest.json"
    )
    assert sources["nsva-2022"]["source_manifest"] == (
        "dataset/public_sources/nsva_source/source-manifest.json"
    )
    assert sources["nsva-2022"]["audit_artifact"] == (
        "analysis_outputs/public_research/nsva_event_text_audit_v1.json"
    )
    assert sources["gamecommbench-basketball"]["source_revision"] == (
        "728f3a67839c10c54a2aba792d8659f94b8fa6ad"
    )
    assert sources["gamecommbench-basketball"]["data_license"] == "other"
    assert sources["gamecommbench-basketball"]["media_included"] is True
    assert sources["gamecommbench-basketball"]["statistics_included"] is False
    assert sources["gamecommbench-basketball"]["runtime_consumable"] is False
    assert sources["gamecommbench-basketball"]["training_media_eligible"] is False
    assert sources["gamecommbench-basketball"]["source_manifest"] == (
        "dataset/public_sources/gcb_basketball_v1/source-manifest.json"
    )
    assert sources["gamecommbench-basketball"]["audit_artifact"] == (
        "analysis_outputs/public_research/gcb_basketball_event_audit_v1.json"
    )
    assert sources["nba-games-v1"]["source_revision"] == (
        "cf59e3a42413e3ab91f6fc0b1618f280df9a7024"
    )
    assert sources["nba-games-v1"]["media_included"] is False
    assert sources["nba-games-v1"]["statistics_included"] is True
    assert sources["nba-games-v1"]["adapter_boundary"] == (
        "offline_pbp_video_index_audit_no_frame_causal_or_runtime_import"
    )
    assert sources["nba-games-v1"]["audit_artifact"] == (
        "analysis_outputs/public_research/nba_games_fullgame_audit_v1.json"
    )
    assert sources["nba-games-v1"]["runtime_consumable"] is False
    assert sources["nba-games-v1"]["training_media_eligible"] is False
    assert sources["nba-identity"]["data_license"] is None
    assert sources["nba-identity"]["code_license"] is None
    assert sources["nba-identity"]["media_included"] is False
    assert sources["nba-identity"]["statistics_included"] is True
    assert sources["nba-identity"]["redistribution_policy"] == (
        "reject_download_no_declared_license_baidu_feature_only_no_raw_media"
    )
    assert sources["nba-identity"]["adapter_boundary"] == (
        "paper_reference_only_no_annotation_or_feature_import_until_terms_and_files_are_explicit"
    )
    assert sources["nba-rebounds-anticipation"]["data_license"] is None
    assert sources["nba-rebounds-anticipation"]["media_included"] is False
    assert sources["nba-rebounds-anticipation"]["statistics_included"] is True
    assert sources["nba-rebounds-anticipation"]["redistribution_policy"] == (
        "reject_not_public_pending_nba_permission_and_media_rights"
    )
    assert sources["nba-rebounds-anticipation"]["adapter_boundary"] == (
        "paper_reference_only_no_dataset_until_explicit_permission_and_manifest"
    )
    assert sources["basketevent-player-grounded"]["source_url"] == (
        "https://huggingface.co/datasets/zaywas/BasketEvent"
    )
    assert sources["basketevent-player-grounded"]["data_license"] is None
    assert sources["basketevent-player-grounded"]["media_included"] is False
    assert sources["basketevent-player-grounded"]["statistics_included"] is True
    assert sources["basketevent-player-grounded"]["source_revision"] == (
        "85aaa3ce62bc096e3995c39ecfa6773fcc9fe5e1"
    )
    assert sources["basketevent-player-grounded"]["redistribution_policy"] == (
        "retain_bounded_trajectory_json_no_declared_license_no_raw_video_rights"
    )
    assert sources["basketevent-player-grounded"]["adapter_boundary"] == (
        "offline_hash_bound_player_ball_trajectory_event_audit_no_runtime_import"
    )
    assert sources["basketevent-player-grounded"]["runtime_consumable"] is False
    assert sources["basketevent-player-grounded"]["training_media_eligible"] is False
    assert sources["basketevent-player-grounded"]["audit_artifact"] == (
        "dataset/public_sources/basketevent_v1/manifest.json"
    )
    assert sources["multi-event-video-mev"]["source_revision"] == (
        "1e9460d5909116807dd45345b174fb91e9244553"
    )
    assert sources["multi-event-video-mev"]["data_license"] == (
        "mixed-third-party-licenses"
    )
    assert sources["multi-event-video-mev"]["media_included"] is True
    assert sources["multi-event-video-mev"]["runtime_consumable"] is False
    assert sources["multi-event-video-mev"]["training_media_eligible"] is False
    assert sources["multi-event-video-mev"]["adapter_boundary"] == (
        "metadata_only_keyword_audit_no_video_or_causal_training_import"
    )
    assert (
        sources["roboflow-basketball-broadcast-v2"]["redistribution_policy"]
        == "attribution_required_download_requires_user_supplied_api_key"
    )
    assert (
        sources["roboflow-basketball-broadcast-v2"]["adapter_boundary"]
        == "candidate_only_authenticated_export_not_available"
    )
    assert sources["hana-ai-basketball-v1"]["code_license"] is None
    assert sources["hana-ai-basketball-v1"]["redistribution_policy"] == (
        "reject_code_and_google_drive_weight_import_no_declared_license"
    )


def test_nba_plan_seals_two_disjoint_benchmarks_with_full_roster_coverage(
    tmp_path: Path,
) -> None:
    _write_game(tmp_path, "target-a", youtube_id="target-a-video", person_ids=(1, 2))
    _write_game(tmp_path, "enroll-a", youtube_id="enroll-a-video", person_ids=(1, 2))
    _write_game(tmp_path, "target-b", youtube_id="target-b-video", person_ids=(3, 4))
    _write_game(tmp_path, "enroll-b", youtube_id="enroll-b-video", person_ids=(3, 4))

    plan = build_nba_research_plan(
        tmp_path,
        benchmark_enrollment_games={
            "target-a": ["enroll-a"],
            "target-b": ["enroll-b"],
        },
        repository_revision="abc123",
        video_metadata_visibility={
            "target-a-video": True,
            "enroll-a-video": True,
            "target-b-video": True,
            "enroll-b-video": True,
        },
        video_playback_probes={
            video_id: {"status": "available"}
            for video_id in (
                "target-a-video",
                "enroll-a-video",
                "target-b-video",
                "enroll-b-video",
            )
        },
    )
    verified = verify_public_research_plan(json.loads(json.dumps(plan)))

    assert verified["two_game_metadata_gate_ready"] is True
    assert verified["two_game_playback_gate_ready"] is True
    assert verified["two_game_media_gate_ready"] is False
    assert verified["runtime_consumable"] is False
    assert verified["truth_access_policy"] == "post_freeze_evaluation_only"
    assert all(item["active_roster_coverage"] == 1.0 for item in verified["benchmarks"])
    assert all(item["eligible_for_acceptance"] is False for item in verified["benchmarks"])
    assert "download_and_sha_seal_benchmark_and_enrollment_media" in verified["remaining_gates"]
    assert set(REQUIRED_PLAYER_STAT_FIELDS) == {
        "fieldGoalsMade",
        "fieldGoalsAttempted",
        "threePointersMade",
        "threePointersAttempted",
        "reboundsOffensive",
        "reboundsDefensive",
        "assists",
        "steals",
        "blocks",
    }


def test_nba_plan_rejects_media_overlap_incomplete_stats_and_tampering(
    tmp_path: Path,
) -> None:
    _write_game(tmp_path, "target-a", youtube_id="same-video", person_ids=(1, 2))
    _write_game(tmp_path, "enroll-a", youtube_id="same-video", person_ids=(1, 2))
    _write_game(tmp_path, "target-b", youtube_id="target-b", person_ids=(3, 4))
    _write_game(tmp_path, "enroll-b", youtube_id="enroll-b", person_ids=(3, 4))
    with pytest.raises(ValueError, match="overlaps its enrollment media"):
        build_nba_research_plan(
            tmp_path,
            benchmark_enrollment_games={
                "target-a": ["enroll-a"],
                "target-b": ["enroll-b"],
            },
            repository_revision="abc123",
        )

    incomplete_root = tmp_path / "incomplete"
    _write_game(
        incomplete_root,
        "target-a",
        youtube_id="target-a",
        person_ids=(1, 2),
        omit_stat="blocks",
    )
    _write_game(incomplete_root, "enroll-a", youtube_id="enroll-a", person_ids=(1, 2))
    _write_game(incomplete_root, "target-b", youtube_id="target-b", person_ids=(3, 4))
    _write_game(incomplete_root, "enroll-b", youtube_id="enroll-b", person_ids=(3, 4))
    plan = build_nba_research_plan(
        incomplete_root,
        benchmark_enrollment_games={
            "target-a": ["enroll-a"],
            "target-b": ["enroll-b"],
        },
        repository_revision="abc123",
        video_metadata_visibility={
            "target-a": True,
            "enroll-a": True,
            "target-b": True,
            "enroll-b": True,
        },
    )
    assert plan["two_game_metadata_gate_ready"] is False

    playback_plan = build_nba_research_plan(
        incomplete_root,
        benchmark_enrollment_games={
            "target-a": ["enroll-a"],
            "target-b": ["enroll-b"],
        },
        repository_revision="abc123",
        video_metadata_visibility={
            "target-a": True,
            "enroll-a": True,
            "target-b": True,
            "enroll-b": True,
        },
        video_playback_probes={
            "target-a": {"status": "unavailable", "error": "geo blocked"},
            "enroll-a": {"status": "available"},
            "target-b": {"status": "available"},
            "enroll-b": {"status": "available"},
        },
    )
    assert playback_plan["two_game_playback_gate_ready"] is False

    plan["benchmarks"][0]["active_roster_coverage"] = 0.5
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_public_research_plan(plan)


def test_nba_plan_sha_seals_complete_local_media(tmp_path: Path) -> None:
    video_ids = {
        "target-a": "target-a-video",
        "enroll-a": "enroll-a-video",
        "target-b": "target-b-video",
        "enroll-b": "enroll-b-video",
    }
    media_paths: dict[str, Path] = {}
    for slug, video_id in video_ids.items():
        _write_game(tmp_path, slug, youtube_id=video_id, person_ids=(1, 2))
        media_path = tmp_path / f"{video_id}.mp4"
        media_path.write_bytes(f"sealed-{video_id}".encode())
        media_paths[video_id] = media_path

    plan = build_nba_research_plan(
        tmp_path,
        benchmark_enrollment_games={
            "target-a": ["enroll-a"],
            "target-b": ["enroll-b"],
        },
        repository_revision="abc123",
        video_metadata_visibility={video_id: True for video_id in video_ids.values()},
        video_playback_probes={video_id: {"status": "available"} for video_id in video_ids.values()},
        local_media_paths=media_paths,
    )

    assert plan["two_game_media_gate_ready"] is True
    assert all(
        item["local_media"]["sha256"]
        for benchmark in plan["benchmarks"]
        for item in [benchmark["benchmark"], *benchmark["enrollment_games"]]
    )
    assert all(benchmark["eligible_for_acceptance"] is False for benchmark in plan["benchmarks"])

    videos = {
        video_id: {
            "status": "completed",
            "filename": media_paths[video_id].name,
            "size_bytes": media_paths[video_id].stat().st_size,
            "sha256": plan_game["local_media"]["sha256"],
        }
        for benchmark in plan["benchmarks"]
        for plan_game in [benchmark["benchmark"], *benchmark["enrollment_games"]]
        for video_id in [plan_game["youtube_id"]]
    }
    handoff = build_public_game_pair_handoff(
        plan,
        {"plan_sha256": plan["plan_sha256"], "videos": videos},
        benchmark_slug="target-a",
    )
    encoded = json.dumps(handoff)
    assert handoff["answer_assets_included"] is False
    assert handoff["runtime_consumable"] is False
    assert handoff["benchmark_inference_media"]["youtube_id"] == "target-a-video"
    assert handoff["face_enrollment_media"][0]["youtube_id"] == "enroll-a-video"
    assert "truth_artifacts" not in encoded
    assert "active_players" not in encoded


def test_nba_plan_seals_internet_archive_media_override(tmp_path: Path) -> None:
    for slug in ("target-a", "enroll-a", "target-b", "enroll-b"):
        _write_game(tmp_path, slug, youtube_id=slug, person_ids=(1, 2))

    plan = build_nba_research_plan(
        tmp_path,
        benchmark_enrollment_games={
            "target-a": ["enroll-a"],
            "target-b": ["enroll-b"],
        },
        repository_revision="abc123",
        media_overrides={
            "target-a": {
                "provider": "internet_archive",
                "identifier": "archive-item",
                "filename": "complete game.ogv",
                "url": ("https://archive.org/download/archive-item/complete%20game.ogv"),
                "declared_license": None,
                "match_basis": "date_teams_and_game_number",
            }
        },
    )

    game = plan["benchmarks"][0]["benchmark"]
    assert game["media_source"] == {
        "provider": "internet_archive",
        "identifier": "archive-item",
        "filename": "complete game.ogv",
        "url": ("https://archive.org/download/archive-item/complete%20game.ogv"),
        "declared_license": None,
        "match_basis": "date_teams_and_game_number",
        "redistribution_permitted": False,
    }
    assert plan["underlying_media_and_statistics_rights"] == ("external_noncommercial_no_redistribution")


def test_nba_plan_rejects_untrusted_media_override(tmp_path: Path) -> None:
    for slug in ("target-a", "enroll-a", "target-b", "enroll-b"):
        _write_game(tmp_path, slug, youtube_id=slug, person_ids=(1, 2))

    with pytest.raises(ValueError, match="Internet Archive download URL"):
        build_nba_research_plan(
            tmp_path,
            benchmark_enrollment_games={
                "target-a": ["enroll-a"],
                "target-b": ["enroll-b"],
            },
            repository_revision="abc123",
            media_overrides={
                "target-a": {
                    "provider": "internet_archive",
                    "identifier": "archive-item",
                    "filename": "game.mp4",
                    "url": "https://example.com/game.mp4",
                    "declared_license": None,
                    "match_basis": "date_teams_and_game_number",
                }
            },
        )


def test_nba_plan_rejects_archive_overlap_and_unused_override(tmp_path: Path) -> None:
    for slug in ("target-a", "enroll-a", "target-b", "enroll-b"):
        _write_game(tmp_path, slug, youtube_id=slug, person_ids=(1, 2))
    archive_source = {
        "provider": "internet_archive",
        "identifier": "same-archive-item",
        "filename": "same-game.mp4",
        "url": ("https://archive.org/download/same-archive-item/same-game.mp4"),
        "declared_license": None,
        "match_basis": "date_teams_and_game_number",
    }

    with pytest.raises(ValueError, match="overlaps its enrollment media"):
        build_nba_research_plan(
            tmp_path,
            benchmark_enrollment_games={
                "target-a": ["enroll-a"],
                "target-b": ["enroll-b"],
            },
            repository_revision="abc123",
            media_overrides={
                "target-a": archive_source,
                "enroll-a": archive_source,
            },
        )

    with pytest.raises(ValueError, match="unused game slugs"):
        build_nba_research_plan(
            tmp_path,
            benchmark_enrollment_games={
                "target-a": ["enroll-a"],
                "target-b": ["enroll-b"],
            },
            repository_revision="abc123",
            media_overrides={"typo-target": archive_source},
        )


def test_public_pair_handoff_waits_for_complete_enrollment(tmp_path: Path) -> None:
    for slug in ("target-a", "enroll-a", "target-b", "enroll-b"):
        _write_game(tmp_path, slug, youtube_id=slug, person_ids=(1, 2))
    plan = build_nba_research_plan(
        tmp_path,
        benchmark_enrollment_games={
            "target-a": ["enroll-a"],
            "target-b": ["enroll-b"],
        },
        repository_revision="abc123",
    )
    state = {
        "plan_sha256": plan["plan_sha256"],
        "videos": {
            "target-a": {
                "status": "completed",
                "filename": "target-a.mp4",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },
            "enroll-a": {"status": "downloading"},
        },
    }
    with pytest.raises(ValueError, match="not completely sealed"):
        build_public_game_pair_handoff(
            plan,
            state,
            benchmark_slug="target-a",
        )


def test_public_research_catalog_registers_sports_grounding_as_offline_causal_candidate() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}

    assert sources["sports-grounding"]["data_license"] == "CC-BY-NC-4.0"
    assert sources["sports-grounding"]["media_included"] is True
    assert sources["sports-grounding"]["statistics_included"] is False
    assert sources["sports-grounding"]["runtime_consumable"] is False
    assert sources["sports-grounding"]["adapter_boundary"] == (
        "offline_causal_caption_and_player_tube_import_no_ball_truth"
    )


def test_public_research_catalog_registers_sports_action_as_gated_action_tube_candidate() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}

    assert sources["sports-action-multisports"]["source_url"] == (
        "https://huggingface.co/datasets/MCG-NJU/SportsAction"
    )
    assert sources["sports-action-multisports"]["data_license"] == (
        "CC-BY-NC-4.0-gated"
    )
    assert sources["sports-action-multisports"]["media_included"] is True
    assert sources["sports-action-multisports"]["statistics_included"] is False
    assert sources["sports-action-multisports"]["runtime_consumable"] is False
    assert sources["sports-action-multisports"]["training_media_eligible"] is False
    assert sources["sports-action-multisports"]["adapter_boundary"] == (
        "metadata_only_until_gate_and_ball_hand_rim_audit"
    )
    assert sources["sportsshot"]["source_revision"] == (
        "2fca05a9c49366d2c52b4d90e322a7a83e634262"
    )
    assert sources["sportsshot"]["data_license"] == "CC-BY-NC-4.0-gated"
    assert sources["sportsshot"]["media_included"] is True
    assert sources["sportsshot"]["statistics_included"] is False
    assert sources["sportsshot"]["runtime_consumable"] is False
    assert sources["sportsshot"]["training_media_eligible"] is False
    assert sources["sportsshot"]["adapter_boundary"] == (
        "metadata_only_until_access_and_causal_label_audit"
    )


def test_public_research_catalog_registers_qlean_as_gated_continuous_video_only() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}

    assert sources["qlean-video-basketball-match"]["source_url"] == (
        "https://huggingface.co/datasets/qleandataset/video-basketball-match"
    )
    assert sources["qlean-video-basketball-match"]["data_license"] == (
        "qlean-academic-research-license-gated"
    )
    assert sources["qlean-video-basketball-match"]["media_included"] is True
    assert sources["qlean-video-basketball-match"]["statistics_included"] is False
    assert sources["qlean-video-basketball-match"]["runtime_consumable"] is False
    assert sources["qlean-video-basketball-match"]["training_media_eligible"] is False
    assert sources["qlean-video-basketball-match"]["adapter_boundary"] == (
        "metadata_only_until_gate_and_annotation_audit"
    )


def test_public_research_catalog_registers_uvy_as_detector_only_auxiliary() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}
    source = sources["uvy-basketball"]
    assert source["source_revision"] == "10.5281/zenodo.21303900"
    assert source["data_license"] == "CC-BY-4.0"
    assert source["media_included"] is True
    assert source["statistics_included"] is False
    assert source["training_media_eligible"] is True
    assert source["runtime_consumable"] is False
    assert source["adapter_boundary"] == (
        "offline_mot_player_ball_goal_referee_detector_and_hard_negative_import_only"
    )


def test_public_research_catalog_registers_sportvu_as_trajectory_only_auxiliary() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}
    source = sources["nba-sportvu-tracking-2015-16-tiny"]

    assert source["source_revision"] == "50ec5611a9128c996ac19d094145bcc4ffa57f22"
    assert source["data_license"] == "unverified-no-license-declared"
    assert source["media_included"] is False
    assert source["statistics_included"] is True
    assert source["runtime_consumable"] is False
    assert source["training_media_eligible"] is False
    assert source["adapter_boundary"] == (
        "offline_hash_bound_trajectory_clock_event_prior_only_no_video_or_runtime"
    )
    assert source["audit_artifact"] == (
        "analysis_outputs/public_research/sportvu_tracking_tiny_audit_2026-08-09.json"
    )
    assert source["download_manifest"] == (
        "dataset/public_sources/nba_tracking_15_16_tiny_v1/manifest.json"
    )


def test_sportvu_tiny_manifest_and_retained_payloads_are_hash_bound() -> None:
    root = Path("dataset/public_sources/nba_tracking_15_16_tiny_v1")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["dataset_id"] == "nba-sportvu-tracking-2015-16-tiny"
    assert manifest["download_policy"]["runtime_consumable"] is False
    assert manifest["download_policy"]["training_media_eligible"] is False
    assert manifest["download_policy"]["causal_truth_eligible"] is False
    assert manifest["selection"]["filtered_pbp_rows"] == 2208

    for game in manifest["games"]:
        payload = root / game["archive"]
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        assert payload.stat().st_size == game["archive_bytes"]
        assert digest == game["archive_sha256"]

    pbp = root / "pbp_selected_5_games.csv"
    assert pbp.stat().st_size == manifest["selection"]["filtered_pbp_bytes"]
    assert hashlib.sha256(pbp.read_bytes()).hexdigest() == manifest["selection"]["filtered_pbp_sha256"]


def test_public_research_catalog_registers_sportvista_as_gated_rejected_reference() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}
    source = sources["sportvista-gated-research-only"]

    assert source["source_revision"] == "ffb720af2a1e23ff7a8f39379a0f9606cc110e3f"
    assert source["data_license"] == "sportvista-research-only-v1.0-gated"
    assert source["media_included"] is False
    assert source["runtime_consumable"] is False
    assert source["training_media_eligible"] is False
    assert source["adapter_boundary"] == (
        "metadata_only_license_audit_no_agu_base_vlm_training_or_evaluation"
    )


def test_public_research_catalog_registers_wikimedia_hctv_as_manual_fullgame_seed() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}
    source = sources["wikimedia-hctv-full-games"]

    assert source["data_license"] == "CC-BY-4.0-per-file"
    assert source["media_included"] is True
    assert source["statistics_included"] is False
    assert source["runtime_consumable"] is False
    assert source["training_media_eligible"] is False
    assert source["audit_artifact"] == (
        "analysis_outputs/public_research/wikimedia_hctv_fullgame_audit_2026-08-09.json"
    )
    assert source["download_manifest"] == (
        "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json"
    )
    assert source["adapter_boundary"].endswith("no_runtime_until_labels")


def test_current_source_gate_keeps_hctv_seed_below_causal_gate() -> None:
    from scripts.audit_continuous_causal_source_gate import current_candidates

    candidates = {item["dataset_id"]: item for item in current_candidates()}
    source = candidates["wikimedia-hctv-full-games"]

    assert source["rights_cleared"] is True
    assert source["continuous_five_by_five_video"] is True
    assert source["broadcast_diverse"] is False
    assert source["subsecond_ball_hand_rim_outcome_labels"] is False
    assert source["exhaustive_non_shot_hard_negatives"] is False


def test_wikimedia_hctv_manifest_keeps_seed_offline_only() -> None:
    manifest = json.loads(
        Path("dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["dataset_id"] == "wikimedia-hctv-full-games"
    assert manifest["selected_payload"]["bytes"] == 1_416_244_469
    assert len(manifest["selected_payload"]["sha256"]) == 64
    assert manifest["download_policy"]["runtime_consumable"] is False
    assert manifest["download_policy"]["training_media_eligible"] is False
    assert manifest["download_policy"]["causal_truth_eligible"] is False
    assert manifest["continuous_causal_gate"]["continuous_five_by_five_video"] is True
    assert manifest["continuous_causal_gate"]["subsecond_ball_hand_rim_outcome_labels"] is False


def test_public_research_catalog_registers_vtv_and_cross_production_pair() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}

    vtv = sources["wikimedia-vtv-full-game"]
    assert vtv["data_license"] == "public-domain-commons-pd-venezuela-official"
    assert vtv["media_included"] is True
    assert vtv["runtime_consumable"] is False
    assert vtv["training_media_eligible"] is False
    assert vtv["audit_artifact"] == (
        "analysis_outputs/public_research/wikimedia_vtv_fullgame_audit_2026-08-09.json"
    )
    assert vtv["download_manifest"] == (
        "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json"
    )

    pair = sources["wikimedia-hctv-vtv-continuous-seeds"]
    assert pair["data_license"] == "public-domain-vtv-plus-cc-by-4.0-hctv-provisional"
    assert pair["runtime_consumable"] is False
    assert pair["training_media_eligible"] is False
    assert pair["download_manifests"] == [
        "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json",
        "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json",
    ]


def test_public_research_catalog_registers_hctv_randolph_and_three_seed_set() -> None:
    sources = {item["dataset_id"]: item for item in build_public_research_catalog()["sources"]}

    randolph = sources["wikimedia-hctv-randolph-full-game"]
    assert randolph["data_license"] == "CC-BY-4.0-per-file"
    assert randolph["media_included"] is True
    assert randolph["runtime_consumable"] is False
    assert randolph["training_media_eligible"] is False
    assert randolph["download_manifest"] == (
        "dataset/public_sources/wikimedia_hctv_randolph_v1/manifest.json"
    )
    assert randolph["audit_artifact"] == (
        "analysis_outputs/public_research/wikimedia_hctv_randolph_audit_2026-08-09.json"
    )

    three_seed = sources["wikimedia-hctv-hazen-randolph-vtv-continuous-seeds"]
    assert three_seed["runtime_consumable"] is False
    assert three_seed["training_media_eligible"] is False
    assert three_seed["download_manifests"] == [
        "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json",
        "dataset/public_sources/wikimedia_hctv_randolph_v1/manifest.json",
        "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json",
    ]


def test_current_source_gate_keeps_hctv_randolph_and_three_seed_set_below_causal_gate() -> None:
    from scripts.audit_continuous_causal_source_gate import current_candidates

    candidates = {item["dataset_id"]: item for item in current_candidates()}
    randolph = candidates["wikimedia-hctv-randolph-full-game"]
    three_seed = candidates["wikimedia-hctv-hazen-randolph-vtv-continuous-seeds"]

    assert randolph["rights_cleared"] is True
    assert randolph["broadcast_diverse"] is False
    assert randolph["continuous_five_by_five_video"] is True
    assert randolph["subsecond_ball_hand_rim_outcome_labels"] is False
    assert randolph["exhaustive_non_shot_hard_negatives"] is False
    assert three_seed["rights_cleared"] is True
    assert three_seed["broadcast_diverse"] is True
    assert three_seed["continuous_five_by_five_video"] is True
    assert three_seed["subsecond_ball_hand_rim_outcome_labels"] is False
    assert three_seed["exhaustive_non_shot_hard_negatives"] is False


def test_wikimedia_hctv_randolph_manifest_keeps_seed_offline_only() -> None:
    manifest = json.loads(
        Path("dataset/public_sources/wikimedia_hctv_randolph_v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["dataset_id"] == "wikimedia-hctv-randolph-full-game"
    assert manifest["selected_payload"]["bytes"] == 2_152_005_777
    assert len(manifest["selected_payload"]["sha256"]) == 64
    assert manifest["download_policy"]["runtime_consumable"] is False
    assert manifest["download_policy"]["training_media_eligible"] is False
    assert manifest["download_policy"]["causal_truth_eligible"] is False
    assert manifest["continuous_causal_gate"]["continuous_five_by_five_video"] is True
    assert manifest["continuous_causal_gate"]["subsecond_ball_hand_rim_outcome_labels"] is False


def test_current_source_gate_records_vtv_and_two_production_pair_but_denies_labels() -> None:
    from scripts.audit_continuous_causal_source_gate import current_candidates

    candidates = {item["dataset_id"]: item for item in current_candidates()}
    vtv = candidates["wikimedia-vtv-full-game"]
    pair = candidates["wikimedia-hctv-vtv-continuous-seeds"]

    assert vtv["rights_cleared"] is True
    assert vtv["broadcast_diverse"] is False
    assert vtv["continuous_five_by_five_video"] is True
    assert pair["rights_cleared"] is True
    assert pair["broadcast_diverse"] is True
    assert pair["continuous_five_by_five_video"] is True
    assert pair["subsecond_ball_hand_rim_outcome_labels"] is False
    assert pair["exhaustive_non_shot_hard_negatives"] is False


def test_wikimedia_vtv_manifest_keeps_independent_seed_offline_only() -> None:
    manifest = json.loads(
        Path("dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["dataset_id"] == "wikimedia-vtv-full-game"
    assert manifest["source"]["provider"] == "Venezolana de Televisión (VTV)"
    assert manifest["source"]["license"] == "Public domain (Commons declaration)"
    assert manifest["selected_payload"]["bytes"] == 2_228_583_591
    assert len(manifest["selected_payload"]["sha256"]) == 64
    assert manifest["download_policy"]["runtime_consumable"] is False
    assert manifest["download_policy"]["training_media_eligible"] is False
    assert manifest["download_policy"]["causal_truth_eligible"] is False
    assert manifest["continuous_causal_gate"]["continuous_five_by_five_video"] is True
    assert manifest["continuous_causal_gate"]["subsecond_ball_hand_rim_outcome_labels"] is False
    assert manifest["cross_source_context"]["independent_production"] is True
