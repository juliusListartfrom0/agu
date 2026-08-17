"""License-aware planning for public, non-commercial basketball research data.

This module seals dataset provenance and benchmark/enrollment separation.  The
result is deliberately not runtime-consumable: official inference must never
read box scores or play-by-play answers from a research plan.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

PUBLIC_RESEARCH_CATALOG_SCHEMA = "agu.public-research-catalog.v1"
PUBLIC_RESEARCH_PLAN_SCHEMA = "agu.public-research-plan.v1"
PUBLIC_GAME_PAIR_HANDOFF_SCHEMA = "agu.public-game-pair-handoff.v1"

REQUIRED_PLAYER_STAT_FIELDS = (
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "threePointersMade",
    "threePointersAttempted",
    "reboundsOffensive",
    "reboundsDefensive",
    "assists",
    "steals",
    "blocks",
)

# A source is allowed to move from metadata review to payload acquisition only
# when every explicit gate below is true.  Missing values deliberately map to
# False; a truthy string such as ``"true"`` is not accepted as evidence.
CONTINUOUS_CAUSAL_SOURCE_GATE_SCHEMA = "agu.continuous-causal-source-gate.v1"
CONTINUOUS_CAUSAL_SOURCE_GATE_FIELDS = (
    "rights_cleared",
    "broadcast_diverse",
    "continuous_five_by_five_video",
    "subsecond_ball_hand_rim_outcome_labels",
    "exhaustive_non_shot_hard_negatives",
)


def evaluate_continuous_causal_source_gate(source: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the fail-closed metadata gate for a candidate data source.

    This function never performs network I/O or opens a payload.  It is the
    single decision boundary used by metadata-only audits: all five evidence
    flags must be literal ``True`` and the immutable source URL, revision and
    declared size must be present before a download may be attempted.
    """

    dataset_id = str(source.get("dataset_id") or "").strip()
    source_url = str(source.get("source_url") or "").strip()
    source_revision = str(source.get("source_revision") or "").strip()
    declared_size = source.get("declared_size_bytes")
    size_valid = (
        isinstance(declared_size, int)
        and not isinstance(declared_size, bool)
        and declared_size >= 0
    )
    checks = {
        field: source.get(field) is True
        for field in CONTINUOUS_CAUSAL_SOURCE_GATE_FIELDS
    }
    missing_checks = [
        field for field in CONTINUOUS_CAUSAL_SOURCE_GATE_FIELDS if not checks[field]
    ]
    missing_metadata = []
    if not dataset_id:
        missing_metadata.append("dataset_id")
    if not source_url:
        missing_metadata.append("source_url")
    if not source_revision:
        missing_metadata.append("source_revision")
    if not size_valid:
        missing_metadata.append("declared_size_bytes")

    eligible = not missing_checks and not missing_metadata
    return {
        "schema_version": CONTINUOUS_CAUSAL_SOURCE_GATE_SCHEMA,
        "dataset_id": dataset_id,
        "source_url": source_url,
        "source_revision": source_revision,
        "declared_size_bytes": declared_size if size_valid else None,
        "declared_license": source.get("data_license"),
        "checks": checks,
        "missing_checks": missing_checks,
        "missing_metadata": missing_metadata,
        "eligible_for_payload_download": eligible,
        "download_decision": "allow" if eligible else "deny",
        # This audit can never make a source runtime-consumable by itself.
        "runtime_consumable": False,
        "training_media_eligible": False,
        "evidence": [str(item) for item in (source.get("evidence") or [])],
    }


def build_continuous_causal_source_gate_audit(
    sources: Sequence[Mapping[str, Any]],
    *,
    generated_on: str,
) -> dict[str, Any]:
    """Build a deterministic metadata-only audit without downloading data."""

    records = [evaluate_continuous_causal_source_gate(source) for source in sources]
    records.sort(key=lambda item: item["dataset_id"])
    payload: dict[str, Any] = {
        "schema_version": CONTINUOUS_CAUSAL_SOURCE_GATE_SCHEMA,
        "generated_on": str(generated_on),
        "purpose": "metadata_first_gate_for_continuous_basketball_causal_sources",
        "required_checks": list(CONTINUOUS_CAUSAL_SOURCE_GATE_FIELDS),
        "payload_downloads_performed": 0,
        "runtime_consumable": False,
        "training_media_eligible": False,
        "sources": records,
        "eligible_source_count": sum(
            1 for item in records if item["eligible_for_payload_download"]
        ),
        "all_sources_eligible": bool(records)
        and all(item["eligible_for_payload_download"] for item in records),
        "download_policy": (
            "download only records with download_decision=allow;"
            " keep all other sources metadata-only"
        ),
    }
    payload["audit_sha256"] = _json_sha256(payload)
    return payload

PUBLIC_RESEARCH_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "trackid3x3",
        "source_url": "https://github.com/open-starlab/TrackID3x3",
        "role": "identity_tracking_pose_pretraining",
        "data_license": "CC-BY-4.0",
        "code_license": "Apache-2.0-with-third-party-exceptions",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": "attribution_required",
        "adapter_boundary": "offline_bbox_track_and_pose_import",
    },
    {
        "dataset_id": "teamtrack",
        "source_url": "https://atomscott.github.io/TeamTrack/",
        "role": "player_detection_and_tracking_pretraining",
        "data_license": "MIT-as-declared-by-dataset-distribution",
        "code_license": "MIT",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": "verify_distribution_copy_before_release",
        "adapter_boundary": "offline_mot_import",
    },
    {
        "dataset_id": "sportsmot",
        "source_url": "https://github.com/MCG-NJU/SportsMOT",
        "role": "professional_sports_player_mot_pretraining_reference",
        "data_license": "CC-BY-NC-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "noncommercial_research_only_no_redistribution_and_player_only_targets"
        ),
        "adapter_boundary": (
            "offline_player_mot_pretraining_reference_no_ball_or_shot_outcome_labels"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "spacejam-basketball-actions",
        "source_url": "https://github.com/simonefrancia/SpaceJam",
        "role": "isolated_single_player_basketball_action_reference",
        "data_license": "MIT-repository",
        "code_license": "MIT",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "repository_license_does_not_establish_source_video_rights_or_full_game_context"
        ),
        "adapter_boundary": (
            "offline_isolated_action_taxonomy_reference_no_continuous_game_or_outcome_labels"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "bard-2026",
        "source_url": "https://github.com/GabrieleGiudic/BARD",
        "role": "event_semantics_pretraining",
        "data_license": "CC-BY-4.0-annotations",
        "code_license": "CC-BY-4.0",
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": "external_media_rights_not_granted",
        "adapter_boundary": ("offline_embedded_validation_video_and_annotation_import"),
    },
    {
        "dataset_id": "e-bard-detection",
        "source_url": ("https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection"),
        "role": "basketball_ball_hoop_player_detection_pretraining",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": "attribution_required_verify_broadcast_media_before_release",
        "adapter_boundary": "offline_yolo_detection_import",
    },
    {
        "dataset_id": "e-bard-object-classification",
        "source_url": (
            "https://huggingface.co/datasets/GabrieleGiudici/E-BARD-ObjectClassification"
        ),
        "role": "basketball_object_role_vlm_pretraining",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_crop_media_no_full_game_or_runtime_answers"
        ),
        "adapter_boundary": (
            "offline_object_role_crop_classification_vlm_finetune_only"
        ),
    },
    {
        "dataset_id": "e-bard-team-attribution",
        "source_url": (
            "https://huggingface.co/datasets/GabrieleGiudici/E-BARD-TeamAttribution"
        ),
        "role": "basketball_team_color_auxiliary_pretraining",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_crop_media_no_team_identity_or_runtime_answers"
        ),
        "adapter_boundary": "offline_team_color_crop_import_only",
        "audit_artifact": (
            "dataset/public_sources/e_bard_team_attribution_v1/source-manifest.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "open-images-v7-ball-subset",
        "source_url": "https://storage.googleapis.com/openimages/web/download_v7.html",
        "role": "generic_ball_and_basketball_image_box_pretraining",
        "data_license": "CC-BY-4.0-annotations-per-image-image-license",
        "code_license": "Apache-2.0-official-downloader",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_verify_each_image_license_no_runtime_answer_channel"
        ),
        "adapter_boundary": "offline_hash_bound_ball_verifier_training_only",
    },
    {
        "dataset_id": "youtube-boundingboxes",
        "source_url": "https://research.google.com/youtube-bb/",
        "role": "generic_object_detection_reference_no_basketball_ball_class",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": "reject_no_basketball_ball_class_or_event_labels",
        "adapter_boundary": "research_reference_only_no_agu_training_import",
    },
    {
        "dataset_id": "spacejam",
        "source_url": "https://github.com/simonefrancia/SpaceJam",
        "role": "coarse_action_pretraining",
        "data_license": "MIT-as-declared-by-repository",
        "code_license": "MIT",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": "do_not_redistribute_broadcast_clips_without_review",
        "adapter_boundary": "existing_v3_training_layout",
    },
    {
        "dataset_id": "basketball-51",
        "source_url": ("https://www.kaggle.com/datasets/sarbagyashakya/basketball-51-dataset"),
        "role": "shot_type_and_outcome_pretraining",
        "data_license": "Apache-2.0-as-declared-by-Kaggle-uploader",
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": ("do_not_redistribute_underlying_broadcast_clips_without-rights-review"),
        "adapter_boundary": "offline_game_grouped_clip_import",
    },
    {
        "dataset_id": "f-16-nba-shot-test",
        "source_url": "https://huggingface.co/datasets/tsinghua-ee/F-16-NBA",
        "role": "shot_outcome_video_screen",
        "data_license": "Apache-2.0-as-declared-by-Hugging-Face-card",
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": (
            "do_not_redistribute_underlying_nba_video_without_rights_review"
        ),
        "adapter_boundary": (
            "offline_shot_outcome_screen_only_no_ball_hand_rim_timing"
        ),
    },
    {
        "dataset_id": "yerx-bb-free-throw",
        "source_url": "https://huggingface.co/datasets/yerx/bb",
        "role": "free_throw_outcome_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_download_no_dataset_card_license_or_media_provenance"),
        "adapter_boundary": ("research_reference_only_no_media_or_annotation_import"),
    },
    {
        "dataset_id": "basketball-detection-github",
        "source_url": "https://github.com/tranvietcuong03/Basketball_Detection",
        "role": "ball_rim_shoot_made_image_detection_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "reject_download_no_repository_license_or_temporal_ball_hand_rim_labels"
        ),
        "adapter_boundary": "research_reference_only_no_media_or_annotation_import",
        "runtime_consumable": False,
        "training_media_eligible": False,
        "audit_artifact": (
            "analysis_outputs/public_research/basketball_detection_github_audit_v1.json"
        ),
    },
    {
        "dataset_id": "leharris3-basketball-shot-test",
        "source_url": "https://huggingface.co/datasets/leharris3/basketball-shot-test-dataset",
        "role": "shot_outcome_research_reference",
        "data_license": "MIT-tagged-gated",
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": "reject_gated_empty_card_and_unverified_broadcast_provenance",
        "adapter_boundary": "research_reference_only_no_media_or_annotation_import",
    },
    {
        "dataset_id": "nus-basketball-detection-cs5260",
        "source_url": (
            "https://huggingface.co/datasets/linhuaian3/"
            "nus-basketball-detection-cs5260"
        ),
        "source_revision": "3ee1a0decfde199117b3a99d79bcf136c902ebc5",
        "role": "shot_type_outcome_and_background_video_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "reject_missing_license_and_broadcast_provenance_no_continuous_causal_labels"
        ),
        "adapter_boundary": "research_reference_only_no_media_or_annotation_import",
        "audit_artifact": (
            "analysis_outputs/public_research/nus_basketball_detection_audit_v1.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "pl-nba-v1",
        "source_url": "https://github.com/holhouse/PL-NBA-Dataset",
        "source_revision": "0b3d5013b5828004062ed27b10c9eddfca313711",
        "role": "possession_level_temporal_event_taxonomy_research_reference",
        "data_license": "CC-BY-NC-4.0-declared-upstream",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "noncommercial_annotation_only_no_underlying_nba_media_rights_or_runtime"
        ),
        "adapter_boundary": (
            "offline_possession_event_taxonomy_audit_only_no_runtime_or_media_training"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/pl_nba_annotation_audit_v1.json"
        ),
        "bounded_annotation_manifest": "dataset/public_sources/pl_nba_v1/manifest.json",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "tal3x3-v1",
        "source_url": "https://github.com/open-starlab/TAL3x3",
        "source_revision": "f1093d62818c3e8930561fda5939db19c0981d95",
        "role": "3x3_temporal_action_and_shot_outcome_annotation_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "retain_annotation_only_until_dataset_license_and_source_media_rights_verified"
        ),
        "adapter_boundary": (
            "offline_3x3_action_timing_reference_only_no_runtime_or_media_training"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/tal3x3_annotation_audit_v1.json"
        ),
        "bounded_annotation_manifest": "dataset/public_sources/tal3x3_v1/manifest.json",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "muvy-basketball-v1",
        "source_url": "https://zenodo.org/records/13883315",
        "source_revision": "10.5281/zenodo.13883315@2024-10-03",
        "role": "multiview_ball_box_and_synchronization_research_reference",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_bounded_multiview_media_only_no_outcome_labels"
        ),
        "adapter_boundary": (
            "offline_multiview_ball_box_sync_audit_only_no_runtime_or_shot_outcome_training"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/muvy_basketball_audit_v1.json"
        ),
        "bounded_annotation_manifest": (
            "dataset/public_sources/muvy_v1/metadata_manifest.json"
        ),
        "bounded_media_manifest": "dataset/public_sources/muvy_v1/media_manifest.json",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "baskethar",
        "source_url": "https://huggingface.co/datasets/Xian-Gao/BasketHAR",
        "role": "wearable_signal_research_reference",
        "data_license": "Apache-2.0",
        "code_license": "not_separately_declared",
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("visual_pretraining_rejected_video_link_removed_upstream"),
        "adapter_boundary": "research_reference_only_no_visual_import",
    },
    {
        "dataset_id": "basketevent-playnet",
        "source_url": "https://github.com/zhangyu2003/BasketEvent",
        "role": "player_ball_global_temporal_relation_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": ("do_not_download_or_train_until_upstream_adds_compatible_terms_and_media"),
        "adapter_boundary": ("research_evidence_only_no_code_weight_or_annotation_import"),
    },
    {
        "dataset_id": "nsva-2022",
        "source_url": "https://github.com/jackwu502/NSVA",
        "source_revision": "97c211f85beec54209b44faea2bff73544a16125",
        "role": "nba_caption_and_event_taxonomy_research_reference",
        "data_license": "fair-noncommercial-research-license",
        "code_license": "CC-BY-NC-with-MIT-UniVL-exception",
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_for_open_commercial_training_noncommercial_license"),
        "adapter_boundary": "offline_text_event_ontology_audit_only_no_raw_video_or_runtime_import",
        "source_manifest": "dataset/public_sources/nsva_source/source-manifest.json",
        "audit_artifact": "analysis_outputs/public_research/nsva_event_text_audit_v1.json",
        "bounded_annotation_manifest": "dataset/public_sources/nsva_subset_v1/source-manifest.json",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "gamecommbench-basketball",
        "source_url": "https://huggingface.co/datasets/A4Blind/GCB",
        "source_revision": "728f3a67839c10c54a2aba792d8659f94b8fa6ad",
        "role": "clip_commentary_event_vlm_research_reference",
        "data_license": "other",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "reject_video_training_license_other_no_continuous_game_or_provenance_contract"
        ),
        "adapter_boundary": "offline_clip_event_audit_only_no_video_or_runtime_import",
        "source_manifest": "dataset/public_sources/gcb_basketball_v1/source-manifest.json",
        "audit_artifact": "analysis_outputs/public_research/gcb_basketball_event_audit_v1.json",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "vc-nba-2022",
        "source_url": "https://arxiv.org/abs/2401.13888",
        "role": "shot_rebound_entity_caption_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": True,
        "redistribution_policy": (
            "reject_download_paper_only_commercial_source_no_public_data_license"
        ),
        "adapter_boundary": "paper_reference_only_no_video_or_annotation_import",
    },
    {
        "dataset_id": "ncaa-basketball-attention",
        "source_url": "https://basketballattention.appspot.com/",
        "role": "shot_outcome_and_free_throw_event_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_download_dead_official_host_and_no_declared_dataset_license"),
        "adapter_boundary": "paper_reference_only_no_annotation_or_media_import",
    },
    {
        "dataset_id": "poseshot-free-throw",
        "source_url": "https://doi.org/10.1038/s41598-026-41025-0",
        "role": "free_throw_phase_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": "dataset_available_from_authors_on_reasonable_request_only",
        "adapter_boundary": "paper_reference_only_request_required_no_public_dataset",
    },
    {
        "dataset_id": "nba-pbp-video-dataset",
        "source_url": "https://github.com/alijkhalil/nba_pbp_video_dataset",
        "role": "legacy_nba_event_clip_downloader_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_download_no_declared_license_and_multi_terabyte_external_media"),
        "adapter_boundary": "research_reference_only_no_code_or_media_import",
    },
    {
        "dataset_id": "fineaction",
        "source_url": "https://github.com/Richard-61/FineAction",
        "role": "generic_fine_grained_temporal_action_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_until_dataset_terms_are_explicit_and_target_labels_are_relevant"),
        "adapter_boundary": ("reject_for_current_basketball_event_gap_generic_taxonomy"),
    },
    {
        "dataset_id": "unidata-basketball-tracking",
        "source_url": ("https://huggingface.co/datasets/UniqueData/basketball_tracking"),
        "role": "ball_box_tracking_research_reference",
        "data_license": "CC-BY-NC-ND-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_open_training_noncommercial_no_derivatives_sample_release"),
        "adapter_boundary": "reject_noncommercial_no_derivatives_sample_only",
    },
    {
        "dataset_id": "basket-skill-estimation",
        "source_url": "https://huggingface.co/datasets/yulupan/BASKET",
        "role": "long_video_skill_estimation_research_reference",
        "data_license": "Apache-2.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("do_not_download_4477_hours_for_unrelated_skill_level_labels"),
        "adapter_boundary": ("research_reference_only_no_ball_track_or_event_timing_labels"),
    },
    {
        "dataset_id": "svi-bench-2026",
        "source_url": "https://huggingface.co/datasets/MVP-Group/SVI-Bench",
        "role": "strategic_sports_video_benchmark_research_reference",
        "data_license": "CC-BY-NC-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_download_noncommercial_gated_edu_only_no_redistribution"),
        "adapter_boundary": "paper_and_taxonomy_reference_only_no_import",
    },
    {
        "dataset_id": "koppolusameer-court-keypoints-2026",
        "source_url": ("https://huggingface.co/koppolusameer/yolo11n-basketball-court-keypoints"),
        "role": "basketball_court_topology_research_reference",
        "data_license": "CC-BY-4.0-Roboflow-project-HF-card-conflicts-MIT",
        "code_license": "AGPL-3.0-model",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_runtime_unverified_nba_media_rights_missing_keypoint_semantics"),
        "adapter_boundary": ("research_probe_only_weight_deleted_after_cross_game_failure"),
    },
    {
        "dataset_id": "kalicalib-deepsport",
        "source_url": "https://github.com/CEA-LIST/KaliCalib",
        "role": "basketball_camera_calibration_research_reference",
        "data_license": "CC-BY-NC-ND-4.0-upstream-DeepSport",
        "code_license": "CeCILL-2.1",
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_open_commercial_training_noncommercial_upstream_data"),
        "adapter_boundary": ("reject_noncommercial_upstream_data_and_trained_weights"),
    },
    {
        "dataset_id": "multisports",
        "source_url": "https://github.com/MCG-NJU/MultiSports",
        "role": "multi_person_action_localization_research_reference",
        "data_license": "CC-BY-NC-4.0",
        "code_license": "MIT",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_open_commercial_training_noncommercial_dataset"),
        "adapter_boundary": ("paper_and_taxonomy_reference_only_no_media_or_annotation_import"),
    },
    {
        "dataset_id": "finesports",
        "source_url": "https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024",
        "role": "fine_grained_basketball_action_research_reference",
        "data_license": "custom-research-only-release-agreement",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_open_commercial_training_signed_research_only_agreement"),
        "adapter_boundary": ("paper_and_taxonomy_reference_only_no_dataset_or_code_import"),
    },
    {
        "dataset_id": "shot7m2",
        "source_url": "https://huggingface.co/datasets/amathislab/SHOT7M2",
        "role": "synthetic_pose_action_segmentation_research_reference",
        "data_license": "ECL-2.0-as-declared-by-Hub",
        "code_license": None,
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("do_not_download_4_67gb_pose_release_for_broadcast_state_gap"),
        "adapter_boundary": ("research_reference_only_no_import_single_agent_synthetic_pose_mismatch"),
    },
    {
        "dataset_id": "cvballtracking",
        "source_url": "https://github.com/brettfazio/CVBallTracking",
        "role": "legacy_basketball_ball_tracking_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_code_and_checkpoint_import_no_repository_license"),
        "adapter_boundary": ("paper_reference_only_generic_coco_yolov3_not_basketball_weight"),
    },
    {
        "dataset_id": "hardik-ai-basketball",
        "source_url": ("https://github.com/hardik0/AI-basketball-analysis-on-google-colab"),
        "role": "legacy_shot_and_pose_analysis_research_reference",
        "data_license": None,
        "code_license": "OpenPose-noncommercial-research-only",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_open_commercial_training_noncommercial_code_private_data"),
        "adapter_boundary": "reject_code_weight_and_private_training_data_import",
    },
    {
        "dataset_id": "roboflow-basketball-players-v17",
        "source_url": ("https://universe.roboflow.com/workspace-5ujvu/basketball-players-fy4c2-vfsuv/dataset/17"),
        "role": "broadcast_ball_hoop_player_clock_detection_pretraining_candidate",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("attribution_required_download_requires_user_supplied_api_key"),
        "adapter_boundary": ("offline_yolo_detection_training_after_versioned_download_and_sha_seal"),
    },
    {
        "dataset_id": "roboflow-basketball-ball-tracking-v2",
        "source_url": (
            "https://universe.roboflow.com/basketball-lez3k/ball-tracking-udopu/dataset/2"
        ),
        "role": "basketball_ball_goal_detection_pretraining_candidate",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_cloudflare_export_blocked_no_auth_bypass"
        ),
        "adapter_boundary": "candidate_only_authenticated_export_not_available",
    },
    {
        "dataset_id": "emirsahin-basketball-ball-v1",
        "source_url": "https://huggingface.co/datasets/emirsahin/basketball-ball",
        "role": "basketball_ball_detection_research_reference",
        "data_license": "MIT-card-conflicts-upstream-CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "reject_missing_labels_rights_conflict_single_shoot_augmented_sample"
        ),
        "adapter_boundary": "research_reference_only_no_training_import",
    },
    {
        "dataset_id": "infactory-ball-tracking-v1",
        "source_url": "https://huggingface.co/datasets/infactory-ai/ball-tracking",
        "role": "broadcast_tiny_ball_detector_auxiliary_pretraining",
        "data_license": "CC-BY-NC-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "research_only_noncommercial_subset_hash_sealed_no_runtime_promotion"
        ),
        "adapter_boundary": (
            "offline_yolo_tiny_ball_auxiliary_pretraining_only_no_runtime_promotion"
        ),
    },
    {
        "dataset_id": "basketball-coco-2025-04-02",
        "source_url": (
            "https://huggingface.co/datasets/koppolusameer/basketball-coco-2025-04-02"
        ),
        "upstream_source_url": (
            "https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2"
        ),
        "source_revision": "d0b72a61490311431c00774db5277cdca737dbe5",
        "role": "basketball_ball_rim_cross_broadcast_supplement_candidate",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_card_mit_conflicts_readme_cc_by4_offline_only_no_runtime"
        ),
        "adapter_boundary": (
            "offline_deduplicated_coco_ball_rim_train_only_import"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/basketball_coco_2025_audit_v1/audit.json"
        ),
        "runtime_consumable": False,
    },
    {
        "dataset_id": "basketball-coco-20260416",
        "source_url": (
            "https://huggingface.co/datasets/koppolusameer/basketball-coco-20260416"
        ),
        "upstream_source_url": (
            "https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo"
        ),
        "source_revision": "92dea3042e1ec707be5b0d4f7ac474a2b314293",
        "role": "basketball_ball_rim_causal_detection_pretraining_candidate",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_source_revision_hash_sealed_no_runtime_promotion"
        ),
        "adapter_boundary": (
            "offline_coco_ball_rim_detector_and_frame_evidence_import_only"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/basketball_coco_2026_audit_v1.json"
        ),
        "runtime_consumable": False,
    },
    {
        "dataset_id": "sports-grounding",
        "source_url": "https://huggingface.co/datasets/MCG-NJU/SportsGrounding",
        "source_revision": "1fb03e7786228ee6d1d7909eb7011e7e3fb7eb2c",
        "role": "basketball_causal_caption_player_tube_vlm_pretraining_candidate",
        "data_license": "CC-BY-NC-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "noncommercial_metadata_only_no_video_runtime_or_external_release"
        ),
        "adapter_boundary": (
            "offline_causal_caption_and_player_tube_import_no_ball_truth"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/sports_grounding_metadata_audit_v1.json"
        ),
        "runtime_consumable": False,
    },
    {
        "dataset_id": "sports-action-multisports",
        "source_url": "https://huggingface.co/datasets/MCG-NJU/SportsAction",
        "role": "gated_multisport_basketball_action_rebound_tube_candidate",
        "data_license": "CC-BY-NC-4.0-gated",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "gated_noncommercial_only_no_payload_until_access_and_source_media_audit"
        ),
        "adapter_boundary": "metadata_only_until_gate_and_ball_hand_rim_audit",
        "audit_artifact": (
            "analysis_outputs/public_research/sports_action_multisports_audit_v1.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "sportsshot",
        "source_url": "https://huggingface.co/datasets/MCG-NJU/SportsShot",
        "source_revision": "2fca05a9c49366d2c52b4d90e322a7a83e634262",
        "role": "sports_shot_segmentation_research_reference",
        "data_license": "CC-BY-NC-4.0-gated",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "gated_noncommercial_183gb_shot_segmentation_no_ball_hand_rim_outcomes"
        ),
        "adapter_boundary": "metadata_only_until_access_and_causal_label_audit",
        "audit_artifact": (
            "analysis_outputs/public_research/sportsshot_online_gate_v1.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "muvy-2024",
        "source_url": "https://zenodo.org/records/13883315",
        "role": "user_generated_multiview_small_ball_detection_pretraining",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("attribution_required_verify_source_video_terms_before_release"),
        "adapter_boundary": ("offline_hash_bound_manual_ball_review_training_only"),
    },
    {
        "dataset_id": "muvs-2026",
        "source_url": "https://zenodo.org/records/20708683",
        "role": "multiview_source_event_state_pretraining",
        "data_license": "CC-BY-4.0-as-declared-by-Zenodo-record",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("attribution_required_record_license_readme_placeholder_verify_before_release"),
        "adapter_boundary": ("offline_hash_bound_manual_event_state_annotation_training_only"),
    },
    {
        "dataset_id": "roboflow-basketball-broadcast-v2",
        "source_url": ("https://universe.roboflow.com/peppes-project/basketball-broadcast-fwdpq/dataset/2"),
        "role": "broadcast_ball_hoop_made_basket_detection_candidate",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("attribution_required_download_requires_user_supplied_api_key"),
        "adapter_boundary": ("candidate_only_authenticated_export_not_available"),
    },
    {
        "dataset_id": "hana-ai-basketball-v1",
        "source_url": "https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1",
        "role": "basketball_detection_and_tracking_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": ("reject_code_and_google_drive_weight_import_no_declared_license"),
        "adapter_boundary": ("dataset_pointer_only_use_separately_licensed_roboflow_source"),
    },
    {
        "dataset_id": "play-by-play-standardized-space",
        "source_url": "https://zenodo.org/records/12698090",
        "role": "standardized_ball_player_trajectory_auxiliary_calibration",
        "data_license": "CC-BY-NC-4.0",
        "code_license": None,
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "noncommercial_annotation_only_no_raw_video_no_runtime_promotion"
        ),
        "adapter_boundary": (
            "offline_standardized_ball_player_trajectory_auxiliary_calibration_only"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/play_by_play_basketball_audit_v1.json"
        ),
        "runtime_consumable": False,
    },
    {
        "dataset_id": "apidis-ball-metadata",
        "source_url": "https://ispgroup.gitlab.io/code/apidis/",
        "role": "multiview_ball_event_pretraining_research_reference",
        "data_license": "non-commercial-research-video-signal-processing-only",
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": "noncommercial_only_mention_apidis_range_extracted_subset",
        "adapter_boundary": "offline_hash_bound_ball_video_and_event_import_no_runtime",
    },
    {
        "dataset_id": "vru-basketball",
        "source_url": "https://huggingface.co/datasets/BestWJH/VRU_Basketball",
        "source_revision": "d256fa0b5fab474663f595abe4f386ac4d5adcc6",
        "role": "continuous_basketball_scene_and_ball_hard_negative_candidate",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_subset_only_no_ball_hand_rim_ground_truth_or_runtime"
        ),
        "adapter_boundary": (
            "offline_continuous_scene_probe_and_codex_ball_candidate_review_only"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/vru_basketball_audit_v1.json"
        ),
        "runtime_consumable": False,
    },
    {
        "dataset_id": "uvy-basketball",
        "source_url": "https://zenodo.org/records/21303900",
        "source_revision": "10.5281/zenodo.21303900",
        "role": "auxiliary_player_ball_goal_referee_detection_and_tracking",
        "data_license": "CC-BY-4.0",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "attribution_required_bounded_basketball_frames_only_no_runtime_or_causal_truth"
        ),
        "adapter_boundary": (
            "offline_mot_player_ball_goal_referee_detector_and_hard_negative_import_only"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/uvy_basketball_audit_2026-08-08.json"
        ),
        "download_manifest": "dataset/public_sources/uvy_v1/manifest.json",
        "runtime_consumable": False,
        "training_media_eligible": True,
    },
    {
        "dataset_id": "qlean-video-basketball-match",
        "source_url": "https://huggingface.co/datasets/qleandataset/video-basketball-match",
        "provider_url": "https://qleandataset.visual-bank.co.jp/en/datasets/video",
        "role": "rights_cleared_continuous_basketball_video_research_candidate",
        "data_license": "qlean-academic-research-license-gated",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "gated_academic_only_manual_access_review_no_download_until_access_granted"
        ),
        "adapter_boundary": "metadata_only_until_gate_and_annotation_audit",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "vstat-visual-state-2026",
        "source_url": "https://huggingface.co/datasets/VSTAT-NeurIPS2026/VSTAT",
        "role": "general_visual_state_tracking_research_reference",
        "data_license": "CC-BY-4.0-annotations-self-recorded-synthetic",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "reject_general_benchmark_no_basketball_event_or_ball_track_labels"
        ),
        "adapter_boundary": (
            "paper_reference_only_no_basketball_event_supervision_import"
        ),
    },
    {
        "dataset_id": "multi-event-video-mev",
        "source_url": "https://huggingface.co/datasets/MultiEventVideo/MEV",
        "source_revision": "1e9460d5909116807dd45345b174fb91e9244553",
        "role": "multi_event_video_basketball_keyword_research_reference",
        "data_license": "mixed-third-party-licenses",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "reject_no_public_per_video_rights_manifest_short_clips_no_continuous_game"
        ),
        "adapter_boundary": (
            "metadata_only_keyword_audit_no_video_or_causal_training_import"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/online_candidate_supplement_2026-08-09.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "exact-basketball-skill-feedback",
        "source_url": "https://huggingface.co/datasets/Alexhimself/ExAct",
        "source_revision": "1bd51bfdbd228f850f69cf81d3b4919c71608c04",
        "role": "basketball_skill_feedback_auxiliary_vlm_research_reference",
        "data_license": "Apache-2.0-card",
        "code_license": "Apache-2.0-card",
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "metadata_only_no_full_game_event_labels_no_runtime_unverified_clip_provenance"
        ),
        "adapter_boundary": (
            "offline_skill_feedback_auxiliary_only_no_shot_outcome_or_player_stats_import"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/exact_basketball_skill_audit_2026-08-09.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "basketball-events-research-only",
        "source_url": "https://huggingface.co/datasets/saveerjain/basketball-events",
        "source_revision": "26d3775286f542b41daf4a94a53190440d426111",
        "role": "basketball_event_semantics_research_reference",
        "data_license": "research-only",
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": (
            "retain_bounded_annotation_and_twenty_sample_clips_research_only_no_runtime"
        ),
        "adapter_boundary": (
            "offline_hash_bound_event_semantics_audit_no_frame_causal_or_runtime_import"
        ),
        "audit_artifact": "dataset/public_sources/basketball_events_v1/manifest.json",
        "bounded_media_manifest": (
            "dataset/public_sources/basketball_events_v1/shot_subset_v1/manifest.json"
        ),
        "latest_offline_screen": (
            "analysis_outputs/public_research/basketball_events_shot_vlm_v1/"
            "screen_summary_v1.json"
        ),
        "re_audit_artifact": (
            "analysis_outputs/public_research/basketball_events_reaudit_v1.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "nba-streaming-2026",
        "source_url": "https://arxiv.org/abs/2608.09200",
        "source_revision": "arxiv-v1-2026-08-10",
        "role": "continuous_full_game_streaming_event_annotation_reference",
        "data_license": "CC-BY-NC-4.0-declared-no_payload_link",
        "code_license": None,
        "media_included": True,
        "statistics_included": True,
        "named_portraits_included": True,
        "redistribution_policy": (
            "paper_only_no_public_payload_or_hash_manifest_do_not_download_until_release_is_verifiable"
        ),
        "adapter_boundary": (
            "offline_taxonomy_and_causal_window_reference_no_runtime_or_training_import"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/nba_streaming_release_audit_2026-08-12.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "nba-games-v1",
        "source_url": "https://huggingface.co/datasets/choucsan/NBA_Games",
        "source_revision": "cf59e3a42413e3ab91f6fc0b1618f280df9a7024",
        "role": "full_game_video_index_and_official_pbp_research_reference",
        "data_license": "MIT-metadata-only-underlying-video-unverified",
        "code_license": None,
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": (
            "retain_metadata_only_no_linked_video_download_until_rights_review"
        ),
        "adapter_boundary": (
            "offline_pbp_video_index_audit_no_frame_causal_or_runtime_import"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/nba_games_fullgame_audit_v1.json"
        ),
        "download_manifest": "dataset/public_sources/nba_games_v1/manifest.json",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "nba-identity",
        "source_url": "https://github.com/Zeyu1226-mt/LLM-IAVC",
        "role": "identity_aware_basketball_event_caption_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": (
            "reject_download_no_declared_license_baidu_feature_only_no_raw_media"
        ),
        "adapter_boundary": (
            "paper_reference_only_no_annotation_or_feature_import_until_terms_and_files_are_explicit"
        ),
    },
    {
        "dataset_id": "nba-rebounds-anticipation",
        "source_url": "https://arxiv.org/abs/2512.15386",
        "role": "basketball_rebound_event_anticipation_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": (
            "reject_not_public_pending_nba_permission_and_media_rights"
        ),
        "adapter_boundary": (
            "paper_reference_only_no_dataset_until_explicit_permission_and_manifest"
        ),
    },
    {
        "dataset_id": "basketevent-player-grounded",
        "source_url": "https://huggingface.co/datasets/zaywas/BasketEvent",
        "role": "player_grounded_basketball_event_trajectory_research_reference",
        "data_license": None,
        "code_license": None,
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "source_revision": "85aaa3ce62bc096e3995c39ecfa6773fcc9fe5e1",
        "redistribution_policy": (
            "retain_bounded_trajectory_json_no_declared_license_no_raw_video_rights"
        ),
        "adapter_boundary": (
            "offline_hash_bound_player_ball_trajectory_event_audit_no_runtime_import"
        ),
        "audit_artifact": "dataset/public_sources/basketevent_v1/manifest.json",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "nba-games-metadata",
        "source_url": "https://huggingface.co/datasets/choucsan/NBA_Games",
        "role": "post_freeze_full_game_acceptance_candidate",
        "data_license": "MIT-metadata",
        "code_license": "MIT",
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": "private_noncommercial_research_only_no_redistribution",
        "adapter_boundary": "offline_truth_sealing_and_post_freeze_evaluation_only",
    },
    {
        "dataset_id": "nba-sportvu-tracking-2015-16-tiny",
        "source_url": "https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16",
        "source_revision": "50ec5611a9128c996ac19d094145bcc4ffa57f22",
        "role": "offline_player_ball_trajectory_and_pbp_prior_research_reference",
        "data_license": "unverified-no-license-declared",
        "code_license": "Apache-2.0-script-header-only",
        "media_included": False,
        "statistics_included": True,
        "named_portraits_included": False,
        "redistribution_policy": (
            "internal_bounded_research_only_no_redistribution_no_video_rights"
        ),
        "adapter_boundary": (
            "offline_hash_bound_trajectory_clock_event_prior_only_no_video_or_runtime"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/sportvu_tracking_tiny_audit_2026-08-09.json"
        ),
        "download_manifest": "dataset/public_sources/nba_tracking_15_16_tiny_v1/manifest.json",
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "sportvista-gated-research-only",
        "source_url": "https://huggingface.co/datasets/bouachalazhar/sportvista",
        "source_revision": "ffb720af2a1e23ff7a8f39379a0f9606cc110e3f",
        "role": "gated_multisport_audio_visual_research_reference",
        "data_license": "sportvista-research-only-v1.0-gated",
        "code_license": None,
        "media_included": False,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "manual_individual_access_no_redistribution_no_general_purpose_model_use"
        ),
        "adapter_boundary": (
            "metadata_only_license_audit_no_agu_base_vlm_training_or_evaluation"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/sportvista_online_audit_2026-08-09.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "wikimedia-hctv-full-games",
        "source_url": (
            "https://commons.wikimedia.org/wiki/"
            "Category:Videos_of_basketball_in_the_United_States"
        ),
        "source_revision": (
            "commons-api-2026-08-09@"
            "a2037464047c5a6598593bd7440be71a77f429a210149939a8a6ba1c79507841"
        ),
        "role": (
            "continuous_broadcast_5x5_hard_negative_and_manual_causal_annotation_candidate"
        ),
        "data_license": "CC-BY-4.0-per-file",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "per_file_cc_by_attribution_hctv_provisional_license_review_warning"
        ),
        "adapter_boundary": (
            "offline_hash_bound_full_game_media_and_manual_causal_annotation_only_no_runtime_until_labels"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/"
            "wikimedia_hctv_fullgame_audit_2026-08-09.json"
        ),
        "download_manifest": (
            "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "wikimedia-vtv-full-game",
        "source_url": (
            "https://commons.wikimedia.org/wiki/File:"
            "Superliga_Profesional_de_Baloncesto_-_Spartans_de_Distrito_Capital_"
            "Vs._Trotamundos_de_Carabobo.webm"
        ),
        "source_revision": (
            "commons-api-2026-08-09@"
            "985c1b2d82fd073065708f5a547dc5b765f548e4f03f33f0ca15a1744b78bae9"
        ),
        "role": (
            "independent_continuous_broadcast_5x5_hard_negative_and_manual_"
            "causal_annotation_candidate"
        ),
        "data_license": "public-domain-commons-pd-venezuela-official",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "commons_public_domain_claim_preserve_vtv_provenance_and_do_not_"
            "infer_rights_from_unlicensed_youtube_record"
        ),
        "adapter_boundary": (
            "offline_hash_bound_full_game_media_and_manual_causal_annotation_"
            "only_no_runtime_until_labels"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/"
            "wikimedia_vtv_fullgame_audit_2026-08-09.json"
        ),
        "download_manifest": (
            "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "wikimedia-hctv-randolph-full-game",
        "source_url": (
            "https://commons.wikimedia.org/wiki/File:"
            "Boys_Varsity_Basketball_v._Randolph_-_February_17,_2026.webm"
        ),
        "source_revision": (
            "commons-api-2026-08-09@"
            "0d3f90add863fe1f8e0b7c2c2ed6af636fe3d6cca25cc07b05f828002c2fa010"
        ),
        "role": (
            "independent_continuous_broadcast_5x5_hard_negative_and_manual_"
            "causal_annotation_candidate"
        ),
        "data_license": "CC-BY-4.0-per-file",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "per_file_cc_by_attribution_hctv_provisional_license_review_warning"
        ),
        "adapter_boundary": (
            "offline_hash_bound_full_game_media_and_manual_causal_annotation_"
            "only_no_runtime_until_labels"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/"
            "wikimedia_hctv_randolph_audit_2026-08-09.json"
        ),
        "download_manifest": (
            "dataset/public_sources/wikimedia_hctv_randolph_v1/manifest.json"
        ),
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "wikimedia-hctv-vtv-continuous-seeds",
        "source_url": "https://commons.wikimedia.org/wiki/Category:Videos_of_basketball",
        "source_revision": (
            "hctv-manifest-33dd43e25764b0cc66d0a56a7ee887e65ff62256c4fb266035b86995a03af71c"
            "+vtv-manifest-df2fed8a6f6810de4e3848d0dab2d2e6cc6c88b595a87b44ad758d5c99ff7c70"
        ),
        "role": "cross_production_continuous_5x5_causal_annotation_candidate",
        "data_license": "public-domain-vtv-plus-cc-by-4.0-hctv-provisional",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "per_file_rights_and_attribution_audit_required_for_cross_production_pair"
        ),
        "adapter_boundary": (
            "offline_cross_production_annotation_pair_only_no_runtime_or_training_"
            "truth_until_causal_labels"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/"
            "wikimedia_hctv_vtv_cross_source_audit_2026-08-09.json"
        ),
        "download_manifests": [
            "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json",
            "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json",
        ],
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "wikimedia-hctv-hazen-randolph-vtv-continuous-seeds",
        "source_url": "https://commons.wikimedia.org/wiki/Category:Videos_of_basketball",
        "source_revision": (
            "hctv-manifest-33dd43e25764b0cc66d0a56a7ee887e65ff62256c4fb266035b86995a03af71c"
            "+hctv-randolph-manifest-4c9d5f1e8a6040faea2e6f686f412cc709a63678ce6eca47a5bb36d10e304c6e"
            "+vtv-manifest-df2fed8a6f6810de4e3848d0dab2d2e6cc6c88b595a87b44ad758d5c99ff7c70"
        ),
        "role": "three_seed_cross_production_continuous_5x5_causal_annotation_candidate",
        "data_license": "public-domain-vtv-plus-cc-by-4.0-hctv-provisional",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": False,
        "redistribution_policy": (
            "per_file_rights_and_attribution_audit_required_for_three_seed_"
            "cross_production_annotation_set"
        ),
        "adapter_boundary": (
            "offline_three_source_annotation_only_no_runtime_or_training_truth_"
            "until_causal_labels"
        ),
        "audit_artifact": (
            "analysis_outputs/public_research/"
            "wikimedia_hctv_hazen_randolph_vtv_cross_source_audit_2026-08-09.json"
        ),
        "download_manifests": [
            "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json",
            "dataset/public_sources/wikimedia_hctv_randolph_v1/manifest.json",
            "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json",
        ],
        "runtime_consumable": False,
        "training_media_eligible": False,
    },
    {
        "dataset_id": "wikimedia-commons-portraits",
        "source_url": "https://commons.wikimedia.org/wiki/Category:Portrait_photographs_of_basketball_players",
        "role": "optional_face_enrollment_supplement",
        "data_license": "per-file-license",
        "code_license": None,
        "media_included": True,
        "statistics_included": False,
        "named_portraits_included": True,
        "redistribution_policy": "store_and_honor_each_file_license_and_attribution",
        "adapter_boundary": "offline_reviewed_face_enrollment_only",
    },
)


def build_public_research_catalog() -> dict[str, Any]:
    """Return the reusable source registry with an integrity hash."""

    payload: dict[str, Any] = {
        "schema_version": PUBLIC_RESEARCH_CATALOG_SCHEMA,
        "purpose": "noncommercial_research_source_governance",
        "runtime_consumable": False,
        "sources": [dict(item) for item in PUBLIC_RESEARCH_SOURCES],
    }
    payload["catalog_sha256"] = _json_sha256(payload)
    return payload


def build_nba_research_plan(
    dataset_root: str | Path,
    *,
    benchmark_enrollment_games: Mapping[str, Sequence[str]],
    repository_revision: str,
    video_metadata_visibility: Mapping[str, bool] | None = None,
    video_playback_probes: Mapping[str, Mapping[str, Any]] | None = None,
    local_media_paths: Mapping[str, str | Path] | None = None,
    media_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Seal two or more benchmark games and disjoint enrollment games.

    The returned plan contains rosters and truth-file hashes, but not player
    statistics values.  It remains ineligible for acceptance until downloaded
    media are SHA-sealed, face galleries are built, and models are frozen.
    """

    root = Path(dataset_root)
    if not repository_revision.strip():
        raise ValueError("NBA_Games import requires an immutable repository revision")
    if len(benchmark_enrollment_games) < 2:
        raise ValueError("the independent acceptance plan requires at least two benchmark games")

    metadata_visibility = dict(video_metadata_visibility or {})
    playback_probes = dict(video_playback_probes or {})
    media_paths = {video_id: Path(path) for video_id, path in (local_media_paths or {}).items()}
    overrides = dict(media_overrides or {})
    selected_slugs = set(benchmark_enrollment_games)
    selected_slugs.update(slug for enrollment_slugs in benchmark_enrollment_games.values() for slug in enrollment_slugs)
    unused_override_slugs = set(overrides) - selected_slugs
    if unused_override_slugs:
        raise ValueError(f"media overrides contain unused game slugs: {sorted(unused_override_slugs)}")
    all_benchmark_ids: set[str] = set()
    all_benchmark_media: set[tuple[str, ...]] = set()
    benchmarks: list[dict[str, Any]] = []
    for benchmark_slug, enrollment_slugs in sorted(benchmark_enrollment_games.items()):
        if not enrollment_slugs:
            raise ValueError(f"benchmark {benchmark_slug} has no enrollment game")
        benchmark = _load_nba_game(
            root,
            benchmark_slug,
            metadata_visibility=metadata_visibility,
            playback_probes=playback_probes,
            media_paths=media_paths,
            media_override=overrides.get(benchmark_slug),
        )
        if benchmark["youtube_id"] in all_benchmark_ids:
            raise ValueError("benchmark YouTube IDs must be unique")
        all_benchmark_ids.add(benchmark["youtube_id"])
        benchmark_media_identity = _media_identity(benchmark)
        if benchmark_media_identity in all_benchmark_media:
            raise ValueError("benchmark media identities must be unique")
        all_benchmark_media.add(benchmark_media_identity)

        enrollment = [
            _load_nba_game(
                root,
                slug,
                metadata_visibility=metadata_visibility,
                playback_probes=playback_probes,
                media_paths=media_paths,
                media_override=overrides.get(slug),
            )
            for slug in enrollment_slugs
        ]
        enrollment_video_ids = {item["youtube_id"] for item in enrollment}
        if benchmark["youtube_id"] in enrollment_video_ids:
            raise ValueError(f"benchmark {benchmark_slug} overlaps its enrollment media")
        if len(enrollment_video_ids) != len(enrollment):
            raise ValueError(f"benchmark {benchmark_slug} repeats enrollment media")
        enrollment_media_identities = {_media_identity(item) for item in enrollment}
        if benchmark_media_identity in enrollment_media_identities:
            raise ValueError(f"benchmark {benchmark_slug} overlaps its enrollment media")
        if len(enrollment_media_identities) != len(enrollment):
            raise ValueError(f"benchmark {benchmark_slug} repeats enrollment media")

        enrolled_people = {player["person_id"] for item in enrollment for player in item["active_players"]}
        target_people = {player["person_id"] for player in benchmark["active_players"]}
        covered_people = target_people & enrolled_people
        missing_people = target_people - enrolled_people
        roster_coverage = len(covered_people) / len(target_people) if target_people else 0.0
        structured_truth_ready = bool(
            benchmark["box_score_complete"] and benchmark["play_by_play_complete"] and roster_coverage == 1.0
        )
        all_games = [benchmark, *enrollment]
        metadata_visible = all(item["youtube_metadata_visible"] is True for item in all_games)
        playback_ready = all(item["youtube_playback_probe"]["status"] == "available" for item in all_games)
        media_sha256_sealed = all(item["local_media"] is not None for item in all_games)
        benchmarks.append(
            {
                "benchmark": benchmark,
                "enrollment_games": enrollment,
                "benchmark_disjoint_by_youtube_id": True,
                "benchmark_disjoint_by_media_identity": True,
                "active_player_count": len(target_people),
                "enrolled_active_player_count": len(covered_people),
                "active_roster_coverage": roster_coverage,
                "missing_enrollment_person_ids": sorted(missing_people),
                "structured_truth_ready": structured_truth_ready,
                "youtube_metadata_visible": metadata_visible,
                "playback_ready": playback_ready,
                "media_sha256_sealed": media_sha256_sealed,
                "face_gallery_sealed": False,
                "model_frozen": False,
                "eligible_for_acceptance": False,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": PUBLIC_RESEARCH_PLAN_SCHEMA,
        "purpose": "post_freeze_independent_acceptance_candidate",
        "noncommercial_research_only": True,
        "runtime_consumable": False,
        "answer_assets_runtime_forbidden": True,
        "truth_access_policy": "post_freeze_evaluation_only",
        "dataset_id": "nba-games-metadata",
        "source_repository": "https://huggingface.co/datasets/choucsan/NBA_Games",
        "repository_revision": repository_revision,
        "metadata_license": "MIT",
        "underlying_media_and_statistics_rights": "external_noncommercial_no_redistribution",
        "benchmark_count": len(benchmarks),
        "two_game_metadata_gate_ready": all(
            item["structured_truth_ready"] and item["youtube_metadata_visible"] for item in benchmarks
        ),
        "two_game_playback_gate_ready": all(item["playback_ready"] for item in benchmarks),
        "two_game_media_gate_ready": all(item["media_sha256_sealed"] for item in benchmarks),
        "benchmarks": benchmarks,
        "remaining_gates": [
            "download_and_sha_seal_benchmark_and_enrollment_media",
            "build_benchmark_disjoint_face_galleries",
            "freeze_models_thresholds_rosters_and_configuration",
            "run_agu_autonomous_inference_without_truth_access",
            "evaluate_each_game_after_freeze_and_reach_95_percent",
        ],
    }
    payload["plan_sha256"] = _json_sha256(payload)
    return payload


def verify_public_research_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify integrity and the non-runtime/non-redistribution boundary."""

    plan = dict(payload)
    claimed_hash = str(plan.pop("plan_sha256", ""))
    if plan.get("schema_version") != PUBLIC_RESEARCH_PLAN_SCHEMA:
        raise ValueError("unsupported public research plan schema")
    if plan.get("runtime_consumable") is not False:
        raise ValueError("public research truth must not be runtime-consumable")
    if plan.get("answer_assets_runtime_forbidden") is not True:
        raise ValueError("public research plan must forbid runtime answer assets")
    if plan.get("noncommercial_research_only") is not True:
        raise ValueError("NBA research plan must retain its non-commercial boundary")
    if _json_sha256(plan) != claimed_hash:
        raise ValueError("public research plan hash mismatch")
    plan["plan_sha256"] = claimed_hash
    return plan


def build_public_game_pair_handoff(
    plan_payload: Mapping[str, Any],
    download_state: Mapping[str, Any],
    *,
    benchmark_slug: str,
) -> dict[str, Any]:
    """Build a truth-free handoff after one benchmark/enrollment pair is sealed."""

    plan = verify_public_research_plan(plan_payload)
    if download_state.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("download state does not match the research plan")
    matches = [item for item in plan.get("benchmarks") or [] if item.get("benchmark", {}).get("slug") == benchmark_slug]
    if len(matches) != 1:
        raise ValueError(f"unknown benchmark slug: {benchmark_slug}")
    benchmark = matches[0]
    videos = download_state.get("videos") or {}

    def sealed_media(game: Mapping[str, Any]) -> dict[str, Any]:
        video_id = str(game.get("youtube_id") or "")
        state = videos.get(video_id) or {}
        if state.get("status") != "completed":
            raise ValueError(f"media is not completely sealed: {video_id}")
        sha256 = str(state.get("sha256") or "")
        filename = str(state.get("filename") or "")
        size_bytes = int(state.get("size_bytes") or 0)
        if len(sha256) != 64 or not filename or size_bytes <= 0:
            raise ValueError(f"sealed media metadata is incomplete: {video_id}")
        return {
            "slug": str(game["slug"]),
            "youtube_id": video_id,
            "filename": filename,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }

    payload: dict[str, Any] = {
        "schema_version": PUBLIC_GAME_PAIR_HANDOFF_SCHEMA,
        "source_plan_sha256": plan["plan_sha256"],
        "runtime_consumable": False,
        "answer_assets_included": False,
        "benchmark_disjoint": True,
        "benchmark_inference_media": sealed_media(benchmark["benchmark"]),
        "face_enrollment_media": [sealed_media(game) for game in benchmark["enrollment_games"]],
        "next_steps": [
            "extract_and_review_face_enrollment_candidates",
            "freeze_face_gallery_and_models",
            "run_benchmark_inference_without_plan_access",
            "evaluate_frozen_prediction_in_separate_process",
        ],
    }
    payload["handoff_sha256"] = _json_sha256(payload)
    return payload


def check_youtube_oembed(video_id: str, *, timeout_seconds: float = 10.0) -> bool:
    """Check current public metadata availability without downloading video."""

    if not video_id.strip():
        return False
    query = urlencode({"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"})
    url = f"https://www.youtube.com/oembed?{query}"
    request = Request(url, headers={"User-Agent": "AGU-public-research-check/1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            return response.status == 200
    except (HTTPError, URLError, TimeoutError):
        return False


def probe_youtube_media(
    video_id: str,
    *,
    executable: str = "yt-dlp",
    maximum_height: int = 720,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Probe actual media extraction separately from public metadata visibility."""

    resolved = shutil.which(executable)
    if resolved is None:
        return {
            "status": "probe_unavailable",
            "error": f"executable not found: {executable}",
        }
    command = [
        resolved,
        "--no-playlist",
        "--simulate",
        "--no-warnings",
        "--print",
        "%(format_id)s|%(height)s|%(filesize_approx)s|%(duration)s",
        "-f",
        f"bv*[height<={maximum_height}]+ba/b[height<={maximum_height}]",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "probe_error",
            "error": f"yt-dlp probe exceeded {timeout_seconds:g} seconds",
        }
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip().splitlines()
        return {
            "status": "unavailable",
            "error": error[-1][:500] if error else "unknown yt-dlp error",
        }
    values = result.stdout.strip().split("|")
    if len(values) != 4:
        return {"status": "probe_error", "error": "unexpected yt-dlp output"}
    format_id, height, estimated_bytes, duration = values
    return {
        "status": "available",
        "format_id": format_id,
        "height": _optional_int(height),
        "estimated_bytes": _optional_int(estimated_bytes),
        "duration_seconds": _optional_float(duration),
    }


def _load_nba_game(
    root: Path,
    slug: str,
    *,
    metadata_visibility: Mapping[str, bool],
    playback_probes: Mapping[str, Mapping[str, Any]],
    media_paths: Mapping[str, Path],
    media_override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    game_root = root / "games" / slug
    metadata_path = game_root / "metadata.json"
    box_score_path = game_root / "box-score.jsonl"
    play_by_play_path = game_root / "play-by-play.jsonl"
    for path in (metadata_path, box_score_path, play_by_play_path):
        if not path.is_file():
            raise ValueError(f"NBA_Games artifact is missing: {slug}/{path.name}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    box_score = _read_jsonl(box_score_path)
    play_by_play = _read_jsonl(play_by_play_path)
    source_game = metadata.get("source_game") or {}
    official_game = metadata.get("official_game") or {}
    youtube_id = str(source_game.get("id") or "")
    game_id = str(official_game.get("game_id") or "")
    if not youtube_id or not game_id:
        raise ValueError(f"NBA_Games metadata lacks stable IDs: {slug}")

    active_players: list[dict[str, Any]] = []
    box_score_complete = True
    for row in box_score:
        if row.get("row_type") != "player" or not _has_minutes(row):
            continue
        statistics = row.get("statistics") or {}
        missing_fields = [field for field in REQUIRED_PLAYER_STAT_FIELDS if field not in statistics]
        if missing_fields:
            box_score_complete = False
        else:
            _validate_player_statistics(statistics, slug=slug, person_id=row.get("person_id"))
        active_players.append(
            {
                "person_id": int(row["person_id"]),
                "name": " ".join(
                    part for part in (str(row.get("first_name") or ""), str(row.get("family_name") or "")) if part
                ),
                "team_tricode": str(row.get("team_tricode") or ""),
            }
        )
    if not active_players:
        raise ValueError(f"NBA_Games box score has no active players: {slug}")

    action_counts = Counter(str(row.get("actionType") or "") for row in play_by_play)
    descriptions = "\n".join(str(row.get("description") or "").upper() for row in play_by_play)
    evidence_counts = {
        "made_shot": action_counts["Made Shot"],
        "missed_shot": action_counts["Missed Shot"],
        "rebound": action_counts["Rebound"],
        "turnover": action_counts["Turnover"],
        "assist": descriptions.count(" AST"),
        "steal": descriptions.count(" STEAL"),
        "block": descriptions.count(" BLOCK"),
    }
    play_by_play_complete = all(value > 0 for value in evidence_counts.values())
    media_path = media_paths.get(youtube_id)
    local_media = None
    if media_path is not None:
        if not media_path.is_file():
            raise ValueError(f"local media is missing: {youtube_id}")
        local_media = {
            "filename": media_path.name,
            "size_bytes": media_path.stat().st_size,
            "sha256": _file_sha256(media_path),
        }
    playback_probe = dict(playback_probes.get(youtube_id, {"status": "not_checked"}))
    if playback_probe.get("status") not in {
        "available",
        "unavailable",
        "probe_unavailable",
        "probe_error",
        "not_checked",
    }:
        raise ValueError(f"invalid playback probe status: {youtube_id}")
    media_source = _validated_media_override(media_override) if media_override is not None else None
    return {
        "slug": slug,
        "game_id": game_id,
        "game_date": str(official_game.get("game_date") or ""),
        "teams": sorted([str(official_game.get("away_abbr") or ""), str(official_game.get("home_abbr") or "")]),
        "youtube_id": youtube_id,
        "youtube_url": str(source_game.get("url") or ""),
        "duration_seconds": int(source_game.get("duration") or 0),
        "youtube_metadata_visible": metadata_visibility.get(youtube_id),
        "youtube_playback_probe": playback_probe,
        "media_source": media_source,
        "local_media": local_media,
        "active_players": sorted(active_players, key=lambda item: item["person_id"]),
        "box_score_complete": box_score_complete,
        "play_by_play_complete": play_by_play_complete,
        "play_by_play_evidence_counts": evidence_counts,
        "truth_artifacts": {
            "box_score_path": f"games/{slug}/box-score.jsonl",
            "box_score_sha256": _file_sha256(box_score_path),
            "play_by_play_path": f"games/{slug}/play-by-play.jsonl",
            "play_by_play_sha256": _file_sha256(play_by_play_path),
            "metadata_path": f"games/{slug}/metadata.json",
            "metadata_sha256": _file_sha256(metadata_path),
        },
    }


def _validated_media_override(value: Mapping[str, Any]) -> dict[str, Any]:
    provider = str(value.get("provider") or "")
    identifier = str(value.get("identifier") or "")
    filename = str(value.get("filename") or "")
    url = str(value.get("url") or "")
    match_basis = str(value.get("match_basis") or "")
    if provider != "internet_archive":
        raise ValueError(f"unsupported media override provider: {provider}")
    parsed = urlparse(url)
    expected_prefix = f"/download/{identifier}/"
    if (
        not identifier
        or not filename
        or not match_basis
        or parsed.scheme != "https"
        or parsed.hostname != "archive.org"
        or not parsed.path.startswith(expected_prefix)
    ):
        raise ValueError("media override requires a trusted Internet Archive download URL")
    return {
        "provider": provider,
        "identifier": identifier,
        "filename": filename,
        "url": url,
        "declared_license": value.get("declared_license"),
        "match_basis": match_basis,
        "redistribution_permitted": False,
    }


def _media_identity(game: Mapping[str, Any]) -> tuple[str, ...]:
    source = game.get("media_source")
    if isinstance(source, Mapping):
        return (
            str(source.get("provider") or ""),
            str(source.get("identifier") or ""),
            str(source.get("filename") or ""),
        )
    return ("youtube", str(game.get("youtube_id") or ""))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path.name}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL artifact is empty: {path.name}")
    return rows


def _has_minutes(row: Mapping[str, Any]) -> bool:
    value = str((row.get("statistics") or {}).get("minutes") or "")
    if not value or value == "0:00":
        return False
    try:
        minutes, seconds = value.split(":", maxsplit=1)
        return int(minutes) > 0 or float(seconds) > 0
    except (TypeError, ValueError):
        return False


def _validate_player_statistics(
    statistics: Mapping[str, Any],
    *,
    slug: str,
    person_id: Any,
) -> None:
    values = {field: int(statistics[field]) for field in REQUIRED_PLAYER_STAT_FIELDS}
    if any(value < 0 for value in values.values()):
        raise ValueError(f"negative player statistics: {slug}/{person_id}")
    if values["threePointersMade"] > values["fieldGoalsMade"]:
        raise ValueError(f"three-point makes exceed field goals: {slug}/{person_id}")
    if values["threePointersAttempted"] > values["fieldGoalsAttempted"]:
        raise ValueError(f"three-point attempts exceed field goals: {slug}/{person_id}")
    if values["fieldGoalsMade"] > values["fieldGoalsAttempted"]:
        raise ValueError(f"field-goal makes exceed attempts: {slug}/{person_id}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
