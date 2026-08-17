#!/usr/bin/env python3
"""Seal and embed the manually reviewed non-blind Bdc extra windows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity_sampling_window import resolve_shot_sampling_window
from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    extract_scene_phase_embeddings,
    file_sha256,
    load_scene_backbone,
    seal_scene_embedding_artifact,
    verify_scene_embedding_artifact,
)
from app.analysis.training_annotation import verify_training_annotation_manifest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/public_research/bdc_extra_hard_negative_v1"
VIDEO = ROOT / "dataset/public_sources/nba_games/media/BdcP-XwUCk8.mp4"
SPEC = OUT / "candidate_spec_v1.json"
LABEL_SPEC = OUT / "labels_v1.json"
BASE_MANIFEST = ROOT / "analysis_outputs/public_research/shot_validity_codex_full511_manifest.json"
BASE_SCENE = ROOT / "analysis_outputs/public_research/shot_validity_mobilenet_v3_small_phase3_full511_ball_release_corrected_batch4_v1.json"
BACKBONE = ROOT / "model_checkpoints/scene/mobilenet_v3_small-047dcff4.pth"


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    label_spec = json.loads(LABEL_SPEC.read_text(encoding="utf-8"))
    by_id = {str(row["candidate_id"]): row for row in label_spec["examples"]}
    candidates = spec["candidates"]
    if set(by_id) != {str(row["candidate_id"]) for row in candidates}:
        raise ValueError("candidate and label ids do not match")

    events = []
    for row in candidates:
        event_id = str(row["event_id"])
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        anchor_frame = int(row["anchor_frame"])
        events.append(
            GameEventResponse.model_validate(
                {
                    "event_id": event_id,
                    "revision": 1,
                    "event_type": "field_goal_attempt",
                    "source_video_id": "video_001",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "outcome": "unknown",
                    "status": "candidate",
                    "confidence": 0.5,
                    "reason": "offline extra window; not a runtime answer",
                    "evidence": [
                        {
                            "evidence_id": f"{event_id}-anchor",
                            "kind": "offline_uniform_extra_window",
                            "source_video_id": "video_001",
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "confidence": 0.5,
                            "details": {"candidate_event_frame": anchor_frame},
                        }
                    ],
                }
            )
        )

    bundle = seal_raw_only_predictions(
        game_id="bdc-extra-hard-negative-v1",
        raw_video_paths=[VIDEO],
        events=events,
        config={"selection": "stratified_nonoverlap_uniform_windows_v1", "runtime": False},
        model_provenance={"producer": "codex", "purpose": "training_only"},
        completed_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
    )
    bundle_path = OUT / "candidate_bundle_v1.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")

    labels = {
        "schema_version": "agu.shot-validity-labels.v1",
        "purpose": "training_annotation_only",
        "runtime_consumable": False,
        "source_video_sha256": file_sha256(VIDEO),
        "candidate_bundle_sha256": bundle.bundle_sha256,
        "review_protocol": label_spec["review_protocol"],
        "examples": [],
    }
    for candidate in candidates:
        decision = by_id[str(candidate["candidate_id"])]
        labels["examples"].append(
            {
                "event_id": str(candidate["event_id"]),
                "start_frame": int(candidate["start_frame"]),
                "end_frame": int(candidate["end_frame"]),
                "release_frame": None,
                "event_present": bool(decision["event_present"]),
                "review_note": str(decision["review_note"]),
            }
        )
    labels_path = OUT / "labels_sealed_v1.json"
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    base_manifest = verify_training_annotation_manifest(
        json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    )
    manifest_payload = dict(base_manifest)
    manifest_payload["annotation_files"] = list(manifest_payload["annotation_files"]) + [
        {"filename": labels_path.name, "sha256": file_sha256(labels_path), "size_bytes": labels_path.stat().st_size}
    ]
    manifest_payload["candidate_bundles"] = [
        {"filename": bundle_path.name, "sha256": file_sha256(bundle_path), "size_bytes": bundle_path.stat().st_size}
    ]
    manifest_payload["base_manifest_sha256"] = base_manifest["manifest_sha256"]
    manifest_payload["augmentation_scope"] = "Bdc non-overlap windows; Bdc remains a held game in the base screen"
    manifest_payload.pop("manifest_sha256", None)
    manifest_payload["manifest_sha256"] = canonical_sha256(manifest_payload)
    manifest_path = OUT / "training_manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    device = torch.device("cpu")
    backbone, transform = load_scene_backbone(BACKBONE, device=device)
    windows = [resolve_shot_sampling_window(event, fps=float(spec["video_fps"])) for event in events]
    embeddings = extract_scene_phase_embeddings(
        video_path=VIDEO,
        events=events,
        backbone=backbone,
        transform=transform,
        device=device,
        batch_size=8,
        sampling_windows=windows,
    )
    extra_examples = []
    for event, window, phases in zip(events, windows, embeddings, strict=True):
        decision = by_id[next(k for k, v in by_id.items() if v["event_id"] == event.event_id)]
        extra_examples.append(
            {
                "source_video_sha256": file_sha256(VIDEO),
                "candidate_bundle_sha256": bundle.bundle_sha256,
                "event_id": event.event_id,
                "event_present": bool(decision["event_present"]),
                "sampling_window": {
                    "start_frame": window.start_frame,
                    "end_frame": window.end_frame,
                    "anchor_frame": window.anchor_frame,
                    "anchor_source": window.anchor_source,
                    "protocol": window.protocol,
                },
                "phase_embeddings": phases,
            }
        )
    extra_scene = seal_scene_embedding_artifact(
        {
            "purpose": "scene_state_screening_training_only",
            "producer": "agu",
            "training_manifest_sha256": manifest_payload["manifest_sha256"],
            "training_annotation_sha256": [file_sha256(labels_path)],
            "source_video_sha256": [file_sha256(VIDEO)],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": file_sha256(BACKBONE),
            "backbone_license": "BSD-3-Clause (torchvision); upstream weight terms apply",
            "backbone_weights_url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "phase_fractions": [0.15, 0.5, 0.85],
            "sampling_protocol": "agu.raw-evidence-atomic-shot-window.v1",
            "examples": extra_examples,
        }
    )
    (OUT / "scene_extra_v1.json").write_text(json.dumps(extra_scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    base_scene = verify_scene_embedding_artifact(json.loads(BASE_SCENE.read_text(encoding="utf-8")))
    merged_payload = dict(base_scene)
    merged_payload["training_manifest_sha256"] = manifest_payload["manifest_sha256"]
    merged_payload["training_annotation_sha256"] = list(base_scene.get("training_annotation_sha256", [])) + [file_sha256(labels_path)]
    merged_payload["source_video_sha256"] = sorted(set(base_scene["source_video_sha256"]) | {file_sha256(VIDEO)})
    merged_payload["base_scene_artifact_sha256"] = base_scene["artifact_sha256"]
    merged_payload["augmentation_artifact_sha256"] = extra_scene["artifact_sha256"]
    merged_payload["augmentation_scope"] = "32 Codex-reviewed Bdc non-overlap windows; training-only"
    merged_payload["examples"] = list(base_scene["examples"]) + extra_examples
    merged_payload.pop("artifact_sha256", None)
    merged_scene = seal_scene_embedding_artifact(merged_payload)
    (OUT / "scene_merged_v1.json").write_text(json.dumps(merged_scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "bundle_sha256": bundle.bundle_sha256,
        "labels_sha256": file_sha256(labels_path),
        "manifest_sha256": manifest_payload["manifest_sha256"],
        "extra_scene_examples": len(extra_examples),
        "extra_scene_artifact_sha256": extra_scene["artifact_sha256"],
        "merged_scene_examples": len(merged_scene["examples"]),
        "merged_scene_artifact_sha256": merged_scene["artifact_sha256"],
        "positive_extra_count": sum(int(x["event_present"]) for x in extra_examples),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
