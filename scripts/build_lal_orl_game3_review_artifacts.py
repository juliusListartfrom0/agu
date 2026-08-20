#!/usr/bin/env python3
"""Seal the raw-reviewed LAL–ORL Game 3 windows for offline training only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.analysis.cross_game_shot_windows import verify_shot_window_spec
from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA
from app.analysis.shot_validity_scene_state import file_sha256
from app.analysis.training_annotation import verify_training_annotation_manifest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/public_research/lal_orl_game3_v2"
VIDEO = ROOT / "dataset/public_sources/nba_games/media_new_dev_v2/Agffi33pz8w.mp4"
SPEC = OUT / "candidate_spec_v1.json"
LABEL_SPEC = OUT / "labels_v1.json"
BASE_MANIFEST = ROOT / "analysis_outputs/public_research/shot_validity_codex_full511_manifest.json"


def main() -> int:
    spec = verify_shot_window_spec(json.loads(SPEC.read_text(encoding="utf-8")))
    if file_sha256(VIDEO) != spec["source_video_sha256"]:
        raise ValueError("review video does not match the hash-bound candidate spec")
    label_spec = json.loads(LABEL_SPEC.read_text(encoding="utf-8"))
    if label_spec.get("source_video_sha256") != spec["source_video_sha256"]:
        raise ValueError("review labels do not match the source video")
    if label_spec.get("candidate_spec_sha256") != spec["artifact_sha256"]:
        raise ValueError("review labels do not match the candidate spec")
    candidates = list(spec["candidates"])
    by_id = {str(row["event_id"]): row for row in label_spec["examples"]}
    if set(by_id) != {str(row["event_id"]) for row in candidates}:
        raise ValueError("candidate and review label ids do not match")

    events = []
    for row in candidates:
        event_id = str(row["event_id"])
        events.append(
            GameEventResponse.model_validate(
                {
                    "event_id": event_id,
                    "revision": 1,
                    "event_type": "field_goal_attempt",
                    "source_video_id": "video_001",
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "outcome": "unknown",
                    "status": "candidate",
                    "confidence": 0.5,
                    "reason": "offline cross-game review window; not a runtime answer",
                    "evidence": [
                        {
                            "evidence_id": f"{event_id}-anchor",
                            "kind": "offline_uniform_extra_window",
                            "source_video_id": "video_001",
                            "start_frame": int(row["start_frame"]),
                            "end_frame": int(row["end_frame"]),
                            "confidence": 0.5,
                            "details": {
                                "candidate_event_frame": int(row["anchor_frame"]),
                                "selection_seed": spec["selection_seed"],
                                "label_selection_not_used": True,
                            },
                        }
                    ],
                }
            )
        )

    bundle = seal_raw_only_predictions(
        game_id="lal-orl-game3-cross-game-v2",
        raw_video_paths=[VIDEO],
        events=events,
        config={
            "selection": spec["selection_protocol"],
            "candidate_spec_sha256": spec["artifact_sha256"],
            "runtime": False,
        },
        model_provenance={"producer": "codex", "purpose": "training_only"},
        completed_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
    )
    bundle_path = OUT / "candidate_bundle_v1.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")

    labels = {
        "schema_version": SHOT_VALIDITY_LABEL_SCHEMA,
        "purpose": "training_annotation_only",
        "runtime_consumable": False,
        "source_video_sha256": file_sha256(VIDEO),
        "candidate_bundle_sha256": bundle.bundle_sha256,
        "review_protocol": label_spec["review_protocol"],
        "reviewer": label_spec.get("reviewer", "codex"),
        "examples": [],
    }
    for candidate in candidates:
        decision = by_id[str(candidate["event_id"])]
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
    source_asset = {
        "filename": VIDEO.name,
        "sha256": file_sha256(VIDEO),
        "size_bytes": VIDEO.stat().st_size,
    }
    annotation_asset = {
        "filename": labels_path.name,
        "sha256": file_sha256(labels_path),
        "size_bytes": labels_path.stat().st_size,
    }
    manifest_payload["source_videos"] = [
        *list(base_manifest["source_videos"]),
        source_asset,
    ]
    manifest_payload["annotation_files"] = [
        *list(base_manifest["annotation_files"]),
        annotation_asset,
    ]
    manifest_payload["candidate_bundles"] = [
        {
            "filename": bundle_path.name,
            "sha256": file_sha256(bundle_path),
            "size_bytes": bundle_path.stat().st_size,
        }
    ]
    manifest_payload["base_manifest_sha256"] = base_manifest["manifest_sha256"]
    manifest_payload["augmentation_scope"] = (
        "LAL-ORL Game 3 hash-bound uniform raw review; 32 windows, one positive; training-only"
    )
    manifest_payload.pop("manifest_sha256", None)
    manifest_payload["manifest_sha256"] = _canonical_sha256(manifest_payload)
    manifest_path = OUT / "training_manifest_v2.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "bundle_sha256": bundle.bundle_sha256,
                "labels_sha256": annotation_asset["sha256"],
                "manifest_sha256": manifest_payload["manifest_sha256"],
                "candidate_count": len(candidates),
                "positive_count": sum(bool(row["event_present"]) for row in labels["examples"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
