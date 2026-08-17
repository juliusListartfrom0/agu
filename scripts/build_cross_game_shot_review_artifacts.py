#!/usr/bin/env python3
"""Seal one independently reviewed cross-game shot-window set offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.analysis.cross_game_shot_windows import verify_shot_window_spec
from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA
from app.analysis.shot_validity_scene_state import file_sha256
from app.analysis.training_annotation import verify_training_annotation_manifest


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--game-id", required=True)
    return parser.parse_args()


def build_artifacts(
    *,
    video_path: Path,
    spec_path: Path,
    labels_path: Path,
    base_manifest_path: Path,
    output_dir: Path,
    game_id: str,
) -> dict[str, Any]:
    spec = verify_shot_window_spec(_read(spec_path))
    source_sha = file_sha256(video_path)
    if source_sha != spec["source_video_sha256"]:
        raise ValueError("video does not match the hash-bound candidate spec")
    label_payload = _read(labels_path)
    if label_payload.get("source_video_sha256") != source_sha:
        raise ValueError("labels do not match the source video")
    if label_payload.get("candidate_spec_sha256") != spec["artifact_sha256"]:
        raise ValueError("labels do not match the candidate spec")
    candidates = list(spec["candidates"])
    rows_by_id = {str(row.get("event_id")): row for row in label_payload.get("examples", [])}
    expected_ids = {str(row["event_id"]) for row in candidates}
    if set(rows_by_id) != expected_ids:
        raise ValueError("labels must contain every candidate exactly once")
    if any(not isinstance(row.get("event_present"), bool) for row in rows_by_id.values()):
        raise ValueError("every event_present label must be boolean")

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

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = seal_raw_only_predictions(
        game_id=game_id,
        raw_video_paths=[video_path],
        events=events,
        config={
            "selection": spec["selection_protocol"],
            "candidate_spec_sha256": spec["artifact_sha256"],
            "runtime": False,
        },
        model_provenance={"producer": "codex", "purpose": "training_only"},
        completed_at=datetime.now(timezone.utc),
    )
    bundle_path = output_dir / "candidate_bundle_v1.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")

    labels = {
        "schema_version": SHOT_VALIDITY_LABEL_SCHEMA,
        "purpose": "training_annotation_only",
        "runtime_consumable": False,
        "source_video_sha256": source_sha,
        "candidate_bundle_sha256": bundle.bundle_sha256,
        "review_protocol": label_payload.get("review_protocol", "raw-frame-contact-sheet"),
        "reviewer": label_payload.get("reviewer", "codex"),
        "examples": [],
    }
    for candidate in candidates:
        decision = rows_by_id[str(candidate["event_id"])]
        labels["examples"].append(
            {
                "event_id": str(candidate["event_id"]),
                "start_frame": int(candidate["start_frame"]),
                "end_frame": int(candidate["end_frame"]),
                "release_frame": decision.get("release_frame"),
                "event_present": bool(decision["event_present"]),
                "review_note": str(decision.get("review_note", "")),
            }
        )
    labels_path_out = output_dir / "labels_sealed_v1.json"
    labels_path_out.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    base_manifest = verify_training_annotation_manifest(_read(base_manifest_path))
    manifest_payload = dict(base_manifest)
    source_asset = {"filename": video_path.name, "sha256": source_sha, "size_bytes": video_path.stat().st_size}
    annotation_asset = {
        "filename": labels_path_out.name,
        "sha256": file_sha256(labels_path_out),
        "size_bytes": labels_path_out.stat().st_size,
    }
    manifest_payload["source_videos"] = [
        *[item for item in base_manifest["source_videos"] if item["sha256"] != source_sha],
        source_asset,
    ]
    manifest_payload["annotation_files"] = [
        *[item for item in base_manifest["annotation_files"] if item["sha256"] != annotation_asset["sha256"]],
        annotation_asset,
    ]
    manifest_payload["candidate_bundles"] = [
        {"filename": bundle_path.name, "sha256": file_sha256(bundle_path), "size_bytes": bundle_path.stat().st_size}
    ]
    manifest_payload["base_manifest_sha256"] = base_manifest["manifest_sha256"]
    manifest_payload["augmentation_scope"] = (
        f"{game_id} hash-bound uniform raw review; {len(candidates)} windows; "
        f"{sum(bool(row['event_present']) for row in labels['examples'])} positive; training-only"
    )
    manifest_payload.pop("manifest_sha256", None)
    manifest_payload["manifest_sha256"] = _canonical_sha256(manifest_payload)
    manifest_path = output_dir / "training_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "bundle_sha256": bundle.bundle_sha256,
        "labels_sha256": annotation_asset["sha256"],
        "manifest_sha256": manifest_payload["manifest_sha256"],
        "candidate_count": len(candidates),
        "positive_count": sum(bool(row["event_present"]) for row in labels["examples"]),
    }


def main() -> int:
    args = parse_args()
    print(json.dumps(build_artifacts(
        video_path=args.video,
        spec_path=args.spec,
        labels_path=args.labels,
        base_manifest_path=args.base_manifest,
        output_dir=args.output_dir,
        game_id=args.game_id,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
