#!/usr/bin/env python3
"""Build offline-only pilot and frame-retention manifests for a sealed VRU review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_review import (  # noqa: E402
    verify_vru_causal_review,
    verify_vru_causal_review_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cross-audit", type=Path, required=True)
    parser.add_argument("--review-spec", type=Path, required=True)
    parser.add_argument("--review-plan", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--review-sealed", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-tag",
        default="v7",
        help="lowercase dataset suffix used for the sealed pilot artifact (default: v7)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention-output", type=Path, required=True)
    return parser.parse_args()


def build_manifests(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_tag = str(getattr(args, "dataset_tag", "v7") or "v7")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", dataset_tag):
        raise ValueError("dataset tag must be lowercase alphanumeric with _ or -")
    source_manifest = _read_json(args.source_manifest)
    if source_manifest.get("schema_version") != "agu.vru-basketball-source-manifest.v1":
        raise ValueError("unsupported VRU source manifest")
    source_audit = _read_json(args.source_cross_audit)
    source_manifest_sha = _file_sha256(args.source_manifest)
    source_audit_sha = str(source_audit.get("audit_sha256") or _file_sha256(args.source_cross_audit))
    review_spec = _read_json(args.review_spec)
    if review_spec.get("schema_version") != "agu.vru-causal-review-spec.v1":
        raise ValueError("unsupported VRU review spec")
    plan = verify_vru_causal_review_plan(_read_json(args.review_plan))
    sealed = verify_vru_causal_review(
        _read_json(args.review_sealed), plan=plan
    )
    decisions = _read_json(args.review_decisions)
    if decisions.get("plan_sha256") != plan["artifact_sha256"]:
        raise ValueError("decision input is not bound to the review plan")
    frame_manifest = _read_json(args.frame_manifest)
    if frame_manifest.get("schema_version") != "agu.vru-causal-review-frame-manifest.v1":
        raise ValueError("unsupported VRU frame manifest")
    if frame_manifest.get("review_plan_sha256") != plan["artifact_sha256"]:
        raise ValueError("frame manifest is not bound to the review plan")
    frame_rows = frame_manifest.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("VRU frame manifest requires frames")

    spec_by_id = {str(row["review_id"]): row for row in review_spec["examples"]}
    plan_by_id = {str(row["review_id"]): row for row in plan["examples"]}
    sealed_by_id = {str(row["review_id"]): row for row in sealed["reviews"]}
    frame_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        if not isinstance(row, dict):
            raise ValueError("VRU frame rows must be objects")
        frame_by_id[str(row.get("review_id") or "")].append(row)
    if set(spec_by_id) != set(plan_by_id) or set(plan_by_id) != set(sealed_by_id):
        raise ValueError("VRU pilot artifacts do not cover the same review IDs")

    windows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for review_id in sorted(plan_by_id):
        spec_row = spec_by_id[review_id]
        plan_row = plan_by_id[review_id]
        sealed_row = sealed_by_id[review_id]
        frame_count = len(frame_by_id.get(review_id, []))
        if frame_count != len(plan_row["frame_indexes"]):
            raise ValueError(f"frame coverage mismatch for {review_id}")
        fps = float(plan_row["source_fps"])
        start_frame = int(plan_row["frame_indexes"][0])
        end_frame = int(plan_row["frame_indexes"][-1])
        sequence = str(sealed_row["shot_sequence"])
        outcome = str(sealed_row["outcome"])
        label_counts[sequence] += 1
        windows.append(
            {
                "review_id": review_id,
                "source": _source_location(plan_row, source_manifest),
                "start_seconds": round(start_frame / fps, 6),
                "end_seconds": round(end_frame / fps, 6),
                "kind": sequence,
                "outcome": outcome,
                "notes": str(sealed_row.get("notes") or ""),
                "frame_count": frame_count,
                "release_frame": sealed_row.get("release_frame"),
                "rim_frame": sealed_row.get("rim_frame"),
                "spec_clip_id": spec_row.get("clip_id"),
            }
        )

    retention = {
        "schema_version": "agu.vru-causal-extension-retention.v1",
        "plan_sha256": plan["artifact_sha256"],
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "frame_root": str(frame_manifest.get("frame_root") or "raw_frames"),
        "frame_manifest_sha256": _file_sha256(args.frame_manifest),
        "frame_count": len(frame_rows),
        "windows": [
            {
                "review_id": review_id,
                "source_video_filename": plan_by_id[review_id]["source_video_filename"],
                "frame_count": len(frame_by_id[review_id]),
                "frames": [
                    {
                        "path": f"{frame_manifest.get('frame_root') or 'raw_frames'}/{row['relative_path']}",
                        "raw_frame_sha256": row["raw_frame_sha256"],
                        "jpeg_sha256": row["jpeg_sha256"],
                        "bytes": row["jpeg_bytes"],
                    }
                    for row in sorted(frame_by_id[review_id], key=lambda value: int(value["position"]))
                ],
            }
            for review_id in sorted(frame_by_id)
        ],
    }
    retention["artifact_sha256"] = _canonical_sha256(retention)

    pilot = {
        "schema_version": f"agu.wikimedia-hctv-hazen-randolph-vtv-pilot-annotation-manifest.{dataset_tag}",
        "generated_on": "2026-08-10",
        "dataset_id": f"wikimedia-hctv-hazen-randolph-vtv-pilot-annotation-{dataset_tag}",
        "purpose": "offline_manual_visual_review_extension_for_cross_production_causal_labels",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "pilot_only": True,
        "exhaustive": False,
        "hard_negative_gate_eligible": False,
        "training_consumable": False,
        "source_cross_audit": _relative(args.output, args.source_cross_audit),
        "source_cross_audit_sha256": source_audit_sha,
        "source_manifest": _relative(args.output, args.source_manifest),
        "source_manifest_sha256": source_manifest_sha,
        "frame_root": _relative(args.output, args.frame_manifest.parent / str(frame_manifest.get("frame_root") or "raw_frames")),
        "frame_count": len(frame_rows),
        "frame_hash_encoding": "raw_cv2_frame_bytes_for_plan_plus_rendered_jpeg_bytes_for_retention",
        "frame_sampling": f"0.2-second nominal samples across bounded {dataset_tag} windows",
        "review_spec": _relative(args.output, args.review_spec),
        "review_spec_sha256": _file_sha256(args.review_spec),
        "review_plan": _relative(args.output, args.review_plan),
        "review_plan_sha256": plan["artifact_sha256"],
        "review_decisions_input": _relative(args.output, args.review_decisions),
        "review_decisions_sha256": _file_sha256(args.review_decisions),
        "review": _relative(args.output, args.review_sealed),
        "review_sha256": sealed["artifact_sha256"],
        "frame_manifest": _relative(args.output, args.frame_manifest),
        "frame_manifest_sha256": _file_sha256(args.frame_manifest),
        "retention_manifest": _relative(args.output, args.retention_output),
        "retention_manifest_sha256": retention["artifact_sha256"],
        "windows": windows,
        "label_counts": dict(sorted(label_counts.items())),
        "caveat": "Pilot-only additive extension, non-exhaustive and pending independent human confirmation; no runtime, training truth, model promotion or blind inference consumes this artifact.",
    }
    pilot["manifest_sha256"] = _canonical_sha256(pilot)
    return pilot, retention


def _source_location(plan_row: dict[str, Any], source_manifest: dict[str, Any]) -> str:
    source_sha = str(plan_row["source_video_sha256"])
    for row in source_manifest.get("videos", []):
        if str(row.get("sha256") or "") == source_sha:
            return str(row.get("location") or row.get("clip_id") or source_sha[:12])
    return source_sha[:12]


def _relative(base_file: Path, target: Path) -> str:
    return Path(target).resolve().relative_to(ROOT).as_posix() if target.resolve().is_relative_to(ROOT) else str(target)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    normalized.pop("manifest_sha256", None)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    pilot, retention = build_manifests(args)
    args.retention_output.parent.mkdir(parents=True, exist_ok=True)
    args.retention_output.write_text(json.dumps(retention, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pilot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pilot": str(args.output), "manifest_sha256": pilot["manifest_sha256"], "retention": str(args.retention_output), "retention_sha256": retention["artifact_sha256"], "windows": len(pilot["windows"]), "frames": pilot["frame_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
