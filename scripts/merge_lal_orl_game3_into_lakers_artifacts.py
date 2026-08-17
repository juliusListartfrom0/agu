#!/usr/bin/env python3
"""Add the Game 3 hard-negative augmentation to the LAL held-game corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.analysis.shot_validity_scene_state import (
    file_sha256,
    seal_scene_embedding_artifact,
    verify_scene_embedding_artifact,
)
from app.analysis.shot_validity_video_backbone import (
    seal_video_embedding_artifact,
    verify_video_embedding_artifact,
)
from app.analysis.training_annotation import verify_training_annotation_manifest

ROOT = Path(__file__).resolve().parents[1]
LAKERS = ROOT / "analysis_outputs/public_research/lakers_magic_extra_v1"
GAME3 = ROOT / "analysis_outputs/public_research/lal_orl_game3_v2"
VIDEO_ASSET = ROOT / "dataset/public_sources/nba_games/media_new_dev_v2/Agffi33pz8w.mp4"


def main() -> int:
    base_manifest = verify_training_annotation_manifest(_read(LAKERS / "training_manifest_v2.json"))
    label_path = GAME3 / "labels_sealed_v1.json"
    bundle_path = GAME3 / "candidate_bundle_v1.json"
    manifest = dict(base_manifest)
    manifest["source_videos"] = list(base_manifest["source_videos"]) + [_asset(VIDEO_ASSET)]
    manifest["annotation_files"] = list(base_manifest["annotation_files"]) + [_asset(label_path)]
    manifest["candidate_bundles"] = list(base_manifest.get("candidate_bundles", [])) + [_asset(bundle_path)]
    manifest["base_manifest_sha256"] = base_manifest["manifest_sha256"]
    manifest["augmentation_scope"] = (
        "LAL-Magic held-game corpus plus LAL-ORL Game 3 hard-negative windows; training-only"
    )
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path = LAKERS / "training_manifest_game3_v3.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scene = _merge_scene(
        _read(LAKERS / "scene_merged_v1.json"),
        _read(GAME3 / "scene_extra_v1.json"),
        manifest,
        label_path,
        "LAL-Magic corpus plus 32 Game 3 raw-reviewed windows",
    )
    scene_path = LAKERS / "scene_merged_game3_v1.json"
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    outputs = {"manifest_sha256": manifest["manifest_sha256"], "scene_artifact_sha256": scene["artifact_sha256"]}
    for name in ("mvit", "swin"):
        base_path = LAKERS / f"video_{name}_merged_v1.json"
        extra_path = GAME3 / f"{name}_extra_v1.json"
        if not base_path.exists() or not extra_path.exists():
            continue
        base = verify_video_embedding_artifact(_read(base_path))
        extra = verify_video_embedding_artifact(_read(extra_path))
        merged = _merge_video(base, extra, manifest, label_path, "LAL-Magic corpus plus 32 Game 3 raw-reviewed windows")
        output = LAKERS / f"video_{name}_merged_game3_v1.json"
        output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs[f"{name}_artifact_sha256"] = merged["artifact_sha256"]
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


def _merge_scene(base: dict[str, object], extra: dict[str, object], manifest: dict[str, object], label_path: Path, scope: str) -> dict[str, object]:
    base = verify_scene_embedding_artifact(base)
    extra = verify_scene_embedding_artifact(extra)
    merged = dict(base)
    merged["training_manifest_sha256"] = str(manifest["manifest_sha256"])
    merged["training_annotation_sha256"] = sorted(set(base.get("training_annotation_sha256", [])) | {file_sha256(label_path)})
    merged["source_video_sha256"] = sorted(set(base.get("source_video_sha256", [])) | set(extra.get("source_video_sha256", [])))
    merged["base_scene_artifact_sha256"] = base["artifact_sha256"]
    merged["augmentation_artifact_sha256"] = extra["artifact_sha256"]
    merged["augmentation_scope"] = scope
    merged["examples"] = list(base["examples"]) + list(extra["examples"])
    merged.pop("artifact_sha256", None)
    return seal_scene_embedding_artifact(merged)


def _merge_video(base: dict[str, object], extra: dict[str, object], manifest: dict[str, object], label_path: Path, scope: str) -> dict[str, object]:
    merged = dict(base)
    merged["training_manifest_sha256"] = str(manifest["manifest_sha256"])
    merged["training_annotation_sha256"] = sorted(set(base.get("training_annotation_sha256", [])) | {file_sha256(label_path)})
    merged["source_video_sha256"] = sorted(set(base.get("source_video_sha256", [])) | set(extra.get("source_video_sha256", [])))
    merged["base_video_artifact_sha256"] = base["artifact_sha256"]
    merged["augmentation_artifact_sha256"] = extra["artifact_sha256"]
    merged["augmentation_scope"] = scope
    merged["examples"] = list(base["examples"]) + list(extra["examples"])
    merged.pop("artifact_sha256", None)
    return seal_video_embedding_artifact(merged)


def _asset(path: Path) -> dict[str, object]:
    return {"filename": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
