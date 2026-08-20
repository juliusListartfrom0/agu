#!/usr/bin/env python3
"""Merge the LAL–ORL Game 3 scene augmentation into the sealed base artifact."""

from __future__ import annotations

import json
from pathlib import Path

from app.analysis.shot_validity_scene_state import (
    file_sha256,
    seal_scene_embedding_artifact,
    verify_scene_embedding_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/public_research/lal_orl_game3_v2"
BASE = ROOT / "analysis_outputs/public_research/shot_validity_mobilenet_v3_small_phase3_full511_ball_release_corrected_batch4_v1.json"


def main() -> int:
    base = verify_scene_embedding_artifact(_read(BASE))
    extra = verify_scene_embedding_artifact(_read(OUT / "scene_extra_v1.json"))
    manifest = _read(OUT / "training_manifest_v2.json")
    labels = OUT / "labels_sealed_v1.json"
    source = "ecbc705c03044ec33dfa16aa50151ebcb5436b213106239ffba7d28c8ff9d04c"
    merged = dict(base)
    merged["training_manifest_sha256"] = str(manifest["manifest_sha256"])
    merged["training_annotation_sha256"] = sorted(
        set(base.get("training_annotation_sha256", [])) | {file_sha256(labels)}
    )
    merged["source_video_sha256"] = sorted(
        set(base.get("source_video_sha256", [])) | {source}
    )
    merged["base_scene_artifact_sha256"] = base["artifact_sha256"]
    merged["augmentation_artifact_sha256"] = extra["artifact_sha256"]
    merged["augmentation_scope"] = "32 LAL-ORL Game 3 raw-reviewed windows; training-only"
    merged["examples"] = list(base["examples"]) + list(extra["examples"])
    merged.pop("artifact_sha256", None)
    sealed = seal_scene_embedding_artifact(merged)
    output = OUT / "scene_merged_v1.json"
    output.write_text(json.dumps(sealed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(sealed["examples"]), "artifact_sha256": sealed["artifact_sha256"]}, indent=2))
    return 0


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
