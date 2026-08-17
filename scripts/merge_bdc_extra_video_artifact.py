#!/usr/bin/env python3
"""Merge the extra Bdc video embeddings with the sealed 511-row artifact."""

from __future__ import annotations

import json
from pathlib import Path

from app.analysis.shot_validity_video_backbone import (
    seal_video_embedding_artifact,
    verify_video_embedding_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/public_research/bdc_extra_hard_negative_v1"
BASE = ROOT / "analysis_outputs/public_research/shot_validity_mvit_v2_s_full511_ball_release_corrected_batch4_v1.json"
EXTRA = OUT / "video_mvit_extra_v1.json"
MANIFEST = OUT / "training_manifest_v2.json"


def main() -> int:
    base = verify_video_embedding_artifact(json.loads(BASE.read_text(encoding="utf-8")))
    extra = verify_video_embedding_artifact(json.loads(EXTRA.read_text(encoding="utf-8")))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if base["backbone"] != extra["backbone"] or base["embedding_dimension"] != extra["embedding_dimension"]:
        raise ValueError("base and extra video artifacts use different backbones")
    keys = {
        (str(row["source_video_sha256"]), str(row["candidate_bundle_sha256"]), str(row["event_id"]))
        for row in base["examples"]
    }
    extra_keys = {
        (str(row["source_video_sha256"]), str(row["candidate_bundle_sha256"]), str(row["event_id"]))
        for row in extra["examples"]
    }
    if keys & extra_keys:
        raise ValueError("extra video rows overlap the base artifact")
    payload = dict(base)
    payload["training_manifest_sha256"] = str(manifest["manifest_sha256"])
    payload["training_annotation_sha256"] = sorted(
        set(base.get("training_annotation_sha256", [])) | set(extra.get("training_annotation_sha256", []))
    )
    payload["source_video_sha256"] = sorted(
        set(base.get("source_video_sha256", [])) | set(extra.get("source_video_sha256", []))
    )
    payload["base_video_artifact_sha256"] = base["artifact_sha256"]
    payload["augmentation_artifact_sha256"] = extra["artifact_sha256"]
    payload["augmentation_scope"] = "32 Codex-reviewed Bdc non-overlap windows; training-only"
    payload["examples"] = list(base["examples"]) + list(extra["examples"])
    payload.pop("artifact_sha256", None)
    merged = seal_video_embedding_artifact(payload)
    path = OUT / "video_mvit_merged_v1.json"
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(merged["examples"]), "artifact_sha256": merged["artifact_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
