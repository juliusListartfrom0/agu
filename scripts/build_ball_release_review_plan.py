#!/usr/bin/env python3
"""Build a hash-bound, label-free hard-example ball-release review plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ball_release_annotation import (  # noqa: E402
    build_hard_review_plan,
    verify_ball_release_plan,
    verify_scene_fusion_screen,
)
from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    verify_scene_embedding_artifact,
)
from app.analysis.training_annotation import (  # noqa: E402
    verify_training_annotation_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-artifact", type=Path, required=True)
    parser.add_argument("--fusion-screen", type=Path, required=True)
    parser.add_argument(
        "--candidate-bundle", type=Path, action="append", required=True
    )
    parser.add_argument("--exclude-plan", type=Path, action="append", default=[])
    parser.add_argument("--per-game-per-class", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_plan(
    *,
    manifest_path: Path,
    scene_artifact_path: Path,
    fusion_screen_path: Path,
    candidate_bundle_paths: list[Path],
    per_game_per_class: int,
    exclude_plan_paths: list[Path] | None = None,
) -> dict[str, object]:
    manifest = verify_training_annotation_manifest(_read_json(manifest_path))
    scene = verify_scene_embedding_artifact(_read_json(scene_artifact_path))
    fusion = verify_scene_fusion_screen(_read_json(fusion_screen_path))
    if scene["training_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("scene artifact does not match the training manifest")
    if fusion["training_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("fusion screen does not match the training manifest")
    if fusion["scene_embedding_artifact_sha256"] != scene["artifact_sha256"]:
        raise ValueError("fusion screen does not match the scene artifact")

    allowed_sources = {
        str(row["sha256"]) for row in manifest.get("source_videos", [])
    }
    events: dict[tuple[str, str, str], dict[str, object]] = {}
    seen_bundle_hashes: set[str] = set()
    for path in candidate_bundle_paths:
        bundle = verify_raw_only_bundle(_read_json(path))
        if len(bundle.raw_videos) != 1:
            raise ValueError("each candidate bundle must bind exactly one raw video")
        source_sha = bundle.raw_videos[0].sha256
        if source_sha not in allowed_sources:
            raise ValueError("candidate bundle source is not in the training manifest")
        bundle_sha = str(bundle.bundle_sha256)
        if bundle_sha in seen_bundle_hashes:
            raise ValueError("candidate bundle was provided more than once")
        seen_bundle_hashes.add(bundle_sha)
        for event in bundle.events:
            key = (source_sha, bundle_sha, event.event_id)
            if key in events:
                raise ValueError("candidate bundle events must be unique")
            events[key] = {
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
            }

    scene_keys = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
        for row in scene["examples"]
    }
    missing = scene_keys - set(events)
    if missing:
        raise ValueError(
            f"candidate bundles do not cover {len(missing)} scene examples"
        )
    best_variant = fusion["best_variant"]
    excluded_keys: set[tuple[str, str]] = set()
    excluded_plan_sha256s = []
    for path in exclude_plan_paths or []:
        prior = verify_ball_release_plan(_read_json(path))
        excluded_plan_sha256s.append(str(prior["artifact_sha256"]))
        excluded_keys.update(
            (
                str(row["source_video_sha256"]),
                str(row["event_id"]),
            )
            for row in prior["examples"]
        )
    return build_hard_review_plan(
        training_manifest_sha256=str(manifest["manifest_sha256"]),
        scene_artifact_sha256=str(scene["artifact_sha256"]),
        fusion_artifact_sha256=str(fusion["artifact_sha256"]),
        scene_examples=scene["examples"],
        oof_predictions=best_variant["oof_predictions"],
        events=events,
        per_game_per_class=per_game_per_class,
        excluded_event_keys=excluded_keys,
        excluded_plan_sha256s=excluded_plan_sha256s,
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def main() -> int:
    args = parse_args()
    artifact = build_plan(
        manifest_path=args.manifest,
        scene_artifact_path=args.scene_artifact,
        fusion_screen_path=args.fusion_screen,
        candidate_bundle_paths=args.candidate_bundle,
        per_game_per_class=args.per_game_per_class,
        exclude_plan_paths=args.exclude_plan,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(artifact["examples"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
