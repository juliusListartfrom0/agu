#!/usr/bin/env python3
"""Build 24-frame player-ball-rim relation features from existing perception."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.causal_phase_ball_windows import (  # noqa: E402
    verify_causal_ball_perception,
)
from app.analysis.causal_phase_entity_relations import (  # noqa: E402
    extract_prepared_entity_relation_features,
    prepare_entity_relation_perception,
    seal_dense_entity_relation_artifact,
)
from app.analysis.causal_shot_phase_review import (  # noqa: E402
    verify_causal_shot_phase_review_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--player", type=Path, action="append", required=True)
    parser.add_argument("--rim", type=Path, action="append", required=True)
    parser.add_argument("--ball", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_entity_relations(
    *,
    plan_path: Path,
    video_paths: list[Path],
    player_paths: list[Path],
    rim_paths: list[Path],
    ball_paths: list[Path],
) -> dict[str, Any]:
    plan = verify_causal_shot_phase_review_plan(_read_json(plan_path))
    if not (
        len(video_paths)
        == len(player_paths)
        == len(rim_paths)
        == len(ball_paths)
        == len(plan["source_video_sha256s"])
    ):
        raise ValueError("entity relation inputs must be paired by source")

    video_by_sha: dict[str, dict[str, Any]] = {}
    video_sha_by_path: dict[Path, str] = {}
    for path in video_paths:
        source_sha = _file_sha256(path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open entity relation video: {path.name}")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if (
            source_sha in video_by_sha
            or width < 1
            or height < 1
            or frame_count < 1
        ):
            raise ValueError("invalid or duplicate entity relation video")
        video_by_sha[source_sha] = {
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }
        video_sha_by_path[path.resolve()] = source_sha
    if set(video_by_sha) != set(plan["source_video_sha256s"]):
        raise ValueError("entity relation videos must exactly cover the plan")

    examples = []
    observed_sources = set()
    source_artifact_hashes = []
    for video_path, player_path, rim_path, ball_path in zip(
        video_paths,
        player_paths,
        rim_paths,
        ball_paths,
        strict=True,
    ):
        source_sha = video_sha_by_path[video_path.resolve()]
        video = video_by_sha[source_sha]
        payloads = {
            "player": _read_json(player_path),
            "rim": _read_json(rim_path),
            "basketball": verify_causal_ball_perception(
                _read_json(ball_path),
                plan=plan,
            ),
        }
        if any(
            str(payload.get("raw_video", {}).get("sha256") or "")
            != source_sha
            for payload in payloads.values()
        ):
            raise ValueError("entity relation perception/video binding mismatch")
        if source_sha in observed_sources:
            raise ValueError("duplicate entity relation perception source")
        observed_sources.add(source_sha)
        perception = {
            "player": prepare_entity_relation_perception(
                payloads["player"],
                expected_object_type="player",
            ),
            "rim": prepare_entity_relation_perception(
                payloads["rim"],
                expected_object_type="rim",
            ),
            "basketball": prepare_entity_relation_perception(
                payloads["basketball"],
                expected_object_type="basketball",
            ),
        }
        source_artifact_hashes.extend(
            _file_sha256(path)
            for path in (player_path, rim_path, ball_path)
        )
        source_rows = [
            row
            for row in plan["examples"]
            if row["source_video_sha256"] == source_sha
        ]
        for row in source_rows:
            if (
                int(row["frame_count"]) != video["frame_count"]
                or max(row["frame_indexes"]) >= video["frame_count"]
            ):
                raise ValueError("entity relation video metadata mismatch")
            examples.append(
                {
                    "phase_review_id": row["phase_review_id"],
                    "source_video_sha256": source_sha,
                    "candidate_bundle_sha256": row[
                        "candidate_bundle_sha256"
                    ],
                    "event_id": row["event_id"],
                    "frame_indexes": row["frame_indexes"],
                    "features": extract_prepared_entity_relation_features(
                        frame_indexes=row["frame_indexes"],
                        frame_width=video["width"],
                        frame_height=video["height"],
                        player_perception=perception["player"],
                        rim_perception=perception["rim"],
                        ball_perception=perception["basketball"],
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "stage": "dense_entity_relations",
                    "source_video": video_path.name,
                    "examples": len(examples),
                    "total_examples": len(plan["examples"]),
                }
            ),
            flush=True,
        )
        del payloads, perception
        gc.collect()
    if observed_sources != set(plan["source_video_sha256s"]):
        raise ValueError("entity relation perception sources do not cover plan")
    return seal_dense_entity_relation_artifact(
        {
            "review_plan_sha256": plan["artifact_sha256"],
            "source_video_sha256s": plan["source_video_sha256s"],
            "sealed_blind_video_sha256s": plan[
                "sealed_blind_video_sha256s"
            ],
            "source_artifact_sha256s": sorted(source_artifact_hashes),
            "examples": examples,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    artifact = build_entity_relations(
        plan_path=args.plan,
        video_paths=args.video,
        player_paths=args.player,
        rim_paths=args.rim,
        ball_paths=args.ball,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "example_count": len(artifact["examples"]),
                "feature_dimension": len(artifact["feature_names"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
