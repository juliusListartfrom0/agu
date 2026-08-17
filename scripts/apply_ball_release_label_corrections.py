#!/usr/bin/env python3
"""Derive and apply sealed ball-release corrections to embedding labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ball_release_annotation import (  # noqa: E402
    apply_ball_release_label_corrections,
    derive_ball_release_label_corrections,
)
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    seal_scene_embedding_artifact,
    verify_scene_embedding_artifact,
)
from app.analysis.shot_validity_video_backbone import (  # noqa: E402
    seal_video_embedding_artifact,
    verify_video_embedding_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--scene-input", type=Path, required=True)
    parser.add_argument("--scene-output", type=Path, required=True)
    parser.add_argument(
        "--video-pair",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        action="append",
        default=[],
    )
    parser.add_argument("--correction-output", type=Path, required=True)
    return parser.parse_args()


def apply_corrections(
    *,
    plan_path: Path,
    review_path: Path,
    scene_input: Path,
    scene_output: Path,
    video_pairs: list[tuple[Path, Path]],
    correction_output: Path,
) -> dict[str, object]:
    plan = _read_json(plan_path)
    review = _read_json(review_path)
    scene = verify_scene_embedding_artifact(_read_json(scene_input))
    corrections = derive_ball_release_label_corrections(
        plan=plan,
        review=review,
        scene_artifact_sha256=str(scene["artifact_sha256"]),
        scene_examples=scene["examples"],
    )
    corrected_scene_payload = {
        key: value
        for key, value in scene.items()
        if key not in {"artifact_sha256", "examples"}
    }
    corrected_scene_payload.update(
        {
            "purpose": "scene_state_screening_with_ball_release_corrections",
            "source_embedding_artifact_sha256": scene["artifact_sha256"],
            "label_correction_artifact_sha256": corrections[
                "artifact_sha256"
            ],
            "examples": apply_ball_release_label_corrections(
                scene["examples"], corrections=corrections
            ),
        }
    )
    corrected_scene = seal_scene_embedding_artifact(corrected_scene_payload)

    corrected_videos = []
    for source_path, output_path in video_pairs:
        video = verify_video_embedding_artifact(_read_json(source_path))
        if (
            video["training_manifest_sha256"]
            != scene["training_manifest_sha256"]
        ):
            raise ValueError("video embeddings do not match the scene manifest")
        payload = {
            key: value
            for key, value in video.items()
            if key not in {"artifact_sha256", "examples"}
        }
        payload.update(
            {
                "purpose": "video_screening_with_ball_release_corrections",
                "source_embedding_artifact_sha256": video["artifact_sha256"],
                "label_correction_artifact_sha256": corrections[
                    "artifact_sha256"
                ],
                "examples": apply_ball_release_label_corrections(
                    video["examples"], corrections=corrections
                ),
            }
        )
        corrected = seal_video_embedding_artifact(payload)
        _write_json(output_path, corrected)
        corrected_videos.append(
            {
                "input": source_path.name,
                "output": output_path.name,
                "artifact_sha256": corrected["artifact_sha256"],
            }
        )

    _write_json(correction_output, corrections)
    _write_json(scene_output, corrected_scene)
    return {
        "corrections": corrections["summary"],
        "correction_artifact_sha256": corrections["artifact_sha256"],
        "scene_artifact_sha256": corrected_scene["artifact_sha256"],
        "video_artifacts": corrected_videos,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    result = apply_corrections(
        plan_path=args.plan,
        review_path=args.review,
        scene_input=args.scene_input,
        scene_output=args.scene_output,
        video_pairs=[
            (Path(source), Path(output)) for source, output in args.video_pair
        ],
        correction_output=args.correction_output,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
