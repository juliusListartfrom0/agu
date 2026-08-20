#!/usr/bin/env python3
"""Apply an existing sealed label-correction chain to re-extracted embeddings."""

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
    verify_ball_release_label_corrections,
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
    parser.add_argument("--scene-input", type=Path, required=True)
    parser.add_argument("--scene-output", type=Path, required=True)
    parser.add_argument("--correction", type=Path, action="append", required=True)
    parser.add_argument(
        "--video-pair",
        nargs=2,
        action="append",
        metavar=("INPUT", "OUTPUT"),
        default=[],
    )
    return parser.parse_args()


def rebind_correction_chain(
    *,
    scene_input: Path,
    scene_output: Path,
    correction_paths: list[Path],
    video_pairs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    scene = verify_scene_embedding_artifact(_read_json(scene_input))
    corrections = [
        verify_ball_release_label_corrections(_read_json(path))
        for path in correction_paths
    ]
    correction_hashes = [str(row["artifact_sha256"]) for row in corrections]
    scene_examples = _apply_chain(scene["examples"], corrections)
    scene_payload = {
        key: value
        for key, value in scene.items()
        if key not in {"artifact_sha256", "examples"}
    }
    scene_payload.update(
        {
            "purpose": "atomic_scene_screening_with_sealed_label_corrections",
            "source_embedding_artifact_sha256": scene["artifact_sha256"],
            "label_correction_artifact_sha256_chain": correction_hashes,
            "examples": scene_examples,
        }
    )
    corrected_scene = seal_scene_embedding_artifact(scene_payload)
    _write_json(scene_output, corrected_scene)

    outputs = []
    for source_path, output_path in video_pairs:
        video = verify_video_embedding_artifact(_read_json(source_path))
        if video["training_manifest_sha256"] != scene["training_manifest_sha256"]:
            raise ValueError("video embeddings do not match the scene manifest")
        video_payload = {
            key: value
            for key, value in video.items()
            if key not in {"artifact_sha256", "examples"}
        }
        video_payload.update(
            {
                "purpose": "atomic_video_screening_with_sealed_label_corrections",
                "source_embedding_artifact_sha256": video["artifact_sha256"],
                "label_correction_artifact_sha256_chain": correction_hashes,
                "examples": _apply_chain(video["examples"], corrections),
            }
        )
        corrected_video = seal_video_embedding_artifact(video_payload)
        _write_json(output_path, corrected_video)
        outputs.append(
            {
                "output": output_path.name,
                "artifact_sha256": corrected_video["artifact_sha256"],
            }
        )
    return {
        "scene_artifact_sha256": corrected_scene["artifact_sha256"],
        "correction_artifact_sha256_chain": correction_hashes,
        "video_artifacts": outputs,
    }


def _apply_chain(
    examples: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = list(examples)
    for correction in corrections:
        output = apply_ball_release_label_corrections(
            output,
            corrections=correction,
        )
    return output


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
    result = rebind_correction_chain(
        scene_input=args.scene_input,
        scene_output=args.scene_output,
        correction_paths=args.correction,
        video_pairs=[
            (Path(source), Path(output)) for source, output in args.video_pair
        ],
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
