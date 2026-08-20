#!/usr/bin/env python3
"""Subset training-only shot-validity embeddings to a sealed VLM plan."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import (  # noqa: E402
    verify_nested_fusion_shot_vlm_plan,
)
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    seal_scene_embedding_artifact,
    verify_scene_embedding_artifact,
)
from app.analysis.shot_validity_video_backbone import (  # noqa: E402
    seal_video_embedding_artifact,
    verify_video_embedding_artifact,
)

SUBSET_SELECTION_PROTOCOL = "sealed_independent_shot_vlm_plan_three_part_key_order_v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ExampleKey = tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scene-input", type=Path, required=True)
    parser.add_argument("--scene-output", type=Path, required=True)
    parser.add_argument(
        "--video-pair",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        action="append",
        required=True,
    )
    return parser.parse_args()


def subset_embedding_artifacts_by_plan(
    *,
    plan: Mapping[str, Any],
    scene: Mapping[str, Any],
    videos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return hash-sealed subsets whose rows exactly follow the frozen plan."""

    verified_plan = verify_nested_fusion_shot_vlm_plan(plan)
    plan_sha256 = _require_sha256(verified_plan.get("plan_sha256"), field="plan")
    manifest_sha256 = _require_sha256(
        verified_plan.get("training_manifest_sha256"),
        field="plan training manifest",
    )
    verified_scene = verify_scene_embedding_artifact(scene)
    verified_videos = [verify_video_embedding_artifact(video) for video in videos]
    if not verified_videos:
        raise ValueError("at least one video embedding artifact is required")
    if verified_scene.get("training_manifest_sha256") != manifest_sha256:
        raise ValueError("scene embedding training manifest does not match the plan")
    for video in verified_videos:
        if video.get("training_manifest_sha256") != manifest_sha256:
            raise ValueError("video embedding training manifest does not match the plan")

    plan_keys = [_example_key(row, artifact_name="plan") for row in verified_plan["examples"]]
    scene_index = _index_examples(verified_scene["examples"], artifact_name="scene embedding artifact")
    video_indexes = [
        _index_examples(
            video["examples"],
            artifact_name=f"video embedding artifact {video['backbone']}",
        )
        for video in verified_videos
    ]
    _require_plan_coverage(plan_keys, scene_index, artifact_name="scene embedding artifact")
    for video, index in zip(verified_videos, video_indexes, strict=True):
        _require_plan_coverage(
            plan_keys,
            index,
            artifact_name=f"video embedding artifact {video['backbone']}",
        )
    _require_consistent_labels(plan_keys, scene_index, video_indexes)

    subset_scene = _seal_scene_subset(
        source=verified_scene,
        plan_sha256=plan_sha256,
        examples=[scene_index[key] for key in plan_keys],
    )
    subset_videos = [
        _seal_video_subset(
            source=video,
            plan_sha256=plan_sha256,
            examples=[index[key] for key in plan_keys],
        )
        for video, index in zip(verified_videos, video_indexes, strict=True)
    ]
    return {"plan_sha256": plan_sha256, "scene": subset_scene, "videos": subset_videos}


def subset_embedding_files_by_plan(
    *,
    plan_path: Path,
    scene_input: Path,
    scene_output: Path,
    video_pairs: Sequence[tuple[Path, Path]],
) -> dict[str, Any]:
    """Validate all inputs, then write plan-ordered embedding subsets."""

    if not video_pairs:
        raise ValueError("at least one video input/output pair is required")
    resolved_plan_path = plan_path.resolve()
    resolved_scene_input = scene_input.resolve()
    resolved_scene_output = scene_output.resolve()
    resolved_video_pairs = [
        (input_path.resolve(), output_path.resolve())
        for input_path, output_path in video_pairs
    ]
    resolved_inputs = {
        resolved_plan_path,
        resolved_scene_input,
        *(input_path for input_path, _output_path in resolved_video_pairs),
    }
    resolved_outputs = [
        resolved_scene_output,
        *(output_path for _input_path, output_path in resolved_video_pairs),
    ]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("scene and video output paths must be unique")
    if resolved_inputs.intersection(resolved_outputs):
        raise ValueError("output paths must not alias plan, scene, or video input paths")
    result = subset_embedding_artifacts_by_plan(
        plan=_read_json(resolved_plan_path),
        scene=_read_json(resolved_scene_input),
        videos=[
            _read_json(input_path)
            for input_path, _output_path in resolved_video_pairs
        ],
    )
    _write_json(resolved_scene_output, result["scene"])
    for artifact, (_input_path, output_path) in zip(
        result["videos"],
        resolved_video_pairs,
        strict=True,
    ):
        _write_json(output_path, artifact)
    return {
        "plan_sha256": result["plan_sha256"],
        "scene_artifact_sha256": result["scene"]["artifact_sha256"],
        "video_artifacts": [
            {
                "output": str(output_path),
                "backbone": artifact["backbone"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
            for artifact, (_input_path, output_path) in zip(
                result["videos"],
                resolved_video_pairs,
                strict=True,
            )
        ],
    }


def _index_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    artifact_name: str,
) -> dict[_ExampleKey, dict[str, Any]]:
    index: dict[_ExampleKey, dict[str, Any]] = {}
    for row in examples:
        key = _example_key(row, artifact_name=artifact_name)
        if key in index:
            raise ValueError(f"{artifact_name} contains duplicate three-part key: {_format_key(key)}")
        index[key] = copy.deepcopy(dict(row))
    return index


def _example_key(row: Mapping[str, Any], *, artifact_name: str) -> _ExampleKey:
    values = (
        row.get("source_video_sha256"),
        row.get("candidate_bundle_sha256"),
        row.get("event_id"),
    )
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{artifact_name} three-part key fields must be strings")
    key = values
    if not all(key):
        raise ValueError(f"{artifact_name} contains an incomplete three-part key")
    if _SHA256_PATTERN.fullmatch(key[0]) is None or _SHA256_PATTERN.fullmatch(key[1]) is None:
        raise ValueError(
            f"{artifact_name} source and candidate key fields must be lowercase SHA-256 digests"
        )
    return key


def _require_plan_coverage(
    plan_keys: Sequence[_ExampleKey],
    index: Mapping[_ExampleKey, Mapping[str, Any]],
    *,
    artifact_name: str,
) -> None:
    for key in plan_keys:
        if key not in index:
            raise ValueError(f"{artifact_name} is missing plan key: {_format_key(key)}")


def _require_consistent_labels(
    plan_keys: Sequence[_ExampleKey],
    scene_index: Mapping[_ExampleKey, Mapping[str, Any]],
    video_indexes: Sequence[Mapping[_ExampleKey, Mapping[str, Any]]],
) -> None:
    for key in plan_keys:
        scene_label = scene_index[key]["event_present"]
        for video_index in video_indexes:
            if video_index[key]["event_present"] != scene_label:
                raise ValueError(f"scene/video labels disagree for plan key: {_format_key(key)}")


def _seal_scene_subset(
    *,
    source: Mapping[str, Any],
    plan_sha256: str,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _subset_payload(source=source, plan_sha256=plan_sha256, examples=examples)
    return verify_scene_embedding_artifact(seal_scene_embedding_artifact(payload))


def _seal_video_subset(
    *,
    source: Mapping[str, Any],
    plan_sha256: str,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _subset_payload(source=source, plan_sha256=plan_sha256, examples=examples)
    return verify_video_embedding_artifact(seal_video_embedding_artifact(payload))


def _subset_payload(
    *,
    source: Mapping[str, Any],
    plan_sha256: str,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {key: copy.deepcopy(value) for key, value in source.items() if key not in {"artifact_sha256", "examples"}}
    payload.update(
        {
            "source_embedding_artifact_sha256": source["artifact_sha256"],
            "plan_sha256": plan_sha256,
            "subset_selection_protocol": SUBSET_SELECTION_PROTOCOL,
            "examples": [copy.deepcopy(dict(row)) for row in examples],
        }
    )
    return payload


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _format_key(key: _ExampleKey) -> str:
    return f"({key[0]}, {key[1]}, {key[2]})"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    result = subset_embedding_files_by_plan(
        plan_path=args.plan,
        scene_input=args.scene_input,
        scene_output=args.scene_output,
        video_pairs=[(Path(input_path), Path(output_path)) for input_path, output_path in args.video_pair],
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
