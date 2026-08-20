#!/usr/bin/env python3
"""Extract resumable, training-only MViT features for BARD non-shot clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.shot_validity_scene_state import file_sha256  # noqa: E402
from app.analysis.shot_validity_video_backbone import (  # noqa: E402
    load_video_backbone,
)
from app.training.resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceThresholds,
    sample_training_resources,
)

PROBE_SCHEMA = "agu.bard-nonshot-mvit-embeddings.v1"
BACKBONE = "torchvision/mvit_v2_s/kinetics400_v1"
EMBEDDING_DIMENSION = 768
DEFAULT_CLIP_FRAMES = 16
RESOURCE_LIMIT_EXIT_CODE = 75


class ResourceLimitExceeded(RuntimeError):
    """Raised when the local resource guard requests a safe stop."""


def normalized_full_clip_frame_indexes(
    frame_count: int,
    *,
    clip_frames: int = DEFAULT_CLIP_FRAMES,
) -> tuple[int, ...]:
    """Return deterministic frame indexes in the interior 5–95% of a clip."""

    if frame_count < clip_frames or clip_frames < 2:
        raise ValueError("clip does not contain enough unique frames")
    indexes = tuple(
        int(round(float(value)))
        for value in np.linspace(
            0.05 * (frame_count - 1),
            0.95 * (frame_count - 1),
            clip_frames,
        )
    )
    if len(set(indexes)) != clip_frames:
        raise ValueError("normalized clip indexes are not unique")
    return indexes


def seal_probe_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a partial or complete MViT probe artifact."""

    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = PROBE_SCHEMA
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    _validate_probe_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_probe_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a hash-sealed MViT probe artifact."""

    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_probe_artifact(artifact)
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("probe artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def extract_probe(
    *,
    manifest_path: Path,
    backbone_checkpoint: Path,
    output_path: Path,
    device_name: str = "cpu",
    clip_frames: int = DEFAULT_CLIP_FRAMES,
    min_available_gib: float = 2.5,
    max_memory_percent: float = 90.0,
    max_cpu_percent: float = 95.0,
    resume: bool = False,
) -> dict[str, Any]:
    """Extract one clip at a time and checkpoint after every successful clip."""

    manifest = _verify_source_manifest(manifest_path)
    if clip_frames < 16:
        raise ValueError("MViT requires at least 16 clip frames")
    if not backbone_checkpoint.is_file():
        raise ValueError("MViT checkpoint is missing")
    backbone_sha256 = file_sha256(backbone_checkpoint)
    device = _resolve_device(device_name)
    model, transform = load_video_backbone(
        BACKBONE,
        backbone_checkpoint,
        device=device,
    )
    torch.set_num_threads(1)

    thresholds = TrainingResourceThresholds(
        max_system_memory_percent=max_memory_percent,
        min_available_memory_gib=min_available_gib,
        max_system_cpu_percent=max_cpu_percent,
        consecutive_breaches=2,
    )
    guard = TrainingResourceGuard(thresholds)

    examples_by_clip: dict[str, dict[str, Any]] = {}
    if resume and output_path.is_file():
        previous = verify_probe_artifact(
            json.loads(output_path.read_text(encoding="utf-8"))
        )
        if (
            previous["source_manifest_sha256"] != manifest["manifest_sha256"]
            or previous["backbone_sha256"] != backbone_sha256
            or int(previous["clip_frames"]) != clip_frames
        ):
            raise ValueError("resume artifact provenance does not match inputs")
        examples_by_clip = {
            str(row["clip_id"]): dict(row) for row in previous["examples"]
        }

    def observe(stage: str) -> None:
        decision = guard.observe(
            sample_training_resources(os.getpid()),
            stage=stage,
        )
        print(decision.to_json(), flush=True)
        if decision.should_stop:
            raise ResourceLimitExceeded(
                "; ".join(decision.reasons) or "resource guard requested stop"
            )

    def write_checkpoint(complete: bool) -> dict[str, Any]:
        artifact = seal_probe_artifact(
            {
                "purpose": "offline_nonshot_mvit_embedding_probe",
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": manifest["manifest_sha256"],
                "source_manifest_file_sha256": file_sha256(manifest_path),
                "backbone": BACKBONE,
                "backbone_sha256": backbone_sha256,
                "backbone_weights_url": (
                    "https://download.pytorch.org/models/mvit_v2_s-ae3be167.pth"
                ),
                "embedding_dimension": EMBEDDING_DIMENSION,
                "clip_frames": clip_frames,
                "sampling_protocol": "normalized_full_clip_5_to_95_percent_v1",
                "clip_count_expected": len(manifest["clips"]),
                "clip_count_completed": len(examples_by_clip),
                "complete": complete,
                "examples": [
                    examples_by_clip[str(row["clip_id"])]
                    for row in manifest["clips"]
                    if str(row["clip_id"]) in examples_by_clip
                ],
            }
        )
        _write_json_atomic(output_path, artifact)
        return artifact

    observe("before_extraction")
    for position, row in enumerate(manifest["clips"], start=1):
        clip_id = str(row["clip_id"])
        if clip_id in examples_by_clip:
            continue
        observe(f"before_clip_{position}")
        path = manifest_path.parent / str(row["relative_path"])
        if file_sha256(path) != str(row["sha256"]):
            raise ValueError(f"BARD clip hash mismatch: {path.name}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open BARD clip: {path.name}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        indexes = normalized_full_clip_frame_indexes(
            frame_count,
            clip_frames=clip_frames,
        )
        frames: list[np.ndarray] = []
        try:
            for frame_index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(
                        f"cannot decode BARD frame {frame_index}: {path.name}"
                    )
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        tensor = transform(clip).unsqueeze(0).to(device)
        with torch.inference_mode():
            embedding = model(tensor).detach().cpu().to(torch.float32).tolist()[0]
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(f"unexpected MViT embedding dimension: {path.name}")
        examples_by_clip[clip_id] = {
            "clip_id": clip_id,
            "source_video_sha256": str(row["sha256"]),
            "relative_path": str(row["relative_path"]),
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_indexes": list(indexes),
            "embedding": embedding,
        }
        write_checkpoint(complete=False)
        observe(f"after_clip_{position}")
        print(
            json.dumps(
                {
                    "stage": "bard_nonshot_mvit_embedding",
                    "clip": position,
                    "total": len(manifest["clips"]),
                    "clip_id": clip_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return write_checkpoint(complete=True)


def _verify_source_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "agu.bard-nonshot-validation-subset.v1"
        or manifest.get("runtime_consumable") is not False
        or manifest.get("codex_runtime_answer_used") is not False
        or manifest.get("promotion_eligible") is not False
    ):
        raise ValueError("invalid BARD non-shot source manifest")
    claimed = str(manifest.get("manifest_sha256") or "")
    if claimed != _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("BARD non-shot source manifest hash mismatch")
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("BARD non-shot source manifest has no clips")
    seen: set[str] = set()
    blind_ids = {str(value) for value in manifest.get("blind_game_ids_excluded", [])}
    for row in clips:
        if not isinstance(row, Mapping):
            raise ValueError("BARD source clip must be an object")
        clip_id = str(row.get("clip_id") or "")
        if not clip_id or clip_id in seen:
            raise ValueError("BARD source clip ids must be unique")
        seen.add(clip_id)
        source_sha = str(row.get("sha256") or "")
        if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
            raise ValueError("BARD source clip hash is invalid")
        relative = Path(str(row.get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("BARD source clip path is unsafe")
        target = path.parent / relative
        if not target.is_file():
            raise ValueError(f"BARD source clip is missing: {target}")
        if any(blind_id in clip_id for blind_id in blind_ids):
            raise ValueError("BARD source manifest contains a sealed blind game")
    return manifest


def _validate_probe_artifact(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != PROBE_SCHEMA
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("backbone") != BACKBONE
        or int(artifact.get("embedding_dimension", 0)) != EMBEDDING_DIMENSION
        or int(artifact.get("clip_frames", 0)) < 16
        or not isinstance(artifact.get("complete"), bool)
    ):
        raise ValueError("invalid BARD non-shot MViT probe artifact")
    for key in ("source_manifest_sha256", "backbone_sha256"):
        value = str(artifact.get(key) or "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("probe provenance hash is invalid")
    examples = artifact.get("examples")
    if not isinstance(examples, list):
        raise ValueError("probe artifact examples must be a list")
    if int(artifact.get("clip_count_completed", -1)) != len(examples):
        raise ValueError("probe completed count does not match examples")
    if int(artifact.get("clip_count_expected", 0)) < len(examples):
        raise ValueError("probe expected count is smaller than completed count")
    if artifact.get("complete") and int(artifact["clip_count_expected"]) != len(examples):
        raise ValueError("complete probe artifact is missing clips")
    seen: set[str] = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("probe example must be an object")
        clip_id = str(row.get("clip_id") or "")
        if not clip_id or clip_id in seen:
            raise ValueError("probe clip ids must be unique")
        seen.add(clip_id)
        indexes = row.get("frame_indexes")
        if not isinstance(indexes, list) or len(indexes) != int(artifact["clip_frames"]):
            raise ValueError("probe frame indexes are invalid")
        if any(not isinstance(value, int) or value < 0 for value in indexes):
            raise ValueError("probe frame indexes must be non-negative integers")
        embedding = row.get("embedding")
        if (
            not isinstance(embedding, list)
            or len(embedding) != EMBEDDING_DIMENSION
            or not np.isfinite(np.asarray(embedding, dtype=np.float64)).all()
        ):
            raise ValueError("probe embedding dimension or values are invalid")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return device


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--clip-frames", type=int, default=DEFAULT_CLIP_FRAMES)
    parser.add_argument("--min-available-gib", type=float, default=2.5)
    parser.add_argument("--max-memory-percent", type=float, default=90.0)
    parser.add_argument("--max-cpu-percent", type=float, default=95.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        artifact = extract_probe(
            manifest_path=args.manifest,
            backbone_checkpoint=args.backbone_checkpoint,
            output_path=args.output,
            device_name=args.device,
            clip_frames=args.clip_frames,
            min_available_gib=args.min_available_gib,
            max_memory_percent=args.max_memory_percent,
            max_cpu_percent=args.max_cpu_percent,
            resume=args.resume,
        )
    except ResourceLimitExceeded as exc:
        print(json.dumps({"event": "probe_resource_guard_stopped", "reason": str(exc)}), flush=True)
        return RESOURCE_LIMIT_EXIT_CODE
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "examples": len(artifact["examples"]),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
