#!/usr/bin/env python3
"""Generate raw-only scene-transition evidence for selected event candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.replay_transition import (  # noqa: E402
    REPLAY_TRANSITION_EVIDENCE_SCHEMA,
    replay_transition_payload,
    seal_replay_transition_artifact,
    summarize_transition_probabilities,
)
from app.analysis.schemas import (  # noqa: E402
    GameEventResponse,
    RawOnlyPredictionBundleResponse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--event-id", action="append")
    parser.add_argument(
        "--event-plan",
        type=Path,
        help="Label-hidden visual-state review plan used only to select bound event IDs.",
    )
    parser.add_argument("--source-module", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--transition-threshold", type=float, default=0.5)
    parser.add_argument("--near-seconds", type=float, default=8.0)
    parser.add_argument("--minimum-transition-count", type=int, default=5)
    parser.add_argument("--minimum-near-transition-count", type=int, default=3)
    return parser.parse_args()


def candidate_anchor(event: GameEventResponse) -> int:
    evidence = [
        item for item in event.evidence if item.kind == "ball_rim_proximity_cluster"
    ]
    if evidence:
        selected = max(
            evidence,
            key=lambda item: (
                int(item.details.get("hit_count") or 0),
                -int(item.details.get("review_anchor_frame") or item.start_frame),
            ),
        )
        return int(
            selected.details.get("review_anchor_frame")
            or selected.details.get("candidate_event_frame")
            or selected.start_frame
        )
    return int(event.release_frame or event.start_frame)


def event_ids_from_review_plan(
    plan: Mapping[str, Any],
    *,
    raw_video_sha256: str,
    candidate_bundle_sha256: str,
) -> set[str]:
    if (
        plan.get("runtime_consumable") is not False
        or plan.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("event plan must remain offline and outside runtime answers")
    sealed = {str(value) for value in plan.get("sealed_blind_video_sha256s", ())}
    if raw_video_sha256 in sealed:
        raise ValueError("event plan rejects sealed blind video")
    examples = plan.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("event plan requires examples")
    selected = {
        str(row.get("event_id") or "")
        for row in examples
        if isinstance(row, Mapping)
        and row.get("source_video_sha256") == raw_video_sha256
        and row.get("candidate_bundle_sha256") == candidate_bundle_sha256
    }
    if not selected or "" in selected:
        raise ValueError("event plan has no exact source-bound events")
    return selected


def load_transnet_model(
    *,
    source_module: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> torch.nn.Module:
    if _file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("TransNetV2 checkpoint hash mismatch")
    module = _load_module(source_module)
    model_type = getattr(module, "TransNetV2", None)
    if model_type is None:
        raise ValueError("TransNetV2 source module does not expose TransNetV2")
    model = model_type()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_transition_probabilities(
    model: torch.nn.Module,
    frames_rgb: np.ndarray,
) -> np.ndarray:
    if (
        frames_rgb.ndim != 4
        or frames_rgb.shape[1:] != (27, 48, 3)
        or frames_rgb.dtype != np.uint8
        or len(frames_rgb) == 0
    ):
        raise ValueError("TransNetV2 frames must be non-empty uint8 [T,27,48,3]")
    frames = torch.from_numpy(frames_rgb)
    remainder = len(frames) % 50
    end_padding = (50 - remainder if remainder else 50) + 25
    padded = torch.cat(
        [
            frames[:1].repeat(25, 1, 1, 1),
            frames,
            frames[-1:].repeat(end_padding, 1, 1, 1),
        ]
    )
    predictions: list[torch.Tensor] = []
    for start in range(0, len(padded) - 99, 50):
        with torch.no_grad():
            logits, _ = model(padded[start : start + 100].unsqueeze(0))
        predictions.append(torch.sigmoid(logits[0, 25:75, 0]).cpu())
    return torch.cat(predictions).numpy()[: len(frames)]


def main() -> int:
    args = parse_args()
    if (
        args.window_seconds <= 0
        or args.near_seconds <= 0
        or args.near_seconds > args.window_seconds
    ):
        raise ValueError("replay transition window configuration is invalid")
    source = verify_raw_only_bundle(
        RawOnlyPredictionBundleResponse.model_validate_json(
            args.candidate_bundle.read_text(encoding="utf-8")
        )
    )
    video_sha256 = _file_sha256(args.video)
    if (
        len(source.raw_videos) != 1
        or source.raw_videos[0].filename != args.video.name
        or source.raw_videos[0].sha256 != video_sha256
    ):
        raise ValueError("candidate bundle does not match the raw video")
    requested = set(args.event_id or ())
    if args.event_plan is not None:
        requested.update(
            event_ids_from_review_plan(
                json.loads(args.event_plan.read_text(encoding="utf-8")),
                raw_video_sha256=video_sha256,
                candidate_bundle_sha256=source.bundle_sha256,
            )
        )
    events = [
        event for event in source.events if not requested or event.event_id in requested
    ]
    if not events or requested - {event.event_id for event in events}:
        raise ValueError("requested replay transition events are missing")
    model = load_transnet_model(
        source_module=args.source_module,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("video metadata is incomplete")
    results = []
    try:
        for index, event in enumerate(events, start=1):
            anchor = candidate_anchor(event)
            radius = int(round(args.window_seconds * fps))
            start = max(0, anchor - radius)
            stop = min(frame_count, anchor + radius)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            while start + len(frames) < stop:
                ok, image = capture.read()
                if not ok or image is None:
                    break
                resized = cv2.resize(image, (48, 27), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            probabilities = predict_transition_probabilities(
                model,
                np.asarray(frames, dtype=np.uint8),
            )
            evidence = summarize_transition_probabilities(
                event_id=event.event_id,
                anchor_frame=anchor,
                window_start_frame=start,
                probabilities=probabilities,
                fps=fps,
                transition_threshold=args.transition_threshold,
                near_seconds=args.near_seconds,
                minimum_transition_count=args.minimum_transition_count,
                minimum_near_transition_count=args.minimum_near_transition_count,
            )
            results.append(replay_transition_payload(evidence))
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(events),
                        "event_id": event.event_id,
                        "broadcast_state": evidence.broadcast_state,
                    }
                ),
                flush=True,
            )
    finally:
        capture.release()
    artifact = seal_replay_transition_artifact(
        {
            "schema_version": REPLAY_TRANSITION_EVIDENCE_SCHEMA,
            "purpose": "offline_raw_only_replay_transition_screening",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": video_sha256,
            "candidate_bundle_sha256": source.bundle_sha256,
            "source": {
                "name": "TransNetV2",
                "license": "MIT",
                "source_revision": args.source_revision,
                "source_module_sha256": _file_sha256(args.source_module),
                "checkpoint_sha256": args.checkpoint_sha256,
                "checkpoint_loaded_with_weights_only": True,
            },
            "configuration": {
                "window_seconds": args.window_seconds,
                "transition_threshold": args.transition_threshold,
                "near_seconds": args.near_seconds,
                "minimum_transition_count": args.minimum_transition_count,
                "minimum_near_transition_count": args.minimum_near_transition_count,
                "device": "cpu",
            },
            "events": results,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_sha256": artifact["artifact_sha256"]}))
    return 0


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("agu_transnetv2_source", path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to load TransNetV2 source module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
