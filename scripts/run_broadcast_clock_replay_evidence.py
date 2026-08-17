#!/usr/bin/env python3
"""Generate offline replay evidence from a broadcast game-clock timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.broadcast_clock import (  # noqa: E402
    BROADCAST_CLOCK_EVIDENCE_SCHEMA,
    BroadcastClockSample,
    clock_evidence_payload,
    parse_broadcast_clock,
    seal_broadcast_clock_artifact,
    summarize_clock_disappearance,
    verify_broadcast_clock_artifact,
)
from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.schemas import (  # noqa: E402
    GameEventResponse,
    RawOnlyPredictionBundleResponse,
)
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    verify_scene_embedding_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--scene-embeddings", type=Path)
    parser.add_argument("--event-id", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--sample-offset-seconds",
        type=float,
        action="append",
        default=None,
    )
    parser.add_argument("--crop-top-ratio", type=float, default=0.58)
    parser.add_argument("--ocr-confidence", type=float, default=0.55)
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
    for item in event.evidence:
        for key in ("review_anchor_frame", "candidate_event_frame"):
            value = item.details.get(key)
            if value is not None:
                return int(value)
    return int(event.release_frame or event.start_frame)


def main() -> int:
    args = parse_args()
    offsets = tuple(args.sample_offset_seconds or (-6.0, -3.0, 0.0, 3.0, 6.0))
    if (
        len(set(offsets)) != len(offsets)
        or sum(value <= -3.0 for value in offsets) < 2
        or sum(value >= 0.0 for value in offsets) < 3
        or not 0.0 <= args.crop_top_ratio < 1.0
        or not 0.0 <= args.ocr_confidence <= 1.0
        or args.checkpoint_every < 1
    ):
        raise ValueError("broadcast clock sampling configuration is invalid")
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
    if args.scene_embeddings is not None:
        scene_requested = select_scene_event_ids(
            json.loads(args.scene_embeddings.read_text(encoding="utf-8")),
            raw_video_sha256=video_sha256,
            candidate_bundle_sha256=str(source.bundle_sha256),
        )
        if requested and not requested.issubset(scene_requested):
            raise ValueError("requested events are outside the scene example set")
        requested = requested or scene_requested
    events = [
        event for event in source.events if not requested or event.event_id in requested
    ]
    if not events or requested - {event.event_id for event in events}:
        raise ValueError("requested broadcast clock events are missing")
    configuration = {
        "sample_offset_seconds": list(offsets),
        "crop_top_ratio": args.crop_top_ratio,
        "ocr_confidence": args.ocr_confidence,
        "maximum_pre_anchor_offset_seconds": -3.0,
        "minimum_pre_anchor_read_count": 2,
        "minimum_post_anchor_missing_count": 3,
    }
    event_ids = {event.event_id for event in events}
    completed = (
        load_resume_events(
            args.output,
            raw_video_sha256=video_sha256,
            candidate_bundle_sha256=str(source.bundle_sha256),
            configuration=configuration,
            allowed_event_ids=event_ids,
        )
        if args.resume and args.output.exists()
        else {}
    )
    pending_events = [event for event in events if event.event_id not in completed]
    if completed:
        print(
            json.dumps(
                {
                    "resumed": len(completed),
                    "remaining": len(pending_events),
                    "total": len(events),
                }
            ),
            flush=True,
        )

    ocr = None
    if pending_events:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("video metadata is incomplete")
    results = dict(completed)
    newly_processed = 0
    try:
        for event in pending_events:
            anchor = candidate_anchor(event)
            samples = []
            for offset in offsets:
                frame = min(
                    frame_count - 1,
                    max(0, round(anchor + offset * fps)),
                )
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                read = None
                if ok and image is not None:
                    crop_top = round(image.shape[0] * args.crop_top_ratio)
                    crop = image[crop_top:]
                    if ocr is None:
                        raise RuntimeError("OCR engine was not initialized")
                    raw, _ = ocr(crop)
                    read = parse_broadcast_clock(
                        raw or (),
                        frame=frame,
                        image_height=crop.shape[0],
                        minimum_confidence=args.ocr_confidence,
                    )
                samples.append(
                    BroadcastClockSample(
                        frame=frame,
                        offset_seconds=offset,
                        read=read,
                    )
                )
            evidence = summarize_clock_disappearance(
                event_id=event.event_id,
                anchor_frame=anchor,
                samples=samples,
            )
            results[event.event_id] = clock_evidence_payload(evidence)
            newly_processed += 1
            print(
                json.dumps(
                    {
                        "processed": len(results),
                        "total": len(events),
                        "event_id": event.event_id,
                        "broadcast_state": evidence.broadcast_state,
                    }
                ),
                flush=True,
            )
            if newly_processed % args.checkpoint_every == 0:
                _write_artifact(
                    args.output,
                    _build_artifact(
                        video_sha256=video_sha256,
                        candidate_bundle_sha256=str(source.bundle_sha256),
                        configuration=configuration,
                        events=[
                            results[event.event_id]
                            for event in events
                            if event.event_id in results
                        ],
                    ),
                )
    finally:
        capture.release()
    artifact = _build_artifact(
        video_sha256=video_sha256,
        candidate_bundle_sha256=str(source.bundle_sha256),
        configuration=configuration,
        events=[
            results[event.event_id]
            for event in events
            if event.event_id in results
        ],
    )
    _write_artifact(args.output, artifact)
    print(json.dumps({"artifact_sha256": artifact["artifact_sha256"]}))
    return 0


def select_scene_event_ids(
    scene_artifact: dict[str, object],
    *,
    raw_video_sha256: str,
    candidate_bundle_sha256: str,
) -> set[str]:
    scene = verify_scene_embedding_artifact(scene_artifact)
    selected = {
        str(row["event_id"])
        for row in scene["examples"]
        if str(row["source_video_sha256"]) == raw_video_sha256
        and str(row["candidate_bundle_sha256"]) == candidate_bundle_sha256
    }
    if not selected:
        raise ValueError("scene examples do not cover this video and candidate bundle")
    return selected


def load_resume_events(
    output: Path,
    *,
    raw_video_sha256: str,
    candidate_bundle_sha256: str,
    configuration: dict[str, object],
    allowed_event_ids: set[str],
) -> dict[str, dict[str, object]]:
    artifact = verify_broadcast_clock_artifact(
        json.loads(output.read_text(encoding="utf-8"))
    )
    if (
        str(artifact.get("raw_video_sha256") or "") != raw_video_sha256
        or str(artifact.get("candidate_bundle_sha256") or "")
        != candidate_bundle_sha256
    ):
        raise ValueError("resume artifact source binding does not match")
    if artifact.get("configuration") != configuration:
        raise ValueError("resume artifact configuration does not match")
    events = {
        str(event["event_id"]): dict(event) for event in artifact["events"]
    }
    if not set(events).issubset(allowed_event_ids):
        raise ValueError("resume artifact contains events outside the requested set")
    return events


def _build_artifact(
    *,
    video_sha256: str,
    candidate_bundle_sha256: str,
    configuration: dict[str, object],
    events: list[dict[str, object]],
) -> dict[str, object]:
    if not events:
        raise ValueError("broadcast clock artifact requires completed events")
    return seal_broadcast_clock_artifact(
        {
            "schema_version": BROADCAST_CLOCK_EVIDENCE_SCHEMA,
            "purpose": "offline_raw_only_broadcast_clock_replay_screening",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": video_sha256,
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "source": {
                "ocr_engine": "RapidOCR",
                "parser": "period_clock_same_row_v1",
            },
            "configuration": configuration,
            "events": events,
        }
    )


def _write_artifact(output: Path, artifact: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
