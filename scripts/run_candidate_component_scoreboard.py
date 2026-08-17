#!/usr/bin/env python3
"""Resolve selected shot states from candidate-local scoreboard reads."""

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

from app.analysis.broadcast_scoreboard import (  # noqa: E402
    BroadcastScoreboardRead,
    BroadcastScoreboardReader,
    attach_score_delta,
    candidate_component_score_delta,
)
from app.analysis.official_evaluation import (  # noqa: E402
    seal_raw_only_predictions,
    verify_raw_only_bundle,
)
from app.analysis.schemas import (  # noqa: E402
    GameEventResponse,
    RawOnlyPredictionBundleResponse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--team-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sample-interval-sec", type=float, default=2.0)
    parser.add_argument("--before-start-sec", type=float, default=20.0)
    parser.add_argument("--before-end-sec", type=float, default=4.0)
    parser.add_argument("--after-end-sec", type=float, default=20.0)
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--crop-top-ratio", type=float, default=0.58)
    return parser.parse_args()


def candidate_anchor(event: GameEventResponse) -> int:
    """Return the strongest raw ball/rim review anchor without label access."""

    evidence = [
        item
        for item in event.evidence
        if item.kind == "ball_rim_proximity_cluster"
    ]
    if not evidence:
        return event.release_frame or event.start_frame
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


def candidate_sample_frames(
    *,
    anchor: int,
    fps: float,
    frame_count: int,
    sample_interval_sec: float,
    before_start_sec: float,
    before_end_sec: float,
    after_end_sec: float,
) -> tuple[list[int], list[int]]:
    """Build disjoint baseline/result samples around one frozen candidate."""

    stride = max(1, int(round(fps * sample_interval_sec)))
    before_start = max(0, anchor - int(round(fps * before_start_sec)))
    before_stop = max(before_start, anchor - int(round(fps * before_end_sec)))
    after_stop = min(frame_count, anchor + int(round(fps * after_end_sec)) + 1)
    return (
        list(range(before_start, before_stop + 1, stride)),
        list(range(anchor, after_stop, stride)),
    )


def main() -> int:
    args = parse_args()
    if (
        len(args.team_id) != 2
        or len(set(args.team_id)) != 2
        or min(
            args.sample_interval_sec,
            args.before_start_sec,
            args.before_end_sec,
            args.after_end_sec,
            float(args.minimum_support),
        )
        <= 0
        or args.before_start_sec <= args.before_end_sec
    ):
        raise ValueError("candidate scoreboard configuration is invalid")
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

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("video metadata is incomplete")
    reader = BroadcastScoreboardReader(
        args.team_id,
        confidence_threshold=args.confidence,
        crop_top_ratio=args.crop_top_ratio,
    )
    header = {
        "schema_version": "agu.candidate-scoreboard-cache.v1",
        "reader_method": reader.method,
        "raw_video_sha256": video_sha256,
        "team_ids": list(args.team_id),
        "sample_interval_sec": args.sample_interval_sec,
        "confidence": args.confidence,
        "crop_top_ratio": args.crop_top_ratio,
    }
    cache = _load_cache(args.cache, expected_header=header)
    windows: dict[str, tuple[list[int], list[int]]] = {}
    requested_frames: set[int] = set()
    for event in source.events:
        before, after = candidate_sample_frames(
            anchor=candidate_anchor(event),
            fps=fps,
            frame_count=frame_count,
            sample_interval_sec=args.sample_interval_sec,
            before_start_sec=args.before_start_sec,
            before_end_sec=args.before_end_sec,
            after_end_sec=args.after_end_sec,
        )
        windows[event.event_id] = before, after
        requested_frames.update(before)
        requested_frames.update(after)
    try:
        for index, frame_number in enumerate(sorted(requested_frames), start=1):
            key = str(frame_number)
            if key in cache["frames"]:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            read = (
                reader.read(frame, frame=frame_number)
                if ok and frame is not None
                else None
            )
            cache["frames"][key] = (
                {
                    "frame": read.frame,
                    "scores": dict(read.scores),
                    "confidence": read.confidence,
                    "source": read.source,
                }
                if read is not None
                else None
            )
            if index % 10 == 0:
                _write_json_atomic(args.cache, cache)
                print(
                    json.dumps(
                        {
                            "sampled": index,
                            "total": len(requested_frames),
                            "readable": sum(
                                value is not None
                                for value in cache["frames"].values()
                            ),
                        }
                    ),
                    flush=True,
                )
    finally:
        capture.release()
        _write_json_atomic(args.cache, cache)

    reads_by_frame = {
        int(value["frame"]): BroadcastScoreboardRead(
            frame=int(value["frame"]),
            scores={
                str(team): int(score)
                for team, score in value["scores"].items()
            },
            confidence=float(value["confidence"]),
            source=str(value.get("source") or reader.method),
        )
        for value in cache["frames"].values()
        if isinstance(value, dict)
    }
    events: list[GameEventResponse] = []
    diagnostics: list[dict[str, object]] = []
    for event in source.events:
        before_frames, after_frames = windows[event.event_id]
        before_reads = [
            reads_by_frame[frame]
            for frame in before_frames
            if frame in reads_by_frame
        ]
        after_reads = [
            reads_by_frame[frame]
            for frame in after_frames
            if frame in reads_by_frame
        ]
        neutral = GameEventResponse.model_validate(
            event.model_copy(
                update={
                    "outcome": "unknown",
                    "outcome_frame": None,
                    "shot_value": None,
                    "team_id": None,
                }
            ).model_dump()
        )
        delta = candidate_component_score_delta(
            event_id=event.event_id,
            before_reads=before_reads,
            after_reads=after_reads,
            minimum_support=args.minimum_support,
        )
        resolved = attach_score_delta(neutral, delta) if delta is not None else neutral
        events.append(resolved)
        diagnostics.append(
            {
                "event_id": event.event_id,
                "anchor_frame": candidate_anchor(event),
                "before_frames": before_frames,
                "after_frames": after_frames,
                "before_reads": [_read_payload(item) for item in before_reads],
                "after_reads": [_read_payload(item) for item in after_reads],
                "score_delta": (
                    {
                        **delta.__dict__,
                        "before_scores": dict(delta.before_scores),
                        "after_scores": dict(delta.after_scores),
                        "unresolved_team_ids": list(delta.unresolved_team_ids),
                    }
                    if delta is not None
                    else None
                ),
                "prediction": {
                    "event_type": resolved.event_type,
                    "outcome": resolved.outcome,
                    "shot_value": resolved.shot_value,
                    "team_id": resolved.team_id,
                },
            }
        )
    evidence = {
        "schema_version": "agu.candidate-component-scoreboard-evidence.v1",
        **header,
        "reference_loaded_by_script": False,
        "source_candidate_bundle_sha256": source.bundle_sha256,
        "before_window_seconds": [
            -args.before_start_sec,
            -args.before_end_sec,
        ],
        "after_window_seconds": [0.0, args.after_end_sec],
        "minimum_support": args.minimum_support,
        "events": diagnostics,
    }
    evidence["artifact_sha256"] = _canonical_sha256(evidence)
    _write_json_atomic(args.evidence_output, evidence)
    result = seal_raw_only_predictions(
        game_id=source.game_id,
        raw_video_paths=[args.video],
        events=events,
        config={
            "pipeline": "candidate_component_scoreboard_v1",
            "source_candidate_bundle_sha256": source.bundle_sha256,
            "scoreboard_evidence_sha256": evidence["artifact_sha256"],
            "score_resolution_method": "independent_team_consensus_v2",
        },
        model_provenance={
            "producer": "agu",
            "artifact_role": "blind_candidate_state_only",
            "candidate_backend": source.model_provenance.get(
                "candidate_backend", ""
            ),
            "semantic_backend": reader.method,
            "score_resolution_method": "independent_team_consensus_v2",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "bundle_sha256": result.bundle_sha256,
                "evidence_sha256": evidence["artifact_sha256"],
                "event_count": len(events),
                "resolved_count": sum(
                    event.outcome == "made" for event in events
                ),
            },
            indent=2,
        )
    )
    return 0


def _read_payload(read: BroadcastScoreboardRead) -> dict[str, object]:
    return {
        "frame": read.frame,
        "scores": dict(read.scores),
        "confidence": read.confidence,
        "source": read.source,
    }


def _load_cache(
    path: Path,
    *,
    expected_header: dict[str, object],
) -> dict[str, object]:
    if not path.exists():
        return {**expected_header, "frames": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected_header.items():
        if payload.get(key) != value:
            raise ValueError(f"candidate scoreboard cache mismatch: {key}")
    if not isinstance(payload.get("frames"), dict):
        raise ValueError("candidate scoreboard cache frames must be an object")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
