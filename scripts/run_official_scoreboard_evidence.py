#!/usr/bin/env python3
"""Attach conservative broadcast-score deltas to raw-only shot candidates."""

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
    CandidateScoreDelta,
    attach_score_delta,
    candidate_ids_for_delta,
    extract_scoreboard_deltas,
    score_delta_event,
)
from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.official_inference import seal_agu_autonomous_predictions  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--team-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--crop-top-ratio", type=float, default=0.58)
    parser.add_argument(
        "--localization-lookback-sec",
        type=float,
        default=12.0,
        help=(
            "Bound generated score-only candidate windows to this interval before "
            "the first stable post-score read; full scoreboard delta bounds remain in evidence"
        ),
    )
    return parser.parse_args()


def run_scoreboard_evidence(
    *,
    candidate_bundle: RawOnlyPredictionBundleResponse,
    video_path: Path,
    team_ids: list[str],
    sample_interval_sec: float,
    minimum_support: int,
    confidence: float,
    crop_top_ratio: float,
    localization_lookback_sec: float,
    cache_path: Path,
    evidence_output: Path,
) -> RawOnlyPredictionBundleResponse:
    source = verify_raw_only_bundle(candidate_bundle)
    if len(source.raw_videos) != 1 or source.raw_videos[0].filename != video_path.name:
        raise ValueError("candidate bundle must declare exactly the provided raw video")
    if len(team_ids) != 2 or len(set(team_ids)) != 2:
        raise ValueError("exactly two distinct --team-id values are required")
    if sample_interval_sec <= 0 or minimum_support <= 0 or localization_lookback_sec <= 0:
        raise ValueError(
            "sample interval, minimum support and localization lookback must be positive"
        )
    video_sha256 = _sha256_file(video_path)
    if video_sha256 != source.raw_videos[0].sha256:
        raise ValueError("raw video hash does not match the candidate bundle")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("video metadata is incomplete")
    stride = max(1, int(round(fps * sample_interval_sec)))
    reader = BroadcastScoreboardReader(
        team_ids,
        confidence_threshold=confidence,
        crop_top_ratio=crop_top_ratio,
    )
    cache_header = {
        "schema_version": "agu.broadcast-scoreboard-cache.v1",
        "reader_method": reader.method,
        "raw_video_sha256": video_sha256,
        "team_ids": list(team_ids),
        "stride_frames": stride,
        "confidence": confidence,
        "crop_top_ratio": crop_top_ratio,
    }
    cache = _load_cache(cache_path, expected_header=cache_header)
    frames = list(range(0, frame_count, stride))
    try:
        for index, frame_number in enumerate(frames, start=1):
            key = str(frame_number)
            if key not in cache["frames"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                read = reader.read(frame, frame=frame_number) if ok and frame is not None else None
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
                if index % 25 == 0:
                    _write_json_atomic(cache_path, cache)
            if index % 100 == 0 or index == len(frames):
                readable = sum(value is not None for value in cache["frames"].values())
                print(json.dumps({"sampled": index, "total": len(frames), "readable": readable}), flush=True)
    finally:
        capture.release()
        _write_json_atomic(cache_path, cache)

    reads = [
        BroadcastScoreboardRead(
            frame=int(value["frame"]),
            scores={str(team): int(score) for team, score in value["scores"].items()},
            confidence=float(value["confidence"]),
            source=str(value.get("source") or reader.method),
        )
        for value in cache["frames"].values()
        if isinstance(value, dict)
    ]
    deltas = extract_scoreboard_deltas(reads, minimum_support=minimum_support)
    events_by_id = {event.event_id: event for event in source.events}
    links: list[dict[str, object]] = []
    for delta in deltas:
        candidate_ids = candidate_ids_for_delta(source.events, delta)
        link = {
            **delta.__dict__,
            "before_scores": dict(delta.before_scores),
            "after_scores": dict(delta.after_scores),
            "candidate_event_ids": candidate_ids,
            "status": "linked" if len(candidate_ids) == 1 else "ambiguous" if candidate_ids else "unlinked",
        }
        links.append(link)
        if len(candidate_ids) != 1:
            generated = score_delta_event(
                delta,
                source_video_id=source.raw_videos[0].video_id,
                candidate_event_ids=candidate_ids,
                localization_lookback_frames=max(
                    1, int(round(fps * localization_lookback_sec))
                ),
            )
            events_by_id[generated.event_id] = generated
            continue
        event_id = candidate_ids[0]
        events_by_id[event_id] = attach_score_delta(
            events_by_id[event_id],
            CandidateScoreDelta(event_id=event_id, **delta.__dict__),
        )
    evidence_payload = {
        "schema_version": "agu.official-scoreboard-evidence.v1",
        **cache_header,
        "source_candidate_bundle_sha256": source.bundle_sha256,
        "read_count": len(reads),
        "reads": [
            {
                "frame": read.frame,
                "scores": dict(read.scores),
                "confidence": read.confidence,
                "source": read.source,
            }
            for read in sorted(reads, key=lambda item: item.frame)
        ],
        "deltas": links,
    }
    evidence_payload["artifact_sha256"] = _canonical_sha256(evidence_payload)
    _write_json_atomic(evidence_output, evidence_payload)
    return seal_agu_autonomous_predictions(
        game_id=source.game_id,
        raw_video_paths=[video_path],
        events=events_by_id.values(),
        config={
            "pipeline": "agu_official_scoreboard_evidence_v1",
            "source_candidate_config_sha256": source.config_sha256,
            "sample_interval_sec": sample_interval_sec,
            "minimum_support": minimum_support,
            "localization_lookback_sec": localization_lookback_sec,
            "scoreboard_evidence_sha256": evidence_payload["artifact_sha256"],
        },
        candidate_backend=source.model_provenance.get("candidate_backend", "agu_traditional_perception"),
        semantic_backend=reader.method,
        semantic_model="rapidocr_onnxruntime",
        model_provenance={
            "source_candidate_bundle_sha256": source.bundle_sha256,
            "scoreboard_evidence_sha256": evidence_payload["artifact_sha256"],
        },
    )


def _load_cache(path: Path, *, expected_header: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return {**expected_header, "frames": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected_header.items():
        if payload.get(key) != value:
            raise ValueError(f"scoreboard cache mismatch: {key}")
    if not isinstance(payload.get("frames"), dict):
        raise ValueError("scoreboard cache frames must be an object")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    args = parse_args()
    source = RawOnlyPredictionBundleResponse.model_validate_json(args.candidate_bundle.read_text(encoding="utf-8"))
    cache_path = args.cache or args.output.with_suffix(".scoreboard-cache.json")
    evidence_output = args.evidence_output or args.output.with_suffix(".scoreboard-evidence.json")
    result = run_scoreboard_evidence(
        candidate_bundle=source,
        video_path=args.video,
        team_ids=args.team_id,
        sample_interval_sec=args.sample_interval_sec,
        minimum_support=args.minimum_support,
        confidence=args.confidence,
        crop_top_ratio=args.crop_top_ratio,
        localization_lookback_sec=args.localization_lookback_sec,
        cache_path=cache_path,
        evidence_output=evidence_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bundle_sha256": result.bundle_sha256, "event_count": len(result.events)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
