#!/usr/bin/env python3
"""Attach traditional ball/rim/pose evidence to scoreboard-localized shots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import (  # noqa: E402
    seal_raw_only_predictions,
    verify_raw_only_bundle,
)
from app.analysis.perception import attach_pose_keypoints  # noqa: E402
from app.analysis.schemas import (  # noqa: E402
    PerceptionDetectionResponse,
    RawOnlyPredictionBundleResponse,
)
from app.analysis.scoreboard_causal import (  # noqa: E402
    ScoreboardCausalConfig,
    attach_scoreboard_causal_evidence,
)
from scripts.run_official_perception import _file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--perception", type=Path, action="append", required=True)
    parser.add_argument("--pose", type=Path, action="append", default=[])
    parser.add_argument(
        "--pose-source-bundle",
        type=Path,
        action="append",
        default=[],
        help="Equivalent bundle named by each pose artifact when it differs from --candidate-bundle",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-ball-confidence", type=float, default=0.03)
    parser.add_argument("--min-rim-confidence", type=float, default=0.10)
    parser.add_argument("--max-rim-frame-gap", type=int, default=6)
    parser.add_argument("--max-normalized-distance", type=float, default=4.0)
    parser.add_argument("--max-ball-rim-iou", type=float, default=0.80)
    parser.add_argument("--cluster-gap-frames", type=int, default=15)
    parser.add_argument("--maximum-cluster-span-frames", type=int, default=45)
    parser.add_argument("--shooter-lookback-frames", type=int, default=90)
    parser.add_argument("--approach-lookback-frames", type=int, default=90)
    parser.add_argument("--minimum-approach-points", type=int, default=2)
    parser.add_argument("--minimum-approach-rise-px", type=float, default=8.0)
    parser.add_argument(
        "--maximum-ball-speed-px-per-frame", type=float, default=80.0
    )
    parser.add_argument("--maximum-player-candidates", type=int, default=16)
    return parser.parse_args()


def enrich_bundle(
    *,
    candidate_bundle_path: Path,
    video_path: Path,
    perception_paths: list[Path],
    pose_paths: list[Path],
    pose_source_bundle_paths: list[Path] | None,
    config: ScoreboardCausalConfig,
) -> tuple[object, dict[str, int]]:
    source = verify_raw_only_bundle(
        json.loads(candidate_bundle_path.read_text(encoding="utf-8"))
    )
    video_sha256 = _file_sha256(video_path)
    matching = [
        item
        for item in source.raw_videos
        if item.filename == video_path.name and item.sha256 == video_sha256
    ]
    if len(matching) != 1:
        raise ValueError("candidate bundle must contain exactly one matching raw video")
    raw_video_id = matching[0].video_id
    detections: list[PerceptionDetectionResponse] = []
    artifact_hashes: list[str] = []
    source_fps: float | None = None
    for artifact_index, path in enumerate(perception_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("raw_video") or {}
        if (
            payload.get("schema_version") != "agu.official-perception.v1"
            or raw.get("filename") != video_path.name
            or raw.get("sha256") != video_sha256
        ):
            raise ValueError(f"perception artifact does not match raw video: {path.name}")
        current_fps = float(raw.get("source_fps") or 0)
        if current_fps <= 0 or (
            source_fps is not None and abs(current_fps - source_fps) > 1e-6
        ):
            raise ValueError("perception artifacts must share a positive source_fps")
        source_fps = current_fps
        artifact_hashes.append(_file_sha256(path))
        for item in payload.get("detections") or []:
            detection = PerceptionDetectionResponse.model_validate(item)
            detections.append(
                detection.model_copy(
                    update={
                        "detection_id": (
                            f"perception-{artifact_index}:{detection.detection_id}"
                        ),
                        "track_id": (
                            f"perception-{artifact_index}:{detection.track_id}"
                            if detection.track_id
                            else None
                        ),
                        "player_id": (
                            f"perception-{artifact_index}:{detection.player_id}"
                            if detection.player_id
                            else None
                        ),
                    }
                )
            )
    pose_detections: list[PerceptionDetectionResponse] = []
    pose_hashes: list[str] = []
    pose_maximum_frame_gap = 0
    pose_source_paths = list(pose_source_bundle_paths or [])
    if pose_source_paths and len(pose_source_paths) != len(pose_paths):
        raise ValueError("pose source bundles must be paired with pose artifacts")
    for pose_index, path in enumerate(pose_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("raw_video") or {}
        if (
            payload.get("schema_version") != "agu.official-pose.v1"
            or raw.get("filename") != video_path.name
            or raw.get("sha256") != video_sha256
        ):
            raise ValueError(f"pose artifact does not match raw video: {path.name}")
        claimed_bundle_sha256 = str(
            (payload.get("candidate_source") or {}).get("bundle_sha256") or ""
        )
        if claimed_bundle_sha256 != source.bundle_sha256:
            if not pose_source_paths:
                raise ValueError(
                    f"pose artifact candidate bundle mismatch: {path.name}"
                )
            equivalent = verify_raw_only_bundle(
                json.loads(
                    pose_source_paths[pose_index].read_text(encoding="utf-8")
                )
            )
            if equivalent.bundle_sha256 != claimed_bundle_sha256:
                raise ValueError("pose source bundle hash does not match pose artifact")
            _verify_equivalent_event_geometry(source, equivalent)
        pose_hashes.append(_file_sha256(path))
        pose_detections.extend(
            PerceptionDetectionResponse.model_validate(item)
            for item in payload.get("detections") or []
        )
        stride = int((payload.get("sampling") or {}).get("stride_frames") or 0)
        pose_maximum_frame_gap = max(pose_maximum_frame_gap, stride // 2)
    if pose_detections:
        detections = attach_pose_keypoints(
            detections,
            pose_detections,
            maximum_frame_gap=pose_maximum_frame_gap,
            minimum_iou=0.25,
        )
    events, stats = attach_scoreboard_causal_evidence(
        source.events,
        detections,
        config=config,
    )
    bundle = seal_raw_only_predictions(
        game_id=source.game_id,
        raw_video_paths=[video_path],
        events=events,
        config={
            "source_candidate_config_sha256": source.config_sha256,
            "scoreboard_causal_enrichment": {
                **config.__dict__,
                "source_bundle_sha256": source.bundle_sha256,
                "perception_artifact_sha256": artifact_hashes,
                "pose_artifact_sha256": pose_hashes,
                **stats,
            },
        },
        model_provenance={
            **source.model_provenance,
            "scoreboard_causal_backend": "agu_ball_rim_pose_v1",
            "scoreboard_causal_source_bundle_sha256": source.bundle_sha256,
        },
    )
    if bundle.raw_videos[0].video_id != raw_video_id:
        raise ValueError("enriched bundle changed source video identity")
    return bundle, stats


def _verify_equivalent_event_geometry(
    source: RawOnlyPredictionBundleResponse,
    equivalent: RawOnlyPredictionBundleResponse,
) -> None:
    source_events = {
        event.event_id: (event.event_type, event.start_frame, event.end_frame)
        for event in source.events
    }
    equivalent_events = {
        event.event_id: (event.event_type, event.start_frame, event.end_frame)
        for event in equivalent.events
    }
    if (
        source.raw_videos != equivalent.raw_videos
        or source_events != equivalent_events
    ):
        raise ValueError("pose source bundle changed raw video or event geometry")


def main() -> int:
    args = parse_args()
    config = ScoreboardCausalConfig(
        min_ball_confidence=args.min_ball_confidence,
        min_rim_confidence=args.min_rim_confidence,
        max_rim_frame_gap=args.max_rim_frame_gap,
        max_normalized_distance=args.max_normalized_distance,
        max_ball_rim_iou=args.max_ball_rim_iou,
        cluster_gap_frames=args.cluster_gap_frames,
        maximum_cluster_span_frames=args.maximum_cluster_span_frames,
        shooter_lookback_frames=args.shooter_lookback_frames,
        approach_lookback_frames=args.approach_lookback_frames,
        minimum_approach_points=args.minimum_approach_points,
        minimum_approach_rise_px=args.minimum_approach_rise_px,
        maximum_ball_speed_px_per_frame=args.maximum_ball_speed_px_per_frame,
        maximum_player_candidates=args.maximum_player_candidates,
    )
    bundle, stats = enrich_bundle(
        candidate_bundle_path=args.candidate_bundle,
        video_path=args.video,
        perception_paths=args.perception,
        pose_paths=args.pose,
        pose_source_bundle_paths=args.pose_source_bundle,
        config=config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"bundle_sha256": bundle.bundle_sha256, **stats},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
