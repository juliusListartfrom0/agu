#!/usr/bin/env python3
"""Convert raw-only perception JSON into sealed shot/rebound review candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.game_state import (  # noqa: E402
    LegacyAnalysisCandidateConfig,
    ShotCandidateConfig,
    attach_traditional_identity_candidates,
    merge_temporal_review_candidates,
    propose_legacy_analysis_candidates,
    propose_shot_and_rebound_candidates,
)
from app.analysis.official_evaluation import seal_raw_only_predictions  # noqa: E402
from app.analysis.official_identity import (  # noqa: E402
    canonicalize_perception_detections,
    verify_official_identity_artifact,
)
from app.analysis.perception import attach_pose_keypoints  # noqa: E402
from app.analysis.schemas import (  # noqa: E402
    OfficialIdentityGraphArtifactResponse,
    PerceptionDetectionResponse,
)
from app.analysis.shot_validity import (  # noqa: E402
    ShotValidityModel,
    apply_shot_validity_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument(
        "--additional-perception",
        type=Path,
        action="append",
        default=[],
        help="Additional same-video perception artifact (for example generic ball + specialist rim fusion)",
    )
    parser.add_argument(
        "--player-perception",
        type=Path,
        action="append",
        default=[],
        help="Same-video perception artifact used only for tracked player/team observations",
    )
    parser.add_argument(
        "--pose",
        type=Path,
        action="append",
        default=[],
        help="Same-video raw-bound pose artifact attached to tracked player observations",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-normalized-distance", type=float, default=4.0)
    parser.add_argument(
        "--max-rim-frame-gap-sec",
        type=float,
        default=0.0,
        help="Optional short hold for intermittently missed static rim detections",
    )
    parser.add_argument("--cluster-gap-sec", type=float, default=3.0)
    parser.add_argument(
        "--max-cluster-span-sec",
        type=float,
        default=0.0,
        help="Optional cap for chained ball/rim hits; 0 keeps legacy gap-only clustering",
    )
    parser.add_argument("--min-shot-candidate-confidence", type=float, default=0.60)
    parser.add_argument("--suppress-rebound-after-vision-make", action="store_true")
    parser.add_argument("--rebound-suppression-make-confidence", type=float, default=0.50)
    parser.add_argument("--shooter-lookback-sec", type=float, default=3.0)
    parser.add_argument("--minimum-approach-points", type=int, default=1)
    parser.add_argument("--minimum-approach-rise-px", type=float, default=0.0)
    parser.add_argument("--approach-lookback-sec", type=float, default=3.0)
    parser.add_argument("--maximum-ball-speed-px-per-frame", type=float, default=80.0)
    parser.add_argument(
        "--maximum-player-candidates",
        type=int,
        default=16,
        help="Maximum raw-grounded player identities retained per shot/rebound candidate",
    )
    parser.add_argument("--legacy-analysis", type=Path)
    parser.add_argument("--identity-graph", type=Path)
    parser.add_argument(
        "--require-face-gallery-identity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exclude anonymous player tracks from official candidates; requires a sealed identity graph "
            "with at least one face-gallery match"
        ),
    )
    parser.add_argument(
        "--identity-graph-perception-index",
        type=int,
        default=0,
        help="Zero-based perception source index whose raw player IDs the identity graph resolves",
    )
    parser.add_argument("--merge-gap-sec", type=float, default=3.0)
    parser.add_argument(
        "--shot-validity-model",
        type=Path,
        help="Optional sealed Extra Trees model used to reject false shot candidates",
    )
    parser.add_argument(
        "--shot-validity-threshold",
        type=float,
        help="Optional override for the calibrated model threshold",
    )
    return parser.parse_args()


def build_candidates(
    *,
    perception_path: Path,
    video_path: Path,
    game_id: str,
    max_normalized_distance: float,
    cluster_gap_sec: float,
    max_cluster_span_sec: float | None = None,
    max_rim_frame_gap_sec: float = 0.0,
    legacy_analysis_path: Path | None = None,
    merge_gap_sec: float = 3.0,
    additional_perception_paths: list[Path] | None = None,
    player_perception_paths: list[Path] | None = None,
    pose_paths: list[Path] | None = None,
    min_shot_candidate_confidence: float = 0.0,
    suppress_rebound_after_vision_make: bool = False,
    rebound_suppression_make_confidence: float = 0.5,
    shooter_lookback_sec: float = 3.0,
    maximum_player_candidates: int = 16,
    identity_graph_path: Path | None = None,
    identity_graph_perception_index: int = 0,
    require_face_gallery_identity: bool = False,
    minimum_approach_points: int = 1,
    minimum_approach_rise_px: float = 0.0,
    approach_lookback_sec: float = 3.0,
    maximum_ball_speed_px_per_frame: float = 80.0,
    shot_validity_model_path: Path | None = None,
    shot_validity_threshold: float | None = None,
) -> object:
    if maximum_player_candidates <= 0:
        raise ValueError("maximum_player_candidates must be positive")
    if minimum_approach_points <= 0 or maximum_ball_speed_px_per_frame <= 0:
        raise ValueError("ball approach thresholds must be positive")
    if max_cluster_span_sec is not None and max_cluster_span_sec <= 0:
        raise ValueError("max_cluster_span_sec must be positive when configured")
    payload = json.loads(perception_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "agu.official-perception.v1":
        raise ValueError("unsupported perception schema")
    raw = payload.get("raw_video") or {}
    if raw.get("filename") != video_path.name:
        raise ValueError("perception raw filename does not match declared video")
    source_fps = float(raw.get("source_fps") or 0)
    if source_fps <= 0:
        raise ValueError("perception source_fps must be positive")
    perception_sources: list[tuple[dict[str, object], set[str] | None]] = [(payload, None)]
    for additional_path in additional_perception_paths or []:
        additional = json.loads(additional_path.read_text(encoding="utf-8"))
        additional_raw = additional.get("raw_video") or {}
        if additional.get("schema_version") != "agu.official-perception.v1":
            raise ValueError("unsupported additional perception schema")
        if (
            additional_raw.get("filename") != raw.get("filename")
            or additional_raw.get("sha256") != raw.get("sha256")
            or float(additional_raw.get("source_fps") or 0) != source_fps
        ):
            raise ValueError("additional perception artifact does not match the declared raw video")
        perception_sources.append((additional, None))
    for player_path in player_perception_paths or []:
        player_payload = json.loads(player_path.read_text(encoding="utf-8"))
        player_raw = player_payload.get("raw_video") or {}
        if player_payload.get("schema_version") != "agu.official-perception.v1":
            raise ValueError("unsupported player perception schema")
        if (
            player_raw.get("filename") != raw.get("filename")
            or player_raw.get("sha256") != raw.get("sha256")
            or float(player_raw.get("source_fps") or 0) != source_fps
        ):
            raise ValueError("player perception artifact does not match the declared raw video")
        perception_sources.append((player_payload, {"player"}))
    identity_artifact = None
    if identity_graph_path is not None:
        if not 0 <= identity_graph_perception_index < len(perception_sources):
            raise ValueError("identity_graph_perception_index is outside perception sources")
        identity_artifact = verify_official_identity_artifact(
            OfficialIdentityGraphArtifactResponse.model_validate_json(identity_graph_path.read_text(encoding="utf-8"))
        )
    face_gallery_player_ids: set[str] = set()
    if require_face_gallery_identity:
        if identity_artifact is None:
            raise ValueError("face-gallery identity enforcement requires --identity-graph")
        face_gallery_player_ids = _face_gallery_anchored_player_ids(identity_artifact)
        if not face_gallery_player_ids:
            raise ValueError("identity graph contains no face-gallery anchored players")
    detections: list[PerceptionDetectionResponse] = []
    authoritative_player_start = 1 + len(additional_perception_paths or [])
    dedicated_player_perception = bool(player_perception_paths)
    for artifact_index, (perception_payload, allowed_types) in enumerate(perception_sources):
        source_detections = [
            PerceptionDetectionResponse.model_validate(item)
            for item in perception_payload.get("detections") or []
            if (allowed_types is None or str(item.get("object_type") or "") in allowed_types)
            and not (
                dedicated_player_perception
                and artifact_index < authoritative_player_start
                and str(item.get("object_type") or "") == "player"
            )
        ]
        preserve_player_id = False
        if identity_artifact is not None and artifact_index == identity_graph_perception_index:
            source_detections, _source_video_id = canonicalize_perception_detections(
                identity_artifact,
                source_detections,
                raw_filename=video_path.name,
                raw_sha256=str(raw.get("sha256") or ""),
            )
            preserve_player_id = True
        detections.extend(
            _load_detection(
                item,
                artifact_index,
                preserve_player_id=preserve_player_id,
            )
            for item in source_detections
        )
    if require_face_gallery_identity:
        detections = [
            item for item in detections if item.object_type != "player" or item.player_id in face_gallery_player_ids
        ]
    pose_detections: list[PerceptionDetectionResponse] = []
    pose_model_hashes: list[str] = []
    pose_maximum_frame_gap = 0
    for pose_path in pose_paths or []:
        pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
        pose_raw = pose_payload.get("raw_video") or {}
        if pose_payload.get("schema_version") != "agu.official-pose.v1":
            raise ValueError("unsupported pose schema")
        if (
            pose_raw.get("filename") != raw.get("filename")
            or pose_raw.get("sha256") != raw.get("sha256")
            or float(pose_raw.get("source_fps") or 0) != source_fps
        ):
            raise ValueError("pose artifact does not match the declared raw video")
        pose_detections.extend(
            PerceptionDetectionResponse.model_validate(item) for item in pose_payload.get("detections") or []
        )
        pose_model_hashes.append(str((pose_payload.get("pose") or {}).get("model_sha256") or ""))
        pose_stride = int((pose_payload.get("sampling") or {}).get("stride_frames") or 0)
        pose_maximum_frame_gap = max(pose_maximum_frame_gap, max(0, pose_stride // 2))
    if pose_detections:
        detections = attach_pose_keypoints(
            detections,
            pose_detections,
            maximum_frame_gap=pose_maximum_frame_gap,
            minimum_iou=0.25,
        )
    settings = ShotCandidateConfig(
        max_normalized_distance=max_normalized_distance,
        cluster_gap_frames=max(1, int(round(cluster_gap_sec * source_fps))),
        maximum_cluster_span_frames=(
            max(1, int(round(max_cluster_span_sec * source_fps))) if max_cluster_span_sec is not None else None
        ),
        pre_roll_frames=max(1, int(round(1.5 * source_fps))),
        post_roll_frames=max(1, int(round(2.0 * source_fps))),
        rebound_window_frames=max(1, int(round(3.0 * source_fps))),
        max_rim_frame_gap=(
            max(1, int(round(max_rim_frame_gap_sec * source_fps)))
            if max_rim_frame_gap_sec > 0
            else max(1, int(payload.get("sampling", {}).get("stride_frames") or 1) * 2)
        ),
        min_candidate_confidence=min_shot_candidate_confidence,
        suppress_rebound_after_vision_make=suppress_rebound_after_vision_make,
        rebound_suppression_make_confidence=rebound_suppression_make_confidence,
        shooter_lookback_frames=max(1, int(round(shooter_lookback_sec * source_fps))),
        maximum_player_candidates=maximum_player_candidates,
        minimum_approach_points=minimum_approach_points,
        minimum_approach_rise_px=minimum_approach_rise_px,
        approach_lookback_frames=max(1, int(round(approach_lookback_sec * source_fps))),
        maximum_ball_speed_px_per_frame=maximum_ball_speed_px_per_frame,
    )
    vision_events = propose_shot_and_rebound_candidates(
        detections,
        source_video_id="video_001",
        config=settings,
        event_id_prefix="raw-vision",
    )
    shot_validity_model = None
    vision_shot_count_before_gate = sum(
        event.event_type == "field_goal_attempt" for event in vision_events
    )
    if shot_validity_model_path is not None:
        shot_validity_model = ShotValidityModel(
            json.loads(shot_validity_model_path.read_text(encoding="utf-8"))
        )
        vision_events = apply_shot_validity_gate(
            vision_events,
            model=shot_validity_model,
            threshold=shot_validity_threshold,
        )
    vision_shot_count_after_gate = sum(
        event.event_type == "field_goal_attempt" for event in vision_events
    )
    candidate_groups = [vision_events]
    if legacy_analysis_path is not None:
        legacy_payload = json.loads(legacy_analysis_path.read_text(encoding="utf-8"))
        candidate_groups.append(
            attach_traditional_identity_candidates(
                propose_legacy_analysis_candidates(
                    legacy_payload,
                    source_video_id="video_001",
                    config=LegacyAnalysisCandidateConfig(
                        min_frame=int(raw.get("start_frame") or payload.get("sampling", {}).get("start_frame") or 0),
                        max_frame=int(payload.get("sampling", {}).get("end_frame") or 0) or None,
                        include_event_types=(
                            "field_goal_attempt",
                            "rebound",
                            "steal",
                            "turnover",
                        ),
                    ),
                ),
                detections,
                context_frames=max(1, int(round(source_fps * 2.0))),
            )
        )
    events = merge_temporal_review_candidates(
        candidate_groups,
        max_center_gap_frames=max(0, int(round(merge_gap_sec * source_fps))),
    )
    return seal_raw_only_predictions(
        game_id=game_id,
        raw_video_paths=[video_path],
        events=events,
        config={
            "pipeline": "ball_rim_candidate_v1",
            "max_normalized_distance": max_normalized_distance,
            "max_rim_frame_gap_sec": max_rim_frame_gap_sec,
            "cluster_gap_sec": cluster_gap_sec,
            "max_cluster_span_sec": max_cluster_span_sec,
            "candidate_sources": ["ball_rim", *(["legacy_analysis"] if legacy_analysis_path else [])],
            "perception_artifact_count": len(perception_sources),
            "player_only_perception_artifact_count": len(player_perception_paths or []),
            "dedicated_player_perception_authoritative": dedicated_player_perception,
            "min_shot_candidate_confidence": min_shot_candidate_confidence,
            "suppress_rebound_after_vision_make": suppress_rebound_after_vision_make,
            "rebound_suppression_make_confidence": rebound_suppression_make_confidence,
            "shooter_lookback_sec": shooter_lookback_sec,
            "maximum_player_candidates": maximum_player_candidates,
            "minimum_approach_points": minimum_approach_points,
            "minimum_approach_rise_px": minimum_approach_rise_px,
            "approach_lookback_sec": approach_lookback_sec,
            "maximum_ball_speed_px_per_frame": maximum_ball_speed_px_per_frame,
            "identity_graph_enabled": identity_artifact is not None,
            "identity_graph_perception_index": identity_graph_perception_index,
            "require_face_gallery_identity": require_face_gallery_identity,
            "face_gallery_anchored_player_count": len(face_gallery_player_ids),
            "pose_artifact_count": len(pose_paths or []),
            "merge_gap_sec": merge_gap_sec,
            "shot_validity_enabled": shot_validity_model is not None,
            "shot_validity_threshold": (
                shot_validity_threshold
                if shot_validity_threshold is not None
                else (
                    float(shot_validity_model.artifact["threshold"])
                    if shot_validity_model is not None
                    else None
                )
            ),
            "vision_shot_count_before_gate": vision_shot_count_before_gate,
            "vision_shot_count_after_gate": vision_shot_count_after_gate,
        },
        model_provenance={
            "producer": "agu",
            "artifact_role": "candidate_only",
            "candidate_backend": (
                "traditional_cv_ball_rim+shot_validity+agu_action_model"
                if shot_validity_model is not None
                else "traditional_cv_ball_rim+agu_action_model"
            ),
            "perception_schema": str(payload["schema_version"]),
            "detector_model_sha256": str((payload.get("detector") or {}).get("model_sha256") or ""),
            "additional_detector_model_sha256": ",".join(
                str((item.get("detector") or {}).get("model_sha256") or "")
                for item, _allowed_types in perception_sources[1:]
            ),
            "player_tracking": str(bool((payload.get("detector") or {}).get("player_tracking"))).lower(),
            "team_assignment": str((payload.get("detector") or {}).get("team_assignment") or ""),
            "identity_graph_sha256": identity_artifact.artifact_sha256 if identity_artifact else "",
            "identity_embedding_model": (
                identity_artifact.model_provenance.get("embedding_model", "") if identity_artifact else ""
            ),
            "pose_model_sha256": ",".join(pose_model_hashes),
            "shot_validity_model_sha256": (
                str(shot_validity_model.artifact["model_sha256"])
                if shot_validity_model is not None
                else ""
            ),
            "vision_shot_count_before_gate": str(vision_shot_count_before_gate),
            "vision_shot_count_after_gate": str(vision_shot_count_after_gate),
        },
    )


def _load_detection(
    item: object,
    artifact_index: int,
    *,
    preserve_player_id: bool = False,
) -> PerceptionDetectionResponse:
    detection = PerceptionDetectionResponse.model_validate(item)
    updates: dict[str, str] = {"detection_id": f"perception-{artifact_index}:{detection.detection_id}"}
    if artifact_index > 0:
        if detection.track_id:
            updates["track_id"] = f"perception-{artifact_index}:{detection.track_id}"
        if detection.player_id and not preserve_player_id:
            updates["player_id"] = f"perception-{artifact_index}:{detection.player_id}"
    return detection.model_copy(update=updates)


def _face_gallery_anchored_player_ids(
    artifact: OfficialIdentityGraphArtifactResponse,
) -> set[str]:
    """Return only canonical identities independently anchored by the enrollment face list."""

    identity_ids = {item.player_id for item in artifact.identities}
    return {
        str(tracklet.gallery_person_id)
        for tracklet in artifact.tracklets
        if tracklet.gallery_person_id and tracklet.gallery_person_id in identity_ids
    }


def main() -> int:
    args = parse_args()
    bundle = build_candidates(
        perception_path=args.perception,
        video_path=args.video,
        game_id=args.game_id,
        max_normalized_distance=args.max_normalized_distance,
        cluster_gap_sec=args.cluster_gap_sec,
        max_cluster_span_sec=args.max_cluster_span_sec or None,
        max_rim_frame_gap_sec=args.max_rim_frame_gap_sec,
        legacy_analysis_path=args.legacy_analysis,
        merge_gap_sec=args.merge_gap_sec,
        additional_perception_paths=args.additional_perception,
        player_perception_paths=args.player_perception,
        pose_paths=args.pose,
        min_shot_candidate_confidence=args.min_shot_candidate_confidence,
        suppress_rebound_after_vision_make=args.suppress_rebound_after_vision_make,
        rebound_suppression_make_confidence=args.rebound_suppression_make_confidence,
        shooter_lookback_sec=args.shooter_lookback_sec,
        maximum_player_candidates=args.maximum_player_candidates,
        identity_graph_path=args.identity_graph,
        identity_graph_perception_index=args.identity_graph_perception_index,
        require_face_gallery_identity=args.require_face_gallery_identity,
        minimum_approach_points=args.minimum_approach_points,
        minimum_approach_rise_px=args.minimum_approach_rise_px,
        approach_lookback_sec=args.approach_lookback_sec,
        maximum_ball_speed_px_per_frame=args.maximum_ball_speed_px_per_frame,
        shot_validity_model_path=args.shot_validity_model,
        shot_validity_threshold=args.shot_validity_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for event in bundle.events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    print(json.dumps({"bundle_sha256": bundle.bundle_sha256, "event_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
