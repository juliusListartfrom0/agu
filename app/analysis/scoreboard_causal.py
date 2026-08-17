"""Attach raw-only ball/rim/pose evidence to scoreboard-localized events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.analysis.game_state.candidates import (
    ShotCandidateConfig,
    propose_shot_and_rebound_candidates,
)
from app.analysis.schemas import (
    EventEvidenceResponse,
    GameEventResponse,
    PerceptionDetectionResponse,
)


@dataclass(frozen=True)
class ScoreboardCausalConfig:
    min_ball_confidence: float = 0.03
    min_rim_confidence: float = 0.10
    max_rim_frame_gap: int = 6
    max_normalized_distance: float = 4.0
    max_ball_rim_iou: float = 0.80
    cluster_gap_frames: int = 15
    maximum_cluster_span_frames: int = 45
    shooter_lookback_frames: int = 90
    approach_lookback_frames: int = 90
    minimum_approach_points: int = 2
    minimum_approach_rise_px: float = 8.0
    maximum_ball_speed_px_per_frame: float = 80.0
    maximum_player_candidates: int = 16


def attach_scoreboard_causal_evidence(
    events: Sequence[GameEventResponse],
    detections: Sequence[PerceptionDetectionResponse],
    *,
    config: ScoreboardCausalConfig | None = None,
) -> tuple[list[GameEventResponse], dict[str, int]]:
    """Add the strongest physical shot proposal inside each scoreboard window."""

    settings = config or ScoreboardCausalConfig()
    _validate_config(settings)
    by_frame = _index_detections(detections)
    enriched: list[GameEventResponse] = []
    matched = 0
    proposals = 0
    for event in events:
        if event.event_type != "field_goal_attempt":
            enriched.append(event)
            continue
        window_detections = [
            detection
            for frame in range(event.start_frame, event.end_frame + 1)
            for detection in by_frame.get(frame, ())
        ]
        candidates = [
            candidate
            for candidate in propose_shot_and_rebound_candidates(
                window_detections,
                source_video_id=event.source_video_id,
                config=ShotCandidateConfig(
                    min_ball_confidence=settings.min_ball_confidence,
                    min_rim_confidence=settings.min_rim_confidence,
                    max_rim_frame_gap=settings.max_rim_frame_gap,
                    max_normalized_distance=settings.max_normalized_distance,
                    max_ball_rim_iou=settings.max_ball_rim_iou,
                    cluster_gap_frames=settings.cluster_gap_frames,
                    maximum_cluster_span_frames=settings.maximum_cluster_span_frames,
                    pre_roll_frames=0,
                    post_roll_frames=0,
                    rebound_window_frames=settings.max_rim_frame_gap,
                    shooter_lookback_frames=settings.shooter_lookback_frames,
                    maximum_player_candidates=settings.maximum_player_candidates,
                    minimum_approach_points=settings.minimum_approach_points,
                    minimum_approach_rise_px=settings.minimum_approach_rise_px,
                    approach_lookback_frames=settings.approach_lookback_frames,
                    maximum_ball_speed_px_per_frame=(
                        settings.maximum_ball_speed_px_per_frame
                    ),
                ),
                event_id_prefix=f"{event.event_id}:causal",
            )
            if candidate.event_type == "field_goal_attempt"
        ]
        proposals += len(candidates)
        selected = _select_candidate(candidates)
        if selected is None:
            enriched.append(event)
            continue
        source_evidence = next(
            (
                evidence
                for evidence in selected.evidence
                if evidence.kind == "ball_rim_proximity_cluster"
            ),
            None,
        )
        if source_evidence is None:
            enriched.append(event)
            continue
        matched += 1
        details = {
            **source_evidence.details,
            "causal_evidence_version": "agu_window_ball_rim_pose_v1",
            "source_candidate_event_id": selected.event_id,
            "window_candidate_count": len(candidates),
            "selection_method": (
                "approach_points_then_rise_then_distance_then_latest"
            ),
        }
        evidence = EventEvidenceResponse(
            evidence_id=f"{event.event_id}:causal-ball-rim",
            kind="ball_rim_proximity_cluster",
            source_video_id=event.source_video_id,
            start_frame=source_evidence.start_frame,
            end_frame=source_evidence.end_frame,
            confidence=source_evidence.confidence,
            details=details,
        )
        enriched.append(
            event.model_copy(
                update={
                    "evidence": [*event.evidence, evidence],
                    "reason": (
                        f"{event.reason}; raw ball/rim trajectory candidate "
                        "attached without changing candidate semantics"
                    ),
                }
            )
        )
    return enriched, {
        "field_goal_count": sum(
            event.event_type == "field_goal_attempt" for event in events
        ),
        "matched_field_goal_count": matched,
        "physical_proposal_count": proposals,
    }


def _select_candidate(
    candidates: Sequence[GameEventResponse],
) -> GameEventResponse | None:
    if not candidates:
        return None

    def score(candidate: GameEventResponse) -> tuple[float, ...]:
        evidence = next(
            (
                item
                for item in candidate.evidence
                if item.kind == "ball_rim_proximity_cluster"
            ),
            candidate.evidence[0],
        )
        details = evidence.details
        return (
            float(details.get("maximum_approach_point_count") or 0),
            float(details.get("maximum_approach_rise_px") or 0),
            -float(details.get("minimum_normalized_distance") or 4.0),
            float(evidence.end_frame),
            float(evidence.confidence),
        )

    return max(candidates, key=score)


def _index_detections(
    detections: Sequence[PerceptionDetectionResponse],
) -> dict[int, list[PerceptionDetectionResponse]]:
    by_frame: dict[int, list[PerceptionDetectionResponse]] = {}
    for detection in detections:
        by_frame.setdefault(detection.frame, []).append(detection)
    return by_frame


def _validate_config(config: ScoreboardCausalConfig) -> None:
    values = (
        config.min_ball_confidence,
        config.min_rim_confidence,
        config.max_rim_frame_gap,
        config.max_normalized_distance,
        config.max_ball_rim_iou,
        config.cluster_gap_frames,
        config.maximum_cluster_span_frames,
        config.shooter_lookback_frames,
        config.approach_lookback_frames,
        config.minimum_approach_points,
        config.maximum_ball_speed_px_per_frame,
        config.maximum_player_candidates,
    )
    if min(values) <= 0 or config.minimum_approach_rise_px < 0:
        raise ValueError("scoreboard causal thresholds must be positive")
    if config.max_ball_rim_iou > 1:
        raise ValueError("maximum ball/rim IoU must be at most one")
