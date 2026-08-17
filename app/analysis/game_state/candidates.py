"""Raw-perception candidate proposals for bounded official-event review."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.analysis.perception.ball_tracking import GlobalBallPathSelector
from app.analysis.schemas import (
    BallTrackPointResponse,
    BoundingBoxResponse,
    EventEvidenceResponse,
    GameEventResponse,
    PerceptionDetectionResponse,
    Point2DResponse,
)

from .shot import ShotAssessment, assess_shot_trajectory


@dataclass(frozen=True)
class ShotCandidateConfig:
    min_ball_confidence: float = 0.08
    min_rim_confidence: float = 0.15
    max_rim_frame_gap: int = 12
    max_normalized_distance: float = 4.0
    max_ball_rim_iou: float = 0.80
    cluster_gap_frames: int = 90
    maximum_cluster_span_frames: int | None = None
    pre_roll_frames: int = 45
    post_roll_frames: int = 60
    rebound_window_frames: int = 90
    min_candidate_confidence: float = 0.0
    suppress_rebound_after_vision_make: bool = False
    rebound_suppression_make_confidence: float = 0.5
    shooter_lookback_frames: int = 90
    maximum_player_candidates: int = 16
    minimum_approach_points: int = 1
    minimum_approach_rise_px: float = 0.0
    approach_lookback_frames: int = 90
    maximum_ball_speed_px_per_frame: float = 80.0


@dataclass(frozen=True)
class LegacyAnalysisCandidateConfig:
    """Bounds for adapting AGU's existing action analysis into review candidates."""

    min_frame: int = 0
    max_frame: int | None = None
    min_shot_confidence: float = 0.0
    include_event_types: tuple[str, ...] = (
        "field_goal_attempt",
        "rebound",
        "steal",
        "turnover",
        "block",
    )


@dataclass(frozen=True)
class RimProximityHit:
    frame: int
    ball_detection_id: str
    rim_detection_id: str
    normalized_distance: float
    confidence: float
    approach_point_count: int = 1
    approach_rise_px: float = 0.0


def propose_shot_and_rebound_candidates(
    detections: Iterable[PerceptionDetectionResponse],
    *,
    source_video_id: str,
    config: ShotCandidateConfig | None = None,
    event_id_prefix: str = "vision",
    ball_path_selector: GlobalBallPathSelector | None = None,
    ball_path_reset_frames: Iterable[int] = (),
) -> list[GameEventResponse]:
    """Propose review candidates from temporally aligned ball/rim evidence.

    A proposal is never automatically accepted. Sparse single-frame hits remain
    useful for recall, while temporal clustering prevents per-frame duplicates.
    """

    settings = config or ShotCandidateConfig()
    items = list(detections)
    balls = [
        item
        for item in items
        if item.object_type == "basketball" and item.confidence >= settings.min_ball_confidence
    ]
    if ball_path_selector is not None:
        balls = [
            detection
            for path in ball_path_selector.select_detection_paths(
                balls,
                reset_frames=ball_path_reset_frames,
            )
            for detection in path
        ]
    rims = [
        item for item in items if item.object_type == "rim" and item.confidence >= settings.min_rim_confidence
    ]
    if not balls or not rims:
        return []
    rim_frames = sorted({item.frame for item in rims})
    rims_by_frame: dict[int, list[PerceptionDetectionResponse]] = {}
    for rim in rims:
        rims_by_frame.setdefault(rim.frame, []).append(rim)

    hits: list[RimProximityHit] = []
    for ball in sorted(balls, key=lambda item: item.frame):
        nearest_frame = min(rim_frames, key=lambda frame: abs(frame - ball.frame))
        if abs(nearest_frame - ball.frame) > settings.max_rim_frame_gap:
            continue
        candidate_rims = rims_by_frame[nearest_frame]
        scored = [(_normalized_ball_rim_distance(ball, rim), rim) for rim in candidate_rims]
        distance, rim = min(scored, key=lambda pair: pair[0])
        if (
            distance > settings.max_normalized_distance
            or _bbox_iou(ball, rim) > settings.max_ball_rim_iou
        ):
            continue
        approach = _backward_ball_approach(ball, balls, settings=settings)
        approach_rise_px = max(
            (item.bbox.y1 + item.bbox.y2) / 2.0 for item in approach
        ) - min((item.bbox.y1 + item.bbox.y2) / 2.0 for item in approach)
        if (
            len(approach) < settings.minimum_approach_points
            or approach_rise_px < settings.minimum_approach_rise_px
        ):
            continue
        proximity = max(0.0, 1.0 - distance / settings.max_normalized_distance)
        confidence = min(ball.confidence, rim.confidence) * (0.55 + 0.45 * proximity)
        hits.append(
            RimProximityHit(
                frame=ball.frame,
                ball_detection_id=ball.detection_id,
                rim_detection_id=rim.detection_id,
                normalized_distance=distance,
                confidence=confidence,
                approach_point_count=len(approach),
                approach_rise_px=approach_rise_px,
            )
        )

    clusters = _cluster_hits(
        hits,
        max_gap_frames=settings.cluster_gap_frames,
        maximum_span_frames=settings.maximum_cluster_span_frames,
    )
    detections_by_id = {item.detection_id: item for item in items}
    events: list[GameEventResponse] = []
    for index, cluster in enumerate(clusters, start=1):
        first_frame = min(hit.frame for hit in cluster)
        last_frame = max(hit.frame for hit in cluster)
        cluster_frames = sorted(hit.frame for hit in cluster)
        review_anchor_frame = cluster_frames[len(cluster_frames) // 2]
        shot_id = f"{event_id_prefix}-shot-{index:05d}"
        trajectory = _assess_cluster_trajectory(
            cluster,
            detections_by_id,
            ball_detections=balls,
            maximum_frame_gap=settings.max_rim_frame_gap,
            maximum_speed_px_per_frame=settings.maximum_ball_speed_px_per_frame,
            lookback_frames=settings.approach_lookback_frames,
            lookahead_frames=settings.rebound_window_frames,
        )
        # Actor evidence must precede the shot result. When trajectory outcome
        # is unresolved, stop at the first rim-proximity hit instead of
        # admitting post-shot rebounders as shooter candidates.
        shooter_anchor_frame = trajectory.outcome_frame or first_frame
        # Actor evidence may need to begin before the compact review window.
        # Bounding it by ``shot_context_start`` silently reduced every larger
        # shooter lookback back to ``pre_roll_frames``.
        shooter_context_start = max(
            0,
            shooter_anchor_frame - settings.shooter_lookback_frames,
        )
        shot_ball_ids = [
            item.detection_id
            for item in items
            if item.object_type == "basketball"
            and shooter_context_start <= item.frame <= shooter_anchor_frame
        ]
        shot_players, shot_teams, shot_observations = _nearby_player_candidates(
            items,
            ball_ids=shot_ball_ids,
            detections_by_id=detections_by_id,
            start_frame=shooter_context_start,
            end_frame=shooter_anchor_frame,
            maximum_frame_gap=settings.max_rim_frame_gap,
            maximum=settings.maximum_player_candidates,
        )
        evidence = EventEvidenceResponse(
            evidence_id=f"{shot_id}:ball-rim",
            kind="ball_rim_proximity_cluster",
            source_video_id=source_video_id,
            start_frame=first_frame,
            end_frame=last_frame,
            confidence=max(hit.confidence for hit in cluster),
            details={
                "hit_count": len(cluster),
                "frames": [hit.frame for hit in cluster],
                "candidate_event_frame": first_frame,
                "review_anchor_frame": review_anchor_frame,
                "ball_detection_ids": [hit.ball_detection_id for hit in cluster],
                "rim_detection_ids": [hit.rim_detection_id for hit in cluster],
                "review_rim_observations": [
                    {
                        "frame": hit.frame,
                        "ball_bbox": detections_by_id[hit.ball_detection_id].bbox.model_dump(),
                        "rim_bbox": detections_by_id[hit.rim_detection_id].bbox.model_dump(),
                    }
                    for hit in cluster
                    if hit.ball_detection_id in detections_by_id
                    and hit.rim_detection_id in detections_by_id
                ],
                "minimum_normalized_distance": min(hit.normalized_distance for hit in cluster),
                "maximum_approach_point_count": max(hit.approach_point_count for hit in cluster),
                "maximum_approach_rise_px": max(hit.approach_rise_px for hit in cluster),
                "candidate_player_ids": shot_players,
                "candidate_team_ids": shot_teams,
                "candidate_player_observations": shot_observations,
                "trajectory_outcome": trajectory.outcome,
                "trajectory_confidence": trajectory.confidence,
                "trajectory_evidence": list(trajectory.evidence),
                **(
                    {"ball_path_backend": "agu_ball_global_path_v1"}
                    if ball_path_selector is not None
                    else {}
                ),
            },
        )
        confidence = min(0.85, 0.25 + max(hit.confidence for hit in cluster) + min(0.2, len(cluster) * 0.03))
        if confidence < settings.min_candidate_confidence:
            continue
        shot = GameEventResponse(
            event_id=shot_id,
            revision=1,
            event_type="field_goal_attempt",
            source_video_id=source_video_id,
            start_frame=max(0, first_frame - settings.pre_roll_frames),
            end_frame=last_frame + settings.post_roll_frames,
            outcome=trajectory.outcome,
            outcome_frame=trajectory.outcome_frame,
            primary_player_id=shot_players[0] if len(shot_players) == 1 else None,
            team_id=shot_teams[0] if len(shot_teams) == 1 else None,
            status="needs_review",
            confidence=confidence,
            evidence=[evidence],
            reason="ball/rim proximity cluster requires raw-video review for release, actor, value and outcome",
        )
        if (
            settings.suppress_rebound_after_vision_make
            and trajectory.outcome == "made"
            and trajectory.confidence >= settings.rebound_suppression_make_confidence
        ):
            events.append(shot)
            continue
        rebound_id = f"{event_id_prefix}-rebound-{index:05d}"
        rebound_ball_ids = [
            item.detection_id
            for item in items
            if item.object_type == "basketball"
            and last_frame <= item.frame <= last_frame + settings.rebound_window_frames
        ]
        rebound_players, rebound_teams, rebound_observations = _nearby_player_candidates(
            items,
            ball_ids=rebound_ball_ids,
            detections_by_id=detections_by_id,
            start_frame=last_frame,
            end_frame=last_frame + settings.rebound_window_frames,
            maximum_frame_gap=settings.max_rim_frame_gap,
            maximum=settings.maximum_player_candidates,
            prefer_stable_control=True,
        )
        rebound_evidence = evidence.model_copy(
            update={
                "evidence_id": f"{rebound_id}:preceding-shot",
                "details": {
                    **evidence.details,
                    "candidate_event_frame": last_frame + settings.rebound_window_frames // 2,
                    "candidate_player_ids": rebound_players,
                    "candidate_team_ids": rebound_teams,
                    "candidate_player_observations": rebound_observations,
                },
            }
        )
        rebound = GameEventResponse(
            event_id=rebound_id,
            revision=1,
            event_type="rebound",
            source_video_id=source_video_id,
            start_frame=last_frame,
            end_frame=last_frame + settings.rebound_window_frames,
            rebound_type="unknown",
            primary_player_id=rebound_players[0] if len(rebound_players) == 1 else None,
            team_id=rebound_teams[0] if len(rebound_teams) == 1 else None,
            status="needs_review",
            confidence=max(0.1, confidence * 0.65),
            evidence=[rebound_evidence],
            related_event_ids=[shot_id],
            reason="possible first control after a shot candidate; reject when the shot was made or no rebound occurred",
        )
        events.extend((shot, rebound))
    return events


def _backward_ball_approach(
    anchor: PerceptionDetectionResponse,
    balls: Sequence[PerceptionDetectionResponse],
    *,
    settings: ShotCandidateConfig,
) -> list[PerceptionDetectionResponse]:
    """Greedily link a causal pre-rim ball path using only raw detections."""
    chain = [anchor]
    current = anchor
    eligible = [
        item
        for item in balls
        if anchor.frame - settings.approach_lookback_frames <= item.frame < anchor.frame
    ]
    for frame in sorted({item.frame for item in eligible}, reverse=True):
        gap = current.frame - frame
        if gap <= 0 or gap > settings.max_rim_frame_gap:
            continue
        current_center = _bbox_center(current)
        candidates = [item for item in eligible if item.frame == frame]
        candidate = min(
            candidates,
            key=lambda item: math.hypot(
                _bbox_center(item)["x"] - current_center["x"],
                _bbox_center(item)["y"] - current_center["y"],
            ),
        )
        candidate_center = _bbox_center(candidate)
        distance = math.hypot(
            candidate_center["x"] - current_center["x"],
            candidate_center["y"] - current_center["y"],
        )
        if distance > settings.maximum_ball_speed_px_per_frame * gap:
            continue
        chain.append(candidate)
        current = candidate
    return list(reversed(chain))


def _assess_cluster_trajectory(
    cluster: Sequence[RimProximityHit],
    detections_by_id: Mapping[str, PerceptionDetectionResponse],
    *,
    ball_detections: Sequence[PerceptionDetectionResponse] = (),
    maximum_frame_gap: int = 12,
    maximum_speed_px_per_frame: float = 80.0,
    lookback_frames: int = 90,
    lookahead_frames: int = 90,
) -> ShotAssessment:
    best_by_frame: dict[int, RimProximityHit] = {}
    for hit in cluster:
        current = best_by_frame.get(hit.frame)
        if current is None or (hit.normalized_distance, -hit.confidence) < (
            current.normalized_distance,
            -current.confidence,
        ):
            best_by_frame[hit.frame] = hit
    points_by_frame: dict[int, BallTrackPointResponse] = {}
    rim_boxes: list[BoundingBoxResponse] = []
    for hit in sorted(best_by_frame.values(), key=lambda item: item.frame):
        ball = detections_by_id.get(hit.ball_detection_id)
        rim = detections_by_id.get(hit.rim_detection_id)
        if ball is None or rim is None:
            continue
        points_by_frame[hit.frame] = BallTrackPointResponse(
                frame=hit.frame,
                center=Point2DResponse(
                    x=(ball.bbox.x1 + ball.bbox.x2) / 2.0,
                    y=(ball.bbox.y1 + ball.bbox.y2) / 2.0,
                ),
                confidence=ball.confidence,
            )
        rim_boxes.append(rim.bbox)
    if not rim_boxes:
        return ShotAssessment("unknown", 0.0, None, ("missing_rim_geometry",))
    rim_box = BoundingBoxResponse(
        x1=float(sorted(box.x1 for box in rim_boxes)[len(rim_boxes) // 2]),
        y1=float(sorted(box.y1 for box in rim_boxes)[len(rim_boxes) // 2]),
        x2=float(sorted(box.x2 for box in rim_boxes)[len(rim_boxes) // 2]),
        y2=float(sorted(box.y2 for box in rim_boxes)[len(rim_boxes) // 2]),
    )
    anchor_hit = min(cluster, key=lambda item: (item.normalized_distance, -item.confidence, item.frame))
    anchor_ball = detections_by_id.get(anchor_hit.ball_detection_id)
    if anchor_ball is not None:
        linked = _linked_ball_trajectory(
            anchor_ball,
            ball_detections,
            start_frame=max(0, min(item.frame for item in cluster) - lookback_frames),
            end_frame=max(item.frame for item in cluster) + lookahead_frames,
            maximum_frame_gap=maximum_frame_gap,
            maximum_speed_px_per_frame=maximum_speed_px_per_frame,
        )
        for ball in linked:
            points_by_frame[ball.frame] = BallTrackPointResponse(
                frame=ball.frame,
                center=Point2DResponse(**_bbox_center(ball)),
                confidence=ball.confidence,
            )
    return assess_shot_trajectory(
        list(points_by_frame.values()),
        rim_box,
        rim_margin_px=max(4.0, (rim_box.y2 - rim_box.y1) * 0.15),
    )


def _linked_ball_trajectory(
    anchor: PerceptionDetectionResponse,
    balls: Sequence[PerceptionDetectionResponse],
    *,
    start_frame: int,
    end_frame: int,
    maximum_frame_gap: int,
    maximum_speed_px_per_frame: float,
) -> list[PerceptionDetectionResponse]:
    """Link a short raw-detection trajectory on both sides of a rim hit."""

    eligible = [item for item in balls if start_frame <= item.frame <= end_frame]

    def extend(current: PerceptionDetectionResponse, *, forward: bool) -> list[PerceptionDetectionResponse]:
        output: list[PerceptionDetectionResponse] = []
        frames = sorted(
            {item.frame for item in eligible if (item.frame > current.frame if forward else item.frame < current.frame)},
            reverse=not forward,
        )
        for frame in frames:
            gap = abs(frame - current.frame)
            if gap > maximum_frame_gap:
                break
            current_center = _bbox_center(current)
            candidates = [item for item in eligible if item.frame == frame]
            candidate = min(
                candidates,
                key=lambda item: (
                    math.hypot(
                        _bbox_center(item)["x"] - current_center["x"],
                        _bbox_center(item)["y"] - current_center["y"],
                    ),
                    -item.confidence,
                    item.detection_id,
                ),
            )
            distance = math.hypot(
                _bbox_center(candidate)["x"] - current_center["x"],
                _bbox_center(candidate)["y"] - current_center["y"],
            )
            if distance > maximum_speed_px_per_frame * gap:
                continue
            output.append(candidate)
            current = candidate
        return output

    before = extend(anchor, forward=False)
    after = extend(anchor, forward=True)
    return [*reversed(before), anchor, *after]


def _nearby_player_candidates(
    detections: Sequence[PerceptionDetectionResponse],
    *,
    ball_ids: Sequence[str],
    detections_by_id: Mapping[str, PerceptionDetectionResponse],
    start_frame: int,
    end_frame: int,
    maximum: int = 8,
    maximum_frame_gap: int = 4,
    prefer_stable_control: bool = False,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    players = [
        item
        for item in detections
        if item.object_type == "player"
        and item.player_id
        and start_frame <= item.frame <= end_frame
    ]
    if not players:
        return [], [], []
    anchor_balls = [detections_by_id[item] for item in ball_ids if item in detections_by_id]
    scores: dict[str, float] = {}
    team_by_player: dict[str, str] = {}
    observations_by_player: dict[
        str,
        list[
            tuple[
                PerceptionDetectionResponse,
                float,
                PerceptionDetectionResponse | None,
            ]
        ],
    ] = {}
    for player in players:
        player_id = str(player.player_id)
        if player.team_id:
            team_by_player[player_id] = player.team_id
        temporally_aligned = [
            ball for ball in anchor_balls if abs(ball.frame - player.frame) <= maximum_frame_gap
        ]
        if not anchor_balls:
            score = -player.confidence
            nearest_ball = None
        elif not temporally_aligned:
            continue
        else:
            nearest_ball, score = min(
                (
                    (
                        ball,
                        _normalized_ball_player_distance(ball, player)
                        + abs(ball.frame - player.frame)
                        / max(1, maximum_frame_gap)
                        * 0.15,
                    )
                    for ball in temporally_aligned
                ),
                key=lambda item: (item[1], -item[0].confidence, item[0].detection_id),
            )
        observations_by_player.setdefault(player_id, []).append(
            (player, score, nearest_ball)
        )
        if score < scores.get(player_id, float("inf")):
            scores[player_id] = score
    control_metrics = {
        player_id: _stable_control_metrics(
            observations,
            start_frame=start_frame,
            end_frame=end_frame,
            maximum_frame_gap=maximum_frame_gap,
        )
        for player_id, observations in observations_by_player.items()
    }
    ranking_scores = {
        player_id: (
            control_metrics[player_id]["score"]
            if prefer_stable_control and math.isfinite(control_metrics[player_id]["score"])
            else score + (4.0 if prefer_stable_control else 0.0)
        )
        for player_id, score in scores.items()
    }
    ranked = _rank_distinct_player_candidates(
        ranking_scores,
        observations_by_player,
        maximum=maximum,
        maximum_frame_gap=maximum_frame_gap,
    )
    teams = sorted({team_by_player[player_id] for player_id in ranked if player_id in team_by_player})
    observations = []
    for player_id in ranked:
        stable_control = control_metrics[player_id]
        sampled = _evenly_sample_observations(observations_by_player[player_id], maximum=32)
        observations.extend(
            {
                "player_id": player_id,
                "team_id": player.team_id,
                "frame": player.frame,
                "bbox": player.bbox.model_dump(mode="json"),
                "ball_player_distance": score,
                "wrist_ball_distance": (
                    _normalized_wrist_ball_distance(ball, player) if ball is not None else None
                ),
                "keypoints": {
                    name: point.model_dump(mode="json")
                    for name, point in player.keypoints.items()
                },
                "ball_frame": ball.frame if ball is not None else None,
                "ball_confidence": ball.confidence if ball is not None else None,
                "ball_center": _bbox_center(ball) if ball is not None else None,
                "ball_bbox": ball.bbox.model_dump(mode="json") if ball is not None else None,
                "stable_control_score": (
                    stable_control["score"] if math.isfinite(stable_control["score"]) else None
                ),
                "stable_contact_count": stable_control["contact_count"],
                "first_stable_control_frame": stable_control["first_frame"],
            }
            for player, score, ball in sampled
        )
    return ranked, teams, observations


def _stable_control_metrics(
    observations: Sequence[
        tuple[
            PerceptionDetectionResponse,
            float,
            PerceptionDetectionResponse | None,
        ]
    ],
    *,
    start_frame: int,
    end_frame: int,
    maximum_frame_gap: int,
) -> dict[str, float | int | None]:
    """Score repeated player/ball proximity as a conservative control prior.

    A single near-ball box is commonly a tip, occlusion, or false ball. A stable
    run needs at least two temporally linked observations. Earlier runs rank
    ahead for rebound ownership because the rule is first controlled possession.
    """

    best_by_frame: dict[int, float] = {}
    for player, distance, ball in observations:
        if ball is None or not math.isfinite(distance) or distance > 0.75:
            continue
        best_by_frame[player.frame] = min(distance, best_by_frame.get(player.frame, float("inf")))
    runs: list[list[tuple[int, float]]] = []
    for item in sorted(best_by_frame.items()):
        if not runs or item[0] - runs[-1][-1][0] > max(2, maximum_frame_gap * 2):
            runs.append([item])
        else:
            runs[-1].append(item)
    stable_runs = [run for run in runs if len(run) >= 2]
    if not stable_runs:
        return {"score": float("inf"), "contact_count": 0, "first_frame": None}
    run = min(stable_runs, key=lambda values: (values[0][0], -len(values)))
    distances = sorted(value for _frame, value in run)
    median_distance = distances[len(distances) // 2]
    window = max(1, end_frame - start_frame)
    first_offset = max(0, run[0][0] - start_frame) / window
    score = first_offset * 0.35 + median_distance * 0.45 + (1.0 / len(run)) * 0.20
    return {"score": score, "contact_count": len(run), "first_frame": run[0][0]}


def _rank_distinct_player_candidates(
    scores: Mapping[str, float],
    observations_by_player: Mapping[
        str,
        Sequence[
            tuple[
                PerceptionDetectionResponse,
                float,
                PerceptionDetectionResponse | None,
            ]
        ],
    ],
    *,
    maximum: int,
    maximum_frame_gap: int,
) -> list[str]:
    """Rank tracks while suppressing persistent cross-backend duplicates."""

    ranked: list[str] = []
    for player_id in sorted(scores, key=lambda item: (scores[item], item)):
        observations = observations_by_player[player_id]
        if any(
            _are_duplicate_player_tracks(
                observations,
                observations_by_player[selected],
                maximum_frame_gap=maximum_frame_gap,
            )
            for selected in ranked
        ):
            continue
        ranked.append(player_id)
        if len(ranked) >= maximum:
            break
    return ranked


def _are_duplicate_player_tracks(
    left: Sequence[
        tuple[PerceptionDetectionResponse, float, PerceptionDetectionResponse | None]
    ],
    right: Sequence[
        tuple[PerceptionDetectionResponse, float, PerceptionDetectionResponse | None]
    ],
    *,
    maximum_frame_gap: int,
) -> bool:
    """Require multi-frame agreement so one occlusion cannot merge real players."""

    overlaps: list[float] = []
    right_detections = [item[0] for item in right]
    for left_detection, _score, _ball in left:
        aligned = [
            item
            for item in right_detections
            if abs(item.frame - left_detection.frame) <= maximum_frame_gap
        ]
        if not aligned:
            continue
        nearest = min(
            aligned,
            key=lambda item: (abs(item.frame - left_detection.frame), item.detection_id),
        )
        overlaps.append(_bbox_iou(left_detection, nearest))
    if len(overlaps) < 3:
        return False
    high_overlap_fraction = sum(value >= 0.65 for value in overlaps) / len(overlaps)
    median_overlap = sorted(overlaps)[len(overlaps) // 2]
    return high_overlap_fraction >= 0.6 and median_overlap >= 0.65


def _bbox_iou(
    left: PerceptionDetectionResponse,
    right: PerceptionDetectionResponse,
) -> float:
    x1 = max(left.bbox.x1, right.bbox.x1)
    y1 = max(left.bbox.y1, right.bbox.y1)
    x2 = min(left.bbox.x2, right.bbox.x2)
    y2 = min(left.bbox.y2, right.bbox.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.bbox.x2 - left.bbox.x1) * max(
        0.0, left.bbox.y2 - left.bbox.y1
    )
    right_area = max(0.0, right.bbox.x2 - right.bbox.x1) * max(
        0.0, right.bbox.y2 - right.bbox.y1
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _evenly_sample_observations(
    observations: Sequence[
        tuple[
            PerceptionDetectionResponse,
            float,
            PerceptionDetectionResponse | None,
        ]
    ],
    *,
    maximum: int,
) -> list[
    tuple[
        PerceptionDetectionResponse,
        float,
        PerceptionDetectionResponse | None,
    ]
]:
    ordered = sorted(observations, key=lambda item: (item[0].frame, item[1], item[0].detection_id))
    if len(ordered) <= maximum:
        return ordered
    indexes = {
        round(index * (len(ordered) - 1) / (maximum - 1)) for index in range(maximum)
    }
    return [ordered[index] for index in sorted(indexes)]


def _bbox_center(detection: PerceptionDetectionResponse) -> dict[str, float]:
    return {
        "x": (detection.bbox.x1 + detection.bbox.x2) / 2.0,
        "y": (detection.bbox.y1 + detection.bbox.y2) / 2.0,
    }


def _normalized_ball_player_distance(
    ball: PerceptionDetectionResponse,
    player: PerceptionDetectionResponse,
) -> float:
    ball_x = (ball.bbox.x1 + ball.bbox.x2) / 2.0
    ball_y = (ball.bbox.y1 + ball.bbox.y2) / 2.0
    player_x = (player.bbox.x1 + player.bbox.x2) / 2.0
    player_y = (player.bbox.y1 + player.bbox.y2) / 2.0
    player_height = max(1.0, player.bbox.y2 - player.bbox.y1)
    return math.hypot(ball_x - player_x, ball_y - player_y) / player_height


def _normalized_wrist_ball_distance(
    ball: PerceptionDetectionResponse,
    player: PerceptionDetectionResponse,
) -> float | None:
    wrists = [
        player.keypoints[name]
        for name in ("left_wrist", "right_wrist")
        if name in player.keypoints
    ]
    if not wrists:
        return None
    ball_x = (ball.bbox.x1 + ball.bbox.x2) / 2.0
    ball_y = (ball.bbox.y1 + ball.bbox.y2) / 2.0
    player_height = max(1.0, player.bbox.y2 - player.bbox.y1)
    return min(math.hypot(ball_x - wrist.x, ball_y - wrist.y) for wrist in wrists) / player_height


def propose_legacy_analysis_candidates(
    analysis: Mapping[str, Any],
    *,
    source_video_id: str,
    config: LegacyAnalysisCandidateConfig | None = None,
    event_id_prefix: str = "legacy",
) -> list[GameEventResponse]:
    """Adapt raw-derived legacy action output without promoting it to official events."""

    settings = config or LegacyAnalysisCandidateConfig()
    events: list[GameEventResponse] = []
    for index, record in enumerate(analysis.get("records") or [], start=1):
        final = record.get("final") or {}
        if str(final.get("action") or "") not in {"shoot", "shot"}:
            continue
        confidence = float(final.get("confidence") or 0.0)
        if confidence < settings.min_shot_confidence:
            continue
        event = _legacy_candidate(
            event_id=f"{event_id_prefix}-record-{index:06d}",
            event_type="field_goal_attempt",
            source_video_id=source_video_id,
            start_frame=int(record.get("start_frame") or 0),
            end_frame=int(record.get("end_frame") or 0),
            confidence=confidence,
            player_id=record.get("global_player_id") or record.get("local_player_id"),
            kind="legacy_action_record",
            details={"source": str(final.get("source") or "record")},
        )
        if _candidate_in_bounds(event, settings) and event.event_type in settings.include_event_types:
            events.append(event)

    aliases = {"rebound": "rebound", "steal": "steal", "block": "block"}
    for index, candidate in enumerate((analysis.get("long_video") or {}).get("event_candidates") or [], start=1):
        raw_type = str(candidate.get("event_type") or "").replace("_candidate", "")
        event_type = aliases.get(raw_type)
        if event_type is None or event_type not in settings.include_event_types:
            continue
        event_id = f"{event_id_prefix}-event-{index:06d}"
        event = _legacy_candidate(
            event_id=event_id,
            event_type=event_type,
            source_video_id=source_video_id,
            start_frame=int(candidate.get("start_frame") or 0),
            end_frame=int(candidate.get("end_frame") or 0),
            confidence=float(candidate.get("confidence") or 0.0),
            player_id=candidate.get("player_id"),
            kind="legacy_long_video_candidate",
            details={
                "method": str(candidate.get("method") or "event_candidate"),
                "legacy_status": str(candidate.get("status") or ""),
            },
            related_event_ids=[f"{event_id}:turnover"] if event_type == "steal" else None,
        )
        if _candidate_in_bounds(event, settings):
            if event_type == "steal" and "turnover" in settings.include_event_types:
                events.append(
                    _legacy_candidate(
                        event_id=f"{event_id}:turnover",
                        event_type="turnover",
                        source_video_id=source_video_id,
                        start_frame=event.start_frame,
                        end_frame=event.end_frame,
                        confidence=event.confidence,
                        player_id=None,
                        kind="legacy_possession_loss_candidate",
                        details={"paired_steal_event_id": event.event_id},
                    )
                )
            events.append(event)
    causal_order = {"field_goal_attempt": 0, "rebound": 1, "turnover": 2, "steal": 3, "block": 4}
    return sorted(
        events,
        key=lambda event: (
            event.start_frame,
            event.end_frame,
            causal_order.get(event.event_type, 99),
            event.event_id,
        ),
    )


def attach_traditional_identity_candidates(
    events: Sequence[GameEventResponse],
    detections: Sequence[PerceptionDetectionResponse],
    *,
    context_frames: int,
    maximum_candidates: int = 8,
) -> list[GameEventResponse]:
    """Ground time-only candidates in current canonical player observations."""

    if context_frames < 0 or maximum_candidates <= 0:
        raise ValueError("identity context must be non-negative and maximum candidates positive")
    detections_by_id = {item.detection_id: item for item in detections}
    grounded: list[GameEventResponse] = []
    for event in events:
        anchor = event.outcome_frame or (event.start_frame + event.end_frame) // 2
        players, teams, observations = _nearby_player_candidates(
            detections,
            ball_ids=[],
            detections_by_id=detections_by_id,
            start_frame=max(0, anchor - context_frames),
            end_frame=anchor + context_frames,
            maximum=maximum_candidates,
        )
        evidence = [
            item.model_copy(
                update={
                    "details": {
                        **item.details,
                        "candidate_event_frame": anchor,
                        "candidate_player_ids": players,
                        "candidate_team_ids": teams,
                        "candidate_player_observations": observations,
                        "identity_grounding": "current_raw_perception",
                    }
                }
            )
            for item in event.evidence
        ]
        grounded.append(
            event.model_copy(
                update={
                    "primary_player_id": players[0] if len(players) == 1 else None,
                    "team_id": teams[0] if len(teams) == 1 else None,
                    "evidence": evidence,
                }
            )
        )
    return grounded


def merge_temporal_review_candidates(
    candidate_groups: Sequence[Iterable[GameEventResponse]],
    *,
    max_center_gap_frames: int,
) -> list[GameEventResponse]:
    """Fuse candidate sources by priority while retaining all audit evidence.

    Earlier groups have higher timing priority. Relations are remapped when an
    incoming event is suppressed as a temporal duplicate.
    """

    if max_center_gap_frames < 0:
        raise ValueError("max_center_gap_frames must be non-negative")
    merged: list[GameEventResponse] = []
    aliases: dict[str, str] = {}
    for group in candidate_groups:
        for event in group:
            center = (event.start_frame + event.end_frame) // 2
            match_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if existing.source_video_id == event.source_video_id
                    and existing.event_type == event.event_type
                    and abs((existing.start_frame + existing.end_frame) // 2 - center) <= max_center_gap_frames
                ),
                None,
            )
            if match_index is None:
                merged.append(event)
                continue
            existing = merged[match_index]
            aliases[event.event_id] = existing.event_id
            evidence_by_id = {item.evidence_id: item for item in [*existing.evidence, *event.evidence]}
            timing_updates: dict[str, object] = {}
            if existing.event_type in {"steal", "turnover"}:
                timing_updates = {
                    "start_frame": min(existing.start_frame, event.start_frame),
                    "end_frame": max(existing.end_frame, event.end_frame),
                    "outcome_frame": max(
                        existing.outcome_frame or existing.end_frame,
                        event.outcome_frame or event.end_frame,
                    ),
                }
            merged[match_index] = existing.model_copy(
                update={
                    **timing_updates,
                    "evidence": list(evidence_by_id.values()),
                    "confidence": max(existing.confidence, event.confidence),
                    "reason": f"{existing.reason}; corroborated by {event.event_id}",
                }
            )

    return [
        GameEventResponse.model_validate(
            event.model_copy(
                update={"related_event_ids": [aliases.get(item, item) for item in event.related_event_ids]}
            ).model_dump()
        )
        for event in sorted(merged, key=lambda item: (item.source_video_id, item.start_frame, item.event_id))
    ]


def link_causal_candidate_relations(
    events: Sequence[GameEventResponse],
    *,
    max_gap_frames: int,
) -> list[GameEventResponse]:
    """Attach only type-compatible nearby parent candidates.

    These links are hypotheses, not confirmation. Autonomous adjudication must
    still confirm the parent before a dependent event can become official.
    """

    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")
    parent_types = {
        "rebound": {"field_goal_attempt", "free_throw_attempt"},
        "assist": {"field_goal_attempt"},
        "block": {"field_goal_attempt"},
        "steal": {"turnover"},
    }
    linked: list[GameEventResponse] = []
    for event in events:
        if event.related_event_ids or event.event_type not in parent_types:
            linked.append(event)
            continue
        center = (event.start_frame + event.end_frame) // 2
        parents = [
            candidate
            for candidate in events
            if candidate.event_id != event.event_id
            and candidate.source_video_id == event.source_video_id
            and candidate.event_type in parent_types[event.event_type]
            and abs((candidate.start_frame + candidate.end_frame) // 2 - center) <= max_gap_frames
            and (
                event.event_type != "rebound"
                or (candidate.start_frame + candidate.end_frame) // 2 <= center
            )
        ]
        if not parents:
            linked.append(event)
            continue
        parent = min(
            parents,
            key=lambda candidate: (
                abs((candidate.start_frame + candidate.end_frame) // 2 - center),
                candidate.event_id,
            ),
        )
        linked.append(event.model_copy(update={"related_event_ids": [parent.event_id]}))
    return linked


def _legacy_candidate(
    *,
    event_id: str,
    event_type: str,
    source_video_id: str,
    start_frame: int,
    end_frame: int,
    confidence: float,
    player_id: object,
    kind: str,
    details: Mapping[str, object],
    related_event_ids: Sequence[str] | None = None,
) -> GameEventResponse:
    evidence = EventEvidenceResponse(
        evidence_id=f"{event_id}:legacy",
        kind=kind,
        source_video_id=source_video_id,
        start_frame=start_frame,
        end_frame=end_frame,
        confidence=max(0.0, min(1.0, confidence)),
        details=dict(details),
    )
    return GameEventResponse(
        event_id=event_id,
        revision=1,
        event_type=event_type,
        source_video_id=source_video_id,
        start_frame=start_frame,
        end_frame=end_frame,
        outcome_frame=end_frame if event_type in {"steal", "turnover"} else None,
        primary_player_id=str(player_id) if player_id else None,
        status="needs_review",
        confidence=max(0.0, min(1.0, confidence)),
        evidence=[evidence],
        related_event_ids=list(related_event_ids or []),
        reason="raw-derived legacy signal requires official-event review",
    )


def _candidate_in_bounds(event: GameEventResponse, config: LegacyAnalysisCandidateConfig) -> bool:
    center = (event.start_frame + event.end_frame) // 2
    return center >= config.min_frame and (config.max_frame is None or center <= config.max_frame)


def _cluster_hits(
    hits: list[RimProximityHit],
    *,
    max_gap_frames: int,
    maximum_span_frames: int | None = None,
) -> list[list[RimProximityHit]]:
    """Cluster nearby rim hits without allowing a sparse chain to span a possession.

    Gap-only clustering can bridge unrelated false ball detections indefinitely:
    every adjacent hit may be close while the first and last hit are many seconds
    apart. The optional span bound keeps the resulting shot/rebound windows causal.
    """

    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")
    if maximum_span_frames is not None and maximum_span_frames <= 0:
        raise ValueError("maximum_span_frames must be positive when configured")
    clusters: list[list[RimProximityHit]] = []
    for hit in sorted(hits, key=lambda item: item.frame):
        exceeds_gap = bool(clusters) and hit.frame - clusters[-1][-1].frame > max_gap_frames
        exceeds_span = (
            bool(clusters)
            and maximum_span_frames is not None
            and hit.frame - clusters[-1][0].frame > maximum_span_frames
        )
        if not clusters or exceeds_gap or exceeds_span:
            clusters.append([hit])
        else:
            clusters[-1].append(hit)
    return clusters


def _normalized_ball_rim_distance(
    ball: PerceptionDetectionResponse,
    rim: PerceptionDetectionResponse,
) -> float:
    ball_x = (ball.bbox.x1 + ball.bbox.x2) / 2.0
    ball_y = (ball.bbox.y1 + ball.bbox.y2) / 2.0
    rim_x = (rim.bbox.x1 + rim.bbox.x2) / 2.0
    rim_y = (rim.bbox.y1 + rim.bbox.y2) / 2.0
    rim_width = max(8.0, rim.bbox.x2 - rim.bbox.x1)
    rim_height = max(4.0, rim.bbox.y2 - rim.bbox.y1)
    return math.hypot((ball_x - rim_x) / rim_width, (ball_y - rim_y) / (rim_height * 3.0))
