from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.analysis.schemas import Point2DResponse, PossessionTransitionResponse


@dataclass(frozen=True)
class PossessionObservation:
    frame: int
    ball: Point2DResponse | None
    player_hands: dict[str, tuple[Point2DResponse, ...]] = field(default_factory=dict)
    player_teams: dict[str, str] = field(default_factory=dict)
    rim_center: Point2DResponse | None = None
    made: bool = False
    missed: bool = False
    dead_ball: bool = False


class PossessionStateMachine:
    STATES = {"UNKNOWN", "CONTROLLED", "PASS_FLIGHT", "SHOT_FLIGHT", "LOOSE_BALL", "DEAD_BALL"}

    def __init__(
        self,
        *,
        control_radius_px: float = 55.0,
        shot_upward_velocity_px: float = 7.0,
        max_unknown_frames: int = 12,
    ) -> None:
        self.control_radius_px = control_radius_px
        self.shot_upward_velocity_px = shot_upward_velocity_px
        self.max_unknown_frames = max_unknown_frames
        self.state = "UNKNOWN"
        self.team_id: str | None = None
        self.player_id: str | None = None
        self.last_ball: tuple[int, Point2DResponse] | None = None
        self.last_control_frame: int | None = None
        self.transitions: list[PossessionTransitionResponse] = []

    def update(self, observation: PossessionObservation) -> PossessionTransitionResponse | None:
        target_state, player_id, team_id, confidence, evidence = self._infer(observation)
        if observation.ball is not None:
            self.last_ball = (observation.frame, observation.ball)
        if target_state == "CONTROLLED":
            self.last_control_frame = observation.frame
        if target_state == self.state and player_id == self.player_id and team_id == self.team_id:
            return None
        transition = PossessionTransitionResponse(
            frame=observation.frame,
            from_state=self.state,
            to_state=target_state,
            team_id=team_id,
            player_id=player_id,
            confidence=confidence,
            evidence=evidence,
        )
        self.state = target_state
        self.player_id = player_id
        self.team_id = team_id
        self.transitions.append(transition)
        return transition

    def _infer(
        self, observation: PossessionObservation
    ) -> tuple[str, str | None, str | None, float, list[str]]:
        if observation.dead_ball:
            return "DEAD_BALL", None, None, 1.0, ["dead_ball_signal"]
        if observation.made:
            return "DEAD_BALL", None, None, 0.98, ["made_shot_signal"]
        if observation.missed and self.state == "SHOT_FLIGHT":
            return "LOOSE_BALL", None, None, 0.92, ["missed_shot_signal"]

        nearest = _nearest_hand(observation)
        if nearest is not None and nearest[0] <= self.control_radius_px:
            distance, player_id = nearest
            team_id = observation.player_teams.get(player_id)
            confidence = max(0.5, min(0.99, 1.0 - distance / (self.control_radius_px * 2.0)))
            return "CONTROLLED", player_id, team_id, confidence, [f"ball_hand_distance_px={distance:.2f}"]

        if observation.ball is None:
            if self.last_control_frame is not None and observation.frame - self.last_control_frame <= self.max_unknown_frames:
                return self.state, self.player_id, self.team_id, 0.25, ["ball_temporarily_occluded"]
            return "UNKNOWN", None, None, 0.0, ["ball_not_visible"]

        if self.state == "CONTROLLED":
            velocity_y = self._vertical_velocity(observation)
            if velocity_y <= -self.shot_upward_velocity_px and _toward_rim(observation, self.last_ball):
                return "SHOT_FLIGHT", self.player_id, self.team_id, 0.82, [f"release_velocity_y={velocity_y:.2f}"]
            return "PASS_FLIGHT", self.player_id, self.team_id, 0.65, [f"release_velocity_y={velocity_y:.2f}"]
        if self.state == "PASS_FLIGHT":
            return "PASS_FLIGHT", self.player_id, self.team_id, 0.55, ["uncontrolled_ball_flight"]
        if self.state == "SHOT_FLIGHT":
            return "SHOT_FLIGHT", self.player_id, self.team_id, 0.55, ["shot_outcome_pending"]
        return "LOOSE_BALL", None, None, 0.4, ["visible_ball_without_control"]

    def _vertical_velocity(self, observation: PossessionObservation) -> float:
        if observation.ball is None or self.last_ball is None:
            return 0.0
        last_frame, last_position = self.last_ball
        elapsed = max(1, observation.frame - last_frame)
        return (observation.ball.y - last_position.y) / elapsed


def _nearest_hand(observation: PossessionObservation) -> tuple[float, str] | None:
    if observation.ball is None:
        return None
    candidates = [
        (math.hypot(hand.x - observation.ball.x, hand.y - observation.ball.y), player_id)
        for player_id, hands in observation.player_hands.items()
        for hand in hands
    ]
    return min(candidates) if candidates else None


def _toward_rim(
    observation: PossessionObservation,
    last_ball: tuple[int, Point2DResponse] | None,
) -> bool:
    if observation.rim_center is None or observation.ball is None or last_ball is None:
        return True
    previous = last_ball[1]
    old_distance = math.hypot(previous.x - observation.rim_center.x, previous.y - observation.rim_center.y)
    new_distance = math.hypot(observation.ball.x - observation.rim_center.x, observation.ball.y - observation.rim_center.y)
    return new_distance < old_distance
