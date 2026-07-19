from __future__ import annotations

from dataclasses import dataclass

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse


@dataclass(frozen=True)
class ControlSpan:
    player_id: str
    team_id: str
    start_frame: int
    end_frame: int
    confidence: float


@dataclass(frozen=True)
class PassObservation:
    passer_id: str
    receiver_id: str
    team_id: str
    release_frame: int
    receive_frame: int
    confidence: float


@dataclass(frozen=True)
class DefensiveContact:
    defender_id: str
    defender_team_id: str
    frame: int
    confidence: float
    ball_trajectory_changed: bool


class EventGraphBuilder:
    """Build dependent basketball events from confirmed primitive evidence."""

    def __init__(self, *, source_video_id: str, ruleset: str = "conservative-amateur-v1") -> None:
        self.source_video_id = source_video_id
        self.ruleset = ruleset

    def rebound_from_miss(
        self,
        *,
        event_id: str,
        missed_shot: GameEventResponse,
        first_control: ControlSpan,
    ) -> GameEventResponse:
        if missed_shot.event_type != "field_goal_attempt" or missed_shot.outcome != "missed":
            raise ValueError("rebound requires a missed field-goal attempt")
        rebound_type = "offensive" if first_control.team_id == missed_shot.team_id else "defensive"
        confidence = min(missed_shot.confidence, first_control.confidence)
        return self._event(
            event_id,
            "rebound",
            first_control.start_frame,
            first_control.end_frame,
            first_control.team_id,
            first_control.player_id,
            confidence,
            related_event_ids=[missed_shot.event_id],
            rebound_type=rebound_type,
            reason="first stable control after confirmed missed FGA",
        )

    def block_from_contact(
        self,
        *,
        event_id: str,
        shot: GameEventResponse,
        contact: DefensiveContact,
    ) -> GameEventResponse:
        if shot.event_type != "field_goal_attempt" or shot.team_id == contact.defender_team_id:
            raise ValueError("block requires an opponent field-goal attempt")
        if not contact.ball_trajectory_changed:
            raise ValueError("block requires a post-release ball trajectory change")
        confidence = min(shot.confidence, contact.confidence)
        return self._event(
            event_id,
            "block",
            contact.frame,
            contact.frame,
            contact.defender_team_id,
            contact.defender_id,
            confidence,
            related_event_ids=[shot.event_id],
            secondary_player_id=shot.primary_player_id,
            reason="defender contact changed post-release ball trajectory",
        )

    def turnover_and_optional_steal(
        self,
        *,
        turnover_event_id: str,
        prior_control: ControlSpan,
        next_control: ControlSpan,
        defender_caused_loss: bool,
        steal_event_id: str | None = None,
    ) -> tuple[GameEventResponse, GameEventResponse | None]:
        if prior_control.team_id == next_control.team_id:
            raise ValueError("turnover requires a team possession change")
        turnover = self._event(
            turnover_event_id,
            "turnover",
            prior_control.end_frame,
            next_control.start_frame,
            prior_control.team_id,
            prior_control.player_id,
            min(prior_control.confidence, next_control.confidence),
            reason="stable live-ball control changed teams",
        )
        if not defender_caused_loss:
            return turnover, None
        if not steal_event_id:
            raise ValueError("steal_event_id is required when defender caused possession loss")
        steal = self._event(
            steal_event_id,
            "steal",
            prior_control.end_frame,
            next_control.start_frame,
            next_control.team_id,
            next_control.player_id,
            min(prior_control.confidence, next_control.confidence),
            related_event_ids=[turnover.event_id],
            secondary_player_id=prior_control.player_id,
            reason="defender caused loss and obtained subsequent live-ball control",
        )
        return turnover, steal

    def assist_from_pass(
        self,
        *,
        event_id: str,
        made_shot: GameEventResponse,
        last_pass: PassObservation,
        shooter_control_changes: int,
        max_receive_to_release_frames: int,
    ) -> GameEventResponse | None:
        if made_shot.event_type != "field_goal_attempt" or made_shot.outcome != "made":
            raise ValueError("assist requires a made field-goal attempt")
        if (
            made_shot.team_id != last_pass.team_id
            or made_shot.primary_player_id != last_pass.receiver_id
            or shooter_control_changes > 0
            or made_shot.release_frame is None
            or made_shot.release_frame - last_pass.receive_frame > max_receive_to_release_frames
        ):
            return None
        return self._event(
            event_id,
            "assist",
            last_pass.release_frame,
            last_pass.receive_frame,
            last_pass.team_id,
            last_pass.passer_id,
            min(made_shot.confidence, last_pass.confidence),
            related_event_ids=[made_shot.event_id],
            secondary_player_id=made_shot.primary_player_id,
            reason=f"last intentional pass satisfies {self.ruleset}",
        )

    def foul(
        self,
        *,
        event_id: str,
        player_id: str,
        team_id: str,
        start_frame: int,
        end_frame: int,
        confidence: float,
        evidence: list[EventEvidenceResponse],
    ) -> GameEventResponse:
        status = "vision_confirmed" if confidence >= 0.9 and evidence else "needs_review"
        return GameEventResponse(
            event_id=event_id,
            revision=1,
            event_type="foul",
            source_video_id=self.source_video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            team_id=team_id,
            primary_player_id=player_id,
            status=status,
            confidence=confidence,
            evidence=evidence,
            reason="foul evidence requires whistle/contact or review confirmation",
        )

    def _event(
        self,
        event_id: str,
        event_type: str,
        start_frame: int,
        end_frame: int,
        team_id: str,
        player_id: str,
        confidence: float,
        *,
        related_event_ids: list[str] | None = None,
        secondary_player_id: str | None = None,
        rebound_type: str | None = None,
        reason: str,
    ) -> GameEventResponse:
        status = "vision_confirmed" if confidence >= 0.9 else "needs_review"
        return GameEventResponse.model_validate(
            {
                "event_id": event_id,
                "revision": 1,
                "event_type": event_type,
                "source_video_id": self.source_video_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "outcome_frame": end_frame if event_type in {"turnover", "steal"} else None,
                "team_id": team_id,
                "primary_player_id": player_id,
                "secondary_player_id": secondary_player_id,
                "rebound_type": rebound_type,
                "status": status,
                "confidence": confidence,
                "related_event_ids": related_event_ids or [],
                "reason": reason,
            }
        )
