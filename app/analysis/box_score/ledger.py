from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Iterable, Mapping

from app.analysis.schemas import (
    BoxScoreReconciliationIssueResponse,
    BoxScoreReconciliationReportResponse,
    GameEventResponse,
    OfficialBoxScoreResponse,
    OfficialPlayerBoxScoreResponse,
    ReviewDecisionResponse,
)

ACCEPTED_STATUSES = {"vision_confirmed", "edge_vlm_confirmed", "codex_confirmed", "human_confirmed"}


class EventLedgerError(ValueError):
    pass


def _canonical_hash(payload: object) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EventLedger:
    """Append-only event revisions with deterministic aggregation.

    The ledger never mutates a prior revision. Re-appending an identical
    revision is idempotent; conflicting content for the same revision fails.
    """

    def __init__(self, events: Iterable[GameEventResponse] = ()) -> None:
        self._revisions: dict[str, list[GameEventResponse]] = defaultdict(list)
        for event in events:
            self.append(event)

    def append(self, event: GameEventResponse) -> bool:
        revisions = self._revisions[event.event_id]
        if revisions and event.revision <= revisions[-1].revision:
            existing = next((item for item in revisions if item.revision == event.revision), None)
            if existing is not None and _canonical_hash(existing) == _canonical_hash(event):
                return False
            raise EventLedgerError(f"conflicting or stale revision for {event.event_id}:{event.revision}")

        expected_revision = 1 if not revisions else revisions[-1].revision + 1
        if event.revision != expected_revision:
            raise EventLedgerError(
                f"non-contiguous revision for {event.event_id}: expected {expected_revision}, got {event.revision}"
            )
        if revisions and event.supersedes_revision != revisions[-1].revision:
            raise EventLedgerError(
                f"revision {event.revision} must supersede {revisions[-1].revision} for {event.event_id}"
            )
        if not revisions and event.supersedes_revision is not None:
            raise EventLedgerError("first revision cannot supersede an earlier revision")
        if event.end_frame < event.start_frame:
            raise EventLedgerError(f"negative event range for {event.event_id}")
        revisions.append(event)
        return True

    def latest(self, event_id: str) -> GameEventResponse:
        try:
            return self._revisions[event_id][-1]
        except (KeyError, IndexError) as exc:
            raise EventLedgerError(f"unknown event: {event_id}") from exc

    def latest_events(self) -> list[GameEventResponse]:
        return sorted((items[-1] for items in self._revisions.values()), key=_event_sort_key)

    def all_revisions(self) -> list[GameEventResponse]:
        """Return immutable history in deterministic event/time/revision order."""

        return sorted(
            (event for revisions in self._revisions.values() for event in revisions),
            key=lambda event: (*_event_sort_key(event), event.revision),
        )

    def apply_decision(self, decision: ReviewDecisionResponse) -> GameEventResponse:
        if decision.decision == "add":
            if self._revisions.get(decision.event_id):
                raise EventLedgerError(f"cannot add existing event: {decision.event_id}")
            if decision.expected_revision != 1:
                raise EventLedgerError("new review event must use expected_revision=1")
            updates = dict(decision.labels)
            forbidden = {"event_id", "revision", "supersedes_revision", "status"}.intersection(updates)
            if forbidden:
                raise EventLedgerError(f"new event labels cannot replace ledger fields: {sorted(forbidden)}")
            added = GameEventResponse.model_validate(
                {
                    **updates,
                    "event_id": decision.event_id,
                    "revision": 1,
                    "supersedes_revision": None,
                    "status": f"{decision.reviewer_type}_confirmed",
                    "reviewer": f"{decision.reviewer_type}:{decision.reviewer}",
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                }
            )
            self.append(added)
            return added

        current = self.latest(decision.event_id)
        if current.revision != decision.expected_revision:
            raise EventLedgerError(
                f"review decision expected revision {decision.expected_revision}, current is {current.revision}"
            )
        updates = dict(decision.labels)
        forbidden = {"event_id", "revision", "supersedes_revision"}.intersection(updates)
        if forbidden:
            raise EventLedgerError(f"review labels cannot replace ledger identity fields: {sorted(forbidden)}")
        status_by_decision = {
            "confirm": f"{decision.reviewer_type}_confirmed",
            "revise": f"{decision.reviewer_type}_confirmed",
            "reject": "rejected",
            "needs_review": "needs_review",
        }
        if status_by_decision[decision.decision] == "edge_vlm_confirmed":
            updates["status"] = "edge_vlm_confirmed"
        elif status_by_decision[decision.decision] in {"codex_confirmed", "human_confirmed"}:
            updates["status"] = status_by_decision[decision.decision]
        else:
            updates["status"] = status_by_decision[decision.decision]
        updates.update(
            revision=current.revision + 1,
            supersedes_revision=current.revision,
            reviewer=f"{decision.reviewer_type}:{decision.reviewer}",
            confidence=decision.confidence,
            reason=decision.reason,
        )
        revised = current.model_copy(update=updates)
        revised = GameEventResponse.model_validate(revised.model_dump())
        self.append(revised)
        return revised

    def aggregate(
        self,
        *,
        expected_team_points: Mapping[str, int] | None = None,
        ruleset: str = "conservative-amateur-v1",
    ) -> OfficialBoxScoreResponse:
        events = self.latest_events()
        accepted = [event for event in events if event.status in ACCEPTED_STATUSES]
        unresolved = [event for event in events if event.status in {"candidate", "needs_review"}]
        issues = _validate_relationships(accepted)

        stats: dict[tuple[str, str], dict[str, object]] = {}
        for event in accepted:
            if not event.primary_player_id or not event.team_id:
                issues.append(
                    _issue("missing_actor", "error", f"Accepted event {event.event_id} has no player/team", event.event_id)
                )
                continue
            key = (event.team_id, event.primary_player_id)
            row = stats.setdefault(key, _empty_stats(event.primary_player_id, event.team_id))
            _apply_event(row, event)
            row["event_ids"].append(event.event_id)  # type: ignore[union-attr]

        for event in unresolved:
            if event.primary_player_id and event.team_id:
                key = (event.team_id, event.primary_player_id)
                row = stats.setdefault(key, _empty_stats(event.primary_player_id, event.team_id))
                row["unresolved_event_count"] = int(row["unresolved_event_count"]) + 1

        players = [OfficialPlayerBoxScoreResponse.model_validate(row) for _, row in sorted(stats.items())]
        team_event_points: dict[str, int] = defaultdict(int)
        for player in players:
            team_event_points[player.team_id] += player.points

        expected = {str(key): int(value) for key, value in (expected_team_points or {}).items()}
        unexplained = {
            team_id: expected_points - int(team_event_points.get(team_id, 0))
            for team_id, expected_points in expected.items()
        }
        for team_id, delta in unexplained.items():
            if delta:
                issues.append(
                    _issue(
                        "score_mismatch",
                        "error",
                        f"Team {team_id} has {delta:+d} points not reconciled by accepted events",
                    )
                )

        reconciliation = BoxScoreReconciliationReportResponse(
            valid=not any(issue.severity == "error" for issue in issues),
            accepted_event_count=len(accepted),
            unresolved_event_count=len(unresolved),
            team_event_points=dict(team_event_points),
            expected_team_points=expected,
            unexplained_points=unexplained,
            issues=issues,
        )
        status = "official" if reconciliation.valid and not unresolved else "needs_review"
        return OfficialBoxScoreResponse(
            ruleset=ruleset,
            status=status,
            players=players,
            accepted_event_ids=[event.event_id for event in accepted],
            reconciliation=reconciliation,
        )


def _event_sort_key(event: GameEventResponse) -> tuple[str, int, int, str]:
    return event.source_video_id, event.start_frame, event.end_frame, event.event_id


def _empty_stats(player_id: str, team_id: str) -> dict[str, object]:
    return {"player_id": player_id, "team_id": team_id, "event_ids": [], "unresolved_event_count": 0}


def _apply_event(row: dict[str, object], event: GameEventResponse) -> None:
    def add(field: str, amount: int = 1) -> None:
        row[field] = int(row.get(field, 0)) + amount

    if event.event_type in {"field_goal_attempt", "free_throw_attempt"}:
        if event.shot_value not in {1, 2, 3}:
            return
        if event.shot_value == 1:
            add("free_throw_attempted")
            if event.outcome == "made":
                add("free_throw_made")
        elif event.shot_value == 2:
            add("two_pt_attempted")
            add("field_goals_attempted")
            if event.outcome == "made":
                add("two_pt_made")
                add("field_goals_made")
        else:
            add("three_pt_attempted")
            add("field_goals_attempted")
            if event.outcome == "made":
                add("three_pt_made")
                add("field_goals_made")
        if event.outcome == "made":
            add("points", event.shot_value)
    elif event.event_type == "rebound":
        add("rebounds")
        if event.rebound_type == "offensive":
            add("offensive_rebounds")
        elif event.rebound_type == "defensive":
            add("defensive_rebounds")
    elif event.event_type == "assist":
        add("assists")
    elif event.event_type == "block":
        add("blocks")
    elif event.event_type == "steal":
        add("steals")
    elif event.event_type == "turnover":
        add("turnovers")
    elif event.event_type == "foul":
        add("fouls")


def _validate_relationships(events: list[GameEventResponse]) -> list[BoxScoreReconciliationIssueResponse]:
    by_id = {event.event_id: event for event in events}
    issues: list[BoxScoreReconciliationIssueResponse] = []
    for event in events:
        related = [by_id[event_id] for event_id in event.related_event_ids if event_id in by_id]
        missing = [event_id for event_id in event.related_event_ids if event_id not in by_id]
        if missing:
            issues.append(
                _issue(
                    "missing_relation",
                    "error",
                    f"Event {event.event_id} refers to unavailable accepted events: {', '.join(missing)}",
                    event.event_id,
                    *missing,
                )
            )
        if event.event_type == "assist" and not any(
            item.event_type == "field_goal_attempt" and item.outcome == "made" and item.team_id == event.team_id
            for item in related
        ):
            issues.append(_issue("invalid_assist_relation", "error", "Assist must link to a same-team made FGA", event.event_id))
        if event.event_type == "block" and not any(
            item.event_type == "field_goal_attempt" and item.team_id != event.team_id for item in related
        ):
            issues.append(_issue("invalid_block_relation", "error", "Block must link to an opponent FGA", event.event_id))
        if event.event_type == "steal" and not any(
            item.event_type == "turnover" and item.team_id != event.team_id for item in related
        ):
            issues.append(_issue("invalid_steal_relation", "error", "Steal must link to an opponent turnover", event.event_id))
        if event.event_type == "rebound" and not any(
            item.event_type in {"field_goal_attempt", "free_throw_attempt"} and item.outcome == "missed"
            for item in related
        ):
            issues.append(
                _issue(
                    "invalid_rebound_relation",
                    "error",
                    "Rebound must link to a missed field-goal or free-throw attempt",
                    event.event_id,
                )
            )
    return issues


def _issue(
    code: str,
    severity: str,
    message: str,
    *event_ids: str,
) -> BoxScoreReconciliationIssueResponse:
    return BoxScoreReconciliationIssueResponse(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        event_ids=list(event_ids),
    )
