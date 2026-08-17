"""Count-only upper-bound diagnostics for sealed blind-game predictions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from app.analysis.official_evaluation import AUTOMATIC_CONFIRMED_STATUSES
from app.analysis.schemas import GameEventResponse

OFFICIAL_ACTION_EVENT_TYPES = {
    "made shot": "field_goal_attempt",
    "missed shot": "field_goal_attempt",
    "rebound": "rebound",
    "free throw": "free_throw_attempt",
    "turnover": "turnover",
    "foul": "foul",
}
DERIVED_BOX_SCORE_EVENT_TYPES = ("assist", "steal", "block")
COMPLETE_EVENT_TYPES = tuple(
    sorted(set(OFFICIAL_ACTION_EVENT_TYPES.values()) | set(DERIVED_BOX_SCORE_EVENT_TYPES))
)


def count_only_upper_bound(
    *,
    truth_counts: Mapping[str, int],
    prediction_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Return the best possible one-to-one F1 before time/identity checks.

    Matching every possible event by type is deliberately optimistic.  A score
    below the acceptance target here proves that stricter temporal, actor, team,
    outcome and shot-value matching cannot pass.
    """

    event_types = sorted(set(truth_counts) | set(prediction_counts))
    by_event_type: dict[str, dict[str, float | int]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for event_type in event_types:
        truth = max(0, int(truth_counts.get(event_type, 0)))
        predicted = max(0, int(prediction_counts.get(event_type, 0)))
        tp = min(truth, predicted)
        fp = predicted - tp
        fn = truth - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        by_event_type[event_type] = _upper_bound_metrics(tp=tp, fp=fp, fn=fn)
    return {
        "metrics": _upper_bound_metrics(tp=total_tp, fp=total_fp, fn=total_fn),
        "by_event_type": by_event_type,
    }


def build_blind_count_audit(
    *,
    prediction_bundle_sha256: str,
    events: Iterable[GameEventResponse],
    official_payload: Mapping[str, Any],
    target_f1: float,
) -> dict[str, Any]:
    """Build a diagnostic report without claiming strict acceptance accuracy."""

    if not 0 < target_f1 <= 1:
        raise ValueError("target_f1 must be in (0, 1]")
    event_list = list(events)
    official_counts = _official_event_counts(official_payload)
    candidate_counts = Counter(event.event_type for event in event_list)
    automatic_counts = Counter(
        event.event_type for event in event_list if event.status in AUTOMATIC_CONFIRMED_STATUSES
    )
    evaluation_scope = sorted(candidate_counts)
    scoped_truth = {event_type: official_counts.get(event_type, 0) for event_type in evaluation_scope}
    candidate_upper_bound = count_only_upper_bound(
        truth_counts=scoped_truth,
        prediction_counts=candidate_counts,
    )
    automatic_upper_bound = count_only_upper_bound(
        truth_counts=scoped_truth,
        prediction_counts=automatic_counts,
    )
    complete_candidate_upper_bound = count_only_upper_bound(
        truth_counts=official_counts,
        prediction_counts=candidate_counts,
    )
    complete_automatic_upper_bound = count_only_upper_bound(
        truth_counts=official_counts,
        prediction_counts=automatic_counts,
    )
    covered = sorted(set(candidate_counts) & set(COMPLETE_EVENT_TYPES))
    missing = sorted(set(COMPLETE_EVENT_TYPES) - set(candidate_counts))
    play_by_play = official_payload.get("playByPlay") or {}
    return {
        "schema_version": "agu.official-count-upper-bound.v1",
        "prediction_bundle_sha256": prediction_bundle_sha256,
        "official_game_id": str(play_by_play.get("gameId") or ""),
        "official_source_url": str(official_payload.get("source_url") or ""),
        "target_f1": target_f1,
        "evaluation_scope": evaluation_scope,
        "official_event_counts": official_counts,
        "candidate_geometry": {
            "prediction_counts": dict(sorted(candidate_counts.items())),
            **candidate_upper_bound,
            "target_status": _target_status(candidate_upper_bound, target_f1),
        },
        "automatic_confirmed": {
            "prediction_counts": dict(sorted(automatic_counts.items())),
            **automatic_upper_bound,
            "target_status": _target_status(automatic_upper_bound, target_f1),
        },
        "complete_scope": {
            "candidate_geometry": {
                "prediction_counts": dict(sorted(candidate_counts.items())),
                **complete_candidate_upper_bound,
                "target_status": _target_status(complete_candidate_upper_bound, target_f1),
            },
            "automatic_confirmed": {
                "prediction_counts": dict(sorted(automatic_counts.items())),
                **complete_automatic_upper_bound,
                "target_status": _target_status(complete_automatic_upper_bound, target_f1),
            },
        },
        "scope_coverage": {
            "required_event_types": list(COMPLETE_EVENT_TYPES),
            "covered_event_types": covered,
            "missing_event_types": missing,
            "covered_count": len(covered),
            "required_count": len(COMPLETE_EVENT_TYPES),
        },
        "blocker_diagnosis": diagnose_prediction_bottlenecks(
            events=event_list,
            official_counts=official_counts,
        ),
        "diagnostic_only_not_acceptance_metric": True,
        "interpretation": (
            "This count-only result assumes perfect type-compatible temporal and identity alignment. "
            "Strict one-to-one event F1 can only be equal or lower."
        ),
    }


def diagnose_prediction_bottlenecks(
    *,
    events: Iterable[GameEventResponse],
    official_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Quantify failure modes that can be diagnosed without event alignment."""

    event_list = list(events)
    prediction_counts = Counter(event.event_type for event in event_list)
    overgeneration = {
        event_type: max(
            0,
            predicted - max(0, int(official_counts.get(event_type, 0))),
        )
        for event_type, predicted in sorted(prediction_counts.items())
    }
    status_counts = Counter(event.status for event in event_list)
    shots = [event for event in event_list if event.event_type == "field_goal_attempt"]
    unknown_outcomes = sum(event.outcome in (None, "unknown") for event in shots)
    event_count = len(event_list)
    missing_team = sum(event.team_id is None for event in event_list)
    missing_player = sum(event.primary_player_id is None for event in event_list)
    reasons = Counter(event.reason.strip() for event in event_list)
    return {
        "candidate_overgeneration": {
            "false_positive_lower_bound": sum(overgeneration.values()),
            "by_event_type": overgeneration,
        },
        "confirmation_collapse": {
            "status_counts": dict(sorted(status_counts.items())),
            "automatic_confirmed_count": sum(
                count
                for status, count in status_counts.items()
                if status in AUTOMATIC_CONFIRMED_STATUSES
            ),
        },
        "shot_outcome_gap": {
            "field_goal_attempt_count": len(shots),
            "unknown_or_missing_count": unknown_outcomes,
            "known_count": len(shots) - unknown_outcomes,
            "known_rate": (len(shots) - unknown_outcomes) / len(shots) if shots else 0.0,
        },
        "causal_gate_blocks": {
            "linked_shot_not_confirmed_miss": reasons[
                "AGU causal gate: linked shot outcome is not an automatic confirmed miss"
            ],
            "rebound_after_confirmed_make": reasons[
                "AGU causal gate: no rebound may follow a confirmed made shot"
            ],
        },
        "identity_gap": {
            "event_count": event_count,
            "missing_team_id_count": missing_team,
            "missing_primary_player_id_count": missing_player,
            "team_id_coverage": (event_count - missing_team) / event_count if event_count else 0.0,
            "primary_player_id_coverage": (
                (event_count - missing_player) / event_count if event_count else 0.0
            ),
        },
    }


def _official_event_counts(official_payload: Mapping[str, Any]) -> dict[str, int]:
    play_by_play = official_payload.get("playByPlay") or {}
    actions = play_by_play.get("actions") or []
    counts: Counter[str] = Counter()
    for action in actions:
        raw_type = str((action or {}).get("actionType") or "").strip().casefold()
        event_type = OFFICIAL_ACTION_EVENT_TYPES.get(raw_type)
        if event_type is not None:
            counts[event_type] += 1
    game = official_payload.get("game") or {}
    teams = (game.get("homeTeam") or {}, game.get("awayTeam") or {})
    for event_type in DERIVED_BOX_SCORE_EVENT_TYPES:
        field = f"{event_type}s" if event_type != "steal" else "steals"
        counts[event_type] = sum(int((team.get("statistics") or {}).get(field) or 0) for team in teams)
    return {event_type: counts.get(event_type, 0) for event_type in COMPLETE_EVENT_TYPES}


def _upper_bound_metrics(*, tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive_upper_bound": tp,
        "false_positive_lower_bound": fp,
        "false_negative_lower_bound": fn,
        "precision_upper_bound": precision,
        "recall_upper_bound": recall,
        "f1_upper_bound": f1,
    }


def _target_status(result: Mapping[str, Any], target_f1: float) -> str:
    metrics = result.get("metrics") or {}
    return "passed" if float(metrics.get("f1_upper_bound") or 0.0) >= target_f1 else "failed"
