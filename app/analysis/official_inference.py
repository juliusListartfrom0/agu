"""AGU-owned autonomous official-event adjudication.

Codex/human decisions intentionally do not appear in this module.  Traditional
perception proposes bounded events; AGU's configured edge VLM may confirm,
revise, reject, or leave them unresolved.  Only schema-complete automatic
events are eligible for official aggregation and autonomous evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from app.analysis.action_ownership import ActionOwnerModel
from app.analysis.box_score import EventLedger
from app.analysis.official_evaluation import (
    AUTONOMOUS_INFERENCE_MODE,
    AUTONOMOUS_PRODUCER,
    seal_raw_only_predictions,
    verify_agu_autonomous_bundle,
)
from app.analysis.official_visuals import build_temporal_contact_sheet
from app.analysis.schemas import GameEventResponse, RawOnlyPredictionBundleResponse, ReviewDecisionResponse
from app.analysis.vlm import clamp_float, encode_frames_jpeg, parse_optional_bool, parse_vlm_payload


@dataclass(frozen=True)
class OfficialVLMResult:
    event_present: bool | None
    confidence: float
    reason: str
    labels: Mapping[str, Any]
    available: bool = True


class OfficialEventReviewer(Protocol):
    name: str
    model: str

    def review(self, event: GameEventResponse, frames: Sequence[np.ndarray]) -> OfficialVLMResult: ...


FrameProvider = Callable[[GameEventResponse], Sequence[np.ndarray]]
FrameBoundsResolver = Callable[[GameEventResponse], tuple[int, int]]
FramePositionsResolver = Callable[[GameEventResponse, int], Sequence[int]]


class OllamaOfficialEventReviewer:
    """Bounded-window official event reviewer using AGU's local Ollama backend."""

    name = "ollama_official_event_v1"

    def __init__(
        self,
        *,
        model: str,
        host: str,
        timeout: float,
        image_width: int,
        max_frames: int,
        context_length: int = 16384,
        seed: int = 0,
        cache_path: Path | None = None,
        contact_sheet: bool = False,
        review_mode: str = "combined",
        frame_bounds_resolver: FrameBoundsResolver | None = None,
        frame_positions_resolver: FramePositionsResolver | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.image_width = image_width
        self.max_frames = max_frames
        self.context_length = context_length
        self.seed = seed
        self.cache_path = cache_path
        self.contact_sheet = contact_sheet
        if review_mode not in {"combined", "event_semantics", "actor_identity"}:
            raise ValueError(f"unsupported official VLM review mode: {review_mode}")
        self.review_mode = review_mode
        self.frame_bounds_resolver = frame_bounds_resolver
        self.frame_positions_resolver = frame_positions_resolver
        self._cache = _load_result_cache(cache_path)

    def review(self, event: GameEventResponse, frames: Sequence[np.ndarray]) -> OfficialVLMResult:
        selected = _evenly_select_frames(frames, self.max_frames)
        if self.contact_sheet:
            sheet = build_temporal_contact_sheet(
                selected,
                columns=3,
                cell_width=max(180, self.image_width // 3),
            )
            image_frames = [sheet] if sheet is not None else []
        else:
            image_frames = list(selected)
        images = encode_frames_jpeg(image_frames, max_width=self.image_width)
        if not images:
            return OfficialVLMResult(None, 0.0, "no event frames available", {}, available=False)
        cache_key = (
            f"{self.model}:input=readable-player-alias-v12-role-gated:mode={self.review_mode}:"
            f"sheet={self.contact_sheet}:width={self.image_width}:"
            f"frames={self.max_frames}:ctx={self.context_length}:seed={self.seed}:"
            f"event={_event_sha256(event)}:images={_encoded_images_sha256(images)}"
        )
        cached = self._cache.get(cache_key)
        if isinstance(cached, dict):
            present = cached.get("event_present")
            cached_labels = dict(cached.get("labels") or {})
            parsed_cached = {**cached_labels, "reason": str(cached.get("reason") or "")}
            if present is True:
                _recover_single_player_alias_from_reason(event, parsed_cached)
            labels = _validated_vlm_labels(event, parsed_cached)
            if self.review_mode == "actor_identity":
                labels.update(_validated_actor_role(parsed_cached))
            return OfficialVLMResult(
                event_present=present,
                confidence=float(cached.get("confidence") or 0.0),
                reason=str(cached.get("reason") or ""),
                labels=labels,
                available=bool(cached.get("available", True)),
            )
        payload = {
            "model": self.model,
            "stream": False,
            "prompt": _official_event_prompt(event, review_mode=self.review_mode),
            "images": images,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 360,
                "num_ctx": self.context_length,
                "seed": self.seed,
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                parsed, _ = parse_vlm_payload(json.loads(response.read().decode("utf-8")))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            result = OfficialVLMResult(None, 0.0, f"official VLM unavailable: {exc}", {}, available=False)
            return result
        present = parse_optional_bool(parsed.get("event_present"))
        parsed_with_frames = dict(parsed)
        if present is True:
            _recover_single_player_alias_from_reason(event, parsed_with_frames)
        selected_frame_numbers = (
            list(self.frame_positions_resolver(event, self.max_frames))
            if self.frame_positions_resolver is not None
            else []
        )
        frame_start, frame_end = (
            self.frame_bounds_resolver(event)
            if self.frame_bounds_resolver is not None
            else (event.start_frame, event.end_frame)
        )
        for source_key, target_key in (
            ("release_frame_index", "release_frame"),
            ("outcome_frame_index", "outcome_frame"),
        ):
            mapped = (
                _frame_index_to_sampled_frame(parsed.get(source_key), selected_frame_numbers)
                if selected_frame_numbers
                else _frame_index_to_absolute(parsed.get(source_key), frame_start, frame_end, len(selected))
            )
            if mapped is not None:
                parsed_with_frames[target_key] = mapped
        labels = _validated_vlm_labels(event, parsed_with_frames)
        if self.review_mode == "actor_identity":
            labels.update(_validated_actor_role(parsed_with_frames))
        result = OfficialVLMResult(
            event_present=present,
            confidence=clamp_float(parsed.get("confidence"), 0.0, 1.0, 0.0),
            reason=str(parsed.get("reason") or ""),
            labels=labels,
        )
        self._cache_result(cache_key, result)
        return result

    def _cache_result(self, key: str, result: OfficialVLMResult) -> None:
        if self.cache_path is None:
            return
        self._cache[key] = {
            "event_present": result.event_present,
            "confidence": result.confidence,
            "reason": result.reason,
            "labels": dict(result.labels),
            "available": result.available,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.cache_path)


class TwoPassOfficialEventReviewer:
    """Separate clean-frame event semantics from overlaid actor identity."""

    name = "two_pass_official_event_v1"

    def __init__(
        self,
        *,
        semantic_reviewer: OfficialEventReviewer,
        actor_reviewer: OfficialEventReviewer,
        actor_frame_provider: FrameProvider,
        action_owner_model: ActionOwnerModel | None = None,
    ) -> None:
        self.semantic_reviewer = semantic_reviewer
        self.actor_reviewer = actor_reviewer
        self.actor_frame_provider = actor_frame_provider
        self.action_owner_model = action_owner_model
        self.model = actor_reviewer.model
        defaults = {"image_width": 0, "max_frames": 0, "context_length": 0, "contact_sheet": False}
        for field, default in defaults.items():
            setattr(self, field, getattr(semantic_reviewer, field, default))

    def review(self, event: GameEventResponse, frames: Sequence[np.ndarray]) -> OfficialVLMResult:
        semantic = self.semantic_reviewer.review(event, frames)
        if semantic.event_present is not True or not semantic.available:
            return semantic
        actor_event = event.model_copy(update=dict(semantic.labels))
        actor_event = _constrain_rebound_actor_candidates(actor_event)
        if self.action_owner_model is not None:
            actor_event = apply_action_owner_prior(
                actor_event,
                model=self.action_owner_model,
                anchor_frame=actor_event.release_frame,
            )
        actor = self.actor_reviewer.review(actor_event, self.actor_frame_provider(actor_event))
        labels = dict(semantic.labels)
        actor_role = actor.labels.get("primary_actor_role")
        if actor_role != "referee":
            for field in ("primary_player_id", "secondary_player_id", "team_id"):
                if field in actor.labels:
                    labels[field] = actor.labels[field]
        confidence = semantic.confidence
        if not any(field in actor.labels for field in ("primary_player_id", "secondary_player_id")):
            confidence = min(confidence, actor.confidence)
        role_reason = "; selected alias rejected by AGU role gate as referee" if actor_role == "referee" else ""
        return OfficialVLMResult(
            event_present=True,
            confidence=confidence,
            reason=f"semantic: {semantic.reason}; actor: {actor.reason}{role_reason}",
            labels=labels,
            available=semantic.available and actor.available,
        )


def apply_action_owner_prior(
    event: GameEventResponse,
    *,
    model: ActionOwnerModel,
    anchor_frame: int | None,
) -> GameEventResponse:
    """Attach AGU traditional-model scores without replacing VLM actor validation."""

    if event.event_type != "field_goal_attempt" or anchor_frame is None:
        return event
    observations = [
        item
        for evidence in event.evidence
        for item in (evidence.details.get("candidate_player_observations") or [])
        if isinstance(item, Mapping)
    ]
    ranked = model.rank(observations, anchor_frame=anchor_frame)
    if not ranked:
        return event
    probabilities = {item.player_id: item.probability for item in ranked}
    updated_evidence = []
    for evidence in event.evidence:
        details = dict(evidence.details)
        values = details.get("candidate_player_observations")
        if isinstance(values, list):
            details["candidate_player_observations"] = [
                {
                    **dict(item),
                    **(
                        {"action_owner_probability": probabilities[str(item.get("player_id") or "")]}
                        if isinstance(item, Mapping) and str(item.get("player_id") or "") in probabilities
                        else {}
                    ),
                }
                if isinstance(item, Mapping)
                else item
                for item in values
            ]
            details["action_owner_ranked_player_ids"] = [item.player_id for item in ranked]
            details["action_owner_model_sha256"] = model.artifact["model_sha256"]
        updated_evidence.append(evidence.model_copy(update={"details": details}))
    return event.model_copy(update={"evidence": updated_evidence})


def _recover_single_player_alias_from_reason(event: GameEventResponse, parsed: dict[str, Any]) -> None:
    """Recover a schema field when the VLM names one overlaid alias in its reason."""

    aliases = candidate_player_aliases(event)
    allowed_aliases = set(aliases.values())
    primary_id = str(parsed.get("primary_player_id") or "").strip()
    primary_alias = str(parsed.get("primary_player_alias") or "").strip().upper()
    if primary_id in aliases or primary_alias in allowed_aliases:
        return
    mentioned = {
        token.upper()
        for token in re.findall(r"(?<![A-Za-z0-9])P\d{2}(?![A-Za-z0-9])", str(parsed.get("reason") or ""), re.I)
        if token.upper() in allowed_aliases
    }
    if len(mentioned) == 1:
        parsed["primary_player_alias"] = next(iter(mentioned))


def adjudicate_official_candidates(
    events: Iterable[GameEventResponse],
    *,
    reviewer: OfficialEventReviewer,
    frame_provider: FrameProvider,
    minimum_confidence: float,
    progress_callback: Callable[[int, int, GameEventResponse, OfficialVLMResult], None] | None = None,
) -> list[GameEventResponse]:
    """Apply AGU VLM decisions without allowing incomplete confirmations."""

    ledger = EventLedger(events)
    original_events = _causal_adjudication_order(ledger.latest_events())
    for index, event in enumerate(original_events, start=1):
        if event.status not in {"candidate", "needs_review", "vision_confirmed"} or event.reviewer:
            raise ValueError(f"autonomous adjudication refuses pre-reviewed event {event.event_id}: {event.status}")
        review_event = _event_with_related_context(event, ledger)
        result = reviewer.review(review_event, frame_provider(review_event))
        if progress_callback is not None:
            progress_callback(index, len(original_events), event, result)
        decision = _decision_from_vlm_result(
            event,
            result=result,
            reviewer_name=f"{reviewer.name}/{reviewer.model}",
            minimum_confidence=minimum_confidence,
            validation_event=review_event,
            related_events={
                event_id: ledger.latest(event_id)
                for event_id in event.related_event_ids
                if event_id in {item.event_id for item in ledger.latest_events()}
            },
        )
        ledger.apply_decision(decision)
    return ledger.latest_events()


def _event_with_related_context(event: GameEventResponse, ledger: EventLedger) -> GameEventResponse:
    """Expose only already-adjudicated causal parent evidence to dependent review."""

    if event.event_type != "rebound" or len(event.related_event_ids) != 1:
        return event
    try:
        parent = ledger.latest(event.related_event_ids[0])
    except KeyError:
        return event
    if parent.event_type != "field_goal_attempt" or parent.outcome != "missed":
        return event
    parent_observations = [
        dict(item)
        for evidence in parent.evidence
        for item in (evidence.details.get("candidate_player_observations") or [])
        if isinstance(item, Mapping) and _observation_at_or_after(item, parent.release_frame)
    ]
    parent_player_ids = {str(item.get("player_id") or "") for item in parent_observations if item.get("player_id")}
    parent_team_ids = {str(item.get("team_id") or "") for item in parent_observations if item.get("team_id")}
    updated_evidence = []
    for evidence in event.evidence:
        details = dict(evidence.details)
        observations = [
            dict(item) for item in (details.get("candidate_player_observations") or []) if isinstance(item, Mapping)
        ]
        observations.extend(parent_observations)
        details.update(
            {
                "candidate_player_observations": observations,
                "candidate_player_ids": sorted(
                    set(str(value) for value in (details.get("candidate_player_ids") or [])) | parent_player_ids
                ),
                "candidate_team_ids": sorted(
                    set(str(value) for value in (details.get("candidate_team_ids") or [])) | parent_team_ids
                ),
                "related_shot_release_frame": parent.release_frame,
                "related_shot_outcome_frame": parent.outcome_frame,
                "related_shot_outcome": parent.outcome,
                "related_shot_team_id": parent.team_id,
            }
        )
        updated_evidence.append(evidence.model_copy(update={"details": details}))
    return event.model_copy(update={"evidence": updated_evidence})


def _constrain_rebound_actor_candidates(event: GameEventResponse) -> GameEventResponse:
    """Apply the semantic rebound type to raw-grounded team candidates.

    Once clean frames establish offensive/defensive rebound semantics, asking
    the actor pass to choose a player from the impossible team is contradictory.
    This filter uses only the already-confirmed shot team and tracked team IDs.
    """

    if event.event_type != "rebound" or event.rebound_type not in {"offensive", "defensive"}:
        return event
    parent_team_ids = {
        str(evidence.details.get("related_shot_team_id") or "")
        for evidence in event.evidence
        if evidence.details.get("related_shot_team_id")
    }
    if len(parent_team_ids) != 1:
        return event
    parent_team = next(iter(parent_team_ids))
    player_teams = _candidate_player_teams(event)
    candidate_teams = {team for team in player_teams.values() if team}
    allowed_teams = (
        {parent_team}
        if event.rebound_type == "offensive"
        else {team for team in candidate_teams if team != parent_team}
    )
    if len(allowed_teams) != 1:
        return event
    allowed_team = next(iter(allowed_teams))
    allowed_players = {player_id for player_id, team_id in player_teams.items() if team_id == allowed_team}
    if not allowed_players:
        return event
    updated_evidence = []
    for evidence in event.evidence:
        details = dict(evidence.details)
        values = details.get("candidate_player_observations")
        if isinstance(values, list):
            details["candidate_player_observations"] = [
                item
                for item in values
                if isinstance(item, Mapping) and str(item.get("player_id") or "") in allowed_players
            ]
        details["candidate_player_ids"] = sorted(allowed_players)
        details["candidate_team_ids"] = [allowed_team]
        details["rebound_actor_team_constraint"] = allowed_team
        updated_evidence.append(evidence.model_copy(update={"details": details}))
    return event.model_copy(
        update={
            "team_id": allowed_team,
            "primary_player_id": (event.primary_player_id if event.primary_player_id in allowed_players else None),
            "evidence": updated_evidence,
        }
    )


def _observation_at_or_after(item: Mapping[str, Any], frame: int | None) -> bool:
    if frame is None:
        return True
    try:
        return int(item.get("frame", -1)) >= frame
    except (TypeError, ValueError):
        return False


def _causal_adjudication_order(events: Sequence[GameEventResponse]) -> list[GameEventResponse]:
    """Topologically process in-bundle parents before dependent candidates."""

    pending = {event.event_id: event for event in events}
    ordered: list[GameEventResponse] = []
    while pending:
        ready = [
            event
            for event in pending.values()
            if not any(parent_id in pending for parent_id in event.related_event_ids)
        ]
        if not ready:
            # Cycles are invalid causal evidence, but deterministic processing
            # lets every member fail closed through the relation gate.
            ready = list(pending.values())
        ready.sort(key=lambda item: (item.source_video_id, item.start_frame, item.event_id))
        for event in ready:
            ordered.append(event)
            pending.pop(event.event_id, None)
    return ordered


def seal_agu_autonomous_predictions(
    *,
    game_id: str,
    raw_video_paths: Sequence[str | Path],
    events: Iterable[GameEventResponse],
    config: Mapping[str, Any],
    candidate_backend: str,
    semantic_backend: str,
    semantic_model: str,
    model_provenance: Mapping[str, str] | None = None,
) -> RawOnlyPredictionBundleResponse:
    provenance = {
        "producer": AUTONOMOUS_PRODUCER,
        "inference_mode": AUTONOMOUS_INFERENCE_MODE,
        "candidate_backend": candidate_backend,
        "semantic_backend": semantic_backend,
        "semantic_model": semantic_model,
        **dict(model_provenance or {}),
    }
    bundle = seal_raw_only_predictions(
        game_id=game_id,
        raw_video_paths=raw_video_paths,
        events=events,
        config=config,
        model_provenance=provenance,
    )
    return verify_agu_autonomous_bundle(bundle)


def _decision_from_vlm_result(
    event: GameEventResponse,
    *,
    result: OfficialVLMResult,
    reviewer_name: str,
    minimum_confidence: float,
    validation_event: GameEventResponse | None = None,
    related_events: Mapping[str, GameEventResponse] | None = None,
) -> ReviewDecisionResponse:
    labels = _validated_vlm_labels(validation_event or event, result.labels)
    labels.update(_causal_secondary_labels(event, related_events or {}))
    proposed = GameEventResponse.model_validate(event.model_copy(update=labels).model_dump())
    complete = _complete_for_automatic_confirmation(proposed)
    causal_action, causal_reason = _causal_event_action(proposed, related_events or {})
    if causal_action == "reject":
        action = "reject"
        labels = {}
    elif causal_action == "needs_review":
        action = "needs_review"
        labels = {}
    elif result.event_present is False and result.available and result.confidence >= minimum_confidence:
        action = "reject"
    elif result.event_present is True and result.confidence >= minimum_confidence and complete:
        action = "revise" if labels else "confirm"
    else:
        action = "needs_review"
        labels = {}
    return ReviewDecisionResponse(
        decision_id=f"agu-vlm-{event.event_id}-{event.revision + 1}",
        event_id=event.event_id,
        expected_revision=event.revision,
        decision=action,
        reviewer_type="edge_vlm",
        reviewer=reviewer_name,
        input_sha256=_event_sha256(event),
        labels=labels,
        confidence=result.confidence,
        reason=causal_reason or result.reason or "AGU official-event VLM decision",
    )


def _causal_event_action(
    event: GameEventResponse,
    related_events: Mapping[str, GameEventResponse],
) -> tuple[str | None, str]:
    """Fail closed when a dependent event contradicts its AGU-produced parent."""

    if event.event_type in {"assist", "block", "steal"}:
        required_type = {
            "assist": "field_goal_attempt",
            "block": "field_goal_attempt",
            "steal": "turnover",
        }[event.event_type]
        parents = [
            related_events[event_id]
            for event_id in event.related_event_ids
            if event_id in related_events and related_events[event_id].event_type == required_type
        ]
        if not parents:
            return "needs_review", f"AGU causal gate: {event.event_type} has no linked {required_type}"
        confirmed = [parent for parent in parents if parent.status in {"vision_confirmed", "edge_vlm_confirmed"}]
        if event.event_type == "assist":
            if any(parent.outcome == "missed" for parent in confirmed):
                return "reject", "AGU causal gate: no assist may follow a confirmed missed shot"
            if not any(parent.outcome == "made" for parent in confirmed):
                return "needs_review", "AGU causal gate: linked shot is not an automatic confirmed make"
            if event.team_id and any(parent.team_id and parent.team_id != event.team_id for parent in confirmed):
                return "reject", "AGU causal gate: assist and made shot must belong to the same team"
        elif not confirmed:
            return "needs_review", (f"AGU causal gate: linked {required_type} is not automatically confirmed")
        elif event.team_id and any(parent.team_id == event.team_id for parent in confirmed if parent.team_id):
            return "reject", f"AGU causal gate: {event.event_type} must belong to the opponent team"
        return None, ""
    if event.event_type != "rebound":
        return None, ""
    linked_shots = [
        related_events[event_id]
        for event_id in event.related_event_ids
        if event_id in related_events
        and related_events[event_id].event_type in {"field_goal_attempt", "free_throw_attempt"}
    ]
    if not linked_shots:
        return "needs_review", "AGU causal gate: rebound has no resolved linked shot"
    if any(
        shot.status in {"vision_confirmed", "edge_vlm_confirmed"} and shot.outcome == "made" for shot in linked_shots
    ):
        return "reject", "AGU causal gate: no rebound may follow a confirmed made shot"
    if not any(
        shot.status in {"vision_confirmed", "edge_vlm_confirmed"} and shot.outcome == "missed" for shot in linked_shots
    ):
        return "needs_review", "AGU causal gate: linked shot outcome is not an automatic confirmed miss"
    confirmed_misses = [
        shot
        for shot in linked_shots
        if shot.status in {"vision_confirmed", "edge_vlm_confirmed"} and shot.outcome == "missed"
    ]
    if event.team_id and event.rebound_type in {"offensive", "defensive"}:
        expected = (
            "offensive"
            if any(shot.team_id == event.team_id for shot in confirmed_misses if shot.team_id)
            else "defensive"
        )
        if event.rebound_type != expected:
            return "reject", f"AGU causal gate: rebound team implies {expected} rebound"
    return None, ""


def _complete_for_automatic_confirmation(event: GameEventResponse) -> bool:
    if not event.primary_player_id or not event.team_id:
        return False
    if event.event_type == "field_goal_attempt":
        return event.outcome in {"made", "missed"} and event.shot_value in {2, 3}
    if event.event_type == "free_throw_attempt":
        return event.outcome in {"made", "missed"} and event.shot_value == 1
    if event.event_type == "rebound":
        return event.rebound_type in {"offensive", "defensive", "team"} and bool(event.related_event_ids)
    if event.event_type in {"assist", "block", "steal"}:
        return bool(event.secondary_player_id and event.related_event_ids)
    return True


def _causal_secondary_labels(
    event: GameEventResponse,
    related_events: Mapping[str, GameEventResponse],
) -> dict[str, Any]:
    """Derive the affected player from an automatically confirmed parent."""

    required_type = {
        "assist": "field_goal_attempt",
        "block": "field_goal_attempt",
        "steal": "turnover",
    }.get(event.event_type)
    if required_type is None:
        return {}
    parents = [
        related_events[event_id]
        for event_id in event.related_event_ids
        if event_id in related_events
        and related_events[event_id].event_type == required_type
        and related_events[event_id].status in {"vision_confirmed", "edge_vlm_confirmed"}
        and related_events[event_id].primary_player_id
    ]
    player_ids = {str(parent.primary_player_id) for parent in parents}
    if len(player_ids) != 1:
        return {}
    return {"secondary_player_id": next(iter(player_ids))}


def _validated_vlm_labels(event: GameEventResponse, parsed: Mapping[str, Any]) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    allowed_candidates = _candidate_ids(event)
    aliases = candidate_player_aliases(event)
    alias_to_player = {alias: player_id for player_id, alias in aliases.items()}
    for field in ("primary_player_id", "secondary_player_id"):
        value = str(parsed.get(field) or "").strip()
        alias = str(parsed.get(field.replace("_id", "_alias")) or "").strip().upper()
        if alias in alias_to_player:
            value = alias_to_player[alias]
        if value and value in allowed_candidates[field]:
            labels[field] = value
    player_teams = _candidate_player_teams(event)
    primary_player = labels.get("primary_player_id") or event.primary_player_id
    bound_team = player_teams.get(str(primary_player)) if primary_player else None
    parsed_team = str(parsed.get("team_id") or "").strip()
    if bound_team:
        labels["team_id"] = bound_team
    elif parsed_team and parsed_team in allowed_candidates["team_id"]:
        labels["team_id"] = parsed_team
    if event.event_type in {"field_goal_attempt", "free_throw_attempt"}:
        # A resolved ball/rim trajectory is stronger than an unconstrained
        # semantic guess and must not be overwritten by the VLM.
        if event.outcome in {"made", "missed"}:
            labels["outcome"] = event.outcome
        else:
            outcome = str(parsed.get("outcome") or "").lower()
            if outcome in {"made", "missed", "unknown"}:
                labels["outcome"] = outcome
        try:
            shot_value = int(parsed.get("shot_value"))
        except (TypeError, ValueError):
            shot_value = 0
        allowed_values = {1} if event.event_type == "free_throw_attempt" else {2, 3}
        if shot_value in allowed_values:
            if shot_value != 3 or all(
                parse_optional_bool(parsed.get(field)) is True
                for field in ("three_point_line_visible", "shooter_feet_visible", "release_beyond_arc")
            ):
                labels["shot_value"] = shot_value
        for frame_field in ("release_frame", "outcome_frame"):
            if (
                frame_field == "outcome_frame"
                and event.outcome in {"made", "missed"}
                and event.outcome_frame is not None
            ):
                continue
            try:
                frame_value = int(parsed.get(frame_field))
            except (TypeError, ValueError):
                continue
            if event.start_frame <= frame_value <= event.end_frame:
                labels[frame_field] = frame_value
    if event.event_type == "rebound":
        rebound_type = str(parsed.get("rebound_type") or "").lower()
        if rebound_type in {"offensive", "defensive", "team", "unknown"}:
            labels["rebound_type"] = rebound_type
    related = parsed.get("related_event_id")
    if isinstance(related, str) and related in set(event.related_event_ids):
        labels["related_event_ids"] = [related]
    return labels


def _validated_actor_role(parsed: Mapping[str, Any]) -> dict[str, str]:
    role = str(parsed.get("primary_actor_role") or "").strip().lower()
    if role in {"player", "referee", "non_player", "unknown"}:
        return {"primary_actor_role": role}
    return {}


def _candidate_player_teams(event: GameEventResponse) -> dict[str, str]:
    result: dict[str, str] = {}
    for evidence in event.evidence:
        observations = evidence.details.get("candidate_player_observations") or []
        if not isinstance(observations, list):
            continue
        for item in observations:
            if not isinstance(item, dict):
                continue
            player_id = str(item.get("player_id") or "").strip()
            team_id = str(item.get("team_id") or "").strip()
            if player_id and team_id:
                result[player_id] = team_id
    return result


def _candidate_ids(event: GameEventResponse) -> dict[str, set[str]]:
    candidates = {
        "team_id": {event.team_id} if event.team_id else set(),
        "primary_player_id": {event.primary_player_id} if event.primary_player_id else set(),
        "secondary_player_id": {event.secondary_player_id} if event.secondary_player_id else set(),
    }
    detail_keys = {
        "team_id": "candidate_team_ids",
        "primary_player_id": "candidate_player_ids",
        "secondary_player_id": "candidate_secondary_player_ids",
    }
    for evidence in event.evidence:
        for field, detail_key in detail_keys.items():
            values = evidence.details.get(detail_key) or []
            if isinstance(values, list):
                candidates[field].update(str(value) for value in values if value)
    return candidates


def candidate_player_aliases(event: GameEventResponse) -> dict[str, str]:
    """Return deterministic short visual aliases for bounded player candidates."""

    return {
        player_id: f"P{index:02d}"
        for index, player_id in enumerate(sorted(_candidate_ids(event)["primary_player_id"]), start=1)
    }


def _official_event_prompt(event: GameEventResponse, *, review_mode: str = "combined") -> str:
    candidates = _candidate_ids(event)
    aliases = candidate_player_aliases(event)
    player_teams = _candidate_player_teams(event)
    alias_candidates = [
        f"{alias}={player_id}({player_teams.get(player_id, 'unknown-team')})" for player_id, alias in aliases.items()
    ]
    if review_mode == "actor_identity":
        return (
            "You are AGU's basketball actor-identification model. The event was already verified in a "
            "separate clean-frame pass; do not re-evaluate whether the basket was made or missed. "
            f"Event type={event.event_type}. Review the chronological frames around release/control and "
            "identify only the green overlaid alias that performs the action. "
            f"primary_player_aliases={alias_candidates}; "
            f"secondary_player_ids={sorted(candidates['secondary_player_id'])}. "
            "Return one flat JSON object with event_present, confidence, reason, team_id, "
            "primary_player_alias and secondary_player_alias. Set event_present=true when one displayed "
            "candidate is visibly the actor. The team_id must equal the team printed beside that alias. "
            "Also return primary_actor_role as player, referee, non_player, or unknown for the selected "
            "alias. Treat team-uniform athletes as players and officials in referee clothing as referees. "
            "Use null for an unproven actor and never invent an identifier."
        )
    if review_mode == "event_semantics":
        if event.event_type == "rebound":
            related_miss = any(evidence.details.get("related_shot_outcome") == "missed" for evidence in event.evidence)
            return (
                "You are AGU's basketball rebound verification model. Review the chronological clean "
                "raw-video frames after a field-goal attempt. "
                + (
                    "The causal parent shot was already confirmed missed; do not require the full shot or "
                    "rim contact to be visible again. "
                    if related_miss
                    else ""
                )
                + "A rebound is present only when a player is first to gain stable control of the loose ball "
                "after the miss; a tip without control is not enough. Return one flat JSON object only with "
                "event_present, confidence, reason, rebound_type (offensive/defensive/team/unknown), and "
                "related_event_id. Do not return player or team identifiers."
            )
        return (
            "You are AGU's basketball event verification model. Review the chronological clean raw-video "
            f"frames for candidate event={event.event_type}. Return one flat JSON object only with "
            "event_present, confidence, reason, outcome (made/missed/unknown), shot_value (1/2/3/null), "
            "rebound_type (offensive/defensive/team/unknown), related_event_id, three_point_line_visible, "
            "shooter_feet_visible, release_beyond_arc, release_frame_index and outcome_frame_index. "
            "Do not return player or team identifiers. For a field goal, made requires visible "
            "ball-through-rim/net evidence. A three requires visible feet, line and beyond-arc geometry; "
            "otherwise return 2 only when release is visibly inside the arc, or null. Use null for any "
            "unproven field. Frame indexes are 1-based positions in the displayed chronological frames."
        )
    return (
        "You are AGU's basketball event verification model. Review the chronological raw-video frames. "
        "Verify both the event semantics and its actor. "
        "Return one flat JSON object only. Never invent a player/team identifier: choose only from the "
        f"provided candidates. Candidate event={event.event_type}; team_ids={sorted(candidates['team_id'])}; "
        f"primary_player_aliases={alias_candidates}; "
        f"secondary_player_ids={sorted(candidates['secondary_player_id'])}; "
        f"related_event_ids={event.related_event_ids}. Required keys: event_present (true/false/null), "
        "confidence (0..1), reason, team_id, primary_player_alias, secondary_player_alias, outcome "
        "(made/missed/unknown), shot_value (1/2/3/null), rebound_type "
        "(offensive/defensive/team/unknown), related_event_id. A selected team_id MUST be the team printed next "
        "to the selected short alias in the green overlay. Return an alias such as P03, not the long internal ID. "
        "For field goals, made requires a visible ball-through-rim/net "
        "result. For shot_value=3, all of the shooter's feet, the three-point line, and their spatial relationship "
        "must be visible at release; otherwise use 2 only when release is visibly inside the arc, or null. Also return "
        "three_point_line_visible, shooter_feet_visible, and release_beyond_arc as booleans/null. "
        "Also return release_frame_index and outcome_frame_index as 1-based displayed frame indexes for shots. "
        "Use empty strings/null when any label is unproven."
    )


def _evenly_select_frames(frames: Sequence[np.ndarray], maximum: int) -> list[np.ndarray]:
    if not frames or maximum <= 0:
        return []
    if len(frames) <= maximum:
        return list(frames)
    indexes = np.linspace(0, len(frames) - 1, maximum, dtype=int)
    return [frames[int(index)] for index in indexes]


def _event_sha256(event: GameEventResponse) -> str:
    payload = json.dumps(
        event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _encoded_images_sha256(images: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in images:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _frame_index_to_absolute(
    value: object,
    start_frame: int,
    end_frame: int,
    frame_count: int,
) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if frame_count <= 0 or not 1 <= index <= frame_count:
        return None
    if frame_count == 1:
        return (start_frame + end_frame) // 2
    ratio = (index - 1) / (frame_count - 1)
    return int(round(start_frame + ratio * (end_frame - start_frame)))


def _frame_index_to_sampled_frame(value: object, frame_numbers: Sequence[int]) -> int | None:
    """Map a one-based displayed-image index to its exact raw-video frame."""

    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if not 1 <= index <= len(frame_numbers):
        return None
    return int(frame_numbers[index - 1])


def _load_result_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
