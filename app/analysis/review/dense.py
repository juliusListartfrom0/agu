"""Validate exhaustive raw-video review decisions and promote them to official events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, model_validator

from app.analysis.box_score import EventLedger
from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import (
    BoxScoreReconciliationIssueResponse,
    EventEvidenceResponse,
    GameEventResponse,
    OfficialBoxScoreResponse,
    RawOnlyPredictionBundleResponse,
)


class DenseReviewError(ValueError):
    """Raised when a dense review package cannot prove exhaustive raw review."""


class DenseReviewedEvent(BaseModel):
    event_id: str
    event_type: Literal[
        "field_goal_attempt",
        "free_throw_attempt",
        "rebound",
        "assist",
        "block",
        "steal",
        "turnover",
        "foul",
    ]
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    release_sec: float | None = Field(default=None, ge=0)
    outcome_sec: float | None = Field(default=None, ge=0)
    team_id: str | None = None
    primary_player_id: str | None = None
    secondary_player_id: str | None = None
    shot_value: Literal[1, 2, 3] | None = None
    outcome: Literal["made", "missed", "unknown"] | None = None
    rebound_type: Literal["offensive", "defensive", "team", "unknown"] | None = None
    related_event_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["codex_confirmed", "needs_review"] = "needs_review"
    evidence_times_sec: list[float] = Field(default_factory=list)
    supporting_window_ids: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def validate_range_and_confirmation(self) -> "DenseReviewedEvent":
        if self.end_sec < self.start_sec:
            raise ValueError("event end_sec precedes start_sec")
        if self.status != "codex_confirmed":
            return self
        if not self.primary_player_id or not self.team_id:
            raise ValueError("confirmed event requires primary_player_id and team_id")
        if self.event_type == "field_goal_attempt":
            if self.shot_value not in {2, 3} or self.outcome not in {"made", "missed"}:
                raise ValueError("confirmed field goal requires 2/3 shot_value and made/missed outcome")
        elif self.event_type == "free_throw_attempt":
            if self.shot_value != 1 or self.outcome not in {"made", "missed"}:
                raise ValueError("confirmed free throw requires shot_value=1 and made/missed outcome")
        elif self.event_type == "rebound" and self.rebound_type in {None, "unknown"}:
            raise ValueError("confirmed rebound requires offensive/defensive/team rebound_type")
        return self


class DenseWindowDecision(BaseModel):
    decision_id: str
    window_id: str
    reviewer: str
    input_sha256: str
    reviewed: Literal[True] = True
    no_event: bool = False
    events: list[DenseReviewedEvent] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def validate_exhaustive_decision(self) -> "DenseWindowDecision":
        if self.no_event == bool(self.events):
            raise ValueError("set no_event=true with no events, or provide at least one event")
        return self


@dataclass(frozen=True)
class DenseReviewImportResult:
    manifest_sha256: str
    events: list[GameEventResponse]
    reviewed_window_count: int
    total_window_count: int
    complete_video_coverage: bool


@dataclass(frozen=True)
class DenseGameReviewResult:
    bundle: RawOnlyPredictionBundleResponse
    box_score: OfficialBoxScoreResponse
    events: list[GameEventResponse]
    manifest_sha256_by_video_id: dict[str, str]
    reviewed_window_count: int
    total_window_count: int
    complete_video_coverage: bool


def import_dense_review(
    *,
    package_dir: str | Path,
    raw_video_path: str | Path,
    decisions_path: str | Path | None = None,
    require_all_windows: bool = True,
) -> DenseReviewImportResult:
    """Import append-only decisions only after validating raw and coverage provenance."""

    package = Path(package_dir)
    manifest_path = package / "manifest.json"
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != "agu.dense-codex-review.v1":
        raise DenseReviewError("unsupported dense review manifest schema")
    manifest_sha256 = _json_sha256(manifest)
    video = Path(raw_video_path)
    _verify_raw_video(video, manifest)

    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise DenseReviewError("manifest contains no review windows")
    windows_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_window in windows:
        if not isinstance(raw_window, Mapping):
            raise DenseReviewError("invalid review window")
        window_id = str(raw_window.get("window_id") or "")
        if not window_id or window_id in windows_by_id:
            raise DenseReviewError(f"duplicate or missing window_id: {window_id!r}")
        windows_by_id[window_id] = raw_window

    decision_file = Path(decisions_path) if decisions_path else package / "codex_events.jsonl"
    decisions = _load_decisions(decision_file)
    decisions_by_window: dict[str, DenseWindowDecision] = {}
    decision_ids: set[str] = set()
    for decision in decisions:
        if decision.input_sha256 != manifest_sha256:
            raise DenseReviewError(f"decision input hash mismatch: {decision.decision_id}")
        if decision.decision_id in decision_ids:
            raise DenseReviewError(f"duplicate decision_id: {decision.decision_id}")
        decision_ids.add(decision.decision_id)
        if decision.window_id not in windows_by_id:
            raise DenseReviewError(f"decision references unknown window: {decision.window_id}")
        if decision.window_id in decisions_by_window:
            raise DenseReviewError(f"window has more than one decision: {decision.window_id}")
        decisions_by_window[decision.window_id] = decision

    missing = sorted(set(windows_by_id) - set(decisions_by_window))
    if require_all_windows and missing:
        raise DenseReviewError(f"unreviewed windows remain: {', '.join(missing[:5])}")

    fps = float(_require_mapping(manifest, "raw_video").get("fps") or 0)
    if fps <= 0:
        raise DenseReviewError("manifest raw-video fps must be positive")
    events: list[GameEventResponse] = []
    event_ids: set[str] = set()
    for window_id in windows_by_id:
        decision = decisions_by_window.get(window_id)
        if decision is None:
            continue
        for reviewed in decision.events:
            if reviewed.event_id in event_ids:
                raise DenseReviewError(f"duplicate event_id: {reviewed.event_id}")
            event_ids.add(reviewed.event_id)
            events.append(
                _to_game_event(
                    reviewed,
                    decision=decision,
                    windows_by_id=windows_by_id,
                    reviewed_window_ids=set(decisions_by_window),
                    fps=fps,
                    package=package,
                    manifest_sha256=manifest_sha256,
                )
            )

    events = _ensure_steal_turnover_companions(events)

    known_ids = {event.event_id for event in events}
    unknown_relations = sorted(
        {
            relation
            for event in events
            for relation in event.related_event_ids
            if relation not in known_ids
        }
    )
    if unknown_relations:
        raise DenseReviewError(f"events reference unknown related ids: {unknown_relations}")

    coverage = _require_mapping(manifest, "coverage")
    raw_video = _require_mapping(manifest, "raw_video")
    duration = float(raw_video.get("frame_count") or 0) / fps
    start_sec = float(coverage.get("start_sec") or 0)
    end_sec = float(coverage.get("end_sec") or 0)
    gap_seconds = float(coverage.get("gap_seconds") or 0)
    manifest_covers_video = (
        start_sec <= 1 / fps
        and end_sec >= duration - 1 / fps
        and gap_seconds <= 1 / fps
    )
    complete_video_coverage = manifest_covers_video and len(decisions_by_window) == len(windows_by_id)
    return DenseReviewImportResult(
        manifest_sha256=manifest_sha256,
        events=events,
        reviewed_window_count=len(decisions_by_window),
        total_window_count=len(windows_by_id),
        complete_video_coverage=complete_video_coverage,
    )


def seal_dense_review_result(
    *,
    game_id: str,
    raw_video_path: str | Path,
    result: DenseReviewImportResult,
    config: Mapping[str, Any],
    expected_team_points: Mapping[str, int] | None = None,
) -> tuple[RawOnlyPredictionBundleResponse, OfficialBoxScoreResponse]:
    """Seal imported events and aggregate them without hiding partial coverage."""

    bundle = seal_raw_only_predictions(
        game_id=game_id,
        raw_video_paths=[raw_video_path],
        events=result.events,
        config=config,
        model_provenance={
            "reviewer": "codex_dense_raw_review",
            "review_manifest_sha256": result.manifest_sha256,
            "review_complete_video_coverage": str(result.complete_video_coverage).lower(),
            "reviewed_window_count": str(result.reviewed_window_count),
            "total_window_count": str(result.total_window_count),
        },
    )
    score = EventLedger(result.events).aggregate(expected_team_points=expected_team_points)
    if not result.complete_video_coverage:
        issue = BoxScoreReconciliationIssueResponse(
            code="incomplete_video_coverage",
            severity="error",
            message="Dense review does not cover the complete declared raw video",
        )
        score = score.model_copy(
            update={
                "status": "needs_review",
                "reconciliation": score.reconciliation.model_copy(
                    update={"valid": False, "issues": [*score.reconciliation.issues, issue]}
                ),
            }
        )
    return bundle, OfficialBoxScoreResponse.model_validate(score.model_dump())


def seal_dense_game_reviews(
    *,
    game_id: str,
    package_video_pairs: Sequence[tuple[str | Path, str | Path]],
    config: Mapping[str, Any],
    expected_team_points: Mapping[str, int] | None = None,
) -> DenseGameReviewResult:
    """Combine independently reviewed raw periods into one sealed game bundle."""

    if not package_video_pairs:
        raise DenseReviewError("at least one dense review package/video pair is required")
    raw_paths: list[Path] = []
    events: list[GameEventResponse] = []
    manifest_hashes: dict[str, str] = {}
    reviewed_window_count = 0
    total_window_count = 0
    complete_coverage = True
    event_ids: set[str] = set()
    for index, (package_dir, video_path) in enumerate(package_video_pairs, start=1):
        video_id = f"video_{index:03d}"
        imported = import_dense_review(package_dir=package_dir, raw_video_path=video_path)
        raw_paths.append(Path(video_path))
        manifest_hashes[video_id] = imported.manifest_sha256
        reviewed_window_count += imported.reviewed_window_count
        total_window_count += imported.total_window_count
        complete_coverage = complete_coverage and imported.complete_video_coverage
        for event in imported.events:
            if event.event_id in event_ids:
                raise DenseReviewError(f"duplicate event_id across raw videos: {event.event_id}")
            event_ids.add(event.event_id)
            remapped_evidence = [
                evidence.model_copy(update={"source_video_id": video_id}) for evidence in event.evidence
            ]
            events.append(
                GameEventResponse.model_validate(
                    event.model_copy(
                        update={"source_video_id": video_id, "evidence": remapped_evidence}
                    ).model_dump()
                )
            )
    bundle = seal_raw_only_predictions(
        game_id=game_id,
        raw_video_paths=raw_paths,
        events=events,
        config=config,
        model_provenance={
            "reviewer": "codex_dense_raw_review",
            "review_manifest_set_sha256": _json_sha256(manifest_hashes),
            "review_complete_video_coverage": str(complete_coverage).lower(),
            "reviewed_window_count": str(reviewed_window_count),
            "total_window_count": str(total_window_count),
        },
    )
    score = EventLedger(events).aggregate(expected_team_points=expected_team_points)
    if not complete_coverage:
        issue = BoxScoreReconciliationIssueResponse(
            code="incomplete_video_coverage",
            severity="error",
            message="At least one dense review does not cover its complete declared raw video",
        )
        score = score.model_copy(
            update={
                "status": "needs_review",
                "reconciliation": score.reconciliation.model_copy(
                    update={"valid": False, "issues": [*score.reconciliation.issues, issue]}
                ),
            }
        )
    return DenseGameReviewResult(
        bundle=bundle,
        box_score=OfficialBoxScoreResponse.model_validate(score.model_dump()),
        events=events,
        manifest_sha256_by_video_id=manifest_hashes,
        reviewed_window_count=reviewed_window_count,
        total_window_count=total_window_count,
        complete_video_coverage=complete_coverage,
    )


def _to_game_event(
    reviewed: DenseReviewedEvent,
    *,
    decision: DenseWindowDecision,
    windows_by_id: Mapping[str, Mapping[str, Any]],
    reviewed_window_ids: set[str],
    fps: float,
    package: Path,
    manifest_sha256: str,
) -> GameEventResponse:
    support_ids = list(dict.fromkeys([decision.window_id, *reviewed.supporting_window_ids]))
    unknown_support = [window_id for window_id in support_ids if window_id not in windows_by_id]
    if unknown_support:
        raise DenseReviewError(f"event {reviewed.event_id} references unknown support windows: {unknown_support}")
    unreviewed_support = [window_id for window_id in support_ids if window_id not in reviewed_window_ids]
    if unreviewed_support:
        raise DenseReviewError(f"event {reviewed.event_id} uses unreviewed support windows: {unreviewed_support}")
    support_windows = sorted(
        (windows_by_id[window_id] for window_id in support_ids),
        key=lambda item: float(item.get("start_sec") or 0),
    )
    for previous, following in zip(support_windows, support_windows[1:]):
        if float(following.get("start_sec") or 0) > float(previous.get("end_sec") or 0) + 1e-6:
            raise DenseReviewError(f"event {reviewed.event_id} support windows are not continuous")
    window_start = float(support_windows[0].get("start_sec") or 0)
    window_end = float(support_windows[-1].get("end_sec") or 0)
    times = [reviewed.start_sec, reviewed.end_sec, *reviewed.evidence_times_sec]
    times.extend(value for value in (reviewed.release_sec, reviewed.outcome_sec) if value is not None)
    if any(value < window_start - 1e-6 or value > window_end + 1e-6 for value in times):
        raise DenseReviewError(f"event {reviewed.event_id} evidence lies outside declared support windows")
    sheet_refs = [str(window.get("contact_sheet") or "") for window in support_windows]
    if any(not sheet_ref or not (package / sheet_ref).is_file() for sheet_ref in sheet_refs):
        raise DenseReviewError(f"missing contact sheet for event {reviewed.event_id}")
    start_frame = int(round(reviewed.start_sec * fps))
    end_frame = int(round(reviewed.end_sec * fps))
    evidence = EventEvidenceResponse(
        evidence_id=f"{reviewed.event_id}:dense-review",
        kind="dense_codex_raw_review",
        source_video_id="video_001",
        start_frame=start_frame,
        end_frame=end_frame,
        confidence=reviewed.confidence,
        artifact_ref=sheet_refs[0],
        details={
            "window_id": decision.window_id,
            "supporting_window_ids": support_ids,
            "supporting_contact_sheets": sheet_refs,
            "decision_id": decision.decision_id,
            "manifest_sha256": manifest_sha256,
            "evidence_times_sec": reviewed.evidence_times_sec,
        },
    )
    return GameEventResponse(
        event_id=reviewed.event_id,
        revision=1,
        event_type=reviewed.event_type,
        source_video_id="video_001",
        start_frame=start_frame,
        end_frame=end_frame,
        release_frame=int(round(reviewed.release_sec * fps)) if reviewed.release_sec is not None else None,
        outcome_frame=int(round(reviewed.outcome_sec * fps)) if reviewed.outcome_sec is not None else None,
        team_id=reviewed.team_id,
        primary_player_id=reviewed.primary_player_id,
        secondary_player_id=reviewed.secondary_player_id,
        shot_value=reviewed.shot_value,
        outcome=reviewed.outcome,
        rebound_type=reviewed.rebound_type,
        status=reviewed.status,
        confidence=reviewed.confidence,
        evidence=[evidence],
        related_event_ids=reviewed.related_event_ids,
        reviewer=f"codex:{decision.reviewer}",
        reason=reviewed.reason,
    )


def _ensure_steal_turnover_companions(events: list[GameEventResponse]) -> list[GameEventResponse]:
    """Make provisional steals reviewable without inventing accepted stats."""

    by_id = {event.event_id: event for event in events}
    result: list[GameEventResponse] = []
    for event in events:
        if event.event_type != "steal":
            result.append(event)
            continue
        linked_turnovers = [
            relation
            for relation in event.related_event_ids
            if relation in by_id and by_id[relation].event_type == "turnover"
        ]
        if linked_turnovers:
            result.append(event)
            continue
        if event.status == "codex_confirmed":
            raise DenseReviewError(f"confirmed steal requires a reviewed turnover companion: {event.event_id}")
        companion_id = f"{event.event_id}:turnover"
        if companion_id in by_id:
            raise DenseReviewError(f"generated turnover companion id already exists: {companion_id}")
        companion_evidence = [
            evidence.model_copy(
                update={
                    "evidence_id": f"{companion_id}:dense-review:{index}",
                    "details": {
                        **evidence.details,
                        "generated_causal_companion_for": event.event_id,
                    },
                }
            )
            for index, evidence in enumerate(event.evidence, start=1)
        ]
        turnover = GameEventResponse(
            event_id=companion_id,
            revision=1,
            event_type="turnover",
            source_video_id=event.source_video_id,
            start_frame=event.start_frame,
            end_frame=event.end_frame,
            outcome_frame=event.outcome_frame or event.end_frame,
            status="needs_review",
            confidence=event.confidence,
            evidence=companion_evidence,
            related_event_ids=[event.event_id],
            reviewer=event.reviewer,
            reason="provisional opponent turnover paired with steal; actor/team require the same raw review",
        )
        result.extend(
            [
                turnover,
                event.model_copy(
                    update={
                        "outcome_frame": event.outcome_frame or event.end_frame,
                        "related_event_ids": [*event.related_event_ids, companion_id],
                        "reason": f"{event.reason}; paired with provisional turnover {companion_id}",
                    }
                ),
            ]
        )
    return result


def _load_decisions(path: Path) -> list[DenseWindowDecision]:
    if not path.is_file():
        raise DenseReviewError(f"decision file does not exist: {path}")
    decisions: list[DenseWindowDecision] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decisions.append(DenseWindowDecision.model_validate_json(line))
        except Exception as exc:
            raise DenseReviewError(f"invalid decision row {line_number}: {exc}") from exc
    return decisions


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DenseReviewError(f"manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DenseReviewError("manifest must be a JSON object")
    return payload


def _verify_raw_video(path: Path, manifest: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise DenseReviewError(f"raw video does not exist: {path}")
    asset = _require_mapping(manifest, "raw_video")
    if path.name != asset.get("filename") or path.stat().st_size != int(asset.get("size_bytes") or -1):
        raise DenseReviewError("raw video identity does not match dense review manifest")
    if _file_sha256(path) != asset.get("sha256"):
        raise DenseReviewError("raw video hash does not match dense review manifest")


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise DenseReviewError(f"manifest {key} must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
