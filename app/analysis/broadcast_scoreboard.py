from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse

_DIGITS = re.compile(r"^\d{1,3}$")
_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_LIVE_PERIOD = re.compile(r"(?:1ST|2ND|3RD|4TH|OT)")


class OCREngine(Protocol):
    def __call__(self, image: np.ndarray) -> tuple[Sequence[Sequence[Any]] | None, Any]: ...


@dataclass(frozen=True)
class BroadcastScoreboardRead:
    frame: int
    scores: Mapping[str, int]
    confidence: float
    source: str = "rapidocr_broadcast_scoreboard_v1"


@dataclass(frozen=True)
class CandidateScoreDelta:
    event_id: str
    team_id: str
    points: int
    before_frame: int
    after_frame: int
    before_scores: Mapping[str, int]
    after_scores: Mapping[str, int]
    confidence: float
    unresolved_team_ids: tuple[str, ...] = ()
    score_resolution_method: str = "full_score_state_consensus_v1"


@dataclass(frozen=True)
class ScoreboardDelta:
    team_id: str
    points: int
    before_frame: int
    after_frame: int
    before_scores: Mapping[str, int]
    after_scores: Mapping[str, int]
    confidence: float


class BroadcastScoreboardReader:
    """Read a lower-third broadcast score bug using expected roster team IDs."""

    method = "rapidocr_broadcast_scoreboard_v3"

    def __init__(
        self,
        team_ids: Sequence[str],
        *,
        confidence_threshold: float = 0.75,
        crop_top_ratio: float = 0.58,
        ocr: OCREngine | None = None,
    ) -> None:
        normalized = tuple(_normalize_token(value) for value in team_ids)
        if len(normalized) != 2 or len(set(normalized)) != 2 or any(not value for value in normalized):
            raise ValueError("broadcast scoreboard reader requires exactly two distinct team IDs")
        if not 0.0 <= crop_top_ratio < 1.0:
            raise ValueError("crop_top_ratio must be in [0,1)")
        self.team_ids = normalized
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.crop_top_ratio = float(crop_top_ratio)
        if ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            ocr = RapidOCR()
        self._ocr = ocr

    def read(self, image_bgr: np.ndarray, *, frame: int) -> BroadcastScoreboardRead | None:
        if image_bgr.size == 0:
            return None
        crop_top = int(round(image_bgr.shape[0] * self.crop_top_ratio))
        results, _ = self._ocr(image_bgr[crop_top:])
        tokens = _ocr_tokens(results or (), minimum_confidence=self.confidence_threshold)
        parsed = _parse_team_score_row(tokens, self.team_ids, image_height=image_bgr.shape[0] - crop_top)
        if parsed is None:
            return None
        scores, confidence = parsed
        return BroadcastScoreboardRead(
            frame=frame,
            scores=scores,
            confidence=confidence,
            source=self.method,
        )


def stable_score_read(
    reads: Sequence[BroadcastScoreboardRead],
    *,
    minimum_support: int = 2,
) -> BroadcastScoreboardRead | None:
    """Return a repeated score state; tied or one-off OCR states fail closed."""

    if minimum_support <= 0:
        raise ValueError("minimum_support must be positive")
    groups: dict[tuple[tuple[str, int], ...], list[BroadcastScoreboardRead]] = {}
    for read in reads:
        key = tuple(sorted((str(team), int(score)) for team, score in read.scores.items()))
        groups.setdefault(key, []).append(read)
    eligible = [(key, values) for key, values in groups.items() if len(values) >= minimum_support]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            len(item[1]),
            sum(read.confidence for read in item[1]) / len(item[1]),
            max(read.frame for read in item[1]),
        ),
        reverse=True,
    )
    if len(eligible) > 1 and len(eligible[0][1]) == len(eligible[1][1]):
        return None
    values = eligible[0][1]
    selected = max(values, key=lambda read: (read.confidence, read.frame))
    return BroadcastScoreboardRead(
        frame=selected.frame,
        scores=dict(eligible[0][0]),
        confidence=min(read.confidence for read in values),
        source=selected.source,
    )


def extract_scoreboard_deltas(
    reads: Sequence[BroadcastScoreboardRead],
    *,
    minimum_support: int = 2,
) -> list[ScoreboardDelta]:
    """Extract monotonic per-team score transitions from a sparse OCR timeline."""

    if minimum_support <= 0:
        raise ValueError("minimum_support must be positive")
    ordered = sorted(reads, key=lambda read: read.frame)
    runs: list[list[BroadcastScoreboardRead]] = []
    for read in ordered:
        key = _score_key(read)
        if runs and _score_key(runs[-1][-1]) == key:
            runs[-1].append(read)
        else:
            runs.append([read])
    stable_runs = [run for run in runs if len(run) >= minimum_support]
    merged: list[list[BroadcastScoreboardRead]] = []
    for run in stable_runs:
        if merged and _score_key(merged[-1][-1]) == _score_key(run[-1]):
            merged[-1].extend(run)
        else:
            merged.append(run)
    deltas: list[ScoreboardDelta] = []
    baseline: list[BroadcastScoreboardRead] | None = None
    for run in merged:
        if baseline is None:
            baseline = run
            continue
        before_scores = dict(baseline[-1].scores)
        after_scores = dict(run[0].scores)
        if set(before_scores) != set(after_scores):
            continue
        changes = {team: after_scores[team] - before_scores[team] for team in before_scores}
        if any(value < 0 for value in changes.values()):
            continue
        if all(value == 0 for value in changes.values()):
            baseline.extend(run)
            continue
        positive = [(team, value) for team, value in changes.items() if value > 0]
        for team_id, points in positive:
            if points not in {1, 2, 3}:
                continue
            deltas.append(
                ScoreboardDelta(
                    team_id=team_id,
                    points=points,
                    before_frame=baseline[-1].frame,
                    after_frame=run[0].frame,
                    before_scores=before_scores,
                    after_scores=after_scores,
                    confidence=min(
                        min(read.confidence for read in baseline),
                        min(read.confidence for read in run),
                    ),
                )
            )
        baseline = run
    return deltas


def candidate_ids_for_delta(
    events: Sequence[GameEventResponse],
    delta: ScoreboardDelta,
) -> list[str]:
    """Return every shot candidate inside a score transition; callers require uniqueness."""

    candidates: list[tuple[int, str]] = []
    for event in events:
        if event.event_type not in {"field_goal_attempt", "free_throw_attempt"}:
            continue
        anchor = _event_review_anchor(event)
        if delta.before_frame <= anchor <= delta.after_frame:
            candidates.append((anchor, event.event_id))
    return [event_id for _, event_id in sorted(candidates)]


def candidate_score_delta(
    *,
    event_id: str,
    before_reads: Sequence[BroadcastScoreboardRead],
    after_reads: Sequence[BroadcastScoreboardRead],
    minimum_support: int = 2,
) -> CandidateScoreDelta | None:
    """Confirm a single-team 1/2/3 point increase around one raw-video candidate."""

    before = stable_score_read(before_reads, minimum_support=minimum_support)
    after = stable_score_read(after_reads, minimum_support=minimum_support)
    if before is None or after is None or set(before.scores) != set(after.scores):
        return None
    changes = {
        team_id: int(after.scores[team_id]) - int(before.scores[team_id])
        for team_id in before.scores
    }
    positive = [(team_id, points) for team_id, points in changes.items() if points > 0]
    if len(positive) != 1 or any(points < 0 for points in changes.values()):
        return None
    team_id, points = positive[0]
    if points not in {1, 2, 3} or any(
        delta != 0 for other_team, delta in changes.items() if other_team != team_id
    ):
        return None
    return CandidateScoreDelta(
        event_id=event_id,
        team_id=team_id,
        points=points,
        before_frame=before.frame,
        after_frame=after.frame,
        before_scores=dict(before.scores),
        after_scores=dict(after.scores),
        confidence=min(before.confidence, after.confidence),
    )


def candidate_component_score_delta(
    *,
    event_id: str,
    before_reads: Sequence[BroadcastScoreboardRead],
    after_reads: Sequence[BroadcastScoreboardRead],
    minimum_support: int = 2,
) -> CandidateScoreDelta | None:
    """Resolve one scoring change while allowing a tied opponent OCR component.

    Each team score is reconciled independently. A resolved decrease, an
    implausible increase, or a second resolved team change fails closed. An
    opponent component may remain unresolved, but it can never contribute
    points or a team identity to the returned result.
    """

    if minimum_support <= 0:
        raise ValueError("minimum_support must be positive")
    before_teams = {
        str(team_id)
        for read in before_reads
        for team_id in read.scores
    }
    after_teams = {
        str(team_id)
        for read in after_reads
        for team_id in read.scores
    }
    team_ids = sorted(before_teams & after_teams)
    if len(team_ids) != 2:
        return None

    resolved: dict[str, tuple[_StableTeamScore, _StableTeamScore]] = {}
    unresolved: list[str] = []
    for team_id in team_ids:
        before = _stable_team_score(
            before_reads,
            team_id=team_id,
            minimum_support=minimum_support,
        )
        after = _stable_team_score(
            after_reads,
            team_id=team_id,
            minimum_support=minimum_support,
        )
        if before is None or after is None:
            unresolved.append(team_id)
            continue
        resolved[team_id] = (before, after)

    changes = {
        team_id: after.score - before.score
        for team_id, (before, after) in resolved.items()
    }
    if any(points < 0 for points in changes.values()):
        return None
    for team_id, points in tuple(changes.items()):
        if points > 3:
            unresolved.append(team_id)
            resolved.pop(team_id)
            changes.pop(team_id)
    positive = [
        (team_id, points)
        for team_id, points in changes.items()
        if points > 0
    ]
    if len(positive) != 1:
        return None
    team_id, points = positive[0]
    if points not in {1, 2, 3}:
        return None
    if any(
        delta != 0
        for other_team, delta in changes.items()
        if other_team != team_id
    ):
        return None

    before, after = resolved[team_id]
    supporting_scores = resolved.values()
    return CandidateScoreDelta(
        event_id=event_id,
        team_id=team_id,
        points=points,
        before_frame=before.frame,
        after_frame=after.frame,
        before_scores={
            resolved_team: values[0].score
            for resolved_team, values in resolved.items()
        },
        after_scores={
            resolved_team: values[1].score
            for resolved_team, values in resolved.items()
        },
        confidence=min(
            value.confidence
            for values in supporting_scores
            for value in values
        ),
        unresolved_team_ids=tuple(unresolved),
        score_resolution_method="independent_team_consensus_v2",
    )


def attach_score_delta(
    event: GameEventResponse,
    delta: CandidateScoreDelta,
) -> GameEventResponse:
    """Attach raw scoreboard evidence without confirming an incomplete actor."""

    if event.event_id != delta.event_id:
        raise ValueError("score delta event_id does not match the candidate")
    if event.event_type not in {"field_goal_attempt", "free_throw_attempt"}:
        raise ValueError("score deltas can only attach to shot attempts")
    evidence = EventEvidenceResponse(
        evidence_id=f"{event.event_id}:score-delta:{delta.after_frame}",
        kind="scoreboard_delta",
        source_video_id=event.source_video_id,
        start_frame=delta.before_frame,
        end_frame=delta.after_frame,
        confidence=delta.confidence,
        details={
            "team_id": delta.team_id,
            "points": delta.points,
            "before_scores": dict(delta.before_scores),
            "after_scores": dict(delta.after_scores),
            "method": BroadcastScoreboardReader.method,
            "score_resolution_method": delta.score_resolution_method,
            "unresolved_team_ids": list(delta.unresolved_team_ids),
        },
    )
    expected_type = "free_throw_attempt" if delta.points == 1 else "field_goal_attempt"
    existing_team_is_provisional = event.team_id is None or event.team_id.startswith("raw-")
    conflict = (
        (not existing_team_is_provisional and event.team_id != delta.team_id)
        or (event.outcome == "missed")
        or (event.shot_value is not None and event.shot_value != delta.points)
        or (event.event_type == "free_throw_attempt" and expected_type != event.event_type)
    )
    updates: dict[str, Any] = {
        "evidence": [*event.evidence, evidence],
        "status": "needs_review",
        "reason": (
            "scoreboard delta conflicts with existing shot evidence; leave unresolved"
            if conflict
            else "stable raw-video scoreboard delta confirms scoring semantics; actor remains unresolved"
        ),
    }
    if conflict:
        updates.update(outcome="unknown", shot_value=None)
    else:
        updates.update(
            event_type=expected_type,
            outcome="made",
            shot_value=delta.points,
            team_id=delta.team_id,
            confidence=max(event.confidence, delta.confidence),
        )
    return GameEventResponse.model_validate(event.model_copy(update=updates).model_dump())


def score_delta_event(
    delta: ScoreboardDelta,
    *,
    source_video_id: str,
    candidate_event_ids: Sequence[str] = (),
    localization_lookback_frames: int | None = None,
) -> GameEventResponse:
    """Create an unresolved scoring event when ball/rim proposals missed the play."""

    if localization_lookback_frames is not None and localization_lookback_frames <= 0:
        raise ValueError("localization_lookback_frames must be positive when configured")
    localized_start_frame = (
        max(delta.before_frame, delta.after_frame - localization_lookback_frames)
        if localization_lookback_frames is not None
        else delta.before_frame
    )
    event_id = (
        f"scoreboard-score-{delta.after_frame:09d}-{delta.team_id.lower()}-{delta.points}"
    )
    event_type = "free_throw_attempt" if delta.points == 1 else "field_goal_attempt"
    evidence = EventEvidenceResponse(
        evidence_id=f"{event_id}:delta",
        kind="scoreboard_delta",
        source_video_id=source_video_id,
        start_frame=delta.before_frame,
        end_frame=delta.after_frame,
        confidence=delta.confidence,
        details={
            "team_id": delta.team_id,
            "points": delta.points,
            "before_scores": dict(delta.before_scores),
            "after_scores": dict(delta.after_scores),
            "candidate_event_ids": list(candidate_event_ids),
            "method": BroadcastScoreboardReader.method,
            "localization_start_frame": localized_start_frame,
            "localization_lookback_frames": localization_lookback_frames,
        },
    )
    return GameEventResponse(
        event_id=event_id,
        revision=1,
        event_type=event_type,
        source_video_id=source_video_id,
        start_frame=localized_start_frame,
        end_frame=delta.after_frame,
        outcome="made",
        shot_value=delta.points,
        team_id=delta.team_id,
        status="needs_review",
        confidence=delta.confidence,
        evidence=[evidence],
        reason="stable raw-video scoreboard increase; actor and exact release remain unresolved",
    )


@dataclass(frozen=True)
class _OCRToken:
    text: str
    confidence: float
    center_x: float
    center_y: float
    x_min: float
    x_max: float


@dataclass(frozen=True)
class _StableTeamScore:
    score: int
    frame: int
    confidence: float


def _stable_team_score(
    reads: Sequence[BroadcastScoreboardRead],
    *,
    team_id: str,
    minimum_support: int,
) -> _StableTeamScore | None:
    groups: dict[int, list[BroadcastScoreboardRead]] = {}
    for read in reads:
        if team_id in read.scores:
            groups.setdefault(int(read.scores[team_id]), []).append(read)
    eligible = [
        (score, values)
        for score, values in groups.items()
        if len(values) >= minimum_support
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            len(item[1]),
            sum(read.confidence for read in item[1]) / len(item[1]),
            max(read.frame for read in item[1]),
        ),
        reverse=True,
    )
    if (
        len(eligible) > 1
        and len(eligible[0][1]) == len(eligible[1][1])
    ):
        return None
    score, values = eligible[0]
    selected = max(values, key=lambda read: (read.confidence, read.frame))
    return _StableTeamScore(
        score=score,
        frame=selected.frame,
        confidence=min(read.confidence for read in values),
    )


@dataclass(frozen=True)
class _ScorePiece:
    kind: str
    value: str | int
    confidence: float
    x_min: float
    x_max: float
    source_index: int


def _ocr_tokens(results: Sequence[Sequence[Any]], *, minimum_confidence: float) -> list[_OCRToken]:
    tokens: list[_OCRToken] = []
    for item in results:
        if len(item) < 3:
            continue
        box, text, confidence = item[:3]
        try:
            confidence_value = float(confidence)
            points = np.asarray(box, dtype=np.float32)
            center_x = float(points[:, 0].mean())
            center_y = float(points[:, 1].mean())
        except (TypeError, ValueError, IndexError):
            continue
        normalized = _normalize_token(text)
        if confidence_value < minimum_confidence or not normalized:
            continue
        tokens.append(
            _OCRToken(
                normalized,
                confidence_value,
                center_x,
                center_y,
                float(points[:, 0].min()),
                float(points[:, 0].max()),
            )
        )
    return tokens


def _parse_team_score_row(
    tokens: Sequence[_OCRToken],
    team_ids: Sequence[str],
    *,
    image_height: int,
) -> tuple[dict[str, int], float] | None:
    team_tokens = [token for token in tokens if any(team in token.text for team in team_ids)]
    if not team_tokens:
        return None
    row_tolerance = max(8.0, image_height * 0.08)
    for anchor in sorted(team_tokens, key=lambda token: -token.confidence):
        row = [token for token in tokens if abs(token.center_y - anchor.center_y) <= row_tolerance]
        nearby = [
            token
            for token in tokens
            if abs(token.center_y - anchor.center_y) <= row_tolerance * 2.5
        ]
        if not any(_LIVE_PERIOD.search(token.text) or token.text == "FINAL" for token in nearby):
            continue
        pieces: list[_ScorePiece] = []
        for token in sorted(row, key=lambda item: item.center_x):
            pieces.extend(_token_pieces(token, team_ids, source_index=id(token)))
        parsed = _score_pattern(pieces, team_ids)
        if parsed is not None:
            return parsed
    return None


def _token_pieces(
    token: _OCRToken,
    team_ids: Sequence[str],
    *,
    source_index: int,
) -> list[_ScorePiece]:
    matches = [(token.text.find(team), team) for team in team_ids if team in token.text]
    if not matches:
        if _DIGITS.fullmatch(token.text):
            return [
                _ScorePiece(
                    "score",
                    int(token.text),
                    token.confidence,
                    token.x_min,
                    token.x_max,
                    source_index,
                )
            ]
        return []
    index, team = min(matches)
    prefix = token.text[:index]
    suffix = token.text[index + len(team) :]
    pieces: list[_ScorePiece] = []
    prefix_digits = re.search(r"(\d{1,3})$", prefix)
    if prefix_digits:
        pieces.append(
            _ScorePiece(
                "score",
                int(prefix_digits.group(1)),
                token.confidence,
                token.x_min,
                token.x_max,
                source_index,
            )
        )
    pieces.append(
        _ScorePiece(
            "team",
            team,
            token.confidence,
            token.x_min,
            token.x_max,
            source_index,
        )
    )
    suffix_digits = re.match(r"^(\d{1,3})", suffix)
    if suffix_digits:
        pieces.append(
            _ScorePiece(
                "score",
                int(suffix_digits.group(1)),
                token.confidence,
                token.x_min,
                token.x_max,
                source_index,
            )
        )
    return pieces


def _score_pattern(
    pieces: Sequence[_ScorePiece],
    team_ids: Sequence[str],
) -> tuple[dict[str, int], float] | None:
    for index in range(len(pieces) - 3):
        first_team, first_score, second_team, second_score = pieces[index : index + 4]
        if (
            first_team.kind == "team"
            and first_score.kind == "score"
            and second_team.kind == "team"
            and second_score.kind == "score"
            and {str(first_team.value), str(second_team.value)} == set(team_ids)
            and _is_adjacent_score(first_team, first_score)
            and _is_adjacent_score(second_team, second_score)
        ):
            scores = {
                str(first_team.value): int(first_score.value),
                str(second_team.value): int(second_score.value),
            }
            if all(0 <= score <= 200 for score in scores.values()):
                return scores, min(item.confidence for item in pieces[index : index + 4])
    return None


def _is_adjacent_score(team: _ScorePiece, score: _ScorePiece) -> bool:
    if team.source_index == score.source_index:
        return True
    gap = score.x_min - team.x_max
    return 0.0 <= gap <= 60.0


def _normalize_token(value: object) -> str:
    return _NON_ALNUM.sub("", str(value or "").upper())


def _score_key(read: BroadcastScoreboardRead) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(team), int(score)) for team, score in read.scores.items()))


def _event_review_anchor(event: GameEventResponse) -> int:
    for evidence in event.evidence:
        value = evidence.details.get("review_anchor_frame")
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return event.outcome_frame or (event.start_frame + event.end_frame) // 2
