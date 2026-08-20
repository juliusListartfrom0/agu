from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, model_validator

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse

_NON_NAME = re.compile(r"[^a-z0-9]+")
_SPACE = re.compile(r"\s+")
_STATISTICAL_CONTEXT = re.compile(
    r"\b(?:averag(?:e|es|ed|ing)|season|career|tonight|game|games|has|had|with)\b"
    r".{0,18}\b(?:\d{1,2}|one|two|three|four|five|six)\b.{0,10}"
    r"\b(?:assist|assists|block|blocks|foul|fouls|rebound|rebounds|steal|steals)\b",
    re.IGNORECASE,
)
_NON_EVENT_CONTEXT = re.compile(
    r"\b(?:on the block|block out|blocking out|shot clock|rebounding team|"
    r"rebounding advantage|rebounding numbers)\b",
    re.IGNORECASE,
)
_FOUL_NON_EVENT_CONTEXT = re.compile(
    r"\b(?:foul\s+(?:line|trouble|shot|to\s+give)|no\s+foul(?:\s+called)?|"
    r"next\s+foul|early\s+fouls?|foul\s+(?:a\s+)?moment\s+ago|without\s+fouling?)\b",
    re.IGNORECASE,
)
_FIELD_GOAL_NON_EVENT_CONTEXT = re.compile(
    r"\b(?:free\s+throws?|(?:at|to)\s+the\s+(?:foul\s+)?line|"
    r"technical|"
    r"miss(?:ed|es|ing)?\s+(?:that|the)\s+call|"
    r"hit\s+with\s+(?:a|the|his|her)\s+(?:foul|technical)|"
    r"(?:miss(?:ed|es|ing)?|connects?)\s+(?:on\s+)?the\s+"
    r"(?:first|second|third)|"
    r"hits?\s+move|hasn['’]?t\b.{0,20}\bhit|"
    r"unless\b.{0,30}\bhits?|"
    r"(?:has|had)\s+(?:already\s+)?(?:hit|made|missed)\b.{0,30}\b"
    r"(?:earlier|tonight|last\s+game)|"
    r"missed\s+(?:\d+|two|three|four|five|six)\s+(?:jump\s+)?shots?)\b",
    re.IGNORECASE,
)
_MADE_SHOT_CUE = re.compile(
    r"\b(?:hits?|makes|connects?|bur(?:y|ies|ied)|"
    r"knocks?\s+(?:it\s+)?(?:down|in)|puts?\s+it\s+in|"
    r"count\s+it|that['’]?s\s+good)\b",
    re.IGNORECASE,
)
_MISSED_SHOT_CUE = re.compile(
    r"\b(?:miss(?:es|ed|ing)?|won['’]?t\s+go|no\s+good|"
    r"shot\s+short|off\s+target)\b",
    re.IGNORECASE,
)

_ACTION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "rebound": (
        re.compile(
            r"\b(?:the|an?|another|offensive|defensive)?\s*(?:rebounds?|boards?)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\brebounded\s+by\b", re.IGNORECASE),
    ),
    "assist": (
        re.compile(r"\b(?:assist|assisted)\s+(?:from|by)\b", re.IGNORECASE),
        re.compile(r"\bwith\s+(?:an?|the)\s+assist\b", re.IGNORECASE),
        re.compile(r"\bnice\s+(?:feed|pass)\b", re.IGNORECASE),
    ),
    "block": (
        re.compile(r"\bblocked\s+by\b", re.IGNORECASE),
        re.compile(r"\b(?:block|blocks)\b", re.IGNORECASE),
    ),
    "steal": (
        re.compile(r"\b(?:steal|stolen)\s+by\b", re.IGNORECASE),
        re.compile(r"\bsteals\b", re.IGNORECASE),
    ),
    "turnover": (
        re.compile(r"\bturns?\s+(?:it|the ball)\s+over\b", re.IGNORECASE),
        re.compile(r"\bturnover\b", re.IGNORECASE),
        re.compile(r"\bbad pass\b", re.IGNORECASE),
    ),
    "foul": (
        re.compile(r"\b(?:foul|fouls|fouled)\b", re.IGNORECASE),
        re.compile(r"\bwhistled\s+(?:for|on)\b", re.IGNORECASE),
    ),
    "field_goal_attempt": (
        re.compile(
            r"\b(?:hits?|makes|connects?|bur(?:y|ies|ied)|"
            r"knocks?\s+(?:it\s+)?down|puts?\s+it\s+in|"
            r"miss(?:es|ed|ing)?|won['’]?t\s+go|no\s+good|"
            r"shot\s+short|off\s+target)\b",
            re.IGNORECASE,
        ),
    ),
}


class AudioRosterPlayer(BaseModel):
    person_id: str
    team_id: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_names(self) -> "AudioRosterPlayer":
        if not _normalize_name(self.person_id):
            raise ValueError("person_id must not be empty")
        if not _normalize_name(self.team_id):
            raise ValueError("team_id must not be empty")
        if not _normalize_name(self.display_name):
            raise ValueError("display_name must not be empty")
        return self


class AudioRosterArtifact(BaseModel):
    schema_version: Literal["agu.audio-roster.v1"] = "agu.audio-roster.v1"
    role: Literal["registration"] = "registration"
    source_roster_sha256: str = ""
    players: list[AudioRosterPlayer]
    artifact_sha256: str = ""

    @model_validator(mode="after")
    def validate_people(self) -> "AudioRosterArtifact":
        ids = [player.person_id for player in self.players]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("audio roster requires unique non-empty person IDs")
        return self


class SpeechActionMention(BaseModel):
    mention_id: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    action: Literal[
        "field_goal_attempt",
        "rebound",
        "assist",
        "block",
        "foul",
        "steal",
        "turnover",
    ]
    disposition: Literal["live_action_candidate", "rejected_context"]
    speech_player_candidate_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    shot_outcome_candidate: Literal["made", "missed", "unknown"] | None = None
    text: str
    reason: str


class SpeechPlayerMention(BaseModel):
    mention_id: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    speech_player_candidate_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    text: str


class AudioEvidenceArtifact(BaseModel):
    schema_version: Literal["agu.raw-audio-evidence.v1"] = "agu.raw-audio-evidence.v1"
    raw_video_sha256: str
    roster_artifact_sha256: str
    transcript_artifact_sha256: str
    model: str
    fps: float = Field(gt=0.0)
    player_mentions: list[SpeechPlayerMention] = Field(default_factory=list)
    mentions: list[SpeechActionMention] = Field(default_factory=list)
    artifact_sha256: str = ""


def seal_audio_roster(
    players: Sequence[AudioRosterPlayer | Mapping[str, Any]],
    *,
    source_roster_sha256: str = "",
) -> AudioRosterArtifact:
    roster = AudioRosterArtifact(
        source_roster_sha256=source_roster_sha256,
        players=[AudioRosterPlayer.model_validate(player) for player in players],
    )
    payload = roster.model_dump(exclude={"artifact_sha256"})
    return roster.model_copy(update={"artifact_sha256": canonical_sha256(payload)})


def seal_audio_roster_from_registration(
    registration: Mapping[str, Any],
    *,
    source_roster_sha256: str,
) -> AudioRosterArtifact:
    """Convert a truth-free registration roster into the audio roster contract."""

    if registration.get("benchmark_answers_included") is not False:
        raise ValueError("audio roster registration must explicitly exclude benchmark answers")
    players = registration.get("players")
    if not isinstance(players, list):
        raise ValueError("registration roster must contain players")
    return seal_audio_roster(
        [
            AudioRosterPlayer(
                person_id=str(player.get("person_id") or ""),
                team_id=str(player.get("team_id") or ""),
                display_name=str(player.get("display_name") or ""),
                aliases=[
                    str(alias)
                    for alias in player.get("aliases", [])
                    if str(alias).strip()
                ],
            )
            for player in players
            if isinstance(player, Mapping)
        ],
        source_roster_sha256=source_roster_sha256,
    )


def verify_audio_roster(roster: AudioRosterArtifact) -> AudioRosterArtifact:
    expected = canonical_sha256(roster.model_dump(exclude={"artifact_sha256"}))
    if roster.artifact_sha256 != expected:
        raise ValueError("audio roster artifact hash mismatch")
    return roster


def build_audio_evidence(
    *,
    transcript_artifact: Mapping[str, Any],
    roster: AudioRosterArtifact,
    raw_video_sha256: str,
    fps: float,
) -> AudioEvidenceArtifact:
    verify_audio_roster(roster)
    verify_transcript_artifact(transcript_artifact)
    if transcript_artifact.get("video_sha256") != raw_video_sha256:
        raise ValueError("transcript is not bound to the provided raw video")
    transcript_roster_sha256 = str(transcript_artifact.get("roster_source_sha256") or "")
    allowed_roster_hashes = {roster.artifact_sha256, roster.source_roster_sha256} - {""}
    if transcript_roster_sha256 not in allowed_roster_hashes:
        raise ValueError("transcript is not bound to the provided registration roster")
    transcript = transcript_artifact.get("transcript")
    if not isinstance(transcript, Mapping) or not isinstance(transcript.get("segments"), list):
        raise ValueError("transcript artifact must contain timestamped segments")
    model = str(transcript_artifact.get("model") or "").strip()
    if not model:
        raise ValueError("transcript artifact must identify its ASR model")

    aliases = _roster_aliases(roster.players)
    mentions: list[SpeechActionMention] = []
    player_mentions: list[SpeechPlayerMention] = []
    for segment_index, segment in enumerate(transcript["segments"]):
        if not isinstance(segment, Mapping):
            continue
        text = str(segment.get("text") or "").strip()
        start_sec = max(0.0, float(segment.get("start") or 0.0))
        end_sec = max(start_sec, float(segment.get("end") or start_sec))
        confidence = _segment_confidence(segment)
        player_ids = _mentioned_people(text, aliases)
        if player_ids:
            player_key = f"{segment_index}:player:{start_sec:.3f}:{end_sec:.3f}:{text}"
            player_mentions.append(
                SpeechPlayerMention(
                    mention_id=f"speech_player_{hashlib.sha256(player_key.encode()).hexdigest()[:16]}",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    speech_player_candidate_ids=player_ids,
                    confidence=confidence,
                    text=text,
                )
            )
        for action, patterns in _ACTION_PATTERNS.items():
            if not any(pattern.search(text) for pattern in patterns):
                continue
            rejected_reason = _rejected_context(text, action)
            disposition = "rejected_context" if rejected_reason else "live_action_candidate"
            reason = rejected_reason or "live commentary action phrase; requires visual/causal confirmation"
            mention_key = f"{segment_index}:{action}:{start_sec:.3f}:{end_sec:.3f}:{text}"
            mentions.append(
                SpeechActionMention(
                    mention_id=f"speech_{hashlib.sha256(mention_key.encode()).hexdigest()[:16]}",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    action=action,
                    disposition=disposition,
                    speech_player_candidate_ids=player_ids,
                    confidence=confidence,
                    shot_outcome_candidate=(
                        _shot_outcome_candidate(text) if action == "field_goal_attempt" else None
                    ),
                    text=text,
                    reason=reason,
                )
            )
    transcript_sha256 = canonical_sha256(transcript_artifact)
    artifact = AudioEvidenceArtifact(
        raw_video_sha256=raw_video_sha256,
        roster_artifact_sha256=roster.artifact_sha256,
        transcript_artifact_sha256=transcript_sha256,
        model=model,
        fps=fps,
        player_mentions=player_mentions,
        mentions=mentions,
    )
    return artifact.model_copy(
        update={"artifact_sha256": canonical_sha256(artifact.model_dump(exclude={"artifact_sha256"}))}
    )


def verify_audio_evidence(artifact: AudioEvidenceArtifact) -> AudioEvidenceArtifact:
    expected = canonical_sha256(artifact.model_dump(exclude={"artifact_sha256"}))
    if artifact.artifact_sha256 != expected:
        raise ValueError("raw-audio evidence artifact hash mismatch")
    return artifact


def derive_audio_evidence_subset(
    artifact: AudioEvidenceArtifact,
    *,
    excluded_mention_ids: set[str],
) -> AudioEvidenceArtifact:
    verify_audio_evidence(artifact)
    derived = artifact.model_copy(
        update={
            "mentions": [
                mention
                for mention in artifact.mentions
                if mention.mention_id not in excluded_mention_ids
            ],
            "artifact_sha256": "",
        }
    )
    return derived.model_copy(
        update={"artifact_sha256": canonical_sha256(derived.model_dump(exclude={"artifact_sha256"}))}
    )


def seal_audio_action_review(
    artifact: AudioEvidenceArtifact,
    *,
    action: str,
    decisions: Sequence[Mapping[str, Any]],
    producer: str,
) -> dict[str, Any]:
    """Seal complete offline review labels without making them runtime evidence."""

    verify_audio_evidence(artifact)
    expected_ids = {
        mention.mention_id
        for mention in artifact.mentions
        if mention.action == action and mention.disposition == "live_action_candidate"
    }
    normalized = [
        {
            "mention_id": str(decision.get("mention_id") or ""),
            "label": str(decision.get("label") or ""),
            "note": str(decision.get("note") or ""),
        }
        for decision in decisions
    ]
    decision_ids = [decision["mention_id"] for decision in normalized]
    if set(decision_ids) != expected_ids or len(decision_ids) != len(set(decision_ids)):
        raise ValueError("audio action review decisions must exactly cover live candidates once")
    allowed_labels = {"live_current_event", "non_event_context", "uncertain"}
    if any(decision["label"] not in allowed_labels for decision in normalized):
        raise ValueError("audio action review label is invalid")
    if not producer.strip():
        raise ValueError("audio action review producer must not be empty")
    payload: dict[str, Any] = {
        "schema_version": "agu.audio-action-review.v1",
        "purpose": "development_evaluation_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "producer": producer,
        "raw_video_sha256": artifact.raw_video_sha256,
        "source_audio_evidence_sha256": artifact.artifact_sha256,
        "action": action,
        "reviewed_count": len(normalized),
        "decisions": sorted(normalized, key=lambda decision: decision["mention_id"]),
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return verify_audio_action_review(payload)


def verify_audio_action_review(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if claimed != canonical_sha256(artifact):
        raise ValueError("audio action review artifact hash mismatch")
    if (
        artifact.get("schema_version") != "agu.audio-action-review.v1"
        or artifact.get("purpose") != "development_evaluation_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("audio action review must remain offline development evidence")
    artifact["artifact_sha256"] = claimed
    return artifact


def speech_candidate_events(
    artifact: AudioEvidenceArtifact,
    *,
    source_video_id: str,
    include_action_types: set[str] | None = None,
) -> list[GameEventResponse]:
    """Create abstaining event candidates; speech names never become confirmed actors."""

    verify_audio_evidence(artifact)
    events: list[GameEventResponse] = []
    for mention in artifact.mentions:
        if (
            mention.disposition != "live_action_candidate"
            or include_action_types is not None
            and mention.action not in include_action_types
        ):
            continue
        start_frame = max(0, int(round(mention.start_sec * artifact.fps)))
        end_frame = max(start_frame, int(round(mention.end_sec * artifact.fps)))
        evidence = EventEvidenceResponse(
            evidence_id=f"{mention.mention_id}:audio",
            kind="raw_audio_action_candidate",
            source_video_id=source_video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            confidence=mention.confidence,
            details={
                "speech_player_candidate_ids": mention.speech_player_candidate_ids,
                "transcript_text": mention.text,
                "audio_evidence_sha256": artifact.artifact_sha256,
                "disposition": mention.disposition,
                "speech_shot_outcome_candidate": mention.shot_outcome_candidate,
                "requires_visual_confirmation": True,
            },
        )
        events.append(
            GameEventResponse(
                event_id=mention.mention_id,
                revision=1,
                event_type=mention.action,
                source_video_id=source_video_id,
                start_frame=start_frame,
                end_frame=end_frame,
                outcome="unknown" if mention.action == "field_goal_attempt" else None,
                status="needs_review",
                confidence=mention.confidence,
                evidence=[evidence],
                reason="Raw-audio candidate only; actor and event require independent visual/causal evidence.",
            )
        )
    return events


def attach_audio_mentions(
    events: Sequence[GameEventResponse],
    artifact: AudioEvidenceArtifact,
    *,
    padding_sec: float = 1.0,
) -> tuple[list[GameEventResponse], set[str]]:
    """Attach same-action speech evidence while leaving every event decision unchanged."""

    verify_audio_evidence(artifact)
    if padding_sec < 0:
        raise ValueError("audio attachment padding must be non-negative")
    padding_frames = int(round(padding_sec * artifact.fps))
    live_mentions = [
        mention for mention in artifact.mentions if mention.disposition == "live_action_candidate"
    ]
    assignments: dict[int, list[SpeechActionMention]] = {}
    attached: set[str] = set()
    for mention in live_mentions:
        mention_start = int(round(mention.start_sec * artifact.fps))
        mention_end = int(round(mention.end_sec * artifact.fps))
        candidates: list[tuple[tuple[float, ...], int]] = []
        for index, event in enumerate(events):
            if (
                mention.action != event.event_type
                or mention_start > event.end_frame + padding_frames
                or mention_end < event.start_frame - padding_frames
            ):
                continue
            overlap = max(
                0,
                min(mention_end, event.end_frame) - max(mention_start, event.start_frame),
            )
            midpoint_distance = abs(
                (mention_start + mention_end) / 2.0
                - (event.start_frame + event.end_frame) / 2.0
            )
            candidates.append(
                (
                    (
                        -float(overlap),
                        midpoint_distance,
                        float(event.end_frame - event.start_frame),
                        -float(event.confidence),
                    ),
                    index,
                )
            )
        if not candidates:
            continue
        _, best_index = min(candidates, key=lambda item: (*item[0], events[item[1]].event_id))
        assignments.setdefault(best_index, []).append(mention)
        attached.add(mention.mention_id)

    updated: list[GameEventResponse] = []
    for index, event in enumerate(events):
        matching = assignments.get(index, [])
        if not matching:
            updated.append(event)
            continue
        evidence_by_id = {item.evidence_id: item for item in event.evidence}
        for mention in matching:
            start_frame = max(0, int(round(mention.start_sec * artifact.fps)))
            end_frame = max(start_frame, int(round(mention.end_sec * artifact.fps)))
            evidence = EventEvidenceResponse(
                evidence_id=f"{event.event_id}:{mention.mention_id}:audio",
                kind="raw_audio_action_candidate",
                source_video_id=event.source_video_id,
                start_frame=start_frame,
                end_frame=end_frame,
                confidence=mention.confidence,
                details={
                    "speech_player_candidate_ids": mention.speech_player_candidate_ids,
                    "transcript_text": mention.text,
                    "audio_evidence_sha256": artifact.artifact_sha256,
                    "speech_shot_outcome_candidate": mention.shot_outcome_candidate,
                    "requires_visual_confirmation": True,
                },
            )
            evidence_by_id[evidence.evidence_id] = evidence
        updated.append(event.model_copy(update={"evidence": list(evidence_by_id.values())}))
    return updated, attached


def ranked_score_window_player_candidates(
    artifact: AudioEvidenceArtifact,
    roster: AudioRosterArtifact,
    *,
    team_id: str,
    before_frame: int,
    after_frame: int,
    expected_scoreboard_lag_sec: float,
    pre_padding_sec: float = 0.0,
    post_padding_sec: float = 0.0,
    minimum_confidence: float = 0.0,
    require_shot_action_context: bool = False,
    require_made_shot_context: bool = False,
) -> list[str]:
    """Rank same-team speech candidates near a score update; never confirm an actor."""

    verify_audio_evidence(artifact)
    verify_audio_roster(roster)
    if before_frame < 0 or after_frame < before_frame:
        raise ValueError("invalid scoreboard frame window")
    if min(expected_scoreboard_lag_sec, pre_padding_sec, post_padding_sec) < 0:
        raise ValueError("score-window timing parameters must be non-negative")
    team_by_person = {player.person_id: player.team_id for player in roster.players}
    window_start = before_frame / artifact.fps - pre_padding_sec
    window_end = after_frame / artifact.fps + post_padding_sec
    target_sec = after_frame / artifact.fps - expected_scoreboard_lag_sec
    ranked: dict[str, tuple[float, float]] = {}
    mentions: Sequence[SpeechPlayerMention | SpeechActionMention]
    if require_made_shot_context and not require_shot_action_context:
        raise ValueError("made-shot context requires shot-action context")
    if require_shot_action_context:
        mentions = [
            mention
            for mention in artifact.mentions
            if mention.disposition == "live_action_candidate"
            and mention.action == "field_goal_attempt"
            and mention.speech_player_candidate_ids
            and (not require_made_shot_context or mention.shot_outcome_candidate == "made")
        ]
    else:
        mentions = artifact.player_mentions
    for mention in mentions:
        if mention.confidence < minimum_confidence:
            continue
        if mention.start_sec > window_end or mention.end_sec < window_start:
            continue
        midpoint = (mention.start_sec + mention.end_sec) / 2.0
        for person_id in mention.speech_player_candidate_ids:
            if team_by_person.get(person_id) != team_id:
                continue
            score = (abs(midpoint - target_sec), -mention.confidence)
            if person_id not in ranked or score < ranked[person_id]:
                ranked[person_id] = score
    return [person_id for person_id, _ in sorted(ranked.items(), key=lambda item: (*item[1], item[0]))]


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def seal_transcript_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["producer"] = "agu_asr"
    sealed.pop("artifact_sha256", None)
    sealed["artifact_sha256"] = canonical_sha256(sealed)
    return sealed


def verify_transcript_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "agu.raw-audio-transcript.v1":
        raise ValueError("unsupported raw-audio transcript schema")
    if payload.get("producer") != "agu_asr":
        raise ValueError("raw-audio transcript producer must be agu_asr")
    claimed = str(payload.get("artifact_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    if not claimed or claimed != canonical_sha256(unsigned):
        raise ValueError("raw-audio transcript artifact hash mismatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_name(value: str) -> str:
    return _SPACE.sub(" ", _NON_NAME.sub(" ", str(value).lower())).strip()


def _roster_aliases(players: Sequence[AudioRosterPlayer]) -> dict[str, list[str]]:
    surname_counts = Counter(_normalize_name(player.display_name).split()[-1] for player in players)
    aliases: dict[str, list[str]] = {}
    for player in players:
        normalized = _normalize_name(player.display_name)
        values = {normalized, *(_normalize_name(alias) for alias in player.aliases)}
        surname = normalized.split()[-1]
        if surname_counts[surname] == 1 and len(surname) >= 4:
            values.add(surname)
        for alias in values:
            if alias:
                aliases.setdefault(alias, []).append(player.person_id)
    return aliases


def _mentioned_people(text: str, aliases: Mapping[str, Sequence[str]]) -> list[str]:
    normalized = f" {_normalize_name(text)} "
    matched: set[str] = set()
    for alias, person_ids in aliases.items():
        if f" {alias} " in normalized:
            matched.update(person_ids)
    return sorted(matched)


def _segment_confidence(segment: Mapping[str, Any]) -> float:
    words = segment.get("words")
    if isinstance(words, list):
        probabilities = [
            float(word["probability"])
            for word in words
            if isinstance(word, Mapping) and word.get("probability") is not None
        ]
        if probabilities:
            return max(0.0, min(1.0, sum(probabilities) / len(probabilities)))
    avg_logprob = float(segment.get("avg_logprob") or -2.0)
    return max(0.0, min(1.0, 1.0 + avg_logprob / 2.0))


def _rejected_context(text: str, action: str) -> str | None:
    if _NON_EVENT_CONTEXT.search(text):
        return "non-event basketball phrase"
    if action == "foul" and _FOUL_NON_EVENT_CONTEXT.search(text):
        return "non-event foul commentary"
    if action == "field_goal_attempt" and _FIELD_GOAL_NON_EVENT_CONTEXT.search(text):
        return "non-event field-goal commentary"
    if action in {"rebound", "assist", "block", "foul", "steal"} and _STATISTICAL_CONTEXT.search(
        text
    ):
        return "retrospective or statistical commentary"
    return None


def _shot_outcome_candidate(text: str) -> Literal["made", "missed", "unknown"]:
    made = bool(_MADE_SHOT_CUE.search(text))
    missed = bool(_MISSED_SHOT_CUE.search(text))
    if made == missed:
        return "unknown"
    return "made" if made else "missed"
