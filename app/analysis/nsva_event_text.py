"""Normalize NSVA basketball video-text intents into AGU event ontology.

NSVA intents are compact, human-written descriptions of short clips.  This
module only parses the text metadata; it never opens a video, reads a box
score, or produces runtime answers.  The resulting audit is therefore safe
for offline ontology design and VLM prompt/evaluation contracts, but not a
replacement for raw-video evidence or full-game supervision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

NSVA_EVENT_TEXT_SCHEMA = "agu.nsva-event-text-audit.v1"
_SEGMENT_SEPARATOR = re.compile(r"\s*,\s*")
_WHITESPACE = re.compile(r"\s+")
_ASSIST_MARKER = re.compile(r"\(\s*AST\s*\)|\bAST\b", re.IGNORECASE)
_MISS_PREFIX = re.compile(r"^MISS\s+", re.IGNORECASE)
_SHOT_WORDS = (
    "SHOT",
    "LAYUP",
    "DUNK",
    "HOOK",
    "FINGER ROLL",
    "FLOATING",
    "PUTBACK",
    "TURNAROUND",
    "FADEAWAY",
    "JUMPER",
)


def _canonical_json(payload: Mapping[str, Any], *, drop: str | None = None) -> bytes:
    normalized = dict(payload)
    if drop is not None:
        normalized.pop(drop, None)
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value.replace("’", "'").strip())


def _parse_segment(segment: str) -> dict[str, Any]:
    text = _normalize_text(segment)
    upper = text.upper()
    if not text:
        return {"kind": "unknown", "text": text}

    if re.search(r"\bEND OF (?:[1-4](?:ST|ND|RD|TH)|OVERTIME) PERIOD\b", upper):
        return {"kind": "period_boundary", "boundary": "end", "text": text}
    if "JUMP BALL" in upper or ("JUMP" in upper and "TIP" in upper):
        return {"kind": "jump_ball", "text": text}
    if "REBOUND" in upper:
        rebound_type = "offensive" if "OFFENSIVE" in upper else "defensive" if "DEFENSIVE" in upper else "unknown"
        return {"kind": "rebound", "rebound_type": rebound_type, "text": text}
    if "TURNOVER" in upper or "LOST BALL" in upper:
        return {"kind": "turnover", "text": text}
    if "FOUL" in upper:
        return {"kind": "foul", "text": text}
    if "EJECTION" in upper:
        return {"kind": "ejection", "text": text}
    if "VIOLATION" in upper:
        return {"kind": "violation", "text": text}
    if "STEAL" in upper:
        return {"kind": "steal", "text": text}
    if "BLOCK" in upper:
        return {"kind": "block", "text": text}
    if "INBOUND" in upper:
        return {"kind": "inbound", "text": text}

    assisted = bool(_ASSIST_MARKER.search(text))
    shot_text = _ASSIST_MARKER.sub("", text).strip()
    shot_upper = shot_text.upper()
    is_free_throw = "FREE THROW" in shot_upper or re.search(r"\bFT\b", shot_upper) is not None
    is_shot = is_free_throw or any(word in shot_upper for word in _SHOT_WORDS)
    if not is_shot:
        return {"kind": "unknown", "text": text}

    miss = bool(_MISS_PREFIX.match(shot_text))
    without_miss = _MISS_PREFIX.sub("", shot_text).strip()
    normalized_shot = without_miss.upper()
    if is_free_throw:
        shot_type = "free_throw"
        kind = "free_throw_attempt"
    elif "3PT" in normalized_shot or "3 PT" in normalized_shot:
        shot_type = "3pt"
        kind = "field_goal_attempt"
    elif "2PT" in normalized_shot or "2 PT" in normalized_shot:
        shot_type = "2pt"
        kind = "field_goal_attempt"
    else:
        shot_type = "unknown"
        kind = "field_goal_attempt"
    return {
        "kind": kind,
        "result": "miss" if miss else "make",
        "shot_type": shot_type,
        "assisted": assisted,
        "text": text,
    }


def parse_nsva_intent(intent: str) -> dict[str, Any]:
    """Parse one NSVA intent without inventing player or team identity."""

    source_intent = _normalize_text(str(intent))
    segments = [segment for segment in _SEGMENT_SEPARATOR.split(source_intent) if segment.strip()]
    events = [_parse_segment(segment) for segment in segments]
    events = [event for event in events if event["kind"] != "unknown"]
    kinds = {str(event["kind"]) for event in events}
    return {
        "source_intent": source_intent,
        "events": events,
        "contains_shot": bool(kinds & {"field_goal_attempt", "free_throw_attempt"}),
        "contains_rebound": "rebound" in kinds,
        "contains_assist": any(bool(event.get("assisted")) for event in events),
        "contains_turnover": "turnover" in kinds,
        "contains_foul": "foul" in kinds,
        "contains_steal": "steal" in kinds,
        "contains_block": "block" in kinds,
        "contains_violation": "violation" in kinds,
        "contains_ejection": "ejection" in kinds,
        "contains_unknown": len(events) != len(segments),
    }


def build_nsva_event_text_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_url: str,
    source_revision: str,
) -> dict[str, Any]:
    """Build a deterministic metadata-only NSVA event ontology audit."""

    normalized_rows: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    for raw in rows:
        video_path = _normalize_text(str(raw.get("video_path") or ""))
        intent = _normalize_text(str(raw.get("intent") or ""))
        if not video_path or not intent:
            continue
        parsed = parse_nsva_intent(intent)
        normalized_rows.append(
            {
                "video_path": video_path,
                "intent": intent,
                "events": parsed["events"],
                "contains_unknown": parsed["contains_unknown"],
            }
        )
        event_counts.update(str(event["kind"]) for event in parsed["events"])
    normalized_rows.sort(key=lambda row: (row["video_path"], row["intent"]))
    audit: dict[str, Any] = {
        "schema_version": NSVA_EVENT_TEXT_SCHEMA,
        "source_url": str(source_url),
        "source_revision": str(source_revision),
        "video_count": len(normalized_rows),
        "event_counts": dict(sorted(event_counts.items())),
        "rows": normalized_rows,
        "runtime_consumable": False,
        "training_media_eligible": False,
        "media_downloaded": False,
        "purpose": "offline_event_ontology_and_vlm_contract_only",
    }
    audit["audit_sha256"] = hashlib.sha256(_canonical_json(audit)).hexdigest()
    return audit


def verify_nsva_event_text_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the self-excluding digest and basic metadata-only boundary."""

    audit = dict(payload)
    if audit.get("schema_version") != NSVA_EVENT_TEXT_SCHEMA:
        raise ValueError("unsupported NSVA event text audit schema")
    digest = audit.get("audit_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("NSVA event text audit hash is missing or malformed")
    expected = hashlib.sha256(_canonical_json(audit, drop="audit_sha256")).hexdigest()
    if digest != expected:
        raise ValueError("NSVA event text audit hash mismatch")
    if audit.get("runtime_consumable") is not False:
        raise ValueError("NSVA event text audit must remain offline-only")
    if audit.get("training_media_eligible") is not False:
        raise ValueError("NSVA media must not be promoted by a text audit")
    return audit
