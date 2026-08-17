"""Audit GameCommBench basketball metadata without opening video payloads.

GameCommBench exposes clip-level basketball metadata and commentary derived
from NSVA.  This module deliberately keeps that source on the offline side of
the AGU boundary: labels are normalized for ontology and VLM-contract audits,
not treated as continuous-game truth or runtime evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.nsva_event_text import parse_nsva_intent

GCB_EVENT_AUDIT_SCHEMA = "agu.gcb-event-audit.v1"
_BACKGROUND_FIELD = re.compile(r"<(?P<field>match_name|date)>(?P<value>.*?)</(?P=field)>")
_LABEL_TO_KIND = {
    "shot": "field_goal_attempt",
    "free_throw": "free_throw_attempt",
    "rebound": "rebound",
    "foul": "foul",
    "turnover": "turnover",
    "violation": "violation",
    "jump_ball": "jump_ball",
    "ejection-other": "ejection",
    "instant_replay": "replay_marker",
    "timeout-regular": "timeout",
    "substitution": "substitution",
    "period-end": "period_boundary",
}


def _canonical_json(payload: Mapping[str, Any], *, drop: str | None = None) -> bytes:
    normalized = dict(payload)
    if drop is not None:
        normalized.pop(drop, None)
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _normalize_label_token(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower())


def parse_gcb_label(label: str) -> list[str]:
    """Map the explicit GCB composite label to AGU event kinds."""

    events: list[str] = []
    for raw_token in str(label).split(","):
        token = _normalize_label_token(raw_token)
        kind = _LABEL_TO_KIND.get(token)
        if kind is not None:
            events.append(kind)
    return events


def _game_key(background: str, file_name: str) -> str:
    fields = {match.group("field"): match.group("value").strip() for match in _BACKGROUND_FIELD.finditer(background)}
    if fields.get("match_name") or fields.get("date"):
        return f"{fields.get('match_name', '')}|{fields.get('date', '')}"
    return str(file_name).split("/", 1)[0]


def _source_events(source_text: str) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    unknown_segments = 0
    for segment in re.split(r"\s*;\s*", str(source_text)):
        if not segment.strip():
            continue
        normalized = segment.replace("_", " ").strip()
        upper = normalized.upper()
        if re.search(r"\b(?:START|END) OF .* PERIOD\b", upper):
            events.append({"kind": "period_boundary", "text": normalized})
            continue
        if "TIMEOUT" in upper:
            events.append({"kind": "timeout", "text": normalized})
            continue
        if "TECHNICAL" in upper:
            events.append({"kind": "foul", "text": normalized, "foul_type": "technical"})
            continue
        parsed = parse_nsva_intent(normalized)
        events.extend(parsed["events"])
        if parsed["contains_unknown"]:
            unknown_segments += 1
    return events, unknown_segments


def build_gcb_event_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_url: str,
    source_revision: str,
) -> dict[str, Any]:
    """Build a deterministic, metadata-only event audit."""

    explicit_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_results: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    game_keys: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    unknown_label_tokens: Counter[str] = Counter()
    unknown_segment_count = 0
    assist_rows = 0
    valid_rows = 0

    for raw in rows:
        file_name = str(raw.get("file_name") or "").strip()
        game_state = raw.get("game_state")
        commentary = raw.get("commentary")
        if not file_name or not isinstance(game_state, Mapping) or not isinstance(commentary, Mapping):
            continue
        label = str(game_state.get("label") or "").strip()
        background = str(raw.get("background") or "")
        source_text = str(commentary.get("source_commentary_text") or "").strip()
        if not label or not source_text:
            continue

        valid_rows += 1
        label_counts[label] += 1
        label_events = parse_gcb_label(label)
        explicit_counts.update(label_events)
        for raw_token in label.split(","):
            token = _normalize_label_token(raw_token)
            if token and token not in _LABEL_TO_KIND:
                unknown_label_tokens[token] += 1

        parsed_events, unknown_count = _source_events(source_text)
        source_counts.update(str(event["kind"]) for event in parsed_events)
        source_results.update(
            str(event["result"])
            for event in parsed_events
            if event.get("result") is not None
        )
        unknown_segment_count += unknown_count
        if any(bool(event.get("assisted")) for event in parsed_events):
            assist_rows += 1
        key = _game_key(background, file_name)
        game_keys.add(key)
        audit_rows.append(
            {
                "file_name": file_name,
                "game_key": key,
                "label": label,
                "events": parsed_events,
                "contains_unknown": unknown_count > 0,
            }
        )

    audit_rows.sort(key=lambda row: row["file_name"])
    audit: dict[str, Any] = {
        "schema_version": GCB_EVENT_AUDIT_SCHEMA,
        "source_url": str(source_url),
        "source_revision": str(source_revision),
        "video_count": valid_rows,
        "unique_game_count": len(game_keys),
        "label_counts": dict(sorted(label_counts.items())),
        "explicit_event_counts": dict(sorted(explicit_counts.items())),
        "source_event_counts": dict(sorted(source_counts.items())),
        "source_shot_result_counts": dict(sorted(source_results.items())),
        "source_assist_marked_rows": assist_rows,
        "source_unknown_segment_count": unknown_segment_count,
        "unknown_label_tokens": dict(sorted(unknown_label_tokens.items())),
        "rows": audit_rows,
        "media_downloaded": False,
        "runtime_consumable": False,
        "training_media_eligible": False,
        "purpose": "offline_clip_event_ontology_and_vlm_contract_only",
    }
    audit["audit_sha256"] = hashlib.sha256(_canonical_json(audit)).hexdigest()
    return audit


def verify_gcb_event_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the audit hash and its offline-only boundary."""

    audit = dict(payload)
    if audit.get("schema_version") != GCB_EVENT_AUDIT_SCHEMA:
        raise ValueError("unsupported GCB event audit schema")
    digest = audit.get("audit_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("GCB event audit hash is missing or malformed")
    expected = hashlib.sha256(_canonical_json(audit, drop="audit_sha256")).hexdigest()
    if digest != expected:
        raise ValueError("GCB event audit hash mismatch")
    if audit.get("media_downloaded") is not False:
        raise ValueError("GCB audit must not claim media was downloaded")
    if audit.get("runtime_consumable") is not False:
        raise ValueError("GCB audit must remain offline-only")
    if audit.get("training_media_eligible") is not False:
        raise ValueError("GCB metadata must not be promoted as training media")
    return audit
