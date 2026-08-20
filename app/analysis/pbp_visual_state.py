from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from app.analysis.broadcast_clock import verify_broadcast_clock_artifact

PBP_VISUAL_STATE_SCHEMA = "agu.pbp-visual-state-training.v1"

_NBA_CLOCK = re.compile(r"PT(\d{1,2})M(\d{2})\.(\d{2})S")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DETERMINATE_STATES = {"field_goal", "free_throw", "foul_only"}
_EXCLUSION_REASONS = {"ambiguous", "other", "unmatched", "no_clock_read"}


def parse_nba_clock_seconds(value: str) -> int:
    """Parse NBA's ISO-like period clock without accepting loose variants."""

    match = _NBA_CLOCK.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("invalid NBA play-by-play clock")
    minutes, seconds, centiseconds = (int(part) for part in match.groups())
    if minutes > 12 or seconds > 59 or centiseconds > 99:
        raise ValueError("NBA play-by-play clock is outside period bounds")
    total = minutes * 60 + seconds
    if total > 12 * 60:
        raise ValueError("NBA play-by-play clock is outside period bounds")
    return total


def build_pbp_visual_state_manifest(
    *,
    clock_artifact: Mapping[str, Any],
    pbp_rows: Sequence[Mapping[str, Any]],
    pbp_sha256: str,
    pbp_game_id: str,
    source_slug: str,
    allowed_clock_delta_seconds: int = 1,
    sealed_blind_video_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """Bind offline PBP state labels to OCR-clock video candidates.

    The result is deliberately non-runtime and omits player, score, and
    description fields. Every clock candidate is either labeled or excluded.
    """

    verified_clock = verify_broadcast_clock_artifact(clock_artifact)
    video_sha = _require_sha256(
        verified_clock.get("raw_video_sha256"), field="raw video"
    )
    blind_hashes = {
        _require_sha256(value, field="sealed blind video")
        for value in sealed_blind_video_sha256s
    }
    if video_sha in blind_hashes:
        raise ValueError("sealed blind video cannot be used for PBP training")
    _require_sha256(pbp_sha256, field="PBP")
    if (
        not pbp_game_id
        or not source_slug
        or isinstance(allowed_clock_delta_seconds, bool)
        or not 0 <= allowed_clock_delta_seconds <= 2
        or not pbp_rows
    ):
        raise ValueError("invalid PBP visual-state manifest configuration")

    by_period_clock: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    action_ids: set[int | str] = set()
    for raw in pbp_rows:
        if not isinstance(raw, Mapping) or str(raw.get("game_id") or "") != pbp_game_id:
            raise ValueError("PBP row does not belong to expected game")
        period = _require_positive_int(raw.get("period"), field="PBP period")
        clock_seconds = parse_nba_clock_seconds(raw.get("clock"))  # type: ignore[arg-type]
        action_id = raw.get("actionId")
        if isinstance(action_id, bool) or not isinstance(action_id, (int, str)):
            raise ValueError("invalid PBP action ID")
        if action_id in action_ids:
            raise ValueError("duplicate PBP action ID")
        action_ids.add(action_id)
        # Legacy NBA feeds leave actionType blank for some steals and blocks.
        # Retain their timestamp/action binding for audit coverage, but make
        # them non-target "other" rows without consulting descriptions.
        action_type = str(raw.get("actionType") or "").strip() or "Unknown"
        by_period_clock[(period, clock_seconds)].append(
            {
                "action_id": action_id,
                "action_type": action_type,
                "sub_type": str(raw.get("subType") or "").strip(),
            }
        )

    examples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for event in verified_clock["events"]:
        event_id = str(event["event_id"])
        anchor_frame = int(event.get("anchor_frame", -1))
        selected_read, sample_offset = _select_anchor_clock_read(event["samples"])
        if selected_read is None:
            reason = "no_clock_read"
            exclusions.append(
                {
                    "event_id": event_id,
                    "anchor_frame": anchor_frame,
                    "reason": reason,
                }
            )
            counts[reason] += 1
            continue

        period = int(selected_read["period"])
        clock_seconds = int(selected_read["clock_seconds"])
        candidates = [
            (abs(pbp_clock - clock_seconds), pbp_clock, actions)
            for (pbp_period, pbp_clock), actions in by_period_clock.items()
            if pbp_period == period
            and abs(pbp_clock - clock_seconds) <= allowed_clock_delta_seconds
        ]
        if not candidates:
            _append_exclusion(
                exclusions,
                counts,
                event_id=event_id,
                anchor_frame=anchor_frame,
                reason="unmatched",
                period=period,
                clock_seconds=clock_seconds,
            )
            continue
        minimum_delta = min(item[0] for item in candidates)
        closest = [item for item in candidates if item[0] == minimum_delta]
        if len(closest) != 1:
            _append_exclusion(
                exclusions,
                counts,
                event_id=event_id,
                anchor_frame=anchor_frame,
                reason="ambiguous",
                period=period,
                clock_seconds=clock_seconds,
            )
            continue

        delta, matched_clock, actions = closest[0]
        state = _classify_actions(actions)
        if state not in _DETERMINATE_STATES:
            _append_exclusion(
                exclusions,
                counts,
                event_id=event_id,
                anchor_frame=anchor_frame,
                reason=state,
                period=period,
                clock_seconds=clock_seconds,
            )
            continue
        examples.append(
            {
                "event_id": event_id,
                "anchor_frame": anchor_frame,
                "period": period,
                "observed_clock_seconds": clock_seconds,
                "matched_clock_seconds": matched_clock,
                "clock_delta_seconds": delta,
                "clock_sample_offset_seconds": sample_offset,
                "state": state,
                "pbp_actions": sorted(actions, key=lambda row: str(row["action_id"])),
            }
        )
        counts[state] += 1

    total_events = len(verified_clock["events"])
    determinate = sum(counts[state] for state in _DETERMINATE_STATES)
    if determinate + len(exclusions) != total_events:
        raise ValueError("PBP visual-state event coverage is incomplete")
    payload: dict[str, Any] = {
        "schema_version": PBP_VISUAL_STATE_SCHEMA,
        "purpose": "offline raw-video visual-state supervision",
        "runtime_consumable": False,
        "truth_used_for_training_only": True,
        "codex_runtime_answer_used": False,
        "raw_video_sha256": video_sha,
        "candidate_bundle_sha256": verified_clock.get("candidate_bundle_sha256"),
        "clock_artifact_sha256": verified_clock["artifact_sha256"],
        "pbp_sha256": pbp_sha256,
        "pbp_game_id": pbp_game_id,
        "source_slug": source_slug,
        "allowed_clock_delta_seconds": allowed_clock_delta_seconds,
        "examples": examples,
        "exclusions": exclusions,
        "summary": {
            "total_clock_events": total_events,
            "determinate": determinate,
            "field_goal": counts["field_goal"],
            "free_throw": counts["free_throw"],
            "foul_only": counts["foul_only"],
            "ambiguous": counts["ambiguous"],
            "other": counts["other"],
            "unmatched": counts["unmatched"],
            "no_clock_read": counts["no_clock_read"],
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return verify_pbp_visual_state_manifest(payload)


def verify_pbp_visual_state_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    if manifest.get("schema_version") != PBP_VISUAL_STATE_SCHEMA:
        raise ValueError("unsupported PBP visual-state manifest schema")
    if (
        manifest.get("runtime_consumable") is not False
        or manifest.get("truth_used_for_training_only") is not True
        or manifest.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("PBP visual-state manifest must remain training-only")
    for field in ("raw_video_sha256", "pbp_sha256", "artifact_sha256"):
        _require_sha256(manifest.get(field), field=field)
    examples = manifest.get("examples")
    exclusions = manifest.get("exclusions")
    summary = manifest.get("summary")
    if not isinstance(examples, list) or not isinstance(exclusions, list):
        raise ValueError("invalid PBP visual-state examples")
    if not isinstance(summary, Mapping):
        raise ValueError("invalid PBP visual-state summary")
    event_ids: set[str] = set()
    for row in [*examples, *exclusions]:
        if not isinstance(row, Mapping) or not str(row.get("event_id") or ""):
            raise ValueError("invalid PBP visual-state event")
        event_id = str(row["event_id"])
        if event_id in event_ids:
            raise ValueError("duplicate PBP visual-state event")
        event_ids.add(event_id)
    if any(row.get("state") not in _DETERMINATE_STATES for row in examples):
        raise ValueError("invalid determinate PBP visual state")
    if any(row.get("reason") not in _EXCLUSION_REASONS for row in exclusions):
        raise ValueError("invalid PBP visual-state exclusion")
    total = int(summary.get("total_clock_events", -1))
    if total != len(event_ids) or int(summary.get("determinate", -1)) != len(examples):
        raise ValueError("PBP visual-state summary coverage mismatch")
    claimed = str(manifest.pop("artifact_sha256"))
    if claimed != _canonical_sha256(manifest):
        raise ValueError("PBP visual-state artifact hash mismatch")
    manifest["artifact_sha256"] = claimed
    return manifest


def _select_anchor_clock_read(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, float | None]:
    candidates: list[tuple[float, float, Mapping[str, Any]]] = []
    for sample in samples:
        read = sample.get("read")
        if isinstance(read, Mapping):
            offset = float(sample["offset_seconds"])
            candidates.append((abs(offset), -float(read["confidence"]), sample))
    if not candidates:
        return None, None
    selected = min(candidates, key=lambda item: (item[0], item[1]))[2]
    return selected["read"], float(selected["offset_seconds"])


def _classify_actions(actions: Sequence[Mapping[str, Any]]) -> str:
    kinds = {str(row["action_type"]).strip().lower() for row in actions}
    has_free_throw = any("free throw" in kind for kind in kinds)
    has_field_goal = any(kind in {"made shot", "missed shot"} for kind in kinds)
    has_foul = any("foul" in kind for kind in kinds)
    if has_free_throw and has_field_goal:
        return "ambiguous"
    if has_free_throw:
        return "free_throw"
    if has_field_goal:
        return "field_goal"
    if has_foul:
        return "foul_only"
    return "other"


def _append_exclusion(
    exclusions: list[dict[str, Any]],
    counts: Counter[str],
    *,
    event_id: str,
    anchor_frame: int,
    reason: str,
    period: int,
    clock_seconds: int,
) -> None:
    exclusions.append(
        {
            "event_id": event_id,
            "anchor_frame": anchor_frame,
            "reason": reason,
            "period": period,
            "observed_clock_seconds": clock_seconds,
        }
    )
    counts[reason] += 1


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid {field}")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
