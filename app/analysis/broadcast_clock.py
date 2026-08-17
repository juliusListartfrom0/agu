from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

BROADCAST_CLOCK_EVIDENCE_SCHEMA = "agu.broadcast-clock-replay-evidence.v1"

_CLOCK = re.compile(r"(\d{1,2})[:.](\d{2})")
_NORMALIZE_CLOCK = re.compile(r"[^A-Z0-9:.]")
_PERIOD = re.compile(r"(1ST|1S[T7]|2ND|3RD|4TH|OT)")
_PERIOD_NUMBER = {
    "1ST": 1,
    "1S7": 1,
    "2ND": 2,
    "3RD": 3,
    "4TH": 4,
    "OT": 5,
}


@dataclass(frozen=True)
class BroadcastClockRead:
    frame: int
    period: int
    clock_seconds: int
    confidence: float
    raw_text: str
    source: str = "rapidocr_broadcast_clock_v1"


@dataclass(frozen=True)
class BroadcastClockSample:
    frame: int
    offset_seconds: float
    read: BroadcastClockRead | None


@dataclass(frozen=True)
class BroadcastClockReplayEvidence:
    event_id: str
    anchor_frame: int
    samples: tuple[BroadcastClockSample, ...]
    pre_anchor_read_count: int
    post_anchor_missing_count: int
    frozen_period: int | None
    frozen_clock_seconds: int | None
    broadcast_state: str
    method: str = "pre_anchor_frozen_clock_disappearance_v1"


def parse_broadcast_clock(
    results: Sequence[Sequence[Any]],
    *,
    frame: int,
    image_height: int,
    minimum_confidence: float = 0.55,
) -> BroadcastClockRead | None:
    """Parse a game clock only when a period marker occurs on the same OCR row."""

    if frame < 0 or image_height <= 0 or not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("broadcast clock parser configuration is invalid")
    tokens = _raw_ocr_tokens(results, minimum_confidence=minimum_confidence)
    row_tolerance = max(8.0, image_height * 0.08)
    candidates: list[BroadcastClockRead] = []
    for anchor in tokens:
        if _PERIOD.search(anchor.text) is None:
            continue
        row = sorted(
            (
                token
                for token in tokens
                if abs(token.center_y - anchor.center_y) <= row_tolerance
            ),
            key=lambda token: token.center_x,
        )
        anchor_index = row.index(anchor)
        for stop in range(anchor_index + 1, min(len(row), anchor_index + 4) + 1):
            selected = row[anchor_index:stop]
            joined = "".join(token.text for token in selected)
            parsed = _parse_period_clock(joined)
            if parsed is None:
                continue
            period, clock_seconds = parsed
            candidates.append(
                BroadcastClockRead(
                    frame=frame,
                    period=period,
                    clock_seconds=clock_seconds,
                    confidence=min(token.confidence for token in selected),
                    raw_text=joined,
                )
            )
            break
    if not candidates:
        return None
    return max(candidates, key=lambda read: (read.confidence, len(read.raw_text)))


def summarize_clock_disappearance(
    *,
    event_id: str,
    anchor_frame: int,
    samples: Sequence[BroadcastClockSample],
    maximum_pre_anchor_offset_seconds: float = -3.0,
    minimum_pre_anchor_read_count: int = 2,
    minimum_post_anchor_missing_count: int = 3,
) -> BroadcastClockReplayEvidence:
    """Flag only a frozen pre-anchor clock followed by sampled disappearance.

    Clock absence alone is not replay evidence. Post-anchor frozen clocks are
    also deliberately ignored because ordinary made shots and dead balls stop
    the game clock.
    """

    ordered = tuple(sorted(samples, key=lambda sample: sample.offset_seconds))
    if (
        not event_id
        or anchor_frame < 0
        or not ordered
        or minimum_pre_anchor_read_count < 2
        or minimum_post_anchor_missing_count <= 0
        or len({sample.offset_seconds for sample in ordered}) != len(ordered)
        or any(sample.frame < 0 for sample in ordered)
    ):
        raise ValueError("broadcast clock replay evidence inputs are invalid")
    pre_reads = [
        sample.read
        for sample in ordered
        if sample.offset_seconds <= maximum_pre_anchor_offset_seconds
        and sample.read is not None
    ]
    post_samples = [sample for sample in ordered if sample.offset_seconds >= 0.0]
    frozen_values = {
        (read.period, read.clock_seconds)
        for read in pre_reads
    }
    frozen = (
        len(pre_reads) >= minimum_pre_anchor_read_count
        and len(frozen_values) == 1
    )
    post_missing_count = sum(sample.read is None for sample in post_samples)
    complete_disappearance = (
        len(post_samples) >= minimum_post_anchor_missing_count
        and post_missing_count == len(post_samples)
    )
    frozen_period, frozen_clock = (
        next(iter(frozen_values)) if frozen else (None, None)
    )
    return BroadcastClockReplayEvidence(
        event_id=event_id,
        anchor_frame=anchor_frame,
        samples=ordered,
        pre_anchor_read_count=len(pre_reads),
        post_anchor_missing_count=post_missing_count,
        frozen_period=frozen_period,
        frozen_clock_seconds=frozen_clock,
        broadcast_state="replay" if frozen and complete_disappearance else "unknown",
    )


def clock_evidence_payload(evidence: BroadcastClockReplayEvidence) -> dict[str, Any]:
    payload = asdict(evidence)
    payload["samples"] = list(payload["samples"])
    return payload


def seal_broadcast_clock_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_broadcast_clock_artifact(artifact)


def verify_broadcast_clock_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    if artifact.get("schema_version") != BROADCAST_CLOCK_EVIDENCE_SCHEMA:
        raise ValueError("unsupported broadcast clock evidence schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("broadcast clock replay evidence must remain raw-only offline")
    events = artifact.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("broadcast clock evidence requires events")
    for raw in events:
        if not isinstance(raw, Mapping) or not str(raw.get("event_id") or ""):
            raise ValueError("invalid broadcast clock evidence event")
        samples = raw.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("broadcast clock event requires samples")
        offsets: set[float] = set()
        for sample in samples:
            if not isinstance(sample, Mapping) or int(sample.get("frame", -1)) < 0:
                raise ValueError("invalid broadcast clock sample")
            offset = float(sample.get("offset_seconds"))
            if offset in offsets:
                raise ValueError("broadcast clock sample offsets must be unique")
            offsets.add(offset)
            read = sample.get("read")
            if read is not None and (
                not isinstance(read, Mapping)
                or int(read.get("frame", -1)) != int(sample["frame"])
                or int(read.get("period", 0)) not in {1, 2, 3, 4, 5}
                or not 0 <= int(read.get("clock_seconds", -1)) <= 12 * 60
                or not 0.0 <= float(read.get("confidence", -1.0)) <= 1.0
            ):
                raise ValueError("invalid broadcast clock read")
        if raw.get("broadcast_state") not in {"replay", "unknown"}:
            raise ValueError("invalid broadcast clock state")
    claimed = str(artifact.pop("artifact_sha256", ""))
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("broadcast clock artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def evaluate_broadcast_clock_proxy(
    *,
    labels: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    minimum_precision: float = 0.95,
    minimum_replay_predictions: int = 5,
) -> dict[str, Any]:
    if (
        not labels
        or len(labels) != len(evidence)
        or not 0.0 < minimum_precision <= 1.0
        or minimum_replay_predictions <= 0
    ):
        raise ValueError("broadcast clock evaluation inputs are invalid")
    true_positive = false_positive = false_negative = true_negative = 0
    false_positive_event_ids: list[str] = []
    evaluated: set[tuple[str, str]] = set()
    per_video: list[dict[str, Any]] = []
    for label_payload, evidence_payload in zip(labels, evidence, strict=True):
        verified = verify_broadcast_clock_artifact(evidence_payload)
        if (
            label_payload.get("schema_version") != "agu.shot-validity-labels.v1"
            or label_payload.get("runtime_consumable") is not False
            or label_payload.get("source_video_sha256")
            != verified.get("raw_video_sha256")
            or label_payload.get("candidate_bundle_sha256")
            != verified.get("candidate_bundle_sha256")
        ):
            raise ValueError("broadcast clock evidence is not bound to offline labels")
        label_by_id = {
            str(row.get("event_id") or ""): row
            for row in label_payload.get("examples", ())
            if isinstance(row, Mapping)
        }
        video_counts = {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
        }
        for event in verified["events"]:
            event_id = str(event["event_id"])
            event_key = (str(verified["raw_video_sha256"]), event_id)
            row = label_by_id.get(event_id)
            if (
                row is None
                or event_key in evaluated
                or not isinstance(row.get("event_present"), bool)
            ):
                raise ValueError("broadcast clock event is absent or duplicated")
            is_live = bool(row["event_present"])
            if not is_live and not any(
                marker in str(row.get("review_note") or "").lower()
                for marker in ("replay", "highlight")
            ):
                raise ValueError("negative clock subset contains a non-replay window")
            predicted_replay = event["broadcast_state"] == "replay"
            is_replay = not is_live
            true_positive += int(predicted_replay and is_replay)
            false_positive += int(predicted_replay and is_live)
            false_negative += int(not predicted_replay and is_replay)
            true_negative += int(not predicted_replay and is_live)
            video_counts["true_positive"] += int(predicted_replay and is_replay)
            video_counts["false_positive"] += int(predicted_replay and is_live)
            video_counts["false_negative"] += int(not predicted_replay and is_replay)
            video_counts["true_negative"] += int(not predicted_replay and is_live)
            if predicted_replay and is_live:
                false_positive_event_ids.append(
                    f"{str(verified['raw_video_sha256'])[:12]}/{event_id}"
                )
            evaluated.add(event_key)
        video_predicted = (
            video_counts["true_positive"] + video_counts["false_positive"]
        )
        video_replay = video_counts["true_positive"] + video_counts["false_negative"]
        per_video.append(
            {
                "raw_video_sha256": str(verified["raw_video_sha256"]),
                **video_counts,
                "precision": (
                    video_counts["true_positive"] / video_predicted
                    if video_predicted
                    else 0.0
                ),
                "recall": (
                    video_counts["true_positive"] / video_replay
                    if video_replay
                    else 0.0
                ),
            }
        )
    predicted_count = true_positive + false_positive
    replay_count = true_positive + false_negative
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / replay_count if replay_count else 0.0
    return {
        "evaluated_count": len(evaluated),
        "live_count": false_positive + true_negative,
        "replay_count": replay_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "minimum_precision": minimum_precision,
        "minimum_replay_predictions": minimum_replay_predictions,
        "promotion_eligible": (
            predicted_count >= minimum_replay_predictions
            and precision >= minimum_precision
        ),
        "false_positive_event_ids": false_positive_event_ids,
        "per_video": per_video,
    }


@dataclass(frozen=True)
class _RawOCRToken:
    text: str
    confidence: float
    center_x: float
    center_y: float


def _raw_ocr_tokens(
    results: Sequence[Sequence[Any]],
    *,
    minimum_confidence: float,
) -> list[_RawOCRToken]:
    tokens: list[_RawOCRToken] = []
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
        normalized = _NORMALIZE_CLOCK.sub("", str(text or "").upper())
        if confidence_value < minimum_confidence or not normalized:
            continue
        tokens.append(_RawOCRToken(normalized, confidence_value, center_x, center_y))
    return tokens


def _parse_period_clock(value: str) -> tuple[int, int] | None:
    period_match = _PERIOD.search(value)
    if period_match is None:
        return None
    clock_match = _CLOCK.search(value[period_match.end() :])
    if clock_match is None:
        return None
    minutes, seconds = (int(part) for part in clock_match.groups())
    if minutes > 12 or seconds >= 60:
        return None
    period_text = period_match.group()
    period = _PERIOD_NUMBER.get(period_text)
    if period is None and period_text.startswith("1S"):
        period = 1
    if period is None:
        return None
    return period, minutes * 60 + seconds


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
