"""Dense AGU VLM discovery for events missed by sparse traditional signals."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from app.analysis.official_visuals import build_temporal_contact_sheet
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse, PerceptionDetectionResponse
from app.analysis.vlm import clamp_float, encode_frames_jpeg, parse_vlm_payload

DISCOVERABLE_EVENT_TYPES = {
    "field_goal_attempt",
    "free_throw_attempt",
    "rebound",
    "assist",
    "block",
    "steal",
    "turnover",
    "foul",
}


@dataclass(frozen=True)
class DiscoveredEvent:
    event_type: str
    relative_time_sec: float
    confidence: float
    reason: str = ""


class DenseEventDiscoverer(Protocol):
    name: str
    model: str

    def discover(
        self,
        frames: Sequence[np.ndarray],
        *,
        window_duration_sec: float,
    ) -> list[DiscoveredEvent]: ...


class OllamaDenseEventDiscoverer:
    name = "ollama_dense_official_event_discovery_v1"

    def __init__(
        self,
        *,
        model: str,
        host: str,
        timeout: float,
        image_width: int,
        context_length: int = 16384,
        contact_sheet: bool = False,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.image_width = image_width
        self.context_length = context_length
        self.contact_sheet = contact_sheet
        self.last_raw_response = ""
        self.last_error = ""

    def discover(
        self,
        frames: Sequence[np.ndarray],
        *,
        window_duration_sec: float,
    ) -> list[DiscoveredEvent]:
        if self.contact_sheet:
            sheet = build_temporal_contact_sheet(
                frames,
                duration_sec=window_duration_sec,
                columns=4,
                cell_width=max(160, self.image_width // 4),
            )
            image_frames = [sheet] if sheet is not None else []
        else:
            image_frames = list(frames)
        images = encode_frames_jpeg(image_frames, max_width=self.image_width)
        if not images:
            return []
        payload = {
            "model": self.model,
            "stream": False,
            "prompt": _discovery_prompt(window_duration_sec, len(images)),
            "images": images,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 700,
                "num_ctx": self.context_length,
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
                parsed, raw = parse_vlm_payload(json.loads(response.read().decode("utf-8")))
                self.last_raw_response = raw
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.last_error = f"HTTP {exc.code}: {detail}"
            raise RuntimeError(f"dense official-event VLM failed: {self.last_error}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"dense official-event VLM failed: {self.last_error}") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
            self.last_error = "VLM response must contain an events array"
            raise RuntimeError(self.last_error)
        self.last_error = ""
        return parse_discovered_events(parsed, window_duration_sec=window_duration_sec)


def parse_discovered_events(
    payload: Any,
    *,
    window_duration_sec: float,
) -> list[DiscoveredEvent]:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return []
    events: list[DiscoveredEvent] = []
    for item in payload["events"]:
        if not isinstance(item, dict):
            continue
        event_type = str(item.get("event_type") or "").strip().lower()
        if event_type not in DISCOVERABLE_EVENT_TYPES:
            continue
        try:
            relative_time = float(item.get("relative_time_sec"))
        except (TypeError, ValueError):
            continue
        if not 0 <= relative_time <= window_duration_sec:
            continue
        events.append(
            DiscoveredEvent(
                event_type=event_type,
                relative_time_sec=relative_time,
                confidence=clamp_float(item.get("confidence"), 0.0, 1.0, 0.0),
                reason=str(item.get("reason") or ""),
            )
        )
    return events


def discovered_events_to_candidates(
    discovered: Sequence[DiscoveredEvent],
    *,
    source_video_id: str,
    window_start_frame: int,
    fps: float,
    detections: Sequence[PerceptionDetectionResponse],
    event_id_prefix: str,
) -> list[GameEventResponse]:
    candidates: list[GameEventResponse] = []
    for index, item in enumerate(discovered, start=1):
        center = window_start_frame + int(round(item.relative_time_sec * fps))
        players, teams, observations = _nearby_identities(detections, center_frame=center, radius_frames=int(fps * 2))
        event_id = f"{event_id_prefix}-{index:03d}"
        evidence = EventEvidenceResponse(
            evidence_id=f"{event_id}:dense-vlm",
            kind="agu_edge_vlm_dense_candidate",
            source_video_id=source_video_id,
            start_frame=max(0, center - int(round(fps * 1.5))),
            end_frame=center + int(round(fps * 2.5)),
            confidence=item.confidence,
            details={
                "candidate_player_ids": players,
                "candidate_team_ids": teams,
                "candidate_player_observations": observations,
                "window_start_frame": window_start_frame,
                "relative_time_sec": item.relative_time_sec,
                "candidate_event_frame": center,
                "discovery_backend": "agu_vlm",
            },
        )
        candidates.append(
            GameEventResponse(
                event_id=event_id,
                revision=1,
                event_type=item.event_type,
                source_video_id=source_video_id,
                start_frame=evidence.start_frame,
                end_frame=evidence.end_frame,
                outcome="unknown" if item.event_type in {"field_goal_attempt", "free_throw_attempt"} else None,
                status="needs_review",
                confidence=item.confidence,
                evidence=[evidence],
                reason=item.reason or "dense AGU VLM candidate requires bounded semantic adjudication",
            )
        )
    return candidates


def _nearby_identities(
    detections: Sequence[PerceptionDetectionResponse],
    *,
    center_frame: int,
    radius_frames: int,
    maximum: int = 10,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    players = [
        item
        for item in detections
        if item.object_type == "player"
        and item.player_id
        and abs(item.frame - center_frame) <= radius_frames
    ]
    best_by_player: dict[str, PerceptionDetectionResponse] = {}
    for player in players:
        player_id = str(player.player_id)
        current = best_by_player.get(player_id)
        if current is None or (abs(player.frame - center_frame), -player.confidence) < (
            abs(current.frame - center_frame),
            -current.confidence,
        ):
            best_by_player[player_id] = player
    ranked = sorted(
        best_by_player.values(),
        key=lambda item: (abs(item.frame - center_frame), -item.confidence, str(item.player_id)),
    )[:maximum]
    observations = [
        {
            "player_id": item.player_id,
            "team_id": item.team_id,
            "frame": item.frame,
            "bbox": item.bbox.model_dump(mode="json"),
        }
        for item in ranked
    ]
    return (
        [str(item.player_id) for item in ranked],
        sorted({str(item.team_id) for item in ranked if item.team_id}),
        observations,
    )


def _discovery_prompt(window_duration_sec: float, frame_count: int) -> str:
    return (
        f"Inspect {frame_count} chronological frames spanning {window_duration_sec:.2f} seconds of a basketball game. "
        "Find only completed, directly visible basketball events. A field-goal attempt REQUIRES a visible ball release "
        "from a shooter's hand followed by ball flight toward the rim; holding, dribbling, gathering, passing, standing "
        "near the rim, or merely setting up is NOT a shot. A rebound REQUIRES visible first controlled possession after "
        "a missed shot, not movement toward the basket. Assists require a completed pass directly leading to a made "
        "basket; fouls require visible illegal contact and a stoppage. Do not infer hidden events outside the frames. Return JSON "
        "with key events, an array of objects: event_type (field_goal_attempt/free_throw_attempt/rebound/assist/"
        "block/steal/turnover/foul), relative_time_sec measured from the first frame, confidence 0..1, and reason. "
        "Return an empty events array when none is visible."
    )
