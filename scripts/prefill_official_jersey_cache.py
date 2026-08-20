#!/usr/bin/env python3
"""Prefill resumable jersey-number VLM cache from a sealed identity graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.jersey_identity import CachedJerseyNumberReader  # noqa: E402
from app.analysis.official_identity import _read_trusted_jersey_number  # noqa: E402
from app.analysis.vlm import OllamaVLMVerifier  # noqa: E402
from app.config import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-graph", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--candidate-player-prefix", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-vl:2b")
    parser.add_argument("--host", default=settings.ollama_host)
    parser.add_argument("--timeout", type=float, default=settings.official_vlm_timeout)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-tracklets", type=int, default=12)
    parser.add_argument("--minimum-crops", type=int, default=4)
    parser.add_argument("--minimum-consensus", type=int, default=2)
    parser.add_argument("--minimum-confidence", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity_payload = json.loads(args.identity_graph.read_text(encoding="utf-8"))
    candidate_payload = json.loads(args.candidate_bundle.read_text(encoding="utf-8"))
    _verify_video_binding(identity_payload, args.video)
    counts = _candidate_source_counts(
        candidate_payload,
        source_prefix=args.candidate_player_prefix,
    )
    selected = _select_tracklets(
        identity_payload,
        candidate_counts=counts,
        maximum_tracklets=args.maximum_tracklets,
        minimum_crops=args.minimum_crops,
    )
    crops_by_tracklet = _read_selected_crops(args.video, selected)
    delegate = OllamaVLMVerifier(
        model=args.model,
        host=args.host,
        timeout=args.timeout,
        image_width=args.image_width,
        context_length=args.context_length,
        seed=args.seed,
    )
    reader = CachedJerseyNumberReader(delegate, args.cache)
    results: list[dict[str, object]] = []
    try:
        for tracklet in selected:
            tracklet_id = str(tracklet["tracklet_id"])
            crops = crops_by_tracklet.get(tracklet_id) or []
            number, confidence = _read_trusted_jersey_number(
                reader,
                crops,
                tracklet_id=tracklet_id,
                source_player_id=str(tracklet["source_player_id"]),
                minimum_confidence=args.minimum_confidence,
                minimum_crops=args.minimum_crops,
                minimum_consensus=args.minimum_consensus,
                allowed_source_player_ids=None,
            )
            results.append(
                {
                    "tracklet_id": tracklet_id,
                    "source_player_id": tracklet["source_player_id"],
                    "candidate_event_count": counts[str(tracklet["source_player_id"])],
                    "crop_count": len(crops),
                    "trusted_number": number,
                    "trusted_confidence": confidence,
                }
            )
    finally:
        _release_ollama_model(args.host, args.model, timeout=min(10.0, args.timeout))
    print(
        json.dumps(
            {
                "model": args.model,
                "selected_tracklets": len(selected),
                "trusted_reads": sum(item["trusted_number"] is not None for item in results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _candidate_source_counts(
    payload: Mapping[str, object],
    *,
    source_prefix: str,
) -> dict[str, int]:
    if payload.get("schema_version") != "agu.raw-only.v1":
        raise ValueError("unsupported official event-candidate schema")
    counts: Counter[str] = Counter()
    for event in payload.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        event_ids: set[str] = set()
        for evidence in event.get("evidence") or []:
            if not isinstance(evidence, Mapping):
                continue
            details = evidence.get("details") or {}
            if not isinstance(details, Mapping):
                continue
            values = list(details.get("candidate_player_ids") or [])
            values.extend(
                observation.get("player_id")
                for observation in details.get("candidate_player_observations") or []
                if isinstance(observation, Mapping)
            )
            event_ids.update(
                str(value).removeprefix(source_prefix)
                for value in values
                if value and str(value).startswith(source_prefix)
            )
        counts.update(event_ids)
    return dict(sorted(counts.items()))


def _select_tracklets(
    identity_payload: Mapping[str, object],
    *,
    candidate_counts: Mapping[str, int],
    maximum_tracklets: int,
    minimum_crops: int,
) -> list[dict[str, Any]]:
    if identity_payload.get("schema_version") != "agu.official-identity.v1":
        raise ValueError("unsupported official identity schema")
    if maximum_tracklets <= 0 or minimum_crops <= 0:
        raise ValueError("tracklet and crop limits must be positive")
    best_by_source: dict[str, tuple[tuple[float, ...], dict[str, Any]]] = {}
    for raw in identity_payload.get("tracklets") or []:
        if not isinstance(raw, Mapping):
            continue
        tracklet = dict(raw)
        source_id = str(tracklet.get("source_player_id") or "")
        boxes = [item for item in tracklet.get("sampled_boxes") or [] if isinstance(item, Mapping)]
        if (
            not source_id
            or candidate_counts.get(source_id, 0) <= 0
            or tracklet.get("gallery_person_id")
            or int(tracklet.get("crop_count") or 0) < minimum_crops
            or len(boxes) < minimum_crops
        ):
            continue
        heights = [max(0.0, float(box["y2"]) - float(box["y1"])) for box in boxes]
        areas = [
            max(0.0, float(box["x2"]) - float(box["x1"]))
            * max(0.0, float(box["y2"]) - float(box["y1"]))
            for box in boxes
        ]
        median_height = statistics.median(heights)
        event_count = float(candidate_counts[source_id])
        rank = (
            event_count * median_height,
            event_count,
            median_height,
            statistics.median(areas),
            float(tracklet.get("observation_count") or 0),
        )
        previous = best_by_source.get(source_id)
        if previous is None or rank > previous[0]:
            best_by_source[source_id] = (rank, tracklet)
    ranked = sorted(
        best_by_source.values(),
        key=lambda item: (*item[0], str(item[1].get("tracklet_id") or "")),
        reverse=True,
    )
    return [item[1] for item in ranked[:maximum_tracklets]]


def _verify_video_binding(identity_payload: Mapping[str, object], video: Path) -> None:
    matches = [
        item
        for item in identity_payload.get("raw_videos") or []
        if isinstance(item, Mapping) and item.get("filename") == video.name
    ]
    if len(matches) != 1:
        raise ValueError("identity graph must contain exactly one matching raw video")
    if matches[0].get("sha256") != _file_sha256(video):
        raise ValueError("identity graph does not match raw video")


def _read_selected_crops(
    video: Path,
    tracklets: Sequence[Mapping[str, object]],
) -> dict[str, list[np.ndarray]]:
    requests: dict[int, list[tuple[str, Mapping[str, object]]]] = {}
    for tracklet in tracklets:
        tracklet_id = str(tracklet["tracklet_id"])
        for box in tracklet.get("sampled_boxes") or []:
            if isinstance(box, Mapping):
                requests.setdefault(int(float(box["frame"])), []).append((tracklet_id, box))
    crops: dict[str, list[np.ndarray]] = {}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open raw video: {video}")
    try:
        for frame_number in sorted(requests):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            for tracklet_id, box in requests[frame_number]:
                crop = _crop_saved_box(frame, box)
                if crop is not None:
                    crops.setdefault(tracklet_id, []).append(crop)
    finally:
        capture.release()
    return crops


def _crop_saved_box(
    frame: np.ndarray,
    box: Mapping[str, object],
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1_raw, y1_raw = float(box["x1"]), float(box["y1"])
    x2_raw, y2_raw = float(box["x2"]), float(box["y2"])
    box_width, box_height = x2_raw - x1_raw, y2_raw - y1_raw
    if box_width < 15 or box_height < 50:
        return None
    x1 = max(0, min(width - 1, int(round(x1_raw - box_width * 0.03))))
    x2 = max(x1 + 1, min(width, int(round(x2_raw + box_width * 0.03))))
    y1 = max(0, min(height - 1, int(round(y1_raw - box_height * 0.02))))
    y2 = max(y1 + 1, min(height, int(round(y2_raw + box_height * 0.02))))
    if x1 == 0 or x2 == width:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


def _release_ollama_model(host: str, model: str, *, timeout: float) -> None:
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            pass
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
