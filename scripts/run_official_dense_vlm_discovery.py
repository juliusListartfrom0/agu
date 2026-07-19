#!/usr/bin/env python3
"""Discover high-recall official-event candidates with AGU's local VLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.game_state import (  # noqa: E402
    link_causal_candidate_relations,
    merge_temporal_review_candidates,
)
from app.analysis.official_discovery import (  # noqa: E402
    DenseEventDiscoverer,
    OllamaDenseEventDiscoverer,
    discovered_events_to_candidates,
    parse_discovered_events,
)
from app.analysis.official_evaluation import seal_raw_only_predictions, verify_raw_only_bundle  # noqa: E402
from app.analysis.official_identity import canonicalize_perception_detections  # noqa: E402
from app.analysis.perception import attach_pose_keypoints  # noqa: E402
from app.analysis.schemas import (  # noqa: E402
    OfficialIdentityGraphArtifactResponse,
    PerceptionDetectionResponse,
    RawOnlyPredictionBundleResponse,
)
from app.config import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--base-candidate-bundle", type=Path)
    parser.add_argument("--identity-graph", type=Path)
    parser.add_argument("--pose", type=Path, help="Same-video raw-bound pose artifact")
    parser.add_argument("--cache", type=Path, help="Resumable per-window VLM discovery cache")
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--host", default=settings.ollama_host)
    parser.add_argument("--timeout", type=float, default=settings.official_vlm_timeout)
    parser.add_argument("--image-width", type=int, default=settings.official_vlm_image_width)
    parser.add_argument("--context-length", type=int, default=settings.official_vlm_context_length)
    parser.add_argument(
        "--contact-sheet", action=argparse.BooleanOptionalAction, default=settings.official_vlm_contact_sheet
    )
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--overlap-sec", type=float, default=2.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=0.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.45)
    return parser.parse_args()


def run_dense_discovery(
    *,
    video_path: Path,
    perception_path: Path,
    game_id: str,
    discoverer: DenseEventDiscoverer,
    window_sec: float,
    overlap_sec: float,
    sample_fps: float,
    start_sec: float,
    end_sec: float,
    minimum_confidence: float,
    base_candidate_bundle: RawOnlyPredictionBundleResponse | None = None,
    identity_artifact: OfficialIdentityGraphArtifactResponse | None = None,
    cache_path: Path | None = None,
    pose_path: Path | None = None,
) -> RawOnlyPredictionBundleResponse:
    if window_sec <= 0 or sample_fps <= 0 or not 0 <= overlap_sec < window_sec:
        raise ValueError("window/sample FPS must be positive and overlap smaller than window")
    perception = json.loads(perception_path.read_text(encoding="utf-8"))
    raw = perception.get("raw_video") or {}
    if perception.get("schema_version") != "agu.official-perception.v1" or raw.get("filename") != video_path.name:
        raise ValueError("perception artifact does not match the raw video")
    source_fps = float(raw.get("source_fps") or 0.0)
    duration_sec = float(raw.get("frame_count") or 0) / source_fps
    stop_sec = min(duration_sec, end_sec) if end_sec > 0 else duration_sec
    detections = [
        PerceptionDetectionResponse.model_validate(item) for item in perception.get("detections") or []
    ]
    if identity_artifact is not None:
        detections, _source_video_id = canonicalize_perception_detections(
            identity_artifact,
            detections,
            raw_filename=video_path.name,
            raw_sha256=str(raw.get("sha256") or ""),
        )
    pose_model_sha256 = ""
    if pose_path is not None:
        pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
        pose_raw = pose_payload.get("raw_video") or {}
        if pose_payload.get("schema_version") != "agu.official-pose.v1" or (
            pose_raw.get("filename") != raw.get("filename")
            or pose_raw.get("sha256") != raw.get("sha256")
            or float(pose_raw.get("source_fps") or 0) != source_fps
        ):
            raise ValueError("pose artifact does not match the raw video")
        pose_stride = int((pose_payload.get("sampling") or {}).get("stride_frames") or 0)
        detections = attach_pose_keypoints(
            detections,
            [
                PerceptionDetectionResponse.model_validate(item)
                for item in pose_payload.get("detections") or []
            ],
            maximum_frame_gap=max(0, pose_stride // 2),
            minimum_iou=0.25,
        )
        pose_model_sha256 = str((pose_payload.get("pose") or {}).get("model_sha256") or "")
    groups: list[list] = []
    if base_candidate_bundle is not None:
        base = verify_raw_only_bundle(base_candidate_bundle)
        if len(base.raw_videos) != 1 or base.raw_videos[0].filename != video_path.name:
            raise ValueError("base candidate bundle does not match the raw video")
        groups.append(base.events)
    cache_fingerprint = _cache_fingerprint(
        raw_sha256=str(raw.get("sha256") or ""),
        perception_sha256=_file_sha256(perception_path),
        identity_sha256=identity_artifact.artifact_sha256 if identity_artifact else "",
        base_bundle_sha256=base_candidate_bundle.bundle_sha256 if base_candidate_bundle else "",
        discoverer_name=discoverer.name,
        discoverer_model=discoverer.model,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        sample_fps=sample_fps,
        start_sec=start_sec,
        end_sec=stop_sec,
    )
    cache = _load_discovery_cache(cache_path, fingerprint=cache_fingerprint)
    discovered_candidates = []
    step = window_sec - overlap_sec
    window_start = max(0.0, start_sec)
    window_index = 0
    while window_start < stop_sec:
        window_end = min(stop_sec, window_start + window_sec)
        window_key = f"{window_start:.6f}:{window_end:.6f}"
        cached_window = cache["windows"].get(window_key)
        cache_hit = isinstance(cached_window, list)
        if cache_hit:
            discovered_all = parse_discovered_events(
                {"events": cached_window},
                window_duration_sec=window_end - window_start,
            )
        else:
            frames = _sample_window(video_path, window_start, window_end, sample_fps)
            discovered_all = discoverer.discover(
                frames,
                window_duration_sec=window_end - window_start,
            )
            cache["windows"][window_key] = [
                {
                    "event_type": item.event_type,
                    "relative_time_sec": item.relative_time_sec,
                    "confidence": item.confidence,
                    "reason": item.reason,
                }
                for item in discovered_all
            ]
            _write_discovery_cache(cache_path, cache)
        discovered = [item for item in discovered_all if item.confidence >= minimum_confidence]
        discovered_candidates.extend(
            discovered_events_to_candidates(
                discovered,
                source_video_id="video_001",
                window_start_frame=int(round(window_start * source_fps)),
                fps=source_fps,
                detections=detections,
                event_id_prefix=f"dense-vlm-{window_index:05d}",
            )
        )
        print(
            json.dumps(
                {
                    "window": window_index,
                    "start_sec": window_start,
                    "end_sec": window_end,
                    "discovered": len(discovered),
                    "total_candidates": len(discovered_candidates),
                    "cache_hit": cache_hit,
                }
            ),
            flush=True,
        )
        window_index += 1
        window_start += step
    groups.append(discovered_candidates)
    events = link_causal_candidate_relations(
        merge_temporal_review_candidates(
            groups,
            max_center_gap_frames=max(1, int(round(source_fps * overlap_sec))),
        ),
        max_gap_frames=max(1, int(round(source_fps * 3.0))),
    )
    return seal_raw_only_predictions(
        game_id=game_id,
        raw_video_paths=[video_path],
        events=events,
        config={
            "pipeline": "agu_dense_vlm_candidate_discovery_v1",
            "window_sec": window_sec,
            "overlap_sec": overlap_sec,
            "sample_fps": sample_fps,
            "start_sec": start_sec,
            "end_sec": stop_sec,
            "minimum_confidence": minimum_confidence,
            "identity_graph_enabled": identity_artifact is not None,
            "pose_enabled": pose_path is not None,
        },
        model_provenance={
            "producer": "agu",
            "artifact_role": "candidate_only",
            "candidate_backend": "traditional_cv+r2plus1d+agu_dense_vlm",
            "semantic_backend": discoverer.name,
            "semantic_model": discoverer.model,
            "detector_model_sha256": str((perception.get("detector") or {}).get("model_sha256") or ""),
            "identity_graph_sha256": identity_artifact.artifact_sha256 if identity_artifact else "",
            "identity_embedding_model": (
                identity_artifact.model_provenance.get("embedding_model", "")
                if identity_artifact
                else ""
            ),
            "pose_model_sha256": pose_model_sha256,
        },
    )


def _sample_window(video_path: Path, start_sec: float, end_sec: float, sample_fps: float) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    start_frame = int(round(start_sec * source_fps))
    end_frame = int(round(end_sec * source_fps))
    stride = max(1, int(round(source_fps / sample_fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    frame_number = start_frame
    try:
        while frame_number <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_number - start_frame) % stride == 0:
                frames.append(frame)
            frame_number += 1
    finally:
        capture.release()
    return frames


def _cache_fingerprint(**payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_discovery_cache(path: Path | None, *, fingerprint: str) -> dict[str, object]:
    empty: dict[str, object] = {
        "schema_version": "agu.official-dense-vlm-cache.v1",
        "fingerprint": fingerprint,
        "windows": {},
    }
    if path is None or not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != empty["schema_version"]
        or payload.get("fingerprint") != fingerprint
        or not isinstance(payload.get("windows"), dict)
    ):
        return empty
    return payload


def _write_discovery_cache(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    base = None
    if args.base_candidate_bundle:
        base = RawOnlyPredictionBundleResponse.model_validate_json(
            args.base_candidate_bundle.read_text(encoding="utf-8")
        )
    identity_artifact = None
    if args.identity_graph:
        identity_artifact = OfficialIdentityGraphArtifactResponse.model_validate_json(
            args.identity_graph.read_text(encoding="utf-8")
        )
    result = run_dense_discovery(
        video_path=args.video,
        perception_path=args.perception,
        game_id=args.game_id,
        discoverer=OllamaDenseEventDiscoverer(
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            image_width=args.image_width,
            context_length=args.context_length,
            contact_sheet=args.contact_sheet,
        ),
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        sample_fps=args.sample_fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        minimum_confidence=args.minimum_confidence,
        base_candidate_bundle=base,
        identity_artifact=identity_artifact,
        cache_path=args.cache or args.output.with_suffix(args.output.suffix + ".discovery-cache.json"),
        pose_path=args.pose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bundle_sha256": result.bundle_sha256, "events": len(result.events)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
