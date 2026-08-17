#!/usr/bin/env python3
"""Use AGU traditional candidates plus its configured VLM to seal predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.action_ownership import ActionOwnerModel  # noqa: E402
from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.official_inference import (  # noqa: E402
    CacheAwareFallbackOfficialEventReviewer,
    OfficialEventReviewer,
    OllamaOfficialEventReviewer,
    TwoPassOfficialEventReviewer,
    adjudicate_official_candidates,
    candidate_player_aliases,
    seal_agu_autonomous_predictions,
)
from app.analysis.schemas import GameEventResponse, RawOnlyPredictionBundleResponse  # noqa: E402
from app.config import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--host", default=settings.ollama_host)
    parser.add_argument("--timeout", type=float, default=settings.official_vlm_timeout)
    parser.add_argument("--confidence", type=float, default=settings.official_vlm_confidence)
    parser.add_argument("--frames", type=int, default=settings.official_vlm_frames)
    parser.add_argument("--image-width", type=int, default=settings.official_vlm_image_width)
    parser.add_argument("--context-length", type=int, default=settings.official_vlm_context_length)
    parser.add_argument(
        "--ollama-keep-alive",
        type=int,
        default=None,
        help="Optional Ollama keep_alive value in seconds; use 0 to unload after each review",
    )
    parser.add_argument(
        "--ollama-semantic-keep-alive",
        type=int,
        default=None,
        help="Optional two-pass semantic keep_alive override so an accepted event can reuse the loaded model",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--contact-sheet", action=argparse.BooleanOptionalAction, default=settings.official_vlm_contact_sheet
    )
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument(
        "--rim-detail-inset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a raw-derived magnified rim inset to semantic frames using traditional detections",
    )
    parser.add_argument("--review-pre-seconds", type=float, default=3.0)
    parser.add_argument("--review-post-seconds", type=float, default=3.0)
    parser.add_argument("--actor-pre-seconds", type=float, default=2.0)
    parser.add_argument("--actor-post-seconds", type=float, default=1.5)
    parser.add_argument(
        "--semantic-fallback-image-width",
        type=int,
        help="For primary semantic-cache misses only, use this lower image width",
    )
    parser.add_argument(
        "--semantic-fallback-model",
        help="For primary semantic-cache misses only, use this resource-safe model",
    )
    parser.add_argument(
        "--semantic-fallback-context-length",
        type=int,
        help="For primary semantic-cache misses only, use this context length",
    )
    parser.add_argument(
        "--semantic-fallback-keep-alive",
        type=int,
        help="Optional Ollama keep_alive seconds for the semantic fallback model",
    )
    parser.add_argument(
        "--actor-fallback-image-width",
        type=int,
        help="For primary actor-cache misses only, use this lower image width",
    )
    parser.add_argument(
        "--actor-fallback-model",
        help="For primary actor-cache misses only, use this resource-safe model",
    )
    parser.add_argument(
        "--actor-fallback-context-length",
        type=int,
        help="For primary actor-cache misses only, use this lower context length",
    )
    parser.add_argument(
        "--actor-fallback-keep-alive",
        type=int,
        help="Optional Ollama keep_alive seconds for the actor fallback model",
    )
    parser.add_argument("--maximum-identity-overlays", type=int, default=2)
    parser.add_argument("--two-pass-vlm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--semantic-only-vlm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Review clean event semantics without spending a second VLM call on actor identity",
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--action-owner-model",
        type=Path,
        default=(
            Path(settings.official_action_owner_model_path) if settings.official_action_owner_model_path else None
        ),
    )
    return parser.parse_args()


class VideoEventFrameProvider:
    def __init__(
        self,
        video_path: Path,
        *,
        sample_fps: float,
        review_pre_seconds: float = 3.0,
        review_post_seconds: float = 3.0,
        overlay_identities: bool = True,
        maximum_identity_overlays: int = 2,
        prefer_release_frame: bool = False,
        rim_detail_inset: bool = False,
    ) -> None:
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.review_pre_seconds = review_pre_seconds
        self.review_post_seconds = review_post_seconds
        self.overlay_identities = overlay_identities
        self.maximum_identity_overlays = maximum_identity_overlays
        self.prefer_release_frame = prefer_release_frame
        self.rim_detail_inset = rim_detail_inset
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {video_path}")
        self.source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.release()

    def __call__(self, event: GameEventResponse) -> Sequence[np.ndarray]:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            return []
        stride = max(1, int(round(self.source_fps / self.sample_fps)))
        frames: list[np.ndarray] = []
        start_frame, end_frame = _review_frame_bounds(
            event,
            source_fps=self.source_fps,
            pre_seconds=self.review_pre_seconds,
            post_seconds=self.review_post_seconds,
            prefer_release_frame=self.prefer_release_frame,
        )
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_number = start_frame
            while frame_number <= end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if (frame_number - start_frame) % stride == 0:
                    if self.rim_detail_inset:
                        frame = _add_rim_detail_inset(frame, event, frame_number=frame_number)
                    if self.overlay_identities:
                        frame = _overlay_candidate_identities(
                            frame,
                            event,
                            frame_number=frame_number,
                            maximum_frame_gap=max(stride, 3),
                            maximum=self.maximum_identity_overlays,
                        )
                    frames.append(frame)
                frame_number += 1
        finally:
            capture.release()
        return frames

    def selected_frame_numbers(self, event: GameEventResponse, maximum: int) -> list[int]:
        start_frame, end_frame = _review_frame_bounds(
            event,
            source_fps=self.source_fps,
            pre_seconds=self.review_pre_seconds,
            post_seconds=self.review_post_seconds,
            prefer_release_frame=self.prefer_release_frame,
        )
        stride = max(1, int(round(self.source_fps / self.sample_fps)))
        sampled = list(range(start_frame, end_frame + 1, stride))
        return _evenly_select_values(sampled, maximum)


def _review_frame_bounds(
    event: GameEventResponse,
    *,
    source_fps: float,
    pre_seconds: float,
    post_seconds: float,
    prefer_release_frame: bool = False,
) -> tuple[int, int]:
    """Concentrate VLM frames around a traditional-model event anchor."""

    related_release_frame = None
    if event.event_type == "rebound":
        for evidence in event.evidence:
            value = evidence.details.get("related_shot_release_frame")
            try:
                related_release_frame = int(value) if value is not None else None
            except (TypeError, ValueError):
                related_release_frame = None
            if related_release_frame is not None:
                break
    anchor = (
        related_release_frame + int(round(source_fps))
        if related_release_frame is not None
        else event.release_frame
        if prefer_release_frame
        else event.outcome_frame
    )
    if anchor is None:
        anchor = event.outcome_frame
    if anchor is None:
        for evidence in event.evidence:
            value = evidence.details.get("review_anchor_frame")
            if value is None:
                value = evidence.details.get("candidate_event_frame")
            try:
                anchor = int(value) if value is not None else None
            except (TypeError, ValueError):
                anchor = None
            if anchor is not None:
                break
    if anchor is None:
        anchor = (event.start_frame + event.end_frame) // 2
    start = max(0, anchor - int(round(source_fps * pre_seconds)))
    end = anchor + int(round(source_fps * post_seconds))
    return start, end


def _add_rim_detail_inset(
    frame: np.ndarray,
    event: GameEventResponse,
    *,
    frame_number: int,
) -> np.ndarray:
    """Magnify detector-grounded rim context without adding semantic labels."""

    observations = [
        item
        for evidence in event.evidence
        for item in (evidence.details.get("review_rim_observations") or [])
        if isinstance(item, dict) and isinstance(item.get("rim_bbox"), dict)
    ]
    if not observations:
        return frame
    nearest = min(
        observations,
        key=lambda item: abs(int(item.get("frame", frame_number)) - frame_number),
    )
    rim = nearest["rim_bbox"]
    try:
        x1, y1, x2, y2 = (float(rim[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return frame
    height, width = frame.shape[:2]
    rim_width = max(1.0, x2 - x1)
    rim_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_width = max(rim_width * 7.0, width * 0.22)
    crop_height = max(rim_height * 9.0, height * 0.32)
    left = max(0, int(round(center_x - crop_width / 2.0)))
    right = min(width, int(round(center_x + crop_width / 2.0)))
    top = max(0, int(round(center_y - crop_height * 0.55)))
    bottom = min(height, int(round(center_y + crop_height * 0.45)))
    if right - left < 4 or bottom - top < 4:
        return frame
    inset_width = max(96, int(round(width * 0.34)))
    inset_height = max(72, int(round(height * 0.34)))
    detail = cv2.resize(frame[top:bottom, left:right], (inset_width, inset_height))
    output = frame.copy()
    target_left = width - inset_width - 8
    target_top = height - inset_height - 8
    output[target_top : target_top + inset_height, target_left : target_left + inset_width] = detail
    cv2.rectangle(
        output,
        (target_left - 1, target_top - 1),
        (target_left + inset_width, target_top + inset_height),
        (0, 255, 255),
        2,
    )
    return output


def _overlay_candidate_identities(
    frame: np.ndarray,
    event: GameEventResponse,
    *,
    frame_number: int,
    maximum_frame_gap: int,
    maximum: int = 2,
) -> np.ndarray:
    observations = []
    for evidence in event.evidence:
        values = evidence.details.get("candidate_player_observations") or []
        if isinstance(values, list):
            for item in values:
                if not isinstance(item, dict):
                    continue
                try:
                    observation_frame = int(item.get("frame"))
                except (TypeError, ValueError):
                    continue
                if abs(observation_frame - frame_number) <= maximum_frame_gap:
                    observations.append(item)
    if not observations:
        return frame
    annotated = frame.copy()
    aliases = candidate_player_aliases(event)
    observations = _select_identity_overlays(
        observations,
        maximum=maximum,
        prefer_stable_control=event.event_type == "rebound",
    )
    for item in observations:
        bbox = item.get("bbox") or {}
        try:
            x1, y1, x2, y2 = (int(float(bbox[key])) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        player_id = str(item.get("player_id") or "unknown")
        team_id = str(item.get("team_id") or "unknown-team")
        alias = aliases.get(player_id, "P??")
        font_scale = max(0.9, frame.shape[1] / 1000.0)
        thickness = max(3, int(round(font_scale * 2.5)))
        label = f"{alias} {team_id}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(text_height + baseline + 4, y1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.rectangle(
            annotated,
            (max(0, x1), max(0, label_y - text_height - baseline - 6)),
            (min(frame.shape[1] - 1, x1 + text_width + 8), min(frame.shape[0] - 1, label_y + 3)),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (max(0, x1 + 4), max(text_height + 2, label_y - baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def _select_identity_overlays(
    observations: Sequence[dict[str, object]],
    *,
    maximum: int,
    prefer_stable_control: bool = False,
) -> list[dict[str, object]]:
    if maximum <= 0:
        return []
    by_ball_distance = sorted(
        observations,
        key=lambda item: (
            (
                _finite_score(item.get("stable_control_score"), default=float("inf"))
                if prefer_stable_control
                else 0.0
            ),
            _candidate_contact_distance(item),
            float(item.get("ball_player_distance", float("inf"))),
            str(item.get("player_id") or ""),
        ),
    )
    scored = [item for item in observations if _finite_score(item.get("action_owner_probability"), default=-1.0) >= 0.0]
    if not scored:
        selected: list[dict[str, object]] = []
        selected_player_ids: set[str] = set()
        for item in by_ball_distance:
            player_id = str(item.get("player_id") or "")
            if not player_id or player_id in selected_player_ids:
                continue
            selected.append(item)
            selected_player_ids.add(player_id)
            if len(selected) >= maximum:
                break
        return selected
    model_top = min(
        scored,
        key=lambda item: (
            -_finite_score(item.get("action_owner_probability"), default=-1.0),
            float(item.get("ball_player_distance", float("inf"))),
            str(item.get("player_id") or ""),
        ),
    )
    selected = [model_top]
    selected_player_ids = {str(model_top.get("player_id") or "")}
    for item in by_ball_distance:
        player_id = str(item.get("player_id") or "")
        if player_id in selected_player_ids:
            continue
        selected.append(item)
        selected_player_ids.add(player_id)
        if len(selected) >= maximum:
            break
    return selected


def _finite_score(value: object, *, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return score if np.isfinite(score) else default


def _candidate_contact_distance(item: dict[str, object]) -> float:
    return _finite_score(
        item.get("wrist_ball_distance"),
        default=float(item.get("ball_player_distance", float("inf"))),
    )


def _evenly_select_values(values: Sequence[int], maximum: int) -> list[int]:
    if not values or maximum <= 0:
        return []
    if len(values) <= maximum:
        return list(values)
    indexes = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return [int(values[int(index)]) for index in indexes]


def run_autonomous_inference(
    *,
    candidate_bundle: RawOnlyPredictionBundleResponse,
    video_path: Path,
    reviewer: OfficialEventReviewer,
    minimum_confidence: float,
    sample_fps: float,
    review_pre_seconds: float = 3.0,
    review_post_seconds: float = 3.0,
    overlay_identities: bool = True,
    rim_detail_inset: bool = False,
) -> RawOnlyPredictionBundleResponse:
    source = verify_raw_only_bundle(candidate_bundle)
    if len(source.raw_videos) != 1 or source.raw_videos[0].filename != video_path.name:
        raise ValueError("candidate bundle must declare exactly the provided raw video")
    provider = VideoEventFrameProvider(
        video_path,
        sample_fps=sample_fps,
        review_pre_seconds=review_pre_seconds,
        review_post_seconds=review_post_seconds,
        overlay_identities=overlay_identities,
        rim_detail_inset=rim_detail_inset,
    )
    events = adjudicate_official_candidates(
        source.events,
        reviewer=reviewer,
        frame_provider=provider,
        minimum_confidence=minimum_confidence,
        progress_callback=lambda index, total, event, result: print(
            json.dumps(
                {
                    "reviewed": index,
                    "total": total,
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "present": result.event_present,
                    "confidence": result.confidence,
                    "available": result.available,
                }
            ),
            flush=True,
        ),
    )
    action_owner_model = getattr(reviewer, "action_owner_model", None)
    semantic_reviewer = getattr(reviewer, "semantic_reviewer", None)
    semantic_fallback_reviewer = getattr(semantic_reviewer, "fallback_reviewer", None)
    actor_reviewer = getattr(reviewer, "actor_reviewer", None)
    actor_fallback_reviewer = getattr(actor_reviewer, "fallback_reviewer", None)
    action_owner_provenance = {}
    if isinstance(action_owner_model, ActionOwnerModel):
        artifact = action_owner_model.artifact
        action_owner_provenance = {
            "action_owner_model_sha256": artifact["model_sha256"],
            "action_owner_model_schema": artifact["schema_version"],
            "training_annotation_producer": str(artifact.get("training_annotation_producer") or ""),
            "training_manifest_sha256": str(artifact.get("training_manifest_sha256") or ""),
            "training_benchmark_overlap": str(artifact.get("training_benchmark_overlap") or ""),
        }
    return seal_agu_autonomous_predictions(
        game_id=source.game_id,
        raw_video_paths=[video_path],
        events=events,
        config={
            "pipeline": "agu_official_autonomous_v1",
            "source_candidate_config_sha256": source.config_sha256,
            "official_vlm_confidence": minimum_confidence,
            "official_vlm_frames": getattr(reviewer, "max_frames"),
            "official_vlm_image_width": getattr(reviewer, "image_width"),
            "official_vlm_context_length": getattr(reviewer, "context_length"),
            "official_vlm_contact_sheet": getattr(reviewer, "contact_sheet"),
            "official_vlm_two_pass": isinstance(reviewer, TwoPassOfficialEventReviewer),
            "official_vlm_review_mode": getattr(reviewer, "review_mode", "two_pass"),
            "official_vlm_actor_fallback_dispatch_count": int(
                getattr(actor_reviewer, "fallback_dispatch_count", 0)
            ),
            "official_vlm_actor_fallback_image_width": int(
                getattr(actor_fallback_reviewer, "image_width", 0)
            ),
            "official_vlm_actor_fallback_context_length": int(
                getattr(actor_fallback_reviewer, "context_length", 0)
            ),
            "official_vlm_actor_fallback_model": str(
                getattr(actor_reviewer, "fallback_model", "")
            ),
            "official_vlm_semantic_fallback_dispatch_count": int(
                getattr(semantic_reviewer, "fallback_dispatch_count", 0)
            ),
            "official_vlm_semantic_fallback_image_width": int(
                getattr(semantic_fallback_reviewer, "image_width", 0)
            ),
            "official_vlm_semantic_fallback_context_length": int(
                getattr(semantic_fallback_reviewer, "context_length", 0)
            ),
            "official_vlm_semantic_fallback_model": str(
                getattr(semantic_reviewer, "fallback_model", "")
            ),
            "action_owner_model_sha256": action_owner_provenance.get("action_owner_model_sha256", ""),
            "event_frame_sample_fps": sample_fps,
            "event_review_pre_seconds": review_pre_seconds,
            "event_review_post_seconds": review_post_seconds,
            "event_rim_detail_inset": rim_detail_inset,
        },
        candidate_backend=source.model_provenance.get("candidate_backend", "agu_traditional_perception"),
        semantic_backend=reviewer.name,
        semantic_model=reviewer.model,
        model_provenance={
            "source_candidate_bundle_sha256": source.bundle_sha256,
            "detector_model_sha256": source.model_provenance.get("detector_model_sha256", ""),
            **action_owner_provenance,
        },
    )


def main() -> int:
    args = parse_args()
    if (
        not 0 <= args.confidence <= 1
        or args.frames <= 0
        or args.sample_fps <= 0
        or args.review_pre_seconds < 0
        or args.review_post_seconds < 0
        or args.actor_pre_seconds < 0
        or args.actor_post_seconds < 0
    ):
        raise ValueError("confidence must be in [0,1]; frames/sample-fps must be positive; review bounds nonnegative")
    if args.two_pass_vlm and args.semantic_only_vlm:
        raise ValueError("--two-pass-vlm and --semantic-only-vlm are mutually exclusive")
    actor_fallback_values = (
        args.actor_fallback_image_width,
        args.actor_fallback_context_length,
    )
    actor_fallback_requested = any(value is not None for value in actor_fallback_values) or any(
        value is not None
        for value in (args.actor_fallback_model, args.actor_fallback_keep_alive)
    )
    if actor_fallback_requested:
        if not args.two_pass_vlm or any(value is None or value <= 0 for value in actor_fallback_values):
            raise ValueError("actor fallback requires two-pass VLM and two positive fallback settings")
        if args.actor_fallback_image_width > args.image_width:
            raise ValueError("actor fallback image width must not exceed the primary setting")
    if args.actor_fallback_keep_alive is not None and args.actor_fallback_keep_alive < 0:
        raise ValueError("actor fallback keep-alive must be nonnegative")
    semantic_fallback_values = (
        args.semantic_fallback_image_width,
        args.semantic_fallback_context_length,
    )
    semantic_fallback_requested = any(
        value is not None for value in semantic_fallback_values
    ) or any(
        value is not None
        for value in (args.semantic_fallback_model, args.semantic_fallback_keep_alive)
    )
    if semantic_fallback_requested:
        if not args.two_pass_vlm or any(
            value is None or value <= 0 for value in semantic_fallback_values
        ):
            raise ValueError("semantic fallback requires two-pass VLM and two positive fallback settings")
        if args.semantic_fallback_image_width > args.image_width:
            raise ValueError("semantic fallback image width must not exceed the primary setting")
    if args.semantic_fallback_keep_alive is not None and args.semantic_fallback_keep_alive < 0:
        raise ValueError("semantic fallback keep-alive must be nonnegative")
    source = RawOnlyPredictionBundleResponse.model_validate_json(args.candidate_bundle.read_text(encoding="utf-8"))
    cache_path = args.cache or args.output.with_suffix(".vlm-cache.json")
    if args.two_pass_vlm:
        action_owner_model = (
            ActionOwnerModel.from_path(args.action_owner_model) if args.action_owner_model is not None else None
        )
        actor_provider = VideoEventFrameProvider(
            args.video,
            sample_fps=args.sample_fps,
            review_pre_seconds=args.actor_pre_seconds,
            review_post_seconds=args.actor_post_seconds,
            overlay_identities=True,
            maximum_identity_overlays=args.maximum_identity_overlays,
            prefer_release_frame=True,
        )
        primary_semantic_reviewer = OllamaOfficialEventReviewer(
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            image_width=args.image_width,
            max_frames=args.frames,
            context_length=args.context_length,
            seed=args.seed,
            keep_alive=(
                args.ollama_semantic_keep_alive
                if args.ollama_semantic_keep_alive is not None
                else args.ollama_keep_alive
            ),
            cache_path=cache_path.with_name(f"{cache_path.stem}.semantic{cache_path.suffix}"),
            contact_sheet=args.contact_sheet,
            review_mode="event_semantics",
            rim_detail_inset=args.rim_detail_inset,
            frame_bounds_resolver=lambda event: _review_frame_bounds(
                event,
                source_fps=actor_provider.source_fps,
                pre_seconds=args.review_pre_seconds,
                post_seconds=args.review_post_seconds,
            ),
            frame_positions_resolver=VideoEventFrameProvider(
                args.video,
                sample_fps=args.sample_fps,
                review_pre_seconds=args.review_pre_seconds,
                review_post_seconds=args.review_post_seconds,
                overlay_identities=False,
                rim_detail_inset=args.rim_detail_inset,
            ).selected_frame_numbers,
        )
        semantic_reviewer: OfficialEventReviewer = primary_semantic_reviewer
        if args.semantic_fallback_image_width is not None:
            semantic_reviewer = CacheAwareFallbackOfficialEventReviewer(
                primary_reviewer=primary_semantic_reviewer,
                fallback_reviewer=OllamaOfficialEventReviewer(
                    model=args.semantic_fallback_model or args.model,
                    host=args.host,
                    timeout=args.timeout,
                    image_width=args.semantic_fallback_image_width,
                    max_frames=args.frames,
                    context_length=args.semantic_fallback_context_length,
                    seed=args.seed,
                    keep_alive=(
                        args.semantic_fallback_keep_alive
                        if args.semantic_fallback_keep_alive is not None
                        else args.ollama_semantic_keep_alive
                    ),
                    cache_path=cache_path.with_name(
                        f"{cache_path.stem}.semantic-fallback{cache_path.suffix}"
                    ),
                    contact_sheet=args.contact_sheet,
                    review_mode="event_semantics",
                    rim_detail_inset=args.rim_detail_inset,
                    frame_bounds_resolver=lambda event: _review_frame_bounds(
                        event,
                        source_fps=actor_provider.source_fps,
                        pre_seconds=args.review_pre_seconds,
                        post_seconds=args.review_post_seconds,
                    ),
                    frame_positions_resolver=VideoEventFrameProvider(
                        args.video,
                        sample_fps=args.sample_fps,
                        review_pre_seconds=args.review_pre_seconds,
                        review_post_seconds=args.review_post_seconds,
                        overlay_identities=False,
                        rim_detail_inset=args.rim_detail_inset,
                    ).selected_frame_numbers,
                ),
            )

        def actor_bounds_resolver(event: GameEventResponse) -> tuple[int, int]:
            return _review_frame_bounds(
                event,
                source_fps=actor_provider.source_fps,
                pre_seconds=args.actor_pre_seconds,
                post_seconds=args.actor_post_seconds,
                prefer_release_frame=True,
            )
        primary_actor_reviewer = OllamaOfficialEventReviewer(
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            image_width=args.image_width,
            max_frames=args.frames,
            context_length=args.context_length,
            seed=args.seed,
            keep_alive=args.ollama_keep_alive,
            cache_path=cache_path.with_name(f"{cache_path.stem}.actor{cache_path.suffix}"),
            contact_sheet=args.contact_sheet,
            review_mode="actor_identity",
            frame_bounds_resolver=actor_bounds_resolver,
            frame_positions_resolver=actor_provider.selected_frame_numbers,
        )
        actor_reviewer: OfficialEventReviewer = primary_actor_reviewer
        if args.actor_fallback_image_width is not None:
            actor_reviewer = CacheAwareFallbackOfficialEventReviewer(
                primary_reviewer=primary_actor_reviewer,
                fallback_reviewer=OllamaOfficialEventReviewer(
                    model=args.actor_fallback_model or args.model,
                    host=args.host,
                    timeout=args.timeout,
                    image_width=args.actor_fallback_image_width,
                    max_frames=args.frames,
                    context_length=args.actor_fallback_context_length,
                    seed=args.seed,
                    keep_alive=(
                        args.actor_fallback_keep_alive
                        if args.actor_fallback_keep_alive is not None
                        else args.ollama_keep_alive
                    ),
                    cache_path=cache_path.with_name(
                        f"{cache_path.stem}.actor-fallback{cache_path.suffix}"
                    ),
                    contact_sheet=args.contact_sheet,
                    review_mode="actor_identity",
                    frame_bounds_resolver=actor_bounds_resolver,
                    frame_positions_resolver=actor_provider.selected_frame_numbers,
                ),
            )
        reviewer: OfficialEventReviewer = TwoPassOfficialEventReviewer(
            semantic_reviewer=semantic_reviewer,
            actor_reviewer=actor_reviewer,
            actor_frame_provider=actor_provider,
            action_owner_model=action_owner_model,
        )
    else:
        semantic_provider = (
            VideoEventFrameProvider(
                args.video,
                sample_fps=args.sample_fps,
                review_pre_seconds=args.review_pre_seconds,
                review_post_seconds=args.review_post_seconds,
                overlay_identities=False,
                rim_detail_inset=args.rim_detail_inset,
            )
            if args.semantic_only_vlm
            else None
        )
        reviewer = OllamaOfficialEventReviewer(
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            image_width=args.image_width,
            max_frames=args.frames,
            context_length=args.context_length,
            seed=args.seed,
            keep_alive=args.ollama_keep_alive,
            cache_path=cache_path,
            contact_sheet=args.contact_sheet,
            review_mode="event_semantics" if args.semantic_only_vlm else "combined",
            rim_detail_inset=args.rim_detail_inset and args.semantic_only_vlm,
            frame_bounds_resolver=(
                lambda event: _review_frame_bounds(
                    event,
                    source_fps=semantic_provider.source_fps,
                    pre_seconds=args.review_pre_seconds,
                    post_seconds=args.review_post_seconds,
                )
            )
            if semantic_provider is not None
            else None,
            frame_positions_resolver=(
                semantic_provider.selected_frame_numbers
                if semantic_provider is not None
                else None
            ),
        )
    result = run_autonomous_inference(
        candidate_bundle=source,
        video_path=args.video,
        reviewer=reviewer,
        minimum_confidence=args.confidence,
        sample_fps=args.sample_fps,
        review_pre_seconds=args.review_pre_seconds,
        review_post_seconds=args.review_post_seconds,
        overlay_identities=not (args.two_pass_vlm or args.semantic_only_vlm),
        rim_detail_inset=args.rim_detail_inset and (args.two_pass_vlm or args.semantic_only_vlm),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for event in result.events:
        counts[event.status] = counts.get(event.status, 0) + 1
    print(json.dumps({"bundle_sha256": result.bundle_sha256, "status_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
