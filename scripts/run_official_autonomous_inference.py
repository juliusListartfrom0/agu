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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--contact-sheet", action=argparse.BooleanOptionalAction, default=settings.official_vlm_contact_sheet
    )
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--review-pre-seconds", type=float, default=3.0)
    parser.add_argument("--review-post-seconds", type=float, default=3.0)
    parser.add_argument("--actor-pre-seconds", type=float, default=2.0)
    parser.add_argument("--actor-post-seconds", type=float, default=1.5)
    parser.add_argument("--maximum-identity-overlays", type=int, default=2)
    parser.add_argument("--two-pass-vlm", action=argparse.BooleanOptionalAction, default=False)
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
    ) -> None:
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.review_pre_seconds = review_pre_seconds
        self.review_post_seconds = review_post_seconds
        self.overlay_identities = overlay_identities
        self.maximum_identity_overlays = maximum_identity_overlays
        self.prefer_release_frame = prefer_release_frame
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
            "action_owner_model_sha256": action_owner_provenance.get("action_owner_model_sha256", ""),
            "event_frame_sample_fps": sample_fps,
            "event_review_pre_seconds": review_pre_seconds,
            "event_review_post_seconds": review_post_seconds,
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
        semantic_reviewer = OllamaOfficialEventReviewer(
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            image_width=args.image_width,
            max_frames=args.frames,
            context_length=args.context_length,
            seed=args.seed,
            cache_path=cache_path.with_name(f"{cache_path.stem}.semantic{cache_path.suffix}"),
            contact_sheet=args.contact_sheet,
            review_mode="event_semantics",
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
            ).selected_frame_numbers,
        )
        reviewer: OfficialEventReviewer = TwoPassOfficialEventReviewer(
            semantic_reviewer=semantic_reviewer,
            actor_reviewer=OllamaOfficialEventReviewer(
                model=args.model,
                host=args.host,
                timeout=args.timeout,
                image_width=args.image_width,
                max_frames=args.frames,
                context_length=args.context_length,
                seed=args.seed,
                cache_path=cache_path.with_name(f"{cache_path.stem}.actor{cache_path.suffix}"),
                contact_sheet=args.contact_sheet,
                review_mode="actor_identity",
                frame_bounds_resolver=lambda event: _review_frame_bounds(
                    event,
                    source_fps=actor_provider.source_fps,
                    pre_seconds=args.actor_pre_seconds,
                    post_seconds=args.actor_post_seconds,
                    prefer_release_frame=True,
                ),
                frame_positions_resolver=actor_provider.selected_frame_numbers,
            ),
            actor_frame_provider=actor_provider,
            action_owner_model=action_owner_model,
        )
    else:
        reviewer = OllamaOfficialEventReviewer(
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            image_width=args.image_width,
            max_frames=args.frames,
            context_length=args.context_length,
            seed=args.seed,
            cache_path=cache_path,
            contact_sheet=args.contact_sheet,
        )
    result = run_autonomous_inference(
        candidate_bundle=source,
        video_path=args.video,
        reviewer=reviewer,
        minimum_confidence=args.confidence,
        sample_fps=args.sample_fps,
        review_pre_seconds=args.review_pre_seconds,
        review_post_seconds=args.review_post_seconds,
        overlay_identities=not args.two_pass_vlm,
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
