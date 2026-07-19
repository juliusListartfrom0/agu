#!/usr/bin/env python3
"""Export candidate-grounded contact sheets for training-only Codex review."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.official_inference import candidate_player_aliases  # noqa: E402
from app.analysis.official_visuals import build_temporal_contact_sheet  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402
from scripts.run_official_autonomous_inference import (  # noqa: E402
    VideoEventFrameProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--pre-seconds", type=float, default=6.0)
    parser.add_argument("--post-seconds", type=float, default=1.0)
    parser.add_argument("--maximum-overlays", type=int, default=8)
    parser.add_argument("--candidate-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--maximum-crops-per-candidate", type=int, default=6)
    return parser.parse_args()


def export_review_sheets(
    *,
    candidate_bundle_path: Path,
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    pre_seconds: float,
    post_seconds: float,
    maximum_overlays: int,
    candidate_crops: bool = True,
    maximum_crops_per_candidate: int = 6,
) -> dict[str, object]:
    bundle = verify_raw_only_bundle(
        RawOnlyPredictionBundleResponse.model_validate_json(
            candidate_bundle_path.read_text(encoding="utf-8")
        )
    )
    if len(bundle.raw_videos) != 1 or bundle.raw_videos[0].filename != video_path.name:
        raise ValueError("candidate bundle does not match the review video")
    if (
        sample_fps <= 0
        or pre_seconds < 0
        or post_seconds < 0
        or maximum_overlays <= 0
        or maximum_crops_per_candidate <= 0
    ):
        raise ValueError("review sampling parameters must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = VideoEventFrameProvider(
        video_path,
        sample_fps=sample_fps,
        review_pre_seconds=pre_seconds,
        review_post_seconds=post_seconds,
        overlay_identities=True,
        maximum_identity_overlays=maximum_overlays,
    )
    records = []
    for event in bundle.events:
        if event.event_type != "field_goal_attempt":
            continue
        frames = list(provider(event))
        sheet = build_temporal_contact_sheet(
            frames,
            duration_sec=pre_seconds + post_seconds,
            columns=4,
            cell_width=480,
        )
        if sheet is None:
            continue
        filename = f"{event.event_id}.jpg"
        if not cv2.imwrite(str(output_dir / filename), sheet):
            raise RuntimeError(f"unable to write review sheet: {filename}")
        aliases = candidate_player_aliases(event)
        crop_sheets = (
            _export_candidate_crop_sheets(
                video_path=video_path,
                output_dir=output_dir,
                event_id=event.event_id,
                observations=_candidate_observations(event),
                aliases=aliases,
                maximum_crops_per_candidate=maximum_crops_per_candidate,
            )
            if candidate_crops
            else []
        )
        records.append(
            {
                "event_id": event.event_id,
                "sheet": filename,
                "outcome_frame": event.outcome_frame,
                "candidate_aliases": aliases,
                "candidate_crop_sheets": crop_sheets,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "agu.action-owner-review-sheets.v2",
        "purpose": "model_training_only",
        "producer": "agu",
        "runtime_consumable": False,
        "candidate_bundle_sha256": bundle.bundle_sha256,
        "video_sha256": bundle.raw_videos[0].sha256,
        "sample_fps": sample_fps,
        "pre_seconds": pre_seconds,
        "post_seconds": post_seconds,
        "candidate_crops": candidate_crops,
        "maximum_crops_per_candidate": maximum_crops_per_candidate,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _candidate_observations(event: object) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for evidence in getattr(event, "evidence", []):
        values = evidence.details.get("candidate_player_observations") or []
        if isinstance(values, list):
            observations.extend(item for item in values if isinstance(item, dict))
    return observations


def _export_candidate_crop_sheets(
    *,
    video_path: Path,
    output_dir: Path,
    event_id: str,
    observations: list[dict[str, Any]],
    aliases: dict[str, str],
    maximum_crops_per_candidate: int,
) -> list[dict[str, object]]:
    by_player: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        player_id = str(observation.get("player_id") or "")
        if player_id in aliases:
            by_player.setdefault(player_id, []).append(observation)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    records: list[dict[str, object]] = []
    try:
        for player_id, alias in sorted(aliases.items(), key=lambda item: item[1]):
            selected = _select_temporal_observations(
                by_player.get(player_id, []), maximum=maximum_crops_per_candidate
            )
            crops = [
                crop
                for observation in selected
                if (crop := _read_annotated_crop(capture, observation, alias=alias)) is not None
            ]
            sheet = build_temporal_contact_sheet(crops, columns=3, cell_width=320)
            if sheet is None:
                continue
            filename = f"{event_id}__{alias}.jpg"
            if not cv2.imwrite(str(output_dir / filename), sheet):
                raise RuntimeError(f"unable to write candidate crop sheet: {filename}")
            records.append(
                {
                    "player_id": player_id,
                    "alias": alias,
                    "sheet": filename,
                    "frames": [int(item["frame"]) for item in selected],
                }
            )
    finally:
        capture.release()
    return records


def _select_temporal_observations(
    observations: list[dict[str, Any]], *, maximum: int
) -> list[dict[str, Any]]:
    ordered = sorted(
        observations,
        key=lambda item: (
            int(item.get("frame") or 0),
            float(item.get("ball_player_distance", float("inf"))),
        ),
    )
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return [min(ordered, key=lambda item: float(item.get("ball_player_distance", float("inf"))))]
    indexes = {
        round(index * (len(ordered) - 1) / (maximum - 1)) for index in range(maximum)
    }
    return [ordered[index] for index in sorted(indexes)]


def _read_annotated_crop(
    capture: cv2.VideoCapture, observation: dict[str, Any], *, alias: str
) -> Any | None:
    try:
        frame_number = int(observation["frame"])
        bbox = observation["bbox"]
        x1, y1, x2, y2 = (float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    if not ok:
        return None
    height, width = frame.shape[:2]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    padding_x = box_width * 0.75
    padding_y = box_height * 0.35
    crop_x1 = max(0, int(math.floor(x1 - padding_x)))
    crop_y1 = max(0, int(math.floor(y1 - padding_y)))
    crop_x2 = min(width, int(math.ceil(x2 + padding_x)))
    crop_y2 = min(height, int(math.ceil(y2 + padding_y)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    cv2.rectangle(
        crop,
        (int(x1) - crop_x1, int(y1) - crop_y1),
        (int(x2) - crop_x1, int(y2) - crop_y1),
        (0, 255, 0),
        2,
    )
    ball_center = observation.get("ball_center") or {}
    try:
        ball_x = int(round(float(ball_center["x"]))) - crop_x1
        ball_y = int(round(float(ball_center["y"]))) - crop_y1
        if 0 <= ball_x < crop.shape[1] and 0 <= ball_y < crop.shape[0]:
            cv2.circle(crop, (ball_x, ball_y), 7, (0, 0, 255), 2)
    except (KeyError, TypeError, ValueError):
        pass
    distance = observation.get("ball_player_distance")
    label = (
        f"{alias} frame={frame_number} d={float(distance):.2f}"
        if distance is not None
        else f"{alias} frame={frame_number}"
    )
    cv2.rectangle(crop, (0, 0), (min(crop.shape[1] - 1, 290), 28), (0, 0, 0), -1)
    cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def main() -> int:
    args = parse_args()
    manifest = export_review_sheets(
        candidate_bundle_path=args.candidate_bundle,
        video_path=args.video,
        output_dir=args.output_dir,
        sample_fps=args.sample_fps,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
        maximum_overlays=args.maximum_overlays,
        candidate_crops=args.candidate_crops,
        maximum_crops_per_candidate=args.maximum_crops_per_candidate,
    )
    print(json.dumps({"sheets": len(manifest["records"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
