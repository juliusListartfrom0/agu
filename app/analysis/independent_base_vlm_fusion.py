"""Strict development-only fusion of AGU v3 player actions and an independent VLM."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r2plus1d_18

from app.analysis.independent_shot_vlm import (
    parse_independent_shot_vlm_decision,
    seal_independent_shot_vlm_predictions,
    verify_independent_shot_vlm_plan,
    verify_independent_shot_vlm_predictions,
)
from app.analysis.inference import LABELS
from app.analysis.official_evaluation import verify_raw_only_bundle
from app.analysis.schemas import RawOnlyPredictionBundleResponse
from app.models.preprocessing import preprocess_clip_frames

INDEPENDENT_BASE_PREDICTIONS_SCHEMA = "agu.independent-base-shot-predictions.v1"
_SHA256_LENGTH = 64


def select_candidate_player_observations(
    evidence: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select unique candidate players by wrist/ball proximity without labels."""

    if limit < 1:
        raise ValueError("candidate player limit must be positive")
    by_player: dict[str, dict[str, Any]] = {}
    for item in evidence:
        details = item.get("details")
        if not isinstance(details, Mapping):
            continue
        rows = details.get("candidate_player_observations")
        if not isinstance(rows, list):
            continue
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            player_id = str(raw.get("player_id") or "")
            bbox = raw.get("bbox")
            frame = int(raw.get("frame", -1))
            if (
                not player_id
                or frame < 0
                or not isinstance(bbox, Mapping)
                or not _valid_bbox(bbox)
            ):
                continue
            row = {
                "player_id": player_id,
                "frame": frame,
                "bbox": {
                    name: float(bbox[name])
                    for name in ("x1", "y1", "x2", "y2")
                },
                "wrist_ball_distance": _finite_or_inf(
                    raw.get("wrist_ball_distance")
                ),
                "ball_player_distance": _finite_or_inf(
                    raw.get("ball_player_distance")
                ),
            }
            prior = by_player.get(player_id)
            if prior is None or _observation_rank(row) < _observation_rank(prior):
                by_player[player_id] = row
    return sorted(by_player.values(), key=_observation_rank)[:limit]


def centered_clip_frame_indexes(
    *,
    center_frame: int,
    start_frame: int,
    end_frame: int,
    clip_frames: int,
) -> list[int]:
    """Return one consecutive player-action clip bounded by the event."""

    if clip_frames < 2 or start_frame < 0 or end_frame < start_frame:
        raise ValueError("invalid base action clip bounds")
    available = end_frame - start_frame + 1
    if available < clip_frames:
        raise ValueError("event is shorter than the base action clip")
    start = center_frame - clip_frames // 2
    start = max(start_frame, min(start, end_frame - clip_frames + 1))
    return list(range(start, start + clip_frames))


def load_agu_v3_action_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the canonical ten-class v3 checkpoint with tensor-only deserialization."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise ValueError("AGU v3 checkpoint must contain a tensor state dict")
    model = r2plus1d_18(weights=None, progress=False)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, {
        "name": "agu-v3-r2plus1d-player-action",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "source_validation_accuracy": float(
            checkpoint.get("best_val_acc", 0.0)
        ),
        "labels": [LABELS[index] for index in sorted(LABELS)],
    }


def run_agu_v3_base_plan(
    *,
    plan: Mapping[str, Any],
    bundle_paths: Sequence[Path],
    video_paths: Sequence[Path],
    checkpoint_path: Path,
    device: torch.device,
    clip_frames: int = 16,
    maximum_player_clips: int = 3,
    bbox_expansion: float = 0.2,
) -> dict[str, Any]:
    """Run current AGU player-action evidence on the exact frozen VLM plan."""

    verified = verify_independent_shot_vlm_plan(plan)
    if (
        clip_frames < 2
        or maximum_player_clips < 1
        or not 0.0 <= bbox_expansion <= 1.0
    ):
        raise ValueError("invalid AGU base extraction contract")
    required_video_hashes = {
        str(row["source_video_sha256"]) for row in verified["examples"]
    }
    videos = {file_sha256(path): path for path in video_paths}
    if set(videos) != required_video_hashes:
        raise ValueError("base videos do not exactly match the frozen plan")
    bundles: dict[str, RawOnlyPredictionBundleResponse] = {}
    for path in bundle_paths:
        bundle = verify_raw_only_bundle(
            RawOnlyPredictionBundleResponse.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        )
        bundles[str(bundle.bundle_sha256)] = bundle
    required_bundle_hashes = {
        str(row["candidate_bundle_sha256"]) for row in verified["examples"]
    }
    if set(bundles) != required_bundle_hashes:
        raise ValueError("base candidate bundles do not exactly match the plan")
    model, model_info = load_agu_v3_action_model(
        checkpoint_path,
        device=device,
    )
    captures: dict[str, cv2.VideoCapture] = {}
    predictions = []
    try:
        for row in verified["examples"]:
            video_sha = str(row["source_video_sha256"])
            bundle_sha = str(row["candidate_bundle_sha256"])
            bundle = bundles[bundle_sha]
            if len(bundle.raw_videos) != 1 or bundle.raw_videos[0].sha256 != video_sha:
                raise ValueError("candidate bundle video does not match the plan")
            events = {event.event_id: event for event in bundle.events}
            event = events.get(str(row["event_id"]))
            if (
                event is None
                or event.start_frame != int(row["start_frame"])
                or event.end_frame != int(row["end_frame"])
            ):
                raise ValueError("base event bounds do not match the frozen plan")
            observations = select_candidate_player_observations(
                [item.model_dump(mode="json") for item in event.evidence],
                limit=maximum_player_clips,
            )
            capture = captures.get(video_sha)
            if capture is None:
                capture = cv2.VideoCapture(str(videos[video_sha]))
                if not capture.isOpened():
                    raise ValueError("cannot open AGU base source video")
                captures[video_sha] = capture
            clips = []
            used = []
            for observation in observations:
                try:
                    indexes = centered_clip_frame_indexes(
                        center_frame=int(observation["frame"]),
                        start_frame=event.start_frame,
                        end_frame=event.end_frame,
                        clip_frames=clip_frames,
                    )
                    clip = _read_static_player_clip(
                        capture,
                        frame_indexes=indexes,
                        bbox=observation["bbox"],
                        expansion=bbox_expansion,
                    )
                except ValueError:
                    continue
                clips.append(clip)
                used.append(observation)
            player_rows: list[dict[str, Any]] = []
            if clips:
                batch = torch.from_numpy(
                    np.stack(
                        [
                            preprocess_clip_frames(list(clip))
                            for clip in clips
                        ]
                    )
                ).to(device)
                with torch.inference_mode():
                    probabilities = torch.softmax(model(batch), dim=1)
                values = probabilities.detach().cpu().to(torch.float32).numpy()
                for observation, scores in zip(used, values, strict=True):
                    action_id = int(np.argmax(scores))
                    player_rows.append(
                        {
                            "player_id": observation["player_id"],
                            "observation_frame": observation["frame"],
                            "bbox": observation["bbox"],
                            "action": LABELS[action_id],
                            "action_confidence": float(scores[action_id]),
                            "shoot_probability": float(scores[4]),
                        }
                    )
            has_shoot = any(item["action"] == "shoot" for item in player_rows)
            max_shoot = max(
                (float(item["shoot_probability"]) for item in player_rows),
                default=0.0,
            )
            predictions.append(
                {
                    "source_video_sha256": video_sha,
                    "candidate_bundle_sha256": bundle_sha,
                    "event_id": row["event_id"],
                    "available": bool(player_rows),
                    "field_goal_state": (
                        "live_field_goal"
                        if has_shoot
                        else "not_field_goal"
                        if player_rows
                        else "unknown"
                    ),
                    "shoot_probability": max_shoot,
                    "confidence": (
                        max_shoot
                        if has_shoot
                        else 1.0 - max_shoot
                        if player_rows
                        else 0.0
                    ),
                    "player_clips": player_rows,
                }
            )
    finally:
        for capture in captures.values():
            capture.release()
    return seal_independent_base_predictions(
        {
            "plan_sha256": verified["plan_sha256"],
            "model": model_info,
            "input_contract": {
                "clip_frames": clip_frames,
                "maximum_player_clips": maximum_player_clips,
                "bbox_expansion": bbox_expansion,
                "player_source": (
                    "candidate_ball_wrist_proximity_static_bbox"
                ),
                "decision": "any_player_argmax_shoot",
                "preprocessing": "agu_v3_bgr_0_255_112_no_normalization",
            },
            "predictions": predictions,
        }
    )


def seal_independent_base_predictions(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = json.loads(json.dumps(payload))
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = INDEPENDENT_BASE_PREDICTIONS_SCHEMA
    artifact["purpose"] = "offline_development_base_model_screening"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    _validate_base_predictions(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_independent_base_predictions(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = json.loads(json.dumps(payload))
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_base_predictions(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("independent base prediction hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def fuse_base_and_vlm_predictions(
    *,
    plan: Mapping[str, Any],
    base_predictions: Mapping[str, Any],
    vlm_predictions: Mapping[str, Any],
    rule: str,
) -> dict[str, Any]:
    """Fuse two already frozen sources with one named, non-learned rule."""

    verified_plan = verify_independent_shot_vlm_plan(plan)
    base = verify_independent_base_predictions(base_predictions)
    vlm = verify_independent_shot_vlm_predictions(vlm_predictions)
    if (
        base.get("plan_sha256") != verified_plan["plan_sha256"]
        or vlm.get("plan_sha256") != verified_plan["plan_sha256"]
    ):
        raise ValueError("base/VLM artifacts do not match the frozen plan")
    if rule not in {"base_only", "vlm_only", "both_confirm", "either_confirms"}:
        raise ValueError("unsupported base/VLM fusion rule")
    expected = {_example_key(row) for row in verified_plan["examples"]}
    base_by_key = {_example_key(row): row for row in base["predictions"]}
    vlm_by_key = {_example_key(row): row for row in vlm["predictions"]}
    if set(base_by_key) != expected or set(vlm_by_key) != expected:
        raise ValueError("base and VLM predictions must exactly cover the plan")
    rows = []
    for planned in verified_plan["examples"]:
        key = _example_key(planned)
        base_row = base_by_key[key]
        vlm_row = vlm_by_key[key]
        base_state = str(base_row["field_goal_state"])
        vlm_state = str(vlm_row["field_goal_state"])
        state = _fused_state(base_state, vlm_state, rule=rule)
        decision = parse_independent_shot_vlm_decision(
            {
                "field_goal_state": state,
                "confidence": _fused_confidence(
                    float(base_row.get("confidence", 0.0)),
                    float(vlm_row.get("confidence", 0.0)),
                    state=state,
                    rule=rule,
                ),
                "reason": (
                    f"fixed {rule}: base={base_state}; vlm={vlm_state}"
                ),
            }
        )
        rows.append(
            {
                "source_video_sha256": key[0],
                "candidate_bundle_sha256": key[1],
                "event_id": key[2],
                **decision,
            }
        )
    return seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": verified_plan["plan_sha256"],
            "model": {
                "name": f"agu-v3-plus-independent-vlm:{rule}",
                "independent_from_codex": True,
                "base_prediction_artifact_sha256": base["artifact_sha256"],
                "vlm_prediction_artifact_sha256": vlm["artifact_sha256"],
                "fusion_rule": rule,
                "learned_on_target_labels": False,
            },
            "predictions": rows,
        }
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_static_player_clip(
    capture: cv2.VideoCapture,
    *,
    frame_indexes: Sequence[int],
    bbox: Mapping[str, Any],
    expansion: float,
) -> np.ndarray:
    frames = []
    for frame_index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("cannot decode AGU base frame")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = _expanded_bbox(
            bbox,
            width=width,
            height=height,
            expansion=expansion,
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("AGU base player crop is empty")
        frames.append(crop)
    return np.stack(frames)


def _expanded_bbox(
    bbox: Mapping[str, Any],
    *,
    width: int,
    height: int,
    expansion: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(bbox[name]) for name in ("x1", "y1", "x2", "y2"))
    grow_x = (x2 - x1) * expansion
    grow_y = (y2 - y1) * expansion
    left = max(0, int(math.floor(x1 - grow_x)))
    top = max(0, int(math.floor(y1 - grow_y)))
    right = min(width, int(math.ceil(x2 + grow_x)))
    bottom = min(height, int(math.ceil(y2 + grow_y)))
    if right <= left or bottom <= top:
        raise ValueError("expanded AGU base bbox is empty")
    return left, top, right, bottom


def _validate_base_predictions(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != INDEPENDENT_BASE_PREDICTIONS_SCHEMA
        or artifact.get("purpose")
        != "offline_development_base_model_screening"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("independent base prediction boundary is invalid")
    _require_sha256(artifact.get("plan_sha256"), "plan")
    model = artifact.get("model")
    if not isinstance(model, Mapping) or not str(model.get("name") or ""):
        raise ValueError("independent base model provenance is missing")
    _require_sha256(model.get("checkpoint_sha256"), "base checkpoint")
    contract = artifact.get("input_contract")
    if (
        not isinstance(contract, Mapping)
        or int(contract.get("clip_frames", 0)) < 2
        or int(contract.get("maximum_player_clips", 0)) < 1
        or contract.get("decision") != "any_player_argmax_shoot"
    ):
        raise ValueError("independent base input contract is invalid")
    rows = artifact.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("independent base predictions are missing")
    maximum_player_clips = int(contract["maximum_player_clips"])
    seen = set()
    for row in rows:
        key = _example_key(row)
        player_clips = row.get("player_clips")
        state = str(row.get("field_goal_state") or "")
        if (
            key in seen
            or state
            not in {"live_field_goal", "not_field_goal", "unknown"}
            or not isinstance(row.get("available"), bool)
            or not isinstance(player_clips, list)
            or not 0.0 <= float(row.get("confidence", -1.0)) <= 1.0
            or not 0.0
            <= float(row.get("shoot_probability", -1.0))
            <= 1.0
            or "event_present" in row
            or "target" in row
        ):
            raise ValueError("independent base prediction row is invalid")
        if bool(player_clips) != row["available"]:
            raise ValueError(
                "base availability must match the presence of player clips"
            )
        if len(player_clips) > maximum_player_clips:
            raise ValueError("base row exceeds the player clip limit")
        if not row["available"] and state != "unknown":
            raise ValueError("unavailable independent base row must be unknown")
        player_ids = set()
        for clip in player_clips:
            if not isinstance(clip, Mapping):
                raise ValueError("independent base player clip is invalid")
            player_id = str(clip.get("player_id") or "")
            action = str(clip.get("action") or "")
            if (
                not player_id
                or player_id in player_ids
                or action not in LABELS.values()
                or not 0.0
                <= float(clip.get("action_confidence", -1.0))
                <= 1.0
                or not 0.0
                <= float(clip.get("shoot_probability", -1.0))
                <= 1.0
            ):
                raise ValueError("independent base player clip is invalid")
            player_ids.add(player_id)
        seen.add(key)


def _fused_state(base: str, vlm: str, *, rule: str) -> str:
    if rule == "base_only":
        return base
    if rule == "vlm_only":
        return vlm
    if rule == "both_confirm":
        if "not_field_goal" in {base, vlm}:
            return "not_field_goal"
        if base == vlm == "live_field_goal":
            return "live_field_goal"
        return "unknown"
    if "live_field_goal" in {base, vlm}:
        return "live_field_goal"
    if base == vlm == "not_field_goal":
        return "not_field_goal"
    return "unknown"


def _fused_confidence(
    base: float,
    vlm: float,
    *,
    state: str,
    rule: str,
) -> float:
    if state == "unknown":
        return 0.0
    if rule == "base_only":
        return base
    if rule == "vlm_only":
        return vlm
    return min(base, vlm) if rule == "both_confirm" else max(base, vlm)


def _observation_rank(row: Mapping[str, Any]) -> tuple[float, float, str]:
    return (
        float(row["wrist_ball_distance"]),
        float(row["ball_player_distance"]),
        str(row["player_id"]),
    )


def _finite_or_inf(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def _valid_bbox(bbox: Mapping[str, Any]) -> bool:
    try:
        values = [float(bbox[name]) for name in ("x1", "y1", "x2", "y2")]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values) and (
        values[2] > values[0] and values[3] > values[1]
    )


def _example_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    key = (
        str(row.get("source_video_sha256") or ""),
        str(row.get("candidate_bundle_sha256") or ""),
        str(row.get("event_id") or ""),
    )
    if not all(key):
        raise ValueError("independent base/VLM example key is incomplete")
    return key


def _require_sha256(value: object, label: str) -> None:
    text = str(value or "")
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} SHA-256 is invalid")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
