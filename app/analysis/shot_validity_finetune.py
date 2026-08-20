"""End-to-end R(2+1)D fine-tuning utilities for raw shot windows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r2plus1d_18

from app.analysis.official_evaluation import verify_raw_only_bundle
from app.analysis.schemas import GameEventResponse, RawOnlyPredictionBundleResponse
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA
from app.analysis.shot_validity_temporal import file_sha256
from app.analysis.training_annotation import verify_training_annotation_manifest

FINETUNED_SHOT_MODEL_SCHEMA = "agu.shot-validity-r2plus1d-model.v1"
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


@dataclass(frozen=True)
class ShotWindowRecord:
    source_video_sha256: str
    candidate_bundle_sha256: str
    event_id: str
    event_present: bool
    start_frame: int
    end_frame: int
    video_path: Path
    anchor_frame: int | None = None
    anchor_source: str = "none"


def resolve_training_anchor(event: GameEventResponse) -> tuple[int | None, str]:
    """Resolve a raw-evidence anchor without consulting the reviewed label."""

    if event.release_frame is not None and event.start_frame <= event.release_frame <= event.end_frame:
        return int(event.release_frame), "release_frame"
    for evidence in event.evidence:
        details = evidence.details
        if not isinstance(details, Mapping):
            continue
        value = details.get("candidate_event_frame")
        if value is None or isinstance(value, bool):
            continue
        try:
            anchor = int(value)
        except (TypeError, ValueError):
            continue
        if event.start_frame <= anchor <= event.end_frame:
            return anchor, "candidate_event_frame"
    return None, "none"


def anchor_window_bounds(
    record: ShotWindowRecord,
    *,
    context_frames: int,
) -> tuple[int, int]:
    """Return a bounded raw-frame context centered on the training anchor."""

    if context_frames < 0:
        raise ValueError("anchor context must be non-negative")
    if record.anchor_frame is None or context_frames == 0:
        return record.start_frame, record.end_frame
    anchor = min(max(record.anchor_frame, record.start_frame), record.end_frame)
    start = max(record.start_frame, anchor - context_frames)
    end = min(record.end_frame, anchor + context_frames)
    return start, end


def load_training_records(
    *,
    manifest_path: Path,
    candidate_bundle_paths: Sequence[Path],
    annotation_paths: Sequence[Path],
    video_paths: Sequence[Path],
) -> tuple[dict[str, Any], list[ShotWindowRecord]]:
    """Load only SHA-authorized labels paired with their raw candidate windows."""

    if not (
        len(candidate_bundle_paths) == len(annotation_paths) == len(video_paths)
    ):
        raise ValueError("candidate bundles, annotations, and videos must be paired")
    manifest = verify_training_annotation_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if "shot_validity" not in manifest.get("task_types", []):
        raise ValueError("training manifest does not authorize shot_validity")
    allowed_sources = {str(item["sha256"]) for item in manifest["source_videos"]}
    allowed_annotations = {
        (str(item["filename"]), str(item["sha256"]))
        for item in manifest["annotation_files"]
    }
    records: list[ShotWindowRecord] = []
    for bundle_path, annotation_path, video_path in zip(
        candidate_bundle_paths, annotation_paths, video_paths, strict=True
    ):
        bundle = verify_raw_only_bundle(
            RawOnlyPredictionBundleResponse.model_validate_json(
                bundle_path.read_text(encoding="utf-8")
            )
        )
        if len(bundle.raw_videos) != 1:
            raise ValueError("shot fine-tuning requires one video per candidate bundle")
        source_hash = file_sha256(video_path)
        if source_hash not in allowed_sources or bundle.raw_videos[0].sha256 != source_hash:
            raise ValueError("video is not SHA-bound to bundle and training manifest")
        annotation_hash = file_sha256(annotation_path)
        if (annotation_path.name, annotation_hash) not in allowed_annotations:
            raise ValueError("annotation is not SHA-bound by training manifest")
        labels = json.loads(annotation_path.read_text(encoding="utf-8"))
        if labels.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA:
            raise ValueError("unsupported shot-validity labels")
        if labels.get("runtime_consumable") is not False:
            raise ValueError("training annotations must not be runtime-consumable")
        if labels.get("source_video_sha256") != source_hash:
            raise ValueError("annotation source video hash mismatch")
        if labels.get("candidate_bundle_sha256") != bundle.bundle_sha256:
            raise ValueError("annotation candidate bundle hash mismatch")
        events_by_id = {event.event_id: event for event in bundle.events}
        for row in labels.get("examples", []):
            event = events_by_id.get(str(row.get("event_id") or ""))
            if event is None or event.event_type != "field_goal_attempt":
                raise ValueError("shot label does not name a field-goal candidate")
            if not isinstance(row.get("event_present"), bool):
                raise ValueError("shot labels must be boolean")
            anchor_frame, anchor_source = resolve_training_anchor(event)
            records.append(
                ShotWindowRecord(
                    source_video_sha256=source_hash,
                    candidate_bundle_sha256=str(bundle.bundle_sha256),
                    event_id=event.event_id,
                    event_present=bool(row["event_present"]),
                    start_frame=event.start_frame,
                    end_frame=event.end_frame,
                    video_path=video_path,
                    anchor_frame=anchor_frame,
                    anchor_source=anchor_source,
                )
            )
    if not records:
        raise ValueError("shot fine-tuning requires labeled records")
    return manifest, records


def decode_dense_windows(
    records: Sequence[ShotWindowRecord],
    *,
    dense_frames: int = 32,
    resize_height: int = 128,
    resize_width: int = 171,
) -> list[torch.Tensor]:
    """Decode raw windows once as compact uint8 tensors shaped ``T,C,H,W``."""

    if dense_frames < 16 or resize_height < 112 or resize_width < 112:
        raise ValueError("dense window and resize dimensions are too small")
    captures: dict[Path, cv2.VideoCapture] = {}
    windows: list[torch.Tensor] = []
    try:
        for record in records:
            capture = captures.get(record.video_path)
            if capture is None:
                capture = cv2.VideoCapture(str(record.video_path))
                if not capture.isOpened():
                    raise ValueError(f"cannot open source video: {record.video_path.name}")
                captures[record.video_path] = capture
            indexes = np.rint(
                np.linspace(record.start_frame, record.end_frame, dense_frames)
            ).astype(int)
            frames = []
            for frame_index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"cannot decode source frame {frame_index}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb, (resize_width, resize_height)))
            windows.append(torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2))
    finally:
        for capture in captures.values():
            capture.release()
    return windows


def prepare_window(
    window: torch.Tensor,
    *,
    training: bool,
    clip_frames: int = 16,
    crop_size: int = 112,
    generator: torch.Generator | None = None,
    preprocessing: str = "kinetics",
    temporal_sampling: str = "uniform",
    anchor_fraction: float | None = None,
) -> torch.Tensor:
    """Apply train-only temporal/spatial broadcast augmentation and normalize."""

    if preprocessing not in {"kinetics", "agu_v3"}:
        raise ValueError("preprocessing must be kinetics or agu_v3")
    if temporal_sampling not in {"uniform", "anchor"}:
        raise ValueError("temporal_sampling must be uniform or anchor")
    if window.ndim != 4 or window.shape[1] != 3:
        raise ValueError("window must have shape T,C,H,W")
    total_frames, _channels, height, width = window.shape
    if total_frames < clip_frames or min(height, width) < crop_size:
        raise ValueError("window is smaller than requested clip")
    generator = generator or torch.default_generator
    use_anchor = temporal_sampling == "anchor" and anchor_fraction is not None
    if use_anchor and not 0.0 <= float(anchor_fraction) <= 1.0:
        raise ValueError("anchor_fraction must be in [0, 1]")
    if use_anchor:
        anchor_index = int(round(float(anchor_fraction) * (total_frames - 1)))
        start_index = anchor_index - clip_frames // 2
        if training:
            start_index += int(torch.randint(-2, 3, (1,), generator=generator))
        start_index = min(max(start_index, 0), total_frames - clip_frames)
        indexes = list(range(start_index, start_index + clip_frames))
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
    elif training:
        segment_edges = torch.linspace(0, total_frames, clip_frames + 1)
        indexes = []
        for left, right in zip(segment_edges[:-1], segment_edges[1:], strict=True):
            low = int(math.floor(float(left)))
            high = max(low + 1, int(math.ceil(float(right))))
            indexes.append(
                int(torch.randint(low, min(high, total_frames), (1,), generator=generator))
            )
        top = int(torch.randint(0, height - crop_size + 1, (1,), generator=generator))
        left = int(torch.randint(0, width - crop_size + 1, (1,), generator=generator))
    else:
        indexes = np.rint(np.linspace(0, total_frames - 1, clip_frames)).astype(int).tolist()
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
    clip = window[indexes, :, top : top + crop_size, left : left + crop_size]
    clip = clip.to(torch.float32).div_(255.0)
    if training:
        if bool(torch.rand((), generator=generator) < 0.5):
            clip = torch.flip(clip, dims=(3,))
        brightness = float(torch.empty((), dtype=torch.float32).uniform_(0.85, 1.15, generator=generator))
        contrast = float(torch.empty((), dtype=torch.float32).uniform_(0.85, 1.15, generator=generator))
        channel_mean = clip.mean(dim=(0, 2, 3), keepdim=True)
        clip = ((clip - channel_mean) * contrast + channel_mean) * brightness
        clip.clamp_(0.0, 1.0)
        # Scoreboard/lower-third masks prevent memorizing broadcast overlays.
        if bool(torch.rand((), generator=generator) < 0.35):
            mask_height = max(1, crop_size // 10)
            if bool(torch.rand((), generator=generator) < 0.5):
                clip[:, :, :mask_height, :] = 0
            else:
                clip[:, :, -mask_height:, :] = 0
    if preprocessing == "agu_v3":
        # Preserve AGU's deployed v3 contract: BGR, [0,255], no normalization.
        return (clip[:, [2, 1, 0]] * 255.0).permute(1, 0, 2, 3).contiguous()
    mean = torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(KINETICS_STD).view(1, 3, 1, 1)
    return ((clip - mean) / std).permute(1, 0, 2, 3).contiguous()


def load_finetune_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    trainable_stage: str = "layer4",
    checkpoint_format: str = "kinetics",
) -> nn.Module:
    """Load local Kinetics weights and expose only the configured late stage."""

    if trainable_stage not in {"fc", "layer4", "layer3"}:
        raise ValueError("trainable_stage must be fc, layer4, or layer3")
    if checkpoint_format not in {"kinetics", "agu"}:
        raise ValueError("checkpoint_format must be kinetics or agu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=checkpoint_format == "kinetics",
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("R(2+1)D checkpoint must contain a state dict")
    state = checkpoint.get("state_dict") if checkpoint_format == "agu" else checkpoint
    if not isinstance(state, Mapping):
        raise ValueError("AGU R(2+1)D checkpoint lacks state_dict")
    model = r2plus1d_18(weights=None, progress=False)
    if checkpoint_format == "kinetics":
        model.load_state_dict(state)
    else:
        compatible = {
            key: value
            for key, value in state.items()
            if not key.startswith("fc.")
            and key in model.state_dict()
            and value.shape == model.state_dict()[key].shape
        }
        missing_backbone = [
            key
            for key in model.state_dict()
            if not key.startswith("fc.") and key not in compatible
        ]
        if missing_backbone:
            raise ValueError(f"AGU checkpoint lacks backbone tensors: {missing_backbone[:3]}")
        model.load_state_dict(compatible, strict=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if trainable_stage in {"layer4", "layer3"}:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    if trainable_stage == "layer3":
        for parameter in model.layer3.parameters():
            parameter.requires_grad = True
    return model.to(device)


def freeze_batch_norm_stats(model: nn.Module) -> None:
    """Keep pretrained normalization stable for tiny domain datasets."""

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def binary_metrics(labels: Sequence[bool], probabilities: Sequence[float], threshold: float) -> dict[str, float | int]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    predicted = [float(value) >= threshold for value in probabilities]
    tp = sum(label and guess for label, guess in zip(labels, predicted, strict=True))
    fp = sum(not label and guess for label, guess in zip(labels, predicted, strict=True))
    fn = sum(label and not guess for label, guess in zip(labels, predicted, strict=True))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def select_precision_threshold(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    *,
    minimum_precision: float = 0.95,
) -> tuple[float, dict[str, float | int]]:
    """Choose the precision-safe threshold with highest recall, deterministically."""

    candidates = sorted({0.0, 1.0, *(float(value) for value in probabilities)})
    eligible = []
    for threshold in candidates:
        metrics = binary_metrics(labels, probabilities, threshold)
        if float(metrics["precision"]) >= minimum_precision:
            eligible.append((float(metrics["recall"]), float(metrics["f1"]), -threshold, threshold, metrics))
    if not eligible:
        threshold = 1.0
        return threshold, binary_metrics(labels, probabilities, threshold)
    _recall, _f1, _negative_threshold, threshold, metrics = max(eligible)
    return threshold, metrics


def normalize_probability_by_threshold(probability: float, threshold: float) -> float:
    """Map a fold-specific raw threshold to 0.5 while preserving logit margin."""

    epsilon = 1e-6
    probability = min(max(float(probability), epsilon), 1.0 - epsilon)
    threshold = min(max(float(threshold), epsilon), 1.0 - epsilon)
    margin = math.log(probability / (1.0 - probability)) - math.log(
        threshold / (1.0 - threshold)
    )
    if margin >= 0:
        return 1.0 / (1.0 + math.exp(-margin))
    exponent = math.exp(margin)
    return exponent / (1.0 + exponent)


def evaluate_game_gate(
    records: Sequence[ShotWindowRecord],
    probabilities: Sequence[float],
    *,
    threshold: float,
    minimum_precision: float = 0.95,
    minimum_recall: float = 0.85,
) -> dict[str, Any]:
    if len(records) != len(probabilities):
        raise ValueError("records and probabilities must be aligned")
    per_game: dict[str, Any] = {}
    for game_hash in sorted({record.source_video_sha256 for record in records}):
        indexes = [i for i, record in enumerate(records) if record.source_video_sha256 == game_hash]
        metrics = binary_metrics(
            [records[i].event_present for i in indexes],
            [probabilities[i] for i in indexes],
            threshold,
        )
        per_game[game_hash] = metrics
    pooled = binary_metrics(
        [record.event_present for record in records], probabilities, threshold
    )
    promoted = (
        float(pooled["precision"]) >= minimum_precision
        and float(pooled["recall"]) >= minimum_recall
        and all(
            float(metrics["precision"]) >= minimum_precision
            and float(metrics["recall"]) >= minimum_recall
            for metrics in per_game.values()
        )
    )
    return {"promoted": promoted, "pooled": pooled, "per_game": per_game}


def seal_finetuned_model_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = FINETUNED_SHOT_MODEL_SCHEMA
    artifact.pop("metadata_sha256", None)
    artifact["metadata_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return artifact
