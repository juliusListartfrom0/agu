#!/usr/bin/env python3
"""Screen a frozen RF-DETR query verifier on a disjoint reviewed game."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analysis.ball_candidate_review import verify_artifact
from app.analysis.ball_candidate_verifier import ball_track_features
from app.analysis.rfdetr_query_verifier import (
    expand_detection_ids_to_tracks,
    match_candidates_to_queries,
    query_feature_vector,
    track_query_features,
)
from scripts.screen_ball_candidate_verifier import _canonical_sha256
from scripts.screen_same_detector_ball_verifier import (
    _determinate_source_rows,
    _external_metrics,
    _fit_variant,
    _validate_bundle,
)

FEATURE_CONTRACT = "last_hidden_state+raw_logits+normalized_cxcywh.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_video(perception: dict[str, Any], video: Path) -> None:
    expected = perception["raw_video"]
    if not video.is_file():
        raise FileNotFoundError(video)
    if video.stat().st_size != int(expected["size_bytes"]):
        raise ValueError(f"video size does not match perception artifact: {video}")
    if _sha256_file(video) != str(expected["sha256"]):
        raise ValueError(f"video hash does not match perception artifact: {video}")


def _model_weight(model_path: Path) -> Path:
    candidates = sorted(model_path.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError("model path must contain exactly one safetensors weight")
    return candidates[0]


def _cache_path(
    cache_dir: Path,
    *,
    perception_sha256: str,
    model_sha256: str,
    detection_ids: list[str],
) -> Path:
    selection_hash = hashlib.sha256("\n".join(detection_ids).encode()).hexdigest()
    return cache_dir / (
        f"{perception_sha256[:16]}-{model_sha256[:16]}-{selection_hash[:16]}.npz"
    )


def _load_cache(
    path: Path,
    *,
    detection_ids: list[str],
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as stored:
        ids = [str(value) for value in stored["detection_ids"].tolist()]
        features = np.asarray(stored["features"], dtype=np.float64)
    if ids != detection_ids or features.ndim != 2 or len(features) != len(ids):
        raise ValueError(f"invalid RF-DETR query feature cache: {path}")
    return dict(zip(ids, features, strict=True))


def _save_cache(
    path: Path,
    *,
    detection_ids: list[str],
    features: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        detection_ids=np.asarray(detection_ids),
        features=np.stack([features[detection_id] for detection_id in detection_ids]),
    )


def _extract_query_features(
    perception: dict[str, Any],
    *,
    video: Path,
    model: Any,
    processor: Any,
    device: str,
    batch_size: int,
    ball_label_id: int,
    confidence_threshold: float,
    model_sha256: str,
    cache_dir: Path | None,
    selected_detection_ids: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows = [
        row
        for row in perception["detections"]
        if selected_detection_ids is None
        or str(row["detection_id"]) in selected_detection_ids
    ]
    if selected_detection_ids is not None and len(rows) != len(selected_detection_ids):
        raise ValueError("selected candidate is missing from perception detections")
    rows.sort(key=lambda row: (int(row["frame"]), str(row["detection_id"])))
    detection_ids = [str(row["detection_id"]) for row in rows]
    if len(set(detection_ids)) != len(detection_ids):
        raise ValueError("perception detection IDs must be unique")
    cache_path = (
        _cache_path(
            cache_dir,
            perception_sha256=str(perception["artifact_sha256"]),
            model_sha256=model_sha256,
            detection_ids=detection_ids,
        )
        if cache_dir is not None
        else None
    )
    if cache_path is not None:
        cached = _load_cache(cache_path, detection_ids=detection_ids)
        if cached is not None:
            print(f"loaded query feature cache {cache_path}", file=sys.stderr)
            return cached, rows

    _verify_video(perception, video)
    candidates_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidates_by_frame[int(row["frame"])].append(row)
    frame_numbers = sorted(candidates_by_frame)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    features: dict[str, np.ndarray] = {}
    try:
        import torch

        for offset in range(0, len(frame_numbers), batch_size):
            batch_numbers = frame_numbers[offset : offset + batch_size]
            bgr_frames: list[np.ndarray] = []
            for frame_number in batch_numbers:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"failed to decode frame {frame_number} from {video}"
                    )
                bgr_frames.append(frame)
            rgb_frames = [
                np.ascontiguousarray(frame[:, :, ::-1]) for frame in bgr_frames
            ]
            inputs = processor(images=rgb_frames, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            if hidden_states is None:
                raise RuntimeError("RF-DETR output has no last_hidden_state")
            for index, (frame_number, frame) in enumerate(
                zip(batch_numbers, bgr_frames, strict=True)
            ):
                candidates = candidates_by_frame[frame_number]
                matches = match_candidates_to_queries(
                    outputs.logits[index],
                    outputs.pred_boxes[index],
                    candidates,
                    image_width=int(frame.shape[1]),
                    image_height=int(frame.shape[0]),
                    confidence_threshold=confidence_threshold,
                    ball_label_id=ball_label_id,
                )
                for candidate in candidates:
                    detection_id = str(candidate["detection_id"])
                    features[detection_id] = query_feature_vector(
                        hidden_states[index],
                        outputs.logits[index],
                        outputs.pred_boxes[index],
                        query_index=int(matches[detection_id]["query_index"]),
                    )
            print(
                f"query features {min(offset + batch_size, len(frame_numbers))}/"
                f"{len(frame_numbers)} frames",
                file=sys.stderr,
            )
    finally:
        capture.release()
    if set(features) != set(detection_ids):
        raise RuntimeError("query feature extraction did not cover every candidate")
    if cache_path is not None:
        _save_cache(
            cache_path,
            detection_ids=detection_ids,
            features=features,
        )
    return features, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-bundle",
        action="append",
        nargs=4,
        type=Path,
        metavar=("PERCEPTION", "PLAN", "REVIEW", "VIDEO"),
        required=True,
    )
    parser.add_argument("--target-perception", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--target-review", type=Path, required=True)
    parser.add_argument("--target-video", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--ball-label-id", type=int, default=0)
    parser.add_argument("--recall-floor", type=float, default=0.85)
    return parser.parse_args()


def _load_artifact(path: Path) -> dict[str, Any]:
    return verify_artifact(json.loads(path.read_text()))


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.torch_threads <= 0:
        raise ValueError("batch size and torch threads must be positive")
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    torch.set_num_threads(args.torch_threads)
    weight = _model_weight(args.model_path)
    model_sha256 = _sha256_file(weight)
    processor = AutoImageProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    model = (
        AutoModelForObjectDetection.from_pretrained(
            args.model_path,
            local_files_only=True,
        )
        .to(args.device)
        .eval()
    )
    if int(model.config.id2label.get(args.ball_label_id, "") == "ball") != 1:
        raise ValueError("configured ball label does not map to 'ball'")

    source_feature_parts: list[np.ndarray] = []
    source_geometry_parts: list[np.ndarray] = []
    source_temporal_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    source_summaries: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    geometry_names: tuple[str, ...] | None = None
    temporal_names: tuple[str, ...] | None = None
    label_width = int(model.config.num_labels)
    context_width: int | None = None
    for perception_path, plan_path, review_path, video_path in args.source_bundle:
        perception = _load_artifact(perception_path)
        plan = _load_artifact(plan_path)
        review = _load_artifact(review_path)
        examples = _determinate_source_rows(perception, plan, review)
        detection_ids = {str(row["detection_id"]) for row in examples}
        expanded_detection_ids = expand_detection_ids_to_tracks(
            perception,
            detection_ids,
        )
        feature_by_id, _ = _extract_query_features(
            perception,
            video=video_path,
            model=model,
            processor=processor,
            device=args.device,
            batch_size=args.batch_size,
            ball_label_id=args.ball_label_id,
            confidence_threshold=args.confidence_threshold,
            model_sha256=model_sha256,
            cache_dir=args.cache_dir,
            selected_detection_ids=expanded_detection_ids,
        )
        current_width = len(next(iter(feature_by_id.values())))
        current_context_width = current_width - label_width - 4
        if context_width is None:
            context_width = current_context_width
        elif current_context_width != context_width:
            raise ValueError("source query feature contracts differ")
        current_temporal_names, temporal_by_id = track_query_features(
            perception,
            feature_by_id,
            context_width=current_context_width,
            label_width=label_width,
        )
        if temporal_names is None:
            temporal_names = current_temporal_names
        elif current_temporal_names != temporal_names:
            raise ValueError("source temporal query-feature contracts differ")
        current_geometry_names, geometry_by_id = ball_track_features(perception)
        if geometry_names is None:
            geometry_names = current_geometry_names
        elif current_geometry_names != geometry_names:
            raise ValueError("source track-feature contracts differ")
        source_feature_parts.append(
            np.stack(
                [feature_by_id[str(row["detection_id"])] for row in examples]
            )
        )
        source_geometry_parts.append(
            np.stack(
                [geometry_by_id[str(row["detection_id"])] for row in examples]
            )
        )
        source_temporal_parts.append(
            np.stack(
                [temporal_by_id[str(row["detection_id"])] for row in examples]
            )
        )
        labels = np.asarray([int(row["label"]) for row in examples], dtype=np.int64)
        groups = np.asarray([str(row["group"]) for row in examples])
        label_parts.append(labels)
        group_parts.append(groups)
        source_hash = str(perception["raw_video"]["sha256"])
        source_hashes.append(source_hash)
        source_summaries.append(
            {
                "perception_sha256": perception["artifact_sha256"],
                "plan_sha256": plan["artifact_sha256"],
                "review_sha256": review["artifact_sha256"],
                "raw_video_sha256": source_hash,
                "determinate_examples": len(examples),
                "query_feature_detections": len(expanded_detection_ids),
                "positive": int(labels.sum()),
                "negative": int((labels == 0).sum()),
                "causal_window_groups": len(set(groups)),
            }
        )
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("source games must be mutually disjoint")

    target_perception = _load_artifact(args.target_perception)
    target_plan = _load_artifact(args.target_plan)
    target_hash = str(target_perception["raw_video"]["sha256"])
    if target_hash in source_hashes:
        raise ValueError("source and target games must be disjoint")
    target_feature_by_id, target_rows = _extract_query_features(
        target_perception,
        video=args.target_video,
        model=model,
        processor=processor,
        device=args.device,
        batch_size=args.batch_size,
        ball_label_id=args.ball_label_id,
        confidence_threshold=args.confidence_threshold,
        model_sha256=model_sha256,
        cache_dir=args.cache_dir,
    )
    target_features = np.stack(
        [target_feature_by_id[str(row["detection_id"])] for row in target_rows]
    )
    if context_width is None:
        raise ValueError("source query context width was not established")
    target_temporal_names, target_temporal_by_id = track_query_features(
        target_perception,
        target_feature_by_id,
        context_width=context_width,
        label_width=label_width,
    )
    if temporal_names != target_temporal_names:
        raise ValueError("source and target temporal query-feature contracts differ")
    target_temporal = np.stack(
        [
            target_temporal_by_id[str(row["detection_id"])]
            for row in target_rows
        ]
    )
    target_geometry_names, target_geometry_by_id = ball_track_features(
        target_perception
    )
    if geometry_names != target_geometry_names:
        raise ValueError("source and target track-feature contracts differ")
    target_geometry = np.stack(
        [target_geometry_by_id[str(row["detection_id"])] for row in target_rows]
    )

    source_features = np.concatenate(source_feature_parts)
    source_geometry = np.concatenate(source_geometry_parts)
    source_temporal = np.concatenate(source_temporal_parts)
    labels = np.concatenate(label_parts)
    groups = np.concatenate(group_parts)
    if set(labels) != {0, 1} or len(set(groups)) < 2:
        raise ValueError("sources require both labels and multiple causal groups")
    if (
        context_width <= 0
        or target_features.shape[1] != source_features.shape[1]
        or source_temporal.shape[1] != target_temporal.shape[1]
    ):
        raise ValueError("invalid or inconsistent RF-DETR query feature width")
    variant_inputs = {
        "query_context": (
            source_features[:, :context_width],
            target_features[:, :context_width],
        ),
        "query_logits_box": (
            source_features[:, context_width:],
            target_features[:, context_width:],
        ),
        "query_context_logits_box": (source_features, target_features),
        "query_context_track_geometry": (
            np.column_stack([source_features, source_geometry]),
            np.column_stack([target_features, target_geometry]),
        ),
        "query_temporal_only": (source_temporal, target_temporal),
        "query_context_temporal": (
            np.column_stack(
                [source_features[:, :context_width], source_temporal]
            ),
            np.column_stack(
                [target_features[:, :context_width], target_temporal]
            ),
        ),
        "query_context_logits_box_temporal": (
            np.column_stack([source_features, source_temporal]),
            np.column_stack([target_features, target_temporal]),
        ),
    }
    fitted: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for name, (source_x, target_x) in variant_inputs.items():
        fitted[name] = _fit_variant(
            source_x,
            labels,
            groups,
            target_x,
            recall_floor=args.recall_floor,
        )

    # Target labels are opened only after all feature choices, fits, OOF
    # thresholds and target scores are fixed.
    target_review = _load_artifact(args.target_review)
    _validate_bundle(target_perception, target_plan, target_review)
    variants: dict[str, Any] = {}
    accepted = False
    for name, (source_evaluation, target_scores) in fitted.items():
        external = _external_metrics(
            target_rows=target_rows,
            target_scores=target_scores,
            plan=target_plan,
            review=target_review,
            threshold=float(source_evaluation["selection"]["threshold"]),
        )
        variant_accepted = bool(
            external["metrics"]["lower"]["precision"] >= 0.85
            and external["metrics"]["lower"]["recall"] >= 0.85
        )
        accepted |= variant_accepted
        variants[name] = {
            "source_evaluation": source_evaluation,
            "external_review": external,
            "accepted": variant_accepted,
        }

    artifact: dict[str, Any] = {
        "schema_version": "agu.rfdetr-query-verifier-screen.v1",
        "purpose": "offline_research_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source": {
            "games": source_summaries,
            "determinate_examples": len(labels),
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
            "causal_window_groups": len(set(groups)),
        },
        "target": {
            "perception_sha256": target_perception["artifact_sha256"],
            "plan_sha256": target_plan["artifact_sha256"],
            "review_sha256": target_review["artifact_sha256"],
            "raw_video_sha256": target_hash,
        },
        "model": {
            "architecture": "RF-DETR frozen query verifier",
            "model_weight_sha256": model_sha256,
            "feature_contract": FEATURE_CONTRACT,
            "query_context_width": context_width,
            "raw_label_width": label_width,
            "normalized_box_width": 4,
            "track_geometry_features": list(geometry_names or ()),
            "track_query_features": list(temporal_names or ()),
            "detector_frozen": True,
            "checkpoint_saved": False,
        },
        "target_review_opened_after_fit": True,
        "variants": variants,
        "accepted": accepted,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": accepted,
                "metrics": {
                    name: value["external_review"]["metrics"]
                    for name, value in variants.items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
