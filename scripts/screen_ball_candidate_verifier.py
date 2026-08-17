#!/usr/bin/env python3
"""Train an E-BARD-only visual verifier and screen a sealed broadcast review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from app.analysis.ball_candidate_verifier import (
    choose_recall_threshold,
    stratified_review_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _verify_artifact(
    value: dict[str, Any], *, hash_field: str = "artifact_sha256"
) -> dict[str, Any]:
    artifact = dict(value)
    claimed = str(artifact.pop(hash_field, ""))
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("artifact hash mismatch")
    artifact[hash_field] = claimed
    return artifact


def _crop(
    image: np.ndarray,
    bbox: list[float] | dict[str, float],
    *,
    context_scale: float = 1.0,
) -> Image.Image:
    if context_scale < 1.0:
        raise ValueError("candidate context scale must be at least one")
    if isinstance(bbox, dict):
        x1, y1, x2, y2 = (bbox[key] for key in ("x1", "y1", "x2", "y2"))
    else:
        x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_width = (x2 - x1) * context_scale / 2
    half_height = (y2 - y1) * context_scale / 2
    x1, x2 = center_x - half_width, center_x + half_width
    y1, y2 = center_y - half_height, center_y + half_height
    height, width = image.shape[:2]
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    return Image.fromarray(cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB))


class MobileNetEmbedder:
    def __init__(self, *, device: str, batch_size: int) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights).eval().to(self.device)
        self.features = model.features
        self.avgpool = model.avgpool
        self.transform = weights.transforms()

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = torch.stack(
                [self.transform(image) for image in images[start : start + self.batch_size]]
            ).to(self.device)
            with torch.inference_mode():
                values = self.avgpool(self.features(batch)).flatten(1)
            rows.append(values.cpu().numpy())
            del batch, values
            if self.device.type == "mps":
                torch.mps.empty_cache()
        return np.concatenate(rows)


def _resolve_e_bard_image(root: Path, relative: str) -> Path:
    split, _, suffix = relative.partition("/images/")
    path = root / "extracted" / "yolo" / split / "images" / suffix
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _e_bard_features(
    manifest: dict[str, Any],
    *,
    root: Path,
    embedder: MobileNetEmbedder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images: list[Image.Image] = []
    labels: list[int] = []
    groups: list[str] = []
    confidence: list[float] = []
    image_cache: dict[str, np.ndarray] = {}
    for row in manifest["examples"]:
        relative = str(row["image_path"])
        if relative not in image_cache:
            path = _resolve_e_bard_image(root, relative)
            if _sha256(path) != row["image_sha256"]:
                raise ValueError(f"E-BARD image hash mismatch: {relative}")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"could not decode E-BARD image: {relative}")
            image_cache[relative] = image
        images.append(_crop(image_cache[relative], row["bbox_xyxy"]))
        labels.append(int(bool(row["ball_seed_present"])))
        groups.append(str(row["game_id"]))
        confidence.append(float(row["detector_confidence"]))
    return (
        embedder.encode(images),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
        np.asarray(confidence, dtype=np.float64),
    )


def _broadcast_features(
    perception: dict[str, Any],
    *,
    video: Path,
    embedder: MobileNetEmbedder,
    context_scale: float = 1.0,
    detection_ids: set[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if _sha256(video) != perception["raw_video"]["sha256"]:
        raise ValueError("broadcast video hash mismatch")
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in perception["detections"]:
        if (
            detection_ids is not None
            and str(row["detection_id"]) not in detection_ids
        ):
            continue
        by_frame[int(row["frame"])].append(row)
    found_ids = {
        str(row["detection_id"])
        for rows in by_frame.values()
        for row in rows
    }
    if detection_ids is not None and found_ids != detection_ids:
        missing = sorted(detection_ids - found_ids)
        raise ValueError(f"selected broadcast detections are missing: {missing}")
    if not by_frame:
        raise ValueError("broadcast feature extraction requires detections")
    cap = cv2.VideoCapture(str(video))
    images: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    try:
        for frame_index in sorted(by_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"could not decode broadcast frame {frame_index}")
            for row in by_frame[frame_index]:
                images.append(
                    _crop(image, row["bbox"], context_scale=context_scale)
                )
                rows.append(row)
    finally:
        cap.release()
    return embedder.encode(images), rows


def _classifier() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
    )


def _with_confidence(features: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    return np.column_stack([features, confidence])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--e-bard-root", type=Path, required=True)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--review-plan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-log", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--recall-floor", type=float, default=0.85)
    parser.add_argument("--exclude-detector-confidence", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch threads must be positive")
    torch.set_num_threads(args.torch_threads)
    manifest = _verify_artifact(
        json.loads(args.candidate_manifest.read_text()), hash_field="manifest_sha256"
    )
    perception = _verify_artifact(json.loads(args.perception.read_text()))
    plan = _verify_artifact(json.loads(args.review_plan.read_text()))
    review = _verify_artifact(json.loads(args.review.read_text()))
    if plan["perception_artifact_sha256"] != perception["artifact_sha256"]:
        raise ValueError("review plan does not bind the perception artifact")
    if review["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("review does not bind the review plan")
    if any(
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used", False) is not False
        for artifact in (plan, review)
    ):
        raise ValueError("review boundary is not offline-only")

    embedder = MobileNetEmbedder(device=args.device, batch_size=args.batch_size)
    source_x, labels, groups, source_conf = _e_bard_features(
        manifest, root=args.e_bard_root, embedder=embedder
    )
    feature_contract = "exact_candidate_crop"
    if not args.exclude_detector_confidence:
        source_x = _with_confidence(source_x, source_conf)
        feature_contract += "+detector_confidence"
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(labels), dtype=np.float64)
    fold_rows: list[dict[str, float]] = []
    for fold, (train, test) in enumerate(splitter.split(source_x, labels, groups), 1):
        classifier = _classifier().fit(source_x[train], labels[train])
        oof[test] = classifier.predict_proba(source_x[test])[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "average_precision": round(
                    float(average_precision_score(labels[test], oof[test])), 6
                ),
            }
        )
    selected = choose_recall_threshold(labels, oof, recall_floor=args.recall_floor)

    classifier = _classifier().fit(source_x, labels)
    target_features, target_rows = _broadcast_features(
        perception, video=args.video, embedder=embedder
    )
    target_conf = np.asarray([float(row["confidence"]) for row in target_rows])
    if not args.exclude_detector_confidence:
        target_features = _with_confidence(target_features, target_conf)
    target_scores = classifier.predict_proba(target_features)[:, 1]
    target_score_by_id = {
        str(row["detection_id"]): float(score)
        for row, score in zip(target_rows, target_scores, strict=True)
    }

    bands = [tuple(float(value) for value in band) for band in plan["sampling"]["bands"]]
    band_names = [f"{low:.2f}-{high:.2f}" for low, high in bands]
    populations = {name: 0 for name in band_names}
    for row in target_rows:
        confidence = float(row["confidence"])
        for name, (low, high) in zip(band_names, bands, strict=True):
            if low <= confidence < high:
                populations[name] += 1
                break
    decisions = {row["candidate_id"]: row["decision"] for row in review["decisions"]}
    reviewed_bounds: dict[str, list[dict[str, object]]] = {"lower": [], "upper": []}
    external_rows: list[dict[str, object]] = []
    for row in plan["candidates"]:
        decision = decisions[row["candidate_id"]]
        score = target_score_by_id[str(row["detection_id"])]
        kept = score >= selected["threshold"]
        band = f"{float(row['confidence_band'][0]):.2f}-{float(row['confidence_band'][1]):.2f}"
        external_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "decision": decision,
                "verifier_score": round(score, 8),
                "kept": bool(kept),
            }
        )
        for bound, uncertain_label in (("lower", 0), ("upper", 1)):
            label = (
                1
                if decision == "valid_ball"
                else uncertain_label
                if decision == "uncertain"
                else 0
            )
            reviewed_bounds[bound].append(
                {"band": band, "label": label, "kept": bool(kept)}
            )
    external = {
        bound: {
            key: round(float(value), 6)
            for key, value in stratified_review_metrics(
                rows, band_population=populations
            ).items()
        }
        for bound, rows in reviewed_bounds.items()
    }
    artifact: dict[str, Any] = {
        "schema_version": "agu.ball-candidate-verifier-screen.v1",
        "purpose": "offline_research_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "perception_artifact_sha256": perception["artifact_sha256"],
        "review_plan_sha256": plan["artifact_sha256"],
        "review_sha256": review["artifact_sha256"],
        "model": {
            "backbone": "MobileNet_V3_Small_Weights.DEFAULT",
            "classifier": "StandardScaler+LogisticRegression(C=0.1,balanced)",
            "features": feature_contract,
            "checkpoint_saved": False,
        },
        "source_evaluation": {
            "splitter": "GroupKFold(n_splits=5, group=game_id)",
            "examples": len(labels),
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
            "folds": fold_rows,
            "oof_average_precision": round(
                float(average_precision_score(labels, oof)), 6
            ),
            "selection": {key: round(value, 8) for key, value in selected.items()},
        },
        "external_review": {
            "population_by_band": populations,
            "reviewed": len(external_rows),
            "metrics": external,
            "rows": external_rows,
        },
    }
    artifact["accepted"] = bool(
        external["lower"]["precision"] >= 0.85
        and external["lower"]["recall"] >= 0.85
    )
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **artifact["external_review"]["metrics"], "accepted": artifact["accepted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
