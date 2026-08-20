#!/usr/bin/env python3
"""Screen a MobileNet ball verifier trained only on Open Images boxes.

This is an offline transfer experiment.  Open Images labels and per-image
licenses are source supervision; the target broadcast review remains a
separate, hash-bound external evaluation and is never used for fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.ball_candidate_review import verify_artifact
from app.analysis.ball_candidate_verifier import (
    choose_recall_threshold,
    stratified_review_metrics,
)
from scripts.screen_ball_candidate_verifier import (
    MobileNetEmbedder,
    _broadcast_features,
    _canonical_sha256,
    _crop,
    _sha256,
)


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def _pixel_box(
    normalized: dict[str, float], width: int, height: int
) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(np.floor(normalized["xmin"] * width))))
    top = max(0, min(height - 1, int(np.floor(normalized["ymin"] * height))))
    right = max(left + 1, min(width, int(np.ceil(normalized["xmax"] * width))))
    bottom = max(top + 1, min(height, int(np.ceil(normalized["ymax"] * height))))
    return left, top, right, bottom


def _source_features(
    manifest: dict[str, Any],
    *,
    root: Path,
    embedder: MobileNetEmbedder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images: list[Any] = []
    labels: list[int] = []
    groups: list[str] = []
    image_cache: dict[str, np.ndarray] = {}
    examples = manifest.get("images")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Open Images manifest has no examples")
    for item in examples:
        image_id = str(item["image_id"])
        path = root / str(item["path"])
        if _sha256(path) != item["download"]["sha256"]:
            raise ValueError(f"Open Images image hash mismatch: {path}")
        image = image_cache.setdefault(image_id, cv2.imread(str(path)))
        if image is None:
            raise ValueError(f"could not decode Open Images image: {path}")
        height, width = image.shape[:2]
        boxes = [
            _pixel_box(box, width, height)
            for box in item.get("boxes", [])
        ]
        for box in boxes:
            images.append(_crop(image, list(box)))
            labels.append(1)
            groups.append(image_id)

        # One deterministic hard background crop per image.  It must not
        # overlap any source ball box, otherwise it would be a false negative.
        widths = [b[2] - b[0] for b in boxes]
        heights = [b[3] - b[1] for b in boxes]
        box_width = max(8, int(np.median(widths))) if widths else 32
        box_height = max(8, int(np.median(heights))) if heights else 32
        candidates = [
            (0, 0, box_width, box_height),
            (max(0, width - box_width), 0, width, box_height),
            (0, max(0, height - box_height), box_width, height),
            (max(0, width - box_width), max(0, height - box_height), width, height),
        ]
        negative = next(
            (candidate for candidate in candidates if max((_iou(candidate, b) for b in boxes), default=0.0) < 0.02),
            None,
        )
        if negative is None:
            continue
        images.append(_crop(image, list(negative)))
        labels.append(0)
        groups.append(image_id)
    if len(set(labels)) != 2 or len(set(groups)) < 5:
        raise ValueError("Open Images source needs both labels and five image groups")
    return embedder.encode(images), np.asarray(labels, dtype=np.int64), np.asarray(groups)


def _classifier(c_value: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=0),
    )


def _external_metrics(
    *,
    target_rows: list[dict[str, Any]],
    scores: np.ndarray,
    plan: dict[str, Any],
    review: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    score_by_id = {
        str(row["detection_id"]): float(score)
        for row, score in zip(target_rows, scores, strict=True)
    }
    bands = [tuple(float(value) for value in band) for band in plan["sampling"]["bands"]]
    names = [f"{low:.2f}-{high:.2f}" for low, high in bands]
    populations = {name: 0 for name in names}
    for row in target_rows:
        for name, (low, high) in zip(names, bands, strict=True):
            if low <= float(row["confidence"]) < high:
                populations[name] += 1
                break
    decisions = {str(row["candidate_id"]): str(row["decision"]) for row in review["decisions"]}
    bounds: dict[str, list[dict[str, object]]] = {"lower": [], "upper": []}
    rows: list[dict[str, object]] = []
    for candidate in plan["candidates"]:
        decision = decisions[str(candidate["candidate_id"])]
        kept = score_by_id[str(candidate["detection_id"])] >= threshold
        band = f"{float(candidate['confidence_band'][0]):.2f}-{float(candidate['confidence_band'][1]):.2f}"
        rows.append({"candidate_id": candidate["candidate_id"], "decision": decision, "kept": bool(kept)})
        for bound, uncertain_label in (("lower", 0), ("upper", 1)):
            label = 1 if decision == "valid_ball" else uncertain_label if decision == "uncertain" else 0
            bounds[bound].append({"band": band, "label": label, "kept": kept})
    return {
        "population_by_band": populations,
        "metrics": {
            bound: {
                key: round(float(value), 6)
                for key, value in stratified_review_metrics(rows, band_population=populations).items()
            }
            for bound, rows in bounds.items()
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-images-root", type=Path, required=True)
    parser.add_argument("--target-perception", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--target-review", type=Path, required=True)
    parser.add_argument("--target-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--recall-floor", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.torch_threads <= 0:
        raise ValueError("batch-size and torch-threads must be positive")
    import torch

    torch.set_num_threads(args.torch_threads)
    source_manifest = json.loads((args.open_images_root / "manifest.json").read_text())
    if (
        source_manifest.get("schema_version") != "agu.open-images-ball-subset.v1"
        or source_manifest.get("purpose") != "offline_training_only"
        or source_manifest.get("runtime_consumable") is not False
        or source_manifest.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("Open Images source manifest is not offline-only")
    perception = verify_artifact(json.loads(args.target_perception.read_text()))
    plan = verify_artifact(json.loads(args.target_plan.read_text()))
    review = verify_artifact(json.loads(args.target_review.read_text()))
    if plan["perception_artifact_sha256"] != perception["artifact_sha256"]:
        raise ValueError("target plan/perception mismatch")
    if review["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("target review/plan mismatch")
    if plan.get("runtime_consumable") is not False or review.get("runtime_consumable") is not False:
        raise ValueError("target review must be offline-only")

    embedder = MobileNetEmbedder(device=args.device, batch_size=args.batch_size)
    source_x, labels, groups = _source_features(
        source_manifest, root=args.open_images_root, embedder=embedder
    )
    target_x, target_rows = _broadcast_features(
        perception, video=args.target_video, embedder=embedder
    )
    splitter = GroupKFold(n_splits=5)
    variants: dict[str, Any] = {}
    for c_value in (0.01, 0.1, 1.0):
        oof = np.zeros(len(labels), dtype=np.float64)
        folds: list[dict[str, float]] = []
        for fold, (train, test) in enumerate(splitter.split(source_x, labels, groups), 1):
            model = _classifier(c_value).fit(source_x[train], labels[train])
            oof[test] = model.predict_proba(source_x[test])[:, 1]
            folds.append({"fold": fold, "average_precision": round(float(average_precision_score(labels[test], oof[test])), 6)})
        selection = choose_recall_threshold(labels, oof, recall_floor=args.recall_floor)
        model = _classifier(c_value).fit(source_x, labels)
        scores = model.predict_proba(target_x)[:, 1]
        external = _external_metrics(target_rows=target_rows, scores=scores, plan=plan, review=review, threshold=float(selection["threshold"]))
        accepted = bool(external["metrics"]["lower"]["precision"] >= 0.85 and external["metrics"]["lower"]["recall"] >= 0.85)
        variants[str(c_value)] = {
            "source": {
                "c": c_value,
                "examples": len(labels),
                "positive": int(labels.sum()),
                "negative": int((labels == 0).sum()),
                "groups": len(set(groups)),
                "oof_average_precision": round(float(average_precision_score(labels, oof)), 6),
                "folds": folds,
                "selection": {key: round(float(value), 8) for key, value in selection.items()},
            },
            "external": external,
            "accepted": accepted,
        }

    artifact: dict[str, Any] = {
        "schema_version": "agu.open-images-ball-verifier-screen.v1",
        "purpose": "offline_research_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source_manifest_sha256": _canonical_sha256(source_manifest),
        "target": {
            "perception_sha256": perception["artifact_sha256"],
            "plan_sha256": plan["artifact_sha256"],
            "review_sha256": review["artifact_sha256"],
        },
        "model": {"backbone": "MobileNet_V3_Small_Weights.DEFAULT", "checkpoint_saved": False},
        "variants": variants,
        "accepted": any(value["accepted"] for value in variants.values()),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "accepted": artifact["accepted"], "variants": {key: value["external"]["metrics"] for key, value in variants.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
