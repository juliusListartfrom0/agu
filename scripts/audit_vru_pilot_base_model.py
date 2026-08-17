#!/usr/bin/env python3
"""Audit the generic AGU v3 action head on sealed VRU pilot frames.

This is a whole-frame proxy diagnostic only.  It intentionally does not feed
the AGU runtime, does not use the pilot labels as training truth, and never
rewrites a checkpoint.  The pilot clips contain no player boxes, so the
diagnostic measures whether the ten-class SpaceJam head is even directionally
useful for a causal shot/not-shot question; it is not a deployment benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_base_vlm_fusion import load_agu_v3_action_model  # noqa: E402
from app.analysis.inference import LABELS  # noqa: E402


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def binary_metrics(
    labels: Sequence[bool], probabilities: Sequence[float], threshold: float
) -> dict[str, float | int]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    predicted = [float(value) >= threshold for value in probabilities]
    tp = sum(label and guess for label, guess in zip(labels, predicted, strict=True))
    fp = sum((not label) and guess for label, guess in zip(labels, predicted, strict=True))
    fn = sum(label and (not guess) for label, guess in zip(labels, predicted, strict=True))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def rank_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute pairwise ROC AUC with 0.5 credit for ties."""

    positives = [float(score) for label, score in zip(labels, scores, strict=True) if label]
    negatives = [float(score) for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return 0.5
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def summarize_diagnostic(
    rows: Sequence[dict[str, Any]], *, threshold: float = 0.5
) -> dict[str, Any]:
    if not rows:
        raise ValueError("diagnostic requires rows")
    labels = [bool(row["event_present"]) for row in rows]
    scores = [float(row["shoot_probability"]) for row in rows]
    argmax = [str(row["action"]) == "shoot" for row in rows]
    argmax_scores = [1.0 if value else 0.0 for value in argmax]
    return {
        "examples": len(rows),
        "positive_examples": sum(labels),
        "negative_examples": sum(not value for value in labels),
        "fixed_threshold": binary_metrics(labels, scores, threshold),
        "argmax_shoot": binary_metrics(labels, argmax_scores, 0.5),
        "rank_auc": rank_auc(labels, scores),
        "max_shoot_probability": max(scores),
        "mean_positive_shoot_probability": (
            float(np.mean([score for label, score in zip(labels, scores, strict=True) if label]))
            if any(labels)
            else 0.0
        ),
        "mean_negative_shoot_probability": float(
            np.mean([score for label, score in zip(labels, scores, strict=True) if not label])
        )
        if any(not value for value in labels)
        else 0.0,
    }


def _read_clip(paths: Sequence[Path], *, clip_frames: int = 16) -> torch.Tensor:
    if len(paths) < clip_frames:
        raise ValueError("pilot window has fewer frames than the action clip")
    selected = np.rint(np.linspace(0, len(paths) - 1, clip_frames)).astype(int)
    frames: list[np.ndarray] = []
    for index in selected:
        frame = cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot read pilot frame: {paths[int(index)]}")
        frames.append(cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR))
    # The canonical v3 contract is BGR, [0,255], C,T,H,W.
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).to(torch.float32)


def _load_windows(root: Path) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for version in ("v23", "v24_rv", "v25_rv"):
        folder = root / f"wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_{version}"
        retention = json.loads((folder / "retention_manifest.json").read_text(encoding="utf-8"))
        sealed = json.loads((folder / "review_sealed.json").read_text(encoding="utf-8"))
        reviews = {str(row["review_id"]): row for row in sealed["reviews"]}
        for window in retention["windows"]:
            review_id = str(window["review_id"])
            review = reviews[review_id]
            sequence = str(review["shot_sequence"])
            if sequence == "uncertain":
                continue
            frame_paths = [folder / "raw_frames" / str(frame["path"]).split("raw_frames/", 1)[-1] for frame in window["frames"]]
            if not all(path.is_file() for path in frame_paths):
                raise FileNotFoundError(review_id)
            windows.append(
                {
                    "version": version,
                    "review_id": review_id,
                    "source": str(review["source_video_filename"]),
                    "event_present": sequence == "shot",
                    "outcome": str(review["outcome"]),
                    "frame_paths": frame_paths,
                }
            )
    return windows


def audit(
    *,
    output: Path,
    checkpoint: Path,
    artifact_root: Path,
    threshold: float,
    device_name: str,
) -> dict[str, Any]:
    windows = _load_windows(artifact_root)
    device = torch.device(device_name)
    model, model_info = load_agu_v3_action_model(checkpoint, device=device)
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        clip = _read_clip(window["frame_paths"])
        with torch.inference_mode():
            probabilities = torch.softmax(model(clip.unsqueeze(0).to(device)), dim=1)[0]
        values = probabilities.detach().cpu().tolist()
        action_id = int(np.argmax(values))
        row = {
            "version": window["version"],
            "review_id": window["review_id"],
            "source": window["source"],
            "event_present": window["event_present"],
            "outcome": window["outcome"],
            "action": LABELS[action_id],
            "action_confidence": float(values[action_id]),
            "shoot_probability": float(values[4]),
        }
        rows.append(row)
        print(f"window={index}/{len(windows)} id={row['review_id']} action={row['action']} shoot={row['shoot_probability']:.6f}", flush=True)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    artifact: dict[str, Any] = {
        "schema_version": "agu.vru-pilot-base-model-audit.v1",
        "purpose": "offline_whole_frame_proxy_diagnostic_of_generic_action_head",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "training_consumable": False,
        "limitations": [
            "pilot-only, non-exhaustive labels",
            "uncertain windows excluded",
            "whole-frame proxy because pilot has no player boxes",
            "not a deployment or promotion benchmark",
        ],
        "inputs": {
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": model_info["checkpoint_sha256"],
            "artifact_root": artifact_root.as_posix(),
            "versions": ["v23", "v24_rv", "v25_rv"],
            "threshold": threshold,
            "device": device_name,
            "preprocessing": "agu_v3_bgr_0_255_112_no_normalization",
            "clip_frames": 16,
        },
        "model": model_info,
        "summary": summarize_diagnostic(rows, threshold=threshold),
        "per_source": {
            source: summarize_diagnostic(source_rows, threshold=threshold)
            for source, source_rows in sorted(by_source.items())
        },
        "rows": rows,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    artifact = audit(
        output=args.output,
        checkpoint=args.checkpoint,
        artifact_root=args.artifact_root,
        threshold=args.threshold,
        device_name=args.device,
    )
    print(json.dumps({"output": args.output.as_posix(), "artifact_sha256": artifact["artifact_sha256"], "examples": artifact["summary"]["examples"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
