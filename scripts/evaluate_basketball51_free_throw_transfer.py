#!/usr/bin/env python3
"""Evaluate a frozen Basketball-51 free-throw model on reviewed AGU windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ball_release_annotation import (  # noqa: E402
    verify_ball_release_plan,
    verify_ball_release_review,
)
from app.analysis.basketball51_transfer import (  # noqa: E402
    evaluate_free_throw_transfer,
)
from app.analysis.shot_validity_video_backbone import file_sha256  # noqa: E402
from app.analysis.training_annotation import (  # noqa: E402
    verify_training_annotation_manifest,
)

SCHEMA = "agu.basketball51-free-throw-transfer.v1"
CHECKPOINT_SCHEMA = "agu.basketball51-pretrained-backbone.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-review",
        nargs=2,
        metavar=("PLAN", "REVIEW"),
        action="append",
        required=True,
    )
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _review_rows(
    pairs: list[tuple[Path, Path]],
    *,
    target_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    plan_hashes: list[str] = []
    review_hashes: list[str] = []
    for plan_path, review_path in pairs:
        plan = verify_ball_release_plan(_read_json(plan_path))
        review = verify_ball_release_review(_read_json(review_path), plan=plan)
        if plan["training_manifest_sha256"] != target_manifest_sha256:
            raise ValueError("review plan does not match target training manifest")
        plan_lookup = {
            (
                str(row["source_video_sha256"]),
                str(row["candidate_bundle_sha256"]),
                str(row["event_id"]),
            ): row
            for row in plan["examples"]
        }
        for annotation in review["reviews"]:
            if annotation["review_confidence"] == "uncertain":
                continue
            key = (
                str(annotation["source_video_sha256"]),
                str(annotation["candidate_bundle_sha256"]),
                str(annotation["event_id"]),
            )
            window = plan_lookup[key]
            rows.append(
                {
                    "source_video_sha256": key[0],
                    "candidate_bundle_sha256": key[1],
                    "event_id": key[2],
                    "start_frame": int(window["start_frame"]),
                    "end_frame": int(window["end_frame"]),
                    "free_throw": bool(annotation["free_throw_formation"]),
                    "review_confidence": str(annotation["review_confidence"]),
                }
            )
        plan_hashes.append(str(plan["artifact_sha256"]))
        review_hashes.append(str(review["artifact_sha256"]))
    keys = {
        (row["source_video_sha256"], row["candidate_bundle_sha256"], row["event_id"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("review transfer rows overlap")
    return rows, plan_hashes, review_hashes


def _video_paths(
    manifest: Mapping[str, Any],
    *,
    video_root: Path,
) -> dict[str, Path]:
    benchmark_hashes = set(manifest["benchmark_raw_sha256"])
    result: dict[str, Path] = {}
    for row in manifest["source_videos"]:
        sha = str(row["sha256"])
        if sha in benchmark_hashes:
            raise ValueError("target transfer manifest overlaps sealed benchmark media")
        path = video_root / str(row["filename"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or file_sha256(path) != sha
        ):
            raise ValueError(f"target transfer video failed integrity check: {path.name}")
        result[sha] = path
    return result


def _load_clip(
    path: Path,
    *,
    start_frame: int,
    end_frame: int,
    clip_frames: int,
) -> torch.Tensor:
    if end_frame < start_frame or clip_frames < 1:
        raise ValueError("invalid target transfer window")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open target transfer video: {path.name}")
    frames = []
    try:
        for frame_number in np.rint(
            np.linspace(start_frame, end_frame, clip_frames)
        ).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"cannot decode target transfer frame {frame_number}: {path.name}"
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    raw = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return MViT_V2_S_Weights.KINETICS400_V1.transforms()(raw)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"target transfer output already exists: {args.output}")
    manifest = verify_training_annotation_manifest(
        _read_json(args.target_manifest)
    )
    rows, plan_hashes, review_hashes = _review_rows(
        [(Path(plan), Path(review)) for plan, review in args.plan_review],
        target_manifest_sha256=str(manifest["manifest_sha256"]),
    )
    paths = _video_paths(manifest, video_root=args.video_root)
    if set(row["source_video_sha256"] for row in rows) - set(paths):
        raise ValueError("review rows reference media outside target training manifest")

    metadata = _read_json(args.checkpoint_metadata)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if (
        metadata.get("schema_version") != CHECKPOINT_SCHEMA
        or metadata.get("source_validation_gate_passed") is not True
        or metadata.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("Basketball-51 checkpoint did not pass its source gate")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("runtime_consumable") is not False
    ):
        raise ValueError("invalid Basketball-51 transfer checkpoint")
    device = torch.device(args.device)
    model = mvit_v2_s(weights=None)
    model.head[-1] = torch.nn.Linear(model.head[-1].in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    probabilities = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            clip = _load_clip(
                paths[str(row["source_video_sha256"])],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                clip_frames=args.clip_frames,
            )
            logits = model(clip.unsqueeze(0).to(device))
            probability = float(torch.softmax(logits, dim=1)[0, 0].cpu())
            probabilities.append(probability)
            print(
                json.dumps(
                    {
                        "event": "target_transfer_prediction",
                        "completed": index + 1,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )

    evaluation = evaluate_free_throw_transfer(
        rows,
        probabilities,
        threshold=args.threshold,
    )
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "purpose": "source_to_target_free_throw_transfer_screening_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "promotion_eligible": evaluation["promotion_eligible"],
        "target_training_manifest_sha256": manifest["manifest_sha256"],
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_subset_manifest_sha256": checkpoint["subset_manifest_sha256"],
        "plan_sha256s": plan_hashes,
        "review_sha256s": review_hashes,
        "clip_frames": args.clip_frames,
        "evaluation": evaluation,
        "predictions": [
            {
                **row,
                "free_throw_probability": probability,
                "predicted_free_throw": probability >= args.threshold,
            }
            for row, probability in zip(rows, probabilities, strict=True)
        ],
    }
    output["artifact_sha256"] = _canonical_sha256(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation, ensure_ascii=False), flush=True)
    return 0 if evaluation["promotion_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
