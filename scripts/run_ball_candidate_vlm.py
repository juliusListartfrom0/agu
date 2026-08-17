#!/usr/bin/env python3
"""Run a label-hidden local VLM probe over broadcast ball candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analysis.ball_candidate_review import seal_artifact, verify_artifact
from app.analysis.ball_candidate_vlm import (
    BALL_CANDIDATE_VLM_PROMPT,
    parse_ball_candidate_vlm_response,
    select_vlm_probe_candidates,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_panel(frame: np.ndarray, bbox: dict[str, float]) -> np.ndarray:
    """Render one label-free full-frame panel with a marked local inset."""
    source_height, source_width = frame.shape[:2]
    panel_width, panel_height = 960, 540
    panel = cv2.resize(
        frame,
        (panel_width, panel_height),
        interpolation=cv2.INTER_AREA,
    )
    x1, y1, x2, y2 = (float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
    scaled = (
        round(x1 * panel_width / source_width),
        round(y1 * panel_height / source_height),
        round(x2 * panel_width / source_width),
        round(y2 * panel_height / source_height),
    )
    cv2.rectangle(
        panel,
        (scaled[0], scaled[1]),
        (scaled[2], scaled[3]),
        (0, 0, 255),
        4,
    )

    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * 5.0
    left = max(0, int(np.floor(center_x - side / 2.0)))
    top = max(0, int(np.floor(center_y - side / 2.0)))
    right = min(source_width, max(left + 1, int(np.ceil(center_x + side / 2.0))))
    bottom = min(source_height, max(top + 1, int(np.ceil(center_y + side / 2.0))))
    crop = frame[top:bottom, left:right]
    inset_width, inset_height = 320, 240
    inset = cv2.resize(
        crop,
        (inset_width, inset_height),
        interpolation=cv2.INTER_NEAREST,
    )
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    inset_box = (
        round((x1 - left) * inset_width / crop_width),
        round((y1 - top) * inset_height / crop_height),
        round((x2 - left) * inset_width / crop_width),
        round((y2 - top) * inset_height / crop_height),
    )
    cv2.rectangle(
        inset,
        (inset_box[0], inset_box[1]),
        (inset_box[2], inset_box[3]),
        (0, 0, 255),
        4,
    )
    cv2.rectangle(inset, (0, 0), (inset_width - 1, inset_height - 1), (0, 255, 255), 5)
    inset_left = panel_width - inset_width - 8
    inset_top = panel_height - inset_height - 8
    panel[
        inset_top : inset_top + inset_height,
        inset_left : inset_left + inset_width,
    ] = inset
    return panel


def configure_image_processor(processor: Any, *, max_pixels: int) -> None:
    """Apply a bounded per-image pixel budget to the local MLX processor."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "max_pixels"):
        raise ValueError("MLX processor does not expose an image processor")
    min_pixels = int(getattr(image_processor, "min_pixels", 0))
    if max_pixels < min_pixels:
        raise ValueError("VLM image max pixels are below the processor minimum")
    image_processor.max_pixels = int(max_pixels)
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        size["longest_edge"] = int(max_pixels)


class MlxBallCandidateReviewer:
    """Load the local MLX VLM once and review independent candidate panels."""

    def __init__(
        self,
        *,
        model_path: Path,
        max_pixels: int,
        max_tokens: int,
        prefill_step_size: int,
    ) -> None:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        self.model, self.processor = load(str(model_path))
        configure_image_processor(self.processor, max_pixels=max_pixels)
        self.prompt = apply_chat_template(
            self.processor,
            self.model.config,
            BALL_CANDIDATE_VLM_PROMPT,
            num_images=1,
        )
        self.generate = generate
        self.max_tokens = max_tokens
        self.prefill_step_size = prefill_step_size

    def review(self, image_path: Path) -> dict[str, object]:
        try:
            result = self.generate(
                self.model,
                self.processor,
                self.prompt,
                image=str(image_path),
                max_tokens=self.max_tokens,
                temperature=0.0,
                prefill_step_size=self.prefill_step_size,
                verbose=False,
            )
            raw = str(getattr(result, "text", result))
        except Exception as exc:
            raw = ""
            return {
                "ball_state": "uncertain",
                "confidence": 0.0,
                "reason": f"local VLM unavailable: {type(exc).__name__}",
                "available": False,
                "raw_response": raw,
            }
        return {
            **parse_ball_candidate_vlm_response(raw),
            "raw_response": raw[:2000],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--review-plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--samples-per-band", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=301_056)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.samples_per_band <= 0
        or args.max_pixels <= 0
        or args.max_tokens <= 0
    ):
        raise ValueError("VLM probe sizes must be positive")
    perception = verify_artifact(json.loads(args.perception.read_text()))
    plan = verify_artifact(json.loads(args.review_plan.read_text()))
    if plan["perception_artifact_sha256"] != perception["artifact_sha256"]:
        raise ValueError("VLM probe plan/perception binding mismatch")
    if (
        plan.get("runtime_consumable") is not False
        or plan.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("VLM probe plan must be offline and label-hidden")
    if _sha256(args.video) != perception["raw_video"]["sha256"]:
        raise ValueError("VLM probe raw-video hash mismatch")
    if _sha256(args.model_path / "model.safetensors") != args.weights_sha256:
        raise ValueError("VLM probe model-weight hash mismatch")

    selected = select_vlm_probe_candidates(
        plan,
        samples_per_band=args.samples_per_band,
    )
    args.panel_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    panel_rows: list[tuple[dict[str, Any], Path, str]] = []
    try:
        for row in selected:
            frame_index = int(row["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"could not decode VLM probe frame {frame_index}")
            panel = _render_panel(frame, row["bbox"])
            path = args.panel_dir / f"{row['candidate_id']}.png"
            if not cv2.imwrite(str(path), panel):
                raise ValueError(f"could not write VLM probe panel: {path}")
            panel_rows.append((row, path, _sha256(path)))
    finally:
        cap.release()

    prompt_sha256 = hashlib.sha256(
        BALL_CANDIDATE_VLM_PROMPT.encode()
    ).hexdigest()
    cache: dict[str, Any] = {
        "schema_version": "agu.ball-candidate-vlm-cache.v1",
        "entries": {},
    }
    if args.cache is not None and args.cache.exists():
        cache = json.loads(args.cache.read_text())
        if (
            cache.get("schema_version") != "agu.ball-candidate-vlm-cache.v1"
            or not isinstance(cache.get("entries"), dict)
        ):
            raise ValueError("invalid ball-candidate VLM cache")
    reviewer: MlxBallCandidateReviewer | None = None
    predictions: list[dict[str, Any]] = []
    for index, (row, path, panel_sha256) in enumerate(panel_rows, 1):
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "candidate_id": row["candidate_id"],
                    "panel_sha256": panel_sha256,
                    "prompt_sha256": prompt_sha256,
                    "weights_sha256": args.weights_sha256,
                    "max_pixels": args.max_pixels,
                    "max_tokens": args.max_tokens,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        cached = cache["entries"].get(cache_key)
        if isinstance(cached, dict):
            decision = dict(cached)
        else:
            if reviewer is None:
                reviewer = MlxBallCandidateReviewer(
                    model_path=args.model_path,
                    max_pixels=args.max_pixels,
                    max_tokens=args.max_tokens,
                    prefill_step_size=args.prefill_step_size,
                )
            decision = reviewer.review(path)
            cache["entries"][cache_key] = decision
            if args.cache is not None:
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.cache.with_suffix(args.cache.suffix + ".tmp")
                temporary.write_text(json.dumps(cache, indent=2) + "\n")
                temporary.replace(args.cache)
        predictions.append(
            {
                "candidate_id": row["candidate_id"],
                "detection_id": row["detection_id"],
                "confidence_band": row["confidence_band"],
                "frame": int(row["frame"]),
                "panel_sha256": panel_sha256,
                **decision,
            }
        )
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(panel_rows),
                    "candidate_id": row["candidate_id"],
                    "ball_state": decision["ball_state"],
                    "available": decision["available"],
                }
            ),
            flush=True,
        )
    artifact = seal_artifact(
        {
            "schema_version": "agu.ball-candidate-independent-vlm-predictions.v1",
            "purpose": "offline_development_model_screening",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "review_plan_sha256": plan["artifact_sha256"],
            "perception_sha256": perception["artifact_sha256"],
            "raw_video_sha256": perception["raw_video"]["sha256"],
            "selection": {
                "strategy": "label_hidden_even_per_confidence_band",
                "samples_per_band": args.samples_per_band,
                "example_count": len(selected),
            },
            "input_contract": {
                "full_frame_width": 960,
                "full_frame_height": 540,
                "red_candidate_box": True,
                "yellow_local_inset": True,
                "labels_or_review_notes_exposed_to_model": False,
                "max_pixels": args.max_pixels,
                "prompt_sha256": prompt_sha256,
            },
            "model": {
                "name": args.model_name,
                "source": args.model_source,
                "revision": args.model_revision,
                "weights_sha256": args.weights_sha256,
                "backend": "mlx-vlm",
            },
            "predictions": predictions,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
