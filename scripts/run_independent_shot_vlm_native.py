#!/usr/bin/env python3
"""Run a frozen independent-shot plan with a local native-video VLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import (  # noqa: E402
    verify_independent_shot_vlm_plan,
)
from app.analysis.independent_shot_vlm_native import (  # noqa: E402
    NATIVE_SHOT_PROMPTS,
    MlxNativeVideoReviewer,
    TransformersNativeVideoReviewer,
    run_native_video_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--backend",
        choices=("mlx", "transformers"),
        default="mlx",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="mps",
        help="Transformers backend device; ignored by MLX.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Transformers backend dtype; ignored by MLX.",
    )
    parser.add_argument("--max-tokens", type=int, default=260)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument(
        "--prompt-variant",
        choices=tuple(NATIVE_SHOT_PROMPTS),
        default="baseline",
        help="offline prompt variant; does not change AGU runtime behavior",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = verify_independent_shot_vlm_plan(
        json.loads(args.plan.read_text(encoding="utf-8"))
    )
    contract = dict(plan.get("input_contract") or {})
    if (
        contract.get("raw_video_clips_only") is not True
        or contract.get("labels_or_review_notes_exposed_to_model") is not False
        or contract.get("native_temporal_position_encoding") is not True
    ):
        raise ValueError("plan does not declare the native-video safety contract")
    if float(contract.get("sample_fps", 0.0)) <= 0:
        raise ValueError("plan has an invalid native-video sample FPS")
    if int(contract.get("max_pixels", 0)) <= 0:
        raise ValueError("plan has an invalid native-video pixel budget")
    prompt = NATIVE_SHOT_PROMPTS[args.prompt_variant]
    if args.backend == "transformers":
        reviewer = TransformersNativeVideoReviewer(
            model_path=str(args.model_path),
            max_pixels=int(contract.get("max_pixels", 0)),
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_tokens=args.max_tokens,
            prompt=prompt,
        )
    else:
        reviewer = MlxNativeVideoReviewer(
            model_path=str(args.model_path),
            max_pixels=int(contract.get("max_pixels", 0)),
            max_tokens=args.max_tokens,
            prefill_step_size=args.prefill_step_size,
            prompt=prompt,
        )
    artifact = run_native_video_plan(
        plan=plan,
        videos=args.video,
        review_window=reviewer.review,
        cache_path=args.cache,
        prompt=prompt,
        prompt_variant=args.prompt_variant,
        model={
            "name": args.model_name,
            "source": args.model_source,
            "revision": args.model_revision,
            "weights_sha256": args.weights_sha256,
        },
        progress=lambda row: print(json.dumps(row), flush=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
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
