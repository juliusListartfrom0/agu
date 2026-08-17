#!/usr/bin/env python3
"""Run a frozen raw-frame shot-validity plan against an independent Ollama VLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import (  # noqa: E402
    parse_independent_shot_vlm_decision,
    seal_independent_shot_vlm_predictions,
    verify_independent_shot_vlm_plan,
)
from app.analysis.vlm import encode_frames_jpeg, parse_vlm_payload  # noqa: E402

BASELINE_PROMPT = """The attached images are chronological, evenly sampled frames from one
basketball broadcast window. Decide whether the window contains a COMPLETE
LIVE-PLAY FIELD-GOAL ATTEMPT.

A live field-goal attempt requires visual evidence of live continuous play,
player control before release, ball separation from the hands, and subsequent
ball motion toward the rim. A replay/highlight, free throw, jump ball, dead-ball
scene, post-shot aftermath, or a window without a complete visible release is
not a live field goal. Do not use score graphics to infer an event. If the
chronological frames are insufficient, answer unknown.

Return exactly one JSON object with:
{
  "field_goal_state": "live_field_goal" | "not_field_goal" | "unknown",
  "continuous_live_play": true | false | null,
  "controlled_ball_before_release": true | false | null,
  "ball_separates_from_hands": true | false | null,
  "ball_moves_toward_rim": true | false | null,
  "replay_or_highlight": true | false | null,
  "free_throw": true | false | null,
  "confidence": number from 0 to 1,
  "reason": "brief visual reason"
}"""

STRICT_RELEASE_PROMPT = """The attached images are chronological frames from one
basketball broadcast candidate window. Inspect ALL frames before deciding.

Return live_field_goal ONLY when the same uninterrupted live possession visibly
contains all of these in order: (1) a player controls the ball, (2) the ball
clearly separates from the hands, and (3) the separated ball is visibly moving
toward the basket. A shooting pose, a player holding the ball, a ball already
in flight without visible control/release, or a post-shot frame is insufficient.

Return not_field_goal for any replay, highlight, slow-motion, dead-ball scene,
free throw, inbound/setup, scoreboard/graphic-only evidence, or window without
a complete visible release. Do not infer an event from score graphics or from
the fact that a player appears to be shooting. Use unknown only if the frames
cannot support either decision, and never invent missing temporal evidence.

Return exactly one JSON object with:
{
  "field_goal_state": "live_field_goal" | "not_field_goal" | "unknown",
  "continuous_live_play": true | false | null,
  "controlled_ball_before_release": true | false | null,
  "ball_separates_from_hands": true | false | null,
  "ball_moves_toward_rim": true | false | null,
  "replay_or_highlight": true | false | null,
  "free_throw": true | false | null,
  "confidence": number from 0 to 1,
  "reason": "brief visual reason"
}"""

STRICT_RELEASE_CONSISTENCY_PROMPT = """The attached images are chronological frames from one
basketball broadcast candidate window. Inspect every frame and perform a contradiction check
before deciding.

Return live_field_goal ONLY when the same uninterrupted live possession visibly proves this
ordered chain across the frames: (1) a player controls the ball, (2) the ball visibly separates
from that player's hands, and (3) after separation the ball is visibly moving toward the basket.
The first frame may not already show an airborne ball unless an earlier release is still visibly
proved in the supplied frames. A shooting pose, a player holding the ball, an airborne ball with
no visible release, or a ball whose direction cannot be established is not enough.

Run this self-consistency check before returning JSON:
- Every required observable must be directly supported by the chronological images, not guessed
  from posture, scoreboard text, the apparent camera direction, or the fact that a player is near
  the rim.
- If the reason contains a contradiction such as "without visible control/release", "no visible
  separation", or "already in flight", set the corresponding observable false or null and do not
  return live_field_goal. Never fill all observable fields with true by default.
- If any link in the ordered chain is missing, ambiguous, replay/highlight, free throw, setup,
  dead-ball, or post-shot, return not_field_goal or unknown; choose unknown when the images cannot
  support either decision.

Return exactly one JSON object with:
{
  "field_goal_state": "live_field_goal" | "not_field_goal" | "unknown",
  "continuous_live_play": true | false | null,
  "controlled_ball_before_release": true | false | null,
  "ball_separates_from_hands": true | false | null,
  "ball_moves_toward_rim": true | false | null,
  "replay_or_highlight": true | false | null,
  "free_throw": true | false | null,
  "confidence": number from 0 to 1,
  "reason": "brief visual reason naming the supported ordered evidence"
}"""

NEGATIVE_FIRST_PROMPT = """You are a conservative visual verifier, not a shot detector. The
attached images are chronological frames from one basketball broadcast candidate window. Inspect
ALL frames and try to disprove a field-goal attempt before accepting it; a false positive is worse
than an abstention.

Return live_field_goal ONLY if the images directly prove every link in this ordered chain within
the same uninterrupted live possession: (1) live play rather than replay/highlight or dead ball,
(2) one player visibly controls the ball before the motion, (3) the ball visibly separates from
that player's hands, and (4) after separation the ball visibly travels toward the basket. If any
link is missing, occluded, ambiguous, inferred from posture/scoreboard, or visible only before or
after the supplied window, return not_field_goal. A shooting pose, ball near a rim, ball already in
flight, free throw, inbound/setup, post-shot aftermath, or graphic is not proof of release.

Every required observable must be set to true only when that fact is directly visible in the
images; otherwise use false or null. Never set all observables true by default. Use unknown only when the frames cannot be
decoded or the JSON decision itself cannot be made. The reason must name the visible evidence or the
missing link, and must not claim an observable that is false or null.

Return exactly one JSON object with:
{
  "field_goal_state": "live_field_goal" | "not_field_goal" | "unknown",
  "continuous_live_play": true | false | null,
  "controlled_ball_before_release": true | false | null,
  "ball_separates_from_hands": true | false | null,
  "ball_moves_toward_rim": true | false | null,
  "replay_or_highlight": true | false | null,
  "free_throw": true | false | null,
  "confidence": number from 0 to 1,
  "reason": "brief evidence-based reason"
}"""

PROMPTS = {
    "baseline": BASELINE_PROMPT,
    "strict_release_v2": STRICT_RELEASE_PROMPT,
    "strict_release_v3": STRICT_RELEASE_CONSISTENCY_PROMPT,
    "negative_first_v4": NEGATIVE_FIRST_PROMPT,
}
# Preserve the historical name for existing callers and cache tests.
PROMPT = BASELINE_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--image-width", type=int, default=704)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument(
        "--prompt-variant",
        choices=tuple(PROMPTS),
        default="baseline",
        help="offline prompt variant; does not change AGU runtime behavior",
    )
    return parser.parse_args()


def run_plan(
    *,
    plan: dict[str, object],
    videos: list[Path],
    model: str,
    model_source: str,
    model_revision: str,
    weights_sha256: str,
    host: str,
    timeout: float,
    max_frames: int,
    image_width: int,
    context_length: int,
    cache_path: Path | None,
    prompt_variant: str = "baseline",
) -> dict[str, object]:
    verified = verify_independent_shot_vlm_plan(plan)
    if max_frames < 2 or image_width <= 0 or context_length <= 0:
        raise ValueError("frame, image-width and context settings must be positive")
    try:
        prompt = PROMPTS[prompt_variant]
    except KeyError as exc:
        raise ValueError(f"unsupported prompt variant: {prompt_variant}") from exc
    input_contract = verified.get("input_contract") or {}
    if (
        max_frames != int(input_contract.get("max_frames", 0))
        or image_width != int(input_contract.get("image_width", 0))
    ):
        raise ValueError("runner sampling does not match the frozen input contract")
    video_by_sha = {_file_sha256(path): path for path in videos}
    required = {str(row["source_video_sha256"]) for row in verified["examples"]}
    if set(video_by_sha) != required:
        raise ValueError("provided videos do not exactly match the frozen plan")
    cache_fingerprint = _cache_fingerprint(
        plan_sha256=str(verified["plan_sha256"]),
        model=model,
        model_source=model_source,
        model_revision=model_revision,
        weights_sha256=weights_sha256,
        max_frames=max_frames,
        image_width=image_width,
        context_length=context_length,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    cache = _load_cache(cache_path, fingerprint=cache_fingerprint)
    predictions = []
    for index, row in enumerate(verified["examples"], start=1):
        key = ":".join(
            (
                str(row["source_video_sha256"]),
                str(row["candidate_bundle_sha256"]),
                str(row["event_id"]),
            )
        )
        cached = cache["predictions"].get(key)
        if isinstance(cached, dict):
            decision = cached
        else:
            frames = _sample_frames(
                video_by_sha[str(row["source_video_sha256"])],
                int(row["start_frame"]),
                int(row["end_frame"]),
                max_frames,
            )
            decision = _review_frames(
                frames=frames,
                model=model,
                host=host,
                timeout=timeout,
                image_width=image_width,
                context_length=context_length,
                prompt=prompt,
            )
            cache["predictions"][key] = decision
            _write_cache(cache_path, cache)
        predictions.append(
            {
                "source_video_sha256": row["source_video_sha256"],
                "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                "event_id": row["event_id"],
                **decision,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "independent_vlm",
                    "completed": index,
                    "total": len(verified["examples"]),
                    "event_id": row["event_id"],
                    "state": decision["field_goal_state"],
                }
            ),
            flush=True,
        )
    return seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": verified["plan_sha256"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_variant": prompt_variant,
            "sampling": {
                "max_frames": max_frames,
                "image_width": image_width,
                "context_length": context_length,
            },
            "model": {
                "name": model,
                "source": model_source,
                "revision": model_revision,
                "weights_sha256": weights_sha256,
                "independent_from_codex": True,
            },
            "predictions": predictions,
        }
    )


def _review_frames(
    *,
    frames: list[np.ndarray],
    model: str,
    host: str,
    timeout: float,
    image_width: int,
    context_length: int,
    prompt: str = PROMPT,
) -> dict[str, object]:
    images = encode_frames_jpeg(frames, max_width=image_width)
    if not images:
        return {
            **parse_independent_shot_vlm_decision(
                {"field_goal_state": "unknown", "reason": "no frames decoded"}
            ),
            "available": False,
        }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(
            {
                "model": model,
                "stream": False,
                "think": False,
                "prompt": prompt,
                "images": images,
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "seed": 0,
                    "num_ctx": context_length,
                    "num_predict": 260,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed, _raw = parse_vlm_payload(json.loads(response.read().decode()))
        return {**parse_independent_shot_vlm_decision(parsed), "available": True}
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return {
            **parse_independent_shot_vlm_decision(
                {"field_goal_state": "unknown", "reason": f"VLM unavailable: {exc}"}
            ),
            "available": False,
        }


def _sample_frames(path: Path, start: int, end: int, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for frame_number in np.linspace(start, end, min(count, end - start + 1), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
    finally:
        capture.release()
    return frames


def _cache_fingerprint(
    *,
    plan_sha256: str,
    model: str,
    model_source: str,
    model_revision: str,
    weights_sha256: str,
    max_frames: int,
    image_width: int,
    context_length: int,
    prompt_sha256: str | None = None,
) -> str:
    if prompt_sha256 is None:
        prompt_sha256 = hashlib.sha256(PROMPT.encode()).hexdigest()
    payload = {
        "plan_sha256": plan_sha256,
        "prompt_sha256": prompt_sha256,
        "model": model,
        "model_source": model_source,
        "model_revision": model_revision,
        "weights_sha256": weights_sha256,
        "max_frames": max_frames,
        "image_width": image_width,
        "context_length": context_length,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_cache(path: Path | None, *, fingerprint: str) -> dict[str, object]:
    empty = {"fingerprint": fingerprint, "predictions": {}}
    if path is None or not path.exists():
        return empty
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("independent VLM cache fingerprint mismatch")
    if not isinstance(payload.get("predictions"), dict):
        raise ValueError("invalid independent VLM cache")
    return payload


def _write_cache(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    artifact = run_plan(
        plan=json.loads(args.plan.read_text(encoding="utf-8")),
        videos=args.video,
        model=args.model,
        model_source=args.model_source,
        model_revision=args.model_revision,
        weights_sha256=args.weights_sha256,
        host=args.host,
        timeout=args.timeout,
        max_frames=args.max_frames,
        image_width=args.image_width,
        context_length=args.context_length,
        cache_path=args.cache,
        prompt_variant=args.prompt_variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "artifact_sha256": artifact["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
