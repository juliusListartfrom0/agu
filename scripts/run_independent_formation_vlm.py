#!/usr/bin/env python3
"""Run a frozen two-frame formation plan with an independent Ollama VLM."""

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

from app.analysis.independent_formation_vlm import (  # noqa: E402
    parse_independent_formation_vlm_decision,
    seal_independent_formation_vlm_predictions,
    verify_independent_formation_vlm_plan,
)
from app.analysis.vlm import encode_frames_jpeg, parse_vlm_payload  # noqa: E402

PROMPT = """The two attached images are chronological raw broadcast frames
from one basketball event window, sampled one second apart. Classify only the
visible court formation.

Use free_throw_setup only when the images visibly support a free-throw
formation, such as a shooter at the free-throw line with lane players aligned
or players stationary for a free throw. Use live_play for continuous ordinary
game action. Use stoppage_other for a dead ball, timeout, close-up, jump ball,
or other non-free-throw stoppage. Use unknown when two frames are insufficient.
Do not infer a shot, score, or outcome from score graphics, captions, filenames,
or outside knowledge.

Return exactly one JSON object:
{
  "formation_state": "free_throw_setup" | "live_play" |
    "stoppage_other" | "unknown",
  "shooter_at_free_throw_line": true | false | null,
  "lane_players_aligned": true | false | null,
  "players_stationary_for_free_throw": true | false | null,
  "continuous_live_motion": true | false | null,
  "confidence": number from 0 to 1,
  "reason": "brief visible reason"
}"""


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
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=3072)
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
    image_width: int,
    context_length: int,
    cache_path: Path | None,
) -> dict[str, object]:
    verified = verify_independent_formation_vlm_plan(plan)
    contract = dict(verified["input_contract"])
    if (
        image_width != int(contract["image_width"])
        or context_length <= 0
    ):
        raise ValueError("runner settings do not match the frozen plan")
    _assert_ollama_model_residency(
        host=host,
        target_model=model,
        timeout=min(timeout, 10.0),
    )
    video_by_sha = {_file_sha256(path): path for path in videos}
    required = {
        str(row["source_video_sha256"]) for row in verified["examples"]
    }
    if set(video_by_sha) != required:
        raise ValueError("provided videos do not exactly match the frozen plan")

    fingerprint = _cache_fingerprint(
        plan_sha256=str(verified["plan_sha256"]),
        model=model,
        model_source=model_source,
        model_revision=model_revision,
        weights_sha256=weights_sha256,
        image_width=image_width,
        context_length=context_length,
    )
    cache = _load_cache(cache_path, fingerprint=fingerprint)
    predictions = []
    for index, row in enumerate(verified["examples"], start=1):
        key = _cache_key(row)
        cached = cache["predictions"].get(key)
        if isinstance(cached, dict):
            decision = cached
        else:
            frames = _read_exact_frames(
                video_by_sha[str(row["source_video_sha256"])],
                [int(value) for value in row["frame_indexes"]],
            )
            decision = _review_frames(
                frames=frames,
                model=model,
                host=host,
                timeout=timeout,
                image_width=image_width,
                context_length=context_length,
            )
            cache["predictions"][key] = decision
            _write_cache(cache_path, cache)
        predictions.append(
            {
                "phase_review_id": row["phase_review_id"],
                "source_video_sha256": row["source_video_sha256"],
                "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                "event_id": row["event_id"],
                **decision,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "independent_formation_vlm",
                    "completed": index,
                    "total": len(verified["examples"]),
                    "phase_review_id": row["phase_review_id"],
                    "state": decision["formation_state"],
                    "available": decision["available"],
                }
            ),
            flush=True,
        )

    return seal_independent_formation_vlm_predictions(
        {
            "plan_sha256": verified["plan_sha256"],
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "sampling": {
                "frame_offsets_seconds": verified["frame_offsets_seconds"],
                "max_frames": 2,
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
) -> dict[str, object]:
    images = encode_frames_jpeg(frames, max_width=image_width)
    if len(images) != 2:
        return {
            **parse_independent_formation_vlm_decision(
                {
                    "formation_state": "unknown",
                    "reason": "two exact frames could not be decoded",
                }
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
                "prompt": PROMPT,
                "images": images,
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "seed": 0,
                    "num_ctx": context_length,
                    "num_predict": 160,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed, _raw = parse_vlm_payload(
                json.loads(response.read().decode())
            )
        return {
            **parse_independent_formation_vlm_decision(parsed),
            "available": True,
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return {
            **parse_independent_formation_vlm_decision(
                {
                    "formation_state": "unknown",
                    "reason": f"VLM unavailable: {exc}",
                }
            ),
            "available": False,
        }


def _assert_ollama_model_residency(
    *,
    host: str,
    target_model: str,
    timeout: float,
) -> None:
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/ps",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"Ollama residency preflight failed: {exc}") from exc
    unexpected = _unexpected_ollama_models(
        payload,
        target_model=target_model,
    )
    if unexpected:
        raise ValueError(
            "unexpected Ollama models are resident: "
            + ", ".join(unexpected)
        )


def _unexpected_ollama_models(
    payload: object,
    *,
    target_model: str,
) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("models"),
        list,
    ):
        raise ValueError("invalid Ollama residency response")
    resident = set()
    for row in payload["models"]:
        if not isinstance(row, dict):
            raise ValueError("invalid Ollama residency model row")
        name = str(row.get("name") or row.get("model") or "")
        if not name:
            raise ValueError("Ollama residency model name is missing")
        resident.add(name)
    return sorted(name for name in resident if name != target_model)


def _read_exact_frames(path: Path, indexes: list[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for frame_number in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
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
    image_width: int,
    context_length: int,
) -> str:
    payload = {
        "plan_sha256": plan_sha256,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "model": model,
        "model_source": model_source,
        "model_revision": model_revision,
        "weights_sha256": weights_sha256,
        "image_width": image_width,
        "context_length": context_length,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_key(row: dict[str, object]) -> str:
    return ":".join(
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
    )


def _load_cache(path: Path | None, *, fingerprint: str) -> dict[str, object]:
    empty = {"fingerprint": fingerprint, "predictions": {}}
    if path is None or not path.exists():
        return empty
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("independent formation VLM cache fingerprint mismatch")
    if not isinstance(payload.get("predictions"), dict):
        raise ValueError("invalid independent formation VLM cache")
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
        image_width=args.image_width,
        context_length=args.context_length,
        cache_path=args.cache,
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
