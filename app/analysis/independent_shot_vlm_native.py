"""Offline native-video runner for the independent basketball shot VLM."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from app.analysis.independent_shot_vlm import (
    parse_independent_shot_vlm_decision,
    seal_independent_shot_vlm_predictions,
    verify_independent_shot_vlm_plan,
)
from app.analysis.vlm import extract_json_object

NATIVE_SHOT_PROMPT = """Review this chronological basketball broadcast video.
Decide whether it contains one COMPLETE LIVE-PLAY FIELD-GOAL ATTEMPT.

A live field-goal attempt requires visual evidence of continuous live play,
player control before release, ball separation from the hands, and subsequent
ball motion toward the rim. A replay/highlight, free throw, jump ball, dead-ball
scene, post-shot aftermath, or a window without a complete visible release is
not a live field goal. Do not infer an event from score graphics. If the video
is insufficient, answer unknown.

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

STRICT_RELEASE_NATIVE_PROMPT = """Review the entire chronological basketball broadcast video before deciding.
Return live_field_goal ONLY when one uninterrupted live possession visibly shows
all three stages in order: (1) a player controls the ball, (2) the ball clearly
separates from the hands, and (3) the separated ball is visibly moving toward
the basket. A shooting pose, a ball already in flight, or a post-shot frame is
not enough when the control or release is not visible.

Return not_field_goal for a replay/highlight/slow-motion, free throw, jump ball,
dead-ball scene, inbound/setup, scoreboard or graphic-only evidence, or any
window without a complete visible release. Do not infer an event from score
graphics or from a player merely appearing to shoot. Use unknown when the video
does not support a defensible decision; never guess missing temporal evidence.

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

NEGATIVE_FIRST_NATIVE_PROMPT = """You are a conservative visual verifier, not a shot detector. Review every
chronological frame of this basketball broadcast window and try to disprove a
field-goal attempt before accepting it; a false positive is worse than an
abstention.

Return live_field_goal ONLY when the same uninterrupted live possession visibly
proves every link in this ordered chain: (1) live play rather than replay,
highlight, dead ball or setup, (2) one player visibly controls the ball before
the motion, (3) the ball visibly separates from that player's hands, and (4)
after separation the ball visibly travels toward the basket. A shooting pose,
ball near a rim, ball already in flight, free throw, inbound/setup, post-shot
aftermath or graphic is not proof of release. If any link is missing, occluded,
ambiguous or only visible before/after the supplied window, return
not_field_goal. Use unknown only when the frames cannot support a defensible
decision.

Every observable must be true only when directly visible in the frames; use
false or null otherwise. Never fill all observable fields with true by default.
The reason must name the visible evidence or the missing link and must not
claim an observable that is false or null.

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

COMPACT_JSON_NATIVE_PROMPT = """Inspect these chronological basketball frames. Decide if they show one
complete live-play field-goal attempt: visible player control, visible ball
release, then visible ball travel toward the basket. Replays, free throws,
dead-ball setup, post-shot scenes, or missing release are not field goals.

Return ONLY one JSON object in one line. No prose and no markdown:
{"field_goal_state":"unknown","continuous_live_play":null,"controlled_ball_before_release":null,"ball_separates_from_hands":null,"ball_moves_toward_rim":null,"replay_or_highlight":null,"free_throw":null,"confidence":0.0,"reason":"short visible evidence"}
Change `field_goal_state` only to `live_field_goal` or `not_field_goal`, and
change nulls only to true or false, when the frames directly support it."""

NATIVE_SHOT_PROMPTS = {
    "baseline": NATIVE_SHOT_PROMPT,
    "strict_release_v2": STRICT_RELEASE_NATIVE_PROMPT,
    "negative_first_v3": NEGATIVE_FIRST_NATIVE_PROMPT,
    "compact_json_v1": COMPACT_JSON_NATIVE_PROMPT,
}


def configure_native_video_processor(processor: Any, *, max_pixels: int) -> None:
    """Apply the frozen video pixel budget to a supported MLX-VLM processor."""
    if max_pixels <= 0:
        raise ValueError("native-video max pixels must be positive")
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None and hasattr(video_processor, "max_pixels"):
        min_pixels = int(getattr(video_processor, "min_pixels", 0))
        if min_pixels > max_pixels:
            raise ValueError("native-video max pixels are below the processor minimum")
        video_processor.max_pixels = int(max_pixels)
        return

    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    max_image_size = getattr(image_processor, "max_image_size", None)
    video_sampling = getattr(image_processor, "video_sampling", None)
    video_size = video_sampling.get("video_size") if isinstance(video_sampling, dict) else None
    if (
        size is None
        or not hasattr(size, "longest_edge")
        or not isinstance(max_image_size, dict)
        or not isinstance(video_size, dict)
    ):
        raise ValueError("MLX processor does not expose a video processor")
    longest_edge = math.isqrt(int(max_pixels))
    if longest_edge <= 0:
        raise ValueError("native-video max pixels are below the processor minimum")
    size.longest_edge = longest_edge
    max_image_size["longest_edge"] = longest_edge
    video_size["longest_edge"] = longest_edge


def sample_native_frame_indices(
    *,
    start_frame: int,
    end_frame: int,
    source_fps: float,
    sample_fps: float,
) -> list[int]:
    """Return deterministic, even-count frame indices for native video input."""
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("native-video frame bounds are invalid")
    if source_fps <= 0 or sample_fps <= 0:
        raise ValueError("native-video FPS values must be positive")
    total_frames = end_frame - start_frame + 1
    if total_frames < 2:
        raise ValueError("native-video frame bounds are too short")
    duration_seconds = total_frames / source_fps
    frame_count = max(4, int(round(duration_seconds * sample_fps)))
    frame_count = min(frame_count, total_frames)
    if frame_count > 2 and frame_count % 2:
        frame_count -= 1
    frame_count = max(2, frame_count)
    return np.linspace(
        start_frame,
        end_frame,
        frame_count,
        dtype=int,
    ).tolist()


def sample_native_video_window(
    path: Path,
    *,
    start_frame: int,
    end_frame: int,
    source_fps: float,
    sample_fps: float,
) -> np.ndarray:
    """Decode one raw-bound window as RGB ``(T,C,H,W)`` video frames."""
    import cv2

    indices = sample_native_frame_indices(
        start_frame=start_frame,
        end_frame=end_frame,
        source_fps=source_fps,
        sample_fps=sample_fps,
    )
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open native-video source: {path.name}")
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not decode native-video frame {frame_index}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.transpose(np.stack(frames, axis=0), (0, 3, 1, 2))


def normalize_native_model_output(text: str) -> dict[str, object]:
    """Parse one model response and fail closed on malformed output."""
    try:
        payload = extract_json_object(text)
    except (ValueError, json.JSONDecodeError):
        return {
            **parse_independent_shot_vlm_decision(
                {
                    "field_goal_state": "unknown",
                    "confidence": 0.0,
                    "reason": "native VLM returned no valid JSON object",
                }
            ),
            "available": False,
            "raw_response": text[:2000],
        }
    return {
        **parse_independent_shot_vlm_decision(payload),
        "available": True,
        "raw_response": text[:2000],
    }


class MlxNativeVideoReviewer:
    """Load one local MLX VLM once and review multiple raw video windows."""

    def __init__(
        self,
        *,
        model_path: str,
        max_pixels: int,
        max_tokens: int = 260,
        prefill_step_size: int = 2048,
        prompt: str = NATIVE_SHOT_PROMPT,
        loader: Callable[[str], tuple[Any, Any]] | None = None,
        formatter: Callable[..., str] | None = None,
        generator: Callable[..., Any] | None = None,
    ) -> None:
        if max_tokens <= 0 or prefill_step_size <= 0:
            raise ValueError("native-video generation limits must be positive")
        if loader is None or formatter is None or generator is None:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template

            loader = loader or load
            formatter = formatter or apply_chat_template
            generator = generator or generate
        self.model, self.processor = loader(model_path)
        configure_native_video_processor(self.processor, max_pixels=max_pixels)
        self._formatter = formatter
        self._generator = generator
        self.max_tokens = int(max_tokens)
        self.prefill_step_size = int(prefill_step_size)
        if not str(prompt).strip():
            raise ValueError("native-video prompt must be non-empty")
        self.prompt = str(prompt)

    def review(
        self,
        frames: np.ndarray,
        *,
        sample_fps: float,
    ) -> dict[str, object]:
        if frames.ndim != 4 or frames.shape[0] < 2 or frames.shape[1] != 3:
            raise ValueError("native-video frames must have shape (T,C,H,W)")
        formatted_prompt = self._formatter(
            self.processor,
            self.model.config,
            self.prompt,
            video=["raw_window"],
            fps=sample_fps,
            num_images=int(frames.shape[0]),
        )
        try:
            result = self._generator(
                self.model,
                self.processor,
                formatted_prompt,
                video=[frames],
                fps=sample_fps,
                max_tokens=self.max_tokens,
                temperature=0.0,
                prefill_step_size=self.prefill_step_size,
                verbose=False,
            )
        except Exception as exc:
            detail = " ".join(str(exc).split())[:300]
            reason = f"native VLM unavailable: {type(exc).__name__}"
            if detail:
                reason = f"{reason}: {detail}"
            return {
                **parse_independent_shot_vlm_decision(
                    {
                        "field_goal_state": "unknown",
                        "confidence": 0.0,
                        "reason": reason,
                    }
                ),
                "available": False,
                "raw_response": "",
            }
        return normalize_native_model_output(str(getattr(result, "text", result)))


class TransformersNativeVideoReviewer:
    """Review raw video tensors with the unquantized local Transformers model."""

    def __init__(
        self,
        *,
        model_path: str,
        max_pixels: int,
        device: str,
        torch_dtype: str = "bfloat16",
        max_tokens: int = 260,
        prompt: str = NATIVE_SHOT_PROMPT,
        model_loader: Callable[..., Any] | None = None,
        processor_loader: Callable[..., Any] | None = None,
    ) -> None:
        if max_pixels <= 0 or max_tokens <= 0:
            raise ValueError("Transformers native-video limits must be positive")
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("unsupported Transformers native-video device")
        import torch

        if model_loader is None or processor_loader is None:
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )

            model_loader = model_loader or Qwen2_5_VLForConditionalGeneration.from_pretrained
            processor_loader = processor_loader or AutoProcessor.from_pretrained
        dtype: object
        if torch_dtype == "auto":
            dtype = "auto"
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError("unsupported Transformers native-video dtype")
        self.model = (
            model_loader(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        self.processor = processor_loader(
            model_path,
            local_files_only=True,
            min_pixels=min(max_pixels, 128 * 28 * 28),
            max_pixels=max_pixels,
        )
        self.device = device
        self.max_tokens = int(max_tokens)
        if not str(prompt).strip():
            raise ValueError("native-video prompt must be non-empty")
        self.prompt = str(prompt)

    def review(
        self,
        frames: np.ndarray,
        *,
        sample_fps: float,
    ) -> dict[str, object]:
        if frames.ndim != 4 or frames.shape[0] < 2 or frames.shape[1] != 3:
            raise ValueError("native-video frames must have shape (T,C,H,W)")
        if sample_fps <= 0:
            raise ValueError("native-video sample FPS must be positive")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            video = np.transpose(frames, (0, 2, 3, 1))
            inputs = self.processor(
                text=[prompt],
                images=None,
                videos=[video],
                fps=sample_fps,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            input_length = int(inputs["input_ids"].shape[1])
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
            response = self.processor.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            detail = " ".join(str(exc).split())[:300]
            reason = f"native Transformers VLM unavailable: {type(exc).__name__}"
            if detail:
                reason = f"{reason}: {detail}"
            return {
                **parse_independent_shot_vlm_decision(
                    {
                        "field_goal_state": "unknown",
                        "confidence": 0.0,
                        "reason": reason,
                    }
                ),
                "available": False,
                "raw_response": "",
            }
        return normalize_native_model_output(str(response))


def run_native_video_plan(
    *,
    plan: Mapping[str, Any],
    videos: Sequence[Path],
    review_window: Callable[..., Mapping[str, Any]],
    model: Mapping[str, Any],
    frame_sampler: Callable[..., np.ndarray] = sample_native_video_window,
    cache_path: Path | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    prompt: str = NATIVE_SHOT_PROMPT,
    prompt_variant: str = "baseline",
) -> dict[str, Any]:
    """Run a frozen label-free native-video plan and seal every prediction."""
    verified = verify_independent_shot_vlm_plan(plan)
    contract = dict(verified.get("input_contract") or {})
    if contract.get("raw_video_clips_only") is not True:
        raise ValueError("native-video plan must use raw clips only")
    if contract.get("labels_or_review_notes_exposed_to_model") is not False:
        raise ValueError("native-video plan must not expose labels")
    if contract.get("native_temporal_position_encoding") is not True:
        raise ValueError("native-video plan must preserve temporal position encoding")
    sample_fps = float(contract.get("sample_fps", 0.0))
    max_pixels = int(contract.get("max_pixels", 0))
    if sample_fps <= 0 or max_pixels <= 0:
        raise ValueError("native-video plan sampling contract is invalid")
    if not str(prompt).strip() or not str(prompt_variant).strip():
        raise ValueError("native-video prompt and variant must be non-empty")
    prompt = str(prompt)
    prompt_variant = str(prompt_variant)

    video_by_sha = {_file_sha256(path): path for path in videos}
    required = {str(row["source_video_sha256"]) for row in verified["examples"]}
    if set(video_by_sha) != required:
        raise ValueError("provided videos do not exactly match the native-video plan")
    for row in verified["examples"]:
        path = video_by_sha[str(row["source_video_sha256"])]
        if path.name != str(row["source_video_filename"]):
            raise ValueError("native-video source filename mismatch")

    model_provenance = {
        "name": str(model.get("name") or ""),
        "source": str(model.get("source") or ""),
        "revision": str(model.get("revision") or ""),
        "weights_sha256": str(model.get("weights_sha256") or ""),
        "independent_from_codex": True,
    }
    if not all(model_provenance[name] for name in ("name", "source", "revision", "weights_sha256")):
        raise ValueError("native-video model provenance is incomplete")

    fingerprint = _cache_fingerprint(
        plan_sha256=str(verified["plan_sha256"]),
        model=model_provenance,
        sample_fps=sample_fps,
        max_pixels=max_pixels,
        prompt=prompt,
        prompt_variant=prompt_variant,
    )
    cache = _load_cache(cache_path, fingerprint=fingerprint)
    predictions = []
    for index, row in enumerate(verified["examples"], start=1):
        key = _example_cache_key(row)
        decision = cache["predictions"].get(key)
        if not isinstance(decision, dict):
            frames = frame_sampler(
                video_by_sha[str(row["source_video_sha256"])],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                source_fps=float(row["source_fps"]),
                sample_fps=sample_fps,
            )
            decision = _normalize_decision(review_window(frames, sample_fps=sample_fps))
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
        if progress is not None:
            progress(
                {
                    "stage": "independent_native_vlm",
                    "completed": index,
                    "total": len(verified["examples"]),
                    "event_id": row["event_id"],
                    "state": decision["field_goal_state"],
                    "available": decision["available"],
                }
            )
    return seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": verified["plan_sha256"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_variant": prompt_variant,
            "sampling": {
                "native_temporal_position_encoding": True,
                "sample_fps": sample_fps,
                "max_pixels": max_pixels,
            },
            "model": model_provenance,
            "predictions": predictions,
        }
    )


def _normalize_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    observables = dict(payload.get("observables") or {})
    available = bool(payload.get("available", False))
    normalized = parse_independent_shot_vlm_decision(
        {
            **payload,
            **observables,
            **({"field_goal_state": "unknown"} if not available else {}),
        }
    )
    return {
        **normalized,
        "available": available,
        "raw_response": str(payload.get("raw_response") or "")[:2000],
    }


def _example_cache_key(row: Mapping[str, Any]) -> str:
    return ":".join(
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
    )


def _cache_fingerprint(
    *,
    plan_sha256: str,
    model: Mapping[str, Any],
    sample_fps: float,
    max_pixels: int,
    prompt: str = NATIVE_SHOT_PROMPT,
    prompt_variant: str = "baseline",
) -> str:
    payload = {
        "plan_sha256": plan_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_variant": prompt_variant,
        "model": dict(model),
        "sample_fps": sample_fps,
        "max_pixels": max_pixels,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_cache(path: Path | None, *, fingerprint: str) -> dict[str, Any]:
    empty: dict[str, Any] = {"fingerprint": fingerprint, "predictions": {}}
    if path is None or not path.exists():
        return empty
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("native-video VLM cache fingerprint mismatch")
    if not isinstance(payload.get("predictions"), dict):
        raise ValueError("native-video VLM cache is invalid")
    return payload


def _write_cache(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
