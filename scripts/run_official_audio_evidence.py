#!/usr/bin/env python3
"""Extract conservative raw-audio action candidates without confirming statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.audio_evidence import (  # noqa: E402
    AudioRosterArtifact,
    attach_audio_mentions,
    build_audio_evidence,
    derive_audio_evidence_subset,
    seal_transcript_artifact,
    sha256_file,
    speech_candidate_events,
    verify_audio_roster,
)
from app.analysis.game_state.candidates import link_causal_candidate_relations  # noqa: E402
from app.analysis.official_inference import seal_agu_autonomous_predictions  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402
from app.config import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--transcript-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--enable", action="store_true", default=settings.official_audio_asr_enabled)
    parser.add_argument("--model", default=settings.official_audio_asr_model)
    parser.add_argument("--language", default=settings.official_audio_asr_language)
    parser.add_argument("--causal-link-seconds", type=float, default=4.0)
    parser.add_argument(
        "--candidate-action",
        action="append",
        choices=(
            "assist",
            "block",
            "field_goal_attempt",
            "foul",
            "rebound",
            "steal",
            "turnover",
        ),
        help="Limit unmatched speech-only candidates; repeat for multiple types.",
    )
    return parser.parse_args()


def run_audio_evidence(
    *,
    candidate_bundle: RawOnlyPredictionBundleResponse,
    video_path: Path,
    roster: AudioRosterArtifact,
    transcript_path: Path,
    evidence_output: Path,
    model: str,
    language: str,
    causal_link_seconds: float,
    candidate_action_types: set[str] | None = None,
) -> RawOnlyPredictionBundleResponse:
    if len(candidate_bundle.raw_videos) != 1:
        raise ValueError("audio evidence currently requires one declared raw video")
    source_video = candidate_bundle.raw_videos[0]
    video_sha256 = sha256_file(video_path)
    if source_video.filename != video_path.name or source_video.sha256 != video_sha256:
        raise ValueError("raw video does not match the candidate bundle")
    verify_audio_roster(roster)
    transcript = _load_or_transcribe(
        transcript_path,
        video_path=video_path,
        video_sha256=video_sha256,
        roster_sha256=roster.artifact_sha256,
        model=model,
        language=language,
        prompt=", ".join(player.display_name for player in roster.players),
    )
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if fps <= 0:
        raise RuntimeError("video FPS is unavailable")
    if causal_link_seconds < 0:
        raise ValueError("causal link seconds must be non-negative")
    evidence = build_audio_evidence(
        transcript_artifact=transcript,
        roster=roster,
        raw_video_sha256=video_sha256,
        fps=fps,
    )
    _write_json_atomic(evidence_output, evidence.model_dump())
    attached_events, attached_mentions = attach_audio_mentions(candidate_bundle.events, evidence)
    unmatched_evidence = derive_audio_evidence_subset(
        evidence,
        excluded_mention_ids=attached_mentions,
    )
    events = [
        *attached_events,
        *speech_candidate_events(
            unmatched_evidence,
            source_video_id=source_video.video_id,
            include_action_types=candidate_action_types,
        ),
    ]
    events = link_causal_candidate_relations(
        events,
        max_gap_frames=int(round(fps * causal_link_seconds)),
    )
    return seal_agu_autonomous_predictions(
        game_id=candidate_bundle.game_id,
        raw_video_paths=[video_path],
        events=events,
        config={
            "pipeline": "agu_official_audio_evidence_v1",
            "source_candidate_config_sha256": candidate_bundle.config_sha256,
            "audio_evidence_sha256": evidence.artifact_sha256,
            "roster_artifact_sha256": roster.artifact_sha256,
            "speech_candidate_action_types": (
                sorted(candidate_action_types) if candidate_action_types is not None else None
            ),
        },
        candidate_backend=candidate_bundle.model_provenance.get("candidate_backend", "agu_traditional_perception"),
        semantic_backend="raw_audio_asr_candidate",
        semantic_model=evidence.model,
        model_provenance={
            "source_candidate_bundle_sha256": candidate_bundle.bundle_sha256,
            "audio_evidence_sha256": evidence.artifact_sha256,
            "roster_artifact_sha256": roster.artifact_sha256,
        },
    )


def _load_or_transcribe(
    path: Path,
    *,
    video_path: Path,
    video_sha256: str,
    roster_sha256: str,
    model: str,
    language: str,
    prompt: str,
) -> dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("video_sha256") != video_sha256:
            raise ValueError("transcript cache raw-video hash mismatch")
        if payload.get("model") != model:
            raise ValueError("transcript cache model mismatch")
        return payload
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper is an optional ASR adapter; install it or provide a matching transcript cache"
        ) from exc
    transcript = mlx_whisper.transcribe(
        str(video_path),
        path_or_hf_repo=model,
        language=language,
        initial_prompt=prompt,
        word_timestamps=True,
    )
    payload = seal_transcript_artifact({
        "schema_version": "agu.raw-audio-transcript.v1",
        "role": "raw_only_inference",
        "video_sha256": video_sha256,
        "roster_source_sha256": roster_sha256,
        "model": model,
        "language": language,
        "transcript": transcript,
    })
    _write_json_atomic(path, payload)
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not args.enable:
        raise RuntimeError(
            "raw-audio ASR is opt-in; pass --enable or set BASKETBALL_OFFICIAL_AUDIO_ASR_ENABLED=true"
        )
    if not str(args.model).strip():
        raise RuntimeError(
            "select an ASR model/path with --model or BASKETBALL_OFFICIAL_AUDIO_ASR_MODEL"
        )
    candidate_bundle = RawOnlyPredictionBundleResponse.model_validate_json(
        args.candidate_bundle.read_text(encoding="utf-8")
    )
    roster = AudioRosterArtifact.model_validate_json(args.roster.read_text(encoding="utf-8"))
    result = run_audio_evidence(
        candidate_bundle=candidate_bundle,
        video_path=args.video,
        roster=roster,
        transcript_path=args.transcript_cache,
        evidence_output=args.evidence_output,
        model=args.model,
        language=args.language,
        causal_link_seconds=args.causal_link_seconds,
        candidate_action_types=set(args.candidate_action) if args.candidate_action else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bundle_sha256": result.bundle_sha256, "event_count": len(result.events)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
