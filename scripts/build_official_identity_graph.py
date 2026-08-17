#!/usr/bin/env python3
"""Build AGU stable raw-player identities from raw video and perception only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_gallery import load_face_gallery  # noqa: E402
from app.analysis.face_identity import build_face_identity_adapter  # noqa: E402
from app.analysis.identity_embedding import build_identity_embedder  # noqa: E402
from app.analysis.jersey_identity import CachedJerseyNumberReader  # noqa: E402
from app.analysis.official_identity import build_official_identity_artifact  # noqa: E402
from app.analysis.vlm import OllamaVLMVerifier  # noqa: E402
from app.config import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception", type=Path, action="append", required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-backend", default="torchvision_mobilenet_v3_small")
    parser.add_argument("--embedding-weights", default="default")
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--embedding-threshold", type=float, default=0.92)
    parser.add_argument("--minimum-observations", type=int, default=3)
    parser.add_argument("--maximum-crops", type=int, default=8)
    parser.add_argument("--maximum-tracklet-gap-sec", type=float, default=2.0)
    parser.add_argument("--maximum-tracklets", type=int, default=2000)
    parser.add_argument(
        "--source-player-id",
        action="append",
        default=[],
        help="Optional raw player ID allowlist for event-scoped identity resolution",
    )
    parser.add_argument(
        "--event-candidate-bundle",
        type=Path,
        help="Derive the raw player allowlist from an official event-candidate bundle",
    )
    parser.add_argument(
        "--event-candidate-player-prefix",
        help=(
            "Optional perception namespace such as 'perception-3:'; only matching "
            "candidate IDs are used and the prefix is removed before raw-track lookup"
        ),
    )
    parser.add_argument("--face-gallery", type=Path)
    parser.add_argument(
        "--face-identity",
        action=argparse.BooleanOptionalAction,
        default=settings.face_identity_backend.strip().lower() not in {"off", "none", "disabled"},
    )
    parser.add_argument("--face-detector-model", type=Path)
    parser.add_argument("--face-recognizer-model", type=Path)
    parser.add_argument("--face-detection-score-threshold", type=float)
    parser.add_argument("--face-gallery-similarity-threshold", type=float)
    parser.add_argument("--face-gallery-minimum-margin", type=float)
    parser.add_argument("--enrolled-minimum-face-quality", type=float)
    parser.add_argument("--face-match-threshold", type=float)
    parser.add_argument("--face-conflict-threshold", type=float)
    parser.add_argument("--jersey-number-vlm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jersey-model")
    parser.add_argument("--jersey-host")
    parser.add_argument("--jersey-timeout", type=float)
    parser.add_argument("--jersey-image-width", type=int, default=512)
    parser.add_argument("--jersey-context-length", type=int)
    parser.add_argument("--jersey-seed", type=int, default=0)
    parser.add_argument("--jersey-cache", type=Path)
    parser.add_argument("--jersey-minimum-confidence", type=float, default=0.90)
    parser.add_argument("--jersey-minimum-crops", type=int, default=4)
    parser.add_argument("--jersey-minimum-consensus", type=int, default=2)
    parser.add_argument(
        "--jersey-source-player-id",
        action="append",
        default=[],
        help="Optional raw player ID allowlist for staged jersey recognition",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.perception]
    source_player_ids = set(args.source_player_id)
    if args.event_candidate_bundle is not None:
        source_player_ids.update(
            _candidate_player_ids(
                json.loads(args.event_candidate_bundle.read_text(encoding="utf-8")),
                source_prefix=args.event_candidate_player_prefix,
            )
        )
    embedder = build_identity_embedder(
        backend=args.embedding_backend,
        weights=args.embedding_weights,
        device=args.device,
        allow_fallback=False,
    )
    face_gallery_path = args.face_gallery or (Path(settings.face_gallery_path) if settings.face_gallery_path else None)
    face_gallery = load_face_gallery(face_gallery_path) if face_gallery_path else None
    face_adapter = None
    if args.face_identity or face_gallery is not None:
        detector_model = args.face_detector_model or Path(settings.face_detection_model_path)
        recognizer_model = args.face_recognizer_model or Path(settings.face_recognition_model_path)
        face_adapter = build_face_identity_adapter(
            backend="opencv_sface" if face_gallery is not None else settings.face_identity_backend,
            detector_model_path=str(detector_model),
            recognizer_model_path=str(recognizer_model),
            score_threshold=(
                args.face_detection_score_threshold
                if args.face_detection_score_threshold is not None
                else settings.face_detection_score_threshold
            ),
            allow_fallback=False if face_gallery is not None else settings.face_identity_allow_fallback,
        )
    jersey_reader = (
        OllamaVLMVerifier(
            model=args.jersey_model or settings.ollama_model,
            host=args.jersey_host or settings.ollama_host,
            timeout=args.jersey_timeout or settings.official_vlm_timeout,
            image_width=args.jersey_image_width,
            context_length=args.jersey_context_length or settings.official_vlm_context_length,
            seed=args.jersey_seed,
        )
        if args.jersey_number_vlm
        else None
    )
    if jersey_reader is not None and args.jersey_cache is not None:
        jersey_reader = CachedJerseyNumberReader(jersey_reader, args.jersey_cache)
    artifact = build_official_identity_artifact(
        perception_payloads=payloads,
        video_paths=args.video,
        embedder=embedder,
        embedding_threshold=args.embedding_threshold,
        minimum_observations=args.minimum_observations,
        maximum_crops=args.maximum_crops,
        maximum_tracklet_gap_sec=args.maximum_tracklet_gap_sec,
        maximum_tracklets=args.maximum_tracklets,
        face_identity_adapter=face_adapter,
        face_gallery=face_gallery,
        face_gallery_similarity_threshold=(
            args.face_gallery_similarity_threshold
            if args.face_gallery_similarity_threshold is not None
            else settings.face_gallery_similarity_threshold
        ),
        face_gallery_minimum_margin=(
            args.face_gallery_minimum_margin
            if args.face_gallery_minimum_margin is not None
            else settings.face_gallery_minimum_margin
        ),
        enrolled_minimum_face_quality=(
            args.enrolled_minimum_face_quality
            if args.enrolled_minimum_face_quality is not None
            else settings.face_enrolled_minimum_quality
        ),
        face_match_threshold=(
            args.face_match_threshold
            if args.face_match_threshold is not None
            else settings.face_identity_match_threshold
        ),
        face_conflict_threshold=(
            args.face_conflict_threshold
            if args.face_conflict_threshold is not None
            else settings.face_identity_conflict_threshold
        ),
        jersey_number_reader=jersey_reader,
        jersey_number_minimum_confidence=args.jersey_minimum_confidence,
        jersey_number_minimum_crops=args.jersey_minimum_crops,
        jersey_number_minimum_consensus=args.jersey_minimum_consensus,
        jersey_source_player_ids=(set(args.jersey_source_player_id) if args.jersey_source_player_id else None),
        source_player_ids=(source_player_ids or None),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    merged = sum(len(identity.tracklet_ids) > 1 for identity in artifact.identities)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact.artifact_sha256,
                "tracklet_count": len(artifact.tracklets),
                "identity_count": len(artifact.identities),
                "merged_identity_count": merged,
                "embedding_model": artifact.model_provenance.get("embedding_model"),
            },
            indent=2,
        )
    )
    return 0


def _candidate_player_ids(
    payload: dict[str, object],
    *,
    source_prefix: str | None = None,
) -> set[str]:
    """Collect raw actor IDs from sealed event evidence without reading truth."""

    if payload.get("schema_version") != "agu.raw-only.v1":
        raise ValueError("unsupported official event-candidate schema")
    result: set[str] = set()
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        for evidence in event.get("evidence") or []:
            if not isinstance(evidence, dict):
                continue
            details = evidence.get("details") or {}
            if not isinstance(details, dict):
                continue
            candidates = [
                str(item) for item in details.get("candidate_player_ids") or [] if item
            ]
            for observation in details.get("candidate_player_observations") or []:
                if isinstance(observation, dict) and observation.get("player_id"):
                    candidates.append(str(observation["player_id"]))
            for candidate in candidates:
                if source_prefix is None:
                    result.add(candidate)
                elif candidate.startswith(source_prefix):
                    result.add(candidate.removeprefix(source_prefix))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
