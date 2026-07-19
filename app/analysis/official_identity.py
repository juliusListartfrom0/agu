"""Raw-video-only identity tracklet extraction and conservative stitching."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np

from app.analysis.face_gallery import FaceGallery, match_face_gallery
from app.analysis.face_identity import FaceIdentityAdapter
from app.analysis.identity_embedding import BaseIdentityEmbedder
from app.analysis.identity_graph import IdentityGraph, IdentityTracklet
from app.analysis.schemas import (
    JerseyNumberCandidateResponse,
    OfficialCanonicalIdentityResponse,
    OfficialIdentityGraphArtifactResponse,
    OfficialIdentityTrackletResponse,
    PerceptionDetectionResponse,
    RawVideoAssetResponse,
)


class JerseyNumberReader(Protocol):
    """AGU-owned jersey OCR/VLM adapter used only on raw tracklet crops."""

    model: str

    def read_jersey_number(
        self,
        frames: Sequence[np.ndarray],
        scope: str = "",
    ) -> list[JerseyNumberCandidateResponse]: ...


def build_official_identity_artifact(
    *,
    perception_payloads: Sequence[Mapping[str, Any]],
    video_paths: Sequence[Path],
    embedder: BaseIdentityEmbedder,
    embedding_threshold: float = 0.92,
    minimum_observations: int = 3,
    maximum_crops: int = 8,
    maximum_tracklet_gap_sec: float = 2.0,
    face_identity_adapter: FaceIdentityAdapter | None = None,
    face_gallery: FaceGallery | None = None,
    face_gallery_similarity_threshold: float = 0.45,
    face_gallery_minimum_margin: float = 0.08,
    enrolled_minimum_face_quality: float = 0.65,
    face_match_threshold: float = 0.45,
    face_conflict_threshold: float = 0.30,
    jersey_number_reader: JerseyNumberReader | None = None,
    jersey_number_minimum_confidence: float = 0.90,
    jersey_number_minimum_crops: int = 4,
    jersey_number_minimum_consensus: int = 2,
    jersey_source_player_ids: set[str] | None = None,
) -> OfficialIdentityGraphArtifactResponse:
    """Build an AGU identity graph, optionally anchored by a sealed face gallery."""

    if len(perception_payloads) != len(video_paths) or not perception_payloads:
        raise ValueError("perception payloads and videos must be non-empty and have equal length")
    if not 0.0 <= embedding_threshold <= 1.0:
        raise ValueError("embedding_threshold must be in [0,1]")
    if minimum_observations <= 0 or maximum_crops <= 0 or maximum_tracklet_gap_sec <= 0:
        raise ValueError("identity observation, crop and gap settings must be positive")
    if face_gallery is not None and face_identity_adapter is None:
        raise ValueError("face gallery requires a face identity adapter")
    if not 0.0 <= enrolled_minimum_face_quality <= 1.0:
        raise ValueError("enrolled_minimum_face_quality must be in [0,1]")
    if not 0.0 <= jersey_number_minimum_confidence <= 1.0:
        raise ValueError("jersey_number_minimum_confidence must be in [0,1]")
    if jersey_number_minimum_crops <= 0:
        raise ValueError("jersey_number_minimum_crops must be positive")
    if jersey_number_minimum_consensus <= 0:
        raise ValueError("jersey_number_minimum_consensus must be positive")

    raw_videos: list[RawVideoAssetResponse] = []
    pending: list[tuple[str, str, str, list[PerceptionDetectionResponse]]] = []
    for video_index, (payload, video_path) in enumerate(zip(perception_payloads, video_paths), start=1):
        if payload.get("schema_version") != "agu.official-perception.v1":
            raise ValueError("unsupported official perception schema")
        detector = payload.get("detector") or {}
        if detector.get("player_tracking") is False:
            raise ValueError(
                "official identity requires perception produced with player_tracking=true"
            )
        raw = payload.get("raw_video") or {}
        if raw.get("filename") != video_path.name or raw.get("sha256") != _file_sha256(video_path):
            raise ValueError("perception artifact does not match raw video")
        source_video_id = f"video_{video_index:03d}"
        raw_videos.append(
            RawVideoAssetResponse(
                video_id=source_video_id,
                filename=video_path.name,
                sha256=str(raw["sha256"]),
                size_bytes=video_path.stat().st_size,
            )
        )
        source_fps = float(raw.get("source_fps") or 0.0)
        if source_fps <= 0:
            raise ValueError("perception source_fps must be positive")
        detections = [PerceptionDetectionResponse.model_validate(item) for item in payload.get("detections") or []]
        grouped: dict[str, list[PerceptionDetectionResponse]] = defaultdict(list)
        for detection in detections:
            if detection.object_type == "player" and detection.player_id and detection.team_id:
                grouped[str(detection.player_id)].append(detection)
        maximum_gap = max(1, int(round(maximum_tracklet_gap_sec * source_fps)))
        for source_player_id, observations in sorted(grouped.items()):
            for segment_index, segment in enumerate(_split_observations(observations, maximum_gap), start=1):
                if len(segment) < minimum_observations:
                    continue
                tracklet_id = f"{source_video_id}:{source_player_id}:{segment_index:03d}"
                team_id = Counter(str(item.team_id) for item in segment if item.team_id).most_common(1)[0][0]
                pending.append((tracklet_id, source_video_id, team_id, segment))

    selected = {
        tracklet_id: _select_observations(observations, maximum_crops)
        for tracklet_id, _source_video_id, _team_id, observations in pending
    }
    crops = _read_tracklet_crops(video_paths, pending, selected)
    tracklet_responses: list[OfficialIdentityTrackletResponse] = []
    graph_tracklets: list[IdentityTracklet] = []
    for tracklet_id, source_video_id, team_id, observations in pending:
        tracklet_crops = crops.get(tracklet_id) or []
        if not tracklet_crops:
            continue
        embedded = embedder.embed_crops(tracklet_crops)
        if not np.any(embedded.embedding):
            continue
        signature = _appearance_signature(tracklet_crops)
        face_result = (
            face_identity_adapter.embed_player_crops(tracklet_crops) if face_identity_adapter is not None else None
        )
        gallery_match = (
            match_face_gallery(
                face_gallery,
                face_result.embedding,
                model_id=face_result.model_id,
                team_id=team_id,
                similarity_threshold=face_gallery_similarity_threshold,
                minimum_margin=face_gallery_minimum_margin,
            )
            if (
                face_gallery is not None
                and face_result is not None
                and face_result.quality >= enrolled_minimum_face_quality
            )
            else None
        )
        resolved_team_id = gallery_match.team_id if gallery_match and gallery_match.team_id else team_id
        source_player_id = str(observations[0].player_id)
        jersey_number, jersey_confidence = _read_trusted_jersey_number(
            jersey_number_reader,
            tracklet_crops,
            tracklet_id=tracklet_id,
            source_player_id=source_player_id,
            minimum_confidence=jersey_number_minimum_confidence,
            minimum_crops=jersey_number_minimum_crops,
            minimum_consensus=jersey_number_minimum_consensus,
            allowed_source_player_ids=jersey_source_player_ids,
        )
        response = OfficialIdentityTrackletResponse(
            tracklet_id=tracklet_id,
            source_video_id=source_video_id,
            source_player_id=source_player_id,
            team_id=resolved_team_id,
            start_frame=min(item.frame for item in observations),
            end_frame=max(item.frame for item in observations),
            observation_count=len(observations),
            crop_count=len(tracklet_crops),
            embedding=[float(value) for value in embedded.embedding.tolist()],
            embedding_model=embedded.model_id,
            appearance_signature=signature,
            sampled_boxes=[
                {
                    "frame": float(item.frame),
                    "x1": float(item.bbox.x1),
                    "y1": float(item.bbox.y1),
                    "x2": float(item.bbox.x2),
                    "y2": float(item.bbox.y2),
                }
                for item in selected.get(tracklet_id) or []
            ],
            face_embedding=(
                [float(value) for value in face_result.embedding.tolist()] if face_result is not None else []
            ),
            face_embedding_model=face_result.model_id if face_result is not None else None,
            face_sample_count=face_result.sample_count if face_result is not None else 0,
            face_embedding_quality=face_result.quality if face_result is not None else 0.0,
            gallery_person_id=gallery_match.person_id if gallery_match is not None else None,
            gallery_confidence=gallery_match.confidence if gallery_match is not None else 0.0,
        )
        tracklet_responses.append(response)
        graph_tracklets.append(
            IdentityTracklet(
                tracklet_id=tracklet_id,
                team_id=resolved_team_id,
                start_frame=response.start_frame,
                end_frame=response.end_frame,
                embedding=tuple(response.embedding),
                source_video_id=source_video_id,
                spatial_samples=tuple(
                    (
                        item.frame,
                        (item.bbox.x1 + item.bbox.x2) / 2.0,
                        (item.bbox.y1 + item.bbox.y2) / 2.0,
                        item.bbox.y2 - item.bbox.y1,
                    )
                    for item in observations
                ),
                gallery_person_id=gallery_match.person_id if gallery_match is not None else None,
                gallery_confidence=gallery_match.confidence if gallery_match is not None else 0.0,
                face_embedding=(
                    tuple(float(value) for value in face_result.embedding.tolist()) if face_result is not None else ()
                ),
                face_embedding_model=face_result.model_id if face_result is not None else None,
                face_quality=face_result.quality if face_result is not None else 0.0,
                jersey_number=jersey_number,
                jersey_confidence=jersey_confidence,
            )
        )

    resolved = IdentityGraph(
        embedding_threshold=embedding_threshold,
        face_match_threshold=face_match_threshold,
        face_conflict_threshold=face_conflict_threshold,
        enrolled_minimum_face_quality=enrolled_minimum_face_quality,
    ).resolve(graph_tracklets)
    response_by_id = {item.tracklet_id: item for item in tracklet_responses}
    identities: list[OfficialCanonicalIdentityResponse] = []
    tracklet_to_player_id: dict[str, str] = {}
    source_player_to_player_ids: dict[str, set[str]] = defaultdict(set)
    for identity in resolved:
        members = [response_by_id[item] for item in identity.tracklet_ids]
        source_ids = sorted({member.source_player_id for member in members})
        identities.append(
            OfficialCanonicalIdentityResponse(
                player_id=identity.player_id,
                team_id=identity.team_id,
                tracklet_ids=list(identity.tracklet_ids),
                source_player_ids=source_ids,
                jersey_number=identity.jersey_number,
                confidence=identity.confidence,
                evidence=list(identity.evidence),
            )
        )
        for member in members:
            tracklet_to_player_id[member.tracklet_id] = identity.player_id
            source_player_to_player_ids[f"{member.source_video_id}:{member.source_player_id}"].add(identity.player_id)

    config = {
        "embedding_threshold": embedding_threshold,
        "minimum_observations": minimum_observations,
        "maximum_crops": maximum_crops,
        "maximum_tracklet_gap_sec": maximum_tracklet_gap_sec,
        "face_gallery_sha256": face_gallery.gallery_sha256 if face_gallery is not None else "",
        "face_gallery_similarity_threshold": face_gallery_similarity_threshold,
        "face_gallery_minimum_margin": face_gallery_minimum_margin,
        "enrolled_minimum_face_quality": enrolled_minimum_face_quality,
        "face_match_threshold": face_match_threshold,
        "face_conflict_threshold": face_conflict_threshold,
        "jersey_number_reader": (
            jersey_number_reader.model if jersey_number_reader is not None else "disabled"
        ),
        "jersey_number_minimum_confidence": jersey_number_minimum_confidence,
        "jersey_number_minimum_crops": jersey_number_minimum_crops,
        "jersey_number_minimum_consensus": jersey_number_minimum_consensus,
        "jersey_source_player_ids": sorted(jersey_source_player_ids or []),
        "enrolled_identity_propagation": "direct_face_or_trusted_jersey",
    }
    artifact = OfficialIdentityGraphArtifactResponse(
        raw_videos=raw_videos,
        config_sha256=_canonical_sha256(config),
        model_provenance={
            "producer": "agu",
            "inference_mode": "traditional_reid",
            "embedding_model": tracklet_responses[0].embedding_model if tracklet_responses else embedder.model_id,
            "graph_backend": "complete_link_identity_graph_v3_enrollment_direct_evidence",
            "face_gallery": face_gallery.gallery_sha256 if face_gallery is not None else "disabled",
            "face_embedding_model": face_gallery.model_id if face_gallery is not None else "disabled",
            "jersey_number_reader": (
                jersey_number_reader.model if jersey_number_reader is not None else "disabled"
            ),
        },
        tracklets=tracklet_responses,
        identities=identities,
        tracklet_to_player_id=tracklet_to_player_id,
        source_player_to_player_ids={key: sorted(value) for key, value in source_player_to_player_ids.items()},
    )
    artifact.artifact_sha256 = _canonical_sha256(
        artifact.model_copy(update={"artifact_sha256": ""}).model_dump(mode="json")
    )
    return artifact


def _read_trusted_jersey_number(
    reader: JerseyNumberReader | None,
    crops: Sequence[np.ndarray],
    *,
    tracklet_id: str,
    source_player_id: str,
    minimum_confidence: float,
    minimum_crops: int,
    minimum_consensus: int,
    allowed_source_player_ids: set[str] | None,
) -> tuple[str | None, float]:
    if (
        reader is None
        or len(crops) < minimum_crops
        or (allowed_source_player_ids is not None and source_player_id not in allowed_source_player_ids)
    ):
        return None, 0.0
    partitions = [list(crops[index::minimum_consensus]) for index in range(minimum_consensus)]
    if any(not partition for partition in partitions):
        return None, 0.0
    accepted: list[JerseyNumberCandidateResponse] = []
    for index, partition in enumerate(partitions, start=1):
        candidates = reader.read_jersey_number(
            partition,
            scope=f"{tracklet_id}:consensus-{index}-of-{minimum_consensus}",
        )
        visible = [
            item
            for item in candidates
            if item.visible
            and item.number is not None
            and item.number.isdigit()
            and item.confidence >= minimum_confidence
        ]
        if len(visible) != 1:
            return None, 0.0
        accepted.append(visible[0])
    numbers = {item.number for item in accepted}
    if len(numbers) != 1:
        return None, 0.0
    return accepted[0].number, min(item.confidence for item in accepted)


def canonical_player_for_observation(
    artifact: OfficialIdentityGraphArtifactResponse,
    *,
    source_video_id: str,
    source_player_id: str,
    frame: int,
) -> str | None:
    """Resolve a raw track observation only when one graph tracklet contains it."""

    candidates = [
        item
        for item in artifact.tracklets
        if item.source_video_id == source_video_id
        and item.source_player_id == source_player_id
        and item.start_frame <= frame <= item.end_frame
    ]
    if len(candidates) != 1:
        return None
    return artifact.tracklet_to_player_id.get(candidates[0].tracklet_id)


def canonicalize_perception_detections(
    artifact: OfficialIdentityGraphArtifactResponse,
    detections: Sequence[PerceptionDetectionResponse],
    *,
    raw_filename: str,
    raw_sha256: str,
) -> tuple[list[PerceptionDetectionResponse], str]:
    """Apply a matching sealed identity graph to traditional detections."""

    verify_official_identity_artifact(artifact)
    matching_assets = [
        item for item in artifact.raw_videos if item.filename == raw_filename and item.sha256 == raw_sha256
    ]
    if len(matching_assets) != 1:
        raise ValueError("identity graph must contain exactly one matching raw video")
    source_video_id = matching_assets[0].video_id
    canonicalized: list[PerceptionDetectionResponse] = []
    for detection in detections:
        if detection.object_type != "player" or not detection.player_id:
            canonicalized.append(detection)
            continue
        canonical = canonical_player_for_observation(
            artifact,
            source_video_id=source_video_id,
            source_player_id=detection.player_id,
            frame=detection.frame,
        )
        canonicalized.append(detection.model_copy(update={"player_id": canonical or detection.player_id}))
    return canonicalized, source_video_id


def verify_official_identity_artifact(
    artifact: OfficialIdentityGraphArtifactResponse,
) -> OfficialIdentityGraphArtifactResponse:
    expected = _canonical_sha256(artifact.model_copy(update={"artifact_sha256": ""}).model_dump(mode="json"))
    if artifact.artifact_sha256 != expected:
        raise ValueError("official identity artifact hash mismatch")
    if artifact.model_provenance.get("producer") != "agu":
        raise ValueError("official identity artifact must be AGU-produced")
    return artifact


def _split_observations(
    observations: Sequence[PerceptionDetectionResponse], maximum_gap: int
) -> list[list[PerceptionDetectionResponse]]:
    result: list[list[PerceptionDetectionResponse]] = []
    for item in sorted(observations, key=lambda value: (value.frame, value.detection_id)):
        if not result or item.frame - result[-1][-1].frame > maximum_gap:
            result.append([item])
        else:
            result[-1].append(item)
    return result


def _select_observations(
    observations: Sequence[PerceptionDetectionResponse], maximum: int
) -> list[PerceptionDetectionResponse]:
    ordered = sorted(observations, key=lambda item: (item.frame, item.detection_id))
    if len(ordered) <= maximum:
        return ordered
    selected: list[PerceptionDetectionResponse] = []
    boundaries = np.linspace(0, len(ordered), maximum + 1, dtype=int)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        bucket = ordered[int(start) : max(int(start) + 1, int(end))]
        selected.append(
            max(
                bucket,
                key=lambda item: (
                    item.confidence * math.sqrt(max(1.0, _box_area(item))),
                    -item.frame,
                    item.detection_id,
                ),
            )
        )
    return selected


def _read_tracklet_crops(
    video_paths: Sequence[Path],
    pending: Sequence[tuple[str, str, str, list[PerceptionDetectionResponse]]],
    selected: Mapping[str, Sequence[PerceptionDetectionResponse]],
) -> dict[str, list[np.ndarray]]:
    requests: dict[str, dict[int, list[tuple[str, PerceptionDetectionResponse]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for tracklet_id, source_video_id, _team_id, _observations in pending:
        for observation in selected.get(tracklet_id) or []:
            requests[source_video_id][observation.frame].append((tracklet_id, observation))
    crops: dict[str, list[np.ndarray]] = defaultdict(list)
    for video_index, video_path in enumerate(video_paths, start=1):
        source_video_id = f"video_{video_index:03d}"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open raw video for identity extraction: {video_path}")
        next_frame = -1
        try:
            for frame_number in sorted(requests[source_video_id]):
                if next_frame < 0 or frame_number < next_frame or frame_number - next_frame > 90:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    next_frame = frame_number
                frame = None
                while next_frame <= frame_number:
                    ok, current = capture.read()
                    if not ok:
                        frame = None
                        break
                    frame = current
                    next_frame += 1
                if frame is None:
                    continue
                for tracklet_id, observation in requests[source_video_id][frame_number]:
                    crop = _crop_detection(frame, observation)
                    if crop is not None:
                        crops[tracklet_id].append(crop)
        finally:
            capture.release()
    return dict(crops)


def _crop_detection(frame: np.ndarray, detection: PerceptionDetectionResponse) -> np.ndarray | None:
    height, width = frame.shape[:2]
    box = detection.bbox
    box_width = box.x2 - box.x1
    box_height = box.y2 - box.y1
    if box_width < 15 or box_height < 50:
        return None
    x1 = max(0, min(width - 1, int(round(box.x1 - box_width * 0.03))))
    x2 = max(x1 + 1, min(width, int(round(box.x2 + box_width * 0.03))))
    y1 = max(0, min(height - 1, int(round(box.y1 - box_height * 0.02))))
    y2 = max(y1 + 1, min(height, int(round(box.y2 + box_height * 0.02))))
    if x1 == 0 or x2 == width:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


def _appearance_signature(crops: Sequence[np.ndarray]) -> dict[str, float]:
    medians: list[np.ndarray] = []
    for crop in crops:
        y1, y2 = int(crop.shape[0] * 0.18), max(1, int(crop.shape[0] * 0.58))
        x1, x2 = int(crop.shape[1] * 0.20), max(1, int(crop.shape[1] * 0.80))
        torso = crop[y1:y2, x1:x2]
        if torso.size:
            medians.append(np.median(cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0))
    if not medians:
        return {}
    values = np.mean(np.stack(medians), axis=0)
    return {"torso_lab_l": float(values[0]), "torso_lab_a": float(values[1]), "torso_lab_b": float(values[2])}


def _box_area(item: PerceptionDetectionResponse) -> float:
    return max(0.0, item.bbox.x2 - item.bbox.x1) * max(0.0, item.bbox.y2 - item.bbox.y1)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
