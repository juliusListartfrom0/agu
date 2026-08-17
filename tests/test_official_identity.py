from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.analysis.face_gallery import parse_face_gallery, seal_face_gallery_payload
from app.analysis.face_identity import FaceIdentityResult
from app.analysis.identity_embedding import BaseIdentityEmbedder, IdentityEmbeddingResult
from app.analysis.identity_graph import CanonicalPlayerIdentity
from app.analysis.official_identity import (
    _attach_registered_jersey_identities,
    _infer_canonical_team_bindings,
    _team_after_inferred_binding,
    build_official_identity_artifact,
    canonical_player_for_observation,
    canonicalize_perception_detections,
    verify_official_identity_artifact,
)
from app.analysis.schemas import JerseyNumberCandidateResponse, PerceptionDetectionResponse
from app.analysis.vlm import OllamaVLMVerifier


class MeanColorEmbedder(BaseIdentityEmbedder):
    model_id = "mean-color-test-v1"
    method = "mean-color-test-v1"

    def embed_crops(self, crops_bgr: list[np.ndarray]) -> IdentityEmbeddingResult:
        vector = np.mean([crop.reshape(-1, 3).mean(axis=0) for crop in crops_bgr], axis=0).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-6)
        return IdentityEmbeddingResult(vector, self.model_id, self.method)


class MeanColorFaceAdapter:
    model_id = "sface-test"

    def embed_player_crops(self, crops_bgr: list[np.ndarray]) -> FaceIdentityResult:
        is_red = float(np.mean([crop[:, :, 2].mean() for crop in crops_bgr])) > float(
            np.mean([crop[:, :, 0].mean() for crop in crops_bgr])
        )
        vector = np.array([1.0, 0.0] if is_red else [0.0, 1.0], dtype=np.float32)
        return FaceIdentityResult(vector, self.model_id, len(crops_bgr), 0.95)


class LowQualityMeanColorFaceAdapter(MeanColorFaceAdapter):
    def embed_player_crops(self, crops_bgr: list[np.ndarray]) -> FaceIdentityResult:
        result = super().embed_player_crops(crops_bgr)
        return FaceIdentityResult(result.embedding, result.model_id, result.sample_count, 0.60)


class MeanColorJerseyReader:
    model = "fixture-jersey-vlm"

    def read_jersey_number(
        self,
        frames: list[np.ndarray],
        scope: str = "",
    ) -> list[JerseyNumberCandidateResponse]:
        del scope
        red = float(np.mean([frame[:, :, 2].mean() for frame in frames]))
        blue = float(np.mean([frame[:, :, 0].mean() for frame in frames]))
        return [
            JerseyNumberCandidateResponse(
                number="7" if red > blue else "9",
                confidence=0.96,
                visible=True,
            )
        ]


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 100))
    assert writer.isOpened()
    for _frame in range(6):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:90, 10:35] = (20, 20, 230)
        image[10:90, 60:85] = (230, 20, 20)
        writer.write(image)
    writer.release()


def _detection(frame: int, player_id: str, box: tuple[int, int, int, int]) -> dict[str, object]:
    return {
        "detection_id": f"{player_id}:{frame}",
        "frame": frame,
        "object_type": "player",
        "bbox": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
        "confidence": 0.9,
        "track_id": player_id,
        "player_id": player_id,
        "team_id": "raw-dark",
        "backend": "fixture",
    }


def test_official_identity_artifact_merges_disjoint_same_player_tracklets(tmp_path: Path) -> None:
    video = tmp_path / "period.mp4"
    _video(video)
    detections = [
        *[_detection(frame, "raw-track-1", (10, 10, 35, 90)) for frame in range(3)],
        *[_detection(frame, "raw-track-2", (10, 10, 35, 90)) for frame in range(3, 6)],
        *[_detection(frame, "raw-track-3", (60, 10, 85, 90)) for frame in range(6)],
    ]
    payload = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "size_bytes": video.stat().st_size,
            "source_fps": 10.0,
            "frame_count": 6,
        },
        "detections": detections,
    }

    artifact = build_official_identity_artifact(
        perception_payloads=[payload],
        video_paths=[video],
        embedder=MeanColorEmbedder(),
        embedding_threshold=0.95,
        minimum_observations=3,
        maximum_crops=3,
    )

    assert len(artifact.tracklets) == 3
    assert len(artifact.identities) == 2
    first = canonical_player_for_observation(
        artifact, source_video_id="video_001", source_player_id="raw-track-1", frame=1
    )
    second = canonical_player_for_observation(
        artifact, source_video_id="video_001", source_player_id="raw-track-2", frame=4
    )
    assert first == second
    mapped, source_video_id = canonicalize_perception_detections(
        artifact,
        [PerceptionDetectionResponse.model_validate(_detection(1, "raw-track-1", (10, 10, 35, 90)))],
        raw_filename=video.name,
        raw_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
    )
    assert source_video_id == "video_001"
    assert mapped[0].player_id == first
    with pytest.raises(ValueError, match="exactly one matching"):
        canonicalize_perception_detections(
            artifact,
            [],
            raw_filename=video.name,
            raw_sha256="wrong",
        )
    assert verify_official_identity_artifact(artifact) is artifact

    artifact.artifact_sha256 = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_official_identity_artifact(artifact)


def test_official_identity_rejects_untracked_perception(tmp_path: Path) -> None:
    video = tmp_path / "period.mp4"
    _video(video)
    payload = {
        "schema_version": "agu.official-perception.v1",
        "detector": {"player_tracking": False},
        "raw_video": {
            "filename": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "size_bytes": video.stat().st_size,
            "source_fps": 10.0,
            "frame_count": 6,
        },
        "detections": [_detection(frame, "player", (10, 10, 35, 90)) for frame in range(6)],
    }

    with pytest.raises(ValueError, match="player_tracking=true"):
        build_official_identity_artifact(
            perception_payloads=[payload],
            video_paths=[video],
            embedder=MeanColorEmbedder(),
        )


def test_official_identity_fails_fast_when_tracking_is_too_fragmented(tmp_path: Path) -> None:
    video = tmp_path / "period.mp4"
    _video(video)
    detections = [
        _detection(frame, f"raw-track-{frame}", (10, 10, 35, 90))
        for frame in range(6)
    ]
    payload = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "size_bytes": video.stat().st_size,
            "source_fps": 10.0,
            "frame_count": 6,
        },
        "detections": detections,
    }

    with pytest.raises(ValueError, match="safe graph limit"):
        build_official_identity_artifact(
            perception_payloads=[payload],
            video_paths=[video],
            embedder=MeanColorEmbedder(),
            minimum_observations=1,
            maximum_tracklets=5,
        )


def test_official_identity_can_limit_work_to_event_source_tracks(tmp_path: Path) -> None:
    video = tmp_path / "period.mp4"
    _video(video)
    payload = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "size_bytes": video.stat().st_size,
            "source_fps": 10.0,
            "frame_count": 6,
        },
        "detections": [
            *[_detection(frame, "event-track", (10, 10, 35, 90)) for frame in range(3)],
            *[_detection(frame, "unrelated-track", (60, 10, 85, 90)) for frame in range(3)],
        ],
    }

    artifact = build_official_identity_artifact(
        perception_payloads=[payload],
        video_paths=[video],
        embedder=MeanColorEmbedder(),
        minimum_observations=1,
        source_player_ids={"event-track"},
    )

    assert {tracklet.source_player_id for tracklet in artifact.tracklets} == {"event-track"}
    assert artifact.model_provenance["source_player_scope_count"] == "1"


def test_official_identity_anchors_canonical_ids_to_face_gallery(tmp_path: Path) -> None:
    video = tmp_path / "period.mp4"
    _video(video)
    detections = [
        *[_detection(frame, "raw-track-1", (10, 10, 35, 90)) for frame in range(3)],
        *[_detection(frame, "raw-track-2", (10, 10, 35, 90)) for frame in range(3, 6)],
        *[_detection(frame, "raw-track-3", (60, 10, 85, 90)) for frame in range(6)],
    ]
    payload = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "size_bytes": video.stat().st_size,
            "source_fps": 10.0,
            "frame_count": 6,
        },
        "detections": detections,
    }
    gallery = parse_face_gallery(
        seal_face_gallery_payload(
            {
                "model_id": "sface-test",
                "source_manifest_sha256": "fixture",
                "benchmark_disjoint": True,
                "entries": [
                    {
                        "person_id": "annotated-red",
                        "team_id": "canonical-team",
                        "jersey_number": "7",
                        "embedding": [1.0, 0.0],
                        "sample_count": 2,
                        "quality": 0.95,
                    },
                    {
                        "person_id": "annotated-blue",
                        "team_id": "canonical-team",
                        "jersey_number": "9",
                        "embedding": [0.0, 1.0],
                        "sample_count": 2,
                        "quality": 0.95,
                    },
                ],
            }
        )
    )

    artifact = build_official_identity_artifact(
        perception_payloads=[payload],
        video_paths=[video],
        embedder=MeanColorEmbedder(),
        embedding_threshold=0.95,
        minimum_observations=3,
        maximum_crops=3,
        face_identity_adapter=MeanColorFaceAdapter(),
        face_gallery=gallery,
    )

    assert {identity.player_id for identity in artifact.identities} == {
        "annotated-red",
        "annotated-blue",
    }
    assert artifact.model_provenance["face_gallery"] == gallery.gallery_sha256
    assert {tracklet.team_id for tracklet in artifact.tracklets} == {"canonical-team"}
    assert {identity.jersey_number for identity in artifact.identities} == {"7", "9"}

    low_quality = build_official_identity_artifact(
        perception_payloads=[payload],
        video_paths=[video],
        embedder=MeanColorEmbedder(),
        embedding_threshold=0.95,
        minimum_observations=3,
        maximum_crops=3,
        face_identity_adapter=LowQualityMeanColorFaceAdapter(),
        face_gallery=gallery,
    )
    assert not any(
        evidence.startswith("face_gallery=")
        for identity in low_quality.identities
        for evidence in identity.evidence
    )
    assert all(tracklet.gallery_person_id for tracklet in artifact.tracklets)


def test_team_binding_requires_multiple_dominant_face_anchors() -> None:
    def tracklet(tracklet_id: str, raw_team: str, canonical_team: str, person_id: str | None):
        return MagicMock(
            tracklet_id=tracklet_id,
            team_id=canonical_team,
            gallery_person_id=person_id,
        )

    tracklets = [
        tracklet("a1", "raw-dark", "ATL", "p1"),
        tracklet("a2", "raw-dark", "ATL", "p2"),
        tracklet("a3", "raw-dark", "CHI", "p3"),
        tracklet("b1", "raw-light", "CHI", "p4"),
        tracklet("b2", "raw-light", "CHI", "p5"),
        tracklet("weak", "raw-third", "ATL", "p6"),
    ]

    raw_teams = {
        "a1": "raw-dark",
        "a2": "raw-dark",
        "a3": "raw-dark",
        "b1": "raw-light",
        "b2": "raw-light",
        "weak": "raw-third",
    }
    assert _infer_canonical_team_bindings(
        tracklets,
        raw_team_by_tracklet=raw_teams,
    ) == {}
    assert _infer_canonical_team_bindings(
        tracklets,
        raw_team_by_tracklet=raw_teams,
        minimum_dominance=0.60,
        minimum_anchor_count=2,
    ) == {"raw-dark": "ATL", "raw-light": "CHI"}


def test_inferred_team_binding_never_overwrites_direct_gallery_team() -> None:
    bindings = {"raw-dark": "ATL"}

    assert _team_after_inferred_binding(
        current_team_id="CHI",
        gallery_person_id="2586",
        raw_team_id="raw-dark",
        inferred_team_bindings=bindings,
    ) == "CHI"
    assert _team_after_inferred_binding(
        current_team_id="raw-dark",
        gallery_person_id=None,
        raw_team_id="raw-dark",
        inferred_team_bindings=bindings,
    ) == "ATL"


def test_official_identity_uses_only_trusted_agu_jersey_reads(tmp_path: Path) -> None:
    video = tmp_path / "period.mp4"
    _video(video)
    detections = [
        *[_detection(frame, "raw-track-1", (10, 10, 35, 90)) for frame in range(3)],
        *[_detection(frame, "raw-track-2", (10, 10, 35, 90)) for frame in range(3, 6)],
        *[_detection(frame, "raw-track-3", (60, 10, 85, 90)) for frame in range(6)],
    ]
    payload = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "size_bytes": video.stat().st_size,
            "source_fps": 10.0,
            "frame_count": 6,
        },
        "detections": detections,
    }

    artifact = build_official_identity_artifact(
        perception_payloads=[payload],
        video_paths=[video],
        embedder=MeanColorEmbedder(),
        embedding_threshold=0.99,
        minimum_observations=3,
        maximum_crops=3,
        jersey_number_reader=MeanColorJerseyReader(),
        jersey_number_minimum_crops=3,
        jersey_number_minimum_consensus=1,
    )

    assert {identity.player_id for identity in artifact.identities} == {
        "raw-dark-jersey-7",
        "raw-dark-jersey-9",
    }
    assert artifact.model_provenance["jersey_number_reader"] == "fixture-jersey-vlm"


def test_registered_jersey_maps_to_face_entry_only_when_graph_key_is_unique() -> None:
    gallery = parse_face_gallery(
        seal_face_gallery_payload(
            {
                "model_id": "sface-test",
                "source_manifest_sha256": "fixture",
                "benchmark_disjoint": True,
                "entries": [
                    {
                        "person_id": "registered-13",
                        "team_id": "CHI",
                        "jersey_number": "13",
                        "embedding": [1.0, 0.0],
                        "sample_count": 2,
                        "quality": 0.95,
                    }
                ],
            }
        )
    )
    identity = CanonicalPlayerIdentity(
        player_id="CHI-jersey-13",
        team_id="CHI",
        tracklet_ids=("track-1",),
        jersey_number="13",
        confidence=0.95,
        evidence=("trusted_jersey=13",),
    )

    mapped = _attach_registered_jersey_identities([identity], face_gallery=gallery)
    ambiguous = _attach_registered_jersey_identities([identity, identity], face_gallery=gallery)

    assert mapped[0].player_id == "registered-13"
    assert "face_gallery_jersey=registered-13" in mapped[0].evidence
    assert all(item.player_id == "CHI-jersey-13" for item in ambiguous)


def test_registered_jersey_ignores_existing_anchor_for_same_enrolled_person() -> None:
    gallery = parse_face_gallery(
        seal_face_gallery_payload(
            {
                "model_id": "sface-test",
                "source_manifest_sha256": "fixture",
                "benchmark_disjoint": True,
                "entries": [
                    {
                        "person_id": "registered-1",
                        "team_id": "CHI",
                        "jersey_number": "1",
                        "embedding": [1.0, 0.0],
                        "sample_count": 2,
                        "quality": 0.95,
                    }
                ],
            }
        )
    )
    anchored = CanonicalPlayerIdentity(
        player_id="registered-1-ambiguous-fixture",
        team_id="CHI",
        tracklet_ids=("face-track",),
        jersey_number="1",
        confidence=0.95,
        evidence=("trusted_jersey=1", "face_gallery=registered-1", "ambiguous_duplicate_anchor"),
    )
    runtime = CanonicalPlayerIdentity(
        player_id="CHI-jersey-1",
        team_id="CHI",
        tracklet_ids=("runtime-track",),
        jersey_number="1",
        confidence=0.95,
        evidence=("trusted_jersey=1",),
    )

    mapped = _attach_registered_jersey_identities([anchored, runtime], face_gallery=gallery)

    assert mapped[0].player_id == "registered-1-ambiguous-fixture"
    assert mapped[1].player_id == "registered-1"
    assert "face_gallery_jersey=registered-1" in mapped[1].evidence


def test_jersey_reader_passes_configured_context_to_ollama() -> None:
    verifier = OllamaVLMVerifier(
        model="fixture-model",
        host="http://127.0.0.1:11434",
        context_length=16384,
        seed=7,
    )
    response = MagicMock()
    response.read.return_value = json.dumps(
        {"response": '{"number":"7","confidence":0.95,"visible":true,"reason":"clear"}'}
    ).encode("utf-8")

    with (
        patch("app.analysis.vlm.encode_frames_jpeg", return_value=["encoded"]),
        patch("urllib.request.urlopen") as urlopen,
    ):
        urlopen.return_value.__enter__.return_value = response
        result = verifier.read_jersey_number([np.zeros((16, 16, 3), dtype=np.uint8)])

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["options"]["num_ctx"] == 16384
    assert payload["options"]["seed"] == 7
    assert result[0].number == "7"
