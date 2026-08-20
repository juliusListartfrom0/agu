from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceIdentityResult:
    embedding: np.ndarray
    model_id: str
    sample_count: int
    quality: float
    prototype_embeddings: tuple[np.ndarray, ...] = ()
    prototype_qualities: tuple[float, ...] = ()
    prototype_sample_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class DetectedFace:
    """One YuNet detection with its SFace embedding and aligned face crop."""

    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray
    aligned_crop: np.ndarray


class FaceIdentityAdapter(Protocol):
    model_id: str

    def embed_player_crops(self, crops_bgr: Sequence[np.ndarray]) -> Optional[FaceIdentityResult]: ...


class OpenCvSFaceIdentityAdapter:
    """Local YuNet face detection and SFace recognition adapter."""

    model_id = "opencv_yunet_2023mar+sface_2021dec_embedding_v1"

    def __init__(
        self,
        detector_model_path: str,
        recognizer_model_path: str,
        score_threshold: float = 0.60,
    ) -> None:
        detector_path = Path(detector_model_path).expanduser()
        recognizer_path = Path(recognizer_model_path).expanduser()
        if not detector_path.is_file():
            raise FileNotFoundError(f"YuNet face detector model not found: {detector_path}")
        if not recognizer_path.is_file():
            raise FileNotFoundError(f"SFace recognizer model not found: {recognizer_path}")

        self.score_threshold = max(0.0, min(1.0, float(score_threshold)))
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path),
            "",
            (320, 320),
            self.score_threshold,
            0.30,
            5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self._lock = threading.Lock()

    def embed_player_crops(self, crops_bgr: Sequence[np.ndarray]) -> Optional[FaceIdentityResult]:
        embeddings = []
        with self._lock:
            for crop in crops_bgr:
                embedding = self._embed_best_face(crop)
                if embedding is not None:
                    embeddings.append(embedding)
        reliable_embeddings, quality = self._select_consistent_embeddings(embeddings)
        if not reliable_embeddings:
            return None
        aggregate = _l2_normalize(np.stack(reliable_embeddings, axis=0).mean(axis=0))
        prototypes = _supported_face_prototypes(reliable_embeddings)
        if not prototypes:
            prototypes = [(aggregate, quality, len(reliable_embeddings))]
        return FaceIdentityResult(
            embedding=aggregate,
            model_id=self.model_id,
            sample_count=len(reliable_embeddings),
            quality=quality,
            prototype_embeddings=tuple(item[0] for item in prototypes),
            prototype_qualities=tuple(item[1] for item in prototypes),
            prototype_sample_counts=tuple(item[2] for item in prototypes),
        )

    def detect_frame_faces(
        self,
        frame_bgr: np.ndarray,
        *,
        minimum_face_size: int = 24,
    ) -> list[DetectedFace]:
        """Detect and embed every sufficiently large face in a full raw frame."""

        with self._lock:
            return self._detect_faces(
                frame_bgr,
                minimum_face_size=minimum_face_size,
                require_upper_player_region=False,
            )

    def _select_consistent_embeddings(
        self,
        embeddings: Sequence[np.ndarray],
    ) -> tuple[list[np.ndarray], float]:
        if len(embeddings) < 2:
            return [], 0.0
        matrix = np.stack(embeddings, axis=0)
        similarities = matrix @ matrix.T
        medoid_index = int(np.argmax(similarities.mean(axis=1)))
        selected_indices = [
            index for index, similarity in enumerate(similarities[medoid_index]) if float(similarity) >= 0.50
        ]
        required_count = max(2, (len(embeddings) + 1) // 2)
        if len(selected_indices) < required_count:
            return [], 0.0
        selected = [embeddings[index] for index in selected_indices]
        pair_scores = [
            float(selected[left] @ selected[right])
            for left in range(len(selected))
            for right in range(left + 1, len(selected))
        ]
        quality = _bounded_quality(np.mean(pair_scores)) if pair_scores else 0.0
        if quality < 0.50:
            return [], quality
        return selected, quality

    def _embed_best_face(self, player_crop: np.ndarray) -> Optional[np.ndarray]:
        if player_crop is None or player_crop.size == 0 or min(player_crop.shape[:2]) < 20:
            return None
        faces = self._detect_faces(
            player_crop,
            minimum_face_size=10,
            require_upper_player_region=True,
        )
        if not faces:
            return None
        height, width = player_crop.shape[:2]
        return max(
            faces,
            key=lambda item: item.confidence - abs(((item.bbox[0] + item.bbox[2] / 2.0) / width) - 0.5) * 0.15,
        ).embedding

    def _detect_faces(
        self,
        image: np.ndarray,
        *,
        minimum_face_size: int,
        require_upper_player_region: bool,
    ) -> list[DetectedFace]:
        if image is None or image.size == 0 or min(image.shape[:2]) < minimum_face_size:
            return []
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None:
            return []
        detected: list[DetectedFace] = []
        for face in faces:
            face_width = float(face[2])
            face_height = float(face[3])
            center_x = (float(face[0]) + face_width / 2.0) / width
            center_y = (float(face[1]) + face_height / 2.0) / height
            if face_width < minimum_face_size or face_height < minimum_face_size:
                continue
            if require_upper_player_region and not (-0.02 <= center_x <= 1.02 and -0.02 <= center_y <= 0.42):
                continue
            try:
                aligned = self._recognizer.alignCrop(image, face)
                embedding = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
            except cv2.error as exc:
                LOGGER.debug("SFace rejected a detected face crop: %s", exc)
                continue
            if embedding.size == 0:
                continue
            detected.append(
                DetectedFace(
                    bbox=(
                        max(0, int(round(float(face[0])))),
                        max(0, int(round(float(face[1])))),
                        max(1, int(round(face_width))),
                        max(1, int(round(face_height))),
                    ),
                    confidence=float(face[-1]),
                    embedding=_l2_normalize(embedding),
                    aligned_crop=aligned.copy(),
                )
            )
        return detected


def build_face_identity_adapter(
    backend: str,
    detector_model_path: str,
    recognizer_model_path: str,
    score_threshold: float = 0.60,
    allow_fallback: bool = True,
) -> Optional[OpenCvSFaceIdentityAdapter]:
    normalized_backend = (backend or "opencv_sface_if_available").strip().lower()
    if normalized_backend in {"off", "none", "haar", "opencv_haar"}:
        return None
    if normalized_backend not in {
        "opencv_sface",
        "opencv_sface_if_available",
        "yunet_sface",
    }:
        raise ValueError(f"Unsupported face identity backend: {backend}")
    try:
        return OpenCvSFaceIdentityAdapter(
            detector_model_path=detector_model_path,
            recognizer_model_path=recognizer_model_path,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        if not allow_fallback or normalized_backend == "opencv_sface":
            raise
        LOGGER.warning("Falling back to Haar face sampling because OpenCV SFace is unavailable: %s", exc)
        return None


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _supported_face_prototypes(
    embeddings: Sequence[np.ndarray],
    *,
    similarity_threshold: float = 0.65,
    maximum_prototypes: int = 4,
) -> list[tuple[np.ndarray, float, int]]:
    """Partition consistent samples into supported view templates.

    A prototype needs at least two mutually compatible observations. This avoids
    turning a single unusual pose into a permissive gallery template while
    preserving distinct frontal/profile or lighting modes that should not be
    averaged into one weak identity vector.
    """

    if len(embeddings) < 2:
        return []
    matrix = np.stack([_l2_normalize(item) for item in embeddings], axis=0)
    similarities = matrix @ matrix.T
    remaining = set(range(len(embeddings)))
    prototypes: list[tuple[np.ndarray, float, int]] = []
    while len(remaining) >= 2 and len(prototypes) < maximum_prototypes:
        seed = max(
            remaining,
            key=lambda index: (
                sum(float(similarities[index, other]) >= similarity_threshold for other in remaining),
                float(np.mean([similarities[index, other] for other in remaining])),
                -index,
            ),
        )
        ordered = sorted(
            remaining - {seed},
            key=lambda index: (float(similarities[seed, index]), -index),
            reverse=True,
        )
        cluster = [seed]
        for candidate in ordered:
            if all(float(similarities[candidate, member]) >= similarity_threshold for member in cluster):
                cluster.append(candidate)
        if len(cluster) < 2:
            remaining.remove(seed)
            continue
        vectors = matrix[cluster]
        pair_scores = [
            float(vectors[left] @ vectors[right])
            for left in range(len(vectors))
            for right in range(left + 1, len(vectors))
        ]
        prototypes.append(
            (
                _l2_normalize(vectors.mean(axis=0)),
                _bounded_quality(np.mean(pair_scores)),
                len(cluster),
            )
        )
        remaining.difference_update(cluster)
    return sorted(prototypes, key=lambda item: (item[2], item[1]), reverse=True)


def _bounded_quality(value: float) -> float:
    """Keep float32 cosine-derived quality inside its public [0, 1] contract."""

    return min(1.0, max(0.0, float(value)))
