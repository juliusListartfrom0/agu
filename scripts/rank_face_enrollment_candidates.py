#!/usr/bin/env python3
"""Rank benchmark-disjoint enrollment face clusters for offline identity review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_gallery import FaceGallery, load_face_gallery  # noqa: E402
from app.analysis.face_identity import OpenCvSFaceIdentityAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument("--gallery", type=Path)
    reference_group.add_argument("--single-reference-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--recognizer-model", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def rank_face_enrollment_candidates(
    candidate_payload: Mapping[str, Any],
    gallery: FaceGallery,
    cluster_embeddings: Mapping[str, np.ndarray],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Score enrollment clusters without promoting any identity automatically."""

    if candidate_payload.get("schema_version") != "agu.face-enrollment-candidates.v1":
        raise ValueError("unsupported face enrollment candidate schema")
    if candidate_payload.get("benchmark_disjoint") is not True:
        raise ValueError("enrollment candidates must be benchmark-disjoint")
    source_sha256 = str(candidate_payload.get("manifest_sha256") or "")
    expected_sha256 = _canonical_sha256({**dict(candidate_payload), "manifest_sha256": ""})
    if not source_sha256 or source_sha256 != expected_sha256:
        raise ValueError("face enrollment candidate manifest hash mismatch")
    if str(candidate_payload.get("model_id") or "") != gallery.model_id:
        raise ValueError("face enrollment and gallery models do not match")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    clusters = {
        str(item.get("cluster_id") or ""): item
        for item in candidate_payload.get("clusters") or []
    }
    if not clusters or "" in clusters or len(clusters) != len(candidate_payload.get("clusters") or []):
        raise ValueError("candidate cluster IDs must be non-empty and unique")
    unknown = sorted(set(cluster_embeddings) - set(clusters))
    if unknown:
        raise ValueError(f"embedding supplied for unknown cluster: {unknown}")

    normalized_embeddings: dict[str, np.ndarray] = {}
    for cluster_id, raw_embedding in cluster_embeddings.items():
        embedding = np.asarray(raw_embedding, dtype=np.float32)
        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
            raise ValueError(f"invalid embedding for cluster: {cluster_id}")
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0:
            raise ValueError(f"invalid embedding for cluster: {cluster_id}")
        normalized_embeddings[cluster_id] = embedding / norm

    scores_by_cluster: dict[str, dict[str, float]] = {}
    for cluster_id, embedding in normalized_embeddings.items():
        person_scores: dict[str, float] = {}
        for entry in gallery.entries:
            person_scores[entry.person_id] = max(
                float(embedding @ np.asarray(prototype, dtype=np.float32))
                for prototype in entry.prototypes
            )
        scores_by_cluster[cluster_id] = person_scores

    rankings = []
    for entry in gallery.entries:
        candidates = []
        for cluster_id, person_scores in scores_by_cluster.items():
            target_score = person_scores[entry.person_id]
            competing = [
                (score, person_id)
                for person_id, score in person_scores.items()
                if person_id != entry.person_id
            ]
            runner_up_score, runner_up_person_id = max(
                competing,
                default=(-1.0, None),
                key=lambda item: (item[0], item[1] or ""),
            )
            cluster = clusters[cluster_id]
            candidates.append(
                {
                    "cluster_id": cluster_id,
                    "similarity": target_score,
                    "runner_up_person_id": runner_up_person_id,
                    "runner_up_similarity": runner_up_score,
                    "person_margin": target_score - runner_up_score,
                    "observation_count": int(cluster.get("observation_count") or 0),
                    "review_sheet": cluster.get("review_sheet"),
                    "source_video_ids": sorted(
                        {
                            str(sample.get("source_video_id") or "")
                            for sample in cluster.get("samples") or []
                            if sample.get("source_video_id")
                        }
                    ),
                }
            )
        candidates.sort(key=lambda row: (-float(row["similarity"]), str(row["cluster_id"])))
        rankings.append(
            {
                "person_id": entry.person_id,
                "team_id": entry.team_id,
                "reference_quality": entry.quality,
                "candidates": candidates[:top_k],
            }
        )

    payload = {
        "schema_version": "agu.face-enrollment-ranking.v1",
        "benchmark_disjoint": True,
        "runtime_consumable": False,
        "identity_verified": False,
        "codex_runtime_answer_used": False,
        "producer": "agu_sface_offline_candidate_ranker",
        "model_id": gallery.model_id,
        "source_candidate_manifest_sha256": source_sha256,
        "source_gallery_sha256": gallery.gallery_sha256,
        "top_k": top_k,
        "rankings": rankings,
        "unembedded_cluster_ids": sorted(set(clusters) - set(normalized_embeddings)),
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def rank_face_enrollment_reference_embeddings(
    candidate_payload: Mapping[str, Any],
    reference_payload: Mapping[str, Any],
    reference_embeddings: Mapping[str, np.ndarray],
    cluster_embeddings: Mapping[str, np.ndarray],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Use single verified portraits only to prioritize enrollment review."""

    if candidate_payload.get("schema_version") != "agu.face-enrollment-candidates.v1":
        raise ValueError("unsupported face enrollment candidate schema")
    if candidate_payload.get("benchmark_disjoint") is not True:
        raise ValueError("enrollment candidates must be benchmark-disjoint")
    candidate_sha256 = str(candidate_payload.get("manifest_sha256") or "")
    if candidate_sha256 != _canonical_sha256(
        {**dict(candidate_payload), "manifest_sha256": ""}
    ):
        raise ValueError("face enrollment candidate manifest hash mismatch")
    if reference_payload.get("schema_version") != "agu.single-face-reference-list.v1":
        raise ValueError("unsupported single face reference schema")
    if (
        reference_payload.get("benchmark_disjoint") is not True
        or reference_payload.get("runtime_consumable") is not False
    ):
        raise ValueError("single face references must be benchmark-disjoint and offline-only")
    reference_sha256 = str(reference_payload.get("manifest_sha256") or "")
    if reference_sha256 != _canonical_sha256(
        {**dict(reference_payload), "manifest_sha256": ""}
    ):
        raise ValueError("single face reference manifest hash mismatch")
    model_id = str(reference_payload.get("model_id") or "")
    if model_id != str(candidate_payload.get("model_id") or ""):
        raise ValueError("face enrollment and reference models do not match")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    clusters = {
        str(item.get("cluster_id") or ""): item
        for item in candidate_payload.get("clusters") or []
    }
    if not clusters or "" in clusters or len(clusters) != len(candidate_payload.get("clusters") or []):
        raise ValueError("candidate cluster IDs must be non-empty and unique")
    references = {
        str(item.get("person_id") or ""): item
        for item in reference_payload.get("references") or []
    }
    if not references or "" in references or len(references) != len(
        reference_payload.get("references") or []
    ):
        raise ValueError("single face reference person IDs must be non-empty and unique")
    if set(reference_embeddings) != set(references):
        raise ValueError("single face reference embeddings must cover every reference")
    unknown_clusters = sorted(set(cluster_embeddings) - set(clusters))
    if unknown_clusters:
        raise ValueError(f"embedding supplied for unknown cluster: {unknown_clusters}")

    normalized_references = _normalize_embedding_map(reference_embeddings, label="reference")
    normalized_clusters = _normalize_embedding_map(cluster_embeddings, label="cluster")
    dimensions = {
        int(vector.size)
        for vector in [*normalized_references.values(), *normalized_clusters.values()]
    }
    if len(dimensions) > 1:
        raise ValueError("reference and cluster embedding dimensions do not match")

    scores_by_cluster = {
        cluster_id: {
            person_id: float(cluster_embedding @ reference_embedding)
            for person_id, reference_embedding in normalized_references.items()
        }
        for cluster_id, cluster_embedding in normalized_clusters.items()
    }
    rankings = []
    for person_id, reference in references.items():
        candidates = []
        for cluster_id, person_scores in scores_by_cluster.items():
            target_score = person_scores[person_id]
            competitor = max(
                (
                    (score, other_person_id)
                    for other_person_id, score in person_scores.items()
                    if other_person_id != person_id
                ),
                default=(-1.0, None),
                key=lambda item: (item[0], item[1] or ""),
            )
            candidates.append(
                {
                    "cluster_id": cluster_id,
                    "similarity": target_score,
                    "runner_up_person_id": competitor[1],
                    "runner_up_similarity": competitor[0],
                    "person_margin": target_score - competitor[0],
                    **_cluster_review_metadata(clusters[cluster_id]),
                }
            )
        candidates.sort(key=lambda row: (-float(row["similarity"]), str(row["cluster_id"])))
        rankings.append(
            {
                "person_id": person_id,
                "team_id": reference.get("team_id"),
                "candidates": candidates[:top_k],
            }
        )
    output = {
        "schema_version": "agu.face-enrollment-single-reference-ranking.v1",
        "benchmark_disjoint": True,
        "runtime_consumable": False,
        "identity_verified": False,
        "codex_runtime_answer_used": False,
        "reference_mode": "single_verified_portrait_ranking_only",
        "model_id": model_id,
        "source_candidate_manifest_sha256": candidate_sha256,
        "source_reference_manifest_sha256": reference_sha256,
        "top_k": top_k,
        "rankings": rankings,
        "unembedded_cluster_ids": sorted(set(clusters) - set(normalized_clusters)),
        "manifest_sha256": "",
    }
    output["manifest_sha256"] = _canonical_sha256(output)
    return output


def _normalize_embedding_map(
    embeddings: Mapping[str, np.ndarray],
    *,
    label: str,
) -> dict[str, np.ndarray]:
    normalized = {}
    for embedding_id, raw_embedding in embeddings.items():
        embedding = np.asarray(raw_embedding, dtype=np.float32)
        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
            raise ValueError(f"invalid {label} embedding: {embedding_id}")
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0:
            raise ValueError(f"invalid {label} embedding: {embedding_id}")
        normalized[embedding_id] = embedding / norm
    return normalized


def _cluster_review_metadata(cluster: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation_count": int(cluster.get("observation_count") or 0),
        "review_sheet": cluster.get("review_sheet"),
        "source_video_ids": sorted(
            {
                str(sample.get("source_video_id") or "")
                for sample in cluster.get("samples") or []
                if sample.get("source_video_id")
            }
        ),
    }


def _load_cluster_images(
    cluster: Mapping[str, Any],
    *,
    base_dir: Path,
) -> list[np.ndarray]:
    images = []
    cluster_id = str(cluster.get("cluster_id") or "")
    for sample in cluster.get("samples") or []:
        path = Path(str(sample.get("path") or ""))
        if not path.is_absolute():
            path = base_dir / path
        expected_sha256 = str(sample.get("sha256") or "")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected_sha256 or actual_sha256 != expected_sha256:
            raise ValueError(f"face sample hash mismatch for {cluster_id}: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"unable to read face sample for {cluster_id}: {path}")
        images.append(image)
    if not images:
        raise ValueError(f"face candidate cluster has no samples: {cluster_id}")
    return images


def _load_reference_embeddings(
    payload: Mapping[str, Any],
    *,
    base_dir: Path,
    adapter: OpenCvSFaceIdentityAdapter,
) -> dict[str, np.ndarray]:
    embeddings = {}
    for reference in payload.get("references") or []:
        person_id = str(reference.get("person_id") or "")
        path = Path(str(reference.get("path") or ""))
        if not path.is_absolute():
            path = base_dir / path
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != str(reference.get("sha256") or ""):
            raise ValueError(f"single face reference hash mismatch for {person_id}: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"unable to read single face reference for {person_id}: {path}")
        bbox = reference.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
            height, width = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"invalid single face reference bbox for {person_id}")
            image = image[y1:y2, x1:x2]
        faces = adapter.detect_frame_faces(image, minimum_face_size=10)
        if not faces:
            raise ValueError(f"no face detected in single face reference for {person_id}")
        best = max(
            faces,
            key=lambda face: (
                face.bbox[2] * face.bbox[3],
                face.confidence,
            ),
        )
        embeddings[person_id] = best.embedding
    return embeddings


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    adapter = OpenCvSFaceIdentityAdapter(
        str(args.detector_model),
        str(args.recognizer_model),
        score_threshold=args.score_threshold,
    )
    embeddings = {}
    clusters = list(candidate_payload.get("clusters") or [])
    for index, cluster in enumerate(clusters, start=1):
        images = _load_cluster_images(cluster, base_dir=args.candidates.parent)
        result = adapter.embed_player_crops(images)
        if result is not None:
            embeddings[str(cluster["cluster_id"])] = result.embedding
        if index % 100 == 0 or index == len(clusters):
            print(
                json.dumps(
                    {
                        "processed_clusters": index,
                        "embedded_clusters": len(embeddings),
                        "total_clusters": len(clusters),
                    }
                ),
                file=sys.stderr,
            )
    if args.gallery is not None:
        ranked = rank_face_enrollment_candidates(
            candidate_payload,
            load_face_gallery(args.gallery),
            embeddings,
            top_k=args.top_k,
        )
    else:
        reference_payload = json.loads(
            args.single_reference_manifest.read_text(encoding="utf-8")
        )
        reference_embeddings = _load_reference_embeddings(
            reference_payload,
            base_dir=args.single_reference_manifest.parent,
            adapter=adapter,
        )
        ranked = rank_face_enrollment_reference_embeddings(
            candidate_payload,
            reference_payload,
            reference_embeddings,
            embeddings,
            top_k=args.top_k,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ranked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_sha256": ranked["manifest_sha256"],
                "embedded_cluster_count": len(embeddings),
                "unembedded_cluster_count": len(ranked["unembedded_cluster_ids"]),
                "person_count": len(ranked["rankings"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
