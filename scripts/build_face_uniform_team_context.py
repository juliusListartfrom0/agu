#!/usr/bin/env python3
"""Fit and apply a face-anchored uniform team filter on enrollment videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_uniform_team import (  # noqa: E402
    FEATURE_NAMES,
    aggregate_cluster_team,
    extract_face_anchored_uniform_features,
    fit_uniform_team_model,
    predict_uniform_team_probability,
    select_uniform_team_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotated-face-list", type=Path, required=True)
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="Bind a candidate source video ID to a local video (SOURCE_ID=PATH).",
    )
    parser.add_argument("--class-id", action="append", required=True)
    parser.add_argument("--output-context", type=Path, required=True)
    parser.add_argument("--team-id")
    parser.add_argument("--output-candidates", type=Path)
    parser.add_argument("--minimum-cross-validated-accuracy", type=float, default=0.85)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--minimum-probability", type=float, default=0.65)
    parser.add_argument("--minimum-consensus", type=float, default=0.75)
    return parser.parse_args()


def _parse_video_bindings(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        source_id, raw_path = source_id.strip(), raw_path.strip()
        if separator != "=" or not source_id or not raw_path or source_id in result:
            raise ValueError(f"invalid or duplicate video binding: {value}")
        result[source_id] = Path(raw_path)
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _extract_sample_features(
    frame: np.ndarray,
    face_bbox: list[float] | tuple[float, ...],
) -> np.ndarray | None:
    """Return uniform features, or None when the face has no torso pixels below it."""

    try:
        return extract_face_anchored_uniform_features(frame, face_bbox)
    except ValueError as error:
        if str(error) == "face-anchored uniform crop is empty":
            return None
        raise


def _extract_all_features(
    candidate_payload: dict[str, Any],
    video_bindings: dict[str, Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    sources = {
        str(source.get("source_video_id") or ""): source
        for source in candidate_payload.get("sources") or []
    }
    if set(video_bindings) != set(sources):
        raise ValueError("video bindings must exactly cover candidate sources")
    samples_by_source_frame: dict[
        str, dict[int, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    paths: set[str] = set()
    for cluster in candidate_payload.get("clusters") or []:
        for sample in cluster.get("samples") or []:
            path = str(sample.get("path") or "")
            source_id = str(sample.get("source_video_id") or "")
            frame = sample.get("frame")
            if not path or path in paths or source_id not in sources or frame is None:
                raise ValueError("candidate face samples must have unique bound paths")
            paths.add(path)
            samples_by_source_frame[source_id][int(frame)].append(sample)

    features_by_path = {}
    source_records = []
    for source_id, video_path in video_bindings.items():
        source = sources[source_id]
        actual_sha256 = _sha256_file(video_path)
        if (
            video_path.name != str(source.get("filename") or "")
            or actual_sha256 != str(source.get("sha256") or "")
        ):
            raise ValueError(f"video binding does not match candidate source: {source_id}")
        targets = samples_by_source_frame[source_id]
        maximum_frame = max(targets, default=-1)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open enrollment video: {video_path}")
        decoded_frames = 0
        extracted_samples = 0
        skipped_empty_uniform_crops = 0
        try:
            frame_index = 0
            while frame_index <= maximum_frame:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"video ended before requested enrollment frame: {source_id}:{frame_index}"
                    )
                decoded_frames += 1
                for sample in targets.get(frame_index, ()):
                    features = _extract_sample_features(frame, sample["bbox"])
                    if features is None:
                        skipped_empty_uniform_crops += 1
                        continue
                    features_by_path[str(sample["path"])] = features
                    extracted_samples += 1
                frame_index += 1
                if frame_index % 25000 == 0:
                    print(
                        json.dumps(
                            {
                                "source_video_id": source_id,
                                "decoded_frames": frame_index,
                                "maximum_target_frame": maximum_frame,
                                "extracted_samples": extracted_samples,
                                "skipped_empty_uniform_crops": (
                                    skipped_empty_uniform_crops
                                ),
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            capture.release()
        source_records.append(
            {
                "source_video_id": source_id,
                "filename": video_path.name,
                "sha256": actual_sha256,
                "decoded_frames": decoded_frames,
                "maximum_target_frame": maximum_frame,
                "target_sample_count": sum(
                    len(frame_samples) for frame_samples in targets.values()
                ),
                "extracted_samples": extracted_samples,
                "skipped_empty_uniform_crops": skipped_empty_uniform_crops,
            }
        )
    if not features_by_path or not set(features_by_path).issubset(paths):
        raise RuntimeError("uniform feature extraction produced no valid candidate samples")
    return features_by_path, source_records


def _training_arrays(
    annotated_payload: dict[str, Any],
    *,
    features_by_path: dict[str, np.ndarray],
    class_ids: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if annotated_payload.get("schema_version") != "agu.annotated-face-list.v1":
        raise ValueError("unsupported annotated face list schema")
    if annotated_payload.get("benchmark_disjoint") is not True:
        raise ValueError("uniform training identities must be benchmark-disjoint")
    features = []
    labels = []
    groups = []
    consumed_paths: set[str] = set()
    class_index = {team_id: index for index, team_id in enumerate(class_ids)}
    for person in annotated_payload.get("persons") or []:
        person_id = str(person.get("person_id") or "")
        team_id = str(person.get("team_id") or "")
        if not person_id or team_id not in class_index:
            raise ValueError("annotated identities must bind one configured team")
        for raw_sample in person.get("samples") or []:
            sample = {"path": raw_sample} if isinstance(raw_sample, str) else raw_sample
            path = str(sample.get("path") or "")
            if path in consumed_paths:
                continue
            if path not in features_by_path:
                continue
            consumed_paths.add(path)
            features.append(features_by_path[path])
            labels.append(class_index[team_id])
            groups.append(person_id)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
    )


def main() -> int:
    args = parse_args()
    if len(args.class_id) != 2 or len(set(args.class_id)) != 2:
        raise ValueError("exactly two unique --class-id values are required")
    if (args.team_id is None) != (args.output_candidates is None):
        raise ValueError("--team-id and --output-candidates must be provided together")
    candidates_bytes = args.candidates.read_bytes()
    candidates = json.loads(candidates_bytes)
    expected_candidate_sha256 = _canonical_sha256(
        {**candidates, "manifest_sha256": ""}
    )
    if candidates.get("manifest_sha256") != expected_candidate_sha256:
        raise ValueError("face enrollment candidate manifest hash mismatch")
    annotated_bytes = args.annotated_face_list.read_bytes()
    annotated = json.loads(annotated_bytes)
    class_ids = (str(args.class_id[0]), str(args.class_id[1]))
    features_by_path, sources = _extract_all_features(
        candidates,
        _parse_video_bindings(args.video),
    )
    training_features, training_labels, training_groups = _training_arrays(
        annotated,
        features_by_path=features_by_path,
        class_ids=class_ids,
    )
    model, audit = fit_uniform_team_model(
        training_features,
        training_labels,
        training_groups,
        class_ids=class_ids,
        minimum_cross_validated_accuracy=args.minimum_cross_validated_accuracy,
    )
    annotated_sample_count = len(
        {
            str(sample.get("path") or "")
            if isinstance(sample, dict)
            else str(sample)
            for person in annotated.get("persons") or []
            for sample in person.get("samples") or []
        }
    )
    audit["available_annotated_sample_count"] = int(training_features.shape[0])
    audit["skipped_annotated_sample_count"] = (
        annotated_sample_count - int(training_features.shape[0])
    )
    cluster_rows = []
    for cluster in candidates.get("clusters") or []:
        sample_rows = []
        probabilities = []
        for sample in cluster.get("samples") or []:
            features = features_by_path.get(str(sample["path"]))
            if features is None:
                continue
            probability = predict_uniform_team_probability(
                model,
                features,
            )
            probabilities.append(probability)
            sample_rows.append(
                {
                    "path": str(sample["path"]),
                    "source_video_id": str(sample["source_video_id"]),
                    "frame": int(sample["frame"]),
                    "class_1_probability": probability,
                    "uniform_features": {
                        name: float(value)
                        for name, value in zip(FEATURE_NAMES, features, strict=True)
                    },
                }
            )
        team_id, consensus, mean_probability = aggregate_cluster_team(
            probabilities,
            class_ids=class_ids,
            minimum_observations=args.minimum_observations,
            minimum_probability=args.minimum_probability,
            minimum_consensus=args.minimum_consensus,
        )
        cluster_rows.append(
            {
                "cluster_id": str(cluster.get("cluster_id") or ""),
                "team_id": team_id,
                "consensus": consensus,
                "mean_class_1_probability": mean_probability,
                "sample_count": len(cluster.get("samples") or []),
                "feature_sample_count": len(sample_rows),
                "samples": sample_rows,
            }
        )
    context = {
        "schema_version": "agu.face-uniform-team-context.v1",
        "benchmark_disjoint": True,
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "producer": "agu_face_anchored_uniform_logistic_v1",
        "source_candidate_manifest_sha256": candidates["manifest_sha256"],
        "source_annotated_face_list_sha256": hashlib.sha256(
            annotated_bytes
        ).hexdigest(),
        "sources": sources,
        "model": model,
        "audit": audit,
        "thresholds": {
            "minimum_observations": args.minimum_observations,
            "minimum_probability": args.minimum_probability,
            "minimum_consensus": args.minimum_consensus,
        },
        "clusters": cluster_rows,
        "manifest_sha256": "",
    }
    context["manifest_sha256"] = _canonical_sha256(context)
    _write_json(args.output_context, context)
    subset = None
    if args.team_id is not None:
        subset = select_uniform_team_candidates(
            candidates,
            context,
            team_id=args.team_id,
        )
        _write_json(args.output_candidates, subset)
    print(
        json.dumps(
            {
                "context_manifest_sha256": context["manifest_sha256"],
                "cross_validated_accuracy": audit["cross_validated_accuracy"],
                "assigned_cluster_count": sum(
                    row["team_id"] is not None for row in cluster_rows
                ),
                "selected_cluster_count": (
                    None if subset is None else len(subset["clusters"])
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
