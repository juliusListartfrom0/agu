"""Deterministic BARD validation subset for broadcast visual-state training."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import numpy as np

BARD_VISUAL_STATE_SCHEMA = "agu.bard-visual-state-subset.v1"
BARD_VISUAL_STATE_EMBEDDING_SCHEMA = "agu.bard-visual-state-embeddings.v1"

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MEDIA_PATH = re.compile(
    r"validation/(2024|2025)/multi/"
    r"([a-z0-9-]+-\d{10})_(\d+)\.mp4"
)
_FIELD_GOAL_ACTIONS = {"2PT Shot", "3PT Shot"}
_BARD_CLIP_SAMPLE_FRACTIONS = (0.05, 0.275, 0.5, 0.725, 0.95)


def bard_clip_frame_indexes(*, frame_count: int) -> tuple[int, ...]:
    """Return five deterministic, ordered clip-wide sampling positions."""

    if isinstance(frame_count, bool) or frame_count < len(
        _BARD_CLIP_SAMPLE_FRACTIONS
    ):
        raise ValueError("BARD clips require at least five frames")
    indexes: list[int] = []
    last_frame = frame_count - 1
    for position, fraction in enumerate(_BARD_CLIP_SAMPLE_FRACTIONS):
        remaining = len(_BARD_CLIP_SAMPLE_FRACTIONS) - position - 1
        raw = int(last_frame * fraction)
        lower = indexes[-1] + 1 if indexes else 0
        indexes.append(min(max(raw, lower), last_frame - remaining))
    return tuple(indexes)


def build_bard_visual_state_subset_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    git_blob_sha1_by_path: Mapping[str, str],
    source_revision: str,
    benchmark_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Pair every pure FT clip with the nearest pure shot from the same game."""

    if _SHA1.fullmatch(source_revision) is None:
        raise ValueError("invalid BARD source revision")
    benchmark_hashes = _require_sha256s(
        benchmark_sha256s,
        field="benchmark",
    )
    free_throws: dict[str, list[tuple[int, str]]] = defaultdict(list)
    field_goals: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_paths: set[str] = set()
    for row in rows:
        path, game, event = _parse_media_path(row.get("path"))
        if path in seen_paths:
            raise ValueError("duplicate BARD validation path")
        seen_paths.add(path)
        blob = str(git_blob_sha1_by_path.get(path) or "")
        if _SHA1.fullmatch(blob) is None:
            raise ValueError("BARD validation path lacks a Git blob SHA-1")
        actions = row.get("actions")
        if (
            not isinstance(actions, list)
            or any(not isinstance(value, str) for value in actions)
        ):
            raise ValueError("invalid BARD action list")
        action_set = set(actions)
        is_free_throw = "Free Throw" in action_set
        is_field_goal = bool(action_set & _FIELD_GOAL_ACTIONS)
        if is_free_throw == is_field_goal:
            continue
        target = free_throws if is_free_throw else field_goals
        target[game].append((event, path))

    pairs = []
    selected_examples = []
    used_field_goals: set[str] = set()
    all_free_throws = sorted(
        (
            game,
            event,
            path,
        )
        for game, candidates in free_throws.items()
        for event, path in candidates
    )
    if not all_free_throws:
        raise ValueError("BARD validation subset has no pure free throws")
    excluded_unpaired = 0
    for game, free_throw_event, free_throw_path in all_free_throws:
        available = [
            (abs(event - free_throw_event), event, path)
            for event, path in field_goals.get(game, [])
            if path not in used_field_goals
        ]
        if not available:
            excluded_unpaired += 1
            continue
        _distance, field_goal_event, field_goal_path = min(available)
        used_field_goals.add(field_goal_path)
        pair_id = f"bard-visual-state-{len(pairs) + 1:03d}"
        pairs.append(
            {
                "pair_id": pair_id,
                "game_key": game,
                "free_throw_path": free_throw_path,
                "field_goal_path": field_goal_path,
                "event_distance": abs(field_goal_event - free_throw_event),
            }
        )
        for state, path in (
            ("free_throw", free_throw_path),
            ("field_goal", field_goal_path),
        ):
            selected_examples.append(
                {
                    "pair_id": pair_id,
                    "state": state,
                    "path": path,
                    "git_blob_sha1": git_blob_sha1_by_path[path],
                }
            )

    artifact: dict[str, Any] = {
        "schema_version": BARD_VISUAL_STATE_SCHEMA,
        "purpose": "offline_broadcast_visual_state_pretraining",
        "runtime_consumable": False,
        "truth_used_for_training_only": True,
        "codex_runtime_answer_used": False,
        "source_repository": "https://github.com/GabrieleGiudic/BARD",
        "source_revision": source_revision,
        "license": "CC-BY-4.0",
        "benchmark_sha256s": list(benchmark_hashes),
        "selection_protocol": (
            "all_pure_free_throw_validation_clips_paired_with_nearest_"
            "unused_pure_2pt_or_3pt_clip_from_same_game"
        ),
        "pairs": pairs,
        "examples": selected_examples,
        "summary": {
            "candidate_free_throw": len(all_free_throws),
            "excluded_unpaired_free_throw": excluded_unpaired,
            "pairs": len(pairs),
            "free_throw": len(pairs),
            "field_goal": len(pairs),
            "examples": len(selected_examples),
        },
    }
    _validate_bard_visual_state_subset_manifest(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_bard_visual_state_subset_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_bard_visual_state_subset_manifest(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("BARD visual-state subset hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_bard_visual_state_embedding_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = BARD_VISUAL_STATE_EMBEDDING_SCHEMA
    artifact["runtime_consumable"] = False
    artifact["truth_used_for_training_only"] = True
    artifact["codex_runtime_answer_used"] = False
    artifact["embedding_dimension"] = 384
    artifact.pop("artifact_sha256", None)
    _validate_bard_visual_state_embedding_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_bard_visual_state_embedding_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_bard_visual_state_embedding_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("BARD visual-state embedding artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _validate_bard_visual_state_subset_manifest(
    artifact: Mapping[str, Any],
) -> None:
    if artifact.get("schema_version") != BARD_VISUAL_STATE_SCHEMA:
        raise ValueError("unsupported BARD visual-state subset schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("truth_used_for_training_only") is not True
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("license") != "CC-BY-4.0"
        or _SHA1.fullmatch(str(artifact.get("source_revision") or "")) is None
    ):
        raise ValueError("invalid BARD visual-state subset provenance")
    _require_sha256s(artifact.get("benchmark_sha256s"), field="benchmark")
    pairs = artifact.get("pairs")
    examples = artifact.get("examples")
    summary = artifact.get("summary")
    if (
        not isinstance(pairs, list)
        or not pairs
        or not isinstance(examples, list)
        or not isinstance(summary, Mapping)
    ):
        raise ValueError("invalid BARD visual-state subset content")
    pair_ids: set[str] = set()
    pair_paths: set[str] = set()
    for row in pairs:
        if not isinstance(row, Mapping):
            raise ValueError("invalid BARD visual-state pair")
        pair_id = str(row.get("pair_id") or "")
        free_throw_path, free_game, _free_event = _parse_media_path(
            row.get("free_throw_path")
        )
        field_goal_path, field_game, _field_event = _parse_media_path(
            row.get("field_goal_path")
        )
        if (
            not pair_id
            or pair_id in pair_ids
            or free_game != field_game
            or row.get("game_key") != free_game
            or free_throw_path == field_goal_path
        ):
            raise ValueError("invalid or duplicate BARD visual-state pair")
        pair_ids.add(pair_id)
        pair_paths.update((free_throw_path, field_goal_path))
    example_paths: set[str] = set()
    state_counts = {"free_throw": 0, "field_goal": 0}
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("invalid BARD visual-state example")
        path, _game, _event = _parse_media_path(row.get("path"))
        state = str(row.get("state") or "")
        if (
            row.get("pair_id") not in pair_ids
            or state not in state_counts
            or path in example_paths
            or _SHA1.fullmatch(str(row.get("git_blob_sha1") or "")) is None
        ):
            raise ValueError("invalid or duplicate BARD visual-state example")
        example_paths.add(path)
        state_counts[state] += 1
    if example_paths != pair_paths:
        raise ValueError("BARD visual-state examples do not cover every pair")
    expected_summary = {
        "candidate_free_throw": int(summary.get("candidate_free_throw", -1)),
        "excluded_unpaired_free_throw": int(
            summary.get("excluded_unpaired_free_throw", -1)
        ),
        "pairs": len(pairs),
        "free_throw": state_counts["free_throw"],
        "field_goal": state_counts["field_goal"],
        "examples": len(examples),
    }
    if (
        expected_summary["candidate_free_throw"]
        != len(pairs) + expected_summary["excluded_unpaired_free_throw"]
        or expected_summary["excluded_unpaired_free_throw"] < 0
        or dict(summary) != expected_summary
        or not (
        len(pairs) == state_counts["free_throw"] == state_counts["field_goal"]
        )
    ):
        raise ValueError("BARD visual-state subset summary mismatch")


def _validate_bard_visual_state_embedding_artifact(
    artifact: Mapping[str, Any],
) -> None:
    if artifact.get("schema_version") != BARD_VISUAL_STATE_EMBEDDING_SCHEMA:
        raise ValueError("unsupported BARD visual-state embedding schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("truth_used_for_training_only") is not True
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("backbone") != "facebook/dinov2-small"
        or int(artifact.get("embedding_dimension", 0)) != 384
    ):
        raise ValueError("invalid BARD visual-state embedding provenance")
    _require_sha256_value(
        artifact.get("acquisition_artifact_sha256"),
        field="acquisition artifact",
    )
    _require_sha256_value(
        artifact.get("backbone_sha256"),
        field="backbone",
    )
    blind = set(
        _require_sha256s(
            artifact.get("sealed_blind_video_sha256s"),
            field="sealed blind video",
        )
    )
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("BARD visual-state embeddings require examples")
    seen_sources: set[str] = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("invalid BARD visual-state embedding example")
        source = _require_sha256_value(
            row.get("source_video_sha256"),
            field="source video",
        )
        frames = row.get("frame_indexes")
        embeddings = row.get("embeddings")
        if (
            source in blind
            or source in seen_sources
            or not str(row.get("pair_id") or "")
            or row.get("state") not in {"free_throw", "field_goal"}
            or not isinstance(frames, list)
            or len(frames) != 5
            or any(isinstance(value, bool) or int(value) < 0 for value in frames)
            or sorted({int(value) for value in frames}) != frames
            or not isinstance(embeddings, list)
            or len(embeddings) != 5
        ):
            raise ValueError("invalid BARD visual-state embedding example")
        seen_sources.add(source)
        for embedding in embeddings:
            if (
                not isinstance(embedding, list)
                or len(embedding) != 384
                or not all(np.isfinite(float(value)) for value in embedding)
            ):
                raise ValueError("invalid BARD visual-state embedding values")


def _parse_media_path(value: object) -> tuple[str, str, int]:
    text = str(value or "")
    if PurePosixPath(text).is_absolute() or ".." in PurePosixPath(text).parts:
        raise ValueError("invalid BARD validation media path")
    match = _MEDIA_PATH.fullmatch(text)
    if match is None:
        raise ValueError("invalid BARD validation media path")
    _year, game, event = match.groups()
    return text, game, int(event)


def _require_sha256s(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{field} SHA-256 values are required")
    hashes = tuple(str(item) for item in value)
    if any(_SHA256.fullmatch(item) is None for item in hashes):
        raise ValueError(f"invalid {field} SHA-256")
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"duplicate {field} SHA-256")
    return hashes


def _require_sha256_value(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
