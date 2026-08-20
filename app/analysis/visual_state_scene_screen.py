"""Nested game-held screen for cut-aware visual-state representations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from app.analysis.pbp_visual_state_frames import WIDE_ANCHOR_OFFSETS_SECONDS
from app.analysis.replay_transition import verify_replay_transition_artifact
from app.analysis.visual_state_temporal_screen import (
    VISUAL_STATE_LABELS,
    _fit_estimator,
    _metrics,
    _select_configuration,
    _summary_features,
)

VISUAL_STATE_SCENE_SCREEN_SCHEMA = "agu.visual-state-scene-screen.v1"
SCENE_REPRESENTATIONS = (
    "wide_summary",
    "anchor_shot_summary",
    "neighbor_shot_means",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_visual_state_scene_examples(
    *,
    temporal_examples: Sequence[Mapping[str, Any]],
    transition_artifacts: Sequence[Mapping[str, Any]],
    expected_event_keys: set[tuple[str, str, str]],
    sealed_blind_video_sha256s: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Exact-join cut positions while deliberately ignoring replay semantics."""

    if not temporal_examples or not transition_artifacts or not expected_event_keys:
        raise ValueError("scene screening requires examples, transitions, and event keys")
    sealed = {
        _require_sha256(value, field="sealed blind video")
        for value in sealed_blind_video_sha256s
    }
    indexed: dict[tuple[str, str, str], list[float]] = {}
    hashes = []
    for raw_artifact in sorted(
        transition_artifacts,
        key=lambda row: str(row.get("raw_video_sha256") or ""),
    ):
        artifact = verify_replay_transition_artifact(raw_artifact)
        if artifact.get("codex_runtime_answer_used") is not False:
            raise ValueError(
                "scene screening rejects Codex runtime answer transition artifact"
            )
        source = _require_sha256(
            artifact.get("raw_video_sha256"),
            field="transition source video",
        )
        bundle = _require_sha256(
            artifact.get("candidate_bundle_sha256"),
            field="transition candidate bundle",
        )
        if source in sealed:
            raise ValueError("scene screening rejects sealed blind source")
        hashes.append(
            _require_sha256(
                artifact.get("artifact_sha256"),
                field="transition artifact",
            )
        )
        for event in artifact["events"]:
            key = (source, bundle, str(event["event_id"]))
            if key in indexed:
                raise ValueError("duplicate scene-transition event")
            indexed[key] = [
                float(value) for value in event["transition_offsets_seconds"]
            ]
    if set(indexed) != expected_event_keys:
        raise ValueError("scene-transition artifacts require exact review-plan coverage")
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate scene-transition artifact")

    rows = []
    seen = set()
    for raw_row in temporal_examples:
        row = dict(raw_row)
        key = _event_key(row)
        if key in seen or key not in indexed:
            raise ValueError("temporal scene example is duplicate or unbound")
        if key[0] in sealed:
            raise ValueError("scene screening rejects sealed blind source")
        seen.add(key)
        row["transition_offsets_seconds"] = indexed[key]
        rows.append(row)
    if not rows:
        raise ValueError("scene screening requires resolved temporal examples")
    return rows, hashes


def screen_visual_state_scene_examples(
    *,
    examples: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    label_correction_artifact_sha256s: Sequence[str],
    configurations: Sequence[Mapping[str, Any]] | None = None,
    acceptance_threshold: float = 0.85,
) -> dict[str, Any]:
    """Select cut-aware representations only inside each outer training split."""

    if not 0.5 <= float(acceptance_threshold) <= 1.0:
        raise ValueError("acceptance threshold must be between 0.5 and 1.0")
    backbone = str(provenance.get("backbone") or "")
    backbone_sha256 = _require_sha256(
        provenance.get("backbone_sha256"),
        field="backbone",
    )
    embedding_hashes = _hash_list(
        provenance.get("embedding_artifact_sha256s"),
        field="embedding artifact",
    )
    transition_hashes = _hash_list(
        provenance.get("transition_artifact_sha256s"),
        field="transition artifact",
    )
    correction_hashes = _hash_list(
        label_correction_artifact_sha256s,
        field="label correction artifact",
    )
    if not backbone or int(provenance.get("embedding_dimension", 0)) < 1:
        raise ValueError("scene-screen provenance is incomplete")

    labels, groups, event_ids, representations = _scene_arrays(examples)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 4:
        raise ValueError("scene screening requires at least four target groups")
    for group in unique_groups:
        if set(labels[groups == group].tolist()) != {0, 1}:
            raise ValueError("every target group must contain both visual states")
    configs = _normalize_configurations(
        configurations or _default_configurations(representations)
    )

    outer_predictions = np.zeros(len(labels), dtype=np.int64)
    outer_probabilities = np.zeros(len(labels), dtype=np.float64)
    outer_folds = []
    for held_group in unique_groups:
        held_mask = groups == held_group
        development_groups = [
            group for group in unique_groups if group != held_group
        ]
        selection = _select_configuration(
            configs=configs,
            representations=representations,
            labels=labels,
            groups=groups,
            development_groups=development_groups,
        )
        config = selection["configuration"]
        features = representations[config["representation"]]
        development_mask = np.isin(groups, development_groups)
        estimator = _fit_estimator(
            features[development_mask],
            labels[development_mask],
            config=config,
        )
        probabilities = estimator.predict_proba(features[held_mask])[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int64)
        outer_predictions[held_mask] = predictions
        outer_probabilities[held_mask] = probabilities
        outer_folds.append(
            {
                "held_target_group": held_group,
                "training_target_groups": development_groups,
                "selection_target_groups": development_groups,
                "selected_configuration": config,
                "inner_selection": selection["metrics"],
                "held_metrics": _metrics(
                    labels[held_mask],
                    predictions,
                    probabilities,
                ),
                "held_example_count": int(held_mask.sum()),
            }
        )

    pooled = _metrics(labels, outer_predictions, outer_probabilities)
    per_game = {
        group: _metrics(
            labels[groups == group],
            outer_predictions[groups == group],
            outer_probabilities[groups == group],
        )
        for group in unique_groups
    }
    worst_game = min(row["balanced_accuracy"] for row in per_game.values())
    metrics = {
        **pooled,
        "worst_game_balanced_accuracy": worst_game,
        "per_game": per_game,
    }
    accepted = (
        worst_game >= acceptance_threshold and pooled["f1"] >= acceptance_threshold
    )
    artifact: dict[str, Any] = {
        "schema_version": VISUAL_STATE_SCENE_SCREEN_SCHEMA,
        "purpose": "training_only_cut_aware_visual_state_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "transition_semantics_used": False,
        "backbone": backbone,
        "backbone_sha256": backbone_sha256,
        "embedding_dimension": int(provenance["embedding_dimension"]),
        "embedding_artifact_sha256s": embedding_hashes,
        "transition_artifact_sha256s": transition_hashes,
        "label_correction_artifact_sha256s": correction_hashes,
        "target_example_count": int(len(labels)),
        "target_group_count": int(len(unique_groups)),
        "target_event_ids_sha256": _canonical_sha256(event_ids),
        "protocol": {
            "outer_split": "leave_one_target_video_out",
            "inner_selection": "leave_one_remaining_target_video_out",
            "decision_threshold": 0.5,
            "transition_threshold": 0.5,
            "transition_cluster_max_gap_seconds": 0.1,
            "replay_or_highlight_semantics_used": False,
            "random_seed": 0,
        },
        "configurations": configs,
        "outer_folds": outer_folds,
        "metrics": metrics,
        "acceptance_gate": {
            "minimum_game_balanced_accuracy": float(acceptance_threshold),
            "minimum_pooled_f1": float(acceptance_threshold),
        },
        "accepted": bool(accepted),
        "promotion_decision": (
            "eligible_for_next_training_stage"
            if accepted
            else "rejected_below_cross_game_gate"
        ),
    }
    _validate_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_visual_state_scene_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("visual-state scene screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _scene_arrays(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], dict[str, np.ndarray]]:
    labels = []
    groups = []
    event_ids = []
    representation_rows = {name: [] for name in SCENE_REPRESENTATIONS}
    seen = set()
    expected_dimension: int | None = None
    wide_offsets = np.asarray(WIDE_ANCHOR_OFFSETS_SECONDS, dtype=np.float64)
    for row in examples:
        key = _event_key(row)
        label = str(row.get("label") or "")
        if key in seen or label not in VISUAL_STATE_LABELS:
            raise ValueError("invalid or duplicate scene-screen example")
        seen.add(key)
        offsets = [float(value) for value in row.get("offsets_seconds") or ()]
        embeddings = np.asarray(row.get("embeddings"), dtype=np.float64)
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(offsets)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("invalid scene-screen embeddings")
        if expected_dimension is None:
            expected_dimension = int(embeddings.shape[1])
        elif embeddings.shape[1] != expected_dimension:
            raise ValueError("scene-screen embedding dimensions differ")
        by_offset = {offset: embeddings[index] for index, offset in enumerate(offsets)}
        if not set(WIDE_ANCHOR_OFFSETS_SECONDS).issubset(by_offset):
            raise ValueError("scene screening requires centered wide embeddings")
        sequence = np.stack([by_offset[offset] for offset in WIDE_ANCHOR_OFFSETS_SECONDS])
        cuts = _cluster_transition_offsets(row.get("transition_offsets_seconds"))
        geometry = _transition_geometry(cuts)
        segment_ids = np.searchsorted(cuts, wide_offsets, side="right")
        anchor_segment = int(np.searchsorted(cuts, 0.0, side="right"))
        relative_segments = segment_ids - anchor_segment
        anchor_rows = sequence[relative_segments == 0]
        if len(anchor_rows) == 0:
            raise ValueError("anchor shot must contain the zero-offset embedding")
        anchor_mean = anchor_rows.mean(axis=0)

        representation_rows["wide_summary"].append(_summary_features(sequence))
        representation_rows["anchor_shot_summary"].append(
            np.concatenate(
                (
                    _summary_features(anchor_rows),
                    geometry,
                    [len(anchor_rows) / len(sequence)],
                )
            )
        )
        neighbor_means = []
        availability = []
        for relative_segment in (-1, 0, 1):
            selected = sequence[relative_segments == relative_segment]
            availability.append(float(len(selected) > 0))
            neighbor_means.append(
                selected.mean(axis=0) if len(selected) else anchor_mean
            )
        representation_rows["neighbor_shot_means"].append(
            np.concatenate((*neighbor_means, geometry, availability))
        )
        labels.append(1 if label == "free_throw" else 0)
        groups.append(key[0])
        event_ids.append({"source_video_sha256": key[0], "event_id": key[2]})
    if not labels:
        raise ValueError("scene screening requires examples")
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
        event_ids,
        {
            name: np.asarray(values, dtype=np.float64)
            for name, values in representation_rows.items()
        },
    )


def _cluster_transition_offsets(value: object) -> np.ndarray:
    if not isinstance(value, (list, tuple)):
        raise ValueError("transition offsets must be a list")
    offsets = sorted(float(item) for item in value)
    if any(not math.isfinite(item) for item in offsets):
        raise ValueError("transition offsets must be finite")
    clustered: list[float] = []
    for offset in offsets:
        if not clustered or offset - clustered[-1] > 0.1:
            clustered.append(offset)
        else:
            clustered[-1] = (clustered[-1] + offset) / 2.0
    return np.asarray(clustered, dtype=np.float64)


def _transition_geometry(cuts: np.ndarray) -> np.ndarray:
    before = cuts[cuts < 0.0]
    after = cuts[cuts >= 0.0]
    distance_before = float(-before[-1]) if len(before) else 10.0
    distance_after = float(after[0]) if len(after) else 10.0
    bins = ((-8.0, -4.0), (-4.0, -1.0), (-1.0, 1.0), (1.0, 4.0), (4.0, 8.0))
    return np.asarray(
        [
            min(distance_before, 10.0) / 10.0,
            min(distance_after, 10.0) / 10.0,
            min(distance_before + distance_after, 20.0) / 20.0,
            min(len(cuts), 10) / 10.0,
            *[
                float(np.sum((cuts >= lower) & (cuts < upper))) / 5.0
                for lower, upper in bins
            ],
            float(np.any(np.abs(cuts) <= 0.5)),
            float(np.any(np.abs(cuts) <= 1.0)),
            float(np.any(np.abs(cuts) <= 2.0)),
        ],
        dtype=np.float64,
    )


def _normalize_configurations(
    configurations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for row in configurations:
        representation = str(row.get("representation") or "")
        c_value = float(row.get("c", 0.0))
        components = row.get("pca_components")
        if (
            representation not in SCENE_REPRESENTATIONS
            or not math.isfinite(c_value)
            or c_value <= 0
            or isinstance(components, bool)
            or int(components or 0) < 1
        ):
            raise ValueError("invalid scene-screen configuration")
        normalized.append(
            {
                "representation": representation,
                "c": c_value,
                "pca_components": int(components),
            }
        )
    if not normalized:
        raise ValueError("scene screening requires configurations")
    if len({_canonical_sha256(row) for row in normalized}) != len(normalized):
        raise ValueError("duplicate scene-screen configuration")
    return normalized


def _default_configurations(
    representations: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    return [
        {
            "representation": representation,
            "c": c_value,
            "pca_components": components,
        }
        for representation in representations
        for c_value in (0.01, 0.1, 1.0)
        for components in (4, 8, 16, 32)
    ]


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != VISUAL_STATE_SCENE_SCREEN_SCHEMA
        or artifact.get("purpose") != "training_only_cut_aware_visual_state_screen"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("transition_semantics_used") is not False
    ):
        raise ValueError("invalid visual-state scene screen")
    _require_sha256(artifact.get("backbone_sha256"), field="backbone")
    _hash_list(artifact.get("embedding_artifact_sha256s"), field="embedding artifact")
    _hash_list(
        artifact.get("transition_artifact_sha256s"),
        field="transition artifact",
    )
    _hash_list(
        artifact.get("label_correction_artifact_sha256s"),
        field="label correction artifact",
    )
    groups = int(artifact.get("target_group_count", 0))
    folds = artifact.get("outer_folds")
    if groups < 4 or not isinstance(folds, list) or len(folds) != groups:
        raise ValueError("invalid scene-screen coverage")
    for fold in folds:
        held = fold.get("held_target_group")
        if (
            held in fold.get("training_target_groups", ())
            or held in fold.get("selection_target_groups", ())
        ):
            raise ValueError("held game leaked into scene-screen training")
    metrics = artifact.get("metrics")
    gate = artifact.get("acceptance_gate")
    if not isinstance(metrics, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("scene-screen metrics are missing")
    expected = (
        float(metrics["worst_game_balanced_accuracy"])
        >= float(gate["minimum_game_balanced_accuracy"])
        and float(metrics["f1"]) >= float(gate["minimum_pooled_f1"])
    )
    if bool(artifact.get("accepted")) != expected:
        raise ValueError("scene-screen acceptance is inconsistent")


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = _require_sha256(row.get("source_video_sha256"), field="source video")
    bundle = _require_sha256(
        row.get("candidate_bundle_sha256"),
        field="candidate bundle",
    )
    event_id = str(row.get("event_id") or "")
    if not event_id:
        raise ValueError("event id is required")
    return source, bundle, event_id


def _hash_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} hashes are required")
    hashes = [_require_sha256(item, field=field) for item in value]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"duplicate {field} hash")
    return hashes


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
