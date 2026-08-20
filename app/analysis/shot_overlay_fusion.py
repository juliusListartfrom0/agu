"""Game-held screening for label-free shot-clock and score-overlay evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.analysis.shot_broadcast_state import (
    verify_broadcast_state_evidence_artifact,
)
from app.analysis.shot_overlay_state import (
    verify_overlay_state_evidence_artifact,
)
from app.analysis.shot_reason_fusion import (
    _base_matrix,
    _example_key,
    _gate_score,
    _reduce_base,
    _variant_result,
)
from app.analysis.shot_validity_scene_state import verify_scene_embedding_artifact
from app.analysis.shot_validity_video_backbone import verify_video_embedding_artifact

VARIANT_NAMES = (
    "base",
    "base+broadcast_raw",
    "base+overlay_raw",
    "base+broadcast+overlay_raw",
)


def screen_shot_overlay_fusion(
    *,
    scene_artifact: Mapping[str, Any],
    video_artifacts: Sequence[Mapping[str, Any]],
    broadcast_state_artifact: Mapping[str, Any],
    overlay_state_artifact: Mapping[str, Any],
    pca_components: int = 16,
    regularization_c: float = 0.01,
) -> dict[str, Any]:
    """Compare four fixed variants under strict outer-game isolation."""

    if pca_components < 1 or regularization_c <= 0:
        raise ValueError("overlay fusion PCA and regularization must be positive")
    scene = verify_scene_embedding_artifact(scene_artifact)
    videos = [verify_video_embedding_artifact(row) for row in video_artifacts]
    broadcast = verify_broadcast_state_evidence_artifact(
        broadcast_state_artifact
    )
    overlay = verify_overlay_state_evidence_artifact(overlay_state_artifact)
    if (
        broadcast["source_scene_artifact_sha256"] != scene["artifact_sha256"]
        or overlay["source_scene_artifact_sha256"] != scene["artifact_sha256"]
    ):
        raise ValueError("state evidence does not bind the scene artifact")

    examples, base_matrix = _base_matrix(scene, videos)
    keys = [_example_key(row) for row in examples]
    broadcast_matrix = _aligned_matrix(
        broadcast["examples"],
        keys,
        name="broadcast",
    )
    overlay_matrix = _aligned_matrix(
        overlay["examples"],
        keys,
        name="overlay",
    )
    target = np.asarray(
        [int(bool(row["event_present"])) for row in examples],
        dtype=np.int64,
    )
    groups = np.asarray(
        [str(row["source_video_sha256"]) for row in examples],
        dtype=object,
    )
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 4:
        raise ValueError("overlay fusion requires at least four games")

    probabilities = {
        name: np.zeros(len(examples), dtype=np.float64) for name in VARIANT_NAMES
    }
    fold_provenance = []
    for held_group in unique_groups:
        training_mask = groups != held_group
        held_mask = ~training_mask
        training_groups = sorted(set(groups[training_mask]))
        reduced_training, reduced_held = _reduce_base(
            base_matrix[training_mask],
            base_matrix[held_mask],
            pca_components=pca_components,
        )
        broadcast_training, broadcast_held = _scale_fold(
            broadcast_matrix,
            training_mask,
            held_mask,
        )
        overlay_training, overlay_held = _scale_fold(
            overlay_matrix,
            training_mask,
            held_mask,
        )
        fold_inputs = {
            "base": (reduced_training, reduced_held),
            "base+broadcast_raw": (
                np.concatenate([reduced_training, broadcast_training], axis=1),
                np.concatenate([reduced_held, broadcast_held], axis=1),
            ),
            "base+overlay_raw": (
                np.concatenate([reduced_training, overlay_training], axis=1),
                np.concatenate([reduced_held, overlay_held], axis=1),
            ),
            "base+broadcast+overlay_raw": (
                np.concatenate(
                    [reduced_training, broadcast_training, overlay_training],
                    axis=1,
                ),
                np.concatenate(
                    [reduced_held, broadcast_held, overlay_held],
                    axis=1,
                ),
            ),
        }
        for name, (training_matrix, held_matrix) in fold_inputs.items():
            classifier = LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ).fit(training_matrix, target[training_mask])
            probabilities[name][held_mask] = classifier.predict_proba(
                held_matrix
            )[:, 1]
        fold_provenance.append(
            {
                "held_game_sha256": held_group,
                "training_game_sha256s": training_groups,
                "training_example_count": int(training_mask.sum()),
                "held_example_count": int(held_mask.sum()),
            }
        )

    variants = [
        _variant_result(
            name=name,
            probabilities=probabilities[name],
            examples=examples,
            target=target,
            groups=groups,
        )
        for name in VARIANT_NAMES
    ]
    best = max(variants, key=lambda row: _gate_score(row["gate"]))
    payload = {
        "schema_version": "agu.shot-overlay-fusion-screen.v1",
        "purpose": "offline_label_free_overlay_state_fusion_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "selection_protocol": "outer_game_held_overlay_state_oof",
        "scene_embedding_artifact_sha256": scene["artifact_sha256"],
        "video_embedding_artifact_sha256s": [
            row["artifact_sha256"] for row in videos
        ],
        "broadcast_state_artifact_sha256": broadcast["artifact_sha256"],
        "overlay_state_artifact_sha256": overlay["artifact_sha256"],
        "pca_components": pca_components,
        "regularization_c": regularization_c,
        "fold_provenance": fold_provenance,
        "variants": variants,
        "best_variant": best,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _aligned_matrix(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[tuple[str, str, str]],
    *,
    name: str,
) -> np.ndarray:
    lookup = {_example_key(row): row for row in rows}
    if set(lookup) != set(keys) or len(lookup) != len(keys):
        raise ValueError(f"{name} state evidence must exactly cover scene examples")
    return np.asarray([lookup[key]["features"] for key in keys], dtype=np.float64)


def _scale_fold(
    matrix: np.ndarray,
    training_mask: np.ndarray,
    held_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(matrix[training_mask])
    return scaler.transform(matrix[training_mask]), scaler.transform(
        matrix[held_mask]
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
