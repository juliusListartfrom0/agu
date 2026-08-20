"""Leakage-safe game-held screening for continuous shot-reason evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from app.analysis.shot_reason_evidence import verify_reason_evidence_artifact
from app.analysis.shot_validity_finetune import binary_metrics
from app.analysis.shot_validity_scene_state import verify_scene_embedding_artifact
from app.analysis.shot_validity_video_backbone import verify_video_embedding_artifact

REASON_NAMES = ("free_throw_formation", "replay_or_stoppage", "causal_release")
VARIANT_NAMES = (
    "base",
    "base+continuous_raw",
    "base+nested_reason_scores",
    "base+continuous_raw+nested_reason_scores",
)


def screen_shot_reason_fusion(
    *,
    scene_artifact: Mapping[str, Any],
    video_artifacts: Sequence[Mapping[str, Any]],
    reason_evidence_artifact: Mapping[str, Any],
    review_rows: Sequence[Mapping[str, Any]],
    pca_components: int = 16,
    regularization_c: float = 0.01,
) -> dict[str, Any]:
    """Compare base and continuous-evidence variants with nested group isolation."""
    if pca_components < 1 or regularization_c <= 0:
        raise ValueError("reason fusion PCA and regularization must be positive")
    scene = verify_scene_embedding_artifact(scene_artifact)
    videos = [verify_video_embedding_artifact(row) for row in video_artifacts]
    reason_evidence = verify_reason_evidence_artifact(reason_evidence_artifact)
    if reason_evidence["source_scene_artifact_sha256"] != scene["artifact_sha256"]:
        raise ValueError("reason evidence does not bind the scene artifact")

    examples, base_matrix = _base_matrix(scene, videos)
    keys = [_example_key(row) for row in examples]
    key_to_index = {key: index for index, key in enumerate(keys)}
    evidence_lookup = {
        _example_key(row): row for row in reason_evidence["examples"]
    }
    if set(evidence_lookup) != set(keys):
        raise ValueError("reason evidence must exactly cover scene examples")
    raw_matrix = np.asarray(
        [evidence_lookup[key]["features"] for key in keys], dtype=np.float64
    )
    target = np.asarray(
        [int(bool(row["event_present"])) for row in examples], dtype=np.int64
    )
    groups = np.asarray(
        [str(row["source_video_sha256"]) for row in examples], dtype=object
    )
    if len(set(groups)) < 4:
        raise ValueError("nested reason fusion requires at least four games")

    review_lookup = {
        _example_key(row): row
        for row in review_rows
        if str(row.get("review_confidence") or "") != "uncertain"
    }
    if not review_lookup or not set(review_lookup).issubset(key_to_index):
        raise ValueError("reason reviews must cover a scene-example subset")

    probabilities = {
        name: np.zeros(len(examples), dtype=np.float64) for name in VARIANT_NAMES
    }
    auxiliary_provenance = []
    for held_group in sorted(set(groups)):
        train_mask = groups != held_group
        held_mask = ~train_mask
        outer_training_groups = sorted(set(groups[train_mask]))
        nested_scores = np.zeros((len(examples), len(REASON_NAMES)), dtype=np.float64)
        reason_provenance = []
        for reason_index, reason_name in enumerate(REASON_NAMES):
            held_training_indices = _review_indices(
                review_lookup,
                key_to_index,
                allowed_groups=set(outer_training_groups),
            )
            held_training_labels = np.asarray(
                [
                    _reason_label(review_lookup[keys[index]], reason_name)
                    for index in held_training_indices
                ],
                dtype=np.int64,
            )
            nested_scores[held_mask, reason_index] = _reason_probabilities(
                raw_matrix[held_training_indices],
                held_training_labels,
                raw_matrix[held_mask],
            )
            training_score_provenance = []
            for scored_group in outer_training_groups:
                training_groups = [
                    group
                    for group in outer_training_groups
                    if group != scored_group
                ]
                training_indices = _review_indices(
                    review_lookup,
                    key_to_index,
                    allowed_groups=set(training_groups),
                )
                training_labels = np.asarray(
                    [
                        _reason_label(review_lookup[keys[index]], reason_name)
                        for index in training_indices
                    ],
                    dtype=np.int64,
                )
                scored_mask = groups == scored_group
                nested_scores[scored_mask, reason_index] = _reason_probabilities(
                    raw_matrix[training_indices],
                    training_labels,
                    raw_matrix[scored_mask],
                )
                training_score_provenance.append(
                    {
                        "scored_game_sha256": scored_group,
                        "training_game_sha256s": training_groups,
                        "training_review_count": len(training_indices),
                    }
                )
            reason_provenance.append(
                {
                    "reason": reason_name,
                    "held_score_training_game_sha256s": outer_training_groups,
                    "held_score_training_review_count": len(held_training_indices),
                    "training_scores": training_score_provenance,
                }
            )

        reduced_train, reduced_held = _reduce_base(
            base_matrix[train_mask],
            base_matrix[held_mask],
            pca_components=pca_components,
        )
        raw_scaler = StandardScaler().fit(raw_matrix[train_mask])
        raw_train = raw_scaler.transform(raw_matrix[train_mask])
        raw_held = raw_scaler.transform(raw_matrix[held_mask])
        score_scaler = StandardScaler().fit(nested_scores[train_mask])
        score_train = score_scaler.transform(nested_scores[train_mask])
        score_held = score_scaler.transform(nested_scores[held_mask])
        fold_inputs = {
            "base": (reduced_train, reduced_held),
            "base+continuous_raw": (
                np.concatenate([reduced_train, raw_train], axis=1),
                np.concatenate([reduced_held, raw_held], axis=1),
            ),
            "base+nested_reason_scores": (
                np.concatenate([reduced_train, score_train], axis=1),
                np.concatenate([reduced_held, score_held], axis=1),
            ),
            "base+continuous_raw+nested_reason_scores": (
                np.concatenate([reduced_train, raw_train, score_train], axis=1),
                np.concatenate([reduced_held, raw_held, score_held], axis=1),
            ),
        }
        for name, (training_matrix, held_matrix) in fold_inputs.items():
            classifier = LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ).fit(training_matrix, target[train_mask])
            probabilities[name][held_mask] = classifier.predict_proba(held_matrix)[:, 1]
        auxiliary_provenance.append(
            {
                "held_game_sha256": held_group,
                "outer_training_game_sha256s": outer_training_groups,
                "reasons": reason_provenance,
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
        "schema_version": "agu.shot-reason-fusion-screen.v1",
        "purpose": "offline_shot_reason_fusion_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "selection_protocol": "outer_game_held_with_nested_reason_game_oof",
        "scene_embedding_artifact_sha256": scene["artifact_sha256"],
        "video_embedding_artifact_sha256s": [
            row["artifact_sha256"] for row in videos
        ],
        "reason_evidence_artifact_sha256": reason_evidence["artifact_sha256"],
        "review_count": len(review_lookup),
        "reason_names": list(REASON_NAMES),
        "pca_components": pca_components,
        "regularization_c": regularization_c,
        "auxiliary_metrics": _reason_metrics(
            raw_matrix=raw_matrix,
            groups=groups,
            keys=keys,
            key_to_index=key_to_index,
            review_lookup=review_lookup,
        ),
        "auxiliary_provenance": auxiliary_provenance,
        "variants": variants,
        "best_variant": best,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _base_matrix(
    scene: Mapping[str, Any],
    videos: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    examples = list(scene["examples"])
    keys = [_example_key(row) for row in examples]
    matrices = [np.stack([_scene_features(row) for row in examples])]
    for video in videos:
        if video["training_manifest_sha256"] != scene["training_manifest_sha256"]:
            raise ValueError("reason fusion artifacts must share a training manifest")
        lookup = {_example_key(row): row for row in video["examples"]}
        if set(lookup) != set(keys):
            raise ValueError("reason fusion video examples do not align")
        matrices.append(np.asarray([lookup[key]["embedding"] for key in keys]))
    return examples, np.concatenate(matrices, axis=1)


def _scene_features(row: Mapping[str, Any]) -> np.ndarray:
    phases = np.asarray(row["phase_embeddings"], dtype=np.float64)
    return np.concatenate(
        [
            phases.mean(axis=0),
            phases.std(axis=0),
            phases[-1] - phases[0],
            phases[0],
            phases[1],
            phases[2],
        ]
    )


def _reduce_base(
    training: np.ndarray,
    held: np.ndarray,
    *,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(training)
    scaled_training = scaler.transform(training)
    scaled_held = scaler.transform(held)
    component_count = min(
        pca_components, scaled_training.shape[0] - 1, scaled_training.shape[1]
    )
    reducer = PCA(n_components=component_count, random_state=0).fit(scaled_training)
    return reducer.transform(scaled_training), reducer.transform(scaled_held)


def _reason_probabilities(
    training: np.ndarray,
    labels: np.ndarray,
    scored: np.ndarray,
) -> np.ndarray:
    if len(labels) == 0:
        return np.full(len(scored), 0.5, dtype=np.float64)
    if len(set(labels.tolist())) < 2:
        return np.full(len(scored), float(labels.mean()), dtype=np.float64)
    classifier = ExtraTreesClassifier(
        n_estimators=256,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=0,
        n_jobs=1,
    ).fit(training, labels)
    class_index = list(classifier.classes_).index(1)
    return classifier.predict_proba(scored)[:, class_index]


def _review_indices(
    reviews: Mapping[tuple[str, str, str], Mapping[str, Any]],
    key_to_index: Mapping[tuple[str, str, str], int],
    *,
    allowed_groups: set[str],
) -> list[int]:
    return sorted(
        key_to_index[key]
        for key in reviews
        if key[0] in allowed_groups
    )


def _reason_label(review: Mapping[str, Any], reason: str) -> int:
    if reason == "causal_release":
        return int(
            review.get("release_observed") is True
            and review.get("ball_moves_toward_rim") is True
            and review.get("free_throw_formation") is not True
            and review.get("replay_or_stoppage") is not True
        )
    return int(review.get(reason) is True)


def _reason_metrics(
    *,
    raw_matrix: np.ndarray,
    groups: np.ndarray,
    keys: Sequence[tuple[str, str, str]],
    key_to_index: Mapping[tuple[str, str, str], int],
    review_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for reason in REASON_NAMES:
        truth: list[int] = []
        probabilities: list[float] = []
        for held_group in sorted({key[0] for key in review_lookup}):
            training_indices = _review_indices(
                review_lookup,
                key_to_index,
                allowed_groups=set(groups) - {held_group},
            )
            held_indices = _review_indices(
                review_lookup,
                key_to_index,
                allowed_groups={held_group},
            )
            labels = np.asarray(
                [
                    _reason_label(review_lookup[keys[index]], reason)
                    for index in training_indices
                ]
            )
            probabilities.extend(
                _reason_probabilities(
                    raw_matrix[training_indices],
                    labels,
                    raw_matrix[held_indices],
                ).tolist()
            )
            truth.extend(
                _reason_label(review_lookup[keys[index]], reason)
                for index in held_indices
            )
        predicted = np.asarray(probabilities) >= 0.5
        precision, recall, f1, _support = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        results.append(
            {
                "reason": reason,
                "review_count": len(truth),
                "positive_count": int(sum(truth)),
                "precision_at_0_5": float(precision),
                "recall_at_0_5": float(recall),
                "f1_at_0_5": float(f1),
                "roc_auc": float(roc_auc_score(truth, probabilities)),
                "average_precision": float(
                    average_precision_score(truth, probabilities)
                ),
            }
        )
    return results


def _variant_result(
    *,
    name: str,
    probabilities: np.ndarray,
    examples: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    threshold, gate = _select_threshold(target, probabilities, groups)
    return {
        "name": name,
        "threshold": threshold,
        "gate": gate,
        "oof_predictions": [
            {
                "source_video_sha256": str(row["source_video_sha256"]),
                "event_id": str(row["event_id"]),
                "event_present": bool(row["event_present"]),
                "probability": float(probability),
            }
            for row, probability in zip(examples, probabilities, strict=True)
        ],
    }


def _select_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    candidates = []
    fallback = []
    for threshold in sorted({0.0, 1.0, *(float(value) for value in probabilities)}):
        pooled = binary_metrics(
            target.astype(bool).tolist(), probabilities.tolist(), threshold
        )
        per_game = {
            group: binary_metrics(
                target[groups == group].astype(bool).tolist(),
                probabilities[groups == group].tolist(),
                threshold,
            )
            for group in sorted(set(groups))
        }
        gate = {
            "promoted": (
                float(pooled["precision"]) >= 0.85
                and all(
                    float(row["precision"]) >= 0.85
                    and float(row["recall"]) >= 0.85
                    for row in per_game.values()
                )
            ),
            "pooled": pooled,
            "per_game": per_game,
        }
        score = _gate_score(gate)
        fallback.append((score, -threshold, threshold, gate))
        if all(float(row["recall"]) >= 0.85 for row in per_game.values()):
            candidates.append(
                (
                    min(float(row["precision"]) for row in per_game.values()),
                    float(pooled["f1"]),
                    -threshold,
                    threshold,
                    gate,
                )
            )
    if candidates:
        _precision, _f1, _negative, threshold, gate = max(candidates)
        return threshold, gate
    _score, _negative, threshold, gate = max(fallback)
    return threshold, gate


def _gate_score(gate: Mapping[str, Any]) -> tuple[float, ...]:
    per_game = list(gate["per_game"].values())
    return (
        float(gate["promoted"]),
        float(all(float(row["recall"]) >= 0.85 for row in per_game)),
        min(float(row["precision"]) for row in per_game),
        min(float(row["recall"]) for row in per_game),
        float(gate["pooled"]["f1"]),
    )


def _example_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    key = (
        str(row.get("source_video_sha256") or ""),
        str(row.get("candidate_bundle_sha256") or ""),
        str(row.get("event_id") or ""),
    )
    if not all(key):
        raise ValueError("reason fusion example key is incomplete")
    return key


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
