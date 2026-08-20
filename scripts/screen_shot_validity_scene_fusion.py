#!/usr/bin/env python3
"""Screen phase-scene and video embeddings with game-held OOF calibration."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import (  # noqa: E402
    verify_nested_fusion_shot_vlm_plan,
)
from app.analysis.shot_validity_finetune import binary_metrics  # noqa: E402
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    verify_scene_embedding_artifact,
)
from app.analysis.shot_validity_video_backbone import (  # noqa: E402
    verify_video_embedding_artifact,
)

NESTED_SCENE_FUSION_SCHEMA = "agu.shot-validity-scene-fusion-screen.v2"
NESTED_SELECTION_PROTOCOL = "nested_outer_game_held_v2"
PLAN_BOUND_SUBSET_SELECTION_PROTOCOL = (
    "sealed_independent_shot_vlm_plan_three_part_key_order_v1"
)
FUSION_THRESHOLD_SELECTION = "nested_inner_game_held_minimum_recall_0_85"
FUSION_PROMOTION_REQUIREMENTS = {
    "minimum_pooled_precision": 0.85,
    "minimum_pooled_recall": 0.85,
    "minimum_per_game_precision": 0.85,
    "minimum_per_game_recall": 0.85,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def screen_scene_fusion(
    *,
    scene_path: Path,
    video_embedding_paths: Sequence[Path] = (),
    pca_components: int = 16,
    regularization_c: float = 0.01,
) -> dict[str, object]:
    """Reproduce label-leaky v1 observations as non-promotable provenance."""

    _validate_hyperparameters(
        pca_components=pca_components,
        regularization_c=regularization_c,
    )
    scene = verify_scene_embedding_artifact(
        json.loads(scene_path.read_text(encoding="utf-8"))
    )
    videos = [
        verify_video_embedding_artifact(json.loads(path.read_text(encoding="utf-8")))
        for path in video_embedding_paths
    ]
    examples, scene_matrix, video_matrices = _align_artifacts(scene, videos)
    target = np.asarray([int(row["event_present"]) for row in examples], dtype=np.int64)
    groups = np.asarray([str(row["source_video_sha256"]) for row in examples])
    if len(set(groups)) < 3 or len(set(target.tolist())) < 2:
        raise ValueError("scene screening requires at least three games and both classes")

    named_matrices = [(str(row["backbone"]), matrix) for row, matrix in zip(
        videos, video_matrices, strict=True
    )]
    variants = [("scene_phase_all", scene_matrix)]
    for count in range(1, len(named_matrices) + 1):
        for combination in itertools.combinations(named_matrices, count):
            variants.append(
                (
                    "scene_phase_all+" + "+".join(name for name, _matrix in combination),
                    np.concatenate(
                        [scene_matrix, *(matrix for _name, matrix in combination)],
                        axis=1,
                    ),
                )
            )
    results = [
        _screen_variant(
            name=name,
            matrix=matrix,
            examples=examples,
            target=target,
            groups=groups,
            pca_components=pca_components,
            regularization_c=regularization_c,
        )
        for name, matrix in variants
    ]
    best = max(results, key=_variant_score)
    for result in results:
        observed_gate = result["gate"]
        result["observed_metrics"] = {
            "pooled": observed_gate["pooled"],
            "per_game": observed_gate["per_game"],
            "legacy_gate_would_have_promoted": bool(observed_gate["promoted"]),
        }
        result["gate"] = {
            "promoted": False,
            "promotion_eligible": False,
            "reason": "legacy_v1_outer_label_leakage_provenance_only",
        }
    payload: dict[str, object] = {
        "schema_version": "agu.shot-validity-scene-fusion-screen.v1",
        "purpose": "scene_video_fusion_screening_training_only",
        "runtime_consumable": False,
        "legacy_provenance_only": True,
        "promotion_eligible": False,
        "promotion_decision": "ineligible_legacy_outer_label_leakage",
        "selection_protocol": "game_held_oof_predeclared_variants",
        "scene_embedding_artifact_sha256": scene["artifact_sha256"],
        "video_embedding_artifact_sha256s": [
            row["artifact_sha256"] for row in videos
        ],
        "training_manifest_sha256": scene["training_manifest_sha256"],
        "pca_components": pca_components,
        "regularization_c": regularization_c,
        "promotion_requirements": {
            "minimum_pooled_precision": 0.85,
            "minimum_per_game_precision": 0.85,
            "minimum_per_game_recall": 0.85,
        },
        "variants": results,
        "best_variant": best,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def screen_nested_outer_game_held_fusion(
    *,
    scene_path: Path,
    video_embedding_paths: Sequence[Path] = (),
    plan: Mapping[str, object],
    pca_components: int = 16,
    regularization_c: float = 0.01,
) -> dict[str, object]:
    """Screen predeclared fusion variants with a true nested game-held split.

    For each outer held game, both variant and threshold selection use inner OOF
    predictions from the remaining games only.  The selected variant is then
    refit on every remaining game before the outer held probabilities are
    produced.  The label-free plan binds the exact three-part event key set.

    Design references:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data
    """

    _validate_hyperparameters(
        pca_components=pca_components,
        regularization_c=regularization_c,
    )

    verified_plan = verify_nested_fusion_shot_vlm_plan(plan)
    plan_sha256 = _require_sha256(
        verified_plan.get("plan_sha256"),
        field="independent-shot VLM plan",
    )
    training_manifest_sha256 = _require_sha256(
        verified_plan.get("training_manifest_sha256"),
        field="training manifest",
    )
    scene = verify_scene_embedding_artifact(json.loads(scene_path.read_text(encoding="utf-8")))
    videos = sorted(
        (
            verify_video_embedding_artifact(json.loads(path.read_text(encoding="utf-8")))
            for path in video_embedding_paths
        ),
        key=lambda row: (str(row["backbone"]), str(row["artifact_sha256"])),
    )
    if len({str(row["backbone"]) for row in videos}) != len(videos):
        raise ValueError("video embedding artifacts require unique backbones")
    if len({str(row["artifact_sha256"]) for row in videos}) != len(videos):
        raise ValueError("video embedding artifacts must not be duplicated")
    _verify_plan_bound_subset(
        scene,
        artifact_name="scene embedding artifact",
        plan_sha256=plan_sha256,
    )
    for video in videos:
        _verify_plan_bound_subset(
            video,
            artifact_name=f"video embedding artifact {video['backbone']}",
            plan_sha256=plan_sha256,
        )
    if scene.get("training_manifest_sha256") != training_manifest_sha256:
        raise ValueError("scene embedding training manifest does not match the plan")
    if any(row.get("training_manifest_sha256") != training_manifest_sha256 for row in videos):
        raise ValueError("video embedding training manifest does not match the plan")

    plan_keys = _strict_example_keys(
        verified_plan["examples"],
        artifact_name="independent-shot VLM plan",
    )
    scene_keys = _strict_example_keys(
        scene["examples"],
        artifact_name="scene embedding artifact",
    )
    for video in videos:
        video_keys = _strict_example_keys(
            video["examples"],
            artifact_name=f"video embedding artifact {video['backbone']}",
        )
        if video_keys != plan_keys:
            raise ValueError(
                "video embedding three-part key order does not match the frozen plan"
            )
    if scene_keys != plan_keys:
        raise ValueError(
            "scene embedding three-part key order does not match the frozen plan"
        )

    examples, scene_matrix, video_matrices = _align_artifacts(scene, videos)
    target = np.asarray(
        [int(bool(row["event_present"])) for row in examples],
        dtype=np.int64,
    )
    groups = np.asarray(
        [str(row["source_video_sha256"]) for row in examples],
        dtype=str,
    )
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 3 or len(set(target.tolist())) < 2:
        raise ValueError("nested scene screening requires at least three games and both classes")

    variants = _fusion_variants(
        scene_matrix=scene_matrix,
        videos=videos,
        video_matrices=video_matrices,
    )
    fusion_screen_contract = _verify_fusion_screen_contract(
        verified_plan=verified_plan,
        scene=scene,
        videos=videos,
        variants=variants,
        pca_components=pca_components,
        regularization_c=regularization_c,
    )
    outer_probabilities = np.full(len(target), np.nan, dtype=np.float64)
    outer_decisions = np.zeros(len(target), dtype=bool)
    outer_folds: list[dict[str, object]] = []
    # Per scikit-learn's nested-CV guidance, inner folds select the variant and
    # threshold; the untouched outer fold estimates generalization.  Games are
    # indivisible groups, matching grouped CV's rule that one group cannot occur
    # in both the paired fitting and held splits (official references above).
    for held_group in unique_groups:
        development_groups = [group for group in unique_groups if group != held_group]
        development_mask = np.isin(groups, development_groups)
        variant_selection: list[dict[str, object]] = []
        for name, matrix in variants:
            inner_probabilities = np.full(len(target), np.nan, dtype=np.float64)
            inner_folds = []
            for inner_held_group in development_groups:
                inner_held_mask = development_mask & (groups == inner_held_group)
                inner_fit_groups = [group for group in development_groups if group != inner_held_group]
                inner_fit_mask = np.isin(groups, inner_fit_groups)
                inner_probabilities[inner_held_mask] = _fit_probability_model(
                    matrix=matrix,
                    target=target,
                    fit_mask=inner_fit_mask,
                    predict_mask=inner_held_mask,
                    pca_components=pca_components,
                    regularization_c=regularization_c,
                )
                inner_folds.append(
                    {
                        "held_game_sha256": inner_held_group,
                        "fit_game_sha256s": inner_fit_groups,
                        "row_count": int(inner_held_mask.sum()),
                    }
                )
            if not np.isfinite(inner_probabilities[development_mask]).all():
                raise ValueError("nested inner OOF predictions are incomplete")
            threshold, gate = _select_threshold(
                target[development_mask],
                inner_probabilities[development_mask],
                groups[development_mask],
            )
            variant_selection.append(
                {
                    "name": name,
                    "input_dimension": int(matrix.shape[1]),
                    "threshold": float(threshold),
                    "gate": gate,
                    "inner_folds": inner_folds,
                }
            )

        selected_index = max(
            range(len(variant_selection)),
            key=lambda index: _gate_score(variant_selection[index]["gate"]),
        )
        selected = variant_selection[selected_index]
        selected_name, selected_matrix = variants[selected_index]
        if selected_name != selected["name"]:
            raise ValueError("nested variant selection is internally inconsistent")
        held_mask = groups == held_group
        held_probabilities = _fit_probability_model(
            matrix=selected_matrix,
            target=target,
            fit_mask=development_mask,
            predict_mask=held_mask,
            pca_components=pca_components,
            regularization_c=regularization_c,
        )
        threshold = float(selected["threshold"])
        held_decisions = held_probabilities >= threshold
        outer_probabilities[held_mask] = held_probabilities
        outer_decisions[held_mask] = held_decisions
        held_examples = [row for row, is_held in zip(examples, held_mask.tolist(), strict=True) if is_held]
        outer_folds.append(
            {
                "held_game_sha256": held_group,
                "inner_selection_game_sha256s": development_groups,
                "fit_game_sha256s": development_groups,
                "selected_variant": selected_name,
                "threshold": threshold,
                "inner_selection_gate": selected["gate"],
                "variant_selection": variant_selection,
                "held_metrics": binary_metrics(
                    target[held_mask].astype(bool).tolist(),
                    held_probabilities.tolist(),
                    threshold,
                ),
                "oof_predictions": [
                    {
                        "source_video_sha256": str(row["source_video_sha256"]),
                        "candidate_bundle_sha256": str(row["candidate_bundle_sha256"]),
                        "event_id": str(row["event_id"]),
                        "probability": float(probability),
                    }
                    for row, probability in zip(
                        held_examples,
                        held_probabilities,
                        strict=True,
                    )
                ],
            }
        )

    if not np.isfinite(outer_probabilities).all():
        raise ValueError("nested outer OOF predictions are incomplete")
    per_game = {
        group: binary_metrics(
            target[groups == group].astype(bool).tolist(),
            outer_decisions[groups == group].astype(float).tolist(),
            0.5,
        )
        for group in unique_groups
    }
    pooled = binary_metrics(
        target.astype(bool).tolist(),
        outer_decisions.astype(float).tolist(),
        0.5,
    )
    promoted = (
        float(pooled["precision"])
        >= FUSION_PROMOTION_REQUIREMENTS["minimum_pooled_precision"]
        and float(pooled["recall"])
        >= FUSION_PROMOTION_REQUIREMENTS["minimum_pooled_recall"]
        and all(
            float(row["precision"])
            >= FUSION_PROMOTION_REQUIREMENTS["minimum_per_game_precision"]
            for row in per_game.values()
        )
        and all(
            float(row["recall"])
            >= FUSION_PROMOTION_REQUIREMENTS["minimum_per_game_recall"]
            for row in per_game.values()
        )
    )
    payload: dict[str, object] = {
        "schema_version": NESTED_SCENE_FUSION_SCHEMA,
        "purpose": "scene_video_fusion_screening_training_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "selection_protocol": NESTED_SELECTION_PROTOCOL,
        "threshold_selection": FUSION_THRESHOLD_SELECTION,
        "random_seed": 0,
        "plan_sha256": plan_sha256,
        "subset_selection_protocol": PLAN_BOUND_SUBSET_SELECTION_PROTOCOL,
        "fusion_screen_contract": fusion_screen_contract,
        "fusion_screen_contract_sha256": _canonical_sha256(
            fusion_screen_contract
        ),
        "training_manifest_sha256": training_manifest_sha256,
        "scene_embedding_artifact_sha256": fusion_screen_contract[
            "scene_embedding_artifact_sha256"
        ],
        "video_embedding_artifact_sha256s": [
            row["artifact_sha256"]
            for row in fusion_screen_contract["video_embedding_artifacts"]
        ],
        "scene_subset_embedding_artifact_sha256": scene["artifact_sha256"],
        "video_subset_embedding_artifact_sha256s": [
            row["artifact_sha256"] for row in videos
        ],
        "pca_components": pca_components,
        "regularization_c": regularization_c,
        "promotion_requirements": dict(FUSION_PROMOTION_REQUIREMENTS),
        "predeclared_variants": fusion_screen_contract["predeclared_variants"],
        "outer_folds": outer_folds,
        "gate": {
            "promoted": promoted,
            "pooled": pooled,
            "per_game": per_game,
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _scene_features(row: dict[str, object]) -> np.ndarray:
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


def _align_artifacts(
    scene: dict[str, object], videos: Sequence[dict[str, object]]
) -> tuple[list[dict[str, object]], np.ndarray, list[np.ndarray]]:
    if any(
        row["training_manifest_sha256"] != scene["training_manifest_sha256"]
        for row in videos
    ):
        raise ValueError("embedding artifacts must share one training manifest")

    def key(row: dict[str, object]) -> tuple[str, str, str]:
        return (
            str(row["source_video_sha256"]),
            str(row.get("candidate_bundle_sha256", "")),
            str(row["event_id"]),
        )

    examples = list(scene["examples"])
    expected_keys = [key(row) for row in examples]
    matrices = []
    for artifact in videos:
        lookup = {key(row): row for row in artifact["examples"]}
        if set(lookup) != set(expected_keys):
            raise ValueError("embedding artifacts must contain identical examples")
        aligned = [lookup[row_key] for row_key in expected_keys]
        if any(
            bool(row["event_present"]) != bool(expected["event_present"])
            for row, expected in zip(aligned, examples, strict=True)
        ):
            raise ValueError("embedding artifacts disagree on example labels")
        matrices.append(np.asarray([row["embedding"] for row in aligned]))
    return examples, np.stack([_scene_features(row) for row in examples]), matrices


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_hyperparameters(
    *,
    pca_components: object,
    regularization_c: object,
) -> None:
    if type(pca_components) is not int or pca_components < 1:
        raise ValueError("PCA components must be a positive integer")
    if (
        isinstance(regularization_c, bool)
        or not isinstance(regularization_c, (int, float))
        or not math.isfinite(float(regularization_c))
        or float(regularization_c) <= 0.0
    ):
        raise ValueError("regularization C must be a positive number")


def _verify_plan_bound_subset(
    artifact: Mapping[str, object],
    *,
    artifact_name: str,
    plan_sha256: str,
) -> None:
    if artifact.get("plan_sha256") != plan_sha256:
        raise ValueError(f"{artifact_name} plan binding does not match the frozen plan")
    if (
        artifact.get("subset_selection_protocol")
        != PLAN_BOUND_SUBSET_SELECTION_PROTOCOL
    ):
        raise ValueError(f"{artifact_name} subset selection protocol is invalid")


def _verify_fusion_screen_contract(
    *,
    verified_plan: Mapping[str, object],
    scene: Mapping[str, object],
    videos: Sequence[Mapping[str, object]],
    variants: Sequence[tuple[str, np.ndarray]],
    pca_components: int,
    regularization_c: float,
) -> dict[str, object]:
    raw_contract = verified_plan.get("fusion_screen_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("frozen plan requires a fusion screen contract")
    source_videos = sorted(
        (
            {
                "backbone": str(video["backbone"]),
                "artifact_sha256": _require_sha256(
                    video.get("source_embedding_artifact_sha256"),
                    field="video subset source embedding artifact",
                ),
            }
            for video in videos
        ),
        key=lambda row: (row["backbone"], row["artifact_sha256"]),
    )
    expected_contract: dict[str, object] = {
        "scene_embedding_artifact_sha256": _require_sha256(
            scene.get("source_embedding_artifact_sha256"),
            field="scene subset source embedding artifact",
        ),
        "video_embedding_artifacts": source_videos,
        "predeclared_variants": _variant_specs(variants),
        "pca_components": pca_components,
        "regularization_c": regularization_c,
        "threshold_selection": FUSION_THRESHOLD_SELECTION,
        "promotion_requirements": dict(FUSION_PROMOTION_REQUIREMENTS),
    }
    contract = dict(raw_contract)
    if _canonical_json(contract) != _canonical_json(expected_contract):
        raise ValueError(
            "frozen plan fusion screen contract does not match source artifacts, "
            "variants, invocation parameters, or promotion policy"
        )
    return contract


def _variant_specs(
    variants: Sequence[tuple[str, np.ndarray]],
) -> list[dict[str, object]]:
    return [
        {"name": name, "input_dimension": int(matrix.shape[1])}
        for name, matrix in variants
    ]


def _strict_example_keys(
    rows: object,
    *,
    artifact_name: str,
) -> list[tuple[str, str, str]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{artifact_name} requires examples")
    keys = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{artifact_name} examples must be objects")
        source_video_sha256 = _require_sha256(
            row.get("source_video_sha256"),
            field=f"{artifact_name} source video",
        )
        candidate_bundle_sha256 = _require_sha256(
            row.get("candidate_bundle_sha256"),
            field=f"{artifact_name} candidate bundle",
        )
        event_id = row.get("event_id")
        if not isinstance(event_id, str):
            raise ValueError(f"{artifact_name} three-part key fields must be strings")
        if not event_id:
            raise ValueError(f"{artifact_name} event ID is missing")
        keys.append((source_video_sha256, candidate_bundle_sha256, event_id))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{artifact_name} contains duplicate three-part keys")
    return keys


def _fusion_variants(
    *,
    scene_matrix: np.ndarray,
    videos: Sequence[dict[str, object]],
    video_matrices: Sequence[np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    named_matrices = [(str(row["backbone"]), matrix) for row, matrix in zip(videos, video_matrices, strict=True)]
    variants = [("scene_phase_all", scene_matrix)]
    for count in range(1, len(named_matrices) + 1):
        for combination in itertools.combinations(named_matrices, count):
            variants.append(
                (
                    "scene_phase_all+" + "+".join(name for name, _matrix in combination),
                    np.concatenate(
                        [scene_matrix, *(matrix for _name, matrix in combination)],
                        axis=1,
                    ),
                )
            )
    return variants


def _fit_probability_model(
    *,
    matrix: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    predict_mask: np.ndarray,
    pca_components: int,
    regularization_c: float,
) -> np.ndarray:
    if fit_mask.shape != target.shape or predict_mask.shape != target.shape:
        raise ValueError("nested model masks must align with labels")
    if not fit_mask.any() or not predict_mask.any():
        raise ValueError("nested model fit and prediction splits must be non-empty")
    if np.any(fit_mask & predict_mask):
        raise ValueError("nested model fit and prediction splits must be disjoint")
    fit_target = target[fit_mask]
    if set(fit_target.tolist()) != {0, 1}:
        raise ValueError("nested model fitting requires both classes")
    scaler = StandardScaler().fit(matrix[fit_mask])
    fit_matrix = scaler.transform(matrix[fit_mask])
    predict_matrix = scaler.transform(matrix[predict_mask])
    component_count = min(
        pca_components,
        fit_matrix.shape[0] - 1,
        fit_matrix.shape[1],
    )
    if component_count < 1:
        raise ValueError("nested PCA requires at least two fitting examples")
    reducer = PCA(n_components=component_count, random_state=0).fit(fit_matrix)
    classifier = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
    ).fit(reducer.transform(fit_matrix), fit_target)
    probabilities = classifier.predict_proba(reducer.transform(predict_matrix))[:, 1]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("nested model produced invalid probabilities")
    return probabilities


def _screen_variant(
    *,
    name: str,
    matrix: np.ndarray,
    examples: list[dict[str, object]],
    target: np.ndarray,
    groups: np.ndarray,
    pca_components: int,
    regularization_c: float,
) -> dict[str, object]:
    probabilities = np.zeros(len(target), dtype=np.float64)
    folds = []
    for group in sorted(set(groups)):
        train = groups != group
        held = ~train
        scaler = StandardScaler().fit(matrix[train])
        training_matrix = scaler.transform(matrix[train])
        held_matrix = scaler.transform(matrix[held])
        component_count = min(
            pca_components, training_matrix.shape[0] - 1, training_matrix.shape[1]
        )
        reducer = PCA(n_components=component_count, random_state=0).fit(training_matrix)
        classifier = LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ).fit(reducer.transform(training_matrix), target[train])
        held_probabilities = classifier.predict_proba(
            reducer.transform(held_matrix)
        )[:, 1]
        probabilities[held] = held_probabilities
        folds.append(
            {
                "held_game_sha256": group,
                "training_game_sha256s": sorted(set(groups[train])),
                "rows": int(held.sum()),
                "positive_count": int(target[held].sum()),
                "roc_auc": float(roc_auc_score(target[held], held_probabilities)),
                "average_precision": float(
                    average_precision_score(target[held], held_probabilities)
                ),
                "best_precision_at_recall_0_85": _best_precision_at_recall(
                    target[held], held_probabilities, minimum_recall=0.85
                ),
            }
        )
    threshold, gate = _select_threshold(target, probabilities, groups)
    return {
        "name": name,
        "input_dimension": int(matrix.shape[1]),
        "threshold": threshold,
        "folds": folds,
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
    target: np.ndarray, probabilities: np.ndarray, groups: np.ndarray
) -> tuple[float, dict[str, object]]:
    best_fallback: tuple[tuple[float, ...], float, dict[str, object]] | None = None
    recall_eligible = []
    promoted_candidates = []
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
        promoted = (
            float(pooled["precision"])
            >= FUSION_PROMOTION_REQUIREMENTS["minimum_pooled_precision"]
            and float(pooled["recall"])
            >= FUSION_PROMOTION_REQUIREMENTS["minimum_pooled_recall"]
            and all(
                float(row["precision"])
                >= FUSION_PROMOTION_REQUIREMENTS["minimum_per_game_precision"]
                for row in per_game.values()
            )
            and all(
                float(row["recall"])
                >= FUSION_PROMOTION_REQUIREMENTS["minimum_per_game_recall"]
                for row in per_game.values()
            )
        )
        gate = {"promoted": promoted, "pooled": pooled, "per_game": per_game}
        fallback_score = (
            min(float(row["recall"]) for row in per_game.values()),
            min(float(row["precision"]) for row in per_game.values()),
            float(pooled["f1"]),
        )
        if best_fallback is None or fallback_score > best_fallback[0]:
            best_fallback = (fallback_score, threshold, gate)
        if all(
            float(row["recall"])
            >= FUSION_PROMOTION_REQUIREMENTS["minimum_per_game_recall"]
            for row in per_game.values()
        ):
            recall_eligible.append(
                (
                    min(float(row["precision"]) for row in per_game.values()),
                    float(pooled["f1"]),
                    -threshold,
                    threshold,
                    gate,
                )
            )
        if promoted:
            promoted_candidates.append(
                (float(pooled["f1"]), float(pooled["recall"]), -threshold, threshold, gate)
            )
    if promoted_candidates:
        _f1, _recall, _negative_threshold, threshold, gate = max(
            promoted_candidates
        )
        return threshold, gate
    if recall_eligible:
        _precision, _f1, _negative_threshold, threshold, gate = max(
            recall_eligible
        )
        return threshold, gate
    assert best_fallback is not None
    _score, threshold, gate = best_fallback
    return threshold, gate


def _gate_score(gate: dict[str, object]) -> tuple[float, ...]:
    per_game = list(gate["per_game"].values())
    recall_eligible = all(
        float(row["recall"])
        >= FUSION_PROMOTION_REQUIREMENTS["minimum_per_game_recall"]
        for row in per_game
    )
    return (
        float(gate["promoted"]),
        float(recall_eligible),
        min(float(row["precision"]) for row in per_game),
        min(float(row["recall"]) for row in per_game),
        float(gate["pooled"]["f1"]),
    )


def _variant_score(result: dict[str, object]) -> tuple[float, ...]:
    return _gate_score(result["gate"])


def _best_precision_at_recall(
    target: np.ndarray, probabilities: np.ndarray, *, minimum_recall: float
) -> dict[str, float | int]:
    candidates = []
    for threshold in sorted({0.0, 1.0, *(float(value) for value in probabilities)}):
        metrics = binary_metrics(
            target.astype(bool).tolist(), probabilities.tolist(), threshold
        )
        if float(metrics["recall"]) >= minimum_recall:
            candidates.append(
                (float(metrics["precision"]), float(metrics["f1"]), threshold, metrics)
            )
    if not candidates:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 1.0}
    _precision, _f1, threshold, metrics = max(candidates)
    return {**metrics, "threshold": threshold}


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_output_disjoint_from_inputs(
    output_path: Path,
    *,
    input_paths: Sequence[Path],
) -> tuple[Path, list[Path]]:
    resolved_output = output_path.resolve()
    resolved_inputs = [path.resolve() for path in input_paths]
    if resolved_output in set(resolved_inputs):
        raise ValueError("output path must not alias a plan, scene, or video input path")
    return resolved_output, resolved_inputs


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-embeddings", type=Path, required=True)
    parser.add_argument("--video-embeddings", action="append", type=Path, default=[])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan",
        type=Path,
        help="frozen independent-shot VLM plan; enables nested v2 screening",
    )
    mode.add_argument(
        "--legacy-v1-provenance-only",
        action="store_true",
        help="reproduce leaky v1 metrics as non-promotable provenance only",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--regularization-c", type=float, default=0.01)
    args = parser.parse_args()
    input_paths = [args.scene_embeddings, *args.video_embeddings]
    if args.plan is not None:
        input_paths.append(args.plan)
    output_path, resolved_inputs = _require_output_disjoint_from_inputs(
        args.output,
        input_paths=input_paths,
    )
    scene_embeddings_path = resolved_inputs[0]
    video_embedding_paths = resolved_inputs[1 : 1 + len(args.video_embeddings)]
    plan_path = resolved_inputs[-1] if args.plan is not None else None
    if args.legacy_v1_provenance_only:
        result = screen_scene_fusion(
            scene_path=scene_embeddings_path,
            video_embedding_paths=video_embedding_paths,
            pca_components=args.pca_components,
            regularization_c=args.regularization_c,
        )
    else:
        assert plan_path is not None
        result = screen_nested_outer_game_held_fusion(
            scene_path=scene_embeddings_path,
            video_embedding_paths=video_embedding_paths,
            plan=json.loads(plan_path.read_text(encoding="utf-8")),
            pca_components=args.pca_components,
            regularization_c=args.regularization_c,
        )
    _write_json_atomic(output_path, result)
    if args.legacy_v1_provenance_only:
        summary = {
            "legacy_provenance_only": True,
            "promotion_eligible": False,
            "best_variant": result["best_variant"]["name"],
            "gate": result["best_variant"]["gate"],
            "observed_metrics": result["best_variant"]["observed_metrics"],
            "artifact_sha256": result["artifact_sha256"],
        }
    else:
        summary = {
            "selection_protocol": result["selection_protocol"],
            "outer_fold_count": len(result["outer_folds"]),
            "gate": result["gate"],
            "artifact_sha256": result["artifact_sha256"],
        }
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )
    if args.legacy_v1_provenance_only:
        return 2
    return 0 if result["gate"]["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
