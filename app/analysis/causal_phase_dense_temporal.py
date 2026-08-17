"""Dense causal-phase temporal training with strict game-held evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.analysis.causal_phase_auxiliary_screen import PHASE_TARGET_NAMES
from app.analysis.causal_shot_phase_review import (
    CAUSAL_PHASE_OFFSETS_SECONDS,
)
from app.analysis.shot_reason_evidence import (
    FEATURE_NAMES as REASON_FEATURE_NAMES,
)
from app.analysis.shot_reason_evidence import (
    verify_reason_evidence_artifact,
)
from app.analysis.visual_state_temporal_screen import _metrics

DENSE_CAUSAL_FRAME_EMBEDDING_SCHEMA = (
    "agu.causal-phase-dense-frame-embeddings.v1"
)
DENSE_CAUSAL_TEMPORAL_SCREEN_SCHEMA = (
    "agu.causal-phase-dense-temporal-screen.v1"
)
VISUAL_STATE_LABELS = ("field_goal", "free_throw")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def crop_dense_causal_sheet_panels(
    image: np.ndarray,
    *,
    example_index: int,
    examples_in_sheet: int,
    panel_width: int = 320,
    grid_columns: int = 6,
    grid_rows_per_example: int = 4,
    header_height: int = 28,
) -> list[np.ndarray]:
    """Recover one event's 24 raw panels while removing rendered headers."""

    if (
        image.ndim != 3
        or image.shape[2] != 3
        or example_index < 0
        or examples_in_sheet < 1
        or panel_width < 1
        or grid_columns < 1
        or grid_rows_per_example < 1
        or header_height < 0
        or image.shape[1] != panel_width * grid_columns
        or image.shape[0] % (grid_rows_per_example * examples_in_sheet)
    ):
        raise ValueError("invalid dense causal review sheet geometry")
    panel_height = image.shape[0] // (
        grid_rows_per_example * examples_in_sheet
    )
    if example_index >= examples_in_sheet or header_height >= panel_height:
        raise ValueError("dense causal example is outside the review sheet")
    start_y = example_index * grid_rows_per_example * panel_height
    panels = []
    for position in range(grid_columns * grid_rows_per_example):
        row = position // grid_columns
        column = position % grid_columns
        top = start_y + row * panel_height + header_height
        bottom = start_y + (row + 1) * panel_height
        left = column * panel_width
        right = left + panel_width
        panels.append(image[top:bottom, left:right].copy())
    if len(panels) != len(CAUSAL_PHASE_OFFSETS_SECONDS):
        raise ValueError("dense causal sheet must contain exactly 24 panels")
    return panels


class DenseCausalTemporalClassifier(nn.Module):
    """Learn temporal phase structure from every ordered frame embedding."""

    def __init__(
        self,
        *,
        embedding_dimension: int,
        hidden_dimension: int,
        kernel_size: int,
        temporal_operator: str = "conv",
    ) -> None:
        super().__init__()
        if embedding_dimension < 1 or hidden_dimension < 1:
            raise ValueError("dense temporal dimensions must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("dense temporal kernel size must be positive and odd")
        if temporal_operator not in {"conv", "gru", "gated"}:
            raise ValueError("unsupported dense temporal operator")
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.temporal_operator = temporal_operator
        self.projection = nn.Linear(embedding_dimension, hidden_dimension)
        if temporal_operator in {"conv", "gated"}:
            self.temporal: nn.Module = nn.Conv1d(
                hidden_dimension,
                hidden_dimension,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            pooled_dimension = hidden_dimension * 3
            self.gate = (
                nn.Linear(hidden_dimension, 1)
                if temporal_operator == "gated"
                else None
            )
        else:
            self.temporal = nn.GRU(
                input_size=hidden_dimension,
                hidden_size=hidden_dimension,
                batch_first=True,
                bidirectional=True,
            )
            pooled_dimension = hidden_dimension * 6
            self.gate = None
        self.main_head = nn.Linear(pooled_dimension, 2)
        self.phase_head = nn.Linear(
            pooled_dimension,
            len(PHASE_TARGET_NAMES),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encode(embeddings)
        if self.temporal_operator == "gated":
            weights = self._attention_weights(encoded)
            pooled = torch.cat(
                (
                    (encoded * weights.unsqueeze(-1)).sum(dim=1),
                    encoded.amax(dim=1),
                    encoded[:, -1] - encoded[:, 0],
                ),
                dim=1,
            )
        else:
            pooled = torch.cat(
                (
                    encoded.mean(dim=1),
                    encoded.std(dim=1, unbiased=False),
                    encoded[:, -1] - encoded[:, 0],
                ),
                dim=1,
            )
        return self.main_head(pooled), self.phase_head(pooled)

    def temporal_attention_weights(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized sparse-evidence weights for a gated model."""

        if self.temporal_operator != "gated":
            raise ValueError("temporal attention requires the gated operator")
        return self._attention_weights(self._encode(embeddings))

    def _encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        if (
            embeddings.ndim != 3
            or embeddings.shape[1] != len(CAUSAL_PHASE_OFFSETS_SECONDS)
            or embeddings.shape[2] != self.embedding_dimension
        ):
            raise ValueError(
                "dense temporal input requires 24 ordered frame embeddings"
            )
        encoded = F.gelu(self.projection(embeddings))
        if self.temporal_operator in {"conv", "gated"}:
            return F.gelu(
                self.temporal(encoded.transpose(1, 2))
            ).transpose(1, 2)
        encoded, _ = self.temporal(encoded)
        return encoded

    def _attention_weights(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            raise ValueError("temporal attention gate is unavailable")
        return torch.softmax(self.gate(encoded).squeeze(-1), dim=1)


def seal_dense_causal_frame_embeddings(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = DENSE_CAUSAL_FRAME_EMBEDDING_SCHEMA
    artifact["runtime_consumable"] = False
    _validate_embedding_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_dense_causal_frame_embeddings(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_embedding_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("dense causal frame embedding hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def build_dense_causal_temporal_examples(
    *,
    embedding_artifact: Mapping[str, Any],
    base_decisions: Sequence[Mapping[str, Any]],
    followup_decisions: Sequence[Mapping[str, Any]],
    phase_reviews: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact-join dense frames, corrected main labels and phase supervision."""

    artifact = verify_dense_causal_frame_embeddings(embedding_artifact)
    embedding_by_key = _index_rows(
        artifact["examples"],
        field="dense embedding",
    )
    base_by_key = _index_rows(base_decisions, field="base decision")
    if set(base_by_key) != set(embedding_by_key):
        raise ValueError(
            "base decisions must exactly cover dense embedding examples"
        )
    unresolved = {
        key
        for key, row in base_by_key.items()
        if row.get("corrected_state") is None
    }
    followup_by_key = _index_rows(
        followup_decisions,
        field="follow-up decision",
    )
    if set(followup_by_key) != unresolved:
        raise ValueError(
            "follow-up decisions must exactly cover unresolved base decisions"
        )
    reviews_by_key = _index_rows(phase_reviews, field="phase review")
    if set(reviews_by_key) != set(embedding_by_key):
        raise ValueError(
            "phase reviews must exactly cover dense embedding examples"
        )

    rows = []
    for key in sorted(embedding_by_key):
        embedding = embedding_by_key[key]
        review = reviews_by_key[key]
        if str(review.get("phase_review_id") or "") != str(
            embedding["phase_review_id"]
        ):
            raise ValueError("dense embedding phase review binding mismatch")
        decision = followup_by_key.get(key, base_by_key[key])
        label = decision.get("corrected_state")
        if label is None:
            continue
        if label not in VISUAL_STATE_LABELS:
            raise ValueError("invalid dense temporal visual-state label")
        rows.append(
            {
                "phase_review_id": embedding["phase_review_id"],
                "source_video_sha256": key[0],
                "candidate_bundle_sha256": key[1],
                "event_id": key[2],
                "label": label,
                "frame_indexes": list(embedding["frame_indexes"]),
                "offsets_seconds": list(
                    artifact["anchor_offsets_seconds"]
                ),
                "embeddings": embedding["embeddings"],
                "phase_targets": _phase_targets(review),
            }
        )
    if not rows:
        raise ValueError("dense temporal screening requires resolved examples")

    return rows, {
        "backbone": artifact["backbone"],
        "backbone_sha256": artifact["backbone_sha256"],
        "embedding_dimension": artifact["embedding_dimension"],
        "embedding_artifact_sha256": artifact["artifact_sha256"],
        "review_plan_sha256": artifact["review_plan_sha256"],
        "sheet_manifest_sha256": artifact["sheet_manifest_sha256"],
        "sealed_blind_video_sha256s": artifact[
            "sealed_blind_video_sha256s"
        ],
        "phase_review_example_count": len(phase_reviews),
        "excluded_unresolved_target_count": len(phase_reviews) - len(rows),
    }


def attach_dense_reason_features(
    examples: Sequence[Mapping[str, Any]],
    *,
    reason_artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append label-free aggregate geometry to every dense temporal position."""

    reason = verify_reason_evidence_artifact(reason_artifact)
    reason_by_key = _index_rows(
        reason["examples"],
        field="reason evidence",
    )
    example_keys = {_event_key(row) for row in examples}
    if not example_keys.issubset(reason_by_key):
        raise ValueError(
            "reason evidence must exactly cover all dense temporal examples"
        )
    rows = []
    for row in examples:
        key = _event_key(row)
        reason_values = [
            float(value) for value in reason_by_key[key]["features"]
        ]
        if (
            len(reason_values) != len(REASON_FEATURE_NAMES)
            or not np.isfinite(reason_values).all()
        ):
            raise ValueError("dense temporal reason features are invalid")
        embeddings = [
            [*[float(value) for value in embedding], *reason_values]
            for embedding in row["embeddings"]
        ]
        rows.append({**row, "embeddings": embeddings})
    return rows, {
        "reason_artifact_sha256": reason["artifact_sha256"],
        "reason_feature_names": list(REASON_FEATURE_NAMES),
        "reason_feature_dimension": len(REASON_FEATURE_NAMES),
        "reason_source_scene_artifact_sha256": reason[
            "source_scene_artifact_sha256"
        ],
        "reason_pose_source_sha256s": list(reason["pose_source_sha256s"]),
    }


def screen_dense_causal_temporal_examples(
    *,
    examples: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    acceptance_threshold: float = 0.85,
) -> dict[str, Any]:
    """Select hyperparameters inside training games and score held games."""

    if not 0.5 <= float(acceptance_threshold) <= 1.0:
        raise ValueError("acceptance threshold must be between 0.5 and 1.0")
    configs = _normalize_configurations(configurations)
    arrays = _example_arrays(examples)
    groups = arrays["groups"]
    labels = arrays["labels"]
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 4:
        raise ValueError("dense temporal screening requires four games")
    for group in unique_groups:
        if set(labels[groups == group].tolist()) != {0, 1}:
            raise ValueError("every dense temporal game must contain both labels")

    predictions = np.zeros(len(labels), dtype=np.int64)
    probabilities = np.zeros(len(labels), dtype=np.float64)
    outer_folds = []
    for outer_index, held_group in enumerate(unique_groups):
        development_groups = [
            group for group in unique_groups if group != held_group
        ]
        selection = _select_configuration(
            arrays=arrays,
            configurations=configs,
            development_groups=development_groups,
            seed=1000 + outer_index * 100,
        )
        train_indexes = np.flatnonzero(
            np.isin(groups, development_groups)
        )
        held_indexes = np.flatnonzero(groups == held_group)
        held_probabilities = _train_predict(
            arrays=arrays,
            train_indexes=train_indexes,
            test_indexes=held_indexes,
            configuration=selection["configuration"],
            seed=2000 + outer_index,
        )
        held_predictions = (held_probabilities >= 0.5).astype(np.int64)
        probabilities[held_indexes] = held_probabilities
        predictions[held_indexes] = held_predictions
        outer_folds.append(
            {
                "held_target_group": held_group,
                "training_target_groups": development_groups,
                "selection_target_groups": development_groups,
                "selection": selection,
                "held_example_count": len(held_indexes),
                "metrics": _metrics(
                    labels[held_indexes],
                    held_predictions,
                    held_probabilities,
                ),
            }
        )

    metrics = _metrics(labels, predictions, probabilities)
    metrics["worst_game_balanced_accuracy"] = min(
        fold["metrics"]["balanced_accuracy"] for fold in outer_folds
    )
    accepted = (
        metrics["f1"] >= acceptance_threshold
        and metrics["worst_game_balanced_accuracy"]
        >= acceptance_threshold
    )
    prediction_rows = [
        {
            "phase_review_id": arrays["phase_review_ids"][index],
            "source_video_sha256": str(groups[index]),
            "event_id": arrays["event_ids"][index],
            "target": int(labels[index]),
            "prediction": int(predictions[index]),
            "probability": float(probabilities[index]),
        }
        for index in range(len(labels))
    ]
    artifact: dict[str, Any] = {
        "schema_version": DENSE_CAUSAL_TEMPORAL_SCREEN_SCHEMA,
        "purpose": "training_only_dense_causal_temporal_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "backbone": str(provenance.get("backbone") or ""),
        "backbone_sha256": _require_sha256(
            provenance.get("backbone_sha256"),
            field="backbone",
        ),
        "embedding_dimension": int(
            provenance.get("embedding_dimension", 0)
        ),
        "embedding_artifact_sha256": _require_sha256(
            provenance.get("embedding_artifact_sha256"),
            field="embedding artifact",
        ),
        "component_artifact_sha256s": _optional_hash_list(
            provenance.get("component_artifact_sha256s"),
            field="component artifact",
        ),
        "review_plan_sha256": _require_sha256(
            provenance.get("review_plan_sha256"),
            field="review plan",
        ),
        "sheet_manifest_sha256": _require_sha256(
            provenance.get("sheet_manifest_sha256"),
            field="sheet manifest",
        ),
        "label_correction_artifact_sha256s": _hash_list(
            provenance.get("label_correction_artifact_sha256s"),
            field="label correction artifact",
        ),
        "phase_review_artifact_sha256": _require_sha256(
            provenance.get("phase_review_artifact_sha256"),
            field="phase review artifact",
        ),
        "sealed_blind_video_sha256s": _hash_list(
            provenance.get("sealed_blind_video_sha256s"),
            field="sealed blind video",
        ),
        "phase_target_names": list(PHASE_TARGET_NAMES),
        "anchor_offsets_seconds": list(CAUSAL_PHASE_OFFSETS_SECONDS),
        "phase_review_example_count": int(
            provenance.get("phase_review_example_count", len(examples))
        ),
        "excluded_unresolved_target_count": int(
            provenance.get("excluded_unresolved_target_count", 0)
        ),
        "target_example_count": len(examples),
        "target_group_count": len(unique_groups),
        "protocol": {
            "outer_split": "leave_one_source_video_out",
            "inner_selection": "leave_one_remaining_source_video_out",
            "feature_standardization": "training_examples_only",
            "main_target": "field_goal_vs_free_throw",
            "phase_auxiliary_targets": list(PHASE_TARGET_NAMES),
            "decision_threshold": 0.5,
            "checkpoint_promotion": "none_unless_gate_passes",
        },
        "configurations": configs,
        "outer_folds": outer_folds,
        "predictions": prediction_rows,
        "metrics": metrics,
        "acceptance_gate": {
            "minimum_game_balanced_accuracy": float(
                acceptance_threshold
            ),
            "minimum_pooled_f1": float(acceptance_threshold),
        },
        "accepted": bool(accepted),
        "promotion_decision": (
            "eligible_for_checkpoint_training"
            if accepted
            else "rejected_below_cross_game_gate"
        ),
    }
    _validate_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_dense_causal_temporal_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("dense causal temporal screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _select_configuration(
    *,
    arrays: Mapping[str, Any],
    configurations: Sequence[dict[str, Any]],
    development_groups: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    groups = arrays["groups"]
    labels = arrays["labels"]
    candidates = []
    for config_index, configuration in enumerate(configurations):
        folds = []
        all_targets = []
        all_predictions = []
        all_probabilities = []
        for inner_index, held_group in enumerate(development_groups):
            training_groups = [
                group
                for group in development_groups
                if group != held_group
            ]
            train_indexes = np.flatnonzero(np.isin(groups, training_groups))
            held_indexes = np.flatnonzero(groups == held_group)
            probabilities = _train_predict(
                arrays=arrays,
                train_indexes=train_indexes,
                test_indexes=held_indexes,
                configuration=configuration,
                seed=seed + config_index * 10 + inner_index,
            )
            predictions = (probabilities >= 0.5).astype(np.int64)
            fold_metrics = _metrics(
                labels[held_indexes],
                predictions,
                probabilities,
            )
            folds.append(
                {
                    "held_selection_group": held_group,
                    "training_target_groups": training_groups,
                    "metrics": fold_metrics,
                }
            )
            all_targets.extend(labels[held_indexes].tolist())
            all_predictions.extend(predictions.tolist())
            all_probabilities.extend(probabilities.tolist())
        pooled = _metrics(
            np.asarray(all_targets, dtype=np.int64),
            np.asarray(all_predictions, dtype=np.int64),
            np.asarray(all_probabilities, dtype=np.float64),
        )
        worst = min(
            fold["metrics"]["balanced_accuracy"] for fold in folds
        )
        candidates.append(
            {
                "configuration": configuration,
                "folds": folds,
                "metrics": {
                    **pooled,
                    "worst_game_balanced_accuracy": worst,
                },
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["metrics"]["worst_game_balanced_accuracy"],
            row["metrics"]["balanced_accuracy"],
            row["metrics"]["f1"],
            -row["configuration"]["hidden_dimension"],
            -row["configuration"]["kernel_size"],
        ),
    )
    return {
        "configuration": selected["configuration"],
        "folds": selected["folds"],
        "metrics": selected["metrics"],
        "candidate_metrics": [
            {
                "configuration": row["configuration"],
                "metrics": row["metrics"],
            }
            for row in candidates
        ],
    }


def _train_predict(
    *,
    arrays: Mapping[str, Any],
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
    configuration: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    embeddings = arrays["embeddings"]
    train_values = embeddings[train_indexes]
    mean = train_values.mean(axis=(0, 1), keepdims=True)
    std = train_values.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = torch.from_numpy(
        ((train_values - mean) / std).astype(np.float32)
    )
    test_x = torch.from_numpy(
        ((embeddings[test_indexes] - mean) / std).astype(np.float32)
    )
    train_y = torch.from_numpy(
        arrays["labels"][train_indexes].astype(np.int64)
    )
    train_phase = torch.from_numpy(
        arrays["phase_targets"][train_indexes].astype(np.float32)
    )
    model = DenseCausalTemporalClassifier(
        embedding_dimension=embeddings.shape[2],
        hidden_dimension=int(configuration["hidden_dimension"]),
        kernel_size=int(configuration["kernel_size"]),
        temporal_operator=str(configuration["temporal_operator"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    counts = torch.bincount(train_y, minlength=2).to(torch.float32)
    class_weights = counts.sum() / (counts.clamp_min(1.0) * 2.0)
    positives = train_phase.sum(dim=0)
    negatives = len(train_phase) - positives
    phase_pos_weight = (negatives / positives.clamp_min(1.0)).clamp(
        0.25,
        4.0,
    )
    model.train()
    for _ in range(int(configuration["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        main_logits, phase_logits = model(train_x)
        loss = F.cross_entropy(
            main_logits,
            train_y,
            weight=class_weights,
        )
        auxiliary_weight = float(configuration["auxiliary_weight"])
        if auxiliary_weight:
            loss = loss + auxiliary_weight * F.binary_cross_entropy_with_logits(
                phase_logits,
                train_phase,
                pos_weight=phase_pos_weight,
            )
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits, _ = model(test_x)
        return (
            torch.softmax(logits, dim=1)[:, 1]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def _example_arrays(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("dense temporal examples are required")
    ordered = sorted(
        examples,
        key=lambda row: (
            str(row.get("source_video_sha256") or ""),
            str(row.get("event_id") or ""),
        ),
    )
    embedding_dimension = len(ordered[0].get("embeddings", [[]])[0])
    embeddings = []
    labels = []
    phases = []
    groups = []
    event_ids = []
    review_ids = []
    seen = set()
    for row in ordered:
        key = _event_key(row)
        if key in seen:
            raise ValueError("duplicate dense temporal example")
        seen.add(key)
        values = np.asarray(row.get("embeddings"), dtype=np.float32)
        if values.shape != (
            len(CAUSAL_PHASE_OFFSETS_SECONDS),
            embedding_dimension,
        ) or not np.isfinite(values).all():
            raise ValueError("invalid dense temporal embedding shape")
        label = str(row.get("label") or "")
        if label not in VISUAL_STATE_LABELS:
            raise ValueError("invalid dense temporal target")
        targets = row.get("phase_targets")
        if not isinstance(targets, Mapping) or set(targets) != set(
            PHASE_TARGET_NAMES
        ):
            raise ValueError("dense temporal phase targets are incomplete")
        phase_values = [int(targets[name]) for name in PHASE_TARGET_NAMES]
        if any(value not in {0, 1} for value in phase_values):
            raise ValueError("dense temporal phase targets must be binary")
        embeddings.append(values)
        labels.append(1 if label == "free_throw" else 0)
        phases.append(phase_values)
        groups.append(key[0])
        event_ids.append(key[2])
        review_ids.append(str(row.get("phase_review_id") or ""))
    return {
        "embeddings": np.stack(embeddings),
        "labels": np.asarray(labels, dtype=np.int64),
        "phase_targets": np.asarray(phases, dtype=np.float32),
        "groups": np.asarray(groups, dtype=object),
        "event_ids": event_ids,
        "phase_review_ids": review_ids,
    }


def _normalize_configurations(
    configurations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for value in configurations:
        row = {
            "temporal_operator": str(
                value.get("temporal_operator", "conv")
            ),
            "hidden_dimension": int(value.get("hidden_dimension", 0)),
            "kernel_size": int(value.get("kernel_size", 0)),
            "auxiliary_weight": float(value.get("auxiliary_weight", -1.0)),
            "epochs": int(value.get("epochs", 0)),
            "learning_rate": float(value.get("learning_rate", 0.0)),
            "weight_decay": float(value.get("weight_decay", -1.0)),
        }
        if (
            row["temporal_operator"] not in {"conv", "gru", "gated"}
            or not 1 <= row["hidden_dimension"] <= 512
            or row["kernel_size"] not in {1, 3, 5, 7}
            or not 0.0 <= row["auxiliary_weight"] <= 10.0
            or not 1 <= row["epochs"] <= 1000
            or not 0.0 < row["learning_rate"] <= 1.0
            or not 0.0 <= row["weight_decay"] <= 1.0
        ):
            raise ValueError("invalid dense temporal configuration")
        normalized.append(row)
    if not normalized:
        raise ValueError("dense temporal configurations are required")
    canonical = {_canonical_sha256(row) for row in normalized}
    if len(canonical) != len(normalized):
        raise ValueError("duplicate dense temporal configuration")
    return normalized


def _validate_embedding_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != DENSE_CAUSAL_FRAME_EMBEDDING_SCHEMA:
        raise ValueError("unsupported dense causal embedding schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("dense causal embeddings must remain training-only")
    _require_sha256(artifact.get("review_plan_sha256"), field="review plan")
    _require_sha256(
        artifact.get("sheet_manifest_sha256"),
        field="sheet manifest",
    )
    _require_sha256(artifact.get("backbone_sha256"), field="backbone")
    dimension = int(artifact.get("embedding_dimension", 0))
    if dimension < 1 or not str(artifact.get("backbone") or ""):
        raise ValueError("dense causal embedding backbone is invalid")
    if tuple(artifact.get("anchor_offsets_seconds") or ()) != (
        CAUSAL_PHASE_OFFSETS_SECONDS
    ):
        raise ValueError("dense causal offsets do not match review protocol")
    sources = set(
        _hash_list(
            artifact.get("source_video_sha256s"),
            field="source video",
        )
    )
    blind = set(
        _hash_list(
            artifact.get("sealed_blind_video_sha256s"),
            field="sealed blind video",
        )
    )
    if sources & blind:
        raise ValueError("sealed blind source entered dense causal embeddings")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("dense causal embedding examples are required")
    keys = set()
    review_ids = set()
    observed_sources = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("invalid dense causal embedding row")
        key = _event_key(row)
        review_id = str(row.get("phase_review_id") or "")
        values = np.asarray(row.get("embeddings"), dtype=np.float32)
        indexes = row.get("frame_indexes")
        if (
            key in keys
            or not review_id
            or review_id in review_ids
            or not isinstance(indexes, list)
            or len(indexes) != len(CAUSAL_PHASE_OFFSETS_SECONDS)
            or values.shape
            != (len(CAUSAL_PHASE_OFFSETS_SECONDS), dimension)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid or duplicate dense causal embedding")
        keys.add(key)
        review_ids.add(review_id)
        observed_sources.add(key[0])
    if observed_sources != sources:
        raise ValueError("dense causal source coverage mismatch")


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != DENSE_CAUSAL_TEMPORAL_SCREEN_SCHEMA:
        raise ValueError("unsupported dense causal temporal screen schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("dense causal temporal screen must be training-only")
    for field in (
        "backbone_sha256",
        "embedding_artifact_sha256",
        "review_plan_sha256",
        "sheet_manifest_sha256",
        "phase_review_artifact_sha256",
    ):
        _require_sha256(artifact.get(field), field=field)
    _hash_list(
        artifact.get("label_correction_artifact_sha256s"),
        field="label correction artifact",
    )
    _optional_hash_list(
        artifact.get("component_artifact_sha256s"),
        field="component artifact",
    )
    _hash_list(
        artifact.get("sealed_blind_video_sha256s"),
        field="sealed blind video",
    )
    if tuple(artifact.get("anchor_offsets_seconds") or ()) != (
        CAUSAL_PHASE_OFFSETS_SECONDS
    ):
        raise ValueError("dense causal temporal offsets are invalid")
    predictions = artifact.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("dense causal temporal predictions are required")
    targets = np.asarray(
        [int(row["target"]) for row in predictions],
        dtype=np.int64,
    )
    predicted = np.asarray(
        [int(row["prediction"]) for row in predictions],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [float(row["probability"]) for row in predictions],
        dtype=np.float64,
    )
    if (
        any(value not in {0, 1} for value in targets.tolist())
        or any(value not in {0, 1} for value in predicted.tolist())
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("dense causal temporal predictions are invalid")
    expected = _metrics(targets, predicted, probabilities)
    folds = artifact.get("outer_folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("dense causal outer folds are required")
    prediction_groups = {
        str(row["source_video_sha256"]) for row in predictions
    }
    held_groups = {str(fold["held_target_group"]) for fold in folds}
    if held_groups != prediction_groups or len(held_groups) != len(folds):
        raise ValueError("dense causal outer fold coverage is inconsistent")
    for fold in folds:
        held_group = str(fold["held_target_group"])
        training_groups = {
            str(value) for value in fold["training_target_groups"]
        }
        selection_groups = {
            str(value) for value in fold["selection_target_groups"]
        }
        if (
            held_group in training_groups
            or held_group in selection_groups
            or training_groups != prediction_groups - {held_group}
            or selection_groups != training_groups
        ):
            raise ValueError("dense causal outer fold leaks held game")
        selection = fold.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("dense causal selection provenance is required")
        for inner in selection.get("folds", []):
            inner_held = str(inner["held_selection_group"])
            inner_training = {
                str(value) for value in inner["training_target_groups"]
            }
            if (
                inner_held not in selection_groups
                or inner_held in inner_training
                or inner_training != selection_groups - {inner_held}
            ):
                raise ValueError("dense causal inner fold leaks held game")
        indexes = [
            index
            for index, row in enumerate(predictions)
            if row["source_video_sha256"] == held_group
        ]
        fold_expected = _metrics(
            targets[indexes],
            predicted[indexes],
            probabilities[indexes],
        )
        if any(
            not np.isclose(
                float(fold["metrics"].get(key, -1.0)),
                float(value),
            )
            for key, value in fold_expected.items()
        ):
            raise ValueError("dense causal outer fold metrics are inconsistent")
    expected["worst_game_balanced_accuracy"] = min(
        float(fold["metrics"]["balanced_accuracy"]) for fold in folds
    )
    metrics = artifact.get("metrics")
    if not isinstance(metrics, Mapping) or any(
        not np.isclose(float(metrics.get(key, -1.0)), float(value))
        for key, value in expected.items()
    ):
        raise ValueError("dense causal temporal metrics are inconsistent")
    gate = artifact.get("acceptance_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("dense causal temporal acceptance gate is required")
    accepted = (
        expected["f1"] >= float(gate["minimum_pooled_f1"])
        and expected["worst_game_balanced_accuracy"]
        >= float(gate["minimum_game_balanced_accuracy"])
    )
    if bool(artifact.get("accepted")) != accepted:
        raise ValueError("dense causal temporal acceptance is inconsistent")
    expected_decision = (
        "eligible_for_checkpoint_training"
        if accepted
        else "rejected_below_cross_game_gate"
    )
    if artifact.get("promotion_decision") != expected_decision:
        raise ValueError("dense causal temporal promotion decision is inconsistent")


def _phase_targets(review: Mapping[str, Any]) -> dict[str, int]:
    return {
        "formation_free_throw": int(
            review.get("formation_state") == "free_throw_setup"
        ),
        "formation_live_play": int(
            review.get("formation_state") == "live_play"
        ),
        "context_replay": int(
            review.get("broadcast_context") == "replay"
        ),
        "context_live_action": int(
            review.get("broadcast_context") == "live_action"
        ),
        "sequence_complete": int(
            review.get("shot_sequence") == "complete"
        ),
        "sequence_not_a_shot": int(
            review.get("shot_sequence") == "not_a_shot"
        ),
        "release_visible": int(
            review.get("release_position") is not None
        ),
        "rim_visible": int(
            review.get("rim_arrival_position") is not None
        ),
    }


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        key = _event_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {field}")
        indexed[key] = row
    return indexed


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = _require_sha256(
        row.get("source_video_sha256"),
        field="source video",
    )
    bundle = _require_sha256(
        row.get("candidate_bundle_sha256"),
        field="candidate bundle",
    )
    event_id = str(row.get("event_id") or "")
    if not event_id:
        raise ValueError("dense causal event ID is required")
    return source, bundle, event_id


def _hash_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} hashes are required")
    hashes = [_require_sha256(item, field=field) for item in value]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"duplicate {field} hash")
    return hashes


def _optional_hash_list(value: object, *, field: str) -> list[str]:
    if value is None or value == [] or value == ():
        return []
    return _hash_list(value, field=field)


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
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
