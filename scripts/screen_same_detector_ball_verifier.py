#!/usr/bin/env python3
"""Train on one reviewed RF-DETR game and evaluate on another sealed game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.ball_candidate_review import verify_artifact
from app.analysis.ball_candidate_verifier import (
    ball_track_features,
    choose_recall_threshold,
    stratified_review_metrics,
)
from scripts.screen_ball_candidate_verifier import (
    MobileNetEmbedder,
    _broadcast_features,
    _canonical_sha256,
    _with_confidence,
)


def _window_group(frame: int, windows: list[dict[str, Any]]) -> str:
    matches = [
        f"window-{index:03d}"
        for index, window in enumerate(windows, 1)
        if int(window["start_frame"]) <= frame <= int(window["end_frame"])
    ]
    if len(matches) != 1:
        raise ValueError(f"frame {frame} does not map to exactly one causal window")
    return matches[0]


def _validate_bundle(
    perception: dict[str, Any],
    plan: dict[str, Any],
    review: dict[str, Any],
) -> None:
    if plan["perception_artifact_sha256"] != perception["artifact_sha256"]:
        raise ValueError("plan/perception binding mismatch")
    if review["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("review/plan binding mismatch")
    if (
        review.get("runtime_consumable") is not False
        or review.get("codex_runtime_answer_used", False) is not False
    ):
        raise ValueError("review is not offline-only")


def _determinate_source_rows(
    perception: dict[str, Any],
    plan: dict[str, Any],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only unambiguous reviewed candidates with game-scoped groups."""
    _validate_bundle(perception, plan, review)
    decisions = {
        str(row["candidate_id"]): str(row["decision"])
        for row in review["decisions"]
    }
    game_key = str(perception["raw_video"]["sha256"])
    rows: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        decision = decisions.get(candidate_id)
        if decision not in {"valid_ball", "false_positive"}:
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "detection_id": str(candidate["detection_id"]),
                "label": int(decision == "valid_ball"),
                "confidence": float(candidate["confidence"]),
                "group": (
                    f"{game_key}:"
                    f"{_window_group(int(candidate['frame']), perception['sampling']['windows'])}"
                ),
            }
        )
    if not rows:
        raise ValueError("source review has no determinate candidates")
    return rows


def _classifier(c_value: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value, class_weight="balanced", max_iter=2000, random_state=0
        ),
    )


def _fit_variant(
    source_x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    target_x: np.ndarray,
    *,
    recall_floor: float,
) -> tuple[dict[str, Any], np.ndarray]:
    splitter = GroupKFold(n_splits=min(5, len(set(groups))))
    candidates: list[tuple[float, float, float, np.ndarray, list[dict[str, float]]]] = []
    for c_value in (0.01, 0.1, 1.0):
        oof = np.zeros(len(labels), dtype=np.float64)
        folds: list[dict[str, float]] = []
        for fold, (train, test) in enumerate(
            splitter.split(source_x, labels, groups), 1
        ):
            model = _classifier(c_value).fit(source_x[train], labels[train])
            oof[test] = model.predict_proba(source_x[test])[:, 1]
            folds.append(
                {
                    "fold": fold,
                    "average_precision": round(
                        float(average_precision_score(labels[test], oof[test])), 6
                    ),
                }
            )
        selection = choose_recall_threshold(
            labels, oof, recall_floor=recall_floor
        )
        candidates.append(
            (
                selection["precision"],
                selection["recall"],
                -c_value,
                oof,
                folds,
            )
        )
    precision, recall, negative_c, oof, folds = max(candidates, key=lambda row: row[:3])
    c_value = -negative_c
    selection = choose_recall_threshold(labels, oof, recall_floor=recall_floor)
    model = _classifier(c_value).fit(source_x, labels)
    target_scores = model.predict_proba(target_x)[:, 1]
    return (
        {
            "c": c_value,
            "folds": folds,
            "oof_average_precision": round(
                float(average_precision_score(labels, oof)), 6
            ),
            "selection": {
                "threshold": round(selection["threshold"], 8),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
            },
        },
        target_scores,
    )


def _external_metrics(
    *,
    target_rows: list[dict[str, Any]],
    target_scores: np.ndarray,
    plan: dict[str, Any],
    review: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    score_by_id = {
        str(row["detection_id"]): float(score)
        for row, score in zip(target_rows, target_scores, strict=True)
    }
    bands = [tuple(float(value) for value in band) for band in plan["sampling"]["bands"]]
    names = [f"{low:.2f}-{high:.2f}" for low, high in bands]
    populations = {name: 0 for name in names}
    for row in target_rows:
        confidence = float(row["confidence"])
        for name, (low, high) in zip(names, bands, strict=True):
            if low <= confidence < high:
                populations[name] += 1
                break
    decisions = {row["candidate_id"]: row["decision"] for row in review["decisions"]}
    bounds: dict[str, list[dict[str, object]]] = {"lower": [], "upper": []}
    output_rows: list[dict[str, object]] = []
    for row in plan["candidates"]:
        decision = decisions[row["candidate_id"]]
        score = score_by_id[str(row["detection_id"])]
        kept = score >= threshold
        band = f"{float(row['confidence_band'][0]):.2f}-{float(row['confidence_band'][1]):.2f}"
        output_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "decision": decision,
                "score": round(score, 8),
                "kept": bool(kept),
            }
        )
        for bound, uncertain_label in (("lower", 0), ("upper", 1)):
            label = (
                1
                if decision == "valid_ball"
                else uncertain_label
                if decision == "uncertain"
                else 0
            )
            bounds[bound].append({"band": band, "label": label, "kept": kept})
    return {
        "population_by_band": populations,
        "metrics": {
            bound: {
                key: round(float(value), 6)
                for key, value in stratified_review_metrics(
                    rows, band_population=populations
                ).items()
            }
            for bound, rows in bounds.items()
        },
        "rows": output_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for prefix in ("target",):
        parser.add_argument(f"--{prefix}-perception", type=Path, required=True)
        parser.add_argument(f"--{prefix}-plan", type=Path, required=True)
        parser.add_argument(f"--{prefix}-review", type=Path, required=True)
        parser.add_argument(f"--{prefix}-video", type=Path, required=True)
    parser.add_argument(
        "--source-bundle",
        action="append",
        nargs=4,
        type=Path,
        metavar=("PERCEPTION", "PLAN", "REVIEW", "VIDEO"),
        required=True,
        help="Repeat for each disjoint reviewed training game.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--recall-floor", type=float, default=0.85)
    parser.add_argument("--context-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch threads must be positive")
    import torch

    torch.set_num_threads(args.torch_threads)
    source_bundles: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]
    ] = []
    for perception_path, plan_path, review_path, video_path in args.source_bundle:
        source_bundles.append(
            (
                verify_artifact(json.loads(perception_path.read_text())),
                verify_artifact(json.loads(plan_path.read_text())),
                verify_artifact(json.loads(review_path.read_text())),
                video_path,
            )
        )
    target_perception = verify_artifact(json.loads(args.target_perception.read_text()))
    target_plan = verify_artifact(json.loads(args.target_plan.read_text()))
    target_review = verify_artifact(json.loads(args.target_review.read_text()))
    _validate_bundle(target_perception, target_plan, target_review)
    source_hashes = [
        str(perception["raw_video"]["sha256"])
        for perception, _, _, _ in source_bundles
    ]
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("source games must be mutually disjoint")
    if str(target_perception["raw_video"]["sha256"]) in source_hashes:
        raise ValueError("source and target games must be disjoint")

    embedder = MobileNetEmbedder(device=args.device, batch_size=args.batch_size)
    source_feature_parts: list[np.ndarray] = []
    source_geometry_parts: list[np.ndarray] = []
    source_confidence_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    source_summaries: list[dict[str, Any]] = []
    for perception, plan, review, video in source_bundles:
        examples = _determinate_source_rows(perception, plan, review)
        detection_ids = {str(row["detection_id"]) for row in examples}
        source_x, source_rows = _broadcast_features(
            perception,
            video=video,
            embedder=embedder,
            context_scale=args.context_scale,
            detection_ids=detection_ids,
        )
        feature_index = {
            str(row["detection_id"]): index
            for index, row in enumerate(source_rows)
        }
        geometry_names, geometry_by_id = ball_track_features(perception)
        order = np.asarray(
            [feature_index[str(row["detection_id"])] for row in examples],
            dtype=np.int64,
        )
        source_feature_parts.append(source_x[order])
        source_geometry_parts.append(
            np.stack(
                [geometry_by_id[str(row["detection_id"])] for row in examples]
            )
        )
        source_confidence_parts.append(
            np.asarray([float(row["confidence"]) for row in examples])
        )
        label_parts.append(
            np.asarray([int(row["label"]) for row in examples], dtype=np.int64)
        )
        group_parts.append(np.asarray([str(row["group"]) for row in examples]))
        labels_for_source = label_parts[-1]
        source_summaries.append(
            {
                "perception_sha256": perception["artifact_sha256"],
                "plan_sha256": plan["artifact_sha256"],
                "review_sha256": review["artifact_sha256"],
                "raw_video_sha256": perception["raw_video"]["sha256"],
                "determinate_examples": len(examples),
                "positive": int(labels_for_source.sum()),
                "negative": int((labels_for_source == 0).sum()),
                "causal_window_groups": len(set(group_parts[-1])),
            }
        )
    source_all_x = np.concatenate(source_feature_parts)
    source_geometry = np.concatenate(source_geometry_parts)
    source_confidence = np.concatenate(source_confidence_parts)
    labels = np.concatenate(label_parts)
    groups = np.concatenate(group_parts)
    if len(set(labels)) != 2:
        raise ValueError("combined sources require both positive and negative labels")
    if len(set(groups)) < 2:
        raise ValueError("combined sources require at least two causal-window groups")
    target_x, target_rows = _broadcast_features(
        target_perception,
        video=args.target_video,
        embedder=embedder,
        context_scale=args.context_scale,
    )
    target_confidence = np.asarray([float(row["confidence"]) for row in target_rows])
    target_geometry_names, target_geometry_by_id = ball_track_features(
        target_perception
    )
    if target_geometry_names != geometry_names:
        raise ValueError("source and target track-feature contracts differ")
    target_geometry = np.stack(
        [target_geometry_by_id[str(row["detection_id"])] for row in target_rows]
    )
    variants: dict[str, Any] = {}
    accepted = False
    for name, train_x, test_x in (
        ("visual_only", source_all_x, target_x),
        (
            "visual_plus_confidence",
            _with_confidence(source_all_x, source_confidence),
            _with_confidence(target_x, target_confidence),
        ),
        ("track_geometry", source_geometry, target_geometry),
        (
            "visual_plus_track_geometry",
            np.column_stack([source_all_x, source_geometry]),
            np.column_stack([target_x, target_geometry]),
        ),
    ):
        source_evaluation, scores = _fit_variant(
            train_x, labels, groups, test_x, recall_floor=args.recall_floor
        )
        external = _external_metrics(
            target_rows=target_rows,
            target_scores=scores,
            plan=target_plan,
            review=target_review,
            threshold=float(source_evaluation["selection"]["threshold"]),
        )
        variant_accepted = bool(
            external["metrics"]["lower"]["precision"] >= 0.85
            and external["metrics"]["lower"]["recall"] >= 0.85
        )
        accepted |= variant_accepted
        variants[name] = {
            "source_evaluation": source_evaluation,
            "external_review": external,
            "accepted": variant_accepted,
        }
    artifact: dict[str, Any] = {
        "schema_version": "agu.same-detector-ball-verifier-screen.v2",
        "purpose": "offline_research_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source": {
            "games": source_summaries,
            "determinate_examples": len(labels),
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
            "causal_window_groups": len(set(groups)),
        },
        "target": {
            "perception_sha256": target_perception["artifact_sha256"],
            "plan_sha256": target_plan["artifact_sha256"],
            "review_sha256": target_review["artifact_sha256"],
        },
        "model": {
            "backbone": "MobileNet_V3_Small_Weights.DEFAULT",
            "context_scale": args.context_scale,
            "track_geometry_features": list(geometry_names),
            "checkpoint_saved": False,
        },
        "variants": variants,
        "accepted": accepted,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": accepted,
                "metrics": {
                    name: value["external_review"]["metrics"]
                    for name, value in variants.items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
