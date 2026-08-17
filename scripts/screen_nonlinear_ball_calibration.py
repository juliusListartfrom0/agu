#!/usr/bin/env python3
"""Screen nonlinear detector/track calibration on a frozen cross-game split.

This is an offline research screen.  It consumes sealed perception, review-plan
and review artifacts, never decodes a video, and never writes a checkpoint.
Target labels are opened only after every source-only fit and threshold is
fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.ball_candidate_calibration import (
    CALIBRATION_FEATURE_NAMES,
    build_nonlinear_track_features,
)
from app.analysis.ball_candidate_review import verify_artifact
from app.analysis.ball_candidate_verifier import (
    ball_track_features,
    choose_recall_threshold,
)
from scripts.screen_same_detector_ball_verifier import (
    _determinate_source_rows,
    _external_metrics,
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_artifact(path: Path) -> dict[str, Any]:
    return verify_artifact(json.loads(path.read_text()))


def _classifier(name: str, *, tree_count: int) -> Any:
    if name == "nonlinear_extra_trees":
        return ExtraTreesClassifier(
            n_estimators=tree_count,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=0,
            n_jobs=1,
        )
    if name == "nonlinear_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            ),
        )
    raise ValueError(f"unknown calibration variant: {name}")


def _fit_variant(
    name: str,
    source_x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    target_x: np.ndarray,
    *,
    recall_floor: float,
    tree_count: int,
) -> tuple[dict[str, Any], np.ndarray]:
    split_count = min(5, len(set(groups)))
    if split_count < 2:
        raise ValueError("grouped calibration requires at least two groups")
    splitter = GroupKFold(n_splits=split_count)
    oof = np.zeros(len(labels), dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(
        splitter.split(source_x, labels, groups),
        1,
    ):
        model = _classifier(name, tree_count=tree_count)
        model.fit(source_x[train], labels[train])
        oof[test] = model.predict_proba(source_x[test])[:, 1]
        folds.append(
            {
                "fold": fold,
                "examples": int(len(test)),
                "groups": int(len(set(groups[test]))),
                "average_precision": round(
                    float(average_precision_score(labels[test], oof[test])),
                    6,
                ),
            }
        )
    selection = choose_recall_threshold(
        labels,
        oof,
        recall_floor=recall_floor,
    )
    model = _classifier(name, tree_count=tree_count)
    model.fit(source_x, labels)
    return (
        {
            "classifier": name,
            "folds": folds,
            "oof_average_precision": round(
                float(average_precision_score(labels, oof)),
                6,
            ),
            "selection": {
                key: round(float(value), 8)
                for key, value in selection.items()
            },
            "checkpoint_saved": False,
        },
        model.predict_proba(target_x)[:, 1],
    )


def _feature_rows(
    perception: dict[str, Any],
    examples: list[dict[str, Any]] | None = None,
) -> tuple[tuple[str, ...], dict[str, np.ndarray], list[dict[str, Any]]]:
    raw_names, raw_by_id = ball_track_features(perception)
    names, by_id = build_nonlinear_track_features(raw_names, raw_by_id)
    rows = list(perception["detections"])
    if examples is not None:
        wanted = {str(row["detection_id"]) for row in examples}
        rows = [row for row in rows if str(row["detection_id"]) in wanted]
    rows.sort(key=lambda row: (int(row["frame"]), str(row["detection_id"])))
    if not rows:
        raise ValueError("feature selection produced no detections")
    return names, by_id, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-bundle",
        action="append",
        nargs=3,
        type=Path,
        metavar=("PERCEPTION", "PLAN", "REVIEW"),
        required=True,
    )
    parser.add_argument("--target-perception", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--target-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.85)
    parser.add_argument("--tree-count", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tree_count <= 0:
        raise ValueError("tree count must be positive")
    source_feature_parts: list[np.ndarray] = []
    source_labels: list[np.ndarray] = []
    source_groups: list[np.ndarray] = []
    source_summaries: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    feature_names: tuple[str, ...] | None = None

    for perception_path, plan_path, review_path in args.source_bundle:
        perception = _load_artifact(perception_path)
        plan = _load_artifact(plan_path)
        review = _load_artifact(review_path)
        examples = _determinate_source_rows(perception, plan, review)
        names, by_id, _ = _feature_rows(perception, examples)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("source feature contracts differ")
        source_feature_parts.append(
            np.stack([by_id[str(row["detection_id"])] for row in examples])
        )
        source_labels.append(
            np.asarray([int(row["label"]) for row in examples], dtype=np.int64)
        )
        source_groups.append(
            np.asarray([str(row["group"]) for row in examples])
        )
        source_hash = str(perception["raw_video"]["sha256"])
        source_hashes.append(source_hash)
        labels = source_labels[-1]
        source_summaries.append(
            {
                "perception_sha256": perception["artifact_sha256"],
                "plan_sha256": plan["artifact_sha256"],
                "review_sha256": review["artifact_sha256"],
                "raw_video_sha256": source_hash,
                "determinate_examples": len(examples),
                "positive": int(labels.sum()),
                "negative": int((labels == 0).sum()),
                "causal_window_groups": len(set(source_groups[-1])),
            }
        )
    if len(source_hashes) < 2 or len(set(source_hashes)) != len(source_hashes):
        raise ValueError("source games must be mutually disjoint")

    target_perception = _load_artifact(args.target_perception)
    target_plan = _load_artifact(args.target_plan)
    target_hash = str(target_perception["raw_video"]["sha256"])
    if target_hash in source_hashes:
        raise ValueError("source and target games must be disjoint")
    target_names, target_by_id, target_rows = _feature_rows(target_perception)
    if target_names != feature_names:
        raise ValueError("source and target feature contracts differ")

    source_x = np.concatenate(source_feature_parts)
    labels = np.concatenate(source_labels)
    groups = np.concatenate(source_groups)
    target_x = np.stack([target_by_id[str(row["detection_id"])] for row in target_rows])
    if set(labels) != {0, 1}:
        raise ValueError("source calibration requires both labels")

    variants: dict[str, Any] = {}
    accepted = False
    for name in ("nonlinear_extra_trees", "nonlinear_logistic"):
        source_evaluation, target_scores = _fit_variant(
            name,
            source_x,
            labels,
            groups,
            target_x,
            recall_floor=args.recall_floor,
            tree_count=args.tree_count,
        )
        # The target review is intentionally opened only after all source-only
        # fitting and threshold selection has completed for this variant.
        target_review = _load_artifact(args.target_review)
        external = _external_metrics(
            target_rows=target_rows,
            target_scores=target_scores,
            plan=target_plan,
            review=target_review,
            threshold=float(source_evaluation["selection"]["threshold"]),
        )
        lower = external["metrics"]["lower"]
        variant_accepted = bool(
            lower["precision"] >= 0.85 and lower["recall"] >= 0.85
        )
        accepted |= variant_accepted
        variants[name] = {
            "source_evaluation": source_evaluation,
            "external_review": external,
            "accepted": variant_accepted,
        }

    artifact: dict[str, Any] = {
        "schema_version": "agu.nonlinear-ball-calibration-screen.v1",
        "purpose": "offline_research_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "pixel_decode": False,
        "source": {
            "games": source_summaries,
            "determinate_examples": int(len(labels)),
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
            "causal_window_groups": int(len(set(groups))),
        },
        "target": {
            "perception_sha256": target_perception["artifact_sha256"],
            "plan_sha256": target_plan["artifact_sha256"],
            "review_sha256": target_review["artifact_sha256"],
            "raw_video_sha256": target_hash,
        },
        "model": {
            "feature_contract": "ball-track+nonlinear-interactions.v1",
            "feature_names": list(CALIBRATION_FEATURE_NAMES),
            "tree_count": int(args.tree_count),
            "checkpoint_saved": False,
        },
        "target_review_opened_after_fit": True,
        "variants": variants,
        "accepted": accepted,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
