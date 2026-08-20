#!/usr/bin/env python3
"""Screen fixed BARD visual-state features on held-out AGU development games."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.bard_visual_state import (  # noqa: E402
    verify_bard_visual_state_embedding_artifact,
)
from app.analysis.pbp_visual_state_frames import (  # noqa: E402
    verify_anchor_state_embedding_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bard-embeddings", type=Path, required=True)
    parser.add_argument("--target-embeddings", type=Path, required=True)
    parser.add_argument("--label-corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bard = verify_bard_visual_state_embedding_artifact(
        json.loads(args.bard_embeddings.read_text(encoding="utf-8"))
    )
    target = verify_anchor_state_embedding_artifact(
        json.loads(args.target_embeddings.read_text(encoding="utf-8"))
    )
    corrections = json.loads(args.label_corrections.read_text(encoding="utf-8"))
    if (
        corrections.get("schema_version")
        != "agu.pbp-visual-state-label-corrections.v1"
        or corrections.get("runtime_consumable") is not False
        or corrections.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("invalid visual-state label corrections")

    source_x, source_y, source_groups = _source_rows(bard)
    source_folds = []
    for train, test in LeaveOneGroupOut().split(
        source_x,
        source_y,
        source_groups,
    ):
        model = _source_model()
        model.fit(source_x[train], source_y[train])
        scores = model.decision_function(source_x[test])
        source_folds.append(
            {
                "pair_id": str(source_groups[test][0]),
                "examples": int(len(test)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(source_y[test], scores >= 0.0)
                ),
                "roc_auc": float(roc_auc_score(source_y[test], scores)),
            }
        )

    source_model = _source_model()
    source_model.fit(source_x, source_y)
    correction_by_key = {
        (str(row["source_video_sha256"]), str(row["event_id"])): str(
            row["corrected_state"]
        )
        for row in corrections["decisions"]
        if row.get("corrected_state") in {"free_throw", "field_goal"}
    }
    target_x = []
    target_y = []
    target_groups = []
    for row in target["examples"]:
        label = correction_by_key.get(
            (
                str(row["source_video_sha256"]),
                str(row["event_id"]),
            )
        )
        if label is None:
            continue
        target_x.append(
            source_model.decision_function(np.asarray(row["embeddings"]))
        )
        target_y.append(label == "free_throw")
        target_groups.append(str(row["source_video_filename"]))
    target_x_array = np.asarray(target_x)
    target_y_array = np.asarray(target_y, dtype=np.int64)
    target_group_array = np.asarray(target_groups)
    target_folds = []
    for game in sorted(set(target_groups)):
        test = target_group_array == game
        train = ~test
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ),
        )
        model.fit(target_x_array[train], target_y_array[train])
        scores = model.decision_function(target_x_array[test])
        row: dict[str, Any] = {
            "held_out_game": game,
            "examples": int(test.sum()),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    target_y_array[test],
                    scores >= 0.0,
                )
            ),
        }
        if len(set(target_y_array[test])) == 2:
            row["roc_auc"] = float(
                roc_auc_score(target_y_array[test], scores)
            )
        target_folds.append(row)

    target_scores = [row["balanced_accuracy"] for row in target_folds]
    artifact = {
        "schema_version": "agu.bard-visual-state-transfer-screen.v1",
        "purpose": "offline_candidate_feature_acceptance_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "bard_embedding_artifact_sha256": bard["artifact_sha256"],
        "target_embedding_artifact_sha256": target["artifact_sha256"],
        "label_correction_artifact_sha256": corrections["artifact_sha256"],
        "protocol": {
            "source": "leave-one-pair-out StandardScaler PCA8 whiten SVC-C1",
            "target": (
                "leave-one-development-game-out StandardScaler "
                "balanced-logistic-C1 over five fixed source margins"
            ),
            "acceptance_metric": "worst_held_out_game_balanced_accuracy",
            "acceptance_threshold": 0.85,
        },
        "source_sanity": {
            "folds": source_folds,
            "mean_balanced_accuracy": float(
                np.mean([row["balanced_accuracy"] for row in source_folds])
            ),
            "worst_pair_balanced_accuracy": float(
                np.min([row["balanced_accuracy"] for row in source_folds])
            ),
            "mean_roc_auc": float(
                np.mean([row["roc_auc"] for row in source_folds])
            ),
        },
        "target_transfer": {
            "resolved_examples": len(target_y),
            "folds": target_folds,
            "mean_balanced_accuracy": float(np.mean(target_scores)),
            "worst_game_balanced_accuracy": float(np.min(target_scores)),
            "accepted": bool(np.min(target_scores) >= 0.85),
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(artifact["target_transfer"]))
    return 0


def _source_rows(
    artifact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    groups = []
    for row in artifact["examples"]:
        for embedding in row["embeddings"]:
            features.append(embedding)
            labels.append(row["state"] == "free_throw")
            groups.append(row["pair_id"])
    return (
        np.asarray(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
    )


def _source_model() -> Any:
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=8, whiten=True, random_state=0),
        SVC(C=1.0, class_weight="balanced", random_state=0),
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
