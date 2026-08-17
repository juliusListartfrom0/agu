#!/usr/bin/env python3
"""Screen an offline E-BARD object-role gate on reviewed broadcast crops.

The source archive is used only for training and a game-disjoint sanity check.
The target crops come from the already-sealed ATL--CHI development review
sheets.  The output is evidence only: it is not a runtime checkpoint or a
blind-game answer source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.e_bard_role_transfer import (  # noqa: E402
    ROLE_TRANSFER_SCHEMA,
    binary_gate_metrics,
    canonical_sha256,
    choose_precision_threshold,
    fit_role_classifier,
    image_role_features,
    load_archive_features,
    split_group_indices,
)

PLAN_SCHEMA = "agu.broadcast-ball-offline-review-plan.v1"
REVIEW_SCHEMA = "agu.broadcast-ball-offline-review.v1"
VALID_DECISIONS = {"valid_ball", "false_positive"}
PANEL_WIDTH = 480
PANEL_HEIGHT = 360
SHEET_COLUMNS = 4
SHEET_ROWS = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify_review(plan: dict[str, Any], review: dict[str, Any]) -> dict[str, str]:
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("runtime_consumable") is not False
        or plan.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("invalid review plan provenance")
    if (
        review.get("schema_version") != REVIEW_SCHEMA
        or review.get("runtime_consumable") is not False
        or review.get("codex_runtime_answer_used") is not False
        or review.get("plan_sha256") != plan.get("artifact_sha256")
    ):
        raise ValueError("invalid sealed review provenance")
    candidates = plan.get("candidates")
    decisions = review.get("decisions")
    if not isinstance(candidates, list) or not isinstance(decisions, list):
        raise ValueError("review plan and decisions must contain lists")
    candidate_ids = [str(row.get("candidate_id")) for row in candidates]
    decision_by_id: dict[str, str] = {}
    for row in decisions:
        candidate_id = str(row.get("candidate_id"))
        decision = str(row.get("decision"))
        if candidate_id in decision_by_id or candidate_id not in candidate_ids:
            raise ValueError("review decisions do not match plan candidates")
        if decision not in {"valid_ball", "false_positive", "uncertain"}:
            raise ValueError(f"unsupported review decision: {decision}")
        decision_by_id[candidate_id] = decision
    if set(decision_by_id) != set(candidate_ids):
        raise ValueError("review decisions are incomplete")
    return decision_by_id


def _sheet_crop(sheet: Image.Image, panel_index: int) -> Image.Image:
    """Extract the border-free enlarged inset from one review panel."""

    if not 1 <= panel_index <= SHEET_COLUMNS * SHEET_ROWS:
        raise ValueError(f"invalid panel index: {panel_index}")
    zero_based = panel_index - 1
    x = (zero_based % SHEET_COLUMNS) * PANEL_WIDTH
    y = (zero_based // SHEET_COLUMNS) * PANEL_HEIGHT
    # The renderer puts a 3px yellow border around the 160x120 inset.  Leave
    # a small margin so the classifier cannot learn the annotation border.
    return sheet.crop((x + 318, y + 148, x + 472, y + 262)).convert("RGB")


def _target_features(
    plan: dict[str, Any],
    decision_by_id: dict[str, str],
    sheets_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, str]]:
    candidates = plan["candidates"]
    max_sheet = max(int(row["sheet_index"]) for row in candidates)
    sheet_paths: dict[int, Path] = {}
    sheet_hashes: dict[str, str] = {}
    sheet_images: dict[int, Image.Image] = {}
    for sheet_index in range(1, max_sheet + 1):
        path = sheets_dir / f"sheet-{sheet_index:02d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as decoded:
            if decoded.size != (PANEL_WIDTH * SHEET_COLUMNS, PANEL_HEIGHT * SHEET_ROWS):
                raise ValueError(f"unexpected review sheet size: {path}")
            sheet_images[sheet_index] = decoded.convert("RGB")
        sheet_paths[sheet_index] = path
        sheet_hashes[path.name] = _sha256_file(path)

    features: list[np.ndarray] = []
    labels: list[bool] = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        decision = decision_by_id[candidate_id]
        crop = _sheet_crop(sheet_images[int(candidate["sheet_index"])], int(candidate["panel_index"]))
        features.append(image_role_features(crop))
        labels.append(decision == "valid_ball")
        rows.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "sheet": f"sheet-{int(candidate['sheet_index']):02d}.jpg",
                "panel_index": int(candidate["panel_index"]),
            }
        )
    return np.asarray(features), np.asarray(labels, dtype=bool), rows, sheet_hashes


def _source_sanity(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    minimum_precision: float,
) -> tuple[dict[str, Any], float]:
    train, valid = split_group_indices(groups)
    model = fit_role_classifier(features[train], labels[train])
    class_names = list(model[-1].classes_)
    basketball_index = class_names.index("basketball")
    scores = model.predict_proba(features[valid])[:, basketball_index]
    truth = labels[valid] == "basketball"
    threshold = choose_precision_threshold(
        truth,
        scores,
        minimum_precision=minimum_precision,
    )
    metrics = binary_gate_metrics(truth, scores, threshold)
    metrics.update(
        {
            "train_examples": int(len(train)),
            "holdout_examples": int(len(valid)),
            "train_games": int(len(set(groups[train].tolist()))),
            "holdout_games": int(len(set(groups[valid].tolist()))),
            "threshold": None if not np.isfinite(threshold) else float(threshold),
            "threshold_policy": f"highest_recall_at_precision>={minimum_precision:.2f}",
        }
    )
    return metrics, threshold


def _target_transfer(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    target_rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    model = fit_role_classifier(source_features, source_labels)
    class_names = list(model[-1].classes_)
    basketball_index = class_names.index("basketball")
    scores = model.predict_proba(target_features)[:, basketball_index]
    determinate = np.asarray(
        [row["decision"] in VALID_DECISIONS for row in target_rows], dtype=bool
    )
    target_metrics = binary_gate_metrics(
        target_labels[determinate],
        scores[determinate],
        threshold,
    )
    predictions = []
    for row, score in zip(target_rows, scores, strict=True):
        predictions.append(
            {
                **row,
                "basketball_score": float(score),
                "predicted_valid_ball": bool(score >= threshold),
            }
        )
    target_metrics.update(
        {
            "reviewed_examples": int(determinate.sum()),
            "uncertain_examples_excluded": int((~determinate).sum()),
            "threshold": None if not np.isfinite(threshold) else float(threshold),
            "predictions": predictions,
            "accepted": bool(
                target_metrics["precision"] >= 0.85
                and target_metrics["recall"] >= 0.85
            ),
        }
    )
    return target_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--sheets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.minimum_precision <= 1.0:
        raise ValueError("--minimum-precision must be in (0, 1]")
    plan = _read_json(args.plan)
    review = _read_json(args.review)
    decision_by_id = _verify_review(plan, review)
    source_features, source_labels, source_groups, split_counts = load_archive_features(args.archive)
    source_sanity, threshold = _source_sanity(
        source_features,
        source_labels,
        source_groups,
        minimum_precision=args.minimum_precision,
    )
    target_features, target_labels, target_rows, sheet_hashes = _target_features(
        plan,
        decision_by_id,
        args.sheets_dir,
    )
    target_transfer = _target_transfer(
        source_features,
        source_labels,
        target_features,
        target_labels,
        target_rows,
        threshold,
    )
    artifact: dict[str, Any] = {
        "schema_version": ROLE_TRANSFER_SCHEMA,
        "purpose": "offline_object_role_gate_transfer_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source": {
            "archive_path": str(args.archive),
            "archive_sha256": _sha256_file(args.archive),
            "split_example_counts": split_counts,
            "example_count": int(len(source_labels)),
            "game_count": int(len(set(source_groups.tolist()))),
            "class_counts": dict(sorted(Counter(source_labels.tolist()).items())),
        },
        "target": {
            "plan_sha256": str(plan["artifact_sha256"]),
            "review_sha256": str(review["artifact_sha256"]),
            "sheet_sha256": dict(sorted(sheet_hashes.items())),
            "candidate_count": int(len(target_rows)),
            "review_counts": dict(sorted(Counter(decision_by_id.values()).items())),
        },
        "protocol": {
            "feature": "16x16 RGB pixels + HSV histograms + channel moments + grayscale gradients",
            "classifier": "StandardScaler balanced LogisticRegression multiclass",
            "source_split": "deterministic 20% held-out game IDs; all E-BARD manifest splits pooled",
            "threshold": f"source holdout highest recall at precision>={args.minimum_precision:.2f}",
            "target_crop": "border-free 154x114 inset from hash-bound ATL-CHI review sheet",
            "acceptance": "target precision and recall both >= 0.85 on determinate human decisions",
        },
        "source_sanity": source_sanity,
        "target_transfer": target_transfer,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "source_sanity": source_sanity,
                "target_transfer": {
                    key: value
                    for key, value in target_transfer.items()
                    if key != "predictions"
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
