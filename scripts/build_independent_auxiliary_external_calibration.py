#!/usr/bin/env python3
"""Build a label-hidden target diagnostic from external game-held calibration.

The calibration screens are independent labeled development games.  The target
transfer file is loaded without labels; the target review evaluation is joined
only after the frozen external threshold has been applied.  The artifact is
never runtime-consumable and deliberately contains no OOF predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_auxiliary_external_calibration import (  # noqa: E402
    build_screen_score_rows,
    evaluate_labeled_scores,
    select_external_score_threshold,
)
from app.analysis.independent_shot_auxiliary_transfer import verify_transfer_artifact  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _target_rows(transfer: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = transfer.get("transfer_predictions")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("target transfer predictions are required")
    rows = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("target transfer row must be a mapping")
        key = (str(raw.get("source_video_sha256") or ""), str(raw.get("event_id") or ""))
        if len(key[0]) != 64 or not key[1] or key in seen:
            raise ValueError("target transfer row key is invalid or duplicated")
        score = float(raw.get("transfer_score"))
        if not 0.0 <= score <= 1.0:
            raise ValueError("target transfer score is outside [0, 1]")
        rows.append(
            {
                "source_video_sha256": key[0],
                "event_id": key[1],
                "score": score,
            }
        )
        seen.add(key)
    return rows


def _target_labeled_rows(
    target_rows: list[dict[str, Any]],
    target_evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    labels: dict[tuple[str, str], bool] = {}
    for raw in target_evaluation.get("joined_rows", []):
        if not isinstance(raw, dict):
            raise ValueError("target evaluation row must be a mapping")
        label = str(raw.get("label") or "")
        if label not in {"positive", "negative"}:
            continue
        key = (str(raw.get("source_video_sha256") or ""), str(raw.get("event_id") or ""))
        if key in labels:
            raise ValueError("target evaluation contains duplicate determinate labels")
        labels[key] = label == "positive"
    output = []
    for row in target_rows:
        key = (row["source_video_sha256"], row["event_id"])
        if key in labels:
            output.append({**row, "event_present": labels[key]})
    return output


def build_artifact(
    *,
    calibration_paths: list[Path],
    target_transfer_path: Path,
    target_evaluation_path: Path,
) -> dict[str, Any]:
    calibration_rows: list[dict[str, Any]] = []
    calibration_sources: list[dict[str, Any]] = []
    for path in calibration_paths:
        screen = _load(path)
        rows = build_screen_score_rows(screen)
        calibration_rows.extend(rows)
        calibration_sources.append(
            {
                "path": path.as_posix(),
                "file_sha256": file_sha256(path),
                "video_sha256": screen["video_sha256"],
                "artifact_sha256": screen.get("artifact_sha256"),
                "row_count": len(rows),
            }
        )
    if len({(row["source_video_sha256"], row["event_id"]) for row in calibration_rows}) != len(calibration_rows):
        raise ValueError("calibration rows contain duplicate keys")
    selection = select_external_score_threshold(calibration_rows, minimum_recall=0.85)
    folds = []
    for held_group in selection["training_groups"]:
        training = [row for row in calibration_rows if row["source_video_sha256"] != held_group]
        held = [row for row in calibration_rows if row["source_video_sha256"] == held_group]
        held_selection = select_external_score_threshold(training, minimum_recall=0.85)
        folds.append(
            {
                "held_group": held_group,
                "training_groups": held_selection["training_groups"],
                "threshold": held_selection["threshold"],
                "training_selection": held_selection,
                "held_group_metrics": evaluate_labeled_scores(
                    held,
                    threshold=float(held_selection["threshold"]),
                ),
            }
        )

    target_transfer = verify_transfer_artifact(_load(target_transfer_path))
    target_rows = _target_rows(target_transfer)
    target_evaluation = _load(target_evaluation_path)
    target_labeled = _target_labeled_rows(target_rows, target_evaluation)
    target_predictions = [
        {
            **row,
            "predicted_present": row["score"] >= float(selection["threshold"]),
        }
        for row in target_rows
    ]
    grouped_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_labeled:
        grouped_target[row["source_video_sha256"]].append(row)
    target_metrics = evaluate_labeled_scores(
        target_labeled,
        threshold=float(selection["threshold"]),
    )
    artifact: dict[str, Any] = {
        "schema_version": "agu.independent-shot-auxiliary-external-calibration.v1",
        "purpose": "offline_external_game_held_auxiliary_score_calibration_diagnostic",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "score_semantics": "uncalibrated_detector_transfer_score; not a probability",
        "calibration": {
            "sources": calibration_sources,
            "row_count": len(calibration_rows),
            "group_count": len(selection["training_groups"]),
            "selected_threshold": selection,
            "group_held_folds": folds,
            "rows": calibration_rows,
        },
        "target": {
            "transfer_artifact": {
                "path": target_transfer_path.as_posix(),
                "file_sha256": file_sha256(target_transfer_path),
                "artifact_sha256": target_transfer["artifact_sha256"],
            },
            "evaluation_artifact": {
                "path": target_evaluation_path.as_posix(),
                "file_sha256": file_sha256(target_evaluation_path),
                "artifact_sha256": target_evaluation.get("artifact_sha256"),
            },
            "label_free_predictions": target_predictions,
            "determinate_post_inference_metrics": target_metrics,
            "per_source_determinate_metrics": {
                source: evaluate_labeled_scores(rows, threshold=float(selection["threshold"]))
                for source, rows in sorted(grouped_target.items())
            },
            "target_labels_used_for_threshold": False,
        },
        "oof_predictions": [],
        "promotion": {
            "auxiliary_oof_evidence_available": False,
            "promotion_eligible": False,
            "reason": (
                "external calibration selects threshold 0.0 with pooled precision below 0.85;"
                " target rows are a post-inference diagnostic, not OOF evidence"
            ),
        },
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--target-transfer", type=Path, required=True)
    parser.add_argument("--target-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_artifact(
        calibration_paths=args.calibration,
        target_transfer_path=args.target_transfer,
        target_evaluation_path=args.target_evaluation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact_sha256": artifact["artifact_sha256"], "target_threshold": artifact["calibration"]["selected_threshold"]["threshold"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
