#!/usr/bin/env python3
"""Evaluate transfer scores descriptively against a sealed pilot review.

This command is an analysis-only companion to the label-hidden transfer screen.
The review is loaded after inference, never passed to the detector, and cannot
select a threshold or produce OOF rows.  All cutoff tables are explicitly
descriptive and promotion is hard-coded false.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_auxiliary_transfer import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    verify_transfer_artifact,
)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(rows: list[dict[str, Any]], cutoff: float) -> dict[str, Any]:
    determinate = [row for row in rows if row["label"] in {"positive", "negative"}]
    tp = sum(row["label"] == "positive" and row["score"] >= cutoff for row in determinate)
    fp = sum(row["label"] == "negative" and row["score"] >= cutoff for row in determinate)
    fn = sum(row["label"] == "positive" and row["score"] < cutoff for row in determinate)
    tn = sum(row["label"] == "negative" and row["score"] < cutoff for row in determinate)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "cutoff": cutoff,
        "evaluated_determinate_count": len(determinate),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "cutoff_is_descriptive_only": True,
    }


def evaluate_transfer(*, transfer_path: Path, review_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    transfer = verify_transfer_artifact(json.loads(transfer_path.read_text(encoding="utf-8")))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_rows = review.get("reviews")
    if not isinstance(review_rows, list) or not review_rows:
        raise ValueError("review contains no rows")
    transfer_rows = transfer["transfer_predictions"]
    by_event = {str(row["event_id"]): row for row in transfer_rows}
    if len(by_event) != len(transfer_rows):
        raise ValueError("transfer rows contain duplicate event IDs")
    joined: list[dict[str, Any]] = []
    for reviewed in review_rows:
        event_id = str(reviewed.get("review_id") or "")
        if event_id not in by_event:
            raise ValueError(f"review event is absent from transfer screen: {event_id}")
        row = by_event[event_id]
        if str(reviewed.get("source_video_sha256") or "") != str(row["source_video_sha256"]):
            raise ValueError(f"review/source hash mismatch: {event_id}")
        sequence = str(reviewed.get("shot_sequence") or "")
        label = (
            "positive"
            if sequence == "shot"
            else "negative"
            if sequence == "not_a_shot"
            else "unknown"
            if sequence == "uncertain"
            else "invalid"
        )
        if label == "invalid":
            raise ValueError(f"unsupported review sequence: {event_id}")
        joined.append(
            {
                "event_id": event_id,
                "source_video_sha256": row["source_video_sha256"],
                "label": label,
                "score": float(row["transfer_score"]),
                "positive_frame_count": row["positive_frame_count"],
                "sampled_frame_count": row["sampled_frame_count"],
            }
        )
    if set(by_event) != {row["event_id"] for row in joined}:
        raise ValueError("review must exactly cover transfer events")

    determinate = [row for row in joined if row["label"] != "unknown"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[str(row["source_video_sha256"])].append(row)
    score_summary: dict[str, dict[str, Any]] = {}
    for label in ("positive", "negative", "unknown"):
        scores = [row["score"] for row in joined if row["label"] == label]
        score_summary[label] = {
            "count": len(scores),
            "scores": scores,
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        }
    cutoffs = [0.0, 0.25, 0.5, 0.75, 0.9]
    artifact: dict[str, Any] = {
        "schema_version": "agu.independent-shot-vlm-auxiliary-transfer-evaluation.v1",
        "purpose": "offline_pilot_diagnostic_of_transfer_scores",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "transfer_artifact": {
            "path": transfer_path.as_posix(),
            "file_sha256": file_sha256(transfer_path),
            "artifact_sha256": transfer["artifact_sha256"],
        },
        "review_artifact": {
            "path": review_path.as_posix(),
            "file_sha256": file_sha256(review_path),
            "plan_sha256": review.get("plan_sha256"),
        },
        "label_policy": "review labels are read only after detector inference; uncertain is excluded from cutoff metrics and never becomes training truth",
        "event_count": len(joined),
        "determinate_event_count": len(determinate),
        "score_summary": score_summary,
        "cutoff_diagnostics": {
            "pooled": [_metrics(joined, cutoff) for cutoff in cutoffs],
            "determinate_only": [_metrics(determinate, cutoff) for cutoff in cutoffs],
            "per_video": {
                source_sha: [_metrics(rows, cutoff) for cutoff in cutoffs]
                for source_sha, rows in sorted(grouped.items())
            },
        },
        "promotion": {
            "threshold_selected": False,
            "auxiliary_oof_evidence_available": False,
            "promotion_eligible": False,
            "reason": "target-game pilot labels cannot select a threshold or create game-held OOF evidence",
        },
        "joined_rows": joined,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = evaluate_transfer(
        transfer_path=args.transfer,
        review_path=args.review,
        output_path=args.output,
    )
    print(json.dumps({"artifact_sha256": artifact["artifact_sha256"], "event_count": artifact["event_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
