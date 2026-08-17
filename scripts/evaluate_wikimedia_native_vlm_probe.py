#!/usr/bin/env python3
"""Evaluate a Wikimedia native-video VLM probe after predictions are sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import (  # noqa: E402
    canonical_sha256,
    evaluate_independent_shot_vlm,
    verify_independent_shot_vlm_plan,
    verify_independent_shot_vlm_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = verify_independent_shot_vlm_plan(
        json.loads(args.plan.read_text(encoding="utf-8"))
    )
    predictions = verify_independent_shot_vlm_predictions(
        json.loads(args.predictions.read_text(encoding="utf-8"))
    )
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    if truth.get("source_plan_sha256") != plan["plan_sha256"]:
        raise ValueError("truth artifact does not bind to the source plan")
    rows = truth.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("truth artifact requires rows")
    truth_map: dict[tuple[str, str, str], bool] = {}
    for row in rows:
        key = (
            str(row.get("source_video_sha256") or ""),
            str(row.get("candidate_bundle_sha256") or ""),
            str(row.get("event_id") or ""),
        )
        if not all(key) or not isinstance(row.get("event_present"), bool):
            raise ValueError("truth row is incomplete")
        if key in truth_map:
            raise ValueError("truth artifact contains duplicate rows")
        truth_map[key] = bool(row["event_present"])
    selected_keys = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
        for row in plan["examples"]
    }
    if selected_keys != set(truth_map):
        raise ValueError("truth artifact does not exactly cover the source plan")
    if predictions["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("prediction artifact does not bind to the source plan")
    evaluation = evaluate_independent_shot_vlm(
        plan=plan,
        predictions=predictions,
        truth=truth_map,
        minimum_precision=0.85,
        minimum_recall=0.85,
    )
    evaluation["truth_artifact_sha256"] = _file_sha256(args.truth)
    evaluation["resource_guard_required"] = True
    evaluation["runtime_consumable"] = False
    evaluation["model_promotion"] = "none"
    evaluation["artifact_sha256"] = canonical_sha256(
        evaluation,
        hash_field="artifact_sha256",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": evaluation["artifact_sha256"],
                "metrics": evaluation["metrics"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
