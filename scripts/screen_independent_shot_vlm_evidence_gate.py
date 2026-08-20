#!/usr/bin/env python3
"""Reproduce the legacy unbound evidence gate as non-promotable provenance."""

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
    verify_independent_shot_vlm_plan,
    verify_independent_shot_vlm_predictions,
)
from app.analysis.independent_shot_vlm_evidence_gate import (  # noqa: E402
    _canonical_sha256,
    screen_evidence_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
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
    if predictions.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("VLM predictions do not match the frozen plan")
    expected_annotation_sha = set(plan.get("training_annotation_sha256") or [])
    supplied_annotation_sha = {_file_sha256(path) for path in args.annotation}
    if supplied_annotation_sha != expected_annotation_sha:
        raise ValueError("annotations do not exactly match the frozen plan")
    truth = _load_truth(args.annotation)
    selected_keys = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
        for row in plan["examples"]
    }
    if set(truth) < selected_keys:
        raise ValueError("annotations do not cover the frozen VLM plan")
    auxiliary = json.loads(args.auxiliary.read_text(encoding="utf-8"))
    if auxiliary.get("runtime_consumable") is not False:
        raise ValueError("auxiliary screen must be training-only")
    auxiliary_rows = auxiliary.get("oof_predictions")
    if not isinstance(auxiliary_rows, list) or not auxiliary_rows:
        raise ValueError("auxiliary screen has no OOF predictions")
    result = screen_evidence_gate(
        vlm_predictions=predictions["predictions"],
        auxiliary_oof_rows=auxiliary_rows,
        target_truth={key: truth[key] for key in selected_keys},
    )
    result.update(
        {
            "plan_sha256": plan["plan_sha256"],
            "vlm_prediction_artifact_sha256": predictions["artifact_sha256"],
            "auxiliary_artifact_sha256": _artifact_sha256(auxiliary),
        }
    )
    result["artifact_sha256"] = _canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **result["metrics"]}))
    return 2


def _load_truth(paths: list[Path]) -> dict[tuple[str, str, str], bool]:
    truth: dict[tuple[str, str, str], bool] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_sha = str(payload.get("source_video_sha256") or "")
        candidate_sha = str(payload.get("candidate_bundle_sha256") or "")
        if not source_sha or not candidate_sha:
            raise ValueError("annotation source/candidate hashes are required")
        for row in payload.get("examples", []):
            if not isinstance(row, dict) or not isinstance(row.get("event_present"), bool):
                raise ValueError("shot-validity truth must be boolean")
            key = (source_sha, candidate_sha, str(row.get("event_id") or ""))
            if not all(key):
                raise ValueError("annotation event key is incomplete")
            truth[key] = bool(row["event_present"])
    return truth


def _artifact_sha256(payload: dict[str, object]) -> str:
    claimed = payload.get("artifact_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("auxiliary artifact hash is missing")
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    if _canonical_sha256(normalized) != claimed:
        raise ValueError("auxiliary artifact hash mismatch")
    return claimed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
