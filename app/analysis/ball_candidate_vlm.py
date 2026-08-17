"""Offline contracts for an independent VLM basketball-candidate reviewer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from app.analysis.ball_candidate_review import select_stratified_candidates
from app.analysis.ball_candidate_verifier import stratified_review_metrics
from app.analysis.vlm import extract_json_object

BALL_CANDIDATE_VLM_PROMPT = """Inspect this basketball broadcast candidate panel.
The full broadcast frame contains one RED rectangle. A yellow-bordered inset
enlarges the same local area. Judge only whether visible pixels of the actual
orange/brown basketball are inside the RED rectangle. Do not infer from game
context or detector confidence. A head, hand, shoe, rim, light, logo, or
background object is not a basketball.

Make the best visual binary decision even when the object is small or blurry:
use "ball" when the red rectangle overlaps a real basketball and "not_ball"
otherwise. Do not default to "uncertain" because confidence is low. Use
"uncertain" only if the supplied image is corrupted or unreadable.

Return exactly one JSON object:
{"ball_state":"ball"|"not_ball"|"uncertain","confidence":0.0,"reason":"brief visual reason"}"""

BALL_STATES = frozenset({"ball", "not_ball", "uncertain"})


def select_vlm_probe_candidates(
    plan: Mapping[str, Any],
    *,
    samples_per_band: int,
) -> list[dict[str, Any]]:
    """Select a deterministic label-free probe from every review band."""
    forbidden = {"decision", "label", "review_note", "ground_truth"}
    candidates = list(plan.get("candidates") or [])
    if any(forbidden.intersection(row) for row in candidates):
        raise ValueError("VLM probe plan contains label-bearing fields")
    bands = [
        (float(values[0]), float(values[1]))
        for values in plan["sampling"]["bands"]
    ]
    return select_stratified_candidates(
        candidates,
        bands=bands,
        samples_per_band=samples_per_band,
    )


def parse_ball_candidate_vlm_response(text: str) -> dict[str, object]:
    """Normalize a local-model response and fail closed on malformed output."""
    try:
        payload = extract_json_object(text)
    except (ValueError, json.JSONDecodeError):
        return {
            "ball_state": "uncertain",
            "confidence": 0.0,
            "reason": "local VLM returned no valid JSON object",
            "available": False,
        }
    state = str(payload.get("ball_state") or "").strip().lower()
    if state not in BALL_STATES:
        state = "uncertain"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "ball_state": state,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(payload.get("reason") or "")[:500],
        "available": True,
    }


def evaluate_ball_candidate_vlm(
    *,
    perception: Mapping[str, Any],
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate label-hidden VLM predictions with stratified uncertainty bounds."""
    if plan.get("perception_artifact_sha256") != perception.get("artifact_sha256"):
        raise ValueError("VLM evaluation plan/perception binding mismatch")
    if review.get("plan_sha256") != plan.get("artifact_sha256"):
        raise ValueError("VLM evaluation review/plan binding mismatch")
    if review.get("runtime_consumable") is not False:
        raise ValueError("VLM evaluation review must be offline-only")
    if predictions.get("review_plan_sha256") != plan.get("artifact_sha256"):
        raise ValueError("VLM predictions/review-plan binding mismatch")

    bands = [
        (float(values[0]), float(values[1]))
        for values in plan["sampling"]["bands"]
    ]
    band_names = [f"{low:.2f}-{high:.2f}" for low, high in bands]
    populations = {name: 0 for name in band_names}
    for row in perception["detections"]:
        confidence = float(row["confidence"])
        for name, (low, high) in zip(band_names, bands, strict=True):
            if low <= confidence < high:
                populations[name] += 1
                break

    plan_rows = {
        str(row["candidate_id"]): row for row in plan["candidates"]
    }
    decisions = {
        str(row["candidate_id"]): str(row["decision"])
        for row in review["decisions"]
    }
    prediction_rows = list(predictions.get("predictions") or [])
    prediction_ids = [str(row["candidate_id"]) for row in prediction_rows]
    if (
        not prediction_ids
        or len(set(prediction_ids)) != len(prediction_ids)
        or any(candidate_id not in plan_rows for candidate_id in prediction_ids)
    ):
        raise ValueError("VLM predictions contain invalid candidate IDs")

    bounds: dict[str, list[dict[str, object]]] = {"lower": [], "upper": []}
    rows: list[dict[str, object]] = []
    for prediction in prediction_rows:
        candidate_id = str(prediction["candidate_id"])
        candidate = plan_rows[candidate_id]
        decision = decisions[candidate_id]
        state = str(prediction.get("ball_state") or "uncertain")
        if state not in BALL_STATES:
            raise ValueError("VLM prediction contains an unsupported state")
        kept = state == "ball"
        band = (
            f"{float(candidate['confidence_band'][0]):.2f}-"
            f"{float(candidate['confidence_band'][1]):.2f}"
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "ball_state": state,
                "kept": kept,
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
            bounds[bound].append(
                {"band": band, "label": label, "kept": kept}
            )
    return {
        "population_by_band": populations,
        "metrics": {
            bound: stratified_review_metrics(
                values,
                band_population=populations,
            )
            for bound, values in bounds.items()
        },
        "rows": rows,
    }
