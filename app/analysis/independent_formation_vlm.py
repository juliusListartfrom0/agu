"""Label-hidden contracts for an independent free-throw formation VLM."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.causal_shot_phase_review import (
    FORMATION_STATES as REVIEW_FORMATION_STATES,
)
from app.analysis.causal_shot_phase_review import (
    verify_causal_shot_phase_review_plan,
)

FORMATION_VLM_PLAN_SCHEMA = "agu.independent-formation-vlm-plan.v1"
FORMATION_VLM_PREDICTIONS_SCHEMA = (
    "agu.independent-formation-vlm-predictions.v1"
)
FORMATION_VLM_EVALUATION_SCHEMA = (
    "agu.independent-formation-vlm-evaluation.v1"
)
FORMATION_VETO_EVALUATION_SCHEMA = (
    "agu.independent-formation-veto-evaluation.v1"
)
PREDICTED_FORMATION_STATES = frozenset(
    {"free_throw_setup", "live_play", "stoppage_other", "unknown"}
)
OBSERVABLE_FIELDS = (
    "shooter_at_free_throw_line",
    "lane_players_aligned",
    "players_stationary_for_free_throw",
    "continuous_live_motion",
)
_FORBIDDEN_PLAN_FIELDS = frozenset(
    {
        "formation_state",
        "broadcast_context",
        "shot_sequence",
        "outcome",
        "confidence",
        "notes",
        "truth",
        "label",
        "event_present",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_sha256(payload: Mapping[str, Any], *, hash_field: str) -> str:
    normalized = dict(payload)
    normalized.pop(hash_field, None)
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal_independent_formation_vlm_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a two-frame plan that exposes no phase-review labels."""
    plan = dict(payload)
    plan.pop("plan_sha256", None)
    plan["schema_version"] = FORMATION_VLM_PLAN_SCHEMA
    plan["purpose"] = "offline_independent_free_throw_formation_screening"
    plan["runtime_consumable"] = False
    plan["codex_runtime_answer_used"] = False
    plan["labels_or_review_notes_exposed_to_model"] = False
    _validate_plan(plan)
    plan["plan_sha256"] = _canonical_sha256(
        plan,
        hash_field="plan_sha256",
    )
    return plan


def verify_independent_formation_vlm_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    plan = dict(payload)
    claimed = str(plan.pop("plan_sha256", ""))
    _validate_plan(plan)
    if claimed != _canonical_sha256(plan, hash_field="plan_sha256"):
        raise ValueError("independent formation VLM plan hash mismatch")
    plan["plan_sha256"] = claimed
    return plan


def derive_independent_formation_vlm_plan(
    phase_plan: Mapping[str, Any],
    *,
    frame_offsets_seconds: Sequence[float] = (-1.0, 0.0),
    image_width: int = 512,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Derive exact two-frame identities from a sealed causal phase plan."""
    source = verify_causal_shot_phase_review_plan(phase_plan)
    offsets = [float(value) for value in frame_offsets_seconds]
    if len(offsets) != 2 or offsets[0] >= offsets[1]:
        raise ValueError("formation VLM requires two increasing frame offsets")
    source_offsets = [float(value) for value in source["anchor_offsets_seconds"]]
    if any(value not in source_offsets for value in offsets):
        raise ValueError("formation offsets must exist in the source phase plan")
    if image_width <= 0:
        raise ValueError("formation image width must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max examples must be positive")

    rows = list(source["examples"])
    if max_examples is not None:
        rows = rows[:max_examples]
    examples = []
    for row in rows:
        indexes = [int(value) for value in row["frame_indexes"]]
        selected = [indexes[source_offsets.index(value)] for value in offsets]
        examples.append(
            {
                "phase_review_id": str(row["phase_review_id"]),
                "source_video_sha256": str(row["source_video_sha256"]),
                "source_video_filename": str(row["source_video_filename"]),
                "candidate_bundle_sha256": str(
                    row["candidate_bundle_sha256"]
                ),
                "event_id": str(row["event_id"]),
                "source_fps": float(row["source_fps"]),
                "frame_indexes": selected,
            }
        )
    selection = {
        "method": "source_phase_plan_order",
        "source_example_count": len(source["examples"]),
        "example_count": len(examples),
        "resource_probe_only": max_examples is not None,
    }
    return seal_independent_formation_vlm_plan(
        {
            "source_phase_plan_sha256": source["artifact_sha256"],
            "source_video_sha256s": sorted(
                {str(row["source_video_sha256"]) for row in examples}
            ),
            "sealed_blind_video_sha256s": list(
                source["sealed_blind_video_sha256s"]
            ),
            "frame_offsets_seconds": offsets,
            "input_contract": {
                "raw_frames_only": True,
                "chronological_fixed_sampling": True,
                "max_frames": 2,
                "image_width": int(image_width),
            },
            "selection": selection,
            "examples": examples,
        }
    )


def parse_independent_formation_vlm_decision(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    state = str(payload.get("formation_state") or "unknown")
    if state not in PREDICTED_FORMATION_STATES:
        state = "unknown"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    observables = {
        field: value if isinstance(value := payload.get(field), bool) else None
        for field in OBSERVABLE_FIELDS
    }
    return {
        "formation_state": state,
        "confidence": confidence,
        "observables": observables,
        "reason": str(payload.get("reason") or "")[:1000],
    }


def seal_independent_formation_vlm_predictions(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = FORMATION_VLM_PREDICTIONS_SCHEMA
    artifact["purpose"] = "offline_independent_free_throw_formation_screening"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    _validate_predictions(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(
        artifact,
        hash_field="artifact_sha256",
    )
    return artifact


def verify_independent_formation_vlm_predictions(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_predictions(artifact)
    if claimed != _canonical_sha256(
        artifact,
        hash_field="artifact_sha256",
    ):
        raise ValueError("independent formation VLM prediction hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def evaluate_independent_formation_vlm(
    *,
    plan: Mapping[str, Any],
    predictions: Mapping[str, Any],
    truth: Mapping[tuple[str, str, str], str],
) -> dict[str, Any]:
    """Evaluate frozen predictions only after phase labels are supplied."""
    verified_plan = verify_independent_formation_vlm_plan(plan)
    verified_predictions = verify_independent_formation_vlm_predictions(
        predictions
    )
    if verified_predictions["plan_sha256"] != verified_plan["plan_sha256"]:
        raise ValueError("formation VLM predictions do not match the plan")

    plan_by_key = {
        _identity(row): row for row in verified_plan["examples"]
    }
    prediction_by_key = {
        _identity(row): row for row in verified_predictions["predictions"]
    }
    if set(prediction_by_key) != set(plan_by_key):
        raise ValueError("formation VLM predictions must exactly cover the plan")
    if set(truth) != set(plan_by_key):
        raise ValueError("formation truth must exactly cover the plan")
    if any(value not in REVIEW_FORMATION_STATES for value in truth.values()):
        raise ValueError("formation truth contains an unsupported state")

    overall = _metrics_for_keys(
        list(plan_by_key),
        prediction_by_key=prediction_by_key,
        truth=truth,
    )
    by_video: dict[str, dict[str, Any]] = {}
    grouped: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in plan_by_key:
        grouped[key[0]].append(key)
    for video_sha, keys in sorted(grouped.items()):
        by_video[video_sha] = _metrics_for_keys(
            keys,
            prediction_by_key=prediction_by_key,
            truth=truth,
        )

    relevant_per_video = [
        metrics
        for metrics in by_video.values()
        if metrics["free_throw_support"] > 0
        and metrics["live_play_support"] > 0
    ]
    overall["promotion_eligible"] = bool(
        overall["free_throw_precision"] >= 0.85
        and overall["free_throw_recall"] >= 0.85
        and overall["live_play_retention"] >= 0.85
        and overall["unknown_rate"] <= 0.05
        and relevant_per_video
        and all(
            metrics["free_throw_recall"] >= 0.85
            and metrics["live_play_retention"] >= 0.85
            for metrics in relevant_per_video
        )
    )
    artifact: dict[str, Any] = {
        "schema_version": FORMATION_VLM_EVALUATION_SCHEMA,
        "purpose": "offline_independent_free_throw_formation_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "plan_sha256": verified_plan["plan_sha256"],
        "predictions_sha256": verified_predictions["artifact_sha256"],
        "metrics": overall,
        "per_video": by_video,
    }
    artifact["artifact_sha256"] = _canonical_sha256(
        artifact,
        hash_field="artifact_sha256",
    )
    return artifact


def evaluate_fixed_formation_veto(
    *,
    formation_predictions: Mapping[str, Any],
    upstream_artifact_sha256: str,
    upstream_predictions: Sequence[Mapping[str, Any]],
    threshold: float,
    upstream_variant: str = "",
) -> dict[str, Any]:
    """Evaluate the fixed rule: positive upstream AND not a predicted free throw."""
    formation = verify_independent_formation_vlm_predictions(
        formation_predictions
    )
    _require_sha256(upstream_artifact_sha256, field="upstream artifact")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("formation-veto threshold must be between zero and one")
    upstream_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in upstream_predictions:
        key = (
            str(row.get("source_video_sha256") or ""),
            str(row.get("event_id") or ""),
        )
        if (
            key in upstream_by_key
            or not key[0]
            or not key[1]
            or not isinstance(row.get("event_present"), bool)
            or not 0.0 <= float(row.get("probability", -1.0)) <= 1.0
        ):
            raise ValueError("invalid upstream formation-veto prediction")
        upstream_by_key[key] = row

    evaluated = []
    for row in formation["predictions"]:
        key = (
            str(row["source_video_sha256"]),
            str(row["event_id"]),
        )
        upstream = upstream_by_key.get(key)
        if upstream is None:
            raise ValueError(
                "upstream predictions must cover every formation prediction"
            )
        baseline = float(upstream["probability"]) >= threshold
        veto_signal = row["formation_state"] == "free_throw_setup"
        evaluated.append(
            {
                "source_video_sha256": key[0],
                "event_id": key[1],
                "event_present": bool(upstream["event_present"]),
                "baseline_positive": baseline,
                "formation_veto_signal": veto_signal,
                "formation_veto_positive": baseline and not veto_signal,
            }
        )

    baseline_metrics = _binary_metrics(
        evaluated,
        prediction_field="baseline_positive",
    )
    veto_metrics = _binary_metrics(
        evaluated,
        prediction_field="formation_veto_positive",
    )
    artifact: dict[str, Any] = {
        "schema_version": FORMATION_VETO_EVALUATION_SCHEMA,
        "purpose": "offline_fixed_independent_formation_veto_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "formation_predictions_sha256": formation["artifact_sha256"],
        "upstream_artifact_sha256": upstream_artifact_sha256,
        "upstream_variant": str(upstream_variant),
        "threshold": float(threshold),
        "example_count": len(evaluated),
        "formation_signal_count": sum(
            row["formation_veto_signal"] for row in evaluated
        ),
        "affected_event_count": sum(
            row["baseline_positive"] and row["formation_veto_signal"]
            for row in evaluated
        ),
        "baseline": baseline_metrics,
        "formation_veto": veto_metrics,
        "accepted": bool(
            veto_metrics["precision"] >= 0.85
            and veto_metrics["recall"] >= 0.85
        ),
    }
    artifact["artifact_sha256"] = _canonical_sha256(
        artifact,
        hash_field="artifact_sha256",
    )
    return artifact


def _metrics_for_keys(
    keys: Sequence[tuple[str, str, str]],
    *,
    prediction_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
    truth: Mapping[tuple[str, str, str], str],
) -> dict[str, Any]:
    tp = fp = fn = tn = unknown = exact = 0
    free_throw_support = live_play_support = live_play_retained = 0
    for key in keys:
        actual = truth[key]
        predicted = str(prediction_by_key[key]["formation_state"])
        is_free_throw = actual == "free_throw_setup"
        predicts_free_throw = predicted == "free_throw_setup"
        unknown += predicted == "unknown"
        exact += predicted == (
            "unknown" if actual == "uncertain" else actual
        )
        if is_free_throw and predicts_free_throw:
            tp += 1
        elif not is_free_throw and predicts_free_throw:
            fp += 1
        elif is_free_throw:
            fn += 1
        elif predicted != "unknown":
            tn += 1
        free_throw_support += is_free_throw
        if actual == "live_play":
            live_play_support += 1
            live_play_retained += predicted == "live_play"

    count = len(keys)
    return {
        "example_count": count,
        "free_throw_support": free_throw_support,
        "live_play_support": live_play_support,
        "free_throw_true_positive": tp,
        "free_throw_false_positive": fp,
        "free_throw_false_negative": fn,
        "free_throw_true_negative": tn,
        "unknown": unknown,
        "free_throw_precision": _ratio(tp, tp + fp),
        "free_throw_recall": _ratio(tp, tp + fn),
        "live_play_retention": _ratio(
            live_play_retained,
            live_play_support,
        ),
        "exact_accuracy": _ratio(exact, count),
        "unknown_rate": _ratio(unknown, count),
    }


def _binary_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_field: str,
) -> dict[str, Any]:
    true_positive = sum(
        bool(row["event_present"]) and bool(row[prediction_field])
        for row in rows
    )
    false_positive = sum(
        not bool(row["event_present"]) and bool(row[prediction_field])
        for row in rows
    )
    false_negative = sum(
        bool(row["event_present"]) and not bool(row[prediction_field])
        for row in rows
    )
    true_negative = sum(
        not bool(row["event_present"]) and not bool(row[prediction_field])
        for row in rows
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": _ratio(
            true_positive,
            true_positive + false_positive,
        ),
        "recall": _ratio(
            true_positive,
            true_positive + false_negative,
        ),
    }


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("schema_version") != FORMATION_VLM_PLAN_SCHEMA
        or plan.get("purpose")
        != "offline_independent_free_throw_formation_screening"
        or plan.get("runtime_consumable") is not False
        or plan.get("codex_runtime_answer_used") is not False
        or plan.get("labels_or_review_notes_exposed_to_model") is not False
    ):
        raise ValueError("invalid independent formation VLM plan contract")
    _require_sha256(
        plan.get("source_phase_plan_sha256"),
        field="source phase plan",
    )
    offsets = plan.get("frame_offsets_seconds")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or float(offsets[0]) >= float(offsets[1])
    ):
        raise ValueError("formation VLM plan requires two increasing offsets")
    input_contract = plan.get("input_contract")
    if (
        not isinstance(input_contract, Mapping)
        or input_contract.get("raw_frames_only") is not True
        or input_contract.get("chronological_fixed_sampling") is not True
        or input_contract.get("max_frames") != 2
        or int(input_contract.get("image_width", 0)) <= 0
    ):
        raise ValueError("invalid formation VLM input contract")
    examples = plan.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("formation VLM plan requires examples")
    identities = []
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("invalid formation VLM plan row")
        if _contains_forbidden_field(row):
            raise ValueError("formation VLM plan contains review labels")
        _require_sha256(row.get("source_video_sha256"), field="source video")
        _require_sha256(
            row.get("candidate_bundle_sha256"),
            field="candidate bundle",
        )
        frames = row.get("frame_indexes")
        if (
            not isinstance(frames, list)
            or len(frames) != 2
            or any(not isinstance(value, int) or value < 0 for value in frames)
            or frames[0] >= frames[1]
            or float(row.get("source_fps", 0.0)) <= 0
            or not str(row.get("phase_review_id") or "")
            or not str(row.get("event_id") or "")
        ):
            raise ValueError("invalid two-frame formation VLM plan row")
        identities.append(_identity(row))
    if len(identities) != len(set(identities)):
        raise ValueError("formation VLM plan identities must be unique")
    source_videos = set(plan.get("source_video_sha256s") or [])
    blind_videos = set(plan.get("sealed_blind_video_sha256s") or [])
    if source_videos and source_videos != {
        str(row["source_video_sha256"]) for row in examples
    }:
        raise ValueError("formation VLM plan source-video set mismatch")
    if source_videos & blind_videos:
        raise ValueError("sealed blind video cannot enter formation VLM plan")


def _validate_predictions(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != FORMATION_VLM_PREDICTIONS_SCHEMA
        or artifact.get("purpose")
        != "offline_independent_free_throw_formation_screening"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("invalid independent formation prediction contract")
    _require_sha256(artifact.get("plan_sha256"), field="plan")
    if not isinstance(artifact.get("model"), Mapping):
        raise ValueError("formation predictions require model provenance")
    rows = artifact.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("formation predictions require rows")
    identities = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("formation_state") not in PREDICTED_FORMATION_STATES
            or not 0.0 <= float(row.get("confidence", -1.0)) <= 1.0
            or not isinstance(row.get("observables"), Mapping)
        ):
            raise ValueError("invalid independent formation prediction")
        identities.append(_identity(row))
    if len(identities) != len(set(identities)):
        raise ValueError("formation prediction identities must be unique")


def _contains_forbidden_field(payload: Mapping[str, Any]) -> bool:
    for key, value in payload.items():
        if str(key).lower() in _FORBIDDEN_PLAN_FIELDS:
            return True
        if isinstance(value, Mapping) and _contains_forbidden_field(value):
            return True
    return False


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_video_sha256") or ""),
        str(row.get("candidate_bundle_sha256") or ""),
        str(row.get("event_id") or ""),
    )


def _require_sha256(value: object, *, field: str) -> str:
    normalized = str(value or "")
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{field} SHA-256 is invalid")
    return normalized


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
