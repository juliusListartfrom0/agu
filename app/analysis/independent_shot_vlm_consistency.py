"""Label-free cross-model consistency screening for independent shot evidence.

This module joins an independent VLM prediction artifact with an auxiliary
transfer screen by the exact frozen-plan example key.  It is deliberately a
diagnostic only: transfer scores are uncalibrated, the transfer artifact has
no OOF rows, and the result can never be consumed by AGU runtime or training.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.independent_shot_auxiliary_transfer import (
    verify_transfer_artifact,
)
from app.analysis.independent_shot_vlm import (
    verify_independent_shot_vlm_predictions,
)

CONSISTENCY_SCHEMA = "agu.independent-shot-vlm-auxiliary-consistency.v1"
_KEY_FIELDS = ("source_video_sha256", "candidate_bundle_sha256", "event_id")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while excluding its own digest field."""

    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def screen_cross_model_consistency(
    *,
    vlm_predictions: Mapping[str, Any],
    auxiliary_transfer: Mapping[str, Any],
    default_transfer_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a sealed, label-free VLM/transfer consistency artifact.

    Both inputs are independently verified before joining.  The exact key set
    must match, while a frozen-plan hash mismatch is retained in the result as
    a non-promotable contract failure rather than silently repaired.
    """

    threshold = _unit(default_transfer_threshold, "default transfer threshold")
    vlm = verify_independent_shot_vlm_predictions(vlm_predictions)
    transfer = verify_transfer_artifact(auxiliary_transfer)
    vlm_rows = vlm["predictions"]
    transfer_rows = transfer["transfer_predictions"]
    vlm_by_key = _index_rows(vlm_rows, "VLM predictions")
    transfer_by_key = _index_rows(transfer_rows, "auxiliary transfer predictions")
    if set(vlm_by_key) != set(transfer_by_key):
        missing = sorted(set(vlm_by_key) - set(transfer_by_key))
        extra = sorted(set(transfer_by_key) - set(vlm_by_key))
        raise ValueError(
            "auxiliary transfer predictions must exactly cover VLM events; "
            f"missing={missing}, extra={extra}"
        )

    vlm_plan_sha = _required_sha(vlm.get("plan_sha256"), "VLM plan hash")
    transfer_plan = transfer.get("plan")
    if not isinstance(transfer_plan, Mapping):
        raise ValueError("auxiliary transfer plan reference is required")
    transfer_plan_sha = _required_sha(
        transfer_plan.get("plan_sha256"), "auxiliary transfer plan hash"
    )
    plan_equal = vlm_plan_sha == transfer_plan_sha

    rows: list[dict[str, Any]] = []
    agreement_count = 0
    transfer_positive_count = 0
    vlm_live_count = 0
    score_total = 0.0
    for key in sorted(vlm_by_key):
        vlm_row = vlm_by_key[key]
        transfer_row = transfer_by_key[key]
        vlm_live = str(vlm_row.get("field_goal_state") or "") == "live_field_goal"
        transfer_score = _unit(transfer_row.get("transfer_score"), "transfer score")
        transfer_positive = transfer_score >= threshold
        agreement = vlm_live == transfer_positive
        agreement_count += int(agreement)
        vlm_live_count += int(vlm_live)
        transfer_positive_count += int(transfer_positive)
        score_total += transfer_score
        rows.append(
            {
                **{field: vlm_row[field] for field in _KEY_FIELDS},
                "vlm_field_goal_state": str(vlm_row.get("field_goal_state") or "unknown"),
                "vlm_live_field_goal": vlm_live,
                "transfer_score": transfer_score,
                "transfer_above_default_threshold": transfer_positive,
                "cross_model_agreement": agreement,
            }
        )

    event_count = len(rows)
    summary = {
        "event_count": event_count,
        "vlm_live_count": vlm_live_count,
        "transfer_above_default_threshold_count": transfer_positive_count,
        "cross_model_agreement_count": agreement_count,
        "cross_model_agreement_rate": agreement_count / event_count if event_count else 0.0,
        "transfer_score_mean": score_total / event_count if event_count else 0.0,
        "labels_read": False,
        "oof_evidence": False,
    }
    reason = (
        "Cross-model keys align, but the VLM and transfer artifacts carry different "
        "frozen plan contract hashes and the transfer screen has no OOF calibration; "
        "retain as consistency evidence only."
        if not plan_equal
        else "Cross-model keys and the frozen plan contract align, but the transfer "
        "screen has no OOF calibration; retain as consistency evidence only."
    )
    artifact: dict[str, Any] = {
        "schema_version": CONSISTENCY_SCHEMA,
        "purpose": "offline_label_hidden_cross_model_consistency_screen",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "contracts": {
            "vlm_plan_sha256": vlm_plan_sha,
            "auxiliary_plan_sha256": transfer_plan_sha,
            "key_alignment": True,
            "plan_contract_hashes_equal": plan_equal,
            "default_transfer_threshold": threshold,
            "threshold_selected": False,
        },
        "summary": summary,
        "decision": {
            "promotion_eligible": False,
            "fusion_output_emitted": False,
            "reason": reason,
        },
        "rows": rows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def verify_consistency_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a sealed consistency artifact and its fail-closed properties."""

    artifact = dict(payload)
    if artifact.get("schema_version") != CONSISTENCY_SCHEMA:
        raise ValueError("unsupported independent-shot consistency schema")
    if artifact.get("purpose") != "offline_label_hidden_cross_model_consistency_screen":
        raise ValueError("cross-model consistency must be research-only")
    for field in ("runtime_consumable", "training_eligible", "codex_runtime_answer_used"):
        if artifact.get(field) is not False:
            raise ValueError(f"cross-model consistency {field} must be false")
    if canonical_sha256(artifact) != artifact.get("artifact_sha256"):
        raise ValueError("cross-model consistency artifact hash mismatch")
    decision = artifact.get("decision")
    if not isinstance(decision, Mapping) or decision.get("promotion_eligible") is not False:
        raise ValueError("cross-model consistency cannot be promotion eligible")
    if decision.get("fusion_output_emitted") is not False:
        raise ValueError("cross-model consistency cannot emit a fusion output")
    summary = artifact.get("summary")
    rows = artifact.get("rows")
    if not isinstance(summary, Mapping) or not isinstance(rows, list) or not rows:
        raise ValueError("cross-model consistency requires summary and rows")
    if summary.get("labels_read") is not False or summary.get("oof_evidence") is not False:
        raise ValueError("cross-model consistency must remain label-free and non-OOF")
    _index_rows(rows, "cross-model consistency rows")
    return artifact


def _index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError(f"{label} are required")
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} must be mappings")
        key = tuple(str(row.get(field) or "") for field in _KEY_FIELDS)
        if not all(key):
            raise ValueError(f"{label} contain incomplete keys")
        if key in indexed:
            raise ValueError(f"{label} contain duplicate keys")
        indexed[key] = row
    return indexed


def _required_sha(value: Any, field: str) -> str:
    parsed = str(value or "")
    if len(parsed) != 64 or any(char not in "0123456789abcdef" for char in parsed):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return parsed


def _unit(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be within [0, 1]") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return parsed
