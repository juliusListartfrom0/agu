"""Label-free fixed-threshold veto for independent shot-VLM evidence.

This is a deliberately small, fail-closed research contract.  A VLM positive
is accepted only when a separately produced transfer score clears a fixed
threshold on the exact same frozen-plan key.  All other rows abstain.  The
threshold is never learned here, target labels are never read during the
screen, and the artifact cannot be consumed by AGU runtime or training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.independent_shot_auxiliary_transfer import (
    verify_transfer_artifact,
)
from app.analysis.independent_shot_vlm import (
    verify_independent_shot_vlm_predictions,
)

TRANSFER_VETO_SCHEMA = "agu.independent-shot-vlm-transfer-veto.v1"
TRANSFER_VETO_EVALUATION_SCHEMA = "agu.independent-shot-vlm-transfer-veto-evaluation.v1"
_KEY_FIELDS = ("source_video_sha256", "candidate_bundle_sha256", "event_id")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while excluding its own digest field."""

    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_label_free_transfer_veto(
    *,
    vlm_predictions: Mapping[str, Any],
    auxiliary_transfer: Mapping[str, Any],
    transfer_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a fixed-threshold, abstaining VLM/transfer screen.

    The input artifacts must cover exactly the same plan examples and carry
    the same plan hash.  No target labels are accepted by this function.
    """

    threshold = _unit(transfer_threshold, "transfer threshold")
    vlm = verify_independent_shot_vlm_predictions(vlm_predictions)
    transfer = verify_transfer_artifact(auxiliary_transfer)
    vlm_by_key = _index_rows(vlm["predictions"], "VLM predictions")
    transfer_by_key = _index_rows(
        transfer["transfer_predictions"], "auxiliary transfer predictions"
    )
    if set(vlm_by_key) != set(transfer_by_key):
        raise ValueError("transfer-veto inputs must exactly cover the frozen plan")
    vlm_plan_sha = _required_sha(vlm.get("plan_sha256"), "VLM plan hash")
    transfer_plan = transfer.get("plan")
    if not isinstance(transfer_plan, Mapping):
        raise ValueError("transfer-veto auxiliary plan is missing")
    transfer_plan_sha = _required_sha(
        transfer_plan.get("plan_sha256"), "auxiliary plan hash"
    )
    if vlm_plan_sha != transfer_plan_sha:
        raise ValueError("transfer-veto inputs do not share the same frozen plan")

    rows: list[dict[str, Any]] = []
    vlm_live_count = 0
    transfer_above_threshold_count = 0
    accepted_live_count = 0
    vetoed_live_count = 0
    unknown_count = 0
    score_total = 0.0
    for key in sorted(vlm_by_key):
        vlm_row = vlm_by_key[key]
        transfer_row = transfer_by_key[key]
        vlm_state = str(vlm_row.get("field_goal_state") or "unknown")
        vlm_live = vlm_state == "live_field_goal"
        score = _unit(transfer_row.get("transfer_score"), "transfer score")
        above = score >= threshold
        accepted = vlm_live and above
        vlm_live_count += int(vlm_live)
        transfer_above_threshold_count += int(above)
        accepted_live_count += int(accepted)
        vetoed_live_count += int(vlm_live and not above)
        unknown_count += int(not accepted)
        score_total += score
        if accepted:
            state = "live_field_goal"
            reason = "vlm_and_transfer_above_fixed_threshold"
        elif vlm_live:
            state = "unknown"
            reason = "transfer_below_fixed_threshold"
        else:
            state = "unknown"
            reason = "vlm_not_live"
        rows.append(
            {
                **{field: vlm_row[field] for field in _KEY_FIELDS},
                "field_goal_state": state,
                "confidence": (
                    float(vlm_row.get("confidence", 0.0)) if accepted else 0.0
                ),
                "vlm_field_goal_state": vlm_state,
                "transfer_score": score,
                "threshold": threshold,
                "decision_reason": reason,
                "runtime_consumable": False,
                "codex_runtime_answer_used": False,
            }
        )

    event_count = len(rows)
    artifact: dict[str, Any] = {
        "schema_version": TRANSFER_VETO_SCHEMA,
        "purpose": "offline_label_free_transfer_veto_screen",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "contracts": {
            "vlm_plan_sha256": vlm_plan_sha,
            "auxiliary_plan_sha256": transfer_plan_sha,
            "plan_contract_hashes_equal": True,
            "transfer_threshold": threshold,
            "threshold_selected": False,
            "oof_evidence": False,
        },
        "summary": {
            "event_count": event_count,
            "vlm_live_count": vlm_live_count,
            "transfer_above_threshold_count": transfer_above_threshold_count,
            "accepted_live_count": accepted_live_count,
            "vetoed_live_count": vetoed_live_count,
            "unknown_count": unknown_count,
            "transfer_score_mean": score_total / event_count if event_count else 0.0,
            "labels_read": False,
            "oof_evidence": False,
        },
        "decision": {
            "promotion_eligible": False,
            "fusion_output_emitted": False,
            "reason": (
                "Fixed threshold is descriptive only; no target labels or OOF "
                "calibration were used, so retain this as research evidence."
            ),
        },
        "rows": rows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return verify_transfer_veto_artifact(artifact)


def evaluate_transfer_veto(
    artifact: Mapping[str, Any],
    *,
    truth: Mapping[tuple[str, str, str], bool],
) -> dict[str, Any]:
    """Evaluate a sealed veto only after inference, never for calibration."""

    verified = verify_transfer_veto_artifact(artifact)
    rows = verified["rows"]
    keys = {_key(row) for row in rows}
    if set(truth) != keys:
        raise ValueError("transfer-veto truth must exactly cover inference rows")
    tp = fp = fn = tn = unknown = 0
    for row in rows:
        key = _key(row)
        predicted = row["field_goal_state"] == "live_field_goal"
        label = bool(truth[key])
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
        unknown += int(row["field_goal_state"] == "unknown")
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    result: dict[str, Any] = {
        "schema_version": TRANSFER_VETO_EVALUATION_SCHEMA,
        "purpose": "offline_post_inference_transfer_veto_evaluation",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "veto_artifact_sha256": verified["artifact_sha256"],
        "evaluated_count": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "unknown": unknown,
        "precision": precision,
        "recall": recall,
        "labels_used_after_inference": True,
        "promotion_eligible": False,
        "reason": "Post-inference diagnostic only; labels cannot select a threshold or create OOF evidence.",
    }
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def verify_transfer_veto_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify fail-closed provenance and the absence of target labels."""

    artifact = dict(payload)
    if artifact.get("schema_version") != TRANSFER_VETO_SCHEMA:
        raise ValueError("unsupported transfer-veto schema")
    if artifact.get("purpose") != "offline_label_free_transfer_veto_screen":
        raise ValueError("transfer-veto artifact must be research-only")
    for field in ("runtime_consumable", "training_eligible", "codex_runtime_answer_used"):
        if artifact.get(field) is not False:
            raise ValueError(f"transfer-veto {field} must be false")
    if canonical_sha256(artifact) != artifact.get("artifact_sha256"):
        raise ValueError("transfer-veto artifact hash mismatch")
    contracts = artifact.get("contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError("transfer-veto contracts are required")
    if contracts.get("plan_contract_hashes_equal") is not True:
        raise ValueError("transfer-veto requires equal frozen plan contracts")
    if contracts.get("threshold_selected") is not False:
        raise ValueError("transfer-veto threshold must not be selected here")
    if contracts.get("oof_evidence") is not False:
        raise ValueError("transfer-veto cannot claim OOF evidence")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("transfer-veto rows are required")
    _index_rows(rows, "transfer-veto rows")
    forbidden = {"event_present", "label", "ground_truth", "target", "review_note"}
    for row in rows:
        if forbidden.intersection(row):
            raise ValueError("transfer-veto rows must not contain target labels")
        _unit(row.get("transfer_score"), "transfer score")
        _unit(row.get("threshold"), "transfer threshold")
        if row.get("runtime_consumable") is not False:
            raise ValueError("transfer-veto row runtime boundary is invalid")
    summary = artifact.get("summary")
    decision = artifact.get("decision")
    if not isinstance(summary, Mapping) or summary.get("labels_read") is not False:
        raise ValueError("transfer-veto summary must remain label-free")
    if summary.get("oof_evidence") is not False:
        raise ValueError("transfer-veto summary cannot claim OOF evidence")
    if not isinstance(decision, Mapping) or decision.get("promotion_eligible") is not False:
        raise ValueError("transfer-veto cannot be promotion eligible")
    if decision.get("fusion_output_emitted") is not False:
        raise ValueError("transfer-veto cannot emit a runtime fusion output")
    return artifact


def _index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError(f"{label} are required")
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} must be mappings")
        key = _key(row)
        if key in indexed:
            raise ValueError(f"{label} contain duplicate keys")
        indexed[key] = row
    return indexed


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    key = tuple(str(row.get(field) or "") for field in _KEY_FIELDS)
    if not all(key):
        raise ValueError("transfer-veto row has an incomplete key")
    return key


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
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return parsed


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
