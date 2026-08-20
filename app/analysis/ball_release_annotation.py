"""Hash-bound offline annotation contracts for ball-release geometry."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

PLAN_SCHEMA = "agu.ball-release-hard-review-plan.v1"
REVIEW_SCHEMA = "agu.ball-release-offline-review.v1"
CORRECTION_SCHEMA = "agu.ball-release-label-corrections.v1"
FUSION_SCREEN_SCHEMA = "agu.shot-validity-scene-fusion-screen.v1"
FRAME_FRACTIONS = (0.05, 0.1625, 0.275, 0.3875, 0.5, 0.6125, 0.725, 0.8375, 0.95)
CONFIDENCE_VALUES = {"high", "medium", "low", "uncertain"}


def build_hard_review_plan(
    *,
    training_manifest_sha256: str,
    scene_artifact_sha256: str,
    fusion_artifact_sha256: str,
    scene_examples: Sequence[Mapping[str, Any]],
    oof_predictions: Sequence[Mapping[str, Any]],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    per_game_per_class: int = 4,
    excluded_event_keys: set[tuple[str, str]] | None = None,
    excluded_plan_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """Select balanced cross-game hard examples while hiding their labels."""

    if per_game_per_class < 1:
        raise ValueError("per-game class sample count must be positive")

    def scene_key(row: Mapping[str, Any]) -> tuple[str, str]:
        return str(row["source_video_sha256"]), str(row["event_id"])

    scene_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in scene_examples:
        row_key = scene_key(row)
        if row_key in scene_lookup:
            raise ValueError("scene examples must have unique source/event keys")
        scene_lookup[row_key] = row
    prediction_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in oof_predictions:
        row_key = scene_key(row)
        if row_key in prediction_lookup:
            raise ValueError("OOF predictions must have unique source/event keys")
        prediction_lookup[row_key] = row
    if set(scene_lookup) != set(prediction_lookup):
        raise ValueError("scene examples and OOF predictions must align")

    excluded = excluded_event_keys or set()
    grouped: dict[str, dict[bool, list[tuple[float, str]]]] = defaultdict(
        lambda: {False: [], True: []}
    )
    for row_key, scene_row in scene_lookup.items():
        prediction = prediction_lookup[row_key]
        label = scene_row.get("event_present")
        if not isinstance(label, bool) or prediction.get("event_present") is not label:
            raise ValueError("OOF labels must match boolean scene labels")
        probability = float(prediction["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("OOF probability must be between zero and one")
        if row_key in excluded:
            continue
        grouped[row_key[0]][label].append((probability, row_key[1]))

    selected: list[tuple[str, str]] = []
    for game in sorted(grouped):
        positives = sorted(grouped[game][True], key=lambda row: (row[0], row[1]))
        negatives = sorted(
            grouped[game][False], key=lambda row: (-row[0], row[1])
        )
        if (
            len(positives) < per_game_per_class
            or len(negatives) < per_game_per_class
        ):
            raise ValueError("each game must contain enough examples from both classes")
        selected.extend((game, event_id) for _probability, event_id in positives[:per_game_per_class])
        selected.extend((game, event_id) for _probability, event_id in negatives[:per_game_per_class])

    output_examples = []
    for game, event_id in sorted(selected):
        scene_row = scene_lookup[(game, event_id)]
        bundle_sha = str(scene_row["candidate_bundle_sha256"])
        event_key = (game, bundle_sha, event_id)
        if event_key not in events:
            raise ValueError("selected review event is missing from candidate bundles")
        event = events[event_key]
        start_frame = int(event["start_frame"])
        end_frame = int(event["end_frame"])
        if end_frame < start_frame:
            raise ValueError("review event ends before it starts")
        span = end_frame - start_frame
        output_examples.append(
            {
                "review_id": f"ball-release-{len(output_examples) + 1:04d}",
                "source_video_sha256": game,
                "candidate_bundle_sha256": bundle_sha,
                "event_id": event_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_numbers": [
                    int(round(start_frame + span * fraction))
                    for fraction in FRAME_FRACTIONS
                ],
            }
        )
    count_per_class = len(grouped) * per_game_per_class
    return seal_ball_release_plan(
        {
            "purpose": "codex_offline_ball_release_training_annotation_plan",
            "annotation_scope": "ball_visibility_path_and_release_geometry_only",
            "training_manifest_sha256": training_manifest_sha256,
            "scene_artifact_sha256": scene_artifact_sha256,
            "fusion_artifact_sha256": fusion_artifact_sha256,
            "selection_protocol": (
                "per_game_lowest_oof_positive_and_highest_oof_negative_hidden_from_reviewer"
            ),
            "excluded_plan_sha256s": sorted(set(excluded_plan_sha256s)),
            "excluded_examples": len(excluded),
            "per_game_per_class": per_game_per_class,
            "frame_fractions": list(FRAME_FRACTIONS),
            "selection_summary": {
                "games": len(grouped),
                "examples": len(output_examples),
                "positive_hard_examples": count_per_class,
                "negative_hard_examples": count_per_class,
            },
            "examples": output_examples,
        }
    )


def seal_ball_release_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = PLAN_SCHEMA
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact.pop("artifact_sha256", None)
    _validate_plan(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_scene_fusion_screen(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the training-only OOF source used to select hard examples."""

    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if claimed != _canonical_sha256(artifact):
        raise ValueError("scene-fusion screen hash mismatch")
    if artifact.get("schema_version") != FUSION_SCREEN_SCHEMA:
        raise ValueError("unsupported scene-fusion screen schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("scene-fusion screen must remain training-only")
    if not artifact.get("training_manifest_sha256") or not artifact.get(
        "scene_embedding_artifact_sha256"
    ):
        raise ValueError("scene-fusion screen provenance is incomplete")
    best = artifact.get("best_variant")
    if not isinstance(best, Mapping):
        raise ValueError("scene-fusion screen requires a best variant")
    predictions = best.get("oof_predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("scene-fusion best variant requires OOF predictions")
    artifact["artifact_sha256"] = claimed
    return artifact


def verify_ball_release_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_plan(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("ball-release plan hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_ball_release_review(
    payload: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    verified_plan = verify_ball_release_plan(plan)
    artifact = dict(payload)
    artifact["schema_version"] = REVIEW_SCHEMA
    artifact["purpose"] = "codex_offline_ball_release_training_annotation"
    artifact["annotation_scope"] = "ball_visibility_path_and_release_geometry_only"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact.pop("artifact_sha256", None)
    _validate_review(artifact, plan=verified_plan)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_ball_release_review(
    payload: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    verified_plan = verify_ball_release_plan(plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_review(artifact, plan=verified_plan)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("ball-release review hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def derive_ball_release_label_corrections(
    *,
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
    scene_artifact_sha256: str,
    scene_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive conservative training-label overrides from sealed visual evidence."""

    verified_plan = verify_ball_release_plan(plan)
    verified_review = verify_ball_release_review(review, plan=verified_plan)
    labels = {_row_key(row): row.get("event_present") for row in scene_examples}
    if any(not isinstance(value, bool) for value in labels.values()):
        raise ValueError("scene examples require boolean labels")
    decisions = []
    changed = 0
    resolved = 0
    for row in verified_review["reviews"]:
        row_key = _row_key(row)
        if row_key not in labels:
            raise ValueError("review is missing from the scene examples")
        corrected, reason = _conservative_corrected_label(row)
        original = bool(labels[row_key])
        if corrected is not None:
            resolved += 1
            changed += corrected is not original
        decisions.append(
            {
                "source_video_sha256": row_key[0],
                "candidate_bundle_sha256": row_key[1],
                "event_id": row_key[2],
                "original_event_present": original,
                "corrected_event_present": corrected,
                "reason": reason,
                "review_confidence": row["review_confidence"],
            }
        )
    payload = {
        "schema_version": CORRECTION_SCHEMA,
        "purpose": "shot_validity_training_label_correction_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "plan_sha256": verified_plan["artifact_sha256"],
        "review_sha256": verified_review["artifact_sha256"],
        "scene_artifact_sha256": scene_artifact_sha256,
        "policy": (
            "exclude_free_throw_replay_or_stoppage; "
            "accept_visible_live_release_toward_rim; "
            "reject_high_confidence_no_release; otherwise_unresolved"
        ),
        "summary": {
            "reviewed": len(decisions),
            "resolved": resolved,
            "changed": changed,
            "unresolved": len(decisions) - resolved,
        },
        "decisions": decisions,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def verify_ball_release_label_corrections(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if claimed != _canonical_sha256(artifact):
        raise ValueError("ball-release label-correction hash mismatch")
    if artifact.get("schema_version") != CORRECTION_SCHEMA:
        raise ValueError("unsupported ball-release label-correction schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("ball-release label corrections must remain training-only")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("Codex runtime answers are prohibited")
    artifact["artifact_sha256"] = claimed
    return artifact


def apply_ball_release_label_corrections(
    examples: Sequence[Mapping[str, Any]],
    *,
    corrections: Mapping[str, Any],
) -> list[dict[str, Any]]:
    verified = verify_ball_release_label_corrections(corrections)
    overrides = {
        _row_key(row): row["corrected_event_present"]
        for row in verified["decisions"]
        if isinstance(row.get("corrected_event_present"), bool)
    }
    output = []
    matched = set()
    for source in examples:
        row = dict(source)
        row_key = _row_key(row)
        if row_key in overrides:
            row["event_present"] = overrides[row_key]
            matched.add(row_key)
        output.append(row)
    if matched != set(overrides):
        raise ValueError("label corrections do not align with embedding examples")
    return output


def _conservative_corrected_label(
    review: Mapping[str, Any],
) -> tuple[bool | None, str]:
    if review.get("free_throw_formation") is True:
        return False, "free_throw_exclusion"
    if review.get("replay_or_stoppage") is True:
        return False, "replay_or_stoppage_exclusion"
    if (
        review.get("release_observed") is True
        and review.get("ball_moves_toward_rim") is True
    ):
        return True, "live_release_toward_rim"
    if (
        review.get("release_observed") is False
        and review.get("review_confidence") == "high"
    ):
        return False, "high_confidence_no_release"
    return None, "unresolved_visual_evidence"


def _validate_plan(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported ball-release plan schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("ball-release plan must remain training-only")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("Codex runtime answers are prohibited")
    if tuple(artifact.get("frame_fractions") or ()) != FRAME_FRACTIONS:
        raise ValueError("ball-release plan frame fractions changed")
    if not all(
        artifact.get(field)
        for field in (
            "training_manifest_sha256",
            "scene_artifact_sha256",
            "fusion_artifact_sha256",
        )
    ):
        raise ValueError("ball-release plan provenance is incomplete")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("ball-release plan requires examples")
    keys = []
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("ball-release plan example must be an object")
        if "event_present" in row or "probability" in row:
            raise ValueError("review examples must not expose training labels or scores")
        frame_numbers = row.get("frame_numbers")
        if not isinstance(frame_numbers, list) or len(frame_numbers) != len(
            FRAME_FRACTIONS
        ):
            raise ValueError("review example must contain every planned frame")
        if frame_numbers != sorted(frame_numbers):
            raise ValueError("review frames must be chronological")
        keys.append(_row_key(row))
    if len(keys) != len(set(keys)):
        raise ValueError("ball-release review examples must be unique")
    summary = artifact.get("selection_summary")
    if not isinstance(summary, Mapping) or int(summary.get("examples", 0)) != len(
        examples
    ):
        raise ValueError("ball-release selection summary is inconsistent")


def _validate_review(
    artifact: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> None:
    if artifact.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("unsupported ball-release review schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("ball-release reviews must remain training-only")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("Codex runtime answers are prohibited")
    if artifact.get("plan_sha256") != plan["artifact_sha256"]:
        raise ValueError("ball-release review plan hash mismatch")
    reviews = artifact.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("ball-release reviews must be a list")
    planned = {_row_key(row): row for row in plan["examples"]}
    reviewed = {_row_key(row): row for row in reviews if isinstance(row, Mapping)}
    if set(reviewed) != set(planned) or len(reviewed) != len(reviews):
        raise ValueError("ball-release reviews must exactly cover the plan")
    boolean_or_null_fields = (
        "release_observed",
        "ball_moves_toward_rim",
        "rim_arrival_observed",
        "free_throw_formation",
        "replay_or_stoppage",
    )
    for row_key, row in reviewed.items():
        for field in boolean_or_null_fields:
            if row.get(field) not in (True, False, None):
                raise ValueError(f"{field} must be boolean or null")
        if row.get("review_confidence") not in CONFIDENCE_VALUES:
            raise ValueError("unsupported ball-release review confidence")
        observations = row.get("frame_observations")
        if not isinstance(observations, list):
            raise ValueError("frame observations must be a list")
        planned_frames = planned[row_key]["frame_numbers"]
        if [item.get("frame") for item in observations] != planned_frames:
            raise ValueError("frame observations must exactly cover planned frames")
        for observation in observations:
            if observation.get("ball_visible") not in (True, False, None):
                raise ValueError("ball visibility must be boolean or null")
            if observation.get("selected_path_is_ball") not in (True, False, None):
                raise ValueError("path correctness must be boolean or null")
            bbox = observation.get("ball_bbox")
            if bbox is not None:
                if not isinstance(bbox, Mapping) or set(bbox) != {
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                }:
                    raise ValueError("ball bbox must use x1/y1/x2/y2")
                if float(bbox["x2"]) <= float(bbox["x1"]) or float(
                    bbox["y2"]
                ) <= float(bbox["y1"]):
                    raise ValueError("ball bbox must have positive area")


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_video_sha256") or ""),
        str(row.get("candidate_bundle_sha256") or ""),
        str(row.get("event_id") or ""),
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
