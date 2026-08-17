from __future__ import annotations

import copy

import pytest

from app.analysis.independent_shot_vlm import (
    canonical_sha256,
    seal_independent_shot_vlm_plan,
    seal_independent_shot_vlm_predictions,
)
from app.analysis.independent_shot_vlm_evidence_gate import (
    _canonical_sha256,
    apply_evidence_gate,
    fuse_vlm_with_frozen_auxiliary,
    screen_evidence_gate,
    select_group_held_threshold,
)


def _vlm(*, event_id: str, state: str = "live_field_goal") -> dict[str, object]:
    return {
        "source_video_sha256": "a" * 64,
        "candidate_bundle_sha256": "b" * 64,
        "event_id": event_id,
        "field_goal_state": state,
        "confidence": 0.95,
        "observables": {
            "continuous_live_play": True,
            "controlled_ball_before_release": True,
            "ball_separates_from_hands": True,
            "ball_moves_toward_rim": False,
            "replay_or_highlight": False,
            "free_throw": False,
        },
    }


def _plan_and_predictions(
    *,
    event_ids: tuple[str, ...] = ("e1", "e2"),
    source_video_sha256: str = "a" * 64,
    candidate_bundle_sha256: str = "b" * 64,
    training_manifest_sha256: str = "c" * 64,
) -> tuple[dict[str, object], dict[str, object]]:
    balanced_event_ids = list(event_ids)
    if len(balanced_event_ids) == 1:
        balanced_event_ids.append("target-balance-negative")
    target_examples = [
        {
            "source_video_sha256": source_video_sha256,
            "source_video_filename": "game.mp4",
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "event_id": event_id,
            "start_frame": index * 100,
            "end_frame": index * 100 + 99,
            "source_fps": 30.0,
        }
        for index, event_id in enumerate(balanced_event_ids)
    ]
    support_examples = [
        {
            "source_video_sha256": source,
            "source_video_filename": f"support-{index}.mp4",
            "candidate_bundle_sha256": bundle,
            "event_id": event_id,
            "start_frame": (len(target_examples) + index) * 100,
            "end_frame": (len(target_examples) + index) * 100 + 99,
            "source_fps": 30.0,
        }
        for index, (source, bundle, event_id) in enumerate(
            (
                ("d" * 64, "1" * 64, "support-d-positive"),
                ("d" * 64, "1" * 64, "support-d-negative"),
                ("e" * 64, "2" * 64, "support-e-positive"),
                ("e" * 64, "2" * 64, "support-e-negative"),
            )
        )
    ]
    examples = [*target_examples, *support_examples]
    fusion_screen_contract = {
        "scene_embedding_artifact_sha256": "3" * 64,
        "video_embedding_artifacts": [],
        "predeclared_variants": [{"name": "scene_phase_all", "input_dimension": 3456}],
        "pca_components": 16,
        "regularization_c": 0.01,
        "threshold_selection": ("nested_inner_game_held_minimum_recall_0_85"),
        "promotion_requirements": {
            "minimum_pooled_precision": 0.85,
            "minimum_pooled_recall": 0.85,
            "minimum_per_game_precision": 0.85,
            "minimum_per_game_recall": 0.85,
        },
    }
    plan = seal_independent_shot_vlm_plan(
        {
            "training_manifest_sha256": training_manifest_sha256,
            "selection": {
                "method": "sha256_rank_within_video_and_boolean_class",
                "positive_per_video": 1,
                "negative_per_video": 1,
                "video_count": 3,
                "example_count": len(examples),
            },
            "input_contract": {
                "raw_frames_only": True,
                "labels_or_review_notes_exposed_to_model": False,
                "chronological_even_sampling": True,
                "max_frames": 6,
                "image_width": 704,
            },
            "fusion_screen_contract": fusion_screen_contract,
            "examples": examples,
        }
    )
    predictions = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {
                "name": "independent-vlm-test",
                "independent_from_codex": True,
            },
            "predictions": [
                {
                    **_vlm(
                        event_id=event_id,
                        state=("not_field_goal" if event_id == "e2" else "live_field_goal"),
                    ),
                    "source_video_sha256": source_video_sha256,
                    "candidate_bundle_sha256": candidate_bundle_sha256,
                }
                for event_id in balanced_event_ids
            ]
            + [
                {
                    **_vlm(event_id=str(row["event_id"]), state="not_field_goal"),
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                }
                for row in support_examples
            ],
        }
    )
    return plan, predictions


def _nested_v2_auxiliary(
    *,
    plan: dict[str, object],
    covered_event_ids: tuple[str, ...] | None = None,
    source_video_sha256: str = "a" * 64,
    candidate_bundle_sha256: str = "b" * 64,
) -> dict[str, object]:
    plan_examples = [dict(row) for row in plan["examples"]]
    source_groups = sorted({str(row["source_video_sha256"]) for row in plan_examples})
    target_event_ids = [
        str(row["event_id"]) for row in plan_examples if row["source_video_sha256"] == source_video_sha256
    ]
    covered = target_event_ids if covered_event_ids is None else list(covered_event_ids)

    def positive_prediction_metrics(group_count: int) -> dict[str, object]:
        tp = group_count
        fp = group_count
        precision = 0.5
        recall = 1.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": 0,
            "tn": 0,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall),
        }

    def fold_payload(held_group: str) -> dict[str, object]:
        development_groups = [group for group in source_groups if group != held_group]
        per_game_metrics = {group: positive_prediction_metrics(1) for group in development_groups}
        selection_gate = {
            "promoted": False,
            "pooled": positive_prediction_metrics(len(development_groups)),
            "per_game": per_game_metrics,
        }
        selected_variant = {
            "name": "scene_phase_all",
            "input_dimension": 3456,
            "threshold": 0.7,
            "gate": copy.deepcopy(selection_gate),
            "inner_folds": [
                {
                    "held_game_sha256": inner_held,
                    "fit_game_sha256s": [group for group in development_groups if group != inner_held],
                    "row_count": sum(row["source_video_sha256"] == inner_held for row in plan_examples),
                }
                for inner_held in development_groups
            ],
        }
        held_examples = [
            row
            for row in plan_examples
            if row["source_video_sha256"] == held_group
            and (held_group != source_video_sha256 or row["event_id"] in covered)
        ]
        return {
            "held_game_sha256": held_group,
            "inner_selection_game_sha256s": development_groups,
            "fit_game_sha256s": development_groups,
            "selected_variant": "scene_phase_all",
            "threshold": 0.7,
            "inner_selection_gate": selection_gate,
            "variant_selection": [selected_variant],
            "held_metrics": positive_prediction_metrics(1),
            "oof_predictions": [
                {
                    "source_video_sha256": str(row["source_video_sha256"]),
                    "candidate_bundle_sha256": str(row["candidate_bundle_sha256"]),
                    "event_id": str(row["event_id"]),
                    "probability": 0.8,
                }
                for row in held_examples
            ],
        }

    artifact: dict[str, object] = {
        "schema_version": "agu.shot-validity-scene-fusion-screen.v2",
        "purpose": "scene_video_fusion_screening_training_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "selection_protocol": "nested_outer_game_held_v2",
        "threshold_selection": "nested_inner_game_held_minimum_recall_0_85",
        "random_seed": 0,
        "plan_sha256": plan["plan_sha256"],
        "training_manifest_sha256": plan["training_manifest_sha256"],
        "scene_embedding_artifact_sha256": "3" * 64,
        "video_embedding_artifact_sha256s": [],
        "scene_subset_embedding_artifact_sha256": "5" * 64,
        "video_subset_embedding_artifact_sha256s": [],
        "subset_selection_protocol": ("sealed_independent_shot_vlm_plan_three_part_key_order_v1"),
        "pca_components": 16,
        "regularization_c": 0.01,
        "promotion_requirements": {
            "minimum_pooled_precision": 0.85,
            "minimum_pooled_recall": 0.85,
            "minimum_per_game_precision": 0.85,
            "minimum_per_game_recall": 0.85,
        },
        "predeclared_variants": [{"name": "scene_phase_all", "input_dimension": 3456}],
        "fusion_screen_contract": copy.deepcopy(plan["fusion_screen_contract"]),
        "fusion_screen_contract_sha256": _canonical_sha256(plan["fusion_screen_contract"]),
        "outer_folds": [fold_payload(group) for group in source_groups],
        "gate": {
            "promoted": False,
            "pooled": positive_prediction_metrics(len(source_groups)),
            "per_game": {group: positive_prediction_metrics(1) for group in source_groups},
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def _reseal_auxiliary(artifact: dict[str, object]) -> dict[str, object]:
    sealed = copy.deepcopy(artifact)
    sealed.pop("artifact_sha256", None)
    sealed["artifact_sha256"] = _canonical_sha256(sealed)
    return sealed


def _reseal_vlm_without_verifier(artifact: dict[str, object]) -> dict[str, object]:
    """Forge a self-consistent artifact to test the consumer's defense in depth."""
    sealed = copy.deepcopy(artifact)
    sealed.pop("artifact_sha256", None)
    sealed["artifact_sha256"] = _canonical_sha256(sealed)
    return sealed


def _reseal_plan_without_verifier(artifact: dict[str, object]) -> dict[str, object]:
    """Forge a self-consistent plan to test the consumer's defense in depth."""
    sealed = copy.deepcopy(artifact)
    sealed.pop("plan_sha256", None)
    sealed["plan_sha256"] = canonical_sha256(sealed, hash_field="plan_sha256")
    return sealed


def _fuse(
    *,
    plan: dict[str, object],
    vlm: dict[str, object],
    auxiliary: dict[str, object],
    expected_sha256: str | None = None,
) -> dict[str, object]:
    return fuse_vlm_with_frozen_auxiliary(
        plan=plan,
        vlm_predictions=vlm,
        auxiliary_artifact=auxiliary,
        expected_auxiliary_artifact_sha256=(
            str(auxiliary["artifact_sha256"]) if expected_sha256 is None else expected_sha256
        ),
    )


def test_group_held_threshold_uses_only_training_groups() -> None:
    rows = [
        {
            "source_video_sha256": "train",
            "candidate_bundle_sha256": "bundle-train",
            "event_id": "t1",
            "probability": 0.9,
            "event_present": True,
        },
        {
            "source_video_sha256": "train",
            "candidate_bundle_sha256": "bundle-train",
            "event_id": "t2",
            "probability": 0.2,
            "event_present": False,
        },
        {
            "source_video_sha256": "held",
            "candidate_bundle_sha256": "bundle-held",
            "event_id": "h1",
            "probability": 0.99,
            "event_present": False,
        },
    ]

    selected = select_group_held_threshold(rows, held_group="held", minimum_recall=0.85)

    assert selected["training_groups"] == ["train"]
    assert selected["threshold"] == pytest.approx(0.9)
    assert selected["target_count"] == 1


def test_apply_gate_rejects_conflicting_or_missing_evidence() -> None:
    accepted = apply_evidence_gate(_vlm(event_id="ok"), auxiliary_probability=0.8, threshold=0.5)
    assert accepted["field_goal_state"] == "live_field_goal"
    assert accepted["decision_reason"] == "vlm_and_auxiliary_confirm"

    replay = _vlm(event_id="replay")
    replay["observables"]["replay_or_highlight"] = True  # type: ignore[index]
    assert apply_evidence_gate(replay, auxiliary_probability=0.8, threshold=0.5)["field_goal_state"] == "unknown"
    assert (
        apply_evidence_gate(_vlm(event_id="missing"), auxiliary_probability=None, threshold=0.5)["field_goal_state"]
        == "unknown"
    )


def test_screen_requires_exact_target_coverage_and_is_not_runtime_consumable() -> None:
    vlm_rows = [_vlm(event_id="e1"), _vlm(event_id="e2", state="not_field_goal")]
    auxiliary_rows = [
        {
            "source_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "b" * 64,
            "event_id": "e1",
            "probability": 0.9,
            "event_present": True,
        },
        {
            "source_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "b" * 64,
            "event_id": "e2",
            "probability": 0.1,
            "event_present": False,
        },
        {
            "source_video_sha256": "c" * 64,
            "candidate_bundle_sha256": "d" * 64,
            "event_id": "train",
            "probability": 0.1,
            "event_present": False,
        },
        {
            "source_video_sha256": "c" * 64,
            "candidate_bundle_sha256": "d" * 64,
            "event_id": "train-pos",
            "probability": 0.8,
            "event_present": True,
        },
    ]
    result = screen_evidence_gate(
        vlm_predictions=vlm_rows,
        auxiliary_oof_rows=auxiliary_rows,
        target_truth={("a" * 64, "b" * 64, "e1"): True, ("a" * 64, "b" * 64, "e2"): False},
        minimum_precision=0.85,
        minimum_recall=0.85,
    )

    assert result["runtime_consumable"] is False
    assert result["codex_runtime_answer_used"] is False
    assert result["metrics"]["evaluated_count"] == 2
    assert result["legacy_provenance_only"] is True
    assert result["promotion_eligible"] is False
    assert result["metrics"]["promotion_eligible"] is False
    assert result["metrics"]["observed_gate_would_have_met_thresholds"] is True
    assert result["metrics"]["true_positive"] == 1

    with pytest.raises(ValueError, match="exactly cover"):
        screen_evidence_gate(
            vlm_predictions=vlm_rows[:1],
            auxiliary_oof_rows=auxiliary_rows,
            target_truth={
                ("a" * 64, "b" * 64, "e1"): True,
                ("a" * 64, "b" * 64, "e2"): False,
            },
        )


def test_screen_requires_auxiliary_coverage_for_every_vlm_event() -> None:
    vlm_rows = [_vlm(event_id="e1"), _vlm(event_id="e2", state="not_field_goal")]
    auxiliary_rows = [
        {
            "source_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "b" * 64,
            "event_id": "e1",
            "probability": 0.9,
            "event_present": True,
        },
        {
            "source_video_sha256": "c" * 64,
            "candidate_bundle_sha256": "d" * 64,
            "event_id": "train",
            "probability": 0.1,
            "event_present": False,
        },
        {
            "source_video_sha256": "c" * 64,
            "candidate_bundle_sha256": "d" * 64,
            "event_id": "train-pos",
            "probability": 0.8,
            "event_present": True,
        },
    ]

    with pytest.raises(ValueError, match="exactly cover VLM events"):
        screen_evidence_gate(
            vlm_predictions=vlm_rows,
            auxiliary_oof_rows=auxiliary_rows,
            target_truth={
                ("a" * 64, "b" * 64, "e1"): True,
                ("a" * 64, "b" * 64, "e2"): False,
            },
        )


def test_screen_keeps_candidate_bundle_in_auxiliary_join_key() -> None:
    first = _vlm(event_id="shared")
    second = _vlm(event_id="shared")
    second["candidate_bundle_sha256"] = "c" * 64
    result = screen_evidence_gate(
        vlm_predictions=[first, second],
        auxiliary_oof_rows=[
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                "event_id": "shared",
                "probability": 0.9,
                "event_present": True,
            },
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "c" * 64,
                "event_id": "shared",
                "probability": 0.1,
                "event_present": False,
            },
            {
                "source_video_sha256": "d" * 64,
                "candidate_bundle_sha256": "e" * 64,
                "event_id": "train-positive",
                "probability": 0.8,
                "event_present": True,
            },
            {
                "source_video_sha256": "d" * 64,
                "candidate_bundle_sha256": "e" * 64,
                "event_id": "train-negative",
                "probability": 0.2,
                "event_present": False,
            },
        ],
        target_truth={
            ("a" * 64, "b" * 64, "shared"): True,
            ("a" * 64, "c" * 64, "shared"): False,
        },
    )

    by_bundle = {row["candidate_bundle_sha256"]: row for row in result["predictions"]}
    assert by_bundle["b" * 64]["field_goal_state"] == "live_field_goal"
    assert by_bundle["c" * 64]["field_goal_state"] == "unknown"


def test_apply_gate_rejects_non_finite_or_out_of_range_scores() -> None:
    with pytest.raises(ValueError, match="auxiliary probability"):
        apply_evidence_gate(_vlm(event_id="bad"), auxiliary_probability=1.1, threshold=0.5)
    with pytest.raises(ValueError, match="auxiliary probability"):
        apply_evidence_gate(_vlm(event_id="bad"), auxiliary_probability=float("nan"), threshold=0.5)

    invalid_confidence = _vlm(event_id="bad-confidence")
    invalid_confidence["confidence"] = float("nan")
    decision = apply_evidence_gate(
        invalid_confidence,
        auxiliary_probability=0.8,
        threshold=0.5,
    )
    assert decision["field_goal_state"] == "unknown"
    assert decision["decision_reason"] == "vlm_confidence_invalid"


def test_apply_gate_abstains_when_vlm_row_is_unavailable() -> None:
    row = _vlm(event_id="unavailable")
    row["available"] = False

    decision = apply_evidence_gate(
        row,
        auxiliary_probability=0.8,
        threshold=0.5,
    )

    assert decision["field_goal_state"] == "unknown"
    assert decision["decision_reason"] == "vlm_unavailable"


def test_apply_gate_rejects_duplicate_required_observables() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        apply_evidence_gate(
            _vlm(event_id="duplicate"),
            auxiliary_probability=0.8,
            threshold=0.5,
            required_observables=("continuous_live_play", "continuous_live_play"),
        )


def test_apply_gate_rejects_noncanonical_required_observable() -> None:
    row = _vlm(event_id="label-observable")
    row["observables"]["event_present"] = True

    with pytest.raises(ValueError, match="canonical visual observables"):
        apply_evidence_gate(
            row,
            auxiliary_probability=0.8,
            threshold=0.5,
            required_observables=("event_present",),
        )


def test_frozen_auxiliary_fusion_accepts_nested_v2_and_fails_closed() -> None:
    plan, vlm = _plan_and_predictions()
    auxiliary = _nested_v2_auxiliary(plan=plan)

    result = _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)

    assert result["runtime_consumable"] is False
    assert result["model"]["raw_target_label_fields_read_by_fuser"] is False
    assert result["model"]["outer_held_metrics_present_in_auxiliary"] is True
    assert result["model"]["outer_held_metrics_used_for_decision"] is False
    assert result["model"]["auxiliary_fit_used_training_labels"] is True
    assert result["model"]["outer_held_labels_used_for_selection"] is False
    assert result["model"]["auxiliary_provenance_verification"] == "structural_only"
    assert result["model"]["formal_promotion_eligible"] is False
    assert result["model"]["selection_protocol"] == "nested_outer_game_held_v2"
    assert result["model"]["input_vlm_prediction_artifact_sha256"] == vlm["artifact_sha256"]
    assert result["coverage"]["missing_auxiliary_event_ids"] == []
    by_event = {row["event_id"]: row for row in result["predictions"]}
    assert by_event["e1"]["field_goal_state"] == "live_field_goal"
    assert by_event["e2"]["field_goal_state"] == "unknown"
    assert all("event_present" not in row for row in result["predictions"])


def test_frozen_auxiliary_fusion_rejects_live_state_marked_unavailable() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    payload = copy.deepcopy(vlm)
    payload.pop("artifact_sha256")
    payload["predictions"][0]["available"] = False
    vlm = _reseal_vlm_without_verifier(payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="availability|unavailable"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_legacy_held_label_threshold_artifact() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    legacy = {
        "schema_version": "agu.shot-validity-scene-fusion-screen.v1",
        "purpose": "scene_video_fusion_screening_training_only",
        "runtime_consumable": False,
        "selection_protocol": "game_held_oof_predeclared_variants",
        "best_variant": {
            "folds": [
                {
                    "held_game_sha256": "a" * 64,
                    "best_precision_at_recall_0_85": {"threshold": 0.7},
                }
            ],
            "oof_predictions": [
                {
                    "source_video_sha256": "a" * 64,
                    "event_id": "e1",
                    "probability": 0.8,
                    "event_present": True,
                }
            ],
        },
    }
    legacy["artifact_sha256"] = _canonical_sha256(legacy)

    with pytest.raises(ValueError, match="nested outer-game-held v2"):
        _fuse(plan=plan, vlm=vlm, auxiliary=legacy)


@pytest.mark.parametrize(
    ("field", "expected_error"),
    (
        ("plan_sha256", "plan"),
        ("training_manifest_sha256", "training manifest"),
        ("candidate_bundle_sha256", "candidate bundle|exactly cover"),
    ),
)
def test_frozen_auxiliary_fusion_fails_closed_on_binding_mismatch(
    field: str,
    expected_error: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    if field == "candidate_bundle_sha256":
        auxiliary["outer_folds"][0]["oof_predictions"][0][field] = "f" * 64
    else:
        auxiliary[field] = "f" * 64
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match=expected_error):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_oof_row_without_candidate_bundle() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    auxiliary["outer_folds"][0]["oof_predictions"][0].pop("candidate_bundle_sha256")
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(
        ValueError,
        match="candidate_bundle_sha256|three-part key|fields are not canonical",
    ):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize(
    "group_field",
    ("inner_selection_game_sha256s", "fit_game_sha256s"),
)
def test_frozen_auxiliary_fusion_rejects_held_game_in_training_groups(
    group_field: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    auxiliary["outer_folds"][0][group_field].append("a" * 64)
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="training groups|held game"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_runtime_auxiliary() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = {"runtime_consumable": True, "artifact_sha256": "a" * 64}
    with pytest.raises(ValueError, match="training-only"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_requires_independently_frozen_hash() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="expected auxiliary artifact SHA-256"):
        _fuse(
            plan=plan,
            vlm=vlm,
            auxiliary=auxiliary,
            expected_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    (
        ("purpose", "forged_runtime_result", "purpose"),
        ("codex_runtime_answer_used", True, "Codex"),
        ("threshold_selection", "outer_label_peeking", "threshold selection"),
        ("subset_selection_protocol", "unordered", "subset selection"),
    ),
)
def test_frozen_auxiliary_fusion_rejects_unsafe_provenance_flags(
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    auxiliary[field] = replacement
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match=expected_error):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_label_bearing_oof_rows() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    auxiliary["outer_folds"][0]["oof_predictions"][0]["event_present"] = True
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="label-bearing|label field|unexpected"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_partial_outer_oof_coverage() -> None:
    plan, vlm = _plan_and_predictions()
    auxiliary = _nested_v2_auxiliary(
        plan=plan,
        covered_event_ids=("e1",),
    )

    with pytest.raises(
        ValueError,
        match=("exactly cover the frozen plan|held metric row count|frozen threshold predictions"),
    ):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_impossible_fold_groups() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    auxiliary["outer_folds"][0]["fit_game_sha256s"] = ["f" * 64]
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="training groups"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize(
    "mutation",
    ("unknown_variant", "threshold_drift", "missing_inner_fold"),
)
def test_frozen_auxiliary_fusion_rejects_variant_selection_drift(
    mutation: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    first_fold = auxiliary["outer_folds"][0]
    if mutation == "unknown_variant":
        first_fold["selected_variant"] = "forged-label-peeking-variant"
    elif mutation == "threshold_drift":
        first_fold["threshold"] = 0.0
    else:
        first_fold["variant_selection"][0]["inner_folds"].pop()
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="variant|threshold|inner folds"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_fusion_contract_drift() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    auxiliary["regularization_c"] = 1.0
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="fusion screen contract"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_inconsistent_variant_gate() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    first_fold = auxiliary["outer_folds"][0]
    first_fold["variant_selection"][0]["gate"]["promoted"] = True
    first_fold["inner_selection_gate"]["promoted"] = True
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="gate promotion"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize("location", ("top_level", "observables"))
def test_frozen_auxiliary_fusion_rejects_vlm_label_field_bypass(
    location: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    payload = copy.deepcopy(vlm)
    payload.pop("artifact_sha256")
    if location == "top_level":
        payload["predictions"][0]["event_present"] = True
    else:
        payload["predictions"][0]["observables"]["ground_truth"] = True
    vlm = _reseal_vlm_without_verifier(payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="label field|observable fields"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_hidden_auxiliary_label_field() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    auxiliary["outer_folds"][0]["variant_selection"][0]["ground_truth"] = [True]
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="label field"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_unfrozen_observable_policy() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="fixed default observables"):
        fuse_vlm_with_frozen_auxiliary(
            plan=plan,
            vlm_predictions=vlm,
            auxiliary_artifact=auxiliary,
            expected_auxiliary_artifact_sha256=str(auxiliary["artifact_sha256"]),
            required_observables=("continuous_live_play",),
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        ("training_manifest", "z" * 64, "training manifest"),
        ("candidate_bundle", "not-a-sha256", "candidate bundle"),
    ),
)
def test_frozen_auxiliary_fusion_rejects_non_sha_bindings(
    field: str,
    value: str,
    expected_error: str,
) -> None:
    kwargs = (
        {
            "training_manifest_sha256": value,
        }
        if field == "training_manifest"
        else {
            "candidate_bundle_sha256": value,
        }
    )
    plan, vlm = _plan_and_predictions(event_ids=("e1",), **kwargs)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match=expected_error):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_string_regularization_value() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    plan_payload = copy.deepcopy(plan)
    plan_payload.pop("plan_sha256")
    plan_payload["fusion_screen_contract"]["regularization_c"] = "0.01"
    plan = seal_independent_shot_vlm_plan(plan_payload)
    vlm_payload = copy.deepcopy(vlm)
    vlm_payload.pop("artifact_sha256")
    vlm_payload["plan_sha256"] = plan["plan_sha256"]
    vlm = seal_independent_shot_vlm_predictions(vlm_payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="regularization C"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_codex_model_provenance_reversal() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    payload = copy.deepcopy(vlm)
    payload.pop("artifact_sha256")
    payload["model"] = {
        "name": "codex-generated-answer",
        "independent_from_codex": False,
    }
    vlm = seal_independent_shot_vlm_predictions(payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="independent from Codex"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize(
    ("container", "field"),
    (
        ("model", "label"),
        ("prediction", "targetLabel"),
        ("auxiliary", "eventPresent"),
        ("auxiliary", "event present"),
        ("auxiliary", "event.present"),
        ("auxiliary", "y_true"),
        ("auxiliary", "goldLabel"),
        ("auxiliary", "actual"),
        ("auxiliary", "class"),
        ("auxiliary", "is_positive"),
    ),
)
def test_frozen_auxiliary_fusion_rejects_normalized_label_aliases(
    container: str,
    field: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    if container == "model":
        vlm_payload = copy.deepcopy(vlm)
        vlm_payload.pop("artifact_sha256")
        vlm_payload["model"][field] = True
        vlm = _reseal_vlm_without_verifier(vlm_payload)
    elif container == "prediction":
        vlm_payload = copy.deepcopy(vlm)
        vlm_payload.pop("artifact_sha256")
        vlm_payload["predictions"][0][field] = True
        vlm = _reseal_vlm_without_verifier(vlm_payload)
    else:
        auxiliary["outer_folds"][0]["variant_selection"][0][field] = True
        auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="label field"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize(
    "mutation",
    (
        "boolean_seed",
        "string_threshold",
        "boolean_probability",
        "boolean_row_count",
        "string_metric",
    ),
)
def test_frozen_auxiliary_fusion_rejects_producer_impossible_json_types(
    mutation: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    if mutation == "boolean_seed":
        auxiliary["random_seed"] = False
    elif mutation == "string_threshold":
        fold = auxiliary["outer_folds"][0]
        fold["threshold"] = "0.7"
        fold["variant_selection"][0]["threshold"] = "0.7"
    elif mutation == "boolean_probability":
        auxiliary["outer_folds"][0]["oof_predictions"][0]["probability"] = True
    elif mutation == "boolean_row_count":
        auxiliary["outer_folds"][0]["variant_selection"][0]["inner_folds"][0]["row_count"] = True
    else:
        fold = auxiliary["outer_folds"][0]
        fold["variant_selection"][0]["gate"]["pooled"]["f1"] = "0.8"
        fold["inner_selection_gate"]["pooled"]["f1"] = "0.8"
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(
        ValueError,
        match="random seed|threshold|probability|inner folds|metrics",
    ):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize(
    "mutation",
    ("metric_arithmetic", "pooled_count_sum", "per_game_row_count"),
)
def test_frozen_auxiliary_fusion_rejects_inconsistent_metric_counts(
    mutation: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    fold = auxiliary["outer_folds"][0]
    variant_gate = fold["variant_selection"][0]["gate"]
    selected_gate = fold["inner_selection_gate"]
    gates = (variant_gate, selected_gate)
    if mutation == "metric_arithmetic":
        for gate in gates:
            gate["pooled"]["precision"] = 1.0
    elif mutation == "pooled_count_sum":
        for gate in gates:
            gate["pooled"]["tn"] += 1
    else:
        group = next(iter(variant_gate["per_game"]))
        for gate in gates:
            gate["per_game"][group]["tn"] += 1
            gate["pooled"]["tn"] += 1
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="metric|counts|row count"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_float_variant_input_dimension() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    auxiliary["outer_folds"][0]["variant_selection"][0]["input_dimension"] = 3456.0
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="variant.*dimension|variant.*declaration"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_integer_sha_in_plan_contract() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    plan_payload = copy.deepcopy(plan)
    plan_payload.pop("plan_sha256")
    plan_payload["fusion_screen_contract"]["scene_embedding_artifact_sha256"] = int("3" * 64)
    plan = seal_independent_shot_vlm_plan(plan_payload)
    vlm_payload = copy.deepcopy(vlm)
    vlm_payload.pop("artifact_sha256")
    vlm_payload["plan_sha256"] = plan["plan_sha256"]
    vlm = seal_independent_shot_vlm_predictions(vlm_payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="scene artifact SHA-256"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_integer_vlm_weights_sha() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    payload = copy.deepcopy(vlm)
    payload.pop("artifact_sha256")
    payload["model"]["weights_sha256"] = int("4" * 64)
    vlm = seal_independent_shot_vlm_predictions(payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="weights SHA-256"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize(
    ("container", "field"),
    (
        ("example", "prior_answers"),
        ("example", "verdict"),
        ("example", "review_state"),
        ("example", "answer"),
        ("top", "prior_answers"),
        ("selection", "review_state"),
        ("input_contract", "answer"),
    ),
)
def test_frozen_auxiliary_fusion_rejects_noncanonical_plan_fields(
    container: str,
    field: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    plan_payload = copy.deepcopy(plan)
    plan_payload.pop("plan_sha256")
    if container == "example":
        plan_payload["examples"][0][field] = {"result": "positive"}
    elif container == "top":
        plan_payload[field] = {"result": "positive"}
    else:
        plan_payload[container][field] = "shot"
    plan = seal_independent_shot_vlm_plan(plan_payload)
    vlm_payload = copy.deepcopy(vlm)
    vlm_payload.pop("artifact_sha256")
    vlm_payload["plan_sha256"] = plan["plan_sha256"]
    vlm = seal_independent_shot_vlm_predictions(vlm_payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="plan.*fields|canonical"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize("replacement", (True, "false", None))
def test_frozen_auxiliary_fusion_requires_label_hidden_input_contract(
    replacement: object,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    plan_payload = copy.deepcopy(plan)
    plan_payload.pop("plan_sha256")
    if replacement is None:
        plan_payload.pop("input_contract")
    else:
        plan_payload["input_contract"]["labels_or_review_notes_exposed_to_model"] = replacement
    plan = _reseal_plan_without_verifier(plan_payload)
    vlm_payload = copy.deepcopy(vlm)
    vlm_payload.pop("artifact_sha256")
    vlm_payload["plan_sha256"] = plan["plan_sha256"]
    vlm = seal_independent_shot_vlm_predictions(vlm_payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(
        ValueError,
        match="unsafe provenance|input contract|top-level fields",
    ):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_numeric_held_game_sha() -> None:
    plan, vlm = _plan_and_predictions(
        event_ids=("e1",),
        source_video_sha256="1" * 64,
        candidate_bundle_sha256="2" * 64,
    )
    auxiliary = _nested_v2_auxiliary(
        plan=plan,
        source_video_sha256="1" * 64,
        candidate_bundle_sha256="2" * 64,
    )
    fold = next(row for row in auxiliary["outer_folds"] if row["held_game_sha256"] == "1" * 64)
    fold["held_game_sha256"] = int("1" * 64)
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="held game SHA-256"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_rejects_plan_quota_row_count_drift() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    payload = copy.deepcopy(plan)
    payload.pop("plan_sha256")
    payload["selection"]["negative_per_video"] = 2
    plan = seal_independent_shot_vlm_plan(payload)
    vlm_payload = copy.deepcopy(vlm)
    vlm_payload.pop("artifact_sha256")
    vlm_payload["plan_sha256"] = plan["plan_sha256"]
    vlm = seal_independent_shot_vlm_predictions(vlm_payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    with pytest.raises(ValueError, match="per-video quotas"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


def test_frozen_auxiliary_fusion_does_not_treat_selection_quota_as_truth() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    payload = copy.deepcopy(plan)
    payload.pop("plan_sha256")
    payload["selection"]["positive_per_video"] = 2
    payload["selection"]["negative_per_video"] = 0
    plan = seal_independent_shot_vlm_plan(payload)
    vlm_payload = copy.deepcopy(vlm)
    vlm_payload.pop("artifact_sha256")
    vlm_payload["plan_sha256"] = plan["plan_sha256"]
    vlm = seal_independent_shot_vlm_predictions(vlm_payload)
    auxiliary = _nested_v2_auxiliary(plan=plan)

    fused = _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)

    assert fused["model"]["formal_promotion_eligible"] is False
    assert fused["coverage"]["vlm_event_count"] == len(plan["examples"])


def test_frozen_auxiliary_fusion_rejects_inner_gate_class_count_drift() -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    fold = auxiliary["outer_folds"][0]
    gates = (
        fold["inner_selection_gate"],
        fold["variant_selection"][0]["gate"],
    )
    group = next(iter(gates[0]["per_game"]))
    for gate in gates:
        gate["per_game"][group] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 2,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        rows = list(gate["per_game"].values())
        tp = sum(row["tp"] for row in rows)
        fp = sum(row["fp"] for row in rows)
        fn = sum(row["fn"] for row in rows)
        tn = sum(row["tn"] for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        gate["pooled"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        }
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="class counts"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)


@pytest.mark.parametrize("mutation", ("held_vs_top", "threshold_predictions"))
def test_frozen_auxiliary_fusion_rejects_outer_metric_evidence_drift(
    mutation: str,
) -> None:
    plan, vlm = _plan_and_predictions(event_ids=("e1",))
    auxiliary = _nested_v2_auxiliary(plan=plan)
    fold = auxiliary["outer_folds"][0]
    if mutation == "held_vs_top":
        fold["held_metrics"] = {
            "tp": 0,
            "fp": 0,
            "fn": 1,
            "tn": 1,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    else:
        for row in fold["oof_predictions"]:
            row["probability"] = 0.6
    auxiliary = _reseal_auxiliary(auxiliary)

    with pytest.raises(ValueError, match="held metrics|threshold predictions"):
        _fuse(plan=plan, vlm=vlm, auxiliary=auxiliary)
