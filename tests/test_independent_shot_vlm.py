from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.analysis.independent_shot_vlm import (
    canonical_sha256,
    derive_frame_sampled_shot_vlm_plan,
    derive_native_video_shot_vlm_plan,
    evaluate_independent_shot_vlm,
    parse_independent_shot_vlm_decision,
    seal_independent_shot_vlm_plan,
    seal_independent_shot_vlm_predictions,
    stable_example_rank,
    verify_independent_shot_vlm_annotation_plan,
    verify_independent_shot_vlm_plan,
    verify_independent_shot_vlm_predictions,
)
from app.analysis.independent_shot_vlm_native import (
    NATIVE_SHOT_PROMPT,
    NATIVE_SHOT_PROMPTS,
    NEGATIVE_FIRST_NATIVE_PROMPT,
    STRICT_RELEASE_NATIVE_PROMPT,
    MlxNativeVideoReviewer,
    TransformersNativeVideoReviewer,
    configure_native_video_processor,
    normalize_native_model_output,
    run_native_video_plan,
    sample_native_frame_indices,
)
from app.analysis.independent_shot_vlm_native import (
    _cache_fingerprint as native_cache_fingerprint,
)
from scripts import evaluate_independent_shot_vlm as evaluate_cli
from scripts.run_independent_shot_vlm import PROMPTS, _cache_fingerprint


def _row(video: str, event: str, start: int) -> dict[str, object]:
    return {
        "source_video_sha256": video,
        "source_video_filename": f"{video}.mp4",
        "candidate_bundle_sha256": f"bundle-{video}",
        "event_id": event,
        "start_frame": start,
        "end_frame": start + 100,
        "source_fps": 30.0,
    }


def _perfect_formal_promotion_case(
    *,
    model: dict[str, object] | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[tuple[str, str, str], bool],
]:
    row = _row("video-a", "positive-a", 0)
    plan = seal_independent_shot_vlm_plan({"selection": {}, "examples": [row]})
    predictions = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": model
            or {
                "name": "independent-test-model",
                "formal_promotion_eligible": True,
                "independent_from_codex": True,
            },
            "predictions": [
                {
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "field_goal_state": "live_field_goal",
                    "confidence": 1.0,
                }
            ],
        }
    )
    key = (
        str(row["source_video_sha256"]),
        str(row["candidate_bundle_sha256"]),
        str(row["event_id"]),
    )
    return plan, predictions, {key: True}


def _evaluate_with_independent_plan_receipt(
    *,
    plan: dict[str, object],
    predictions: dict[str, object],
    truth: dict[tuple[str, str, str], object],
    **kwargs: object,
) -> dict[str, object]:
    evaluation_kwargs: dict[str, object] = {
        "plan": plan,
        "predictions": predictions,
        "truth": truth,
        **kwargs,
    }
    if "expected_plan_sha256" in inspect.signature(evaluate_independent_shot_vlm).parameters:
        evaluation_kwargs["expected_plan_sha256"] = plan["plan_sha256"]
    return evaluate_independent_shot_vlm(**evaluation_kwargs)


def test_plan_is_sealed_label_free_and_tamper_evident() -> None:
    plan = seal_independent_shot_vlm_plan(
        {
            "selection": {"positive_per_video": 1, "negative_per_video": 1},
            "examples": [_row("video-a", "event-1", 10)],
        }
    )
    assert plan["runtime_consumable"] is False
    assert plan["codex_runtime_answer_used"] is False

    leaked = copy.deepcopy(plan)
    leaked["examples"][0]["event_present"] = True
    leaked["plan_sha256"] = ""
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_independent_shot_vlm_plan(leaked)

    leaked["plan_sha256"] = plan["plan_sha256"]
    with pytest.raises(ValueError):
        verify_independent_shot_vlm_plan(leaked)


def test_native_video_plan_derivation_preserves_examples_without_labels() -> None:
    source = seal_independent_shot_vlm_plan(
        {
            "selection": {"method": "frozen"},
            "input_contract": {"max_frames": 6, "image_width": 704},
            "examples": [_row("video-a", "event-1", 10)],
        }
    )

    derived = derive_native_video_shot_vlm_plan(
        source,
        sample_fps=2.0,
        max_pixels=360 * 420,
    )

    assert derived["source_plan_sha256"] == source["plan_sha256"]
    assert derived["examples"] == source["examples"]
    assert derived["input_contract"] == {
        "raw_video_clips_only": True,
        "labels_or_review_notes_exposed_to_model": False,
        "native_temporal_position_encoding": True,
        "sample_fps": 2.0,
        "max_pixels": 360 * 420,
    }
    assert "event_present" not in str(derived)
    with pytest.raises(ValueError, match="sample FPS"):
        derive_native_video_shot_vlm_plan(source, sample_fps=0, max_pixels=100)


def test_frame_sampled_plan_derivation_preserves_examples_without_labels() -> None:
    source = seal_independent_shot_vlm_plan(
        {
            "selection": {"method": "frozen"},
            "training_annotation_sha256": ["a" * 64],
            "input_contract": {"max_frames": 6, "image_width": 704},
            "examples": [_row("video-a", "event-1", 10)],
        }
    )

    derived = derive_frame_sampled_shot_vlm_plan(
        source,
        max_frames=4,
        image_width=512,
    )

    assert derived["source_plan_sha256"] == source["plan_sha256"]
    assert derived["examples"] == source["examples"]
    assert derived["input_contract"] == {
        "raw_frames_only": True,
        "labels_or_review_notes_exposed_to_model": False,
        "chronological_even_sampling": True,
        "max_frames": 4,
        "image_width": 512,
    }
    assert "training_annotation_sha256" not in derived
    assert "event_present" not in str(derived)
    with pytest.raises(ValueError, match="frame-sampled"):
        derive_frame_sampled_shot_vlm_plan(
            source,
            max_frames=1,
            image_width=512,
        )


def test_native_video_resource_probe_is_explicitly_bounded() -> None:
    source = seal_independent_shot_vlm_plan(
        {
            "selection": {"example_count": 3},
            "examples": [
                _row("video-a", "event-1", 10),
                _row("video-a", "event-2", 20),
                _row("video-b", "event-3", 30),
            ],
        }
    )

    probe = derive_native_video_shot_vlm_plan(
        source,
        sample_fps=1.0,
        max_pixels=100_000,
        max_examples=1,
    )

    assert len(probe["examples"]) == 1
    assert probe["selection"]["resource_probe_only"] is True
    assert probe["selection"]["source_example_count"] == 3
    assert probe["selection"]["example_count"] == 1
    assert probe["source_plan_sha256"] == source["plan_sha256"]
    with pytest.raises(ValueError, match="positive"):
        derive_native_video_shot_vlm_plan(
            source,
            sample_fps=1.0,
            max_pixels=100_000,
            max_examples=0,
        )


def test_derived_plan_requires_exact_annotation_source_plan() -> None:
    source = seal_independent_shot_vlm_plan(
        {
            "selection": {"method": "frozen"},
            "training_annotation_sha256": ["a" * 64],
            "examples": [_row("video-a", "event-1", 10)],
        }
    )
    derived = derive_native_video_shot_vlm_plan(
        source,
        sample_fps=2.0,
        max_pixels=151_200,
    )
    unrelated = seal_independent_shot_vlm_plan(
        {"selection": {"method": "other"}, "examples": [_row("video-a", "event-1", 10)]}
    )

    assert verify_independent_shot_vlm_annotation_plan(derived, source)["plan_sha256"] == source["plan_sha256"]
    with pytest.raises(ValueError, match="derived source"):
        verify_independent_shot_vlm_annotation_plan(derived, unrelated)


def test_annotation_plan_may_cover_a_derived_resource_probe_subset() -> None:
    rows = [
        _row("video-a", "event-1", 10),
        _row("video-b", "event-2", 20),
    ]
    source = seal_independent_shot_vlm_plan(
        {
            "training_annotation_sha256": ["a" * 64],
            "selection": {"method": "frozen"},
            "examples": rows,
        }
    )
    derived = derive_native_video_shot_vlm_plan(
        source,
        sample_fps=2.0,
        max_pixels=151_200,
        max_examples=1,
    )

    assert verify_independent_shot_vlm_annotation_plan(derived, source)["plan_sha256"] == source["plan_sha256"]


def test_decision_parser_fails_unknown_and_clamps_confidence() -> None:
    parsed = parse_independent_shot_vlm_decision(
        {
            "field_goal_state": "something else",
            "confidence": 9,
            "continuous_live_play": "yes",
            "free_throw": False,
        }
    )
    assert parsed["field_goal_state"] == "unknown"
    assert parsed["confidence"] == 1.0
    assert parsed["observables"]["continuous_live_play"] is None
    assert parsed["observables"]["free_throw"] is False


def test_evaluation_requires_exact_plan_and_per_video_gate() -> None:
    rows = [
        _row("video-a", "positive-a", 0),
        _row("video-a", "negative-a", 100),
        _row("video-b", "positive-b", 0),
        _row("video-b", "negative-b", 100),
    ]
    plan = seal_independent_shot_vlm_plan({"selection": {}, "examples": rows})
    decisions = []
    states = {
        "positive-a": "live_field_goal",
        "negative-a": "not_field_goal",
        "positive-b": "unknown",
        "negative-b": "not_field_goal",
    }
    for row in rows:
        decisions.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "source_video_sha256",
                        "candidate_bundle_sha256",
                        "event_id",
                    )
                },
                **parse_independent_shot_vlm_decision(
                    {
                        "field_goal_state": states[str(row["event_id"])],
                        "confidence": 0.8,
                    }
                ),
            }
        )
    predictions = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {"name": "independent-test-model"},
            "predictions": decisions,
        }
    )
    truth = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        ): str(row["event_id"]).startswith("positive")
        for row in rows
    }
    evaluation = evaluate_independent_shot_vlm(
        plan=plan,
        predictions=predictions,
        truth=truth,
        expected_prediction_artifact_sha256=predictions["artifact_sha256"],
    )
    assert evaluation["metrics"]["true_positive"] == 1
    assert evaluation["metrics"]["false_negative"] == 1
    assert evaluation["metrics"]["unknown"] == 1
    assert evaluation["metrics"]["promotion_eligible"] is False

    incomplete = copy.deepcopy(predictions)
    incomplete["predictions"].pop()
    incomplete = seal_independent_shot_vlm_predictions(incomplete)
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate_independent_shot_vlm(
            plan=plan,
            predictions=incomplete,
            truth=truth,
        )


def test_evaluation_cannot_promote_structural_only_fused_predictions() -> None:
    rows = [
        _row("video-a", "positive-a", 0),
        _row("video-b", "positive-b", 0),
    ]
    plan = seal_independent_shot_vlm_plan({"selection": {}, "examples": rows})
    predictions = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {
                "name": "structural-only-fused-diagnostic",
                "formal_promotion_eligible": False,
            },
            "predictions": [
                {
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "field_goal_state": "live_field_goal",
                    "confidence": 1.0,
                }
                for row in rows
            ],
        }
    )
    truth = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        ): True
        for row in rows
    }

    evaluation = evaluate_independent_shot_vlm(
        plan=plan,
        predictions=predictions,
        truth=truth,
    )

    assert evaluation["metrics"]["observed_metric_requirements_met"] is True
    assert evaluation["metrics"]["promotion_eligible"] is False
    assert evaluation["metrics"]["input_prediction_formal_promotion_eligible"] is False


@pytest.mark.parametrize("formal_marker", (None, True))
def test_legacy_v1_evaluation_never_promotes_from_marker_or_receipts(
    formal_marker: bool | None,
) -> None:
    row = _row("video-a", "positive-a", 0)
    plan = seal_independent_shot_vlm_plan({"selection": {}, "examples": [row]})
    model: dict[str, object] = {"name": "independent-test-model"}
    if formal_marker is not None:
        model["formal_promotion_eligible"] = formal_marker
    if formal_marker is True:
        model["independent_from_codex"] = True
    predictions = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": model,
            "predictions": [
                {
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "field_goal_state": "live_field_goal",
                    "confidence": 1.0,
                }
            ],
        }
    )
    key = (
        str(row["source_video_sha256"]),
        str(row["candidate_bundle_sha256"]),
        str(row["event_id"]),
    )

    without_receipt = evaluate_independent_shot_vlm(
        plan=plan,
        predictions=predictions,
        truth={key: True},
    )
    with_receipt = _evaluate_with_independent_plan_receipt(
        plan=plan,
        predictions=predictions,
        truth={key: True},
        expected_prediction_artifact_sha256=predictions["artifact_sha256"],
    )

    assert without_receipt["metrics"]["promotion_eligible"] is False
    assert with_receipt["metrics"]["promotion_eligible"] is False
    assert with_receipt["metrics"]["input_provenance_verification"] == ("open_world_structural_only")
    assert with_receipt["metrics"]["input_label_free_verified"] is False
    assert with_receipt["metrics"]["input_codex_independence_verified"] is False
    assert with_receipt["input_provenance_verification"] == ("open_world_structural_only")


def test_prediction_sealer_rejects_explicit_codex_runtime_answer() -> None:
    row = _row("video-a", "positive-a", 0)

    with pytest.raises(ValueError):
        seal_independent_shot_vlm_predictions(
            {
                "plan_sha256": "a" * 64,
                "codex_runtime_answer_used": True,
                "model": {
                    "name": "independent-test-model",
                    "independent_from_codex": True,
                },
                "predictions": [
                    {
                        "source_video_sha256": row["source_video_sha256"],
                        "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                        "event_id": row["event_id"],
                        "field_goal_state": "live_field_goal",
                        "confidence": 1.0,
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "independence_case",
    ("false", "missing", "integer_true"),
)
def test_formal_evaluation_rejects_model_without_exact_codex_independence(
    independence_case: str,
) -> None:
    model: dict[str, object] = {
        "name": "independent-test-model",
        "formal_promotion_eligible": True,
    }
    if independence_case == "false":
        model["independent_from_codex"] = False
    elif independence_case == "integer_true":
        model["independent_from_codex"] = 1
        with pytest.raises(ValueError, match="noncanonical provenance"):
            _perfect_formal_promotion_case(model=model)
        return
    plan, predictions, truth = _perfect_formal_promotion_case(model=model)

    with pytest.raises(ValueError):
        _evaluate_with_independent_plan_receipt(
            plan=plan,
            predictions=predictions,
            truth=truth,
            expected_prediction_artifact_sha256=predictions["artifact_sha256"],
        )


_FORMAL_LABEL_REVIEW_ALIASES = (
    pytest.param(
        {"review-notes": "human review says shot"},
        id="normalized-review-alias",
    ),
    pytest.param({"groundTruth": True}, id="camel-case-label-alias"),
    pytest.param(
        {"metadata": {"review": {"label": True}}},
        id="nested-label-review-alias",
    ),
)


@pytest.mark.parametrize("leaked_fields", _FORMAL_LABEL_REVIEW_ALIASES)
def test_formal_evaluation_rejects_resealed_plan_example_label_aliases(
    leaked_fields: dict[str, object],
) -> None:
    plan, predictions, truth = _perfect_formal_promotion_case()
    leaked_plan = copy.deepcopy(plan)
    leaked_plan["examples"][0].update(copy.deepcopy(leaked_fields))
    leaked_plan["plan_sha256"] = canonical_sha256(
        leaked_plan,
        hash_field="plan_sha256",
    )
    rebound_predictions = seal_independent_shot_vlm_predictions(
        {
            **predictions,
            "plan_sha256": leaked_plan["plan_sha256"],
        }
    )

    with pytest.raises(ValueError):
        _evaluate_with_independent_plan_receipt(
            plan=leaked_plan,
            predictions=rebound_predictions,
            truth=truth,
            expected_prediction_artifact_sha256=rebound_predictions["artifact_sha256"],
        )


@pytest.mark.parametrize("leaked_fields", _FORMAL_LABEL_REVIEW_ALIASES)
def test_formal_evaluation_rejects_resealed_prediction_label_aliases(
    leaked_fields: dict[str, object],
) -> None:
    plan, predictions, truth = _perfect_formal_promotion_case()
    leaked_predictions = copy.deepcopy(predictions)
    leaked_predictions["predictions"][0].update(copy.deepcopy(leaked_fields))
    leaked_predictions["artifact_sha256"] = canonical_sha256(
        leaked_predictions,
        hash_field="artifact_sha256",
    )

    with pytest.raises(ValueError):
        _evaluate_with_independent_plan_receipt(
            plan=plan,
            predictions=leaked_predictions,
            truth=truth,
            expected_prediction_artifact_sha256=leaked_predictions["artifact_sha256"],
        )


@pytest.mark.parametrize(
    "truth_value",
    (
        pytest.param(1, id="integer-one"),
        pytest.param("true", id="string-true"),
        pytest.param(np.bool_(True), id="numpy-bool"),
    ),
)
def test_formal_evaluation_requires_truth_values_to_be_exact_bool(
    truth_value: object,
) -> None:
    plan, predictions, truth = _perfect_formal_promotion_case()
    key = next(iter(truth))

    with pytest.raises(ValueError):
        _evaluate_with_independent_plan_receipt(
            plan=plan,
            predictions=predictions,
            truth={key: truth_value},
            expected_prediction_artifact_sha256=predictions["artifact_sha256"],
        )


@pytest.mark.parametrize(
    "threshold_overrides",
    (
        pytest.param(
            {"minimum_precision": float("nan")},
            id="non-finite-precision",
        ),
        pytest.param(
            {"minimum_recall": float("inf")},
            id="non-finite-recall",
        ),
        pytest.param(
            {"minimum_precision": "0.95"},
            id="non-numeric-precision",
        ),
        pytest.param(
            {"minimum_recall": True},
            id="boolean-recall",
        ),
        pytest.param(
            {"minimum_precision": -0.01},
            id="out-of-range-precision",
        ),
        pytest.param(
            {"minimum_recall": 1.01},
            id="out-of-range-recall",
        ),
        pytest.param(
            {"minimum_precision": 0.94},
            id="weaker-than-formal-precision",
        ),
        pytest.param(
            {"minimum_recall": 0.84},
            id="weaker-than-formal-recall",
        ),
    ),
)
def test_formal_evaluation_rejects_invalid_or_weakened_gate_thresholds(
    threshold_overrides: dict[str, object],
) -> None:
    plan, predictions, truth = _perfect_formal_promotion_case()

    with pytest.raises(ValueError):
        _evaluate_with_independent_plan_receipt(
            plan=plan,
            predictions=predictions,
            truth=truth,
            expected_prediction_artifact_sha256=predictions["artifact_sha256"],
            **threshold_overrides,
        )


def test_legacy_v1_records_expected_plan_receipt_but_never_promotes() -> None:
    plan, predictions, truth = _perfect_formal_promotion_case()

    evaluation = evaluate_independent_shot_vlm(
        plan=plan,
        predictions=predictions,
        truth=truth,
        expected_plan_sha256=plan["plan_sha256"],
        expected_prediction_artifact_sha256=predictions["artifact_sha256"],
    )

    assert evaluation["metrics"]["plan_artifact_receipt_verified"] is True
    assert evaluation["metrics"]["promotion_eligible"] is False


def test_formal_evaluation_without_expected_plan_receipt_is_ineligible() -> None:
    plan, predictions, truth = _perfect_formal_promotion_case()

    evaluation = evaluate_independent_shot_vlm(
        plan=plan,
        predictions=predictions,
        truth=truth,
        expected_prediction_artifact_sha256=predictions["artifact_sha256"],
    )

    assert evaluation["metrics"]["plan_artifact_receipt_verified"] is False
    assert evaluation["metrics"]["promotion_eligible"] is False


def test_evaluation_cli_accepts_independent_expected_plan_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_plan_sha256 = "a" * 64
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            "plan.json",
            "--expected-plan-sha256",
            expected_plan_sha256,
            "--predictions",
            "predictions.json",
            "--annotation",
            "annotations.json",
            "--output",
            "evaluation.json",
        ],
    )

    args = evaluate_cli.parse_args()

    assert args.expected_plan_sha256 == expected_plan_sha256


def test_stable_rank_does_not_depend_on_input_order() -> None:
    first = stable_example_rank("video", "bundle", "event-1")
    second = stable_example_rank("video", "bundle", "event-2")
    assert first == stable_example_rank("video", "bundle", "event-1")
    assert first != second


def test_cache_fingerprint_binds_sampling_and_weights() -> None:
    kwargs = {
        "plan_sha256": "plan",
        "model": "model",
        "model_source": "source",
        "model_revision": "revision",
        "weights_sha256": "weights",
        "max_frames": 6,
        "image_width": 704,
        "context_length": 8192,
    }
    baseline = _cache_fingerprint(**kwargs)
    assert baseline == _cache_fingerprint(**kwargs)
    assert baseline != _cache_fingerprint(**{**kwargs, "max_frames": 12})
    assert baseline != _cache_fingerprint(**{**kwargs, "weights_sha256": "other"})
    assert set(PROMPTS) == {
        "baseline",
        "strict_release_v2",
        "strict_release_v3",
        "negative_first_v4",
    }
    assert "contradiction" in PROMPTS["strict_release_v3"].lower()
    assert "already in flight" in PROMPTS["strict_release_v3"]
    assert "false positive" in PROMPTS["negative_first_v4"].lower()
    assert "every required observable" in PROMPTS["negative_first_v4"].lower()
    assert baseline != _cache_fingerprint(
        **kwargs,
        prompt_sha256=hashlib.sha256(PROMPTS["strict_release_v2"].encode()).hexdigest(),
    )


def test_native_video_processor_enforces_frozen_pixel_budget() -> None:
    video_processor = SimpleNamespace(min_pixels=12_544, max_pixels=940_800)
    processor = SimpleNamespace(video_processor=video_processor)

    configure_native_video_processor(processor, max_pixels=151_200)

    assert video_processor.max_pixels == 151_200
    assert video_processor.min_pixels == 12_544
    with pytest.raises(ValueError, match="video processor"):
        configure_native_video_processor(SimpleNamespace(), max_pixels=151_200)


def test_smolvlm_processor_enforces_frozen_square_video_budget() -> None:
    image_processor = SimpleNamespace(
        size=SimpleNamespace(longest_edge=2048),
        max_image_size={"longest_edge": 512},
        video_sampling={
            "fps": 1,
            "max_frames": 64,
            "video_size": {"longest_edge": 512},
        },
    )
    processor = SimpleNamespace(image_processor=image_processor)

    configure_native_video_processor(processor, max_pixels=65_536)

    assert image_processor.size.longest_edge == 256
    assert image_processor.max_image_size == {"longest_edge": 256}
    assert image_processor.video_sampling == {
        "fps": 1,
        "max_frames": 64,
        "video_size": {"longest_edge": 256},
    }


def test_native_frame_indices_are_even_temporal_samples() -> None:
    indices = sample_native_frame_indices(
        start_frame=100,
        end_frame=219,
        source_fps=30.0,
        sample_fps=2.0,
    )

    assert len(indices) == 8
    assert indices[0] == 100
    assert indices[-1] == 219
    assert indices == sorted(set(indices))
    with pytest.raises(ValueError, match="frame bounds"):
        sample_native_frame_indices(
            start_frame=5,
            end_frame=5,
            source_fps=30.0,
            sample_fps=2.0,
        )


def test_native_output_fails_closed_on_non_json() -> None:
    valid = normalize_native_model_output(
        '{"field_goal_state":"live_field_goal","confidence":0.9,"continuous_live_play":true,"reason":"visible release"}'
    )
    invalid = normalize_native_model_output("<|im_start|>" * 8)

    assert valid["available"] is True
    assert valid["field_goal_state"] == "live_field_goal"
    assert valid["observables"]["continuous_live_play"] is True
    assert invalid["available"] is False
    assert invalid["field_goal_state"] == "unknown"


def test_native_prompt_variants_are_distinct_and_cache_bound() -> None:
    assert set(NATIVE_SHOT_PROMPTS) == {
        "baseline",
        "strict_release_v2",
        "negative_first_v3",
        "compact_json_v1",
    }
    assert NATIVE_SHOT_PROMPTS["baseline"] == NATIVE_SHOT_PROMPT
    assert NATIVE_SHOT_PROMPTS["strict_release_v2"] == STRICT_RELEASE_NATIVE_PROMPT
    assert NATIVE_SHOT_PROMPTS["negative_first_v3"] == NEGATIVE_FIRST_NATIVE_PROMPT
    assert "JSON object" in NATIVE_SHOT_PROMPTS["compact_json_v1"]
    assert "No prose" in NATIVE_SHOT_PROMPTS["compact_json_v1"]
    kwargs = {
        "plan_sha256": "plan",
        "model": {"name": "model", "weights_sha256": "weights"},
        "sample_fps": 2.0,
        "max_pixels": 151_200,
    }
    baseline = native_cache_fingerprint(
        **kwargs,
        prompt=NATIVE_SHOT_PROMPTS["baseline"],
        prompt_variant="baseline",
    )
    strict = native_cache_fingerprint(
        **kwargs,
        prompt=NATIVE_SHOT_PROMPTS["strict_release_v2"],
        prompt_variant="strict_release_v2",
    )
    assert baseline != strict


def test_native_runner_forces_unavailable_decisions_to_unknown(tmp_path: Path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"sealed-video")
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    row = {
        **_row(video_sha, "event-1", 10),
        "source_video_filename": video.name,
    }
    plan = derive_native_video_shot_vlm_plan(
        seal_independent_shot_vlm_plan({"selection": {}, "examples": [row]}),
        sample_fps=2.0,
        max_pixels=151_200,
    )

    artifact = run_native_video_plan(
        plan=plan,
        videos=[video],
        review_window=lambda *_args, **_kwargs: {
            "available": False,
            "field_goal_state": "live_field_goal",
            "confidence": 1.0,
        },
        frame_sampler=lambda *_args, **_kwargs: np.zeros((4, 3, 32, 32), dtype=np.uint8),
        model={
            "name": "test",
            "source": "test",
            "revision": "test",
            "weights_sha256": "test",
        },
    )

    assert artifact["predictions"][0]["field_goal_state"] == "unknown"
    assert artifact["predictions"][0]["available"] is False


def test_native_reviewer_loads_once_and_uses_video_processor_budget() -> None:
    loads: list[str] = []
    calls: list[dict[str, object]] = []
    prompts: list[str] = []
    formatted_image_counts: list[int | None] = []
    processor = SimpleNamespace(video_processor=SimpleNamespace(min_pixels=12_544, max_pixels=940_800))
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"))

    def loader(path: str):
        loads.append(path)
        return model, processor

    def formatter(*args, **kwargs) -> str:
        prompts.append(str(args[2]))
        formatted_image_counts.append(kwargs.get("num_images"))
        return "formatted native-video prompt"

    def generator(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"field_goal_state":"not_field_goal","confidence":0.8}')

    reviewer = MlxNativeVideoReviewer(
        model_path="local-model",
        max_pixels=151_200,
        max_tokens=120,
        prefill_step_size=2048,
        loader=loader,
        formatter=formatter,
        generator=generator,
    )
    frames = np.zeros((4, 3, 32, 32), dtype=np.uint8)
    first = reviewer.review(frames, sample_fps=2.0)
    second = reviewer.review(frames, sample_fps=2.0)

    assert loads == ["local-model"]
    assert processor.video_processor.max_pixels == 151_200
    assert first["field_goal_state"] == "not_field_goal"
    assert second["available"] is True
    assert len(calls) == 2
    assert isinstance(calls[0]["video"], list)
    assert len(calls[0]["video"]) == 1
    assert calls[0]["video"][0] is frames
    assert calls[0]["fps"] == 2.0
    assert calls[0]["prefill_step_size"] == 2048
    assert prompts == [NATIVE_SHOT_PROMPT, NATIVE_SHOT_PROMPT]
    assert formatted_image_counts == [4, 4]


def test_native_reviewer_accepts_strict_prompt_variant() -> None:
    prompts: list[str] = []
    processor = SimpleNamespace(video_processor=SimpleNamespace(min_pixels=12_544, max_pixels=940_800))
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"))

    def formatter(*args, **_kwargs) -> str:
        prompts.append(str(args[2]))
        return "formatted strict native-video prompt"

    reviewer = MlxNativeVideoReviewer(
        model_path="local-model",
        max_pixels=151_200,
        prompt=NATIVE_SHOT_PROMPTS["strict_release_v2"],
        loader=lambda _path: (model, processor),
        formatter=formatter,
        generator=lambda *_args, **_kwargs: SimpleNamespace(text='{"field_goal_state":"unknown","confidence":0.0}'),
    )
    reviewer.review(np.zeros((4, 3, 32, 32), dtype=np.uint8), sample_fps=2.0)

    assert prompts == [STRICT_RELEASE_NATIVE_PROMPT]


def test_native_reviewer_preserves_sanitized_generation_error() -> None:
    processor = SimpleNamespace(video_processor=SimpleNamespace(min_pixels=12_544, max_pixels=940_800))
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"))

    def fail_generation(*_args, **_kwargs):
        raise ValueError("video grid does not match temporal tokens")

    reviewer = MlxNativeVideoReviewer(
        model_path="local-model",
        max_pixels=151_200,
        loader=lambda _path: (model, processor),
        formatter=lambda *_args, **_kwargs: "prompt",
        generator=fail_generation,
    )
    decision = reviewer.review(
        np.zeros((4, 3, 32, 32), dtype=np.uint8),
        sample_fps=2.0,
    )

    assert decision["available"] is False
    assert "video grid does not match temporal tokens" in decision["reason"]


def test_transformers_native_reviewer_loads_once_and_preserves_video_order() -> None:
    loads: list[tuple[str, dict[str, object]]] = []
    processor_loads: list[tuple[str, dict[str, object]]] = []
    generated: list[dict[str, object]] = []

    class Batch(dict[str, object]):
        def to(self, device: str):
            self["moved_to"] = device
            return self

    class Model:
        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            generated.append(kwargs)
            return np.asarray([[10, 11, 12, 13]])

    class Processor:
        def apply_chat_template(self, *_args, **_kwargs):
            return "formatted native-video prompt"

        def __call__(self, **kwargs):
            video = kwargs["videos"][0]
            assert video.shape == (4, 32, 32, 3)
            assert int(video[0, 0, 0, 0]) == 1
            assert int(video[-1, 0, 0, 0]) == 4
            return Batch(input_ids=np.asarray([[10, 11]]))

        def batch_decode(self, values, **_kwargs):
            assert values.tolist() == [[12, 13]]
            return ['{"field_goal_state":"not_field_goal","confidence":0.8}']

    reviewer = TransformersNativeVideoReviewer(
        model_path="local-native-model",
        max_pixels=151_200,
        device="cpu",
        torch_dtype="bfloat16",
        max_tokens=120,
        model_loader=lambda path, **kwargs: loads.append((path, kwargs)) or Model(),
        processor_loader=lambda path, **kwargs: processor_loads.append((path, kwargs)) or Processor(),
    )
    frames = np.stack([np.full((3, 32, 32), value, dtype=np.uint8) for value in range(1, 5)])

    first = reviewer.review(frames, sample_fps=2.0)
    second = reviewer.review(frames, sample_fps=2.0)

    assert len(loads) == 1
    assert len(processor_loads) == 1
    assert loads[0][1]["local_files_only"] is True
    assert first["field_goal_state"] == "not_field_goal"
    assert second["available"] is True
    assert len(generated) == 2
    assert generated[0]["max_new_tokens"] == 120
    assert generated[0]["do_sample"] is False
    assert generated[0]["moved_to"] == "cpu"


def test_native_plan_runner_seals_exact_label_free_coverage(tmp_path: Path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"sealed-video")
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    row = {
        **_row(video_sha, "event-1", 10),
        "source_video_filename": video.name,
    }
    plan = derive_native_video_shot_vlm_plan(
        seal_independent_shot_vlm_plan({"selection": {"method": "frozen"}, "examples": [row]}),
        sample_fps=2.0,
        max_pixels=151_200,
    )
    reviews: list[tuple[tuple[int, ...], float]] = []

    def sampler(*_args, **_kwargs) -> np.ndarray:
        return np.zeros((4, 3, 32, 32), dtype=np.uint8)

    def review(frames: np.ndarray, *, sample_fps: float):
        reviews.append((frames.shape, sample_fps))
        return normalize_native_model_output('{"field_goal_state":"unknown","confidence":0.4}')

    artifact = run_native_video_plan(
        plan=plan,
        videos=[video],
        review_window=review,
        frame_sampler=sampler,
        model={
            "name": "EBQwen2.5-VL-3B-MLX-4bit",
            "source": "GabrieleGiudici/EBQwen2.5-VL-3B",
            "revision": "revision",
            "weights_sha256": "weights",
        },
    )

    assert reviews == [((4, 3, 32, 32), 2.0)]
    assert artifact["plan_sha256"] == plan["plan_sha256"]
    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    assert artifact["predictions"][0]["field_goal_state"] == "unknown"
    assert artifact["prompt_variant"] == "baseline"
    assert artifact["prompt_sha256"] == hashlib.sha256(NATIVE_SHOT_PROMPT.encode()).hexdigest()


def _write_cli_evaluation_inputs(
    tmp_path: Path,
) -> tuple[dict[str, Path], bytes]:
    annotation_path = tmp_path / "annotation.json"
    row = {
        "source_video_sha256": "a" * 64,
        "source_video_filename": "video.mp4",
        "candidate_bundle_sha256": "b" * 64,
        "event_id": "event-1",
        "start_frame": 0,
        "end_frame": 30,
        "source_fps": 30.0,
    }
    annotation_bytes = (
        json.dumps(
            {
                "source_video_sha256": row["source_video_sha256"],
                "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                "examples": [{"event_id": row["event_id"], "event_present": True}],
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    annotation_path.write_bytes(annotation_bytes)
    plan = seal_independent_shot_vlm_plan(
        {
            "selection": {},
            "training_annotation_sha256": [hashlib.sha256(annotation_bytes).hexdigest()],
            "examples": [row],
        }
    )
    predictions = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {
                "name": "independent-test-model",
                "formal_promotion_eligible": True,
                "independent_from_codex": True,
            },
            "predictions": [
                {
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "field_goal_state": "live_field_goal",
                    "confidence": 1.0,
                }
            ],
        }
    )
    plan_path = tmp_path / "plan.json"
    predictions_path = tmp_path / "predictions.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    prediction_bytes = (json.dumps(predictions, sort_keys=True) + "\n").encode()
    predictions_path.write_bytes(prediction_bytes)
    return (
        {
            "plan": plan_path,
            "predictions": predictions_path,
            "annotation": annotation_path,
            "output": tmp_path / "evaluation.json",
        },
        prediction_bytes,
    )


def _observe_read_opens(
    monkeypatch: pytest.MonkeyPatch,
    tracked_paths: set[Path],
) -> list[tuple[Path, str]]:
    original_open = Path.open
    resolved_tracked_paths = {path.resolve() for path in tracked_paths}
    reads: list[tuple[Path, str]] = []

    def observed_open(path: Path, *args: object, **kwargs: object):
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        resolved_path = path.resolve()
        if resolved_path in resolved_tracked_paths and "r" in mode:
            reads.append((resolved_path, mode))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    return reads


def test_evaluation_cli_rejects_prediction_receipt_before_opening_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _prediction_bytes = _write_cli_evaluation_inputs(tmp_path)
    reads = _observe_read_opens(monkeypatch, set(paths.values()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            str(paths["plan"]),
            "--predictions",
            str(paths["predictions"]),
            "--expected-prediction-sha256",
            "0" * 64,
            "--annotation",
            str(paths["annotation"]),
            "--output",
            str(paths["output"]),
        ],
    )

    with pytest.raises(ValueError):
        evaluate_cli.main()

    annotation_path = paths["annotation"].resolve()
    assert [entry for entry in reads if entry[0] == annotation_path] == []


def test_evaluation_cli_reads_prediction_then_each_annotation_once_as_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, prediction_bytes = _write_cli_evaluation_inputs(tmp_path)
    reads = _observe_read_opens(monkeypatch, set(paths.values()))
    captured_evaluation_inputs: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        captured_evaluation_inputs.update(kwargs)
        return {"metrics": {"promotion_eligible": False}}

    monkeypatch.setattr(evaluate_cli, "evaluate_independent_shot_vlm", fake_evaluate)
    monkeypatch.setattr(evaluate_cli, "_write_json_atomic", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            str(paths["plan"]),
            "--predictions",
            str(paths["predictions"]),
            "--expected-prediction-sha256",
            hashlib.sha256(prediction_bytes).hexdigest(),
            "--annotation",
            str(paths["annotation"]),
            "--output",
            str(paths["output"]),
        ],
    )

    assert evaluate_cli.main() == 2

    predictions_path = paths["predictions"].resolve()
    annotation_path = paths["annotation"].resolve()
    prediction_reads = [entry for entry in reads if entry[0] == predictions_path]
    annotation_reads = [entry for entry in reads if entry[0] == annotation_path]
    assert prediction_reads == [(predictions_path, "rb")]
    assert annotation_reads == [(annotation_path, "rb")]
    assert reads.index(prediction_reads[0]) < reads.index(annotation_reads[0])
    assert captured_evaluation_inputs["predictions"] == json.loads(prediction_bytes)
    prediction_file_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    assert captured_evaluation_inputs["prediction_file_sha256"] == prediction_file_sha256
    assert captured_evaluation_inputs["expected_prediction_file_sha256"] == (prediction_file_sha256)


def test_evaluation_cli_verifies_prediction_before_truth_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _prediction_bytes = _write_cli_evaluation_inputs(tmp_path)
    reads = _observe_read_opens(monkeypatch, set(paths.values()))
    monkeypatch.setattr(
        evaluate_cli,
        "verify_independent_shot_vlm_predictions",
        lambda _payload: (_ for _ in ()).throw(ValueError("invalid prediction")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            str(paths["plan"]),
            "--predictions",
            str(paths["predictions"]),
            "--annotation",
            str(paths["annotation"]),
            "--output",
            str(paths["output"]),
        ],
    )

    with pytest.raises(ValueError, match="invalid prediction"):
        evaluate_cli.main()

    annotation_path = paths["annotation"].resolve()
    assert [entry for entry in reads if entry[0] == annotation_path] == []


def test_evaluation_cli_rejects_duplicate_annotation_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, prediction_bytes = _write_cli_evaluation_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            str(paths["plan"]),
            "--predictions",
            str(paths["predictions"]),
            "--expected-prediction-sha256",
            hashlib.sha256(prediction_bytes).hexdigest(),
            "--annotation",
            str(paths["annotation"]),
            "--annotation",
            str(paths["annotation"]),
            "--output",
            str(paths["output"]),
        ],
    )

    with pytest.raises(ValueError, match="duplicate|alias"):
        evaluate_cli.main()


def test_evaluation_cli_rejects_duplicate_truth_key_across_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _prediction_bytes = _write_cli_evaluation_inputs(tmp_path)
    first_payload = json.loads(paths["annotation"].read_bytes())
    second_payload = copy.deepcopy(first_payload)
    second_payload["examples"][0]["event_present"] = False
    second_path = tmp_path / "annotation-second.json"
    second_bytes = (json.dumps(second_payload, sort_keys=True) + "\n").encode()
    second_path.write_bytes(second_bytes)

    plan = json.loads(paths["plan"].read_bytes())
    plan["training_annotation_sha256"] = sorted(
        {
            hashlib.sha256(paths["annotation"].read_bytes()).hexdigest(),
            hashlib.sha256(second_bytes).hexdigest(),
        }
    )
    plan["plan_sha256"] = canonical_sha256(plan, hash_field="plan_sha256")
    paths["plan"].write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    predictions = json.loads(paths["predictions"].read_bytes())
    predictions["plan_sha256"] = plan["plan_sha256"]
    predictions["artifact_sha256"] = canonical_sha256(
        predictions,
        hash_field="artifact_sha256",
    )
    prediction_bytes = (json.dumps(predictions, sort_keys=True) + "\n").encode()
    paths["predictions"].write_bytes(prediction_bytes)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            str(paths["plan"]),
            "--predictions",
            str(paths["predictions"]),
            "--expected-prediction-sha256",
            hashlib.sha256(prediction_bytes).hexdigest(),
            "--annotation",
            str(paths["annotation"]),
            "--annotation",
            str(second_path),
            "--output",
            str(paths["output"]),
        ],
    )

    with pytest.raises(ValueError, match="duplicate.*truth|truth.*duplicate"):
        evaluate_cli.main()


def test_evaluation_cli_rejects_duplicate_json_keys_before_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, prediction_bytes = _write_cli_evaluation_inputs(tmp_path)
    duplicate_bytes = prediction_bytes.replace(
        b'"codex_runtime_answer_used": false',
        b'"codex_runtime_answer_used": true, "codex_runtime_answer_used": false',
        1,
    )
    assert duplicate_bytes != prediction_bytes
    paths["predictions"].write_bytes(duplicate_bytes)
    reads = _observe_read_opens(monkeypatch, {paths["annotation"]})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            str(paths["plan"]),
            "--predictions",
            str(paths["predictions"]),
            "--expected-prediction-sha256",
            hashlib.sha256(duplicate_bytes).hexdigest(),
            "--annotation",
            str(paths["annotation"]),
            "--output",
            str(paths["output"]),
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_cli.main()
    assert reads == []


@pytest.mark.parametrize(
    "unsafe_payload",
    (
        {"codexRuntimeAnswerUsed": True},
        {"metadata": {"codex_runtime_answer_used": True}},
        {"codex_involved": True},
        {"input_contract": {"labels_or_review_notes_exposed_to_model": True}},
        {"humanReviewLabel": True},
        {"human_reviewed_label": True},
        {"reviewer_notes": "human says positive"},
        {"metadata": ({"label": True},)},
        {"purpose": "codex_runtime_answer"},
        {"Purpose": "codex_runtime_answer"},
        {"schemaVersion": "agu.runtime-result.v1"},
        {"reviewer_label": True},
        {"human_reviewed_labels": [True]},
        {"codex_assisted": True},
        {"labels_exposed": True},
        {"trained_on_target_labels": True},
        {"used_target_labels": True},
        {"target_label_selection_used": True},
    ),
)
@pytest.mark.parametrize("artifact_kind", ("plan", "predictions"))
def test_independent_vlm_sealers_reject_nested_or_normalized_unsafe_fields(
    artifact_kind: str,
    unsafe_payload: dict[str, object],
) -> None:
    row = _row("video-a", "event-1", 0)
    if artifact_kind == "plan":
        payload: dict[str, object] = {
            "selection": {},
            "examples": [row],
            **copy.deepcopy(unsafe_payload),
        }
        sealer = seal_independent_shot_vlm_plan
    else:
        payload = {
            "plan_sha256": "a" * 64,
            "model": {"name": "independent-test-model"},
            "predictions": [
                {
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "field_goal_state": "unknown",
                    "confidence": 0.0,
                }
            ],
            **copy.deepcopy(unsafe_payload),
        }
        sealer = seal_independent_shot_vlm_predictions

    with pytest.raises(ValueError):
        sealer(payload)


@pytest.mark.parametrize(
    "unsafe_payload",
    (
        {"codex_involved": True},
        {"input_contract": {"labels_or_review_notes_exposed_to_model": True}},
        {"human_reviewed_label": True},
        {"reviewer_notes": "human says positive"},
        {"Purpose": "codex_runtime_answer"},
        {"schemaVersion": "agu.runtime-result.v1"},
        {"reviewer_label": True},
        {"human_reviewed_labels": [True]},
        {"codex_assisted": True},
        {"labels_exposed": True},
        {"trained_on_target_labels": True},
        {"used_target_labels": True},
        {"target_label_selection_used": True},
    ),
)
@pytest.mark.parametrize("artifact_kind", ("plan", "predictions"))
def test_independent_vlm_verifiers_reject_resealed_unsafe_fields(
    artifact_kind: str,
    unsafe_payload: dict[str, object],
) -> None:
    row = _row("video-a", "event-1", 0)
    plan = seal_independent_shot_vlm_plan({"selection": {}, "examples": [row]})
    if artifact_kind == "plan":
        artifact = {**plan, **copy.deepcopy(unsafe_payload)}
        artifact["plan_sha256"] = canonical_sha256(
            artifact,
            hash_field="plan_sha256",
        )
        verifier = verify_independent_shot_vlm_plan
    else:
        artifact = seal_independent_shot_vlm_predictions(
            {
                "plan_sha256": plan["plan_sha256"],
                "model": {"name": "independent-test-model"},
                "predictions": [
                    {
                        "source_video_sha256": row["source_video_sha256"],
                        "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                        "event_id": row["event_id"],
                        "field_goal_state": "unknown",
                        "confidence": 0.0,
                    }
                ],
            }
        )
        artifact.update(copy.deepcopy(unsafe_payload))
        artifact["artifact_sha256"] = canonical_sha256(
            artifact,
            hash_field="artifact_sha256",
        )
        verifier = verify_independent_shot_vlm_predictions

    with pytest.raises(ValueError):
        verifier(artifact)


def test_evaluation_records_external_prediction_file_receipt_separately() -> None:
    plan, predictions, truth = _perfect_formal_promotion_case()
    prediction_file_sha256 = "d" * 64

    evaluation = evaluate_independent_shot_vlm(
        plan=plan,
        predictions=predictions,
        truth=truth,
        expected_plan_sha256=plan["plan_sha256"],
        expected_prediction_artifact_sha256=predictions["artifact_sha256"],
        prediction_file_sha256=prediction_file_sha256,
        expected_prediction_file_sha256=prediction_file_sha256,
    )

    assert evaluation["prediction_file_sha256"] == prediction_file_sha256
    assert evaluation["expected_prediction_file_sha256"] == prediction_file_sha256
    assert evaluation["metrics"]["prediction_file_receipt_verified"] is True

    with pytest.raises(ValueError, match="prediction file SHA-256"):
        evaluate_independent_shot_vlm(
            plan=plan,
            predictions=predictions,
            truth=truth,
            prediction_file_sha256="c" * 64,
            expected_prediction_file_sha256=prediction_file_sha256,
        )


@pytest.mark.parametrize("input_name", ("plan", "annotation_plan", "predictions", "annotation"))
@pytest.mark.parametrize("alias_kind", ("direct", "output_symlink", "hardlink", "casefold"))
def test_evaluation_cli_rejects_output_input_alias_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    input_paths = {
        "plan": tmp_path / "plan.json",
        "annotation_plan": tmp_path / "annotation-plan.json",
        "predictions": tmp_path / "predictions.json",
        "annotation": tmp_path / "annotation.json",
    }
    for name, path in input_paths.items():
        path.write_text(f"{name} evidence\n", encoding="utf-8")
    output = input_paths[input_name]
    if alias_kind == "output_symlink":
        output = tmp_path / f"{input_name}-output-alias.json"
        output.symlink_to(input_paths[input_name])
    elif alias_kind == "hardlink":
        output = tmp_path / f"{input_name}-output-hardlink.json"
        output.hardlink_to(input_paths[input_name])
    elif alias_kind == "casefold":
        output = input_paths[input_name].with_name(input_paths[input_name].name.upper())
        if not output.exists() or not output.samefile(input_paths[input_name]):
            pytest.skip("case-fold alias requires a case-insensitive filesystem")
    snapshots = {name: path.read_bytes() for name, path in input_paths.items()}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            str(input_paths["plan"]),
            "--annotation-plan",
            str(input_paths["annotation_plan"]),
            "--predictions",
            str(input_paths["predictions"]),
            "--annotation",
            str(input_paths["annotation"]),
            "--output",
            str(output),
        ],
    )

    def reject_read(_path: Path) -> bytes:
        raise AssertionError("output/input alias must be rejected before JSON reading")

    monkeypatch.setattr(evaluate_cli, "_read_bytes", reject_read)

    with pytest.raises(ValueError, match="output.*input"):
        evaluate_cli.main()

    assert {name: path.read_bytes() for name, path in input_paths.items()} == snapshots


def test_evaluation_json_writer_stages_and_fsyncs_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "evaluation.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    writer = getattr(evaluate_cli, "_write_json_atomic", None)
    assert callable(writer)
    original_replace = Path.replace
    replace_observations: list[Path] = []
    fsync_calls: list[int] = []

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.suffix == ".tmp"
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": True}
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        replace_observations.append(source)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    monkeypatch.setattr(
        evaluate_cli.os,
        "fsync",
        lambda file_descriptor: fsync_calls.append(file_descriptor),
    )

    writer(destination, {"new": True})

    assert replace_observations
    assert fsync_calls
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_evaluation_cli_uses_frozen_resolved_output_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.json"
    predictions_path = tmp_path / "predictions.json"
    annotation_path = tmp_path / "annotation.json"
    row = {
        "source_video_sha256": "a" * 64,
        "candidate_bundle_sha256": "b" * 64,
        "event_id": "event-1",
    }
    plan_path.write_text("{}\n", encoding="utf-8")
    predictions_path.write_text("{}\n", encoding="utf-8")
    annotation_path.write_text(
        json.dumps(
            {
                "source_video_sha256": row["source_video_sha256"],
                "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                "examples": [{"event_id": row["event_id"], "event_present": True}],
            }
        ),
        encoding="utf-8",
    )
    annotation_sha256 = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    first_destination = tmp_path / "first-destination"
    second_destination = tmp_path / "second-destination"
    first_destination.mkdir()
    second_destination.mkdir()
    output_parent = tmp_path / "output-parent"
    output_parent.symlink_to(first_destination, target_is_directory=True)
    output = output_parent / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_shot_vlm.py",
            "--plan",
            str(plan_path),
            "--predictions",
            str(predictions_path),
            "--annotation",
            str(annotation_path),
            "--output",
            str(output),
        ],
    )
    verified_plan = {
        "training_annotation_sha256": [annotation_sha256],
        "examples": [row],
    }
    monkeypatch.setattr(
        evaluate_cli,
        "verify_independent_shot_vlm_plan",
        lambda _payload: verified_plan,
    )
    monkeypatch.setattr(
        evaluate_cli,
        "verify_independent_shot_vlm_predictions",
        lambda payload: payload,
    )

    def redirect_after_validation(**_kwargs: object) -> dict[str, object]:
        output_parent.unlink()
        output_parent.symlink_to(second_destination, target_is_directory=True)
        return {"metrics": {"promotion_eligible": False}}

    monkeypatch.setattr(
        evaluate_cli,
        "evaluate_independent_shot_vlm",
        redirect_after_validation,
    )

    assert evaluate_cli.main() == 2

    assert json.loads((first_destination / "evaluation.json").read_text(encoding="utf-8")) == {
        "metrics": {"promotion_eligible": False}
    }
    assert not (second_destination / "evaluation.json").exists()
