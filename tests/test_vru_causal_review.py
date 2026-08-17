from __future__ import annotations

import copy
import hashlib
import json

import pytest

from app.analysis.continuous_causal_selection import neutral_continuous_review_id
from app.analysis.vru_causal_review import (
    build_vru_causal_review_plan,
    seal_vru_causal_review,
    verify_vru_causal_review,
    verify_vru_causal_review_plan,
)


def _plan() -> dict[str, object]:
    return build_vru_causal_review_plan(
        source_manifest_sha256="a" * 64,
        examples=[
            {
                "review_id": "vru-causal-0001",
                "source_video_sha256": "b" * 64,
                "source_video_filename": "clip-a.mp4",
                "source_fps": 25.0,
                "frame_count": 250,
                "frame_indexes": [100, 110, 120, 130, 140, 150],
            },
            {
                "review_id": "vru-causal-0002",
                "source_video_sha256": "c" * 64,
                "source_video_filename": "clip-b.mp4",
                "source_fps": 25.0,
                "frame_count": 250,
                "frame_indexes": [0, 10, 20, 30, 40, 50],
            },
        ],
    )


def _selection_bound_examples(*, filename: str = "all_windows_are_shots.webm") -> list[dict[str, object]]:
    source_sha = "b" * 64
    examples: list[dict[str, object]] = []
    for ordinal in range(1, 25):
        first = ordinal * 300
        indexes = [round(first + offset * 30.0 / 8.0) for offset in range(64)]
        examples.append(
            {
                "review_id": neutral_continuous_review_id(source_sha, ordinal),
                "source_video_sha256": source_sha,
                "source_video_filename": filename,
                "source_fps": 30.0,
                "frame_count": 10_000,
                "frame_indexes": indexes,
                "frame_sha256s": {str(index): "d" * 64 for index in indexes},
            }
        )
    return examples


def _selection_bound_plan() -> dict[str, object]:
    return build_vru_causal_review_plan(
        source_manifest_sha256="a" * 64,
        source_selection_artifact_sha256="c" * 64,
        examples=_selection_bound_examples(),
    )


def _valid_sealed_review(plan: dict[str, object]) -> dict[str, object]:
    return seal_vru_causal_review(
        {
            "reviews": [
                {
                    "review_id": "vru-causal-0001",
                    "shot_sequence": "shot",
                    "release_position": 2,
                    "rim_position": 5,
                    "outcome": "unknown",
                    "confidence": "high",
                    "evidence": {
                        "controlled_ball_before_release": True,
                        "ball_separated_from_hand": True,
                        "ball_progresses_toward_rim": True,
                        "rim_proximity_visible": True,
                        "rim_contact_visible": False,
                    },
                    "notes": "Release and rim approach are visible.",
                },
                {
                    "review_id": "vru-causal-0002",
                    "shot_sequence": "not_a_shot",
                    "release_position": None,
                    "rim_position": None,
                    "outcome": "not_applicable",
                    "confidence": "medium",
                    "evidence": {
                        "controlled_ball_before_release": False,
                        "ball_separated_from_hand": False,
                        "ball_progresses_toward_rim": False,
                        "rim_proximity_visible": False,
                        "rim_contact_visible": False,
                    },
                    "notes": "No visible release-to-rim chain.",
                },
            ]
        },
        plan=plan,
    )


def _canonical_reseal(artifact: dict[str, object]) -> dict[str, object]:
    resealed = copy.deepcopy(artifact)
    resealed.pop("artifact_sha256", None)
    encoded = json.dumps(
        resealed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    resealed["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    return resealed


def test_vru_causal_plan_is_label_hidden_and_hash_bound() -> None:
    plan = _plan()

    assert plan["schema_version"] == "agu.vru-causal-review-plan.v1"
    assert plan["runtime_consumable"] is False
    assert plan["codex_runtime_answer_used"] is False
    assert plan["labels_hidden_from_reviewer"] is True
    assert "shot" not in json.dumps(plan["examples"], sort_keys=True)
    assert verify_vru_causal_review_plan(plan)["artifact_sha256"] == plan["artifact_sha256"]


def test_selection_bound_plan_carries_receipt_and_hides_source_strings() -> None:
    plan = _selection_bound_plan()

    assert plan["source_selection_artifact_sha256"] == "c" * 64
    assert plan["reviewer_visible_fields"] == [
        "review_id",
        "source_video_handle",
        "source_fps",
        "frame_indexes",
        "frame_sha256s",
    ]
    assert plan["examples"][0]["source_video_handle"].startswith("source-")
    assert plan["examples"][0]["source_video_filename"].startswith("source-")
    assert "shot" not in plan["examples"][0]["source_video_filename"]
    assert "source_video_filename" not in plan["reviewer_visible_fields"]
    with pytest.raises(ValueError, match="external plan receipt"):
        verify_vru_causal_review_plan(plan)
    assert (
        verify_vru_causal_review_plan(
            plan,
            expected_artifact_sha256=plan["artifact_sha256"],
        )
        == plan
    )


def test_external_plan_receipt_rejects_validly_resealed_selection_geometry() -> None:
    plan = _selection_bound_plan()
    drifted = copy.deepcopy(plan)
    indexes = drifted["examples"][0]["frame_indexes"]
    assert isinstance(indexes, list)
    shifted = [int(index) + 300 for index in indexes]
    drifted["examples"][0]["frame_indexes"] = shifted
    drifted["examples"][0]["frame_sha256s"] = {str(index): "e" * 64 for index in shifted}
    drifted = _canonical_reseal(drifted)

    with pytest.raises(ValueError, match="external plan receipt"):
        verify_vru_causal_review_plan(
            drifted,
            expected_artifact_sha256=plan["artifact_sha256"],
        )


def test_selection_bound_plan_cannot_be_resealed_as_legacy() -> None:
    plan = _selection_bound_plan()
    downgraded = copy.deepcopy(plan)
    downgraded.pop("source_selection_artifact_sha256")
    downgraded = _canonical_reseal(downgraded)

    with pytest.raises(ValueError, match="selection.*receipt"):
        verify_vru_causal_review_plan(downgraded)


def test_selection_bound_plan_rejects_label_bearing_reviewer_id() -> None:
    examples = _selection_bound_examples(filename="private.webm")
    examples[0]["review_id"] = "made-shot-harwood-0001"
    with pytest.raises(ValueError, match="neutral.*review ID"):
        build_vru_causal_review_plan(
            source_manifest_sha256="a" * 64,
            source_selection_artifact_sha256="c" * 64,
            examples=examples,
        )


def test_selection_bound_plan_rejects_resealed_extra_label_channel() -> None:
    plan = _selection_bound_plan()
    injected = copy.deepcopy(plan)
    injected["examples"][0]["outcome"] = "made"
    injected = _canonical_reseal(injected)

    with pytest.raises(ValueError, match="selection-bound.*fields"):
        verify_vru_causal_review_plan(injected)


def test_selection_bound_plan_rejects_resealed_geometry_and_hash_truncation() -> None:
    plan = _selection_bound_plan()
    drifted = copy.deepcopy(plan)
    drifted["examples"] = drifted["examples"][:1]
    drifted["examples"][0]["frame_indexes"] = ["7", "8", "9"]
    drifted["examples"][0]["frame_sha256s"] = {}
    drifted["examples"][0]["source_video_filename"] = {"label": "made"}
    drifted["examples"][0]["source_fps"] = {"model_score": 0.99}
    drifted = _canonical_reseal(drifted)

    with pytest.raises(ValueError, match="24|64|filename|FPS"):
        verify_vru_causal_review_plan(drifted)


def test_vru_plan_sha_receipts_require_actual_string_scalars() -> None:
    class _LooksLikeSha:
        def __str__(self) -> str:
            return "a" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        build_vru_causal_review_plan(
            source_manifest_sha256=_LooksLikeSha(),  # type: ignore[arg-type]
            examples=[
                {
                    "review_id": "vru-causal-0001",
                    "source_video_sha256": "b" * 64,
                    "source_video_filename": "clip.mp4",
                    "source_fps": 25.0,
                    "frame_count": 250,
                    "frame_indexes": [0, 10, 20],
                }
            ],
        )

    plan = _plan()
    claimed_sha = plan["artifact_sha256"]

    class _LooksLikeArtifactSha:
        def __str__(self) -> str:
            return str(claimed_sha)

    plan["artifact_sha256"] = _LooksLikeArtifactSha()
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_vru_causal_review_plan(plan)


@pytest.mark.parametrize("review_id", ("../escape", "nested/escape", "nested\\escape"))
def test_vru_causal_plan_rejects_path_unsafe_review_ids(review_id: str) -> None:
    with pytest.raises(ValueError, match="review ID.*path-safe"):
        build_vru_causal_review_plan(
            source_manifest_sha256="a" * 64,
            examples=[
                {
                    "review_id": review_id,
                    "source_video_sha256": "b" * 64,
                    "source_video_filename": "clip-a.mp4",
                    "source_fps": 25.0,
                    "frame_count": 250,
                    "frame_indexes": [100, 110, 120],
                }
            ],
        )


def test_vru_causal_review_resolves_positions_and_requires_exact_coverage() -> None:
    plan = _plan()
    review = seal_vru_causal_review(
        {
            "reviews": [
                {
                    "review_id": "vru-causal-0001",
                    "shot_sequence": "shot",
                    "release_position": 2,
                    "rim_position": 5,
                    "outcome": "unknown",
                    "confidence": "high",
                    "evidence": {
                        "controlled_ball_before_release": True,
                        "ball_separated_from_hand": True,
                        "ball_progresses_toward_rim": True,
                        "rim_proximity_visible": True,
                        "rim_contact_visible": False,
                    },
                    "notes": "Release and rim approach are visible; result is unresolved.",
                },
                {
                    "review_id": "vru-causal-0002",
                    "shot_sequence": "not_a_shot",
                    "release_position": None,
                    "rim_position": None,
                    "outcome": "not_applicable",
                    "confidence": "medium",
                    "evidence": {
                        "controlled_ball_before_release": False,
                        "ball_separated_from_hand": False,
                        "ball_progresses_toward_rim": False,
                        "rim_proximity_visible": False,
                        "rim_contact_visible": False,
                    },
                    "notes": "Continuous play without a visible release-to-rim chain.",
                },
            ]
        },
        plan=plan,
    )

    assert review["runtime_consumable"] is False
    assert review["reviews"][0]["release_frame"] == 120
    assert review["reviews"][0]["rim_frame"] == 150
    assert verify_vru_causal_review(review, plan=plan)["artifact_sha256"] == review["artifact_sha256"]

    incomplete = copy.deepcopy(review)
    incomplete.pop("artifact_sha256")
    incomplete["reviews"] = incomplete["reviews"][:1]
    with pytest.raises(ValueError, match="exactly cover"):
        seal_vru_causal_review(incomplete, plan=plan)


def test_vru_causal_review_rejects_broken_causal_order_or_tampering() -> None:
    plan = _plan()
    payload = {
        "reviews": [
            {
                "review_id": "vru-causal-0001",
                "shot_sequence": "shot",
                "release_position": 5,
                "rim_position": 2,
                "outcome": "unknown",
                "confidence": "high",
                "evidence": {
                    "controlled_ball_before_release": True,
                    "ball_separated_from_hand": True,
                    "ball_progresses_toward_rim": True,
                    "rim_proximity_visible": True,
                    "rim_contact_visible": False,
                },
            },
            {
                "review_id": "vru-causal-0002",
                "shot_sequence": "uncertain",
                "release_position": None,
                "rim_position": None,
                "outcome": "unknown",
                "confidence": "low",
                "evidence": {
                    "controlled_ball_before_release": False,
                    "ball_separated_from_hand": False,
                    "ball_progresses_toward_rim": False,
                    "rim_proximity_visible": False,
                    "rim_contact_visible": False,
                },
            },
        ]
    }
    with pytest.raises(ValueError, match="precede"):
        seal_vru_causal_review(payload, plan=plan)

    valid = seal_vru_causal_review(
        {
            **payload,
            "reviews": [
                {**payload["reviews"][0], "release_position": 2, "rim_position": 5},
                payload["reviews"][1],
            ],
        },
        plan=plan,
    )
    tampered = copy.deepcopy(valid)
    tampered["reviews"][0]["rim_frame"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_vru_causal_review(tampered, plan=plan)


def test_vru_causal_review_preserves_legacy_uncertain_outcome_semantics() -> None:
    plan = _plan()
    reviews = _valid_sealed_review(plan)["reviews"]
    assert isinstance(reviews, list)
    uncertain = copy.deepcopy(reviews[1])
    uncertain["shot_sequence"] = "uncertain"
    uncertain["outcome"] = "made"

    sealed = seal_vru_causal_review(
        {"reviews": [reviews[0], uncertain]},
        plan=plan,
    )

    assert sealed["reviews"][1]["outcome"] == "made"
    assert verify_vru_causal_review(sealed, plan=plan) == sealed


def test_vru_causal_verifier_accepts_sealed_rows_without_optional_positions() -> None:
    plan = _plan()
    reviews = _valid_sealed_review(plan)["reviews"]
    assert isinstance(reviews, list)
    inputs = copy.deepcopy(reviews)
    inputs[1].pop("release_position")
    inputs[1].pop("rim_position")

    sealed = seal_vru_causal_review({"reviews": inputs}, plan=plan)

    assert "release_position" not in sealed["reviews"][1]
    assert "rim_position" not in sealed["reviews"][1]
    assert verify_vru_causal_review(sealed, plan=plan) == sealed


@pytest.mark.parametrize(
    "malformation",
    (
        "missing_review",
        "duplicate_review",
        "release_frame_mapping",
        "invalid_shot_evidence",
        "invalid_shot_outcome",
        "extra_review_field",
        "extra_artifact_field",
    ),
)
def test_vru_causal_verifier_rejects_canonical_resealed_malformed_reviews(
    malformation: str,
) -> None:
    plan = _plan()
    malformed = _valid_sealed_review(plan)
    reviews = malformed["reviews"]
    assert isinstance(reviews, list)

    if malformation == "missing_review":
        reviews.pop()
    elif malformation == "duplicate_review":
        reviews[1] = copy.deepcopy(reviews[0])
    elif malformation == "release_frame_mapping":
        reviews[0]["release_frame"] = 121
    elif malformation == "invalid_shot_evidence":
        reviews[0]["evidence"]["ball_separated_from_hand"] = False
    elif malformation == "invalid_shot_outcome":
        reviews[0]["outcome"] = "not_applicable"
    elif malformation == "extra_review_field":
        reviews[0]["untrusted_extra"] = "must not survive verification"
    else:
        malformed["untrusted_extra"] = "must not survive verification"

    with pytest.raises(ValueError):
        verify_vru_causal_review(_canonical_reseal(malformed), plan=plan)
