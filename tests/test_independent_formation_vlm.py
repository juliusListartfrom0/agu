from __future__ import annotations

import copy

import pytest

import app.analysis.independent_formation_vlm as formation_module
from app.analysis.independent_formation_vlm import (
    derive_independent_formation_vlm_plan,
    evaluate_fixed_formation_veto,
    evaluate_independent_formation_vlm,
    parse_independent_formation_vlm_decision,
    seal_independent_formation_vlm_plan,
    seal_independent_formation_vlm_predictions,
    verify_independent_formation_vlm_plan,
)
from scripts.run_independent_formation_vlm import (
    _cache_fingerprint,
    _unexpected_ollama_models,
)


def _row(video: str, event: str, frames: tuple[int, int]) -> dict[str, object]:
    return {
        "phase_review_id": f"phase-{event}",
        "source_video_sha256": video * 64,
        "source_video_filename": f"{video}.mp4",
        "candidate_bundle_sha256": ("b" if video == "a" else "c") * 64,
        "event_id": event,
        "source_fps": 30.0,
        "frame_indexes": list(frames),
    }


def test_formation_plan_is_two_frame_label_free_and_tamper_evident() -> None:
    plan = seal_independent_formation_vlm_plan(
        {
            "source_phase_plan_sha256": "d" * 64,
            "frame_offsets_seconds": [-1.0, 0.0],
            "input_contract": {
                "raw_frames_only": True,
                "chronological_fixed_sampling": True,
                "max_frames": 2,
                "image_width": 512,
            },
            "examples": [_row("a", "event-1", (100, 130))],
        }
    )

    assert plan["runtime_consumable"] is False
    assert plan["codex_runtime_answer_used"] is False
    assert plan["labels_or_review_notes_exposed_to_model"] is False
    assert plan["examples"][0]["frame_indexes"] == [100, 130]
    assert "formation_state" not in str(plan)

    tampered = copy.deepcopy(plan)
    tampered["examples"][0]["frame_indexes"][0] = 99
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_independent_formation_vlm_plan(tampered)

    leaked = copy.deepcopy(plan)
    leaked["examples"][0]["formation_state"] = "free_throw_setup"
    leaked["plan_sha256"] = ""
    with pytest.raises(ValueError):
        verify_independent_formation_vlm_plan(leaked)


def test_formation_decision_parser_fails_closed_and_clamps_confidence() -> None:
    parsed = parse_independent_formation_vlm_decision(
        {
            "formation_state": "not-a-state",
            "confidence": 9,
            "shooter_at_free_throw_line": "yes",
            "lane_players_aligned": False,
        }
    )

    assert parsed["formation_state"] == "unknown"
    assert parsed["confidence"] == 1.0
    assert parsed["observables"]["shooter_at_free_throw_line"] is None
    assert parsed["observables"]["lane_players_aligned"] is False


def test_formation_plan_derivation_selects_exact_label_hidden_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "artifact_sha256": "d" * 64,
        "anchor_offsets_seconds": [
            -4.0,
            -3.5,
            -3.25,
            -3.0,
            -2.75,
            -2.5,
            -2.25,
            -2.0,
            -1.75,
            -1.5,
            -1.25,
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
            4.0,
        ],
        "sealed_blind_video_sha256s": ["f" * 64],
        "examples": [
            {
                **_row("a", "event-1", (100, 130)),
                "frame_indexes": list(range(100, 124)),
            },
            {
                **_row("b", "event-2", (200, 230)),
                "frame_indexes": list(range(200, 224)),
            },
        ],
    }
    monkeypatch.setattr(
        formation_module,
        "verify_causal_shot_phase_review_plan",
        lambda payload: source,
    )

    plan = derive_independent_formation_vlm_plan(
        {"source": "stubbed"},
        image_width=384,
        max_examples=1,
    )

    assert plan["source_phase_plan_sha256"] == "d" * 64
    assert plan["examples"][0]["frame_indexes"] == [111, 115]
    assert plan["input_contract"]["image_width"] == 384
    assert plan["selection"]["resource_probe_only"] is True
    assert "formation_state" not in str(plan)


def test_formation_cache_fingerprint_binds_context_and_weights() -> None:
    kwargs = {
        "plan_sha256": "plan",
        "model": "model",
        "model_source": "source",
        "model_revision": "revision",
        "weights_sha256": "weights",
        "image_width": 384,
        "context_length": 2048,
    }
    baseline = _cache_fingerprint(**kwargs)

    assert baseline == _cache_fingerprint(**kwargs)
    assert baseline != _cache_fingerprint(
        **{**kwargs, "context_length": 3072}
    )
    assert baseline != _cache_fingerprint(
        **{**kwargs, "weights_sha256": "other"}
    )


def test_formation_runner_rejects_unexpected_ollama_residency() -> None:
    payload = {
        "models": [
            {"name": "qwen3-vl:2b"},
            {"model": "qwen3-vl:4b"},
        ]
    }

    assert _unexpected_ollama_models(payload, target_model="qwen3-vl:2b") == [
        "qwen3-vl:4b"
    ]
    assert _unexpected_ollama_models(
        {"models": [{"name": "qwen3-vl:2b"}]},
        target_model="qwen3-vl:2b",
    ) == []


def test_formation_evaluation_requires_exact_plan_coverage() -> None:
    rows = [
        _row("a", "free-throw", (100, 130)),
        _row("a", "live-play", (200, 230)),
        _row("b", "other", (300, 330)),
        _row("b", "unknown", (400, 430)),
    ]
    plan = seal_independent_formation_vlm_plan(
        {
            "source_phase_plan_sha256": "d" * 64,
            "frame_offsets_seconds": [-1.0, 0.0],
            "input_contract": {
                "raw_frames_only": True,
                "chronological_fixed_sampling": True,
                "max_frames": 2,
                "image_width": 512,
            },
            "examples": rows,
        }
    )
    predicted_states = {
        "free-throw": "free_throw_setup",
        "live-play": "free_throw_setup",
        "other": "stoppage_other",
        "unknown": "unknown",
    }
    predictions = seal_independent_formation_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {"name": "independent-test-model"},
            "predictions": [
                {
                    "phase_review_id": row["phase_review_id"],
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    **parse_independent_formation_vlm_decision(
                        {
                            "formation_state": predicted_states[str(row["event_id"])],
                            "confidence": 0.8,
                        }
                    ),
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
        ): {
            "free-throw": "free_throw_setup",
            "live-play": "live_play",
            "other": "stoppage_other",
            "unknown": "uncertain",
        }[str(row["event_id"])]
        for row in rows
    }

    evaluation = evaluate_independent_formation_vlm(
        plan=plan,
        predictions=predictions,
        truth=truth,
    )

    assert evaluation["metrics"]["free_throw_true_positive"] == 1
    assert evaluation["metrics"]["free_throw_false_positive"] == 1
    assert evaluation["metrics"]["free_throw_precision"] == pytest.approx(0.5)
    assert evaluation["metrics"]["free_throw_recall"] == pytest.approx(1.0)
    assert evaluation["metrics"]["unknown"] == 1
    assert evaluation["metrics"]["promotion_eligible"] is False

    incomplete = copy.deepcopy(predictions)
    incomplete["predictions"].pop()
    incomplete = seal_independent_formation_vlm_predictions(incomplete)
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate_independent_formation_vlm(
            plan=plan,
            predictions=incomplete,
            truth=truth,
        )


def test_fixed_formation_veto_only_suppresses_predicted_free_throws() -> None:
    rows = [
        _row("a", "false-positive", (100, 130)),
        _row("a", "true-positive", (200, 230)),
    ]
    plan = seal_independent_formation_vlm_plan(
        {
            "source_phase_plan_sha256": "d" * 64,
            "frame_offsets_seconds": [-1.0, 0.0],
            "input_contract": {
                "raw_frames_only": True,
                "chronological_fixed_sampling": True,
                "max_frames": 2,
                "image_width": 512,
            },
            "examples": rows,
        }
    )
    states = {
        "false-positive": "free_throw_setup",
        "true-positive": "live_play",
    }
    predictions = seal_independent_formation_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {"name": "independent-test-model"},
            "predictions": [
                {
                    "phase_review_id": row["phase_review_id"],
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    **parse_independent_formation_vlm_decision(
                        {
                            "formation_state": states[str(row["event_id"])],
                            "confidence": 0.8,
                        }
                    ),
                }
                for row in rows
            ],
        }
    )

    evaluation = evaluate_fixed_formation_veto(
        formation_predictions=predictions,
        upstream_artifact_sha256="e" * 64,
        upstream_predictions=[
            {
                "source_video_sha256": row["source_video_sha256"],
                "event_id": row["event_id"],
                "event_present": row["event_id"] == "true-positive",
                "probability": 0.9,
            }
            for row in rows
        ],
        threshold=0.5,
    )

    assert evaluation["baseline"]["false_positive"] == 1
    assert evaluation["formation_veto"]["false_positive"] == 0
    assert evaluation["formation_veto"]["true_positive"] == 1
    assert evaluation["affected_event_count"] == 1
