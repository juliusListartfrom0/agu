from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_vru_causal_temporal_feature_plan import _verified_task_fixture

from app.analysis import vru_causal_temporal_retrospective as temporal_module
from app.analysis.vru_causal_temporal_retrospective import (
    build_sealed_final_evaluator,
    verify_final_evaluator_computation,
)

GAME_COUNTS = {
    "hazen": (6, 2),
    "randolph": (4, 3),
    "vtv": (4, 3),
    "harwood": (3, 20),
}

MANDATORY_MECHANICAL_EVIDENCE_CHECKS = frozenset(
    {
        "two_empty_state_receipt_bound_replays",
        "held_label_invariance",
        "per_fit_backend_pre_and_post",
        "disk_budget_revalidation",
        "atomic_publication_and_no_residue",
        "complete_resume_chain_external_receipts",
    }
)


@pytest.fixture(autouse=True)
def _verified_fit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        temporal_module,
        "observe_temporal_environment_contract",
        lambda: temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT,
    )


def _inputs() -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...]]:
    labels: list[bool] = []
    games: list[str] = []
    strong: list[list[float]] = []
    weak: list[list[float]] = []
    for game_index, (game, (positive_count, negative_count)) in enumerate(GAME_COUNTS.items()):
        for label, count in ((True, positive_count), (False, negative_count)):
            for row_index in range(count):
                labels.append(label)
                games.append(game)
                strong.append(
                    [
                        2.0 if label else -2.0,
                        game_index / 10.0,
                        row_index / 100.0,
                    ]
                )
                weak.append(
                    [
                        float(game_index),
                        row_index / 100.0,
                        0.0,
                    ]
                )
    weak_matrix = np.zeros((45, 768), dtype=np.float64)
    weak_matrix[:, :3] = np.asarray(weak, dtype=np.float64)
    strong_matrix = np.zeros((45, 1536), dtype=np.float64)
    strong_matrix[:, :3] = np.asarray(strong, dtype=np.float64)
    return (
        {
            "swin3d_t": weak_matrix,
            "mvit_v2_s+swin3d_t": strong_matrix,
            temporal_module.TEMPORAL_REPRESENTATION_CONTRACT["name"]: strong_matrix,
        },
        np.asarray(labels, dtype=np.bool_),
        tuple(games),
    )


def _ordered_keys(games: tuple[str, ...]) -> list[dict[str, str]]:
    source_by_game = {game: f"{index + 1:064x}" for index, game in enumerate(GAME_COUNTS)}
    bundle_by_game = {game: f"{index + 11:064x}" for index, game in enumerate(GAME_COUNTS)}
    return [
        {
            "source_video_sha256": source_by_game[game],
            "candidate_bundle_sha256": bundle_by_game[game],
            "event_id": f"{game}-{index:04d}",
        }
        for index, game in enumerate(games)
    ]


def test_baseline_evaluator_runs_both_variants_and_refits_selected_on_all_rows() -> None:
    features, labels, games = _inputs()
    rows = _ordered_keys(games)
    result = build_sealed_final_evaluator(
        feature_matrices={
            "swin3d_t": features["swin3d_t"],
            "mvit_v2_s+swin3d_t": features["mvit_v2_s+swin3d_t"],
        },
        labels=labels,
        game_ids=games,
        representation_order=("swin3d_t", "mvit_v2_s+swin3d_t"),
        evaluator_role="baseline",
        ordered_training_rows=rows,
        input_receipts={"plan": "1" * 64},
    )

    assert [row["representation_name"] for row in result["logo_selection"]] == [
        "swin3d_t",
        "mvit_v2_s+swin3d_t",
    ]
    assert result["selected_evaluator"]["representation_name"] == ("mvit_v2_s+swin3d_t")
    assert result["all_45_refit"]["row_count"] == 45
    assert result["runtime_consumable"] is False
    assert result["formal_evaluation_eligible"] is False
    assert result["promotion_eligible"] is False
    assert result["promoted"] is False


def test_candidate_evaluator_has_only_the_predeclared_tiled_variant() -> None:
    features, labels, games = _inputs()
    rows = _ordered_keys(games)
    result = build_sealed_final_evaluator(
        feature_matrices={
            temporal_module.TEMPORAL_REPRESENTATION_CONTRACT["name"]: features[
                temporal_module.TEMPORAL_REPRESENTATION_CONTRACT["name"]
            ]
        },
        labels=labels,
        game_ids=games,
        representation_order=(temporal_module.TEMPORAL_REPRESENTATION_CONTRACT["name"],),
        evaluator_role="candidate",
        ordered_training_rows=rows,
        input_receipts={"plan": "1" * 64},
    )

    assert len(result["logo_selection"]) == 1
    assert (
        result["selected_evaluator"]["representation_name"] == temporal_module.TEMPORAL_REPRESENTATION_CONTRACT["name"]
    )
    assert len(result["logo_selection"][0]["oof_predictions"]) == 45
    assert len(result["logo_selection"][0]["threshold_candidates"]) == 21
    assert (
        verify_final_evaluator_computation(
            result,
            feature_matrices={
                temporal_module.TEMPORAL_REPRESENTATION_CONTRACT["name"]: features[
                    temporal_module.TEMPORAL_REPRESENTATION_CONTRACT["name"]
                ]
            },
            labels=labels,
            game_ids=games,
            ordered_training_rows=rows,
            input_receipts={"plan": "1" * 64},
        )
        == result
    )


def test_private_final_generation_cannot_pass_without_mandatory_external_evidence(
    tmp_path: Path,
) -> None:
    inputs = _verified_task_fixture(tmp_path)
    plan_payload = temporal_module.seal_vru_causal_temporal_feature_plan(
        inputs=inputs,
        environment_contract=temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT,
    )
    plan_path = tmp_path / "plan.json"
    encoded = temporal_module._canonical_json_bytes(plan_payload)
    plan_path.write_bytes(encoded)
    plan = temporal_module.load_verified_temporal_feature_plan(
        plan_path=plan_path,
        expected_artifact_sha256=plan_payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    old_artifacts = []
    for backbone, strong in (
        ("torchvision/mvit_v2_s/kinetics400_v1", True),
        ("torchvision/swin3d_t/kinetics400_v1", False),
    ):
        old_artifacts.append(
            {
                "backbone": backbone,
                "embedding_dimension": 768,
                "examples": [
                    {
                        "source_video_sha256": row["source_video_sha256"],
                        "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                        "event_id": row["event_id"],
                        "event_present": row["event_present"],
                        "embedding": [
                            (2.0 if row["event_present"] else -2.0) if strong and index == 0 else 0.0
                            for index in range(768)
                        ],
                    }
                    for row in inputs.ordered_rows
                ],
            }
        )
    inputs.old_embeddings = tuple(old_artifacts)
    inputs.replayed_nested_probe = {
        "selection_protocol": "nested_outer_game_held_v2",
        "row_count": 45,
        "observed_metrics": {"pooled": {"f1": 0.1}},
    }
    temporal_module._register_task0257_capability(inputs)
    tiled = [
        [
            [(2.0 if row["event_present"] else -2.0) if index == 0 else float(tile) / 100.0 for index in range(768)]
            for tile in range(4)
        ]
        for row in inputs.ordered_rows
    ]
    attempt_chain = [
        {
            "attempt_ordinal": 1,
            "attempt_record": {
                "schema_version": "agu.vru-causal-tiled-swin-attempt.v1",
                "internal_sha256_field": "artifact_sha256",
                "internal_sha256": "1" * 64,
                "file_sha256": "2" * 64,
                "filename": "attempt.json",
                "size_bytes": 1,
            },
            "resource_log": {
                "file_sha256": "3" * 64,
                "filename": "resource.jsonl",
                "size_bytes": 0,
            },
            "resume_output_cas": None,
        }
    ]
    fresh = temporal_module._build_fresh_tiled_swin_embeddings(
        plan=plan,
        tile_embeddings=tiled,
        attempt_chain=attempt_chain,
    )

    generation = temporal_module._build_vru_causal_temporal_final_generation(
        inputs=inputs,
        plan=plan,
        tiled_embeddings=fresh,
    )

    assert generation.retrospective["error_bound_result"]["passed"] is True
    assert generation.baseline_evaluator["selected_evaluator"]["representation_name"] == ("mvit_v2_s+swin3d_t")
    assert (
        generation.candidate_evaluator["selected_evaluator"]["representation_name"]
        == (temporal_module.TEMPORAL_REPRESENTATION_CONTRACT["name"])
    )
    passed_checks = {
        row["check"] for row in generation.mechanical_gate["ordered_check_results"] if row.get("passed") is True
    }
    missing_evidence = MANDATORY_MECHANICAL_EVIDENCE_CHECKS - passed_checks
    assert generation.mechanical_gate["decision"] != "mechanical_pass" or not missing_evidence, (
        f"mechanical_pass omitted mandatory external evidence: {sorted(missing_evidence)}"
    )


def test_public_final_generation_verifier_rejects_unbound_mandatory_evidence(
    tmp_path: Path,
) -> None:
    def pretty(payload: dict[str, object]) -> bytes:
        return temporal_module._canonical_json_bytes(payload)

    inputs = _verified_task_fixture(tmp_path)
    plan_payload = temporal_module.seal_vru_causal_temporal_feature_plan(
        inputs=inputs,
        environment_contract=temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT,
    )
    plan_path = tmp_path / "plan.json"
    plan_bytes = pretty(plan_payload)
    plan_path.write_bytes(plan_bytes)
    plan = temporal_module.load_verified_temporal_feature_plan(
        plan_path=plan_path,
        expected_artifact_sha256=plan_payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(plan_bytes).hexdigest(),
    )
    inputs.old_embeddings = tuple(
        {
            "backbone": backbone,
            "embedding_dimension": 768,
            "examples": [
                {
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "event_present": row["event_present"],
                    "embedding": [
                        (2.0 if row["event_present"] else -2.0) if index == 0 else 0.0 for index in range(768)
                    ],
                }
                for row in inputs.ordered_rows
            ],
        }
        for backbone in (
            "torchvision/mvit_v2_s/kinetics400_v1",
            "torchvision/swin3d_t/kinetics400_v1",
        )
    )
    inputs.replayed_nested_probe = {
        "selection_protocol": "nested_outer_game_held_v2",
        "row_count": 45,
        "observed_metrics": {"pooled": {"f1": 0.1}},
    }
    temporal_module._register_task0257_capability(inputs)
    final_dir = tmp_path / "final_v1"
    terminal_attempt = final_dir / "terminal_attempt"
    terminal_attempt.mkdir(parents=True)
    resource_path = terminal_attempt / "resource_guard.jsonl"
    resource_path.write_bytes(b"")
    resource_receipt = {
        "file_sha256": hashlib.sha256(b"").hexdigest(),
        "filename": "resource_guard.jsonl",
        "size_bytes": 0,
    }
    attempt = temporal_module.seal_tiled_swin_attempt(
        {
            "attempt_ordinal": 1,
            "prior_attempt_receipt": None,
            "plan_receipt": {
                "artifact_sha256": plan._artifact_sha256,
                "file_sha256": plan._file_sha256,
            },
            "task0257_input_receipts": plan._payload["task0257_receipts"],
            "started_prefix_count": 0,
            "completed_prefix_count": 45,
            "new_rows_verified": 45,
            "cumulative_active_runtime_nanoseconds": 0,
            "cumulative_resource_samples": 0,
            "cumulative_resource_log_bytes": 0,
            "resource_log_receipt": resource_receipt,
            "resume_input_cas": None,
            "resume_output_cas": None,
            "received_signal": None,
            "disposition": "completed",
            "stop_reason": None,
        }
    )
    attempt_path = terminal_attempt / "attempt_record.json"
    attempt_bytes = pretty(attempt)
    attempt_path.write_bytes(attempt_bytes)
    attempt_chain = [
        {
            "attempt_ordinal": 1,
            "attempt_record": {
                "schema_version": attempt["schema_version"],
                "internal_sha256_field": "artifact_sha256",
                "internal_sha256": attempt["artifact_sha256"],
                "file_sha256": hashlib.sha256(attempt_bytes).hexdigest(),
                "filename": "attempt_record.json",
                "size_bytes": len(attempt_bytes),
            },
            "resource_log": resource_receipt,
            "resume_output_cas": None,
        }
    ]
    tiled = [
        [
            [(2.0 if row["event_present"] else -2.0) if index == 0 else float(tile) / 100.0 for index in range(768)]
            for tile in range(4)
        ]
        for row in inputs.ordered_rows
    ]
    fresh = temporal_module._build_fresh_tiled_swin_embeddings(
        plan=plan,
        tile_embeddings=tiled,
        attempt_chain=attempt_chain,
    )
    generation = temporal_module._build_vru_causal_temporal_final_generation(
        inputs=inputs,
        plan=plan,
        tiled_embeddings=fresh,
    )
    payloads = {
        "tiled_swin_embeddings.json": fresh._payload,
        "temporal_retrospective.json": generation.retrospective,
        "baseline_final_evaluator.json": generation.baseline_evaluator,
        "candidate_final_evaluator.json": generation.candidate_evaluator,
        "mechanical_gate.json": generation.mechanical_gate,
    }
    for filename, payload in payloads.items():
        (final_dir / filename).write_bytes(pretty(payload))
    file_receipts = {
        str(path.relative_to(final_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(final_dir.rglob("*"))
        if path.is_file()
    }
    artifact_receipts = {
        key: json.loads((final_dir / key).read_text(encoding="utf-8"))["artifact_sha256"]
        for key in file_receipts
        if not key.endswith("resource_guard.jsonl")
    }
    verified_tiled = temporal_module.load_verified_tiled_swin_embeddings(
        embeddings_path=final_dir / "tiled_swin_embeddings.json",
        expected_artifact_sha256=fresh._payload["artifact_sha256"],
        expected_file_sha256=file_receipts["tiled_swin_embeddings.json"],
        plan=plan,
    )
    assert (
        temporal_module.verify_vru_causal_final_evaluator(
            evaluator_path=final_dir / "baseline_final_evaluator.json",
            expected_artifact_sha256=generation.baseline_evaluator["artifact_sha256"],
            expected_file_sha256=file_receipts["baseline_final_evaluator.json"],
            verified_inputs=inputs,
            tiled_embeddings=verified_tiled,
        )
        == generation.baseline_evaluator
    )
    noncanonical_evaluator = tmp_path / "noncanonical-evaluator.json"
    noncanonical_bytes = (json.dumps(generation.baseline_evaluator, indent=2) + "\n").encode("utf-8")
    noncanonical_evaluator.write_bytes(noncanonical_bytes)
    with pytest.raises(ValueError, match="canonical"):
        temporal_module.verify_vru_causal_final_evaluator(
            evaluator_path=noncanonical_evaluator,
            expected_artifact_sha256=generation.baseline_evaluator["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(noncanonical_bytes).hexdigest(),
            verified_inputs=inputs,
            tiled_embeddings=verified_tiled,
        )
    registry = temporal_module.seal_module_a_final_receipt_registry(
        generation_dir=final_dir,
    )
    assert (
        temporal_module.verify_module_a_final_receipt_registry(
            registry,
            expected_artifact_sha256=registry["artifact_sha256"],
        )["file_receipts"]
        == file_receipts
    )

    with pytest.raises(ValueError):
        temporal_module.verify_vru_causal_temporal_final_generation(
            generation_dir=final_dir,
            expected_file_receipts=file_receipts,
            expected_artifact_receipts=artifact_receipts,
            verified_inputs=inputs,
            plan=plan,
            tiled_embeddings=verified_tiled,
        )


def test_bounded_binary_reader_rejects_oversize_before_reading_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = tmp_path / "resource_guard.jsonl"
    with oversized.open("wb") as handle:
        handle.truncate(17 * 1024 * 1024)
    monkeypatch.setattr(
        temporal_module.os,
        "read",
        lambda *_args, **_kwargs: pytest.fail("oversized payload was read"),
    )

    with pytest.raises(ValueError, match="size limit"):
        temporal_module._read_bounded_bytes(oversized, max_bytes=16 * 1024 * 1024)
