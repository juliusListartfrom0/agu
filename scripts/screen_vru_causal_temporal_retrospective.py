#!/usr/bin/env python3
"""Replay and verify a published TASK-0258 Module-A final generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.analysis.vru_causal_temporal_retrospective import (
    TEMPORAL_ENVIRONMENT_CONTRACT,
    _read_bounded_json,
    _sha256_bytes,
    load_verified_temporal_feature_plan,
    load_verified_tiled_swin_embeddings,
    observe_temporal_environment_contract,
    seal_vru_causal_temporal_feature_plan,
    verify_module_a_final_receipt_registry,
    verify_module_a_spec_approval,
    verify_task0257_temporal_inputs,
    verify_vru_causal_temporal_final_generation,
)
from scripts import seal_vru_causal_temporal_feature_plan as plan_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-approval-artifact-sha256", required=True)
    parser.add_argument("--expected-approval-file-sha256", required=True)
    parser.add_argument("--requirement", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--gate-review", type=Path, required=True)
    parser.add_argument("--expected-fresh-review-internal-sha256", required=True)
    parser.add_argument("--expected-fresh-review-file-sha256", required=True)
    parser.add_argument("--expected-approval-statement-sha256", required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--expected-input-contract-file-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-artifact-sha256", required=True)
    parser.add_argument("--expected-plan-file-sha256", required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--receipt-registry", type=Path, required=True)
    parser.add_argument("--expected-receipt-registry-artifact-sha256", required=True)
    parser.add_argument("--expected-receipt-registry-file-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_module_a_spec_approval(
        approval_path=args.approval,
        expected_artifact_sha256=args.expected_approval_artifact_sha256,
        expected_file_sha256=args.expected_approval_file_sha256,
        approved_spec_paths=(args.requirement, args.solution, args.gate_review),
        expected_fresh_review_internal_sha256=args.expected_fresh_review_internal_sha256,
        expected_fresh_review_file_sha256=args.expected_fresh_review_file_sha256,
        expected_approval_statement_sha256=args.expected_approval_statement_sha256,
    )
    contract = plan_cli._read_contract(
        args.input_contract,
        expected_file_sha256=args.expected_input_contract_file_sha256,
    )
    paths, receipts = plan_cli._parse_input_contract(contract)
    environment = observe_temporal_environment_contract()
    if environment != TEMPORAL_ENVIRONMENT_CONTRACT:
        raise ValueError("screening environment is invalid")
    verified_inputs = verify_task0257_temporal_inputs(
        paths=paths,
        expected_receipts=receipts,
    )
    plan = load_verified_temporal_feature_plan(
        plan_path=args.plan,
        expected_artifact_sha256=args.expected_plan_artifact_sha256,
        expected_file_sha256=args.expected_plan_file_sha256,
    )
    if (
        seal_vru_causal_temporal_feature_plan(
            inputs=verified_inputs,
            environment_contract=environment,
        )
        != plan._payload
    ):
        raise ValueError("stored temporal plan does not replay from TASK-0257")
    registry, registry_bytes = _read_bounded_json(
        args.receipt_registry,
        max_bytes=1_048_576,
    )
    if _sha256_bytes(registry_bytes) != args.expected_receipt_registry_file_sha256:
        raise ValueError("receipt registry file SHA-256 does not match")
    verified_registry = verify_module_a_final_receipt_registry(
        registry,
        expected_artifact_sha256=args.expected_receipt_registry_artifact_sha256,
    )
    tiled_embeddings = load_verified_tiled_swin_embeddings(
        embeddings_path=args.generation_dir / "tiled_swin_embeddings.json",
        expected_artifact_sha256=verified_registry["artifact_receipts"]["tiled_swin_embeddings.json"],
        expected_file_sha256=verified_registry["file_receipts"]["tiled_swin_embeddings.json"],
        plan=plan,
    )
    generation = verify_vru_causal_temporal_final_generation(
        generation_dir=args.generation_dir,
        expected_file_receipts=verified_registry["file_receipts"],
        expected_artifact_receipts=verified_registry["artifact_receipts"],
        verified_inputs=verified_inputs,
        plan=plan,
        tiled_embeddings=tiled_embeddings,
    )
    print(
        json.dumps(
            {
                "decision": generation.mechanical_gate["decision"],
                "mechanical_gate_artifact_sha256": generation.mechanical_gate["artifact_sha256"],
                "receipt_registry_verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
