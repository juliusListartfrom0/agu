"""Tests for the TASK-0258 v2 gate vocabulary (check sequences + failure matrix)."""

from __future__ import annotations

import pytest

from app.analysis.task0258_v2_gate import (
    CANDIDATE_METRIC_OUTCOMES,
    CANDIDATE_PREPUBLICATION_CHECKS,
    FAILED_PHASES,
    RESULT_ORDERED_CHECKS,
    STOP_REASONS,
    TRUST_CHECKS,
    expected_failure_check_sequence,
    verify_candidate_metric_outcome,
    verify_failed_phase,
    verify_failed_phase_stop_reason,
    verify_failure_check_sequence,
    verify_ordered_checks,
)


def test_candidate_check_sequence_exact():
    assert CANDIDATE_PREPUBLICATION_CHECKS == (
        "parent_spec_authorization",
        "amendment_implementation_authorization",
        "amended_implementation_review",
        "exact_v2_rerun_authorization",
        "reviewed_runtime_dependency_closure",
        "run_history_ledger",
        "run_admission",
        "static_input_replay",
        "prior_producer_chain_replay",
        "checkpoint_and_source_identity",
        "disk_revalidation",
        "terminal_exclusivity",
        "staging_topology",
        "producer_extraction",
        "verification_extraction",
        "projection_equality",
        "held_label_invariance",
        "per_fit_environment",
        "retrospective",
        "baseline_evaluator",
        "candidate_evaluator",
        "global_resource_caps",
        "candidate_canonical_coverage",
        "candidate_publication_preconditions",
    )


def test_trust_checks_are_first_thirteen():
    assert TRUST_CHECKS == CANDIDATE_PREPUBLICATION_CHECKS[:13]
    assert TRUST_CHECKS[-1] == "staging_topology"


def test_result_check_sequence_length_and_tail():
    assert len(RESULT_ORDERED_CHECKS) == 26
    assert RESULT_ORDERED_CHECKS[-1] == "frozen_error_bounds"
    assert RESULT_ORDERED_CHECKS[-2] == "result_publication_preconditions"


def test_failed_phases_exact():
    assert FAILED_PHASES == (
        "producer_attempt_1",
        "producer_attempt_2",
        "producer_attempt_3",
        "pre_verification",
        "verification_before_embedding",
        "verification_after_embedding",
        "projection_equality",
        "held_label_invariance",
        "per_fit_environment",
        "retrospective",
        "baseline_evaluator",
        "candidate_evaluator",
        "global_resource_caps",
        "candidate_sealing",
    )


def test_stop_reasons_exact():
    assert STOP_REASONS == (
        "resource_breach",
        "runtime_cap",
        "sample_cap",
        "log_byte_cap",
        "decode_failure",
        "model_failure",
        "determinism_failure",
        "non_advancing_prefix",
        "attempt_limit",
        "external_sigint",
        "external_sigterm",
        "unsupported_signal",
        "computation_failure",
    )


def test_verify_failed_phase():
    verify_failed_phase("candidate_sealing")
    with pytest.raises(ValueError):
        verify_failed_phase("bogus_phase")


def test_phase_stop_reason_matrix():
    # attempt 3 allows attempt_limit; attempt 1/2 do not
    verify_failed_phase_stop_reason("producer_attempt_3", "attempt_limit")
    with pytest.raises(ValueError):
        verify_failed_phase_stop_reason("producer_attempt_1", "attempt_limit")
    # computation phase allows computation_failure, not decode_failure
    verify_failed_phase_stop_reason("retrospective", "computation_failure")
    with pytest.raises(ValueError):
        verify_failed_phase_stop_reason("retrospective", "decode_failure")
    # global_resource_caps allows caps, not computation_failure
    verify_failed_phase_stop_reason("global_resource_caps", "runtime_cap")
    with pytest.raises(ValueError):
        verify_failed_phase_stop_reason("global_resource_caps", "computation_failure")


def test_expected_failure_sequence_candidate_sealing():
    seq = expected_failure_check_sequence("candidate_sealing")
    assert seq[:13] == TRUST_CHECKS
    # additional true prefix producer_extraction..candidate_canonical_coverage
    assert seq[13:-1] == (
        "producer_extraction",
        "verification_extraction",
        "projection_equality",
        "held_label_invariance",
        "per_fit_environment",
        "retrospective",
        "baseline_evaluator",
        "candidate_evaluator",
        "global_resource_caps",
        "candidate_canonical_coverage",
    )
    assert seq[-1] == "candidate_publication_preconditions"


def test_expected_failure_sequence_producer_attempt_1():
    seq = expected_failure_check_sequence("producer_attempt_1")
    assert seq == TRUST_CHECKS + ("producer_attempt_1",)


def _rows(names):
    return [{"check_name": n, "passed": (i < len(names) - 1)} for i, n in enumerate(names)]


def test_verify_failure_check_sequence_valid():
    verify_failure_check_sequence("producer_attempt_1", _rows(expected_failure_check_sequence("producer_attempt_1")))
    verify_failure_check_sequence("candidate_sealing", _rows(expected_failure_check_sequence("candidate_sealing")))


def test_verify_failure_check_sequence_rejects():
    names = list(expected_failure_check_sequence("candidate_sealing"))
    with pytest.raises(ValueError):
        verify_failure_check_sequence("candidate_sealing", _rows(names[:-1]))
    # wrong false-check name
    bad = _rows(names)
    bad[-1] = {"check_name": "bogus", "passed": False}
    with pytest.raises(ValueError):
        verify_failure_check_sequence("candidate_sealing", bad)
    # final row must be False, all others True
    bad2 = _rows(names)
    bad2[-1] = {"check_name": names[-1], "passed": True}
    with pytest.raises(ValueError):
        verify_failure_check_sequence("candidate_sealing", bad2)


def test_candidate_metric_outcome():
    assert CANDIDATE_METRIC_OUTCOMES == (
        "within_frozen_error_bounds",
        "outside_frozen_error_bounds",
    )
    verify_candidate_metric_outcome("within_frozen_error_bounds")
    verify_candidate_metric_outcome("outside_frozen_error_bounds")
    with pytest.raises(ValueError):
        verify_candidate_metric_outcome("mechanical_pass")


def test_verify_ordered_checks():
    names = ("a", "b", "c")
    verify_ordered_checks(
        [{"check_name": "a", "passed": True}, {"check_name": "b", "passed": True}, {"check_name": "c", "passed": True}],
        names,
    )
    # allow_last_false
    verify_ordered_checks(
        [
            {"check_name": "a", "passed": True},
            {"check_name": "b", "passed": True},
            {"check_name": "c", "passed": False},
        ],
        names,
        allow_last_false=True,
    )
    # last must not be True when allow_last_false
    with pytest.raises(ValueError):
        verify_ordered_checks(
            [
                {"check_name": "a", "passed": True},
                {"check_name": "b", "passed": True},
                {"check_name": "c", "passed": True},
            ],
            names,
            allow_last_false=True,
        )
    # a middle row must not be False
    with pytest.raises(ValueError):
        verify_ordered_checks(
            [
                {"check_name": "a", "passed": True},
                {"check_name": "b", "passed": False},
                {"check_name": "c", "passed": False},
            ],
            names,
            allow_last_false=True,
        )
