"""TASK-0258 Amendment-001 v2 gate vocabulary: candidate/result check sequences
and the mechanical-failure phase/check/reason matrix.

These are the exact, closed enum sequences from amendment-001 §"Candidate
generation" and §"Failure generation before candidate publication". They are
pure data plus closed validators; no heavy computation, filesystem, or model
work happens here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

# Candidate gate `ordered_prepublication_check_results` (exact order, all true).
CANDIDATE_PREPUBLICATION_CHECKS: tuple[str, ...] = (
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

# Post-publication result `ordered_check_results` (exact order; all but the
# final error-bound row must pass).
RESULT_ORDERED_CHECKS: tuple[str, ...] = (
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
    "producer_extraction",
    "verification_extraction",
    "projection_equality",
    "held_label_invariance",
    "per_fit_environment",
    "retrospective",
    "baseline_evaluator",
    "candidate_evaluator",
    "global_resource_caps",
    "disk_revalidation",
    "candidate_canonical_coverage",
    "candidate_receipt_bundle",
    "prerename_terminal_exclusivity",
    "bound_active_stage_only",
    "result_publication_preconditions",
    "frozen_error_bounds",
)

# The first thirteen candidate checks are the global trust spine; every
# mechanical-failure `ordered_check_results` begins with these, all true.
TRUST_CHECKS: tuple[str, ...] = CANDIDATE_PREPUBLICATION_CHECKS[:13]

# Exact failed phases.
FAILED_PHASES: tuple[str, ...] = (
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

# Exact stop reasons (union across all phases).
STOP_REASONS: tuple[str, ...] = (
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

_PRODUCER_REASONS = frozenset(
    {
        "resource_breach",
        "runtime_cap",
        "sample_cap",
        "log_byte_cap",
        "decode_failure",
        "model_failure",
        "determinism_failure",
        "non_advancing_prefix",
        "external_sigint",
        "external_sigterm",
        "unsupported_signal",
    }
)
_WORKER_REASONS = frozenset(
    {
        "resource_breach",
        "runtime_cap",
        "sample_cap",
        "log_byte_cap",
        "decode_failure",
        "model_failure",
        "determinism_failure",
        "external_sigint",
        "external_sigterm",
        "unsupported_signal",
    }
)
_COMPUTATION_REASONS = frozenset(
    {
        "computation_failure",
        "external_sigint",
        "external_sigterm",
        "unsupported_signal",
    }
)
_CAP_REASONS = frozenset(
    {
        "resource_breach",
        "runtime_cap",
        "sample_cap",
        "log_byte_cap",
        "external_sigint",
        "external_sigterm",
        "unsupported_signal",
    }
)

# Exact phase -> allowed stop-reason set.
FAILED_PHASE_ALLOWED_REASONS: dict[str, frozenset[str]] = {
    "producer_attempt_1": _PRODUCER_REASONS,
    "producer_attempt_2": _PRODUCER_REASONS,
    "producer_attempt_3": _PRODUCER_REASONS | {"attempt_limit"},
    "pre_verification": _COMPUTATION_REASONS,
    "verification_before_embedding": _WORKER_REASONS,
    "verification_after_embedding": _WORKER_REASONS,
    "projection_equality": _COMPUTATION_REASONS,
    "held_label_invariance": _COMPUTATION_REASONS,
    "per_fit_environment": _COMPUTATION_REASONS,
    "retrospective": _COMPUTATION_REASONS,
    "baseline_evaluator": _COMPUTATION_REASONS,
    "candidate_evaluator": _COMPUTATION_REASONS,
    "global_resource_caps": _CAP_REASONS,
    "candidate_sealing": _COMPUTATION_REASONS,
}

# candidate check index of `producer_extraction` (first post-trust check).
_PRODUCER_INDEX = CANDIDATE_PREPUBLICATION_CHECKS.index("producer_extraction")


def _candidate_prefix_through(last_check: str) -> tuple[str, ...]:
    end = CANDIDATE_PREPUBLICATION_CHECKS.index(last_check)
    return CANDIDATE_PREPUBLICATION_CHECKS[_PRODUCER_INDEX : end + 1]


# Exact phase -> (additional true checks, one false check).
FAILED_PHASE_CHECK_PREFIX: dict[str, tuple[tuple[str, ...], str]] = {
    "producer_attempt_1": ((), "producer_attempt_1"),
    "producer_attempt_2": ((), "producer_attempt_2"),
    "producer_attempt_3": ((), "producer_attempt_3"),
    "pre_verification": (("producer_extraction",), "pre_verification"),
    "verification_before_embedding": (("producer_extraction",), "verification_before_embedding"),
    "verification_after_embedding": (
        ("producer_extraction", "verification_extraction"),
        "verification_after_embedding",
    ),
    "projection_equality": (
        ("producer_extraction", "verification_extraction"),
        "projection_equality",
    ),
    "held_label_invariance": (
        _candidate_prefix_through("projection_equality"),
        "held_label_invariance",
    ),
    "per_fit_environment": (
        _candidate_prefix_through("held_label_invariance"),
        "per_fit_environment",
    ),
    "retrospective": (
        _candidate_prefix_through("per_fit_environment"),
        "retrospective",
    ),
    "baseline_evaluator": (
        _candidate_prefix_through("retrospective"),
        "baseline_evaluator",
    ),
    "candidate_evaluator": (
        _candidate_prefix_through("baseline_evaluator"),
        "candidate_evaluator",
    ),
    "global_resource_caps": (
        _candidate_prefix_through("candidate_evaluator"),
        "global_resource_caps",
    ),
    "candidate_sealing": (
        _candidate_prefix_through("candidate_canonical_coverage"),
        "candidate_publication_preconditions",
    ),
}


def is_failed_phase(phase: str) -> bool:
    return phase in FAILED_PHASES


def verify_failed_phase(phase: str) -> None:
    if not is_failed_phase(phase):
        raise ValueError(f"unknown failed phase: {phase!r}")


def verify_failed_phase_stop_reason(phase: str, stop_reason: str) -> None:
    """Raise when ``stop_reason`` is not an allowed reason for ``phase``."""
    verify_failed_phase(phase)
    if stop_reason not in FAILED_PHASE_ALLOWED_REASONS[phase]:
        raise ValueError(f"stop_reason {stop_reason!r} not allowed for phase {phase!r}")


def expected_failure_check_sequence(phase: str) -> tuple[str, ...]:
    """Return the exact full ordered check sequence for a failed phase.

    The sequence is the first thirteen trust checks, all true, then the phase's
    additional true prefix, then exactly one false row (the failure subject).
    """
    verify_failed_phase(phase)
    extra_true, false_check = FAILED_PHASE_CHECK_PREFIX[phase]
    return TRUST_CHECKS + extra_true + (false_check,)


# candidate_metric_outcome closed enum (never mechanical_pass/rejected/failure).
CANDIDATE_METRIC_OUTCOMES: tuple[str, ...] = (
    "within_frozen_error_bounds",
    "outside_frozen_error_bounds",
)


def verify_candidate_metric_outcome(value: object) -> None:
    if value not in CANDIDATE_METRIC_OUTCOMES:
        raise ValueError(f"candidate_metric_outcome invalid: {value!r}")


def verify_ordered_checks(
    check_results: Sequence[Mapping[str, object]],
    expected_names: Sequence[str],
    *,
    allow_last_false: bool = False,
) -> None:
    """Validate an exact ordered check list.

    Every row is ``{check_name, passed}``. The names must equal ``expected_names``
    in order with no omission/reorder/duplication. When ``allow_last_false`` is
    False every row must be ``passed=True``; when True, all rows except the last
    must pass and the last must be ``passed=False``.
    """
    if len(check_results) != len(expected_names):
        raise ValueError("ordered check sequence length mismatch")
    for i, (row, expected_name) in enumerate(zip(check_results, expected_names)):
        if set(row) != {"check_name", "passed"}:
            raise ValueError(f"check row {i} shape is invalid")
        if row["check_name"] != expected_name:
            raise ValueError(f"check row {i} name mismatch: {row['check_name']!r} != {expected_name!r}")
        is_last = i == len(expected_names) - 1
        if allow_last_false:
            required = not is_last
        else:
            required = True
        if row["passed"] is not required:
            raise ValueError(f"check row {i} passed value invalid (expected {required})")


def verify_failure_check_sequence(
    phase: str,
    check_results: Sequence[Mapping[str, object]],
) -> None:
    """Validate an exact mechanical-failure ``ordered_check_results``.

    Every row is ``{check_name, passed}``; all rows except the final one must be
    ``passed=True``; the final row is the phase's exact false check. No row may
    be omitted, renamed, duplicated, reordered, or replaced.
    """
    expected = expected_failure_check_sequence(phase)
    verify_ordered_checks(check_results, expected, allow_last_false=True)


__all__ = [
    "CANDIDATE_PREPUBLICATION_CHECKS",
    "RESULT_ORDERED_CHECKS",
    "TRUST_CHECKS",
    "FAILED_PHASES",
    "STOP_REASONS",
    "FAILED_PHASE_ALLOWED_REASONS",
    "FAILED_PHASE_CHECK_PREFIX",
    "CANDIDATE_METRIC_OUTCOMES",
    "is_failed_phase",
    "verify_failed_phase",
    "verify_failed_phase_stop_reason",
    "expected_failure_check_sequence",
    "verify_failure_check_sequence",
    "verify_candidate_metric_outcome",
    "verify_ordered_checks",
]
