"""TASK-0258 v2 review-only contexts and receipt-bound loaders.

The review contexts in this module are diagnostics.  They deliberately carry
no production authorization and are rejected by production publication APIs.
The loaders are the small, reusable no-follow/canonical-byte boundary used by
later admission work: a caller supplies an externally frozen path and hashes;
the loader reopens the exact object and returns an opaque, read-only result.
"""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

from app.analysis.task0258_module_a_v2 import (
    compact_canonical_json,
    is_safe_slug,
    is_sha256,
    verify_internal_artifact_hash,
)
from app.analysis.task0258_run_history import replay_run_history_registry
from app.analysis.task0258_v2_artifacts import verify_candidate_receipt_bundle

_DISCOVERY_TOKEN = object()
_SANDBOX_TOKEN = object()
_SYNTHETIC_DISCOVERY_TOKEN = object()
_SYNTHETIC_REVIEW_TOKEN = object()
_JSON_ARTIFACT_TOKEN = object()
_RUN_HISTORY_TOKEN = object()

_REVIEW_CHECKS = frozenset({"focused_pytest", "full_pytest"})


class VerifiedReviewDiscoveryContext:
    __slots__ = ("_token", "expected_check_name", "expected_command_sha256")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _DISCOVERY_TOKEN:
            raise TypeError("VerifiedReviewDiscoveryContext is review-sandbox-only")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.expected_check_name = kwargs["expected_check_name"]
        self.expected_command_sha256 = kwargs["expected_command_sha256"]


class VerifiedImplementationReviewSandboxContext:
    __slots__ = ("_token", "expected_check_name", "expected_command_sha256")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _SANDBOX_TOKEN:
            raise TypeError("VerifiedImplementationReviewSandboxContext is review-sandbox-only")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.expected_check_name = kwargs["expected_check_name"]
        self.expected_command_sha256 = kwargs["expected_command_sha256"]


class VerifiedSyntheticDiscoveryTransactionContext:
    __slots__ = ("_token", "ancestor", "identity")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _SYNTHETIC_DISCOVERY_TOKEN:
            raise TypeError("VerifiedSyntheticDiscoveryTransactionContext is review-only")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.ancestor = kwargs["ancestor"]
        self.identity = kwargs["identity"]


class VerifiedSyntheticModuleATransactionContext:
    __slots__ = ("_token", "ancestor", "identity")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _SYNTHETIC_REVIEW_TOKEN:
            raise TypeError("VerifiedSyntheticModuleATransactionContext is review-only")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.ancestor = kwargs["ancestor"]
        self.identity = kwargs["identity"]


class VerifiedJsonArtifact:
    __slots__ = ("_token", "path", "payload", "artifact_sha256", "file_sha256")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _JSON_ARTIFACT_TOKEN:
            raise TypeError("VerifiedJsonArtifact cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.path = kwargs["path"]
        self.payload = kwargs["payload"]
        self.artifact_sha256 = kwargs["artifact_sha256"]
        self.file_sha256 = kwargs["file_sha256"]


class VerifiedRunHistoryLedger:
    __slots__ = ("_token", "directory", "authorization_sha256", "payloads")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _RUN_HISTORY_TOKEN:
            raise TypeError("VerifiedRunHistoryLedger cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.directory = kwargs["directory"]
        self.authorization_sha256 = kwargs["authorization_sha256"]
        self.payloads = tuple(kwargs["payloads"])


def _verify_sha(value: object, name: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _verify_review_check(check_name: object, command_sha256: object) -> None:
    if check_name not in _REVIEW_CHECKS:
        raise ValueError("review check name is not allowed")
    _verify_sha(command_sha256, "review command hash")


def _verify_real_directory(path: Path) -> tuple[int, int]:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("path must be an absolute real directory")
    identity = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(identity.st_mode):
        raise ValueError("path is not a directory")
    return identity.st_dev, identity.st_ino


def _verify_temp_ancestor(path: Path) -> tuple[int, int]:
    identity = _verify_real_directory(path)
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        path.resolve().relative_to(temp_root)
    except ValueError as exc:
        raise ValueError("synthetic context must be under the OS temporary directory") from exc
    return identity


def _reopen_temp_context(context: object) -> Path:
    if not isinstance(
        context,
        (VerifiedSyntheticDiscoveryTransactionContext, VerifiedSyntheticModuleATransactionContext),
    ):
        raise TypeError("synthetic context type is invalid")
    if context._token not in {_SYNTHETIC_DISCOVERY_TOKEN, _SYNTHETIC_REVIEW_TOKEN}:
        raise PermissionError("synthetic context token is invalid")
    identity = _verify_real_directory(context.ancestor)
    if identity != context.identity:
        raise ValueError("synthetic temporary ancestor identity changed")
    return context.ancestor


def bind_implementation_review_discovery_context(
    *, expected_check_name: str, expected_command_sha256: str
) -> VerifiedReviewDiscoveryContext:
    """Bind a diagnostic context for namespace discovery only."""
    _verify_review_check(expected_check_name, expected_command_sha256)
    return VerifiedReviewDiscoveryContext(
        _DISCOVERY_TOKEN,
        expected_check_name=expected_check_name,
        expected_command_sha256=expected_command_sha256,
    )


def bind_implementation_review_sandbox_context(
    *, expected_check_name: str, expected_command_sha256: str
) -> VerifiedImplementationReviewSandboxContext:
    """Bind a diagnostic context for synthetic review transactions only."""
    _verify_review_check(expected_check_name, expected_command_sha256)
    return VerifiedImplementationReviewSandboxContext(
        _SANDBOX_TOKEN,
        expected_check_name=expected_check_name,
        expected_command_sha256=expected_command_sha256,
    )


def replay_module_a_read_traversal_for_discovery(
    *,
    discovery_context: VerifiedReviewDiscoveryContext,
    operation: str,
    operation_input_manifest_path: Path,
    expected_manifest_artifact_sha256: str,
    expected_manifest_file_sha256: str,
) -> Mapping[str, object]:
    """Reopen one temporary manifest and return observation-only data.

    This function never returns a production capability and never accepts a
    repository/output path.  A real external runner must still provide the OS
    FD and kernel-audit evidence before an execution context can be issued.
    """
    if (
        type(discovery_context) is not VerifiedReviewDiscoveryContext
        or discovery_context._token is not _DISCOVERY_TOKEN
    ):
        raise PermissionError("discovery context is invalid")
    if not is_safe_slug(operation):
        raise ValueError("discovery operation is invalid")
    path = Path(operation_input_manifest_path)
    temp_root = Path(tempfile.gettempdir()).resolve()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("discovery manifest must be an absolute regular file")
    try:
        path.resolve().relative_to(temp_root)
    except ValueError as exc:
        raise ValueError("discovery manifest must be under the OS temporary directory") from exc
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _verify_sha(expected_manifest_file_sha256, "manifest file hash"):
        raise ValueError("discovery manifest file hash does not match")
    if not raw.endswith(b"\n"):
        raise ValueError("discovery manifest must end with one LF")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("discovery manifest is not JSON") from exc
    if not isinstance(payload, Mapping) or raw != (compact_canonical_json(payload) + "\n").encode("utf-8"):
        raise ValueError("discovery manifest is not canonical JSON")
    verify_internal_artifact_hash(payload)
    if payload["artifact_sha256"] != _verify_sha(expected_manifest_artifact_sha256, "manifest artifact hash"):
        raise ValueError("discovery manifest artifact hash does not match")
    return {
        "operation": operation,
        "manifest_file_sha256": expected_manifest_file_sha256,
        "manifest_artifact_sha256": expected_manifest_artifact_sha256,
        "observed_member_count": len(payload),
        "production_capability": False,
    }


def bind_synthetic_module_a_discovery_context(
    *, discovery_context: VerifiedReviewDiscoveryContext, isolated_temp_ancestor: Path
) -> VerifiedSyntheticDiscoveryTransactionContext:
    if (
        type(discovery_context) is not VerifiedReviewDiscoveryContext
        or discovery_context._token is not _DISCOVERY_TOKEN
    ):
        raise PermissionError("discovery context is invalid")
    identity = _verify_temp_ancestor(Path(isolated_temp_ancestor))
    return VerifiedSyntheticDiscoveryTransactionContext(
        _SYNTHETIC_DISCOVERY_TOKEN,
        ancestor=Path(isolated_temp_ancestor),
        identity=identity,
    )


def bind_synthetic_module_a_transaction_context(
    *, review_sandbox: VerifiedImplementationReviewSandboxContext, isolated_temp_ancestor: Path
) -> VerifiedSyntheticModuleATransactionContext:
    if (
        type(review_sandbox) is not VerifiedImplementationReviewSandboxContext
        or review_sandbox._token is not _SANDBOX_TOKEN
    ):
        raise PermissionError("review sandbox context is invalid")
    identity = _verify_temp_ancestor(Path(isolated_temp_ancestor))
    return VerifiedSyntheticModuleATransactionContext(
        _SYNTHETIC_REVIEW_TOKEN,
        ancestor=Path(isolated_temp_ancestor),
        identity=identity,
    )


def _exercise_synthetic(context: object, scenario: Mapping[str, object]) -> Mapping[str, object]:
    ancestor = _reopen_temp_context(context)
    if not isinstance(scenario, Mapping) or set(scenario) != {"scenario_id", "steps"}:
        raise ValueError("synthetic scenario field set is invalid")
    if not is_safe_slug(scenario["scenario_id"]):
        raise ValueError("synthetic scenario id is invalid")
    steps = scenario["steps"]
    if not isinstance(steps, (list, tuple)) or len(steps) > 128:
        raise ValueError("synthetic scenario steps are invalid")
    for step in steps:
        if not isinstance(step, Mapping) or set(step) != {"state", "event"}:
            raise ValueError("synthetic scenario step is invalid")
        if not is_safe_slug(step["state"]) or not is_safe_slug(step["event"]):
            raise ValueError("synthetic scenario step values are invalid")
    return {
        "scenario_id": scenario["scenario_id"],
        "observed_step_count": len(steps),
        "ancestor_device": ancestor.stat().st_dev,
        "ancestor_inode": ancestor.stat().st_ino,
        "production_capability": False,
    }


def exercise_module_a_v2_state_machine_for_discovery(
    *, synthetic_context: VerifiedSyntheticDiscoveryTransactionContext, scenario: Mapping[str, object]
) -> Mapping[str, object]:
    if type(synthetic_context) is not VerifiedSyntheticDiscoveryTransactionContext:
        raise PermissionError("discovery synthetic context is invalid")
    return _exercise_synthetic(synthetic_context, scenario)


def exercise_module_a_v2_state_machine_for_review(
    *, synthetic_context: VerifiedSyntheticModuleATransactionContext, scenario: Mapping[str, object]
) -> Mapping[str, object]:
    if type(synthetic_context) is not VerifiedSyntheticModuleATransactionContext:
        raise PermissionError("review synthetic context is invalid")
    return _exercise_synthetic(synthetic_context, scenario)


def _load_verified_json_artifact(
    *, path: Path, expected_artifact_sha256: str, expected_file_sha256: str
) -> VerifiedJsonArtifact:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("verified artifact must be an absolute regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _verify_sha(expected_file_sha256, "artifact file hash"):
        raise ValueError("verified artifact file hash does not match")
    if not raw.endswith(b"\n"):
        raise ValueError("verified artifact must end with LF")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("verified artifact is not JSON") from exc
    if not isinstance(payload, Mapping) or raw != (compact_canonical_json(payload) + "\n").encode("utf-8"):
        raise ValueError("verified artifact is not canonical JSON")
    verify_internal_artifact_hash(payload)
    if payload["artifact_sha256"] != _verify_sha(expected_artifact_sha256, "artifact hash"):
        raise ValueError("verified artifact hash does not match")
    return VerifiedJsonArtifact(
        _JSON_ARTIFACT_TOKEN,
        path=path,
        payload=dict(payload),
        artifact_sha256=expected_artifact_sha256,
        file_sha256=expected_file_sha256,
    )


def load_verified_candidate_receipt_bundle(
    *,
    execution_context: object,
    bundle_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
) -> VerifiedJsonArtifact:
    """Load a candidate bundle only inside an externally bound review context."""
    if type(execution_context) not in {
        VerifiedImplementationReviewSandboxContext,
    }:
        raise PermissionError("candidate bundle loader requires a verified review context")
    artifact = _load_verified_json_artifact(
        path=bundle_path,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_file_sha256=expected_file_sha256,
    )
    verify_candidate_receipt_bundle(artifact.payload)
    return artifact


def load_verified_run_history_ledger(
    *, registry_directory: Path, authorization_sha256: str
) -> VerifiedRunHistoryLedger:
    """Replay a complete durable registry and return an opaque ledger."""
    _verify_sha(authorization_sha256, "authorization hash")
    directory = Path(registry_directory)
    payloads = replay_run_history_registry(directory, authorization_sha256)
    return VerifiedRunHistoryLedger(
        _RUN_HISTORY_TOKEN,
        directory=directory,
        authorization_sha256=authorization_sha256,
        payloads=payloads,
    )


__all__ = [
    "VerifiedReviewDiscoveryContext",
    "VerifiedImplementationReviewSandboxContext",
    "VerifiedSyntheticDiscoveryTransactionContext",
    "VerifiedSyntheticModuleATransactionContext",
    "VerifiedJsonArtifact",
    "VerifiedRunHistoryLedger",
    "bind_implementation_review_discovery_context",
    "bind_implementation_review_sandbox_context",
    "replay_module_a_read_traversal_for_discovery",
    "bind_synthetic_module_a_discovery_context",
    "bind_synthetic_module_a_transaction_context",
    "exercise_module_a_v2_state_machine_for_discovery",
    "exercise_module_a_v2_state_machine_for_review",
    "load_verified_candidate_receipt_bundle",
    "load_verified_run_history_ledger",
]
