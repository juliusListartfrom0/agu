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
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

from app.analysis.task0258_module_a_v2 import (
    AUTHORIZATION_PROVIDER_ORDER,
    compact_canonical_json,
    is_safe_slug,
    is_sha256,
    verify_internal_artifact_hash,
)
from app.analysis.task0258_run_history import (
    claim_filename,
    completion_filename,
    replay_run_history_registry,
    verify_run_admission,
    verify_run_consumption_claim,
    verify_run_consumption_completed,
)
from app.analysis.task0258_v2_artifacts import (
    CANDIDATE_MEMBER_PATHS,
    verify_candidate_gate,
    verify_candidate_receipt_bundle,
    verify_postpublication_failure,
    verify_postpublication_verification,
)
from app.analysis.task0258_v2_fs import verify_generation_directory
from app.analysis.task0258_v2_pipeline import build_member_receipts
from app.analysis.task0258_v2_read_isolation import verify_read_isolation_binding
from app.analysis.task0258_v2_verification import verify_verification_attempt

_DISCOVERY_TOKEN = object()
_SANDBOX_TOKEN = object()
_SYNTHETIC_DISCOVERY_TOKEN = object()
_SYNTHETIC_REVIEW_TOKEN = object()
_JSON_ARTIFACT_TOKEN = object()
_READ_ISOLATION_TOKEN = object()
_VERIFICATION_ATTEMPT_TOKEN = object()
_ATTEMPT_SPINE_TOKEN = object()
_RUN_HISTORY_TOKEN = object()
_RUN_ADMISSION_TOKEN = object()

_REVIEW_CHECKS = frozenset({"focused_pytest", "full_pytest"})
_MAX_VERIFIED_FILE_BYTES = 67_108_864


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


class VerifiedReviewReadIsolationBinding:
    """Review-only pair of canonical read-isolation provider artifacts.

    The object proves only that the two supplied artifacts are internally
    valid and cross-bound.  It does not prove that the provider is an
    authenticated kernel-audit source and cannot be used as production
    authorization.
    """

    __slots__ = ("_token", "policy", "attestation")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _READ_ISOLATION_TOKEN:
            raise TypeError("VerifiedReviewReadIsolationBinding cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.policy = kwargs["policy"]
        self.attestation = kwargs["attestation"]


class VerifiedReviewVerificationAttempt:
    """Review-only canonical verification-attempt artifact.

    Loading this object proves only the local schema and nested read-isolation
    binding. It is not a worker admission, kernel attestation, or production
    execution capability.
    """

    __slots__ = ("_token", "artifact")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _VERIFICATION_ATTEMPT_TOKEN:
            raise TypeError("VerifiedReviewVerificationAttempt cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.artifact = kwargs["artifact"]


class VerifiedReviewVerificationAttemptRunSpine:
    """Review-only binding of one attempt to its loaded run trust spine."""

    __slots__ = ("_token", "attempt", "run_admission", "run_history")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _ATTEMPT_SPINE_TOKEN:
            raise TypeError("VerifiedReviewVerificationAttemptRunSpine cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.attempt = kwargs["attempt"]
        self.run_admission = kwargs["run_admission"]
        self.run_history = kwargs["run_history"]


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


class VerifiedReviewRunAdmission:
    """Review-only replay of claim, admission, and completion bytes.

    This deliberately is not a production admission capability.  It exists
    so local adversarial tests can prove the three-file receipt spine and
    physical output-root identity before a future OS-bound runner is present.
    """

    __slots__ = ("_token", "claim", "admission", "completion", "root_identity")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _RUN_ADMISSION_TOKEN:
            raise TypeError("VerifiedReviewRunAdmission cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.claim = kwargs["claim"]
        self.admission = kwargs["admission"]
        self.completion = kwargs["completion"]
        self.root_identity = kwargs["root_identity"]


def _verify_sha(value: object, name: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _verify_review_check(check_name: object, command_sha256: object) -> None:
    if check_name not in _REVIEW_CHECKS:
        raise ValueError("review check name is not allowed")
    _verify_sha(command_sha256, "review command hash")


def _verify_real_directory(path: Path) -> tuple[int, int]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("path must be an absolute real directory")
    identity = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(identity.st_mode):
        raise ValueError("path is not a directory")
    return identity.st_dev, identity.st_ino


def _verify_absolute_no_symlink_path(path: Path) -> None:
    path = Path(path)
    if not path.is_absolute() or os.path.normpath(os.fspath(path)) != os.fspath(path):
        raise ValueError("path must be an absolute canonical path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                raise ValueError(f"path contains a symlinked component: {current}")
        except OSError as exc:
            raise ValueError(f"cannot inspect path component: {current}") from exc


def _verify_review_temp_directory(path: Path) -> tuple[int, int]:
    """Require a canonical real directory inside the OS temporary root."""
    path = Path(path)
    _verify_absolute_no_symlink_path(path)
    identity = _verify_real_directory(path)
    try:
        path.relative_to(Path(tempfile.gettempdir()).resolve())
    except ValueError as exc:
        raise ValueError("review-only path must be under the OS temporary directory") from exc
    return identity


def _verify_temp_ancestor(path: Path) -> tuple[int, int]:
    identity = _verify_real_directory(path)
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        path.resolve().relative_to(temp_root)
    except ValueError as exc:
        raise ValueError("synthetic context must be under the OS temporary directory") from exc
    return identity


def _verify_no_symlink_ancestors(path: Path, *, allowed_prefix: Path) -> None:
    """Reject symlinked components before descriptor-relative file opening."""
    if not path.is_absolute() or "\x00" in os.fspath(path):
        raise ValueError("path must be an absolute path without NUL bytes")
    try:
        relative_parts = path.relative_to(allowed_prefix).parts
    except ValueError as exc:
        raise ValueError("path is outside its allowed prefix") from exc
    current = allowed_prefix
    for part in relative_parts:
        current /= part
        try:
            if current.is_symlink():
                raise ValueError(f"path contains a symlinked component: {current}")
        except OSError as exc:
            raise ValueError(f"cannot inspect path component: {current}") from exc


def _read_no_follow_temp_file(path: Path, *, description: str) -> bytes:
    """Read one bounded temporary regular file through no-follow descriptors."""
    path = Path(path)
    if not path.is_absolute() or os.path.normpath(os.fspath(path)) != os.fspath(path):
        raise ValueError(f"{description} must be an absolute canonical path")
    raw_temp_root = Path(tempfile.gettempdir())
    temp_root = raw_temp_root.resolve()
    lexical_temp_root = raw_temp_root
    try:
        path.relative_to(raw_temp_root)
    except ValueError:
        lexical_temp_root = temp_root
    try:
        canonical_path = path.resolve(strict=True)
        canonical_path.relative_to(temp_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"{description} must be under the OS temporary directory") from exc
    _verify_no_symlink_ancestors(path, allowed_prefix=lexical_temp_root)

    components = canonical_path.parts[1:]
    if not components or any(not component for component in components):
        raise ValueError(f"{description} contains an empty path component")
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = common_flags | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = common_flags | getattr(os, "O_NOFOLLOW", 0)
    parent_fd: int | None = None
    file_fd: int | None = None
    try:
        parent_fd = os.open("/", directory_flags)
        for component in components[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        file_fd = os.open(components[-1], file_flags, dir_fd=parent_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > _MAX_VERIFIED_FILE_BYTES:
            raise ValueError(f"{description} is not a bounded regular file")
        remaining = file_stat.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(file_fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"{description} ended before its recorded size")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError(f"{description} cannot be opened without following links") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if parent_fd is not None:
            os.close(parent_fd)


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
    raw = _read_no_follow_temp_file(path, description="discovery manifest")
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
    raw = _read_no_follow_temp_file(path, description="verified artifact")
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
    payload = dict(payload)
    # Canonical JSON sorts object keys on disk; restore the schema's fixed
    # provider iteration order only after the canonical bytes are accepted.
    authorization_receipts = payload.get("authorization_receipts")
    if isinstance(authorization_receipts, Mapping) and set(authorization_receipts) == set(AUTHORIZATION_PROVIDER_ORDER):
        payload["authorization_receipts"] = {
            provider: authorization_receipts[provider] for provider in AUTHORIZATION_PROVIDER_ORDER
        }
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


def load_verified_read_isolation_binding(
    *,
    execution_context: object,
    policy_path: Path,
    expected_policy_artifact_sha256: str,
    expected_policy_file_sha256: str,
    attestation_path: Path,
    expected_attestation_artifact_sha256: str,
    expected_attestation_file_sha256: str,
) -> VerifiedReviewReadIsolationBinding:
    """Load and cross-bind read-isolation artifacts in a review sandbox.

    This loader is deliberately diagnostic-only.  It reopens both canonical
    JSON files through the existing bounded no-follow loader, then validates
    the policy↔attestation binding.  The external kernel-audit provenance
    required for production remains outside this function.
    """
    if (
        type(execution_context) is not VerifiedImplementationReviewSandboxContext
        or execution_context._token is not _SANDBOX_TOKEN
    ):
        raise PermissionError("read-isolation loader requires a verified review context")
    policy = _load_verified_json_artifact(
        path=policy_path,
        expected_artifact_sha256=expected_policy_artifact_sha256,
        expected_file_sha256=expected_policy_file_sha256,
    )
    attestation = _load_verified_json_artifact(
        path=attestation_path,
        expected_artifact_sha256=expected_attestation_artifact_sha256,
        expected_file_sha256=expected_attestation_file_sha256,
    )
    verify_read_isolation_binding(policy.payload, attestation.payload)
    return VerifiedReviewReadIsolationBinding(
        _READ_ISOLATION_TOKEN,
        policy=policy,
        attestation=attestation,
    )


def load_verified_verification_attempt(
    *,
    execution_context: object,
    attempt_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
) -> VerifiedReviewVerificationAttempt:
    """Reopen and validate one verification attempt in a review sandbox.

    The attempt validator includes the policy↔attestation binder. This
    function therefore verifies the complete canonical attempt bytes before
    returning an opaque diagnostic object, while leaving production admission
    and external kernel-audit provenance outside the local loader.
    """
    if (
        type(execution_context) is not VerifiedImplementationReviewSandboxContext
        or execution_context._token is not _SANDBOX_TOKEN
    ):
        raise PermissionError("verification attempt loader requires a verified review context")
    artifact = _load_verified_json_artifact(
        path=attempt_path,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_file_sha256=expected_file_sha256,
    )
    verify_verification_attempt(artifact.payload)
    return VerifiedReviewVerificationAttempt(_VERIFICATION_ATTEMPT_TOKEN, artifact=artifact)


def _artifact_file_receipt(artifact: VerifiedJsonArtifact) -> dict[str, str]:
    return {
        "artifact_sha256": artifact.artifact_sha256,
        "file_sha256": artifact.file_sha256,
    }


def _payload_file_receipt(payload: Mapping[str, object]) -> dict[str, str]:
    raw = (compact_canonical_json(payload) + "\n").encode("utf-8")
    artifact_sha256 = payload.get("artifact_sha256")
    if not is_sha256(artifact_sha256):
        raise ValueError("history payload artifact hash is invalid")
    return {"artifact_sha256": artifact_sha256, "file_sha256": hashlib.sha256(raw).hexdigest()}


def _history_contract(ledger: VerifiedRunHistoryLedger) -> dict[str, object]:
    if type(ledger) is not VerifiedRunHistoryLedger or len(ledger.payloads) < 2:
        raise TypeError("verified run-history ledger is invalid")
    completion = ledger.payloads[1]
    head = ledger.payloads[-1]
    return {
        "run_identity_receipt": _payload_file_receipt(completion),
        "head_receipt": _payload_file_receipt(head),
        "marker_count": len(ledger.payloads) - 2,
    }


def bind_verified_review_attempt_to_run_spine(
    *,
    attempt: VerifiedReviewVerificationAttempt,
    run_admission: VerifiedReviewRunAdmission,
    run_history: VerifiedRunHistoryLedger,
) -> VerifiedReviewVerificationAttemptRunSpine:
    """Bind a verified attempt to one loaded admission/history trust spine.

    This is a review-only cross-artifact check. It proves that the attempt's
    run-admission, stable run identity, history head, rerun authorization, and
    read-isolation output-root tuple describe the same local replay. It does
    not issue production admission or external kernel evidence.
    """
    if type(attempt) is not VerifiedReviewVerificationAttempt or attempt._token is not _VERIFICATION_ATTEMPT_TOKEN:
        raise TypeError("attempt is not a verified review artifact")
    if type(run_admission) is not VerifiedReviewRunAdmission or run_admission._token is not _RUN_ADMISSION_TOKEN:
        raise TypeError("run admission is not a verified review artifact")
    if type(run_history) is not VerifiedRunHistoryLedger or run_history._token is not _RUN_HISTORY_TOKEN:
        raise TypeError("run history is not a verified review ledger")

    payload = attempt.artifact.payload
    admission_payload = run_admission.admission.payload
    history_contract = _history_contract(run_history)
    authorization_receipts = payload["authorization_receipts"]
    if not isinstance(authorization_receipts, Mapping):
        raise ValueError("attempt authorization receipts are invalid")
    if payload["run_admission_receipt"] != _artifact_file_receipt(run_admission.admission):
        raise ValueError("attempt admission receipt is not bound to admission")
    if payload["run_identity_receipt"] != history_contract["run_identity_receipt"]:
        raise ValueError("attempt run identity receipt is not bound to history")
    if payload["history_head_receipt"] != history_contract["head_receipt"]:
        raise ValueError("attempt history head receipt is not bound to history")
    if authorization_receipts["rerun_authorization"] != admission_payload["authorization_receipt"]:
        raise ValueError("attempt rerun authorization is not bound to admission")
    if run_history.authorization_sha256 != admission_payload["authorization_receipt"]["artifact_sha256"]:
        raise ValueError("run history authorization is not bound to admission")

    policy = payload["read_isolation_policy"]
    attestation = payload["read_isolation_attestation"]
    if policy["run_identity_receipt"] != payload["run_identity_receipt"]:
        raise ValueError("attempt read-isolation policy identity is not bound to attempt")
    if attestation["run_identity_receipt"] != payload["run_identity_receipt"]:
        raise ValueError("attempt read-isolation attestation identity is not bound to attempt")
    policy_root = policy["output_root_identity"]
    if policy_root["absolute_path"] != admission_payload["output_root_absolute_path"]:
        raise ValueError("attempt read-isolation root is not bound to admission")
    if policy_root["device"] != run_admission.completion.payload["root_identity"]["device"]:
        raise ValueError("attempt read-isolation root device is not bound to completion")
    if policy_root["inode"] != run_admission.completion.payload["root_identity"]["inode"]:
        raise ValueError("attempt read-isolation root inode is not bound to completion")

    return VerifiedReviewVerificationAttemptRunSpine(
        _ATTEMPT_SPINE_TOKEN,
        attempt=attempt,
        run_admission=run_admission,
        run_history=run_history,
    )


def _load_candidate_gate_artifact(
    *, candidate_dir: Path, member_receipts: list[dict[str, object]]
) -> VerifiedJsonArtifact:
    gate_row = next((row for row in member_receipts if row["relative_path"] == "candidate_gate.json"), None)
    if not isinstance(gate_row, Mapping) or gate_row["receipt_kind"] != "json":
        raise ValueError("candidate gate member receipt is missing")
    gate = _load_verified_json_artifact(
        path=Path(candidate_dir) / "candidate_gate.json",
        expected_artifact_sha256=gate_row["artifact_sha256"],
        expected_file_sha256=gate_row["file_sha256"],
    )
    verify_candidate_gate(gate.payload)
    return gate


def load_verified_run_admission(
    *,
    execution_context: object,
    claim_path: Path,
    admission_path: Path,
    completion_path: Path,
    expected_authorization_sha256: str,
    expected_claim_artifact_sha256: str,
    expected_claim_file_sha256: str,
    expected_admission_artifact_sha256: str,
    expected_admission_file_sha256: str,
    expected_completion_artifact_sha256: str,
    expected_completion_file_sha256: str,
) -> VerifiedReviewRunAdmission:
    """Replay the claim/admission/completion spine in a review context.

    The loader validates canonical bytes, receipt edges, output-root identity,
    and the completion's admission CAS.  It remains review-only and cannot
    issue the production admission capability described by the amendment.
    """
    if type(execution_context) is not VerifiedImplementationReviewSandboxContext:
        raise PermissionError("run admission loader requires a verified review context")
    authorization_sha256 = _verify_sha(expected_authorization_sha256, "authorization hash")
    claim_path = Path(claim_path)
    admission_path = Path(admission_path)
    completion_path = Path(completion_path)
    if claim_path.name != claim_filename(authorization_sha256):
        raise ValueError("claim basename is not bound to the authorization")
    if completion_path.name != completion_filename(authorization_sha256):
        raise ValueError("completion basename is not bound to the authorization")
    if claim_path.parent != completion_path.parent:
        raise ValueError("claim and completion must share one registry directory")
    if admission_path.name != "run_admission.json":
        raise ValueError("admission basename is not authorized")

    claim = _load_verified_json_artifact(
        path=claim_path,
        expected_artifact_sha256=expected_claim_artifact_sha256,
        expected_file_sha256=expected_claim_file_sha256,
    )
    admission = _load_verified_json_artifact(
        path=admission_path,
        expected_artifact_sha256=expected_admission_artifact_sha256,
        expected_file_sha256=expected_admission_file_sha256,
    )
    completion = _load_verified_json_artifact(
        path=completion_path,
        expected_artifact_sha256=expected_completion_artifact_sha256,
        expected_file_sha256=expected_completion_file_sha256,
    )
    verify_run_consumption_claim(claim.payload)
    verify_run_admission(admission.payload)
    verify_run_consumption_completed(completion.payload)

    for payload in (claim.payload, admission.payload, completion.payload):
        authorization_receipt = payload["authorization_receipt"]
        if authorization_receipt["artifact_sha256"] != authorization_sha256:
            raise ValueError("run admission authorization binding drifted")
        if payload["run_id"] != claim.payload["run_id"] or payload["nonce"] != claim.payload["nonce"]:
            raise ValueError("run admission identity binding drifted")
        if payload["output_root_absolute_path"] != claim.payload["output_root_absolute_path"]:
            raise ValueError("run admission output-root binding drifted")

    claim_receipt = _artifact_file_receipt(claim)
    admission_receipt = _artifact_file_receipt(admission)
    if admission.payload["claim_receipt"] != claim_receipt:
        raise ValueError("admission claim receipt does not match claim bytes")
    if completion.payload["claim_receipt"] != claim_receipt:
        raise ValueError("completion claim receipt does not match claim bytes")
    if completion.payload["admission_receipt"] != admission_receipt:
        raise ValueError("completion admission receipt does not match admission bytes")

    output_root = Path(admission.payload["output_root_absolute_path"])
    if output_root != admission.path.parent:
        raise ValueError("admission output root is not the loaded directory")
    root_identity = _verify_review_temp_directory(output_root)
    expected_root_identity = {
        "device": root_identity[0],
        "inode": root_identity[1],
    }
    if completion.payload["root_identity"] != expected_root_identity:
        raise ValueError("completion root identity does not match the loaded output root")

    admission_stat = admission.path.stat(follow_symlinks=False)
    if not stat.S_ISREG(admission_stat.st_mode):
        raise ValueError("admission path is not a regular file")
    expected_admission_identity = {
        "device": admission_stat.st_dev,
        "inode": admission_stat.st_ino,
        "size_bytes": admission_stat.st_size,
        "internal_sha256": admission.artifact_sha256,
        "file_sha256": admission.file_sha256,
    }
    if completion.payload["admission_identity"] != expected_admission_identity:
        raise ValueError("completion admission identity does not match the loaded admission bytes")

    return VerifiedReviewRunAdmission(
        _RUN_ADMISSION_TOKEN,
        claim=claim,
        admission=admission,
        completion=completion,
        root_identity=expected_root_identity,
    )


def load_verified_candidate_receipt_bundle(
    *,
    execution_context: object,
    bundle_path: Path,
    candidate_dir: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
) -> VerifiedJsonArtifact:
    """Load a bundle and replay its exact member receipts against ``candidate_v2``."""
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
    candidate_dir = Path(candidate_dir)
    if candidate_dir.name != "candidate_v2":
        raise ValueError("candidate bundle replay requires candidate_v2")
    _verify_review_temp_directory(candidate_dir)
    verify_generation_directory(candidate_dir, CANDIDATE_MEMBER_PATHS)
    actual_member_receipts = build_member_receipts(candidate_dir)
    if list(artifact.payload["ordered_member_receipts"]) != actual_member_receipts:
        raise ValueError("candidate bundle member receipts do not match candidate_v2 bytes")
    _load_candidate_gate_artifact(candidate_dir=candidate_dir, member_receipts=actual_member_receipts)
    verify_generation_directory(candidate_dir, CANDIDATE_MEMBER_PATHS)
    return artifact


def load_verified_terminal_artifact(
    *,
    execution_context: object,
    terminal_kind: str,
    output_root: Path,
    candidate_dir: Path,
    candidate_bundle_path: Path,
    expected_candidate_bundle_artifact_sha256: str,
    expected_candidate_bundle_file_sha256: str,
    run_admission: VerifiedReviewRunAdmission,
    run_history: VerifiedRunHistoryLedger,
    terminal_path: Path,
    expected_terminal_artifact_sha256: str,
    expected_terminal_file_sha256: str,
) -> VerifiedJsonArtifact:
    """Replay one existing terminal generation and its trust-spine inputs.

    Only the two locally implemented post-candidate terminal variants are
    accepted.  The loader requires exactly one terminal sibling, replays the
    candidate bundle against candidate bytes, and binds terminal receipts to
    the loaded admission and durable history ledger.
    """
    if type(execution_context) is not VerifiedImplementationReviewSandboxContext:
        raise PermissionError("terminal loader requires a verified review context")
    if type(run_admission) is not VerifiedReviewRunAdmission:
        raise TypeError("terminal loader requires a verified review-only admission")
    if type(run_history) is not VerifiedRunHistoryLedger:
        raise TypeError("terminal loader requires a verified run-history ledger")
    targets = {
        "verified_result": ("verified_result_v2", "verification_registry.json"),
        "postverification_failure": ("postverification_failure_v2", "failure.json"),
    }
    if terminal_kind not in targets:
        raise ValueError("terminal kind is not implemented by this review loader")
    terminal_name, member_name = targets[terminal_kind]

    output_root = Path(output_root)
    candidate_dir = Path(candidate_dir)
    terminal_path = Path(terminal_path)
    _verify_review_temp_directory(output_root)
    if candidate_dir != output_root / "candidate_v2":
        raise ValueError("terminal candidate path is not bound to the output root")
    _verify_review_temp_directory(candidate_dir)
    verify_generation_directory(candidate_dir, CANDIDATE_MEMBER_PATHS)
    expected_terminal_dir = output_root / terminal_name
    if terminal_path != expected_terminal_dir / member_name:
        raise ValueError("terminal artifact path is not bound to its fixed generation")
    for sibling in ("terminal_failure_v2", "verified_result_v2", "postverification_failure_v2"):
        sibling_path = output_root / sibling
        if sibling == terminal_name:
            verify_generation_directory(sibling_path, (member_name,))
        elif sibling_path.exists() or sibling_path.is_symlink():
            raise ValueError("terminal topology contains a competing generation")
    if terminal_path.is_symlink():
        raise ValueError("terminal artifact cannot be a symlink")

    bundle = load_verified_candidate_receipt_bundle(
        execution_context=execution_context,
        bundle_path=candidate_bundle_path,
        candidate_dir=candidate_dir,
        expected_artifact_sha256=expected_candidate_bundle_artifact_sha256,
        expected_file_sha256=expected_candidate_bundle_file_sha256,
    )
    if bundle.path.is_relative_to(output_root):
        raise ValueError("candidate receipt bundle must remain outside the output root")

    admission_payload = run_admission.admission.payload
    admission_receipt = _artifact_file_receipt(run_admission.admission)
    history_contract = _history_contract(run_history)
    candidate_gate = _load_candidate_gate_artifact(
        candidate_dir=candidate_dir,
        member_receipts=bundle.payload["ordered_member_receipts"],
    ).payload
    gate_receipts = {slot["provider"]: slot["receipt"] for slot in candidate_gate["input_receipts"]}
    if run_history.authorization_sha256 != bundle.payload["authorization_receipt"]["artifact_sha256"]:
        raise ValueError("terminal history authorization is not bound to the bundle")
    if bundle.payload["authorization_receipt"] != admission_payload["authorization_receipt"]:
        raise ValueError("candidate bundle authorization is not bound to admission")
    if bundle.payload["run_admission_receipt"] != admission_receipt:
        raise ValueError("candidate bundle admission receipt is not bound to admission bytes")
    if bundle.payload["static_input_contract"] != admission_payload["static_input_contract"]:
        raise ValueError("candidate bundle static inputs are not bound to admission")
    if bundle.payload["run_identity_receipt"] != history_contract["run_identity_receipt"]:
        raise ValueError("candidate bundle run identity is not bound to history")
    if bundle.payload["candidate_published_history_head_receipt"] != history_contract["head_receipt"]:
        raise ValueError("candidate bundle history head is not bound to history")
    if gate_receipts["exact_v2_rerun_authorization"] != bundle.payload["authorization_receipt"]:
        raise ValueError("candidate gate authorization is not bound to the bundle")
    if gate_receipts["run_admission"] != admission_receipt:
        raise ValueError("candidate gate admission is not bound to admission bytes")
    if gate_receipts["static_inputs"] != admission_payload["static_input_contract"]:
        raise ValueError("candidate gate static inputs are not bound to admission")
    if gate_receipts["run_history_ledger"] != history_contract:
        raise ValueError("candidate gate history is not bound to history")

    terminal = _load_verified_json_artifact(
        path=terminal_path,
        expected_artifact_sha256=expected_terminal_artifact_sha256,
        expected_file_sha256=expected_terminal_file_sha256,
    )
    bundle_receipt = _artifact_file_receipt(bundle)
    if terminal_kind == "verified_result":
        verify_postpublication_verification(terminal.payload)
        result = terminal.payload
        if result["run_history_contract_receipt"] != history_contract:
            raise ValueError("verified result history contract is not bound to history")
        if result["run_admission_receipt"] != admission_receipt:
            raise ValueError("verified result admission receipt is not bound to admission")
        if result["static_input_contract"] != admission_payload["static_input_contract"]:
            raise ValueError("verified result static inputs are not bound to admission")
        if result["authorization_receipts"]["rerun_authorization"] != bundle.payload["authorization_receipt"]:
            raise ValueError("verified result authorization is not bound to the bundle")
        if result["candidate_receipt_bundle_receipt"] != bundle_receipt:
            raise ValueError("verified result bundle receipt does not match bundle bytes")
        if result["candidate_member_receipts"] != bundle.payload["ordered_member_receipts"]:
            raise ValueError("verified result candidate receipts do not match candidate bytes")
    else:
        verify_postpublication_failure(terminal.payload)
        slots = {slot["provider"]: slot for slot in terminal.payload["dependency_provider_slots"]}
        if slots["exact_v2_rerun_authorization"]["receipt"] != bundle.payload["authorization_receipt"]:
            raise ValueError("postverification failure authorization is not bound to the bundle")
        if slots["run_history_ledger"]["receipt"] != history_contract:
            raise ValueError("postverification failure history is not bound to history")
        if slots["run_admission"]["receipt"] != admission_receipt:
            raise ValueError("postverification failure admission is not bound to admission")
        if slots["static_inputs"]["receipt"] != admission_payload["static_input_contract"]:
            raise ValueError("postverification failure static inputs are not bound to admission")
        if slots["candidate_receipt_bundle"]["receipt"] != bundle_receipt:
            raise ValueError("postverification failure bundle receipt does not match bundle bytes")
    return terminal


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
    "VerifiedReviewReadIsolationBinding",
    "VerifiedReviewVerificationAttempt",
    "VerifiedReviewVerificationAttemptRunSpine",
    "VerifiedRunHistoryLedger",
    "VerifiedReviewRunAdmission",
    "bind_implementation_review_discovery_context",
    "bind_implementation_review_sandbox_context",
    "replay_module_a_read_traversal_for_discovery",
    "bind_synthetic_module_a_discovery_context",
    "bind_synthetic_module_a_transaction_context",
    "exercise_module_a_v2_state_machine_for_discovery",
    "exercise_module_a_v2_state_machine_for_review",
    "load_verified_run_admission",
    "load_verified_read_isolation_binding",
    "load_verified_verification_attempt",
    "bind_verified_review_attempt_to_run_spine",
    "load_verified_candidate_receipt_bundle",
    "load_verified_terminal_artifact",
    "load_verified_run_history_ledger",
]
