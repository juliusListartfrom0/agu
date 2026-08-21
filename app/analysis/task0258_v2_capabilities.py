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
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from app.analysis.task0258_module_a_v2 import (
    AUTHORIZATION_PROVIDER_ORDER,
    compact_canonical_json,
    is_rfc3339,
    is_safe_slug,
    is_sha256,
    verify_artifact_file_receipt,
    verify_internal_artifact_hash,
)
from app.analysis.task0258_run_history import (
    claim_filename,
    completion_filename,
    registry_history_filenames,
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
_PREFLIGHT_TOKEN = object()
_IMPLEMENTATION_APPROVAL_TOKEN = object()
_PARENT_SPEC_APPROVAL_TOKEN = object()
_AMENDED_IMPLEMENTATION_REVIEW_TOKEN = object()
_RERUN_AUTHORIZATION_TOKEN = object()
_RUN_HISTORY_TOKEN = object()
_RUN_ADMISSION_TOKEN = object()

_REVIEW_CHECKS = frozenset({"focused_pytest", "full_pytest"})
_MAX_VERIFIED_FILE_BYTES = 67_108_864
_IMPLEMENTATION_APPROVAL_SCHEMA = "agu.task0258-module-a-amendment-implementation-approval.v1"
_IMPLEMENTATION_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "repository_root_absolute_path",
        "repository_root_device",
        "repository_root_inode",
        "parent_spec_approval_receipt",
        "approved_parent_file_receipts",
        "approved_amendment_file_receipt",
        "amendment_fresh_review_receipt",
        "implementation_scope_baseline_receipt",
        "approval_scope",
        "model_execution_authorized",
        "module_b_authorized",
        "approval_statement_sha256",
        "approved_at_utc",
        "artifact_sha256",
    }
)
_PARENT_SPEC_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "approved_files",
        "fresh_review_receipt",
        "approval_scope",
        "approval_statement_sha256",
        "approved_at_utc",
        "artifact_sha256",
    }
)
_IMPLEMENTATION_SCOPE_BASELINE_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "repository_root_absolute_path",
        "repository_root_device",
        "repository_root_inode",
        "ordered_root_paths",
        "check_output_directory_absolute_path",
        "check_output_directory_device",
        "check_output_directory_inode",
        "ordered_entry_receipts",
        "ordered_repository_executable_receipts",
        "captured_at_utc",
        "artifact_sha256",
    }
)
_AMENDED_IMPLEMENTATION_REVIEW_SCHEMA = "agu.task0258-module-a-fresh-code-review.v1"
_AMENDED_IMPLEMENTATION_REVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "repository_root_absolute_path",
        "repository_root_device",
        "repository_root_inode",
        "amendment_implementation_approval_receipt",
        "implementation_context_id",
        "reviewer_context_id",
        "reviewer_independence",
        "ordered_code_file_receipts",
        "ordered_test_file_receipts",
        "ordered_runtime_dependency_file_receipts",
        "ordered_check_configuration_file_receipts",
        "ordered_check_input_receipt_sets",
        "implementation_scope_baseline_receipt",
        "implementation_scope_delta",
        "bootstrap_launcher_receipt",
        "ordered_check_receipts",
        "review_resource_summary",
        "critical_count",
        "required_count",
        "optional_count",
        "heavy_execution_performed",
        "reviewed_at_utc",
        "artifact_sha256",
    }
)
_FRESH_IMPLEMENTATION_REVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "repository_root_absolute_path",
        "repository_root_device",
        "repository_root_inode",
        "amendment_implementation_approval_receipt",
        "implementation_context_id",
        "reviewer_context_id",
        "reviewer_independence",
        "ordered_code_file_receipts",
        "ordered_test_file_receipts",
        "ordered_runtime_dependency_file_receipts",
        "ordered_check_configuration_file_receipts",
        "ordered_check_input_receipt_sets",
        "implementation_scope_baseline_receipt",
        "implementation_scope_delta",
        "bootstrap_launcher_receipt",
        "ordered_pre_review_check_receipts",
        "fresh_review_governance_observation",
        "critical_count",
        "required_count",
        "optional_count",
        "heavy_execution_performed",
        "reviewed_at_utc",
        "artifact_sha256",
    }
)
_AMENDED_REVIEW_CODE_PATHS = (
    "app/analysis/vru_causal_temporal_retrospective.py",
    "scripts/extract_vru_causal_tiled_swin_embeddings.py",
    "scripts/screen_vru_causal_temporal_retrospective.py",
    "scripts/seal_vru_causal_temporal_feature_plan.py",
    "scripts/task0258_module_a_verified_bootstrap.py",
)
_AMENDED_REVIEW_TEST_PATHS = (
    "tests/test_task0258_module_a_cli.py",
    "tests/test_vru_causal_final_evaluator.py",
    "tests/test_vru_causal_temporal_feature_plan.py",
    "tests/test_vru_causal_temporal_retrospective.py",
    "tests/test_vru_causal_tiled_swin_embeddings.py",
)
_AMENDED_REVIEW_CHECK_NAMES = (
    "focused_pytest",
    "full_pytest",
    "ruff_check",
    "ruff_format_check",
    "diff_check",
    "fresh_context_code_review",
)
_AMENDED_REVIEW_CHECK_FIELDS = frozenset(
    {
        "check_name",
        "execution_protocol",
        "sandbox_attestation_sha256",
        "command_sha256",
        "exit_code",
        "output_path",
        "output_size_bytes",
        "output_artifact_sha256",
        "output_file_sha256",
        "completed_at_utc",
    }
)
_RERUN_AUTHORIZATION_SCHEMA = "agu.task0258-module-a-v2-rerun-authorization.v1"
_RERUN_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "repository_root_absolute_path",
        "repository_root_device",
        "repository_root_inode",
        "amendment_implementation_approval_receipt",
        "implementation_review_receipt",
        "approved_amendment_file_receipt",
        "approved_code_receipts",
        "approved_test_receipts",
        "approved_runtime_dependency_receipts",
        "approved_check_configuration_receipts",
        "approved_check_input_receipt_sets",
        "approved_check_receipts",
        "approved_implementation_scope_baseline_receipt",
        "approved_implementation_scope_delta",
        "approved_bootstrap_launcher_receipt",
        "approved_static_input_contract",
        "allowed_operations",
        "output_root_absolute_path",
        "candidate_receipt_bundle_absolute_path",
        "run_id",
        "authorization_scope",
        "maximum_run_count",
        "run_admission_relative_path",
        "run_consumption_registry_directory_absolute_path",
        "run_consumption_registry_directory_identity",
        "module_b_authorized",
        "approval_statement_sha256",
        "approved_at_utc",
        "artifact_sha256",
    }
)
_RERUN_ALLOWED_OPERATIONS = (
    "preflight_and_publish_admission",
    "recover_admission_completion",
    "publish_candidate",
    "seal_candidate_receipt_bundle",
    "postpublication_verify",
    "load_existing_terminal",
)


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


class VerifiedReviewNoWritePreflight:
    """Review-only replay of the complete no-write publication preflight.

    This object records the identities and planned paths that a future
    externally authorized runner would revalidate immediately before
    publication. It deliberately carries no production authorization and is
    rejected by :func:`run_v2_pipeline`.
    """

    __slots__ = (
        "_token",
        "attempt_spine",
        "output_root",
        "output_root_identity",
        "registry_directory",
        "registry_identity",
        "candidate_bundle_path",
        "planned_paths",
        "production_capability",
    )

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _PREFLIGHT_TOKEN:
            raise TypeError("VerifiedReviewNoWritePreflight cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.attempt_spine = kwargs["attempt_spine"]
        self.output_root = kwargs["output_root"]
        self.output_root_identity = kwargs["output_root_identity"]
        self.registry_directory = kwargs["registry_directory"]
        self.registry_identity = kwargs["registry_identity"]
        self.candidate_bundle_path = kwargs["candidate_bundle_path"]
        self.planned_paths = tuple(kwargs["planned_paths"])
        self.production_capability = False


class VerifiedReviewParentModuleASpecApproval:
    """Review-only replay of the exact parent Module-A specification approval."""

    __slots__ = ("_token", "artifact", "approved_files", "production_capability")

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _PARENT_SPEC_APPROVAL_TOKEN:
            raise TypeError("VerifiedReviewParentModuleASpecApproval cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.artifact = kwargs["artifact"]
        self.approved_files = tuple(kwargs["approved_files"])
        self.production_capability = False


class VerifiedReviewImplementationApproval:
    """Review-only replay of the sealed amendment implementation approval."""

    __slots__ = (
        "_token",
        "artifact",
        "parent_approval",
        "amendment",
        "amendment_review",
        "implementation_scope_baseline",
        "repository_root_identity",
        "production_capability",
    )

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _IMPLEMENTATION_APPROVAL_TOKEN:
            raise TypeError("VerifiedReviewImplementationApproval cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.artifact = kwargs["artifact"]
        self.parent_approval = kwargs.get("parent_approval")
        self.amendment = kwargs.get("amendment")
        self.amendment_review = kwargs.get("amendment_review")
        self.implementation_scope_baseline = kwargs.get("implementation_scope_baseline")
        self.repository_root_identity = kwargs["repository_root_identity"]
        self.production_capability = False


class VerifiedReviewAmendedImplementationReview:
    """Review-only replay of the amended implementation and fresh review.

    This object binds the externally receipted implementation-review artifact
    to the already loaded implementation approval and its separate fresh
    reviewer artifact.  It is evidence for local diagnostics only; it cannot
    authorize a rerun, a worker, or production publication.
    """

    __slots__ = (
        "_token",
        "artifact",
        "fresh_review",
        "implementation_approval",
        "repository_root_identity",
        "production_capability",
    )

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _AMENDED_IMPLEMENTATION_REVIEW_TOKEN:
            raise TypeError("VerifiedReviewAmendedImplementationReview cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.artifact = kwargs["artifact"]
        self.fresh_review = kwargs["fresh_review"]
        self.implementation_approval = kwargs["implementation_approval"]
        self.repository_root_identity = kwargs["repository_root_identity"]
        self.production_capability = False


class VerifiedReviewRerunAuthorization:
    """Review-only replay of the exact v2 rerun-authorization receipt."""

    __slots__ = (
        "_token",
        "artifact",
        "implementation_approval",
        "implementation_review",
        "repository_root_identity",
        "output_root",
        "candidate_bundle_path",
        "production_capability",
    )

    def __new__(cls, token: object = None, **kwargs: object):
        if token is not _RERUN_AUTHORIZATION_TOKEN:
            raise TypeError("VerifiedReviewRerunAuthorization cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object = None, **kwargs: object) -> None:
        self._token = token
        self.artifact = kwargs["artifact"]
        self.implementation_approval = kwargs["implementation_approval"]
        self.implementation_review = kwargs["implementation_review"]
        self.repository_root_identity = kwargs["repository_root_identity"]
        self.output_root = kwargs["output_root"]
        self.candidate_bundle_path = kwargs["candidate_bundle_path"]
        self.production_capability = False


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


def _read_no_follow_file_under_root(path: Path, *, allowed_root: Path, description: str) -> bytes:
    """Read one bounded regular file below a caller-bound real repository root."""
    path = Path(path)
    allowed_root = Path(allowed_root)
    _verify_absolute_no_symlink_path(allowed_root)
    _verify_real_directory(allowed_root)
    if not path.is_absolute() or os.path.normpath(os.fspath(path)) != os.fspath(path):
        raise ValueError(f"{description} must be an absolute canonical path")
    try:
        canonical_root = allowed_root.resolve(strict=True)
        canonical_path = path.resolve(strict=True)
        canonical_path.relative_to(canonical_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"{description} must be below the repository root") from exc
    _verify_no_symlink_ancestors(path, allowed_prefix=allowed_root)
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


def _load_verified_json_artifact_under_root(
    *,
    path: Path,
    repository_root: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    description: str,
) -> VerifiedJsonArtifact:
    path = Path(path)
    raw = _read_no_follow_file_under_root(path, allowed_root=repository_root, description=description)
    if hashlib.sha256(raw).hexdigest() != _verify_sha(expected_file_sha256, f"{description} file hash"):
        raise ValueError(f"{description} file hash does not match")
    if not raw.endswith(b"\n"):
        raise ValueError(f"{description} must end with LF")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not JSON") from exc
    if not isinstance(payload, Mapping) or raw != (compact_canonical_json(payload) + "\n").encode("utf-8"):
        raise ValueError(f"{description} is not canonical JSON")
    payload = dict(payload)
    authorization_receipts = payload.get("authorization_receipts")
    if isinstance(authorization_receipts, Mapping) and set(authorization_receipts) == set(AUTHORIZATION_PROVIDER_ORDER):
        payload["authorization_receipts"] = {
            provider: authorization_receipts[provider] for provider in AUTHORIZATION_PROVIDER_ORDER
        }
    verify_internal_artifact_hash(payload)
    if payload["artifact_sha256"] != _verify_sha(expected_artifact_sha256, f"{description} artifact hash"):
        raise ValueError(f"{description} artifact hash does not match")
    return VerifiedJsonArtifact(
        _JSON_ARTIFACT_TOKEN,
        path=path,
        payload=dict(payload),
        artifact_sha256=expected_artifact_sha256,
        file_sha256=expected_file_sha256,
    )


def _verify_reviewed_file_receipt(value: object, *, basename_only: bool = False) -> None:
    expected_fields = (
        {"filename", "size_bytes", "file_sha256"}
        if basename_only
        else {
            "path",
            "size_bytes",
            "file_sha256",
        }
    )
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("reviewed file receipt shape is invalid")
    path = value["filename"] if basename_only else value["path"]
    if not isinstance(path, str) or not path or "\x00" in path or not is_sha256(value["file_sha256"]):
        raise ValueError("reviewed file receipt path or hash is invalid")
    if basename_only:
        if Path(path).name != path or path in {".", ".."} or "/" in path or "\\" in path:
            raise ValueError("reviewed parent file receipt filename is invalid")
    else:
        relative = Path(path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("reviewed file receipt path is not a safe relative path")
    size = value["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("reviewed file receipt size is invalid")


def _verify_file_receipt_against_path(
    receipt: Mapping[str, object], *, path: Path, repository_root: Path, basename_only: bool
) -> bytes:
    _verify_reviewed_file_receipt(receipt, basename_only=basename_only)
    path = Path(path)
    _verify_absolute_no_symlink_path(path.parent)
    if basename_only:
        if path.name != receipt["filename"]:
            raise ValueError("reviewed file receipt filename does not match supplied path")
    else:
        relative = path.relative_to(repository_root).as_posix()
        if relative != receipt["path"]:
            raise ValueError("reviewed file receipt path does not match supplied path")
    raw = _read_no_follow_file_under_root(path, allowed_root=repository_root, description="reviewed file")
    if len(raw) != receipt["size_bytes"]:
        raise ValueError("reviewed file receipt size does not match")
    if hashlib.sha256(raw).hexdigest() != receipt["file_sha256"]:
        raise ValueError("reviewed file receipt hash does not match")
    return raw


def _verify_parent_spec_approval_artifact(
    payload: Mapping[str, object],
    *,
    approved_spec_paths: Mapping[str, Path],
    expected_fresh_review_internal_sha256: str,
    expected_fresh_review_file_sha256: str,
    expected_approval_statement_sha256: str,
) -> tuple[dict[str, object], ...]:
    if set(payload) != _PARENT_SPEC_APPROVAL_FIELDS:
        raise ValueError("parent spec approval field set is invalid")
    if (
        payload["schema_version"] != "agu.module-a-spec-approval.v1"
        or payload["module_id"] != "existing-45-temporal-retrospective"
    ):
        raise ValueError("parent spec approval identity is invalid")
    if payload["approval_scope"] != "module_a_implementation_only":
        raise ValueError("parent spec approval scope is invalid")
    if payload["approval_statement_sha256"] != _verify_sha(
        expected_approval_statement_sha256, "parent approval statement hash"
    ):
        raise ValueError("parent approval statement hash does not match")
    if not is_rfc3339(payload["approved_at_utc"]):
        raise ValueError("parent spec approval timestamp is invalid")
    fresh_review = payload["fresh_review_receipt"]
    if not isinstance(fresh_review, Mapping) or set(fresh_review) != {"internal_sha256", "file_sha256"}:
        raise ValueError("parent spec approval fresh review receipt is invalid")
    if fresh_review["internal_sha256"] != _verify_sha(
        expected_fresh_review_internal_sha256, "parent fresh review internal hash"
    ) or fresh_review["file_sha256"] != _verify_sha(expected_fresh_review_file_sha256, "parent fresh review file hash"):
        raise ValueError("parent fresh review receipt does not match")
    approved_files = payload["approved_files"]
    if not isinstance(approved_files, list) or len(approved_files) != 3:
        raise ValueError("parent spec approval files are invalid")
    if tuple(approved_spec_paths) != tuple(
        receipt.get("filename") for receipt in approved_files if isinstance(receipt, Mapping)
    ):
        raise ValueError("parent spec approval file order is invalid")
    for receipt in approved_files:
        _verify_reviewed_file_receipt(receipt, basename_only=True)
    for receipt in approved_files:
        path = Path(approved_spec_paths[receipt["filename"]])
        _verify_file_receipt_against_path(receipt, path=path, repository_root=Path(path.anchor), basename_only=True)
    return tuple(dict(receipt) for receipt in approved_files)


def _verify_implementation_scope_baseline_artifact(
    payload: Mapping[str, object], *, repository_root: Path, root_identity: Mapping[str, int]
) -> None:
    if set(payload) != _IMPLEMENTATION_SCOPE_BASELINE_FIELDS:
        raise ValueError("implementation scope baseline field set is invalid")
    if (
        payload["schema_version"] != "agu.task0258-module-a-implementation-scope-baseline.v1"
        or payload["module_id"] != "existing-45-temporal-retrospective"
    ):
        raise ValueError("implementation scope baseline identity is invalid")
    if payload["repository_root_absolute_path"] != str(repository_root):
        raise ValueError("implementation scope baseline repository root is invalid")
    if (
        payload["repository_root_device"] != root_identity["device"]
        or payload["repository_root_inode"] != root_identity["inode"]
    ):
        raise ValueError("implementation scope baseline repository root identity drifted")
    if payload["ordered_root_paths"] != ["."]:
        raise ValueError("implementation scope baseline root path order is invalid")
    for field in (
        "repository_root_device",
        "repository_root_inode",
        "check_output_directory_device",
        "check_output_directory_inode",
    ):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"implementation scope baseline {field} is invalid")
    check_output = Path(payload["check_output_directory_absolute_path"])
    _verify_absolute_no_symlink_path(check_output)
    entries = payload["ordered_entry_receipts"]
    executables = payload["ordered_repository_executable_receipts"]
    if not isinstance(entries, list) or not isinstance(executables, list) or not entries:
        raise ValueError("implementation scope baseline receipt arrays are invalid")
    entry_fields = {
        "path",
        "entry_kind",
        "mode_bits",
        "link_count",
        "size_bytes",
        "file_sha256",
        "symlink_target_text",
        "hardlink_group_sha256",
        "ordered_child_names",
    }
    entry_paths: set[str] = set()
    entry_casefold_paths: set[str] = set()
    previous_path: str | None = None
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != entry_fields:
            raise ValueError("implementation scope baseline entry receipt shape is invalid")
        path = entry["path"]
        if not isinstance(path, str) or "\\" in path or path == "" or PurePosixPath(path).is_absolute():
            raise ValueError("implementation scope baseline entry path is invalid")
        if any(part in {"", ".", ".."} for part in PurePosixPath(path).parts) and path != ".":
            raise ValueError("implementation scope baseline entry path escapes root")
        if path in entry_paths or path.casefold() in entry_casefold_paths:
            raise ValueError("implementation scope baseline entry paths are not unique")
        if previous_path is not None and path <= previous_path:
            raise ValueError("implementation scope baseline entry paths are not ordered")
        previous_path = path
        entry_paths.add(path)
        entry_casefold_paths.add(path.casefold())
        kind = entry["entry_kind"]
        if kind not in {"absent", "directory", "excluded_directory", "regular", "symlink"}:
            raise ValueError("implementation scope baseline entry kind is invalid")
        mode = entry["mode_bits"]
        if mode is not None and (not isinstance(mode, int) or isinstance(mode, bool) or mode < 0):
            raise ValueError("implementation scope baseline mode is invalid")
        if kind == "excluded_directory":
            if path not in {".git", ".venv"} or mode is None:
                raise ValueError("implementation scope baseline excluded directory is invalid")
        elif path in {".git", ".venv"}:
            raise ValueError("implementation scope baseline excluded directory kind is missing")
        if kind in {"absent", "directory", "excluded_directory", "symlink"}:
            if entry["link_count"] is not None or entry["size_bytes"] is not None or entry["file_sha256"] is not None:
                raise ValueError("implementation scope baseline conditional file fields are invalid")
            if entry["hardlink_group_sha256"] is not None:
                raise ValueError("implementation scope baseline hardlink field is invalid")
        if kind in {"absent", "excluded_directory", "symlink"} and entry["ordered_child_names"] is not None:
            raise ValueError("implementation scope baseline child names are invalid")
        if kind == "directory":
            children = entry["ordered_child_names"]
            if (
                not isinstance(children, list)
                or any(not isinstance(child, str) or not child or "/" in child or "\\" in child for child in children)
                or children != sorted(children)
                or len(children) != len(set(children))
            ):
                raise ValueError("implementation scope baseline directory children are invalid")
        if kind == "regular":
            if (
                mode is None
                or entry["link_count"] != 1
                or not isinstance(entry["size_bytes"], int)
                or isinstance(entry["size_bytes"], bool)
                or entry["size_bytes"] < 0
                or not is_sha256(entry["file_sha256"])
                or not is_sha256(entry["hardlink_group_sha256"])
                or entry["symlink_target_text"] is not None
                or entry["ordered_child_names"] is not None
            ):
                raise ValueError("implementation scope baseline regular entry is invalid")
        if kind == "symlink" and (
            not isinstance(entry["symlink_target_text"], str) or not entry["symlink_target_text"]
        ):
            raise ValueError("implementation scope baseline symlink entry is invalid")
    if entries[0]["path"] != "." or entries[0]["entry_kind"] != "directory":
        raise ValueError("implementation scope baseline root entry is invalid")
    executable_fields = {"path", "size_bytes", "file_sha256", "mode_bits", "link_count"}
    previous_path = None
    for receipt in executables:
        if not isinstance(receipt, Mapping) or set(receipt) != executable_fields:
            raise ValueError("implementation scope baseline executable receipt shape is invalid")
        path = receipt["path"]
        if (
            not isinstance(path, str)
            or path not in entry_paths
            or path <= (previous_path or "")
            or not isinstance(receipt["size_bytes"], int)
            or isinstance(receipt["size_bytes"], bool)
            or receipt["size_bytes"] < 0
            or not is_sha256(receipt["file_sha256"])
            or not isinstance(receipt["mode_bits"], int)
            or isinstance(receipt["mode_bits"], bool)
            or receipt["mode_bits"] < 0
            or receipt["link_count"] != 1
        ):
            raise ValueError("implementation scope baseline executable receipt is invalid")
        previous_path = path
    if not is_rfc3339(payload["captured_at_utc"]):
        raise ValueError("implementation scope baseline timestamp is invalid")


def _verify_implementation_approval_artifact(payload: Mapping[str, object]) -> tuple[Path, dict[str, int]]:
    if set(payload) != _IMPLEMENTATION_APPROVAL_FIELDS:
        raise ValueError("implementation approval field set is invalid")
    if (
        payload["schema_version"] != _IMPLEMENTATION_APPROVAL_SCHEMA
        or payload["module_id"] != "existing-45-temporal-retrospective"
    ):
        raise ValueError("implementation approval identity is invalid")
    root = Path(payload["repository_root_absolute_path"])
    _verify_absolute_no_symlink_path(root)
    root_device, root_inode = _verify_real_directory(root)
    if payload["repository_root_device"] != root_device or payload["repository_root_inode"] != root_inode:
        raise ValueError("implementation approval repository root identity drifted")
    for field in (
        "repository_root_device",
        "repository_root_inode",
    ):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"implementation approval {field} is invalid")
    verify_artifact_file_receipt(payload["parent_spec_approval_receipt"])
    verify_artifact_file_receipt(payload["amendment_fresh_review_receipt"])
    verify_artifact_file_receipt(payload["implementation_scope_baseline_receipt"])
    parent_receipts = payload["approved_parent_file_receipts"]
    if not isinstance(parent_receipts, list) or len(parent_receipts) != 3:
        raise ValueError("implementation approval parent file receipts are invalid")
    for receipt in parent_receipts:
        _verify_reviewed_file_receipt(receipt, basename_only=True)
    _verify_reviewed_file_receipt(payload["approved_amendment_file_receipt"])
    if payload["approval_scope"] != "amendment_implementation_only":
        raise ValueError("implementation approval scope is invalid")
    if payload["model_execution_authorized"] is not False or payload["module_b_authorized"] is not False:
        raise ValueError("implementation approval execution flags must be false")
    if not is_sha256(payload["approval_statement_sha256"]):
        raise ValueError("implementation approval statement hash is invalid")
    if not is_rfc3339(payload["approved_at_utc"]):
        raise ValueError("implementation approval timestamp is invalid")
    return root, {"device": root_device, "inode": root_inode}


def _verify_nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _verify_reviewed_file_rows(
    payload: Mapping[str, object],
    *,
    field: str,
    repository_root: Path,
    expected_paths: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], ...]:
    rows = payload[field]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{field} must be a non-empty list")
    normalized: list[dict[str, object]] = []
    for row in rows:
        _verify_reviewed_file_receipt(row)
        normalized.append(dict(row))
    paths = tuple(row["path"] for row in normalized)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise ValueError(f"{field} is not lexically ordered and unique")
    if expected_paths is not None and paths != expected_paths:
        raise ValueError(f"{field} does not use the frozen path set")
    for row in normalized:
        _verify_file_receipt_against_path(
            row,
            path=repository_root / row["path"],
            repository_root=repository_root,
            basename_only=False,
        )
    return tuple(normalized)


def _verify_amended_review_governance(value: object) -> dict[str, object]:
    expected = {
        "review_scope": "external_governance_only",
        "model_execution_scope": "review_reasoning_only",
        "local_module_a_worker_or_media_execution_performed": False,
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("fresh review governance observation is invalid")
    return dict(value)


def _verify_amended_review_scope_delta(
    value: object,
    *,
    code_rows: tuple[dict[str, object], ...],
    test_rows: tuple[dict[str, object], ...],
    baseline_entries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"ordered_leaf_rows", "ordered_directory_rows"}:
        raise ValueError("implementation scope delta shape is invalid")
    expected_rows = {row["path"]: row for row in code_rows + test_rows}
    leaf_rows = value["ordered_leaf_rows"]
    if not isinstance(leaf_rows, list) or len(leaf_rows) != len(expected_rows):
        raise ValueError("implementation scope delta leaf rows are invalid")
    seen: list[str] = []
    for row in leaf_rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "before_entry", "after_file", "change_kind"}:
            raise ValueError("implementation scope delta leaf row shape is invalid")
        path = row["path"]
        if path not in expected_rows or path in seen:
            raise ValueError("implementation scope delta leaf path is invalid")
        if not isinstance(row["before_entry"], Mapping):
            raise ValueError("implementation scope delta before entry is invalid")
        if path not in baseline_entries or dict(row["before_entry"]) != dict(baseline_entries[path]):
            raise ValueError("implementation scope delta before entry is not baseline-bound")
        _verify_reviewed_file_receipt(row["after_file"])
        if dict(row["after_file"]) != expected_rows[path]:
            raise ValueError("implementation scope delta after receipt is not current")
        if row["change_kind"] not in {"unchanged", "modified", "created"}:
            raise ValueError("implementation scope delta change kind is invalid")
        if path.endswith("task0258_module_a_verified_bootstrap.py"):
            if row["change_kind"] != "created":
                raise ValueError("bootstrap delta must be created")
        elif row["change_kind"] == "created":
            raise ValueError("only the bootstrap may be created")
        seen.append(path)
    if tuple(seen) != tuple(sorted(seen)) or set(seen) != set(expected_rows):
        raise ValueError("implementation scope delta leaf order is invalid")

    directory_rows = value["ordered_directory_rows"]
    if not isinstance(directory_rows, list) or len(directory_rows) != 1:
        raise ValueError("implementation scope delta directory rows are invalid")
    directory = directory_rows[0]
    expected_directory_fields = {
        "path",
        "before_child_names",
        "after_child_names",
        "authorized_added_children",
        "authorized_removed_children",
    }
    if not isinstance(directory, Mapping) or set(directory) != expected_directory_fields:
        raise ValueError("implementation scope delta directory row shape is invalid")
    if directory["path"] != "scripts":
        raise ValueError("implementation scope delta directory path is invalid")
    for field in (
        "before_child_names",
        "after_child_names",
        "authorized_added_children",
        "authorized_removed_children",
    ):
        names = directory[field]
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("implementation scope delta directory children are invalid")
        if names != sorted(names) or len(set(names)) != len(names):
            raise ValueError("implementation scope delta directory children are not ordered")
    if directory["authorized_added_children"] != ["task0258_module_a_verified_bootstrap.py"]:
        raise ValueError("implementation scope delta added children are invalid")
    if directory["authorized_removed_children"]:
        raise ValueError("implementation scope delta may not remove children")
    expected_after = sorted(set(directory["before_child_names"]) | {"task0258_module_a_verified_bootstrap.py"})
    if directory["after_child_names"] != expected_after:
        raise ValueError("implementation scope delta after children are invalid")
    return {"ordered_leaf_rows": [dict(row) for row in leaf_rows], "ordered_directory_rows": [dict(directory)]}


def _verify_amended_review_bootstrap(value: object, *, code_rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    expected_fields = {"protocol", "source_sha256", "runtime_snapshot_receipt"}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("bootstrap launcher receipt shape is invalid")
    if value["protocol"] != "task0258-module-a-verified-python-bootstrap-v2":
        raise ValueError("bootstrap launcher protocol is invalid")
    bootstrap_row = next(row for row in code_rows if row["path"].endswith("task0258_module_a_verified_bootstrap.py"))
    if value["source_sha256"] != bootstrap_row["file_sha256"]:
        raise ValueError("bootstrap launcher source hash is not bound")
    runtime = value["runtime_snapshot_receipt"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "contract_absolute_path",
        "artifact_sha256",
        "file_sha256",
        "postpublication_free_bytes",
    }:
        raise ValueError("runtime snapshot receipt shape is invalid")
    _verify_absolute_no_symlink_path(Path(runtime["contract_absolute_path"]))
    _verify_sha(runtime["artifact_sha256"], "runtime snapshot artifact hash")
    _verify_sha(runtime["file_sha256"], "runtime snapshot file hash")
    _verify_nonnegative_integer(runtime["postpublication_free_bytes"], "runtime snapshot free bytes")
    return {key: value[key] for key in expected_fields}


def _verify_namespace_query_receipt(value: object, *, repository_root: Path) -> dict[str, object]:
    fields = {
        "operation",
        "path",
        "arguments",
        "follow_policy",
        "result_kind",
        "errno",
        "stat_result",
        "access_result",
        "readlink_target_text",
        "ordered_directory_entries",
        "xattr_result",
        "file_content",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("namespace query receipt shape is invalid")
    operation = value["operation"]
    if operation not in {"access", "exec", "fstat", "getdents", "lstat", "mmap", "open", "readlink", "stat", "xattr"}:
        raise ValueError("namespace query operation is invalid")
    path = value["path"]
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts)
    ):
        raise ValueError("namespace query path is invalid")
    argument_shapes = {
        "access": {"mode", "follow_symlinks"},
        "stat": {"follow_symlinks"},
        "open": {"flags", "creation_mode"},
        "fstat": {"source_open_query_sha256"},
        "getdents": {"source_open_query_sha256"},
        "mmap": {"source_open_query_sha256", "offset", "length", "access"},
        "xattr": {"name", "options", "follow_symlinks"},
        "exec": set(),
        "lstat": set(),
        "readlink": set(),
    }
    if not isinstance(value["arguments"], Mapping) or set(value["arguments"]) != argument_shapes[operation]:
        raise ValueError("namespace query arguments are invalid")
    if value["follow_policy"] not in {"follow", "no_follow", "not_applicable"}:
        raise ValueError("namespace query follow policy is invalid")
    if value["result_kind"] not in {"success", "error"}:
        raise ValueError("namespace query result kind is invalid")
    if value["result_kind"] == "error":
        if value["errno"] not in {"EACCES", "ELOOP", "ENOENT", "ENOTDIR"}:
            raise ValueError("namespace query errno is invalid")
        if any(
            value[field] is not None
            for field in fields - {"operation", "path", "arguments", "follow_policy", "result_kind", "errno"}
        ):
            raise ValueError("namespace query error projection is not empty")
    else:
        if value["errno"] is not None:
            raise ValueError("namespace query success errno is invalid")
        stat_result = value["stat_result"]
        if stat_result is not None:
            stat_fields = {
                "entry_kind",
                "mode",
                "device",
                "inode",
                "nlink",
                "uid",
                "gid",
                "rdev",
                "size_bytes",
                "atime_ns",
                "mtime_ns",
                "ctime_ns",
                "birthtime_ns",
                "flags",
            }
            if not isinstance(stat_result, Mapping) or set(stat_result) != stat_fields:
                raise ValueError("namespace query stat projection is invalid")
            if stat_result["entry_kind"] not in {"directory", "regular", "symlink"}:
                raise ValueError("namespace query stat entry kind is invalid")
            for field in stat_fields - {"entry_kind"}:
                _verify_nonnegative_integer(stat_result[field], f"namespace query stat {field}")
        if value["access_result"] is not None and not isinstance(value["access_result"], bool):
            raise ValueError("namespace query access result is invalid")
        if value["readlink_target_text"] is not None and not isinstance(value["readlink_target_text"], str):
            raise ValueError("namespace query readlink result is invalid")
        if value["ordered_directory_entries"] is not None and not isinstance(value["ordered_directory_entries"], list):
            raise ValueError("namespace query directory projection is invalid")
        if value["xattr_result"] is not None and not isinstance(value["xattr_result"], Mapping):
            raise ValueError("namespace query xattr projection is invalid")
        file_content = value["file_content"]
        if file_content is not None:
            if not isinstance(file_content, Mapping) or set(file_content) != {"size_bytes", "file_sha256"}:
                raise ValueError("namespace query file projection is invalid")
            _verify_nonnegative_integer(file_content["size_bytes"], "namespace query file size")
            _verify_sha(file_content["file_sha256"], "namespace query file hash")
            actual = _read_no_follow_file_under_root(
                repository_root / path, allowed_root=repository_root, description="namespace query file"
            )
            if (
                len(actual) != file_content["size_bytes"]
                or hashlib.sha256(actual).hexdigest() != file_content["file_sha256"]
            ):
                raise ValueError("namespace query file projection does not match current bytes")
    return dict(value)


def _verify_amended_review_check_inputs(value: object, *, repository_root: Path) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("implementation review check-input sets are invalid")
    rows: list[dict[str, object]] = []
    expected_names = _AMENDED_REVIEW_CHECK_NAMES[:5]
    for expected_name, row in zip(expected_names, value, strict=True):
        if not isinstance(row, Mapping) or set(row) != {"check_name", "ordered_query_receipts", "projection_sha256"}:
            raise ValueError("implementation review check-input row shape is invalid")
        if row["check_name"] != expected_name or not isinstance(row["ordered_query_receipts"], list):
            raise ValueError("implementation review check-input name or rows are invalid")
        queries = [
            _verify_namespace_query_receipt(query, repository_root=repository_root)
            for query in row["ordered_query_receipts"]
        ]
        query_keys = [
            (
                query["operation"],
                query["path"],
                compact_canonical_json(query["arguments"]),
                query["follow_policy"],
            )
            for query in queries
        ]
        if query_keys != sorted(query_keys) or len(set(query_keys)) != len(query_keys):
            raise ValueError("implementation review check-input query order is invalid")
        if (
            not queries
            or hashlib.sha256(compact_canonical_json(queries).encode()).hexdigest() != row["projection_sha256"]
        ):
            raise ValueError("implementation review check-input projection is invalid")
        rows.append(
            {
                "check_name": row["check_name"],
                "ordered_query_receipts": queries,
                "projection_sha256": row["projection_sha256"],
            }
        )
    return tuple(rows)


def _verify_amended_review_check_rows(
    value: object,
    *,
    check_output_directory: Path,
    fresh_review_artifact_sha256: str,
    fresh_review_path: Path,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or len(value) != len(_AMENDED_REVIEW_CHECK_NAMES):
        raise ValueError("implementation review check receipts are invalid")
    rows: list[dict[str, object]] = []
    for expected_name, row in zip(_AMENDED_REVIEW_CHECK_NAMES, value, strict=True):
        if not isinstance(row, Mapping) or set(row) != _AMENDED_REVIEW_CHECK_FIELDS:
            raise ValueError("implementation review check receipt shape is invalid")
        if row["check_name"] != expected_name or row["execution_protocol"] != (
            "fresh-context-read-only-code-review-v1"
            if expected_name == "fresh_context_code_review"
            else "task0258-review-snapshot-fd-v2"
        ):
            raise ValueError("implementation review check receipt identity is invalid")
        if expected_name == "fresh_context_code_review":
            if (
                row["sandbox_attestation_sha256"] is not None
                or row["output_artifact_sha256"] != fresh_review_artifact_sha256
            ):
                raise ValueError("fresh review check receipt is not bound")
            if Path(row["output_path"]) != fresh_review_path:
                raise ValueError("fresh review output path is not bound")
        else:
            if not is_sha256(row["sandbox_attestation_sha256"]) or row["output_artifact_sha256"] is not None:
                raise ValueError("implementation check receipt attestation is invalid")
            if Path(row["output_path"]).name != f"{expected_name}.out":
                raise ValueError("implementation check output basename is invalid")
        if not is_sha256(row["command_sha256"]) or row["exit_code"] != 0 or not is_rfc3339(row["completed_at_utc"]):
            raise ValueError("implementation check receipt command or status is invalid")
        output_path = Path(row["output_path"])
        if not output_path.is_absolute() or output_path.parent != check_output_directory:
            raise ValueError("implementation check output path is outside the bound directory")
        _verify_absolute_no_symlink_path(output_path)
        raw = _read_no_follow_file_under_root(
            output_path, allowed_root=Path(output_path.anchor), description="implementation check output"
        )
        if len(raw) != row["output_size_bytes"] or hashlib.sha256(raw).hexdigest() != row["output_file_sha256"]:
            raise ValueError("implementation check output receipt does not match bytes")
        _verify_nonnegative_integer(row["output_size_bytes"], "implementation check output size")
        rows.append(dict(row))
    return tuple(rows)


def _verify_amended_review_resource_summary(
    value: object, *, fresh_review_size: int, governance: Mapping[str, object]
) -> None:
    fields = {"runtime_build_observation", "ordered_check_observations", "fresh_review_observation"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("implementation review resource summary shape is invalid")
    runtime = value["runtime_build_observation"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "build_resource_limits",
        "build_resource_observation",
        "postpublication_free_bytes",
    }:
        raise ValueError("implementation review runtime resource summary is invalid")
    _verify_nonnegative_integer(runtime["postpublication_free_bytes"], "implementation review free bytes")
    observations = value["ordered_check_observations"]
    if not isinstance(observations, list) or len(observations) != 5:
        raise ValueError("implementation review check resource summary is invalid")
    for expected_name, observation in zip(_AMENDED_REVIEW_CHECK_NAMES[:5], observations, strict=True):
        if not isinstance(observation, Mapping) or set(observation) != {
            "check_name",
            "process_resource_observation",
            "output_publication_observation",
        }:
            raise ValueError("implementation review check observation shape is invalid")
        if observation["check_name"] != expected_name:
            raise ValueError("implementation review check observation order is invalid")
        if not isinstance(observation["process_resource_observation"], Mapping) or not isinstance(
            observation["output_publication_observation"], Mapping
        ):
            raise ValueError("implementation review check observation values are invalid")
    fresh = value["fresh_review_observation"]
    if not isinstance(fresh, Mapping) or set(fresh) != {"governance_observation", "fresh_review_artifact_bytes"}:
        raise ValueError("implementation review fresh resource summary is invalid")
    if (
        dict(fresh["governance_observation"]) != dict(governance)
        or fresh["fresh_review_artifact_bytes"] != fresh_review_size
    ):
        raise ValueError("implementation review fresh resource summary is not bound")


def _verify_amended_review_payload(
    payload: Mapping[str, object],
    *,
    repository_root: Path,
    root_identity: Mapping[str, int],
    implementation_approval: VerifiedReviewImplementationApproval,
    check_output_directory: Path,
    fresh_review_artifact_sha256: str,
    fresh_review_path: Path,
    fresh_review_size: int,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], dict[str, object]]:
    if set(payload) != _AMENDED_IMPLEMENTATION_REVIEW_FIELDS:
        raise ValueError("amended implementation review field set is invalid")
    if (
        payload["schema_version"] != _AMENDED_IMPLEMENTATION_REVIEW_SCHEMA
        or payload["module_id"] != "existing-45-temporal-retrospective"
    ):
        raise ValueError("amended implementation review identity is invalid")
    payload_root = Path(payload["repository_root_absolute_path"])
    _verify_absolute_no_symlink_path(payload_root)
    if payload_root != repository_root:
        raise ValueError("amended implementation review repository root path is not bound")
    if (
        payload["repository_root_device"] != root_identity["device"]
        or payload["repository_root_inode"] != root_identity["inode"]
    ):
        raise ValueError("amended implementation review repository root identity drifted")
    for field in ("repository_root_device", "repository_root_inode"):
        _verify_nonnegative_integer(payload[field], f"amended implementation review {field}")
    if payload["amendment_implementation_approval_receipt"] != _artifact_file_receipt(implementation_approval.artifact):
        raise ValueError("amended implementation review approval receipt is not bound")
    implementation_context_id = payload["implementation_context_id"]
    reviewer_context_id = payload["reviewer_context_id"]
    if (
        not is_safe_slug(implementation_context_id)
        or not is_safe_slug(reviewer_context_id)
        or implementation_context_id == reviewer_context_id
    ):
        raise ValueError("amended implementation review contexts are invalid")
    if payload["reviewer_independence"] != "different_fresh_context":
        raise ValueError("amended implementation review independence is invalid")
    code_rows = _verify_reviewed_file_rows(
        payload,
        field="ordered_code_file_receipts",
        repository_root=repository_root,
        expected_paths=_AMENDED_REVIEW_CODE_PATHS,
    )
    test_rows = _verify_reviewed_file_rows(
        payload,
        field="ordered_test_file_receipts",
        repository_root=repository_root,
        expected_paths=_AMENDED_REVIEW_TEST_PATHS,
    )
    _verify_reviewed_file_rows(
        payload, field="ordered_runtime_dependency_file_receipts", repository_root=repository_root
    )
    _verify_reviewed_file_rows(
        payload,
        field="ordered_check_configuration_file_receipts",
        repository_root=repository_root,
        expected_paths=("pyproject.toml", "pytest.ini"),
    )
    _verify_amended_review_check_inputs(payload["ordered_check_input_receipt_sets"], repository_root=repository_root)
    baseline_receipt = implementation_approval.artifact.payload["implementation_scope_baseline_receipt"]
    if payload["implementation_scope_baseline_receipt"] != baseline_receipt:
        raise ValueError("amended implementation review baseline receipt is not bound")
    delta = _verify_amended_review_scope_delta(
        payload["implementation_scope_delta"],
        code_rows=code_rows,
        test_rows=test_rows,
        baseline_entries={
            entry["path"]: entry
            for entry in implementation_approval.implementation_scope_baseline.payload["ordered_entry_receipts"]
            if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
        },
    )
    _verify_amended_review_bootstrap(payload["bootstrap_launcher_receipt"], code_rows=code_rows)
    _verify_amended_review_check_rows(
        payload["ordered_check_receipts"],
        check_output_directory=check_output_directory,
        fresh_review_artifact_sha256=fresh_review_artifact_sha256,
        fresh_review_path=fresh_review_path,
    )
    if (
        payload["critical_count"] != 0
        or payload["required_count"] != 0
        or not isinstance(payload["optional_count"], int)
        or isinstance(payload["optional_count"], bool)
        or payload["optional_count"] < 0
    ):
        raise ValueError("amended implementation review counts are invalid")
    if payload["heavy_execution_performed"] is not False or not is_rfc3339(payload["reviewed_at_utc"]):
        raise ValueError("amended implementation review status is invalid")
    _verify_amended_review_resource_summary(
        payload["review_resource_summary"],
        fresh_review_size=fresh_review_size,
        governance=_verify_amended_review_governance(
            payload["review_resource_summary"]["fresh_review_observation"]["governance_observation"]
        ),
    )
    return code_rows, test_rows, delta


def _verify_fresh_amended_review_payload(
    payload: Mapping[str, object],
    *,
    repository_root: Path,
    root_identity: Mapping[str, int],
    implementation_approval: VerifiedReviewImplementationApproval,
    implementation_payload: Mapping[str, object],
) -> None:
    if set(payload) != _FRESH_IMPLEMENTATION_REVIEW_FIELDS:
        raise ValueError("fresh amended implementation review field set is invalid")
    if (
        payload["schema_version"] != _AMENDED_IMPLEMENTATION_REVIEW_SCHEMA
        or payload["module_id"] != "existing-45-temporal-retrospective"
    ):
        raise ValueError("fresh amended implementation review identity is invalid")
    payload_root = Path(payload["repository_root_absolute_path"])
    _verify_absolute_no_symlink_path(payload_root)
    if (
        payload_root != repository_root
        or payload["repository_root_device"] != root_identity["device"]
        or payload["repository_root_inode"] != root_identity["inode"]
    ):
        raise ValueError("fresh amended implementation review root identity is invalid")
    if payload["amendment_implementation_approval_receipt"] != _artifact_file_receipt(implementation_approval.artifact):
        raise ValueError("fresh amended implementation review approval receipt is not bound")
    if not is_safe_slug(payload["implementation_context_id"]) or not is_safe_slug(payload["reviewer_context_id"]):
        raise ValueError("fresh amended implementation review context is invalid")
    if (
        payload["implementation_context_id"] == payload["reviewer_context_id"]
        or payload["reviewer_independence"] != "different_fresh_context"
    ):
        raise ValueError("fresh amended implementation review independence is invalid")
    for field in (
        "ordered_code_file_receipts",
        "ordered_test_file_receipts",
        "ordered_runtime_dependency_file_receipts",
        "ordered_check_configuration_file_receipts",
        "ordered_check_input_receipt_sets",
        "implementation_scope_baseline_receipt",
        "implementation_scope_delta",
        "bootstrap_launcher_receipt",
    ):
        if payload[field] != implementation_payload[field]:
            raise ValueError(f"fresh amended implementation review {field} is not bound")
    if payload["ordered_pre_review_check_receipts"] != implementation_payload["ordered_check_receipts"][:5]:
        raise ValueError("fresh amended implementation review pre-review checks are not bound")
    if _verify_amended_review_governance(payload["fresh_review_governance_observation"])[
        "local_module_a_worker_or_media_execution_performed"
    ]:
        raise ValueError("fresh amended implementation review claims local execution")
    if (
        payload["critical_count"] != 0
        or payload["required_count"] != 0
        or not isinstance(payload["optional_count"], int)
        or isinstance(payload["optional_count"], bool)
        or payload["optional_count"] < 0
    ):
        raise ValueError("fresh amended implementation review counts are invalid")
    if payload["heavy_execution_performed"] is not False or not is_rfc3339(payload["reviewed_at_utc"]):
        raise ValueError("fresh amended implementation review status is invalid")


def load_verified_amended_implementation_review(
    *,
    execution_context: object,
    repository_root: Path,
    implementation_approval: VerifiedReviewImplementationApproval,
    implementation_review_path: Path,
    expected_implementation_review_artifact_sha256: str,
    expected_implementation_review_file_sha256: str,
    fresh_review_path: Path,
    expected_fresh_review_artifact_sha256: str,
    expected_fresh_review_file_sha256: str,
) -> VerifiedReviewAmendedImplementationReview:
    """Replay the amended implementation review and its fresh reviewer receipt.

    This is intentionally review-only.  It reopens the canonical review and
    fresh-review bytes, checks the exact closed receipt shapes and current code,
    test, runtime, configuration, check-output, and check-input bindings, and
    returns no production authority.
    """
    if (
        type(execution_context) is not VerifiedImplementationReviewSandboxContext
        or execution_context._token is not _SANDBOX_TOKEN
    ):
        raise PermissionError("amended implementation review loader requires a verified review context")
    if type(implementation_approval) is not VerifiedReviewImplementationApproval:
        raise PermissionError("amended implementation review requires a verified implementation approval")
    repository_root = Path(repository_root)
    _verify_absolute_no_symlink_path(repository_root)
    root_device, root_inode = _verify_real_directory(repository_root)
    root_identity = {"device": root_device, "inode": root_inode}
    baseline = implementation_approval.implementation_scope_baseline
    if not isinstance(baseline, VerifiedJsonArtifact):
        raise PermissionError("implementation approval baseline is not loaded")
    baseline_payload = baseline.payload
    check_output_directory = Path(baseline_payload["check_output_directory_absolute_path"])
    _verify_absolute_no_symlink_path(check_output_directory)
    try:
        check_output_directory.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise ValueError("implementation review check-output directory must be outside the repository")
    check_output_identity = _verify_real_directory(check_output_directory)
    if check_output_identity != (
        baseline_payload["check_output_directory_device"],
        baseline_payload["check_output_directory_inode"],
    ):
        raise ValueError("implementation review check-output directory identity drifted")
    review_artifact = _load_verified_json_artifact_under_root(
        path=Path(implementation_review_path),
        repository_root=Path(Path(implementation_review_path).anchor),
        expected_artifact_sha256=expected_implementation_review_artifact_sha256,
        expected_file_sha256=expected_implementation_review_file_sha256,
        description="amended implementation review",
    )
    fresh_artifact = _load_verified_json_artifact_under_root(
        path=Path(fresh_review_path),
        repository_root=Path(Path(fresh_review_path).anchor),
        expected_artifact_sha256=expected_fresh_review_artifact_sha256,
        expected_file_sha256=expected_fresh_review_file_sha256,
        description="fresh amended implementation review",
    )
    _verify_amended_review_payload(
        review_artifact.payload,
        repository_root=repository_root,
        root_identity=root_identity,
        implementation_approval=implementation_approval,
        check_output_directory=check_output_directory,
        fresh_review_artifact_sha256=fresh_artifact.artifact_sha256,
        fresh_review_path=Path(fresh_review_path),
        fresh_review_size=len(
            _read_no_follow_file_under_root(
                Path(fresh_review_path),
                allowed_root=Path(Path(fresh_review_path).anchor),
                description="fresh amended implementation review",
            )
        ),
    )
    _verify_fresh_amended_review_payload(
        fresh_artifact.payload,
        repository_root=repository_root,
        root_identity=root_identity,
        implementation_approval=implementation_approval,
        implementation_payload=review_artifact.payload,
    )
    if review_artifact.payload["ordered_check_receipts"][-1]["output_file_sha256"] != fresh_artifact.file_sha256:
        raise ValueError("fresh amended implementation review file receipt is not bound")
    return VerifiedReviewAmendedImplementationReview(
        _AMENDED_IMPLEMENTATION_REVIEW_TOKEN,
        artifact=review_artifact,
        fresh_review=fresh_artifact,
        implementation_approval=implementation_approval,
        repository_root_identity=root_identity,
    )


def _verify_rerun_authorization_path_boundary(
    *,
    path: Path,
    repository_root: Path,
    output_root: Path,
    registry_directory: Path,
    description: str,
    must_be_absent: bool,
) -> None:
    _verify_absolute_no_symlink_path(path)
    if must_be_absent and path.exists():
        raise ValueError(f"{description} must be absent")
    try:
        path.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise ValueError(f"{description} must be outside the repository")
    try:
        path.relative_to(output_root)
    except ValueError:
        pass
    else:
        raise ValueError(f"{description} must be outside the output root")
    try:
        path.relative_to(registry_directory)
    except ValueError:
        pass
    else:
        raise ValueError(f"{description} must be outside the consumption registry")


def _verify_rerun_authorization_payload(
    payload: Mapping[str, object],
    *,
    repository_root: Path,
    root_identity: Mapping[str, int],
    implementation_approval: VerifiedReviewImplementationApproval,
    implementation_review: VerifiedReviewAmendedImplementationReview,
    expected_static_input_contract: Mapping[str, object],
    output_root: Path,
    candidate_bundle_path: Path,
) -> None:
    if set(payload) != _RERUN_AUTHORIZATION_FIELDS:
        raise ValueError("rerun authorization field set is invalid")
    if (
        payload["schema_version"] != _RERUN_AUTHORIZATION_SCHEMA
        or payload["module_id"] != "existing-45-temporal-retrospective"
    ):
        raise ValueError("rerun authorization identity is invalid")
    payload_root = Path(payload["repository_root_absolute_path"])
    _verify_absolute_no_symlink_path(payload_root)
    if payload_root != repository_root:
        raise ValueError("rerun authorization repository root path is not bound")
    if (
        payload["repository_root_device"] != root_identity["device"]
        or payload["repository_root_inode"] != root_identity["inode"]
    ):
        raise ValueError("rerun authorization repository root identity drifted")
    for field in ("repository_root_device", "repository_root_inode"):
        _verify_nonnegative_integer(payload[field], f"rerun authorization {field}")

    approval_receipt = _artifact_file_receipt(implementation_approval.artifact)
    review_receipt = _artifact_file_receipt(implementation_review.artifact)
    if payload["amendment_implementation_approval_receipt"] != approval_receipt:
        raise ValueError("rerun authorization implementation approval is not bound")
    if payload["implementation_review_receipt"] != review_receipt:
        raise ValueError("rerun authorization implementation review is not bound")
    if (
        payload["approved_amendment_file_receipt"]
        != implementation_approval.artifact.payload["approved_amendment_file_receipt"]
    ):
        raise ValueError("rerun authorization amendment receipt is not bound")
    review_payload = implementation_review.artifact.payload
    equality_fields = (
        ("approved_code_receipts", "ordered_code_file_receipts"),
        ("approved_test_receipts", "ordered_test_file_receipts"),
        ("approved_runtime_dependency_receipts", "ordered_runtime_dependency_file_receipts"),
        ("approved_check_configuration_receipts", "ordered_check_configuration_file_receipts"),
        ("approved_check_input_receipt_sets", "ordered_check_input_receipt_sets"),
        ("approved_check_receipts", "ordered_check_receipts"),
        ("approved_implementation_scope_baseline_receipt", "implementation_scope_baseline_receipt"),
        ("approved_implementation_scope_delta", "implementation_scope_delta"),
        ("approved_bootstrap_launcher_receipt", "bootstrap_launcher_receipt"),
    )
    for authorization_field, review_field in equality_fields:
        if payload[authorization_field] != review_payload[review_field]:
            raise ValueError(f"rerun authorization {authorization_field} is not review-bound")
    if (
        payload["approved_implementation_scope_baseline_receipt"]
        != implementation_approval.artifact.payload["implementation_scope_baseline_receipt"]
    ):
        raise ValueError("rerun authorization baseline is not approval-bound")
    static_fields = {
        "temporal_plan_artifact_sha256",
        "temporal_plan_file_sha256",
        "task0257_receipts_projection_sha256",
    }
    if set(payload["approved_static_input_contract"]) != static_fields or dict(
        payload["approved_static_input_contract"]
    ) != dict(expected_static_input_contract):
        raise ValueError("rerun authorization static-input contract is invalid")
    for field in static_fields:
        _verify_sha(payload["approved_static_input_contract"][field], f"rerun authorization {field}")
    if tuple(payload["allowed_operations"]) != _RERUN_ALLOWED_OPERATIONS:
        raise ValueError("rerun authorization operation set is invalid")

    payload_output_root = Path(payload["output_root_absolute_path"])
    payload_bundle_path = Path(payload["candidate_receipt_bundle_absolute_path"])
    if payload_output_root != output_root or payload_bundle_path != candidate_bundle_path:
        raise ValueError("rerun authorization output paths are not caller-bound")
    if payload_output_root.name != "vru_causal_temporal_retrospective_v2":
        raise ValueError("rerun authorization output root basename is invalid")
    _verify_absolute_no_symlink_path(payload_output_root)
    if payload_output_root.exists():
        raise ValueError("rerun authorization output root must be absent")
    _verify_real_directory(payload_output_root.parent)

    registry_directory = Path(payload["run_consumption_registry_directory_absolute_path"])
    _verify_absolute_no_symlink_path(registry_directory)
    registry_device, registry_inode = _verify_real_directory(registry_directory)
    if payload["run_consumption_registry_directory_identity"] != {
        "device": registry_device,
        "inode": registry_inode,
    }:
        raise ValueError("rerun authorization registry identity drifted")
    for field in ("device", "inode"):
        _verify_nonnegative_integer(
            payload["run_consumption_registry_directory_identity"][field], f"rerun authorization registry {field}"
        )
    try:
        registry_directory.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise ValueError("rerun authorization registry must be outside the repository")
    try:
        registry_directory.relative_to(payload_output_root)
    except ValueError:
        pass
    else:
        raise ValueError("rerun authorization registry must be outside the output root")
    _verify_rerun_authorization_path_boundary(
        path=payload_bundle_path,
        repository_root=repository_root,
        output_root=payload_output_root,
        registry_directory=registry_directory,
        description="rerun authorization candidate bundle",
        must_be_absent=True,
    )
    if not _verify_real_directory(payload_bundle_path.parent):
        raise ValueError("rerun authorization candidate bundle parent is invalid")
    if not is_safe_slug(payload["run_id"]):
        raise ValueError("rerun authorization run id is invalid")
    _verify_nonnegative_integer(payload["maximum_run_count"], "rerun authorization maximum run count")
    if payload["authorization_scope"] != "one_module_a_v2_rerun" or payload["maximum_run_count"] != 1:
        raise ValueError("rerun authorization scope or run count is invalid")
    if payload["run_admission_relative_path"] != "run_admission.json":
        raise ValueError("rerun authorization admission path is invalid")
    if payload["module_b_authorized"] is not False:
        raise ValueError("rerun authorization must not authorize Module B")
    if not is_sha256(payload["approval_statement_sha256"]) or not is_rfc3339(payload["approved_at_utc"]):
        raise ValueError("rerun authorization approval metadata is invalid")


def load_verified_review_rerun_authorization(
    *,
    execution_context: object,
    repository_root: Path,
    authorization_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    implementation_approval: VerifiedReviewImplementationApproval,
    implementation_review: VerifiedReviewAmendedImplementationReview,
    expected_static_input_contract: Mapping[str, object],
    output_root: Path,
    candidate_bundle_path: Path,
) -> VerifiedReviewRerunAuthorization:
    """Replay a rerun authorization without issuing production authority.

    The production API will require a distinct externally authenticated
    bootstrap context.  This local loader intentionally accepts only the
    review sandbox context and returns an opaque diagnostic object so the
    authorization graph can be tested before that platform boundary exists.
    """
    if (
        type(execution_context) is not VerifiedImplementationReviewSandboxContext
        or execution_context._token is not _SANDBOX_TOKEN
    ):
        raise PermissionError("rerun authorization loader requires a verified review context")
    if (
        type(implementation_approval) is not VerifiedReviewImplementationApproval
        or implementation_approval.production_capability
    ):
        raise PermissionError("rerun authorization requires a review-only implementation approval")
    if (
        type(implementation_review) is not VerifiedReviewAmendedImplementationReview
        or implementation_review.production_capability
    ):
        raise PermissionError("rerun authorization requires a review-only implementation review")
    repository_root = Path(repository_root)
    _verify_absolute_no_symlink_path(repository_root)
    root_device, root_inode = _verify_real_directory(repository_root)
    root_identity = {"device": root_device, "inode": root_inode}
    artifact = _load_verified_json_artifact_under_root(
        path=Path(authorization_path),
        repository_root=Path(Path(authorization_path).anchor),
        expected_artifact_sha256=expected_artifact_sha256,
        expected_file_sha256=expected_file_sha256,
        description="review rerun authorization",
    )
    current_review = _load_verified_json_artifact_under_root(
        path=implementation_review.artifact.path,
        repository_root=Path(Path(implementation_review.artifact.path).anchor),
        expected_artifact_sha256=implementation_review.artifact.artifact_sha256,
        expected_file_sha256=implementation_review.artifact.file_sha256,
        description="bound amended implementation review",
    )
    current_fresh_review = _load_verified_json_artifact_under_root(
        path=implementation_review.fresh_review.path,
        repository_root=Path(Path(implementation_review.fresh_review.path).anchor),
        expected_artifact_sha256=implementation_review.fresh_review.artifact_sha256,
        expected_file_sha256=implementation_review.fresh_review.file_sha256,
        description="bound fresh amended implementation review",
    )
    if (
        current_review.payload != implementation_review.artifact.payload
        or current_fresh_review.payload != implementation_review.fresh_review.payload
    ):
        raise ValueError("bound implementation review bytes changed")
    baseline_payload = implementation_approval.implementation_scope_baseline.payload
    check_output_directory = Path(baseline_payload["check_output_directory_absolute_path"])
    _verify_amended_review_payload(
        current_review.payload,
        repository_root=repository_root,
        root_identity=root_identity,
        implementation_approval=implementation_approval,
        check_output_directory=check_output_directory,
        fresh_review_artifact_sha256=current_fresh_review.artifact_sha256,
        fresh_review_path=current_fresh_review.path,
        fresh_review_size=len(
            _read_no_follow_file_under_root(
                current_fresh_review.path,
                allowed_root=Path(Path(current_fresh_review.path).anchor),
                description="bound fresh amended implementation review",
            )
        ),
    )
    _verify_fresh_amended_review_payload(
        current_fresh_review.payload,
        repository_root=repository_root,
        root_identity=root_identity,
        implementation_approval=implementation_approval,
        implementation_payload=current_review.payload,
    )
    _verify_rerun_authorization_payload(
        artifact.payload,
        repository_root=repository_root,
        root_identity=root_identity,
        implementation_approval=implementation_approval,
        implementation_review=implementation_review,
        expected_static_input_contract=expected_static_input_contract,
        output_root=Path(output_root),
        candidate_bundle_path=Path(candidate_bundle_path),
    )
    return VerifiedReviewRerunAuthorization(
        _RERUN_AUTHORIZATION_TOKEN,
        artifact=artifact,
        implementation_approval=implementation_approval,
        implementation_review=implementation_review,
        repository_root_identity=root_identity,
        output_root=Path(output_root),
        candidate_bundle_path=Path(candidate_bundle_path),
    )


def load_verified_parent_module_a_spec_approval(
    *,
    execution_context: object,
    approval_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    approved_spec_paths: Mapping[str, Path],
    expected_fresh_review_internal_sha256: str,
    expected_fresh_review_file_sha256: str,
    expected_approval_statement_sha256: str,
) -> VerifiedReviewParentModuleASpecApproval:
    """Replay the parent specification approval and all three approved files."""
    if (
        type(execution_context) is not VerifiedImplementationReviewSandboxContext
        or execution_context._token is not _SANDBOX_TOKEN
    ):
        raise PermissionError("parent spec approval loader requires a verified review context")
    if tuple(approved_spec_paths) != ("requirement.md", "solution.md", "gate-review.md"):
        raise ValueError("parent spec approval paths must use the frozen order")
    artifact = _load_verified_json_artifact_under_root(
        path=approval_path,
        repository_root=Path(Path(approval_path).anchor),
        expected_artifact_sha256=expected_artifact_sha256,
        expected_file_sha256=expected_file_sha256,
        description="parent spec approval",
    )
    approved_files = _verify_parent_spec_approval_artifact(
        artifact.payload,
        approved_spec_paths=approved_spec_paths,
        expected_fresh_review_internal_sha256=expected_fresh_review_internal_sha256,
        expected_fresh_review_file_sha256=expected_fresh_review_file_sha256,
        expected_approval_statement_sha256=expected_approval_statement_sha256,
    )
    return VerifiedReviewParentModuleASpecApproval(
        _PARENT_SPEC_APPROVAL_TOKEN,
        artifact=artifact,
        approved_files=approved_files,
    )


def load_verified_amendment_implementation_approval(
    *,
    execution_context: object,
    repository_root: Path,
    approval_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    parent_approval: VerifiedReviewParentModuleASpecApproval,
    amendment_path: Path,
    expected_amendment_file_sha256: str,
    amendment_review_path: Path,
    expected_amendment_review_artifact_sha256: str,
    expected_amendment_review_file_sha256: str,
    implementation_scope_baseline_path: Path,
    expected_implementation_scope_baseline_artifact_sha256: str,
    expected_implementation_scope_baseline_file_sha256: str,
) -> VerifiedReviewImplementationApproval:
    """Replay every local receipt edge of the amendment implementation approval.

    The returned object remains review-only.  This function does not issue a
    rerun authorization, admit a producer, or provide a production capability.
    """
    if (
        type(execution_context) is not VerifiedImplementationReviewSandboxContext
        or execution_context._token is not _SANDBOX_TOKEN
    ):
        raise PermissionError("implementation approval loader requires a verified review context")
    if type(parent_approval) is not VerifiedReviewParentModuleASpecApproval:
        raise PermissionError("implementation approval requires a verified parent spec approval")
    repository_root = Path(repository_root)
    _verify_absolute_no_symlink_path(repository_root)
    root_device, root_inode = _verify_real_directory(repository_root)
    root_identity = {"device": root_device, "inode": root_inode}
    artifact = _load_verified_json_artifact_under_root(
        path=approval_path,
        repository_root=Path(Path(approval_path).anchor),
        expected_artifact_sha256=expected_artifact_sha256,
        expected_file_sha256=expected_file_sha256,
        description="implementation approval",
    )
    root, payload_identity = _verify_implementation_approval_artifact(artifact.payload)
    if root != repository_root or payload_identity != root_identity:
        raise ValueError("implementation approval repository root does not match the bound root")
    payload = artifact.payload
    parent_receipt = payload["parent_spec_approval_receipt"]
    if parent_receipt != {
        "artifact_sha256": parent_approval.artifact.artifact_sha256,
        "file_sha256": parent_approval.artifact.file_sha256,
    }:
        raise ValueError("implementation approval parent receipt is not bound")
    if tuple(payload["approved_parent_file_receipts"]) != parent_approval.approved_files:
        raise ValueError("implementation approval parent file receipts are not bound")

    amendment_receipt = payload["approved_amendment_file_receipt"]
    amendment_path = Path(amendment_path)
    amendment_raw = _verify_file_receipt_against_path(
        amendment_receipt, path=amendment_path, repository_root=repository_root, basename_only=False
    )
    if hashlib.sha256(amendment_raw).hexdigest() != _verify_sha(expected_amendment_file_sha256, "amendment file hash"):
        raise ValueError("amendment expected file hash does not match")

    review_raw = _read_no_follow_file_under_root(
        Path(amendment_review_path), allowed_root=repository_root, description="amendment fresh review"
    )
    review_file_sha256 = hashlib.sha256(review_raw).hexdigest()
    if review_file_sha256 != _verify_sha(expected_amendment_review_file_sha256, "amendment review file hash"):
        raise ValueError("amendment fresh review file hash does not match")
    review_artifact_sha256 = _verify_sha(expected_amendment_review_artifact_sha256, "amendment review artifact hash")
    if payload["amendment_fresh_review_receipt"] != {
        "artifact_sha256": review_artifact_sha256,
        "file_sha256": review_file_sha256,
    }:
        raise ValueError("amendment fresh review receipt hashes are invalid")
    review_text = review_raw.decode("utf-8")
    if not re.search(r"^reviewer_context:\s+independent-fresh-context$", review_text, re.MULTILINE):
        raise ValueError("amendment fresh review context is not independent")
    reviewed_hash = re.search(r"^reviewed_file_sha256:\s+([0-9a-f]{64})$", review_text, re.MULTILINE)
    if reviewed_hash is None or reviewed_hash.group(1) != hashlib.sha256(amendment_raw).hexdigest():
        raise ValueError("amendment fresh review does not cover the amendment bytes")
    if not re.search(r"^critical_count:\s+0$", review_text, re.MULTILINE) or not re.search(
        r"^required_count:\s+0$", review_text, re.MULTILINE
    ):
        raise ValueError("amendment fresh review is not Critical/Required 0/0")

    baseline = _load_verified_json_artifact_under_root(
        path=implementation_scope_baseline_path,
        repository_root=Path(Path(implementation_scope_baseline_path).anchor),
        expected_artifact_sha256=expected_implementation_scope_baseline_artifact_sha256,
        expected_file_sha256=expected_implementation_scope_baseline_file_sha256,
        description="implementation scope baseline",
    )
    _verify_implementation_scope_baseline_artifact(
        baseline.payload, repository_root=repository_root, root_identity=root_identity
    )
    if payload["implementation_scope_baseline_receipt"] != {
        "artifact_sha256": baseline.artifact_sha256,
        "file_sha256": baseline.file_sha256,
    }:
        raise ValueError("implementation approval baseline receipt is not bound")
    return VerifiedReviewImplementationApproval(
        _IMPLEMENTATION_APPROVAL_TOKEN,
        artifact=artifact,
        parent_approval=parent_approval,
        amendment=amendment_raw,
        amendment_review=review_raw,
        implementation_scope_baseline=baseline,
        repository_root_identity=root_identity,
    )


load_verified_review_implementation_approval = load_verified_amendment_implementation_approval


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


def bind_verified_review_no_write_preflight(
    *,
    execution_context: object,
    attempt_spine: VerifiedReviewVerificationAttemptRunSpine,
    candidate_bundle_path: Path,
) -> VerifiedReviewNoWritePreflight:
    """Replay the local no-write preflight for one verified review spine.

    The checks intentionally stop before lock creation, directory creation,
    model/device access, or output publication.  The returned object is a
    diagnostic record only; an external runner must still bind kernel audit,
    retained runtime, and exact rerun authorization before production use.
    """
    if type(execution_context) is not VerifiedImplementationReviewSandboxContext:
        raise PermissionError("no-write preflight requires a verified review context")
    if (
        type(attempt_spine) is not VerifiedReviewVerificationAttemptRunSpine
        or attempt_spine._token is not _ATTEMPT_SPINE_TOKEN
    ):
        raise TypeError("attempt spine is not a verified review object")

    admission = attempt_spine.run_admission
    history = attempt_spine.run_history
    if type(admission) is not VerifiedReviewRunAdmission or admission._token is not _RUN_ADMISSION_TOKEN:
        raise TypeError("no-write preflight admission is invalid")
    if type(history) is not VerifiedRunHistoryLedger or history._token is not _RUN_HISTORY_TOKEN:
        raise TypeError("no-write preflight history is invalid")

    output_root = Path(admission.admission.payload["output_root_absolute_path"])
    output_identity = _verify_review_temp_directory(output_root)
    expected_root_identity = {
        "device": output_identity[0],
        "inode": output_identity[1],
    }
    if expected_root_identity != admission.root_identity:
        raise ValueError("no-write preflight output-root identity drifted")
    if admission.completion.payload["root_identity"] != expected_root_identity:
        raise ValueError("no-write preflight completion root identity drifted")

    reserved_output_names = {
        "candidate_v2",
        "terminal_failure_v2",
        "verified_result_v2",
        "postverification_failure_v2",
    }
    for entry in output_root.iterdir():
        if entry.name.startswith(".") or entry.name in reserved_output_names:
            raise ValueError("no-write preflight found reserved output residue")

    registry = Path(history.directory)
    registry_identity = _verify_review_temp_directory(registry)
    if registry == output_root or registry in output_root.parents or output_root in registry.parents:
        raise ValueError("no-write preflight registry and output root must be distinct")
    expected_registry_names = registry_history_filenames(registry, history.authorization_sha256)
    actual_registry_entries = list(registry.iterdir())
    if {entry.name for entry in actual_registry_entries} != set(expected_registry_names):
        raise ValueError("no-write preflight registry contains unstable residue")
    if any(entry.is_symlink() or not entry.is_file() for entry in actual_registry_entries):
        raise ValueError("no-write preflight registry contains a non-regular member")

    bundle_path = Path(candidate_bundle_path)
    _verify_absolute_no_symlink_path(bundle_path)
    bundle_parent = bundle_path.parent
    _verify_review_temp_directory(bundle_parent)
    if bundle_path.exists() or bundle_path.is_symlink():
        raise ValueError("no-write preflight candidate bundle target is not absent")
    if bundle_parent == output_root or output_root in bundle_parent.parents:
        raise ValueError("no-write preflight candidate bundle is inside the output root")
    if bundle_parent == registry or registry in bundle_parent.parents:
        raise ValueError("no-write preflight candidate bundle is inside the registry")

    planned_paths = (
        output_root / "candidate_v2",
        bundle_path,
        output_root / "verified_result_v2",
        output_root / "postverification_failure_v2",
    )
    return VerifiedReviewNoWritePreflight(
        _PREFLIGHT_TOKEN,
        attempt_spine=attempt_spine,
        output_root=output_root,
        output_root_identity=expected_root_identity,
        registry_directory=registry,
        registry_identity={"device": registry_identity[0], "inode": registry_identity[1]},
        candidate_bundle_path=bundle_path,
        planned_paths=planned_paths,
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
    "VerifiedReviewNoWritePreflight",
    "VerifiedReviewParentModuleASpecApproval",
    "VerifiedReviewImplementationApproval",
    "VerifiedReviewAmendedImplementationReview",
    "VerifiedReviewRerunAuthorization",
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
    "bind_verified_review_no_write_preflight",
    "load_verified_parent_module_a_spec_approval",
    "load_verified_amendment_implementation_approval",
    "load_verified_amended_implementation_review",
    "load_verified_review_rerun_authorization",
    "load_verified_review_implementation_approval",
    "load_verified_candidate_receipt_bundle",
    "load_verified_terminal_artifact",
    "load_verified_run_history_ledger",
]
