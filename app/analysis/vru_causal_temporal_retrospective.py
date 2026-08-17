"""Sealed TASK-0258 temporal-retrospective contracts and computations."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, is_dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.analysis.vru_causal_training_v2 import (
    verify_vru_causal_shot_validity_training_extension_export,
)
from app.analysis.vru_causal_video_probe_v2 import (
    screen_vru_causal_video_embeddings_nested_v2,
)

MODULE_A_APPROVAL_SCHEMA = "agu.module-a-spec-approval.v1"
MODULE_A_ID = "existing-45-temporal-retrospective"
MODULE_A_APPROVAL_SCOPE = "module_a_implementation_only"
APPROVED_SPEC_FILENAMES = ("requirement.md", "solution.md", "gate-review.md")
TILED_SWIN_DIMENSION = 768
TILED_SWIN_TILE_COUNT = 4
TILED_SWIN_OUTPUT_DIMENSION = 1536
REGULARIZATION_C = 0.01
THRESHOLD_GRID = tuple(index / 20 for index in range(21))
RESERVE_BYTES = 3_758_096_384
_MAX_APPROVAL_BYTES = 1_048_576
_SHA256 = re.compile(r"[0-9a-f]{64}")
_APPROVAL_FIELDS = {
    "approval_scope",
    "approval_statement_sha256",
    "approved_at_utc",
    "approved_files",
    "artifact_sha256",
    "fresh_review_receipt",
    "module_id",
    "schema_version",
}
_APPROVED_FILE_FIELDS = {"filename", "file_sha256", "size_bytes"}
_FRESH_REVIEW_FIELDS = {"file_sha256", "internal_sha256"}
TEMPORAL_FEATURE_PLAN_SCHEMA = "agu.vru-causal-temporal-feature-plan.v1"
TEMPORAL_REPRESENTATION_CONTRACT = {
    "name": "swin3d-t-tiled-4x2s-mean-delta-v1",
    "backbone": "torchvision/swin3d_t/kinetics400_v1",
    "checkpoint_sha256": "7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130",
    "window_seconds": 8.0,
    "tile_count": 4,
    "tile_seconds": 2.0,
    "clip_frames_per_tile": 16,
    "review_plan_frames": 64,
    "tile_position_slices": [[0, 16], [16, 32], [32, 48], [48, 64]],
    "tile_embedding_dimension": 768,
    "model_input_dimension": 1536,
    "model_input_dtype": "float32",
    "frame_source": "verified_review_plan_frame_indexes",
    "preprocessing": "torchvision/Swin3D_T_Weights.KINETICS400_V1.transforms",
    "aggregation": "concat(mean(e0,e1,e2,e3),mean(e2,e3)-mean(e0,e1))",
}
TEMPORAL_EVALUATION_PROTOCOL = {
    "game_order": ["hazen", "randolph", "vtv", "harwood"],
    "game_row_counts": [8, 7, 7, 23],
    "outer_split": "leave_one_game_out",
    "inner_split": "leave_one_game_out_within_outer_training",
    "standardization": "StandardScaler_fit_training_split_only",
    "pca_enabled": False,
    "logistic_regression": {
        "C": 0.01,
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": 0,
    },
    "threshold_grid": list(THRESHOLD_GRID),
    "decision_rule": "probability_greater_than_or_equal_to_threshold",
    "threshold_selection": "balanced_accuracy_then_f1_then_precision_then_recall",
    "threshold_tie_break": "higher_threshold",
}
TEMPORAL_RESOURCE_POLICY = {
    "batch_size": 1,
    "max_memory_percent": 90.0,
    "min_available_memory_bytes": 2_147_483_648,
    "min_free_swap_bytes": 268_435_456,
    "max_system_cpu_percent": 95.0,
    "sample_interval_seconds": 2,
    "consecutive_breach_limit": 3,
    "reserve_bytes": RESERVE_BYTES,
    "max_cumulative_active_seconds": 43_200,
    "max_resource_samples": 21_600,
    "max_resource_log_bytes": 16_777_216,
    "process_tree_reaping_required": True,
}
TEMPORAL_ENVIRONMENT_CONTRACT = {
    "virtual_environment": ".venv",
    "python_version": "3.11.15",
    "operating_system": {
        "name": "macOS",
        "version": "26.5.2",
        "build": "25F84",
        "architecture": "arm64",
    },
    "hardware": {
        "model": "Mac mini Mac16,10",
        "processor": "Apple M4",
        "cpu_core_count": 10,
        "gpu_core_count": 10,
        "unified_memory_bytes": 17_179_869_184,
    },
    "device": "mps:0",
    "process_environment": {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    },
    "opencv": {
        "version": "4.13.0",
        "video_backend": "cv2.CAP_FFMPEG",
        "build_information_sha256": "6fa24b2683931738b55c6b8a2d2fe42ff5663b9d8b4eca4fc53fb1803a55e46e",
    },
    "numpy": {
        "version": "2.4.6",
        "show_config_sha256": "470ee8aedb85b9fa43dcd664ea25e13f2fad1bf6c822d5b9dddebcff81d2a3a7",
    },
    "scikit_learn_version": "1.9.0",
    "pytorch": {
        "version": "2.13.0",
        "config_sha256": "1943041ad11240ec18f123ae0f5aa98ed16a9ebf53b479bb4f0c11ea4c6f12f1",
    },
    "torchvision_version": "0.28.0",
    "scipy": {
        "version": "1.17.1",
        "show_config_sha256": "23f41e614b4eaa40a20a60411c0700c189258aa52c47569b60b59b6683e78438",
        "blas": "Accelerate",
        "lapack": "Accelerate",
    },
    "threadpoolctl": {
        "version": "3.6.0",
        "normalized_backends": [
            {
                "owner": "scikit-learn",
                "relative_library": ".dylibs/libomp.dylib",
                "user_api": "openmp",
                "internal_api": "openmp",
                "prefix": "libomp",
                "version": None,
                "num_threads": 1,
            },
            {
                "owner": "PyTorch",
                "relative_library": "lib/libomp.dylib",
                "user_api": "openmp",
                "internal_api": "openmp",
                "prefix": "libomp",
                "version": None,
                "num_threads": 1,
            },
        ],
    },
}
_PLAN_FIELDS = {
    "artifact_sha256",
    "environment_contract",
    "evaluation_protocol",
    "formal_evaluation_eligible",
    "label_hidden",
    "module_id",
    "ordered_examples",
    "promoted",
    "promotion_eligible",
    "purpose",
    "representation",
    "resource_policy",
    "runtime_consumable",
    "schema_version",
    "task0257_receipts",
    "training_consumable",
}
_PLAN_EXAMPLE_FIELDS = {
    "candidate_bundle_sha256",
    "event_id",
    "game_id",
    "ordinal",
    "production_family",
    "review_plan_receipt",
    "source_video_sha256",
    "tile_frame_indexes",
}
_PLAN_REPRESENTATION_FIELDS = {
    "aggregation",
    "backbone",
    "checkpoint_sha256",
    "clip_frames_per_tile",
    "frame_source",
    "model_input_dimension",
    "model_input_dtype",
    "name",
    "preprocessing",
    "review_plan_frames",
    "tile_count",
    "tile_embedding_dimension",
    "tile_position_slices",
    "tile_seconds",
    "window_seconds",
}
_VERIFIED_PLAN_TOKEN = object()


class DiskWriteBudgetError(ValueError):
    """Raised when a valid write request would cross the frozen disk reserve."""


TILED_SWIN_EMBEDDING_SCHEMA = "agu.vru-causal-tiled-swin-embeddings.v1"
RESOURCE_SAMPLE_SCHEMA = "agu.vru-causal-resource-sample.v1"
TILED_SWIN_ATTEMPT_SCHEMA = "agu.vru-causal-tiled-swin-attempt.v1"
TILED_SWIN_RESUME_SCHEMA = "agu.vru-causal-tiled-swin-embeddings-resume.v1"
TEMPORAL_RETROSPECTIVE_SCHEMA = "agu.vru-causal-temporal-retrospective.v1"
FINAL_EVALUATOR_SCHEMA = "agu.vru-causal-final-evaluator.v1"
MECHANICAL_GATE_SCHEMA = "agu.vru-causal-temporal-mechanical-gate.v1"
_EMBEDDING_FIELDS = {
    "artifact_sha256",
    "attempt_chain",
    "examples",
    "formal_evaluation_eligible",
    "module_id",
    "plan_receipt",
    "promoted",
    "promotion_eligible",
    "producer_environment",
    "purpose",
    "representation",
    "row_count",
    "runtime_consumable",
    "schema_version",
    "task0257_input_receipts",
    "training_consumable",
}
_EMBEDDING_EXAMPLE_FIELDS = {
    "derivation_only_tile_embeddings",
    "key",
    "model_input",
    "ordinal",
    "tile_frame_indexes",
}
_EMBEDDING_KEY_FIELDS = {
    "candidate_bundle_sha256",
    "event_id",
    "source_video_sha256",
}
_VERIFIED_EMBEDDING_TOKEN = object()
_VERIFIED_TASK0257_TOKEN = object()
_FRESH_EMBEDDING_TOKEN = object()
_COMMON_ARTIFACT_FIELDS = {
    "schema_version",
    "module_id",
    "purpose",
    "runtime_consumable",
    "training_consumable",
    "formal_evaluation_eligible",
    "promotion_eligible",
    "promoted",
}
_ATTEMPT_FIELDS = _COMMON_ARTIFACT_FIELDS | {
    "attempt_ordinal",
    "prior_attempt_receipt",
    "plan_receipt",
    "task0257_input_receipts",
    "started_prefix_count",
    "completed_prefix_count",
    "new_rows_verified",
    "cumulative_active_runtime_nanoseconds",
    "cumulative_resource_samples",
    "cumulative_resource_log_bytes",
    "resource_log_receipt",
    "resume_input_cas",
    "resume_output_cas",
    "received_signal",
    "disposition",
    "stop_reason",
    "artifact_sha256",
}
_RESUME_CAS_FIELDS = {"device", "inode", "size_bytes", "internal_sha256", "file_sha256"}
_ATTEMPT_CHAIN_RECEIPT_FIELDS = {"attempt_ordinal", "attempt_record", "resource_log", "resume_output_cas"}
_RESUME_FIELDS = _COMMON_ARTIFACT_FIELDS | {
    "artifact_sha256",
    "attempt_chain_receipts",
    "checkpoint_receipt",
    "completed_count",
    "completed_examples",
    "plan_receipt",
    "producer_environment",
    "representation",
    "row_count",
    "source_video_receipts",
    "task0257_input_receipts",
}


@dataclass(frozen=True)
class StoredArtifactReceipt:
    schema_version: str
    internal_sha256_field: str
    internal_sha256: str
    file_sha256: str
    filename: str
    size_bytes: int


@dataclass(frozen=True)
class FileReceipt:
    file_sha256: str
    filename: str
    size_bytes: int


@dataclass(frozen=True)
class ResumeCAS:
    device: int
    inode: int
    size_bytes: int
    internal_sha256: str
    file_sha256: str


@dataclass(frozen=True)
class ModuleAFinalGeneration:
    retrospective: Mapping[str, object]
    baseline_evaluator: Mapping[str, object]
    candidate_evaluator: Mapping[str, object]
    mechanical_gate: Mapping[str, object]


@dataclass(frozen=True)
class Task0257InputPaths:
    parent_selection: Path
    parent_source_manifest: Path
    parent_review_plan: Path
    parent_raw_frame_manifest: Path
    parent_review_jpegs: tuple[Path, ...]
    parent_sealed_review: Path
    parent_v1_export: Path
    parent_candidate_children: tuple[Path, Path, Path]
    parent_label_children: tuple[Path, Path, Path]
    v2_export: Path
    v2_candidate_child: Path
    v2_label_child: Path
    source_groups: Path
    four_video_training_manifest: Path
    old_embedding_files: tuple[Path, Path]
    old_nested_probe_plan: Path
    old_nested_probe: Path
    harwood_source_manifest: Path
    harwood_selection: Path
    harwood_review_plan: Path
    harwood_raw_frame_manifest: Path
    harwood_review_jpegs: tuple[Path, ...]
    harwood_sealed_review: Path
    source_videos: tuple[Path, Path, Path, Path]
    checkpoints: tuple[Path, Path]


@dataclass(frozen=True)
class Task0257ExpectedReceipts:
    parent_selection: StoredArtifactReceipt
    parent_source_manifest: FileReceipt
    parent_review_plan: StoredArtifactReceipt
    parent_raw_frame_manifest: StoredArtifactReceipt
    parent_review_jpegs: tuple[FileReceipt, ...]
    parent_sealed_review: StoredArtifactReceipt
    parent_v1_export: StoredArtifactReceipt
    parent_candidate_children: tuple[StoredArtifactReceipt, StoredArtifactReceipt, StoredArtifactReceipt]
    parent_label_children: tuple[FileReceipt, FileReceipt, FileReceipt]
    v2_export: StoredArtifactReceipt
    v2_candidate_child: StoredArtifactReceipt
    v2_label_child: FileReceipt
    source_groups: StoredArtifactReceipt
    four_video_training_manifest: StoredArtifactReceipt
    old_embedding_files: tuple[StoredArtifactReceipt, StoredArtifactReceipt]
    old_nested_probe_plan: StoredArtifactReceipt
    old_nested_probe: StoredArtifactReceipt
    harwood_source_manifest: FileReceipt
    harwood_selection: StoredArtifactReceipt
    harwood_review_plan: StoredArtifactReceipt
    harwood_raw_frame_manifest: StoredArtifactReceipt
    harwood_review_jpegs: tuple[FileReceipt, ...]
    harwood_sealed_review: StoredArtifactReceipt
    source_videos: tuple[FileReceipt, FileReceipt, FileReceipt, FileReceipt]
    checkpoints: tuple[FileReceipt, FileReceipt]


class VerifiedTask0257TemporalInputs:
    """Opaque TASK-0257 capability constructible only by full receipt replay."""

    __slots__ = (
        "_token",
        "expected_receipts",
        "harwood_review_plan",
        "old_embeddings",
        "ordered_rows",
        "parent_review_plan",
        "replayed_nested_probe",
        "source_groups",
        "training_chain",
        "validated_input_paths",
        "validated_leaf_identities",
        "validated_leaf_receipts",
    )

    def __new__(cls, token: object | None = None) -> VerifiedTask0257TemporalInputs:
        if token is not _VERIFIED_TASK0257_TOKEN:
            raise TypeError("VerifiedTask0257TemporalInputs cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object | None = None) -> None:
        if token is not _VERIFIED_TASK0257_TOKEN:
            raise TypeError("VerifiedTask0257TemporalInputs cannot be constructed directly")


class VerifiedTemporalFeaturePlan:
    """Opaque plan capability constructible only by the verified loader."""

    __slots__ = (
        "_artifact_sha256",
        "_device",
        "_file_sha256",
        "_inode",
        "_mtime_ns",
        "_path",
        "_payload",
        "_size_bytes",
        "_token",
    )

    def __new__(cls, token: object | None = None) -> VerifiedTemporalFeaturePlan:
        if token is not _VERIFIED_PLAN_TOKEN:
            raise TypeError("VerifiedTemporalFeaturePlan cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object | None = None) -> None:
        if token is not _VERIFIED_PLAN_TOKEN:
            raise TypeError("VerifiedTemporalFeaturePlan cannot be constructed directly")


class VerifiedTiledSwinEmbeddings:
    """Opaque embedding capability constructible only by the verified loader."""

    __slots__ = (
        "_artifact_sha256",
        "_device",
        "_file_sha256",
        "_inode",
        "_mtime_ns",
        "_path",
        "_payload",
        "_plan_artifact_sha256",
        "_size_bytes",
        "_token",
    )

    def __new__(cls, token: object | None = None) -> VerifiedTiledSwinEmbeddings:
        if token is not _VERIFIED_EMBEDDING_TOKEN:
            raise TypeError("VerifiedTiledSwinEmbeddings cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object | None = None) -> None:
        if token is not _VERIFIED_EMBEDDING_TOKEN:
            raise TypeError("VerifiedTiledSwinEmbeddings cannot be constructed directly")


class FreshlyProducedTiledSwinEmbeddings:
    """Transaction-local embedding capability that cannot cross public replay."""

    __slots__ = ("_file_sha256", "_payload", "_resource_summary", "_token")

    def __new__(cls, token: object | None = None) -> FreshlyProducedTiledSwinEmbeddings:
        if token is not _FRESH_EMBEDDING_TOKEN:
            raise TypeError("FreshlyProducedTiledSwinEmbeddings cannot be constructed directly")
        return super().__new__(cls)

    def __init__(self, token: object | None = None) -> None:
        if token is not _FRESH_EMBEDDING_TOKEN:
            raise TypeError("FreshlyProducedTiledSwinEmbeddings cannot be constructed directly")


_REGISTERED_TASK0257_CAPABILITIES: dict[int, tuple[VerifiedTask0257TemporalInputs, bytes]] = {}
_REGISTERED_PLAN_CAPABILITIES: dict[int, VerifiedTemporalFeaturePlan] = {}
_REGISTERED_EMBEDDING_CAPABILITIES: dict[int, VerifiedTiledSwinEmbeddings] = {}
_REGISTERED_FRESH_EMBEDDING_CAPABILITIES: dict[
    int,
    tuple[FreshlyProducedTiledSwinEmbeddings, bytes],
] = {}


def _json_capability_value(value: object) -> object:
    """Project capability state to deterministic JSON for mutation detection."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_capability_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_capability_value(value.tolist())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("verified capability mapping keys must be strings")
        return {key: _json_capability_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_capability_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"unsupported verified capability value: {type(value).__name__}")


def _task0257_capability_fingerprint(inputs: VerifiedTask0257TemporalInputs) -> bytes:
    state = {
        field: _json_capability_value(getattr(inputs, field))
        for field in VerifiedTask0257TemporalInputs.__slots__
        if field != "_token"
    }
    return json.dumps(
        state,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _register_task0257_capability(inputs: VerifiedTask0257TemporalInputs) -> None:
    _REGISTERED_TASK0257_CAPABILITIES[id(inputs)] = (
        inputs,
        _task0257_capability_fingerprint(inputs),
    )


def _register_capability(registry: dict[int, Any], capability: Any) -> None:
    registry[id(capability)] = capability


def _require_registered_capability(registry: Mapping[int, Any], capability: Any, message: str) -> None:
    if registry.get(id(capability)) is not capability:
        raise TypeError(message)


def _fresh_embedding_fingerprint(capability: FreshlyProducedTiledSwinEmbeddings) -> bytes:
    return json.dumps(
        {
            "payload": _json_capability_value(capability._payload),
            "resource_summary": _json_capability_value(capability._resource_summary),
            "file_sha256": capability._file_sha256,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _register_fresh_embedding_capability(capability: FreshlyProducedTiledSwinEmbeddings) -> None:
    _REGISTERED_FRESH_EMBEDDING_CAPABILITIES[id(capability)] = (
        capability,
        _fresh_embedding_fingerprint(capability),
    )


def _require_fresh_embedding_capability(capability: FreshlyProducedTiledSwinEmbeddings) -> None:
    if type(capability) is not FreshlyProducedTiledSwinEmbeddings:
        raise TypeError("a transaction-local tiled embedding capability is required")
    registered = _REGISTERED_FRESH_EMBEDDING_CAPABILITIES.get(id(capability))
    if registered is None or registered[0] is not capability:
        raise TypeError("a transaction-local tiled embedding capability is required")
    if registered[1] != _fresh_embedding_fingerprint(capability):
        raise ValueError("transaction-local tiled embedding capability was mutated")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _command_text(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        list(arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def observe_temporal_environment_contract() -> dict[str, object]:
    """Observe and validate the exact approved deterministic fit environment."""
    import cv2
    import scipy
    import sklearn
    import torch
    import torchvision
    from threadpoolctl import threadpool_info, threadpool_limits

    for name, expected in TEMPORAL_ENVIRONMENT_CONTRACT["process_environment"].items():
        if os.environ.get(name) != expected:
            raise ValueError(f"deterministic process environment {name} is invalid")
    numpy_output = StringIO()
    with redirect_stdout(numpy_output):
        np.show_config()
    scipy_output = StringIO()
    with redirect_stdout(scipy_output):
        scipy.show_config()
    scipy_config = scipy_output.getvalue()
    if scipy_config.count("Accelerate") < 2:
        raise ValueError("SciPy BLAS/LAPACK backend is not Accelerate")
    system_profile = json.loads(
        _command_text(("/usr/sbin/system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"))
    )
    hardware = system_profile.get("SPHardwareDataType")
    displays = system_profile.get("SPDisplaysDataType")
    if not isinstance(hardware, list) or len(hardware) != 1 or not isinstance(displays, list) or len(displays) != 1:
        raise ValueError("hardware profile is invalid")
    hardware_row = hardware[0]
    display_row = displays[0]
    if not isinstance(hardware_row, Mapping) or not isinstance(display_row, Mapping):
        raise ValueError("hardware profile rows are invalid")
    with threadpool_limits(limits=1):
        raw_backends = threadpool_info()
    normalized_backends: list[dict[str, object]] = []
    backend_suffixes = (
        ("scikit-learn", ".dylibs/libomp.dylib"),
        ("PyTorch", "lib/libomp.dylib"),
    )
    if not isinstance(raw_backends, list) or len(raw_backends) != len(backend_suffixes):
        raise ValueError("threadpool backend count is invalid")
    matched_indexes: set[int] = set()
    for owner, suffix in backend_suffixes:
        matches = [
            (index, raw)
            for index, raw in enumerate(raw_backends)
            if isinstance(raw, Mapping) and isinstance(raw.get("filepath"), str) and raw["filepath"].endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError("threadpool backend path is invalid")
        index, raw = matches[0]
        if index in matched_indexes:
            raise ValueError("threadpool backend ownership is ambiguous")
        matched_indexes.add(index)
        normalized_backends.append(
            {
                "owner": owner,
                "relative_library": suffix,
                "user_api": raw.get("user_api"),
                "internal_api": raw.get("internal_api"),
                "prefix": raw.get("prefix"),
                "version": raw.get("version"),
                "num_threads": raw.get("num_threads"),
            }
        )
    observed = {
        "virtual_environment": ".venv",
        "python_version": platform.python_version(),
        "operating_system": {
            "name": "macOS",
            "version": platform.mac_ver()[0],
            "build": _command_text(("/usr/bin/sw_vers", "-buildVersion")),
            "architecture": platform.machine(),
        },
        "hardware": {
            "model": f"{hardware_row.get('machine_name')} {hardware_row.get('machine_model')}",
            "processor": hardware_row.get("chip_type"),
            "cpu_core_count": int(_command_text(("/usr/sbin/sysctl", "-n", "hw.physicalcpu"))),
            "gpu_core_count": int(display_row.get("sppci_cores", "0")),
            "unified_memory_bytes": int(_command_text(("/usr/sbin/sysctl", "-n", "hw.memsize"))),
        },
        "device": "mps:0",
        "process_environment": copy.deepcopy(TEMPORAL_ENVIRONMENT_CONTRACT["process_environment"]),
        "opencv": {
            "version": cv2.__version__,
            "video_backend": "cv2.CAP_FFMPEG",
            "build_information_sha256": _sha256_bytes(cv2.getBuildInformation().encode("utf-8")),
        },
        "numpy": {
            "version": np.__version__,
            "show_config_sha256": _sha256_bytes(numpy_output.getvalue().encode("utf-8")),
        },
        "scikit_learn_version": sklearn.__version__,
        "pytorch": {
            "version": torch.__version__,
            "config_sha256": _sha256_bytes(torch.__config__.show().encode("utf-8")),
        },
        "torchvision_version": torchvision.__version__,
        "scipy": {
            "version": scipy.__version__,
            "show_config_sha256": _sha256_bytes(scipy_config.encode("utf-8")),
            "blas": "Accelerate",
            "lapack": "Accelerate",
        },
        "threadpoolctl": {
            "version": __import__("threadpoolctl").__version__,
            "normalized_backends": normalized_backends,
        },
    }
    if observed != TEMPORAL_ENVIRONMENT_CONTRACT:
        raise ValueError("observed environment does not match the approved contract")
    return observed


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path.absolute()
    while not os.path.lexists(candidate):
        parent = candidate.parent
        if parent == candidate:
            raise ValueError("output path has no existing filesystem ancestor")
        candidate = parent
    if candidate.is_symlink():
        raise ValueError("output path ancestor must not be a symlink")
    return candidate.resolve(strict=True)


def verify_disk_write_budget(
    *,
    output_paths: Sequence[Path],
    worst_case_new_bytes: int,
    reserve_bytes: int = RESERVE_BYTES,
) -> None:
    """Fail before writes when any output filesystem cannot retain the reserve."""
    if (
        type(worst_case_new_bytes) is not int
        or worst_case_new_bytes <= 0
        or type(reserve_bytes) is not int
        or reserve_bytes < RESERVE_BYTES
    ):
        raise ValueError("disk budget byte values are invalid or weakened")
    if not output_paths:
        raise ValueError("at least one output path is required for disk accounting")
    filesystems: dict[int, Path] = {}
    for raw_path in output_paths:
        if not isinstance(raw_path, Path):
            raise ValueError("disk budget output paths must be Path values")
        ancestor = _nearest_existing_ancestor(raw_path)
        device = ancestor.stat(follow_symlinks=False).st_dev
        filesystems.setdefault(device, ancestor)
    for ancestor in filesystems.values():
        free_bytes = shutil.disk_usage(ancestor).free
        if free_bytes - worst_case_new_bytes < reserve_bytes:
            raise DiskWriteBudgetError("disk write would cross the mandatory reserve")


def build_resource_sample(
    *,
    attempt_ordinal: int,
    attempt_sample_ordinal: int,
    cumulative_sample_ordinal: int,
    observed_attempt_active_nanoseconds: int,
    system_memory_percent: float,
    available_memory_bytes: int,
    free_swap_bytes: int,
    system_cpu_percent: float,
    prior_consecutive_breach_count: int,
) -> dict[str, object]:
    """Build one exact scheduled TASK-0258 resource observation."""
    integer_values = (
        attempt_ordinal,
        attempt_sample_ordinal,
        cumulative_sample_ordinal,
        observed_attempt_active_nanoseconds,
        available_memory_bytes,
        free_swap_bytes,
        prior_consecutive_breach_count,
    )
    if any(type(value) is not int for value in integer_values):
        raise ValueError("resource sample integer fields are invalid")
    if (
        attempt_ordinal not in (1, 2, 3)
        or attempt_sample_ordinal <= 0
        or not 0 < cumulative_sample_ordinal <= TEMPORAL_RESOURCE_POLICY["max_resource_samples"]
        or observed_attempt_active_nanoseconds < 0
        or available_memory_bytes < 0
        or free_swap_bytes < 0
        or prior_consecutive_breach_count < 0
    ):
        raise ValueError("resource sample integer ranges are invalid")
    percentages = (system_memory_percent, system_cpu_percent)
    if any(
        type(value) not in (int, float) or isinstance(value, bool) or not np.isfinite(value) for value in percentages
    ):
        raise ValueError("resource sample percentages are invalid")
    breached_limits: list[str] = []
    if system_memory_percent > TEMPORAL_RESOURCE_POLICY["max_memory_percent"]:
        breached_limits.append("memory_percent")
    if available_memory_bytes < TEMPORAL_RESOURCE_POLICY["min_available_memory_bytes"]:
        breached_limits.append("available_memory")
    if free_swap_bytes < TEMPORAL_RESOURCE_POLICY["min_free_swap_bytes"]:
        breached_limits.append("free_swap")
    if system_cpu_percent > TEMPORAL_RESOURCE_POLICY["max_system_cpu_percent"]:
        breached_limits.append("system_cpu")
    consecutive = prior_consecutive_breach_count + 1 if breached_limits else 0
    scheduled = TEMPORAL_RESOURCE_POLICY["sample_interval_seconds"] * cumulative_sample_ordinal
    if scheduled > TEMPORAL_RESOURCE_POLICY["max_cumulative_active_seconds"]:
        raise ValueError("resource sample exceeds the cumulative runtime schedule")
    return {
        "schema_version": RESOURCE_SAMPLE_SCHEMA,
        "attempt_ordinal": attempt_ordinal,
        "attempt_sample_ordinal": attempt_sample_ordinal,
        "cumulative_sample_ordinal": cumulative_sample_ordinal,
        "scheduled_cumulative_active_seconds": scheduled,
        "observed_attempt_active_nanoseconds": observed_attempt_active_nanoseconds,
        "system_memory_percent": float(system_memory_percent),
        "available_memory_bytes": available_memory_bytes,
        "free_swap_bytes": free_swap_bytes,
        "system_cpu_percent": float(system_cpu_percent),
        "consecutive_breach_count": consecutive,
        "breached_limits": breached_limits,
    }


def encode_resource_sample_line(payload: Mapping[str, object]) -> bytes:
    expected_fields = {
        "schema_version",
        "attempt_ordinal",
        "attempt_sample_ordinal",
        "cumulative_sample_ordinal",
        "scheduled_cumulative_active_seconds",
        "observed_attempt_active_nanoseconds",
        "system_memory_percent",
        "available_memory_bytes",
        "free_swap_bytes",
        "system_cpu_percent",
        "consecutive_breach_count",
        "breached_limits",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != RESOURCE_SAMPLE_SCHEMA:
        raise ValueError("resource sample fields are invalid")
    line = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(line) > TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"]:
        raise ValueError("resource sample line exceeds the log cap")
    return line


def verify_resource_log_bytes(
    encoded: bytes,
    *,
    expected_attempt_ordinal: int,
    prior_cumulative_samples: int,
    prior_consecutive_breach_count: int | None,
) -> tuple[int, int]:
    """Verify exact JSONL schedule, values, and canonical bytes for one attempt."""
    if (
        type(expected_attempt_ordinal) is not int
        or expected_attempt_ordinal not in (1, 2, 3)
        or type(prior_cumulative_samples) is not int
        or prior_cumulative_samples < 0
        or (
            prior_consecutive_breach_count is not None
            and (type(prior_consecutive_breach_count) is not int or prior_consecutive_breach_count < 0)
        )
    ):
        raise ValueError("resource log verification inputs are invalid")
    if len(encoded) > TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"]:
        raise ValueError("resource log exceeds the cumulative byte cap")
    if encoded and not encoded.endswith(b"\n"):
        raise ValueError("resource log has a truncated final row")
    rows = encoded.splitlines(keepends=True)
    consecutive = prior_consecutive_breach_count
    for attempt_sample_ordinal, line in enumerate(rows, start=1):
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("resource log JSON is invalid") from error
        if consecutive is None:
            breached = payload.get("breached_limits") if isinstance(payload, Mapping) else None
            observed_consecutive = payload.get("consecutive_breach_count") if isinstance(payload, Mapping) else None
            if isinstance(breached, list) and breached:
                if type(observed_consecutive) is not int or observed_consecutive <= 0:
                    raise ValueError("resource log initial breach count is invalid")
                consecutive = observed_consecutive - 1
            else:
                consecutive = 0
        cumulative_ordinal = prior_cumulative_samples + attempt_sample_ordinal
        if isinstance(payload, Mapping):
            try:
                rebuilt = build_resource_sample(
                    attempt_ordinal=payload["attempt_ordinal"],
                    attempt_sample_ordinal=payload["attempt_sample_ordinal"],
                    cumulative_sample_ordinal=payload["cumulative_sample_ordinal"],
                    observed_attempt_active_nanoseconds=payload["observed_attempt_active_nanoseconds"],
                    system_memory_percent=payload["system_memory_percent"],
                    available_memory_bytes=payload["available_memory_bytes"],
                    free_swap_bytes=payload["free_swap_bytes"],
                    system_cpu_percent=payload["system_cpu_percent"],
                    prior_consecutive_breach_count=consecutive,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("resource log row values are invalid") from error
        else:
            rebuilt = None
        if (
            not isinstance(payload, Mapping)
            or payload.get("attempt_ordinal") != expected_attempt_ordinal
            or payload.get("attempt_sample_ordinal") != attempt_sample_ordinal
            or payload.get("cumulative_sample_ordinal") != cumulative_ordinal
            or rebuilt != payload
            or encode_resource_sample_line(payload) != line
        ):
            raise ValueError("resource log schedule or canonical bytes are invalid")
        consecutive = payload["consecutive_breach_count"]
    return len(rows), consecutive if consecutive is not None else 0


def _verify_resume_cas_mapping(value: object, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, Mapping) or set(value) != _RESUME_CAS_FIELDS:
        raise ValueError("resume CAS fields are invalid")
    if (
        any(type(value[field]) is not int or value[field] < 0 for field in ("device", "inode", "size_bytes"))
        or not _is_sha256(value.get("internal_sha256"))
        or not _is_sha256(value.get("file_sha256"))
    ):
        raise ValueError("resume CAS values are invalid")


def _verify_common_false_artifact(payload: Mapping[str, object], *, schema_version: str) -> None:
    if payload.get("schema_version") != schema_version or payload.get("module_id") != MODULE_A_ID:
        raise ValueError("Module-A artifact identity is invalid")
    if payload.get("purpose") != "development_diagnostic_only":
        raise ValueError("Module-A artifact purpose is invalid")
    for field in (
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    ):
        if payload.get(field) is not False:
            raise ValueError("Module-A eligibility flags are invalid")


def verify_tiled_swin_attempt(payload: Mapping[str, object]) -> dict[str, object]:
    if set(payload) != _ATTEMPT_FIELDS:
        raise ValueError("tiled Swin attempt fields are invalid")
    _verify_common_false_artifact(payload, schema_version=TILED_SWIN_ATTEMPT_SCHEMA)
    if payload.get("artifact_sha256") != _canonical_sha256(payload):
        raise ValueError("tiled Swin attempt artifact SHA-256 is invalid")
    attempt_ordinal = payload.get("attempt_ordinal")
    if type(attempt_ordinal) is not int or attempt_ordinal not in (1, 2, 3):
        raise ValueError("tiled Swin attempt ordinal is invalid")
    prior = payload.get("prior_attempt_receipt")
    if attempt_ordinal == 1:
        if prior is not None:
            raise ValueError("attempt one cannot have a prior receipt")
    else:
        _verify_stored_receipt_mapping(prior)
    if (
        payload.get("plan_receipt") is None
        or set(payload["plan_receipt"])
        != {
            "artifact_sha256",
            "file_sha256",
        }
        or not all(_is_sha256(value) for value in payload["plan_receipt"].values())
    ):
        raise ValueError("tiled Swin attempt plan receipt is invalid")
    _verify_task0257_receipt_mapping(payload.get("task0257_input_receipts"))
    counter_fields = (
        "started_prefix_count",
        "completed_prefix_count",
        "new_rows_verified",
        "cumulative_active_runtime_nanoseconds",
        "cumulative_resource_samples",
        "cumulative_resource_log_bytes",
    )
    if any(type(payload.get(field)) is not int or payload[field] < 0 for field in counter_fields):
        raise ValueError("tiled Swin attempt counters are invalid")
    started = payload["started_prefix_count"]
    completed = payload["completed_prefix_count"]
    if not 0 <= started <= completed <= 45 or payload["new_rows_verified"] != completed - started:
        raise ValueError("tiled Swin attempt prefix counters are invalid")
    if (
        payload["cumulative_resource_samples"] > TEMPORAL_RESOURCE_POLICY["max_resource_samples"]
        or payload["cumulative_resource_log_bytes"] > TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"]
        or payload["cumulative_active_runtime_nanoseconds"]
        > TEMPORAL_RESOURCE_POLICY["max_cumulative_active_seconds"] * 1_000_000_000
    ):
        raise ValueError("tiled Swin attempt cumulative caps are invalid")
    _verify_file_receipt_mapping(payload.get("resource_log_receipt"))
    _verify_resume_cas_mapping(payload.get("resume_input_cas"), allow_none=True)
    _verify_resume_cas_mapping(payload.get("resume_output_cas"), allow_none=True)
    if attempt_ordinal == 1 and payload.get("resume_input_cas") is not None:
        raise ValueError("attempt one cannot have a resume input CAS")
    if attempt_ordinal > 1 and payload.get("resume_input_cas") is None:
        raise ValueError("resumed attempt requires a resume input CAS")
    disposition = payload.get("disposition")
    signal_name = payload.get("received_signal")
    stop_reason = payload.get("stop_reason")
    if disposition == "completed":
        if (
            completed != 45
            or signal_name is not None
            or stop_reason is not None
            or payload["resume_output_cas"] is not None
        ):
            raise ValueError("completed attempt state is invalid")
    elif disposition == "interrupted_recoverable":
        expected_reason = {"SIGINT": "external_sigint", "SIGTERM": "external_sigterm"}.get(signal_name)
        if (
            attempt_ordinal not in (1, 2)
            or not 0 < payload["new_rows_verified"]
            or not completed < 45
            or stop_reason != expected_reason
            or payload["resume_output_cas"] is None
        ):
            raise ValueError("recoverable attempt state is invalid")
    elif disposition == "terminal_failure":
        terminal_reasons = {
            "receipt_failure",
            "schema_failure",
            "input_identity_failure",
            "resume_cas_failure",
            "disk_failure",
            "resource_breach",
            "runtime_cap",
            "sample_cap",
            "log_byte_cap",
            "decode_failure",
            "model_failure",
            "determinism_failure",
            "non_advancing_prefix",
            "unsupported_signal",
            "attempt_limit",
            "publication_failure",
        }
        if stop_reason not in terminal_reasons or payload["resume_output_cas"] is not None:
            raise ValueError("terminal attempt state is invalid")
    else:
        raise ValueError("tiled Swin attempt disposition is invalid")
    if signal_name not in (None, "SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"):
        raise ValueError("tiled Swin attempt signal is invalid")
    return copy.deepcopy(dict(payload))


def seal_tiled_swin_attempt(body: Mapping[str, object]) -> dict[str, object]:
    artifact = {
        "schema_version": TILED_SWIN_ATTEMPT_SCHEMA,
        "module_id": MODULE_A_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        **copy.deepcopy(dict(body)),
    }
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_tiled_swin_attempt(artifact)


def verify_tiled_swin_resume(
    payload: Mapping[str, object],
    *,
    plan: VerifiedTemporalFeaturePlan,
) -> dict[str, object]:
    """Verify an interrupted attempt's exact plan-prefix feature state."""
    _require_verified_plan(plan)
    if set(payload) != _RESUME_FIELDS:
        raise ValueError("tiled Swin resume fields are invalid")
    _verify_common_false_artifact(payload, schema_version=TILED_SWIN_RESUME_SCHEMA)
    if payload.get("artifact_sha256") != _canonical_sha256(payload):
        raise ValueError("tiled Swin resume artifact SHA-256 is invalid")
    if payload.get("plan_receipt") != {
        "artifact_sha256": plan._artifact_sha256,
        "file_sha256": plan._file_sha256,
    }:
        raise ValueError("tiled Swin resume plan receipt is invalid")
    if (
        payload.get("task0257_input_receipts") != plan._payload["task0257_receipts"]
        or payload.get("representation") != plan._payload["representation"]
        or payload.get("producer_environment") != plan._payload["environment_contract"]
        or payload.get("checkpoint_receipt") != plan._payload["task0257_receipts"]["checkpoints"][1]
        or payload.get("source_video_receipts") != plan._payload["task0257_receipts"]["source_videos"]
    ):
        raise ValueError("tiled Swin resume input contract is invalid")
    chain = payload.get("attempt_chain_receipts")
    if not isinstance(chain, list) or len(chain) > 2:
        raise ValueError("tiled Swin resume attempt chain is invalid")
    if chain:
        _verify_attempt_chain_receipts(chain)
    row_count = payload.get("row_count")
    completed_count = payload.get("completed_count")
    completed_examples = payload.get("completed_examples")
    if (
        type(row_count) is not int
        or row_count != 45
        or type(completed_count) is not int
        or not 0 < completed_count < row_count
        or not isinstance(completed_examples, list)
        or len(completed_examples) != completed_count
    ):
        raise ValueError("tiled Swin resume prefix is invalid")
    for row in completed_examples:
        if (
            not isinstance(row, list)
            or len(row) != TILED_SWIN_TILE_COUNT
            or any(
                not isinstance(tile, list)
                or len(tile) != TILED_SWIN_DIMENSION
                or any(type(value) is not float or not np.isfinite(value) for value in tile)
                for tile in row
            )
        ):
            raise ValueError("tiled Swin resume prefix values are invalid")
    return copy.deepcopy(dict(payload))


def seal_tiled_swin_resume(
    body: Mapping[str, object],
    *,
    plan: VerifiedTemporalFeaturePlan,
) -> dict[str, object]:
    artifact = {
        "schema_version": TILED_SWIN_RESUME_SCHEMA,
        "module_id": MODULE_A_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        **copy.deepcopy(dict(body)),
    }
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_tiled_swin_resume(artifact, plan=plan)


def _verify_attempt_chain_receipts(value: object) -> None:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError("attempt chain must contain one to three receipts")
    for ordinal, row in enumerate(value, start=1):
        if not isinstance(row, Mapping) or set(row) != _ATTEMPT_CHAIN_RECEIPT_FIELDS:
            raise ValueError("attempt chain receipt fields are invalid")
        if row.get("attempt_ordinal") != ordinal or type(row.get("attempt_ordinal")) is not int:
            raise ValueError("attempt chain order is invalid")
        _verify_stored_receipt_mapping(row.get("attempt_record"))
        _verify_file_receipt_mapping(row.get("resource_log"))
        _verify_resume_cas_mapping(row.get("resume_output_cas"), allow_none=True)


def _stored_artifact_receipt(
    path: Path,
    payload: Mapping[str, object],
    encoded: bytes,
) -> dict[str, object]:
    internal_sha256 = payload.get("artifact_sha256")
    schema_version = payload.get("schema_version")
    if not _is_sha256(internal_sha256) or not isinstance(schema_version, str):
        raise ValueError("stored artifact receipt source is invalid")
    return {
        "schema_version": schema_version,
        "internal_sha256_field": "artifact_sha256",
        "internal_sha256": internal_sha256,
        "file_sha256": _sha256_bytes(encoded),
        "filename": path.name,
        "size_bytes": len(encoded),
    }


def _file_receipt_from_bytes(path: Path, encoded: bytes) -> dict[str, object]:
    return {
        "file_sha256": _sha256_bytes(encoded),
        "filename": path.name,
        "size_bytes": len(encoded),
    }


def _resume_cas_from_bytes(
    path: Path,
    payload: Mapping[str, object],
    encoded: bytes,
) -> dict[str, object]:
    metadata = path.stat(follow_symlinks=False)
    internal_sha256 = payload.get("artifact_sha256")
    if not stat.S_ISREG(metadata.st_mode) or not _is_sha256(internal_sha256):
        raise ValueError("published resume CAS source is invalid")
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "internal_sha256": internal_sha256,
        "file_sha256": _sha256_bytes(encoded),
    }


def _replay_published_prior_attempts(
    *,
    output_root: Path,
    attempt_chain: Sequence[Mapping[str, object]],
    plan: VerifiedTemporalFeaturePlan,
) -> dict[str, object]:
    """Replay every durable recoverable attempt from its published bytes."""
    _require_verified_plan(plan)
    chain = [copy.deepcopy(dict(row)) for row in attempt_chain]
    if chain:
        _verify_attempt_chain_receipts(chain)
    if len(chain) > 2:
        raise ValueError("prior-attempt replay cannot include a terminal attempt")
    root = Path(output_root)
    attempts_root = root / "attempts"
    if not chain:
        if attempts_root.exists() or attempts_root.is_symlink():
            raise ValueError("unexpected prior-attempt directory without a receipt chain")
        return {
            "attempt_records": [],
            "latest_resume_cas": None,
            "completed_prefix_count": 0,
            "cumulative_active_runtime_nanoseconds": 0,
            "cumulative_resource_samples": 0,
            "cumulative_resource_log_bytes": 0,
            "ending_consecutive_breach_count": 0,
        }
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise ValueError("prior-attempt root is invalid")
    expected_names = {f"attempt-{ordinal:04d}" for ordinal in range(1, len(chain) + 1)}
    if {path.name for path in attempts_root.iterdir()} != expected_names:
        raise ValueError("prior-attempt directory coverage is invalid")

    records: list[dict[str, object]] = []
    latest_resume_cas: dict[str, object] | None = None
    completed_prefix_count = 0
    cumulative_runtime = 0
    cumulative_samples = 0
    cumulative_log_bytes = 0
    consecutive_breach_count = 0
    for ordinal, chain_row in enumerate(chain, start=1):
        directory = attempts_root / f"attempt-{ordinal:04d}"
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or {path.name for path in directory.iterdir()}
            != {"attempt_record.json", "resource_guard.jsonl", "resume.json"}
        ):
            raise ValueError("prior-attempt generation coverage is invalid")

        attempt_path = directory / "attempt_record.json"
        attempt, attempt_bytes = _read_bounded_json(attempt_path, max_bytes=67_108_864)
        if attempt_bytes != _canonical_json_bytes(attempt):
            raise ValueError("prior-attempt record encoding is invalid")
        verified_attempt = verify_tiled_swin_attempt(attempt)
        attempt_receipt = _stored_artifact_receipt(attempt_path, verified_attempt, attempt_bytes)
        if attempt_receipt != chain_row["attempt_record"]:
            raise ValueError("prior-attempt record receipt does not match")

        resource_path = directory / "resource_guard.jsonl"
        resource_bytes = _read_bounded_bytes(
            resource_path,
            max_bytes=TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"],
        )
        resource_receipt = _file_receipt_from_bytes(resource_path, resource_bytes)
        if (
            resource_receipt != chain_row["resource_log"]
            or resource_receipt != verified_attempt["resource_log_receipt"]
        ):
            raise ValueError("prior-attempt resource receipt does not match")
        current_sample_count, consecutive_breach_count = verify_resource_log_bytes(
            resource_bytes,
            expected_attempt_ordinal=ordinal,
            prior_cumulative_samples=cumulative_samples,
            prior_consecutive_breach_count=consecutive_breach_count,
        )

        resume_path = directory / "resume.json"
        resume, resume_bytes = _read_bounded_json(resume_path, max_bytes=67_108_864)
        if resume_bytes != _canonical_json_bytes(resume):
            raise ValueError("prior-attempt resume encoding is invalid")
        verified_resume = verify_tiled_swin_resume(resume, plan=plan)
        resume_cas = _resume_cas_from_bytes(resume_path, verified_resume, resume_bytes)
        expected_prior_receipt = chain[ordinal - 2]["attempt_record"] if ordinal > 1 else None
        if (
            verified_attempt["attempt_ordinal"] != ordinal
            or verified_attempt["disposition"] != "interrupted_recoverable"
            or verified_attempt["plan_receipt"]
            != {"artifact_sha256": plan._artifact_sha256, "file_sha256": plan._file_sha256}
            or verified_attempt["task0257_input_receipts"] != plan._payload["task0257_receipts"]
            or verified_attempt["prior_attempt_receipt"] != expected_prior_receipt
            or verified_attempt["resume_input_cas"] != latest_resume_cas
            or verified_attempt["resume_output_cas"] != resume_cas
            or chain_row["resume_output_cas"] != resume_cas
            or verified_attempt["started_prefix_count"] != completed_prefix_count
            or verified_attempt["completed_prefix_count"] != verified_resume["completed_count"]
            or verified_attempt["new_rows_verified"]
            != verified_attempt["completed_prefix_count"] - completed_prefix_count
            or verified_resume["attempt_chain_receipts"] != chain[: ordinal - 1]
            or verified_attempt["cumulative_resource_samples"] != cumulative_samples + current_sample_count
            or verified_attempt["cumulative_resource_log_bytes"] != cumulative_log_bytes + len(resource_bytes)
            or verified_attempt["cumulative_active_runtime_nanoseconds"] < cumulative_runtime
        ):
            raise ValueError("prior-attempt chain binding is invalid")
        records.append(verified_attempt)
        latest_resume_cas = resume_cas
        completed_prefix_count = verified_attempt["completed_prefix_count"]
        cumulative_runtime = verified_attempt["cumulative_active_runtime_nanoseconds"]
        cumulative_samples = verified_attempt["cumulative_resource_samples"]
        cumulative_log_bytes = verified_attempt["cumulative_resource_log_bytes"]

    return {
        "attempt_records": records,
        "latest_resume_cas": latest_resume_cas,
        "completed_prefix_count": completed_prefix_count,
        "cumulative_active_runtime_nanoseconds": cumulative_runtime,
        "cumulative_resource_samples": cumulative_samples,
        "cumulative_resource_log_bytes": cumulative_log_bytes,
        "ending_consecutive_breach_count": consecutive_breach_count,
    }


def _task0257_path_receipt_pairs(
    paths: Task0257InputPaths,
    receipts: Task0257ExpectedReceipts,
) -> tuple[tuple[Path, StoredArtifactReceipt | FileReceipt], ...]:
    paired_sequences = (
        (paths.parent_review_jpegs, receipts.parent_review_jpegs),
        (paths.parent_candidate_children, receipts.parent_candidate_children),
        (paths.parent_label_children, receipts.parent_label_children),
        (paths.old_embedding_files, receipts.old_embedding_files),
        (paths.harwood_review_jpegs, receipts.harwood_review_jpegs),
        (paths.source_videos, receipts.source_videos),
        (paths.checkpoints, receipts.checkpoints),
    )
    if any(len(path_values) != len(receipt_values) for path_values, receipt_values in paired_sequences):
        raise ValueError("TASK-0257 path and receipt cardinalities do not match")
    scalar_pairs = (
        (paths.parent_selection, receipts.parent_selection),
        (paths.parent_source_manifest, receipts.parent_source_manifest),
        (paths.parent_review_plan, receipts.parent_review_plan),
        (paths.parent_raw_frame_manifest, receipts.parent_raw_frame_manifest),
        (paths.parent_sealed_review, receipts.parent_sealed_review),
        (paths.parent_v1_export, receipts.parent_v1_export),
        (paths.v2_export, receipts.v2_export),
        (paths.v2_candidate_child, receipts.v2_candidate_child),
        (paths.v2_label_child, receipts.v2_label_child),
        (paths.source_groups, receipts.source_groups),
        (paths.four_video_training_manifest, receipts.four_video_training_manifest),
        (paths.old_nested_probe_plan, receipts.old_nested_probe_plan),
        (paths.old_nested_probe, receipts.old_nested_probe),
        (paths.harwood_source_manifest, receipts.harwood_source_manifest),
        (paths.harwood_selection, receipts.harwood_selection),
        (paths.harwood_review_plan, receipts.harwood_review_plan),
        (paths.harwood_raw_frame_manifest, receipts.harwood_raw_frame_manifest),
        (paths.harwood_sealed_review, receipts.harwood_sealed_review),
    )
    expanded = tuple(
        pair
        for path_values, receipt_values in paired_sequences
        for pair in zip(path_values, receipt_values, strict=True)
    )
    return (*scalar_pairs, *expanded)


def _require_unique_leaf_paths(
    pairs: Sequence[tuple[Path, StoredArtifactReceipt | FileReceipt]],
) -> tuple[Path, ...]:
    normalized: set[str] = set()
    identities: set[tuple[int, int]] = set()
    resolved_paths: list[Path] = []
    for raw_path, _receipt in pairs:
        if not isinstance(raw_path, Path):
            raise ValueError("TASK-0257 leaf paths must be Path values")
        absolute = raw_path.absolute()
        casefolded = os.path.normcase(str(absolute)).casefold()
        if casefolded in normalized:
            raise ValueError("TASK-0257 inputs require unique leaf paths")
        normalized.add(casefolded)
        if os.path.lexists(absolute):
            if absolute.is_symlink():
                raise ValueError("TASK-0257 inputs require non-symlink unique leaf paths")
            resolved = absolute.resolve(strict=True)
            metadata = resolved.stat(follow_symlinks=False)
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in identities:
                raise ValueError("TASK-0257 inputs require unique leaf inode identities")
            identities.add(identity)
            resolved_paths.append(resolved)
        else:
            resolved_paths.append(absolute)
    return tuple(resolved_paths)


def _verify_leaf_receipt(
    raw_path: Path,
    receipt: StoredArtifactReceipt | FileReceipt,
) -> tuple[dict[str, Any] | None, tuple[int, int, int, int]]:
    path = Path(raw_path)
    if path.is_symlink():
        raise ValueError("receipt-bound leaf must not be a symlink")
    path = path.resolve(strict=True)
    if (
        not isinstance(receipt.filename, str)
        or receipt.filename != path.name
        or Path(receipt.filename).name != receipt.filename
        or type(receipt.size_bytes) is not int
        or receipt.size_bytes < 0
        or not _is_sha256(receipt.file_sha256)
    ):
        raise ValueError("leaf receipt fields are invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != receipt.size_bytes:
            raise ValueError("receipt-bound leaf size or type is invalid")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1_048_576)
            if not chunk:
                break
            digest.update(chunk)
        final_metadata = os.fstat(descriptor)
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        if identity != (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
        ):
            raise ValueError("receipt-bound leaf changed during hashing")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != receipt.file_sha256:
        raise ValueError("receipt-bound leaf file SHA-256 does not match")
    if isinstance(receipt, FileReceipt):
        return None, identity
    if (
        not isinstance(receipt.schema_version, str)
        or not receipt.schema_version
        or not isinstance(receipt.internal_sha256_field, str)
        or re.fullmatch(r"[a-z][a-z0-9_]*", receipt.internal_sha256_field) is None
        or not _is_sha256(receipt.internal_sha256)
        or receipt.size_bytes > 67_108_864
    ):
        raise ValueError("stored artifact receipt fields are invalid")
    payload, encoded = _read_bounded_json(path, max_bytes=67_108_864)
    if _sha256_bytes(encoded) != receipt.file_sha256:
        raise ValueError("stored artifact changed between receipt checks")
    if (
        payload.get("schema_version") != receipt.schema_version
        or payload.get(receipt.internal_sha256_field) != receipt.internal_sha256
    ):
        raise ValueError("stored artifact internal receipt does not match")
    return payload, identity


def verify_task0257_temporal_inputs(
    *,
    paths: Task0257InputPaths,
    expected_receipts: Task0257ExpectedReceipts,
) -> VerifiedTask0257TemporalInputs:
    """Verify the complete frozen TASK-0257 leaf graph before model work."""
    if type(paths) is not Task0257InputPaths or type(expected_receipts) is not Task0257ExpectedReceipts:
        raise TypeError("TASK-0257 inputs require the closed path and receipt contracts")
    pairs = _task0257_path_receipt_pairs(paths, expected_receipts)
    resolved_paths = _require_unique_leaf_paths(pairs)
    payloads: dict[Path, dict[str, Any]] = {}
    identities: dict[Path, tuple[int, int, int, int]] = {}
    for (raw_path, receipt), resolved_path in zip(pairs, resolved_paths, strict=True):
        payload, identity = _verify_leaf_receipt(raw_path, receipt)
        identities[resolved_path] = identity
        if payload is not None:
            payloads[resolved_path] = payload

    def payload_for(path: Path) -> dict[str, Any]:
        try:
            return payloads[path.resolve(strict=True)]
        except KeyError as error:
            raise ValueError("required TASK-0257 JSON payload was not receipt verified") from error

    parent_roots = {
        path.resolve(strict=True).parent for path in (*paths.parent_candidate_children, *paths.parent_label_children)
    }
    extension_roots = {
        paths.v2_candidate_child.resolve(strict=True).parent,
        paths.v2_label_child.resolve(strict=True).parent,
    }
    if len(parent_roots) != 1 or len(extension_roots) != 1:
        raise ValueError("TASK-0257 child assets must share their frozen roots")
    parent_asset_root = next(iter(parent_roots))
    extension_asset_root = next(iter(extension_roots))
    v2_payload = payload_for(paths.v2_export)
    training_chain = verify_vru_causal_shot_validity_training_extension_export(
        v2_payload,
        expected_artifact_sha256=expected_receipts.v2_export.internal_sha256,
        extension_asset_root=extension_asset_root,
        parent_export_path=paths.parent_v1_export,
        expected_parent_artifact_sha256=expected_receipts.parent_v1_export.internal_sha256,
        expected_parent_file_sha256=expected_receipts.parent_v1_export.file_sha256,
        parent_asset_root=parent_asset_root,
        harwood_selection_path=paths.harwood_selection,
        expected_harwood_selection_artifact_sha256=(expected_receipts.harwood_selection.internal_sha256),
        harwood_review_plan_path=paths.harwood_review_plan,
        expected_harwood_review_plan_artifact_sha256=(expected_receipts.harwood_review_plan.internal_sha256),
        harwood_sealed_review_path=paths.harwood_sealed_review,
        expected_harwood_sealed_review_artifact_sha256=(expected_receipts.harwood_sealed_review.internal_sha256),
        harwood_raw_frame_manifest_path=paths.harwood_raw_frame_manifest,
        expected_harwood_raw_frame_manifest_artifact_sha256=(
            expected_receipts.harwood_raw_frame_manifest.internal_sha256
        ),
        harwood_source_manifest_path=paths.harwood_source_manifest,
        source_groups_path=paths.source_groups,
        expected_source_groups_artifact_sha256=(expected_receipts.source_groups.internal_sha256),
    )
    ordered_rows = tuple(copy.deepcopy(row) for row in training_chain.resolved_examples)
    if (
        len(ordered_rows) != 45
        or sum(row.get("event_present") is True for row in ordered_rows) != 17
        or sum(row.get("event_present") is False for row in ordered_rows) != 28
        or any(type(row.get("event_present")) is not bool for row in ordered_rows)
    ):
        raise ValueError("TASK-0257 resolved training view is not exactly 45 rows / 17+ / 28-")
    replayed_probe = screen_vru_causal_video_embeddings_nested_v2(
        embedding_paths=paths.old_embedding_files,
        expected_embedding_artifact_sha256s=tuple(
            receipt.internal_sha256 for receipt in expected_receipts.old_embedding_files
        ),
        expected_embedding_file_sha256s=tuple(receipt.file_sha256 for receipt in expected_receipts.old_embedding_files),
        training_export_path=paths.v2_export,
        expected_training_export_artifact_sha256=expected_receipts.v2_export.internal_sha256,
        expected_training_export_file_sha256=expected_receipts.v2_export.file_sha256,
        extension_asset_root=extension_asset_root,
        parent_export_path=paths.parent_v1_export,
        expected_parent_artifact_sha256=expected_receipts.parent_v1_export.internal_sha256,
        expected_parent_file_sha256=expected_receipts.parent_v1_export.file_sha256,
        parent_asset_root=parent_asset_root,
        harwood_selection_path=paths.harwood_selection,
        expected_harwood_selection_artifact_sha256=(expected_receipts.harwood_selection.internal_sha256),
        harwood_review_plan_path=paths.harwood_review_plan,
        expected_harwood_review_plan_artifact_sha256=(expected_receipts.harwood_review_plan.internal_sha256),
        harwood_sealed_review_path=paths.harwood_sealed_review,
        expected_harwood_sealed_review_artifact_sha256=(expected_receipts.harwood_sealed_review.internal_sha256),
        harwood_raw_frame_manifest_path=paths.harwood_raw_frame_manifest,
        expected_harwood_raw_frame_manifest_artifact_sha256=(
            expected_receipts.harwood_raw_frame_manifest.internal_sha256
        ),
        harwood_source_manifest_path=paths.harwood_source_manifest,
        source_groups_path=paths.source_groups,
        expected_source_groups_artifact_sha256=expected_receipts.source_groups.internal_sha256,
        expected_source_groups_file_sha256=expected_receipts.source_groups.file_sha256,
        training_manifest_path=paths.four_video_training_manifest,
        expected_training_manifest_sha256=(expected_receipts.four_video_training_manifest.internal_sha256),
        expected_training_manifest_file_sha256=(expected_receipts.four_video_training_manifest.file_sha256),
        probe_plan_path=paths.old_nested_probe_plan,
        expected_probe_plan_sha256=expected_receipts.old_nested_probe_plan.internal_sha256,
        expected_probe_plan_file_sha256=expected_receipts.old_nested_probe_plan.file_sha256,
    )
    stored_probe = payload_for(paths.old_nested_probe)
    if replayed_probe != stored_probe:
        raise ValueError("TASK-0257 nested probe does not reproduce its stored bytes")
    chain_paths = {path.resolve(strict=True) for path in training_chain.validated_input_paths}
    if not chain_paths.issubset(set(resolved_paths)):
        raise ValueError("TASK-0257 verified chain used an undeclared leaf path")
    for path, expected_identity in identities.items():
        metadata = path.stat(follow_symlinks=False)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != expected_identity:
            raise ValueError("TASK-0257 input identity changed during full replay")
    verified = VerifiedTask0257TemporalInputs(_VERIFIED_TASK0257_TOKEN)
    verified._token = _VERIFIED_TASK0257_TOKEN
    verified.training_chain = training_chain
    verified.ordered_rows = ordered_rows
    verified.parent_review_plan = copy.deepcopy(payload_for(paths.parent_review_plan))
    verified.harwood_review_plan = copy.deepcopy(payload_for(paths.harwood_review_plan))
    verified.source_groups = copy.deepcopy(payload_for(paths.source_groups))
    verified.old_embeddings = tuple(copy.deepcopy(payload_for(path)) for path in paths.old_embedding_files)
    verified.replayed_nested_probe = copy.deepcopy(replayed_probe)
    verified.validated_input_paths = resolved_paths
    verified.validated_leaf_identities = tuple(identities[path] for path in resolved_paths)
    verified.validated_leaf_receipts = tuple(copy.deepcopy(receipt) for _path, receipt in pairs)
    verified.expected_receipts = copy.deepcopy(expected_receipts)
    _register_task0257_capability(verified)
    return verified


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("artifact_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _read_bounded_bytes(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("bounded input must be a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError("bounded input exceeds the size limit")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                raise ValueError("bounded input changed during the read")
            chunks.append(chunk)
            remaining -= len(chunk)
        final_metadata = os.fstat(descriptor)
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        final_identity = (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_mode,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
        )
        if final_identity != identity:
            raise ValueError("bounded input changed during the read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _read_bounded_json(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    encoded = _read_bounded_bytes(path, max_bytes=max_bytes)
    try:
        decoded = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JSON input is invalid") from error
    if not isinstance(decoded, dict):
        raise ValueError("JSON root must be an object")
    return decoded, encoded


def _spec_file_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    if len(paths) != len(APPROVED_SPEC_FILENAMES):
        raise ValueError("approved specification file order is invalid")
    rows: list[dict[str, object]] = []
    for expected_filename, raw_path in zip(APPROVED_SPEC_FILENAMES, paths, strict=True):
        path = Path(raw_path)
        if path.name != expected_filename or not path.is_file():
            raise ValueError("approved specification file order is invalid")
        encoded = path.read_bytes()
        rows.append(
            {
                "filename": expected_filename,
                "file_sha256": _sha256_bytes(encoded),
                "size_bytes": len(encoded),
            }
        )
    return rows


def _verify_temporal_feature_plan_payload(payload: Mapping[str, object]) -> None:
    if set(payload) != _PLAN_FIELDS:
        raise ValueError("temporal feature plan fields are invalid")
    if payload.get("schema_version") != TEMPORAL_FEATURE_PLAN_SCHEMA:
        raise ValueError("temporal feature plan schema is invalid")
    if payload.get("module_id") != MODULE_A_ID:
        raise ValueError("temporal feature plan module is invalid")
    if payload.get("purpose") != "development_diagnostic_only":
        raise ValueError("temporal feature plan purpose is invalid")
    if payload.get("label_hidden") is not True:
        raise ValueError("temporal feature plan must be label hidden")
    for field in (
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    ):
        if payload.get(field) is not False:
            raise ValueError("temporal feature plan eligibility flags are invalid")
    artifact_sha256 = payload.get("artifact_sha256")
    if not _is_sha256(artifact_sha256) or artifact_sha256 != _canonical_sha256(payload):
        raise ValueError("temporal feature plan artifact SHA-256 is invalid")
    representation = payload.get("representation")
    if (
        not isinstance(representation, Mapping)
        or set(representation) != _PLAN_REPRESENTATION_FIELDS
        or representation != TEMPORAL_REPRESENTATION_CONTRACT
    ):
        raise ValueError("temporal feature plan representation is invalid")
    _verify_task0257_receipt_mapping(payload.get("task0257_receipts"))
    if payload.get("evaluation_protocol") != TEMPORAL_EVALUATION_PROTOCOL:
        raise ValueError("temporal feature plan evaluation protocol is invalid")
    if payload.get("resource_policy") != TEMPORAL_RESOURCE_POLICY:
        raise ValueError("temporal feature plan resource policy is invalid")
    if payload.get("environment_contract") != TEMPORAL_ENVIRONMENT_CONTRACT:
        raise ValueError("temporal feature plan environment contract is invalid")
    examples = payload.get("ordered_examples")
    if not isinstance(examples, list) or len(examples) != 45:
        raise ValueError("temporal feature plan examples are invalid")
    expected_games = tuple(
        game_id
        for game_id, row_count in zip(
            TEMPORAL_EVALUATION_PROTOCOL["game_order"],
            TEMPORAL_EVALUATION_PROTOCOL["game_row_counts"],
            strict=True,
        )
        for _ in range(row_count)
    )
    expected_families = {"hazen": "hctv", "randolph": "hctv", "vtv": "vtv", "harwood": "hctv"}
    game_source_sha256s: dict[str, str] = {}
    game_bundle_sha256s: dict[str, str] = {}
    previous_event_id_by_game: dict[str, str] = {}
    seen_keys: set[tuple[str, str, str]] = set()
    receipts = payload["task0257_receipts"]
    source_video_receipts = receipts["source_videos"]
    expected_review_receipts = {
        "hazen": receipts["parent_review_plan"]["internal_sha256"],
        "randolph": receipts["parent_review_plan"]["internal_sha256"],
        "vtv": receipts["parent_review_plan"]["internal_sha256"],
        "harwood": receipts["harwood_review_plan"]["internal_sha256"],
    }
    for ordinal, (row, expected_game) in enumerate(zip(examples, expected_games, strict=True)):
        if not isinstance(row, Mapping) or set(row) != _PLAN_EXAMPLE_FIELDS:
            raise ValueError("temporal feature plan example fields are invalid")
        if row.get("ordinal") != ordinal or type(row.get("ordinal")) is not int:
            raise ValueError("temporal feature plan example ordinal is invalid")
        source_sha256 = row.get("source_video_sha256")
        bundle_sha256 = row.get("candidate_bundle_sha256")
        event_id = row.get("event_id")
        if (
            not _is_sha256(source_sha256)
            or not _is_sha256(bundle_sha256)
            or not isinstance(event_id, str)
            or not event_id
            or row.get("game_id") != expected_game
            or row.get("production_family") != expected_families[expected_game]
            or row.get("review_plan_receipt") != expected_review_receipts[expected_game]
        ):
            raise ValueError("temporal feature plan example identity is invalid")
        game_source_sha256s.setdefault(expected_game, source_sha256)
        game_bundle_sha256s.setdefault(expected_game, bundle_sha256)
        if (
            game_source_sha256s[expected_game] != source_sha256
            or game_bundle_sha256s[expected_game] != bundle_sha256
            or source_video_receipts[TEMPORAL_EVALUATION_PROTOCOL["game_order"].index(expected_game)]["file_sha256"]
            != source_sha256
            or event_id <= previous_event_id_by_game.get(expected_game, "")
        ):
            raise ValueError("temporal feature plan game binding or order is invalid")
        previous_event_id_by_game[expected_game] = event_id
        key = (source_sha256, bundle_sha256, event_id)
        if key in seen_keys:
            raise ValueError("temporal feature plan example key is duplicated")
        seen_keys.add(key)
        tiles = row.get("tile_frame_indexes")
        if (
            not isinstance(tiles, list)
            or len(tiles) != 4
            or any(
                not isinstance(tile, list)
                or len(tile) != 16
                or any(type(value) is not int or value < 0 for value in tile)
                for tile in tiles
            )
        ):
            raise ValueError("temporal feature plan tile indexes are invalid")


def _verify_file_receipt_mapping(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"file_sha256", "filename", "size_bytes"}:
        raise ValueError("TASK-0257 file receipt fields are invalid")
    if (
        not _is_sha256(value.get("file_sha256"))
        or not isinstance(value.get("filename"), str)
        or not value["filename"]
        or Path(value["filename"]).name != value["filename"]
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] < 0
    ):
        raise ValueError("TASK-0257 file receipt values are invalid")


def _verify_stored_receipt_mapping(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "internal_sha256_field",
        "internal_sha256",
        "file_sha256",
        "filename",
        "size_bytes",
    }:
        raise ValueError("TASK-0257 stored receipt fields are invalid")
    _verify_file_receipt_mapping({field: value[field] for field in ("file_sha256", "filename", "size_bytes")})
    if (
        not isinstance(value.get("schema_version"), str)
        or not value["schema_version"]
        or not isinstance(value.get("internal_sha256_field"), str)
        or re.fullmatch(r"[a-z][a-z0-9_]*", value["internal_sha256_field"]) is None
        or not _is_sha256(value.get("internal_sha256"))
    ):
        raise ValueError("TASK-0257 stored receipt values are invalid")


def _verify_task0257_receipt_mapping(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(Task0257ExpectedReceipts.__dataclass_fields__):
        raise ValueError("TASK-0257 receipt graph fields are invalid")
    stored_sequences = {"parent_candidate_children": 3, "old_embedding_files": 2}
    file_sequences = {
        "parent_label_children": 3,
        "parent_review_jpegs": 1536,
        "harwood_review_jpegs": 1536,
        "source_videos": 4,
        "checkpoints": 2,
    }
    file_scalars = {"parent_source_manifest", "v2_label_child", "harwood_source_manifest"}
    for field in Task0257ExpectedReceipts.__dataclass_fields__:
        item = value[field]
        if field in stored_sequences:
            if not isinstance(item, list) or len(item) != stored_sequences[field]:
                raise ValueError("TASK-0257 stored receipt sequence is invalid")
            for row in item:
                _verify_stored_receipt_mapping(row)
        elif field in file_sequences:
            if not isinstance(item, list) or len(item) != file_sequences[field]:
                raise ValueError("TASK-0257 file receipt sequence is invalid")
            for row in item:
                _verify_file_receipt_mapping(row)
        elif field in file_scalars:
            _verify_file_receipt_mapping(item)
        else:
            _verify_stored_receipt_mapping(item)
    if value["checkpoints"][1]["file_sha256"] != TEMPORAL_REPRESENTATION_CONTRACT["checkpoint_sha256"]:
        raise ValueError("TASK-0257 Swin checkpoint receipt is invalid")


def _require_verified_task0257_inputs(inputs: VerifiedTask0257TemporalInputs) -> None:
    if (
        type(inputs) is not VerifiedTask0257TemporalInputs
        or getattr(inputs, "_token", None) is not _VERIFIED_TASK0257_TOKEN
    ):
        raise TypeError("verified TASK-0257 temporal inputs are required")
    registered = _REGISTERED_TASK0257_CAPABILITIES.get(id(inputs))
    if registered is None or registered[0] is not inputs:
        raise TypeError("verified TASK-0257 temporal inputs are required")
    _registered, expected_fingerprint = registered
    if _task0257_capability_fingerprint(inputs) != expected_fingerprint:
        raise ValueError("verified TASK-0257 temporal inputs were mutated")
    if not (
        len(inputs.validated_input_paths)
        == len(inputs.validated_leaf_identities)
        == len(inputs.validated_leaf_receipts)
    ):
        raise ValueError("verified TASK-0257 leaf capability is incomplete")
    for path, identity, receipt in zip(
        inputs.validated_input_paths,
        inputs.validated_leaf_identities,
        inputs.validated_leaf_receipts,
        strict=True,
    ):
        metadata = path.stat(follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != identity:
            raise ValueError("verified TASK-0257 leaf identity changed")
        if not isinstance(receipt, (StoredArtifactReceipt, FileReceipt)):
            raise ValueError("verified TASK-0257 leaf receipt type changed")


def seal_vru_causal_temporal_feature_plan(
    *,
    inputs: VerifiedTask0257TemporalInputs,
    environment_contract: Mapping[str, object],
) -> dict[str, object]:
    """Seal the exact 45-row label-hidden plan from verified TASK-0257 inputs."""
    _require_verified_task0257_inputs(inputs)
    if dict(environment_contract) != TEMPORAL_ENVIRONMENT_CONTRACT:
        raise ValueError("temporal environment does not match the approved contract")
    projected = build_temporal_feature_examples(
        ordered_rows=inputs.ordered_rows,
        parent_review_plan=inputs.parent_review_plan,
        harwood_review_plan=inputs.harwood_review_plan,
    )
    source_groups = inputs.source_groups.get("games")
    if not isinstance(source_groups, list):
        raise ValueError("TASK-0257 source groups are invalid")
    games_by_source = {
        row.get("source_video_sha256"): (row.get("game_id"), row.get("production_family"))
        for row in source_groups
        if isinstance(row, Mapping)
    }
    ordered_examples: list[dict[str, object]] = []
    for ordinal, row in enumerate(projected):
        source_sha256 = row["source_video_sha256"]
        try:
            game_id, production_family = games_by_source[source_sha256]
        except KeyError as error:
            raise ValueError("temporal example source is absent from source groups") from error
        ordered_examples.append(
            {
                "ordinal": ordinal,
                "game_id": game_id,
                "production_family": production_family,
                "source_video_sha256": source_sha256,
                "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                "event_id": row["event_id"],
                "review_plan_receipt": row["review_plan_artifact_sha256"],
                "tile_frame_indexes": copy.deepcopy(row["tile_frame_indexes"]),
            }
        )
    receipt_mapping = json.loads(json.dumps(asdict(inputs.expected_receipts), allow_nan=False))
    payload: dict[str, object] = {
        "schema_version": TEMPORAL_FEATURE_PLAN_SCHEMA,
        "module_id": MODULE_A_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "label_hidden": True,
        "representation": copy.deepcopy(TEMPORAL_REPRESENTATION_CONTRACT),
        "task0257_receipts": receipt_mapping,
        "ordered_examples": ordered_examples,
        "evaluation_protocol": copy.deepcopy(TEMPORAL_EVALUATION_PROTOCOL),
        "resource_policy": copy.deepcopy(TEMPORAL_RESOURCE_POLICY),
        "environment_contract": copy.deepcopy(TEMPORAL_ENVIRONMENT_CONTRACT),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    _verify_temporal_feature_plan_payload(payload)
    return payload


def load_verified_temporal_feature_plan(
    *,
    plan_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
) -> VerifiedTemporalFeaturePlan:
    """Load a plan only through caller-frozen internal and file receipts."""
    if not _is_sha256(expected_artifact_sha256) or not _is_sha256(expected_file_sha256):
        raise ValueError("temporal feature plan expected SHA-256 is invalid")
    raw_path = Path(plan_path)
    if raw_path.is_symlink():
        raise ValueError("temporal feature plan path must not be a symlink")
    path = raw_path.resolve(strict=True)
    payload, encoded = _read_bounded_json(path, max_bytes=67_108_864)
    if _sha256_bytes(encoded) != expected_file_sha256:
        raise ValueError("temporal feature plan file SHA-256 does not match")
    if encoded != _canonical_json_bytes(payload):
        raise ValueError("temporal feature plan bytes are not canonical JSON plus LF")
    _verify_temporal_feature_plan_payload(payload)
    if payload.get("artifact_sha256") != expected_artifact_sha256:
        raise ValueError("temporal feature plan artifact SHA-256 does not match")
    metadata = path.stat(follow_symlinks=False)
    verified = VerifiedTemporalFeaturePlan(_VERIFIED_PLAN_TOKEN)
    verified._token = _VERIFIED_PLAN_TOKEN
    verified._payload = copy.deepcopy(payload)
    verified._path = path
    verified._device = metadata.st_dev
    verified._inode = metadata.st_ino
    verified._size_bytes = metadata.st_size
    verified._mtime_ns = metadata.st_mtime_ns
    verified._artifact_sha256 = expected_artifact_sha256
    verified._file_sha256 = expected_file_sha256
    _register_capability(_REGISTERED_PLAN_CAPABILITIES, verified)
    return verified


def _require_verified_plan(plan: VerifiedTemporalFeaturePlan) -> None:
    if type(plan) is not VerifiedTemporalFeaturePlan or getattr(plan, "_token", None) is not _VERIFIED_PLAN_TOKEN:
        raise TypeError("a verified temporal feature plan capability is required")
    _require_registered_capability(
        _REGISTERED_PLAN_CAPABILITIES,
        plan,
        "a verified temporal feature plan capability is required",
    )
    metadata = plan._path.stat(follow_symlinks=False)
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    if identity != (
        plan._device,
        plan._inode,
        plan._size_bytes,
        plan._mtime_ns,
    ):
        raise ValueError("verified temporal feature plan identity changed")
    payload, encoded = _read_bounded_json(plan._path, max_bytes=67_108_864)
    if _sha256_bytes(encoded) != plan._file_sha256 or payload.get("artifact_sha256") != plan._artifact_sha256:
        raise ValueError("verified temporal feature plan bytes changed")
    _verify_temporal_feature_plan_payload(payload)
    if plan._payload != payload:
        raise ValueError("verified temporal feature plan local snapshot changed")


def _verify_tiled_embedding_payload(payload: Mapping[str, object], *, plan: VerifiedTemporalFeaturePlan) -> None:
    if set(payload) != _EMBEDDING_FIELDS:
        raise ValueError("tiled embedding artifact fields are invalid")
    if payload.get("schema_version") != TILED_SWIN_EMBEDDING_SCHEMA:
        raise ValueError("tiled embedding artifact schema is invalid")
    if payload.get("module_id") != MODULE_A_ID:
        raise ValueError("tiled embedding module is invalid")
    if payload.get("purpose") != "development_diagnostic_only":
        raise ValueError("tiled embedding artifact purpose is invalid")
    for field in (
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    ):
        if payload.get(field) is not False:
            raise ValueError("tiled embedding eligibility flags are invalid")
    artifact_sha256 = payload.get("artifact_sha256")
    if not _is_sha256(artifact_sha256) or artifact_sha256 != _canonical_sha256(payload):
        raise ValueError("tiled embedding artifact SHA-256 is invalid")
    if payload.get("plan_receipt") != {
        "artifact_sha256": plan._artifact_sha256,
        "file_sha256": plan._file_sha256,
    }:
        raise ValueError("tiled embedding plan receipt is invalid")
    if payload.get("representation") != plan._payload.get("representation"):
        raise ValueError("tiled embedding representation is invalid")
    if payload.get("task0257_input_receipts") != plan._payload.get("task0257_receipts"):
        raise ValueError("tiled embedding input receipts are invalid")
    if payload.get("producer_environment") != plan._payload.get("environment_contract"):
        raise ValueError("tiled embedding producer environment is invalid")
    _verify_attempt_chain_receipts(payload.get("attempt_chain"))
    examples = payload.get("examples")
    plan_examples = plan._payload.get("ordered_examples")
    if (
        not isinstance(examples, list)
        or not isinstance(plan_examples, list)
        or payload.get("row_count") != len(examples)
        or type(payload.get("row_count")) is not int
        or len(examples) != len(plan_examples)
    ):
        raise ValueError("tiled embedding row coverage is invalid")
    for ordinal, (row, plan_row) in enumerate(zip(examples, plan_examples, strict=True)):
        if not isinstance(row, Mapping) or set(row) != _EMBEDDING_EXAMPLE_FIELDS:
            raise ValueError("tiled embedding example fields are invalid")
        key = row.get("key")
        if not isinstance(key, Mapping) or set(key) != _EMBEDDING_KEY_FIELDS:
            raise ValueError("tiled embedding example key is invalid")
        expected_key = {
            "source_video_sha256": plan_row["source_video_sha256"],
            "candidate_bundle_sha256": plan_row["candidate_bundle_sha256"],
            "event_id": plan_row["event_id"],
        }
        if (
            row.get("ordinal") != ordinal
            or type(row.get("ordinal")) is not int
            or key != expected_key
            or row.get("tile_frame_indexes") != plan_row["tile_frame_indexes"]
        ):
            raise ValueError("tiled embedding example does not match the plan")
        tile_embeddings = row.get("derivation_only_tile_embeddings")
        model_input = row.get("model_input")
        if not isinstance(tile_embeddings, list) or not isinstance(model_input, list):
            raise ValueError("tiled embedding example vectors are invalid")
        if (
            len(tile_embeddings) != TILED_SWIN_TILE_COUNT
            or any(
                not isinstance(tile, list)
                or len(tile) != TILED_SWIN_DIMENSION
                or any(type(value) is not float or not np.isfinite(value) for value in tile)
                for tile in tile_embeddings
            )
            or len(model_input) != TILED_SWIN_OUTPUT_DIMENSION
            or any(type(value) is not float or not np.isfinite(value) for value in model_input)
        ):
            raise ValueError("tiled embedding vectors must be exact finite floats")
        expected_model_input = combine_tiled_swin_embeddings(tile_embeddings)
        stored = np.asarray(model_input, dtype=np.float32)
        expected = np.asarray(expected_model_input, dtype=np.float32)
        if (
            stored.shape != (TILED_SWIN_OUTPUT_DIMENSION,)
            or not np.isfinite(stored).all()
            or not np.array_equal(stored, expected)
        ):
            raise ValueError("tiled embedding model input formula does not match")


def load_verified_tiled_swin_embeddings(
    *,
    embeddings_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    plan: VerifiedTemporalFeaturePlan,
) -> VerifiedTiledSwinEmbeddings:
    """Load tiled embeddings through their plan and two external receipts."""
    _require_verified_plan(plan)
    if not _is_sha256(expected_artifact_sha256) or not _is_sha256(expected_file_sha256):
        raise ValueError("tiled embedding expected SHA-256 is invalid")
    raw_path = Path(embeddings_path)
    if raw_path.is_symlink():
        raise ValueError("tiled embedding path must not be a symlink")
    path = raw_path.resolve(strict=True)
    payload, encoded = _read_bounded_json(path, max_bytes=67_108_864)
    if _sha256_bytes(encoded) != expected_file_sha256:
        raise ValueError("tiled embedding file SHA-256 does not match")
    if encoded != _canonical_json_bytes(payload):
        raise ValueError("tiled embedding bytes are not canonical JSON plus LF")
    _verify_tiled_embedding_payload(payload, plan=plan)
    if payload.get("artifact_sha256") != expected_artifact_sha256:
        raise ValueError("tiled embedding artifact SHA-256 does not match")
    metadata = path.stat(follow_symlinks=False)
    verified = VerifiedTiledSwinEmbeddings(_VERIFIED_EMBEDDING_TOKEN)
    verified._token = _VERIFIED_EMBEDDING_TOKEN
    verified._payload = copy.deepcopy(payload)
    verified._path = path
    verified._device = metadata.st_dev
    verified._inode = metadata.st_ino
    verified._size_bytes = metadata.st_size
    verified._mtime_ns = metadata.st_mtime_ns
    verified._artifact_sha256 = expected_artifact_sha256
    verified._file_sha256 = expected_file_sha256
    verified._plan_artifact_sha256 = plan._artifact_sha256
    _register_capability(_REGISTERED_EMBEDDING_CAPABILITIES, verified)
    return verified


def _require_verified_tiled_swin_embeddings(
    embeddings: VerifiedTiledSwinEmbeddings,
    *,
    plan: VerifiedTemporalFeaturePlan,
) -> dict[str, object]:
    _require_verified_plan(plan)
    if (
        type(embeddings) is not VerifiedTiledSwinEmbeddings
        or getattr(embeddings, "_token", None) is not _VERIFIED_EMBEDDING_TOKEN
    ):
        raise TypeError("verified tiled Swin embeddings are required")
    _require_registered_capability(
        _REGISTERED_EMBEDDING_CAPABILITIES,
        embeddings,
        "verified tiled Swin embeddings are required",
    )
    metadata = embeddings._path.stat(follow_symlinks=False)
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ) != (
        embeddings._device,
        embeddings._inode,
        embeddings._size_bytes,
        embeddings._mtime_ns,
    ):
        raise ValueError("verified tiled embedding identity changed")
    payload, encoded = _read_bounded_json(embeddings._path, max_bytes=67_108_864)
    if (
        _sha256_bytes(encoded) != embeddings._file_sha256
        or payload.get("artifact_sha256") != embeddings._artifact_sha256
        or embeddings._plan_artifact_sha256 != plan._artifact_sha256
    ):
        raise ValueError("verified tiled embedding bytes changed")
    _verify_tiled_embedding_payload(payload, plan=plan)
    if embeddings._payload != payload:
        raise ValueError("verified tiled embedding local snapshot changed")
    return copy.deepcopy(payload)


def _build_fresh_tiled_swin_embeddings(
    *,
    plan: VerifiedTemporalFeaturePlan,
    tile_embeddings: Sequence[Sequence[Sequence[float]]],
    attempt_chain: Sequence[Mapping[str, object]],
    resource_summary: Mapping[str, object] | None = None,
) -> FreshlyProducedTiledSwinEmbeddings:
    _require_verified_plan(plan)
    plan_examples = plan._payload["ordered_examples"]
    if len(tile_embeddings) != len(plan_examples):
        raise ValueError("fresh tiled embeddings do not exactly cover the plan")
    examples: list[dict[str, object]] = []
    for row, plan_row in zip(tile_embeddings, plan_examples, strict=True):
        if len(row) != TILED_SWIN_TILE_COUNT or any(
            len(tile) != TILED_SWIN_DIMENSION
            or any(type(value) is not float or not np.isfinite(value) for value in tile)
            for tile in row
        ):
            raise ValueError("fresh tiled embeddings must be exact finite float vectors")
        normalized_tiles = [[float(np.float32(value)) for value in tile] for tile in row]
        model_input = combine_tiled_swin_embeddings(normalized_tiles)
        examples.append(
            {
                "ordinal": plan_row["ordinal"],
                "key": {
                    "source_video_sha256": plan_row["source_video_sha256"],
                    "candidate_bundle_sha256": plan_row["candidate_bundle_sha256"],
                    "event_id": plan_row["event_id"],
                },
                "tile_frame_indexes": copy.deepcopy(plan_row["tile_frame_indexes"]),
                "derivation_only_tile_embeddings": normalized_tiles,
                "model_input": model_input,
            }
        )
    payload: dict[str, object] = {
        "schema_version": TILED_SWIN_EMBEDDING_SCHEMA,
        "module_id": MODULE_A_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "plan_receipt": {
            "artifact_sha256": plan._artifact_sha256,
            "file_sha256": plan._file_sha256,
        },
        "task0257_input_receipts": copy.deepcopy(plan._payload["task0257_receipts"]),
        "representation": copy.deepcopy(plan._payload["representation"]),
        "producer_environment": copy.deepcopy(plan._payload["environment_contract"]),
        "attempt_chain": [copy.deepcopy(dict(row)) for row in attempt_chain],
        "row_count": len(examples),
        "examples": examples,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    _verify_tiled_embedding_payload(payload, plan=plan)
    fresh = FreshlyProducedTiledSwinEmbeddings(_FRESH_EMBEDDING_TOKEN)
    fresh._token = _FRESH_EMBEDDING_TOKEN
    fresh._payload = payload
    fresh._file_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
    fresh._resource_summary = copy.deepcopy(
        dict(resource_summary)
        if resource_summary is not None
        else {
            "attempt_count": len(attempt_chain),
            "cumulative_active_runtime_nanoseconds": 0,
            "cumulative_resource_samples": 0,
            "cumulative_resource_log_bytes": 0,
            "all_sustained_limits_passed": True,
            "process_tree_reaped": True,
        }
    )
    _register_fresh_embedding_capability(fresh)
    return fresh


def _read_checkpoint_snapshot(
    path: Path,
    *,
    expected_file_sha256: str,
    maximum_bytes: int = 209_715_200,
) -> bytes:
    if not _is_sha256(expected_file_sha256) or type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("checkpoint snapshot receipt is invalid")
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError("checkpoint path must not be a symlink")
    resolved = raw.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_bytes:
            raise ValueError("checkpoint must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                raise ValueError("checkpoint changed during snapshot")
            chunks.append(chunk)
            remaining -= len(chunk)
        final = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            raise ValueError("checkpoint changed during snapshot")
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    if _sha256_bytes(snapshot) != expected_file_sha256:
        raise ValueError("checkpoint file SHA-256 does not match")
    return snapshot


def _load_swin_backbone_from_verified_descriptor(
    checkpoint_path: Path,
    *,
    expected_file_sha256: str,
    expected_size_bytes: int,
    device: Any,
) -> tuple[Any, Any]:
    """Hash and load the Swin state dict through one no-follow descriptor."""
    import torch
    from torch import nn
    from torchvision.models.video import Swin3D_T_Weights, swin3d_t

    if (
        not _is_sha256(expected_file_sha256)
        or type(expected_size_bytes) is not int
        or not 0 < expected_size_bytes <= 209_715_200
    ):
        raise ValueError("Swin checkpoint receipt is invalid")
    raw_path = Path(checkpoint_path)
    if raw_path.is_symlink():
        raise ValueError("Swin checkpoint path must not be a symlink")
    path = raw_path.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        initial = os.fstat(descriptor)
        identity = (
            initial.st_dev,
            initial.st_ino,
            initial.st_mode,
            initial.st_size,
            initial.st_mtime_ns,
        )
        if not stat.S_ISREG(initial.st_mode) or initial.st_size != expected_size_bytes:
            raise ValueError("Swin checkpoint size or type is invalid")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1_048_576):
            digest.update(chunk)
        if digest.hexdigest() != expected_file_sha256:
            raise ValueError("Swin checkpoint SHA-256 does not match")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            try:
                state = torch.load(handle, map_location="cpu", weights_only=True)
            finally:
                final = os.fstat(descriptor)
                if (
                    final.st_dev,
                    final.st_ino,
                    final.st_mode,
                    final.st_size,
                    final.st_mtime_ns,
                ) != identity:
                    raise ValueError("Swin checkpoint identity changed during model load")
        if not isinstance(state, Mapping):
            raise ValueError("Swin checkpoint must contain a state dict")
    finally:
        os.close(descriptor)
    model = swin3d_t(weights=None, progress=False)
    model.load_state_dict(state)
    model.head = nn.Identity()
    transform = Swin3D_T_Weights.KINETICS400_V1.transforms()
    return model.to(device).eval(), transform


def _file_sha256_no_follow(path: Path) -> tuple[str, tuple[int, int, int, int]]:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError("source video path must not be a symlink")
    resolved = raw.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("source video must be a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1_048_576)
            if not chunk:
                break
            digest.update(chunk)
        final = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
        if identity != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            raise ValueError("source video changed during hash")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), identity


def _decode_exact_frame_tiles(capture: Any, tile_frame_indexes: Sequence[Sequence[int]]) -> list[np.ndarray]:
    import cv2

    if len(tile_frame_indexes) != 4 or any(len(tile) != 16 for tile in tile_frame_indexes):
        raise ValueError("decoder requires exactly four 16-frame tiles")
    flattened = [index for tile in tile_frame_indexes for index in tile]
    if any(type(index) is not int or index < 0 for index in flattened) or any(
        left >= right for left, right in zip(flattened, flattened[1:])
    ):
        raise ValueError("decoder frame indexes are invalid")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError("decoder source dimensions are invalid")
    frames: dict[int, np.ndarray] = {}
    for frame_index in flattened:
        if capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index) is False:
            raise ValueError(f"cannot seek source frame {frame_index}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode source frame {frame_index}")
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != (height, width, 3)
            or not frame.flags.c_contiguous
        ):
            raise ValueError("decoded frame violates the frozen pixel contract")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3) or not rgb.flags.c_contiguous:
            raise ValueError("converted frame violates the frozen pixel contract")
        frames[frame_index] = rgb
    return [np.ascontiguousarray(np.stack([frames[index] for index in tile])) for tile in tile_frame_indexes]


def _infer_exact_tile_embeddings(
    *,
    frame_tiles: Sequence[np.ndarray],
    backbone: Any,
    transform: Any,
    device: Any,
) -> list[list[float]]:
    import torch

    if len(frame_tiles) != 4:
        raise ValueError("inference requires exactly four frame tiles")
    output: list[list[float]] = []
    for frames in frame_tiles:
        if not isinstance(frames, np.ndarray) or frames.shape[0] != 16 or frames.ndim != 4:
            raise ValueError("frame tile shape is invalid")
        clip = torch.from_numpy(frames).permute(0, 3, 1, 2)
        batch = transform(clip).unsqueeze(0).to(device)
        values = None
        cpu_values = None
        try:
            with torch.inference_mode():
                values = backbone(batch)
            cpu_values = values.detach().cpu().to(torch.float32)
            if tuple(cpu_values.shape) != (1, TILED_SWIN_DIMENSION):
                raise ValueError("Swin tile embedding dimension is invalid")
            row = [float(value) for value in cpu_values[0].tolist()]
            if any(type(value) is not float or not np.isfinite(value) for value in row):
                raise ValueError("Swin tile embedding is non-finite")
            output.append(row)
        finally:
            batch = None
            values = None
            cpu_values = None
            if getattr(device, "type", None) == "mps":
                torch.mps.empty_cache()
    return output


def _extract_tiled_swin_rows(
    *,
    plan: VerifiedTemporalFeaturePlan,
    source_video_paths: Sequence[Path],
    checkpoint_path: Path,
    row_callback: Any | None = None,
    initial_rows: Sequence[Sequence[Sequence[float]]] = (),
) -> list[list[list[float]]]:
    """Extract the one approved representation with one capture per source."""
    import cv2
    import torch

    _require_verified_plan(plan)
    observe_temporal_environment_contract()
    if len(source_video_paths) != 4 or any(not isinstance(path, Path) for path in source_video_paths):
        raise ValueError("exactly four source video paths are required")
    source_receipts = plan._payload["task0257_receipts"]["source_videos"]
    checkpoint_receipt = plan._payload["task0257_receipts"]["checkpoints"][1]
    if (
        Path(checkpoint_path).name != checkpoint_receipt["filename"]
        or Path(checkpoint_path).stat(follow_symlinks=False).st_size != checkpoint_receipt["size_bytes"]
    ):
        raise ValueError("Swin checkpoint filename or size is invalid")
    resolved_sources: list[Path] = []
    pre_hashes: list[str] = []
    pre_identities: list[tuple[int, int, int, int]] = []
    seen_identities: set[tuple[int, int]] = set()
    for path, receipt in zip(source_video_paths, source_receipts, strict=True):
        if path.is_symlink():
            raise ValueError("source video must not be a symlink")
        resolved = path.resolve(strict=True)
        digest, identity = _file_sha256_no_follow(resolved)
        if (
            digest != receipt["file_sha256"]
            or identity[2] != receipt["size_bytes"]
            or (identity[0], identity[1]) in seen_identities
        ):
            raise ValueError("source video receipt or identity is invalid")
        seen_identities.add((identity[0], identity[1]))
        resolved_sources.append(resolved)
        pre_hashes.append(digest)
        pre_identities.append(identity)
    device = torch.device("mps:0")
    if not torch.backends.mps.is_available():
        raise ValueError("approved MPS device is unavailable")
    backbone, transform = _load_swin_backbone_from_verified_descriptor(
        checkpoint_path,
        expected_file_sha256=checkpoint_receipt["file_sha256"],
        expected_size_bytes=checkpoint_receipt["size_bytes"],
        device=device,
    )
    examples_by_source: dict[str, list[Mapping[str, object]]] = {
        receipt["file_sha256"]: [] for receipt in source_receipts
    }
    for row in plan._payload["ordered_examples"]:
        examples_by_source[row["source_video_sha256"]].append(row)
    if len(initial_rows) >= 45:
        raise ValueError("initial tiled embedding prefix must be shorter than the plan")
    extracted: list[list[list[float]] | None] = [None] * 45
    for ordinal, row in enumerate(initial_rows):
        if len(row) != TILED_SWIN_TILE_COUNT or any(
            len(tile) != TILED_SWIN_DIMENSION
            or any(type(value) is not float or not np.isfinite(value) for value in tile)
            for tile in row
        ):
            raise ValueError("initial tiled embedding prefix is invalid")
        extracted[ordinal] = [list(tile) for tile in row]
    try:
        for source_path, source_receipt in zip(resolved_sources, source_receipts, strict=True):
            capture = cv2.VideoCapture(str(source_path), cv2.CAP_FFMPEG)
            if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_BACKEND)) != cv2.CAP_FFMPEG:
                capture.release()
                raise ValueError("source video did not open with the frozen FFmpeg backend")
            try:
                for row in examples_by_source[source_receipt["file_sha256"]]:
                    if row["ordinal"] < len(initial_rows):
                        continue
                    frame_tiles = _decode_exact_frame_tiles(capture, row["tile_frame_indexes"])
                    tile_values = _infer_exact_tile_embeddings(
                        frame_tiles=frame_tiles,
                        backbone=backbone,
                        transform=transform,
                        device=device,
                    )
                    ordinal = row["ordinal"]
                    extracted[ordinal] = tile_values
                    if row_callback is not None:
                        row_callback(ordinal, copy.deepcopy(tile_values))
            finally:
                capture.release()
    finally:
        backbone = None
        transform = None
        torch.mps.empty_cache()
        for path, expected_hash, expected_identity in zip(
            resolved_sources,
            pre_hashes,
            pre_identities,
            strict=True,
        ):
            digest, identity = _file_sha256_no_follow(path)
            if digest != expected_hash or identity != expected_identity:
                raise ValueError("source video changed during extraction")
    if any(row is None for row in extracted):
        raise ValueError("tiled extraction did not cover all 45 rows")
    return [row for row in extracted if row is not None]


def seal_module_a_spec_approval(
    *,
    approved_spec_paths: Sequence[Path],
    fresh_review_internal_sha256: str,
    fresh_review_file_sha256: str,
    approval_statement: str,
    approved_at_utc: str,
) -> dict[str, object]:
    """Seal the exact user-approved Module-A specification tuple."""
    if not _is_sha256(fresh_review_internal_sha256) or not _is_sha256(fresh_review_file_sha256):
        raise ValueError("fresh review receipt SHA-256 is invalid")
    if not isinstance(approval_statement, str) or not approval_statement:
        raise ValueError("approval statement is required")
    if not isinstance(approved_at_utc, str) or not approved_at_utc.endswith("Z") or "T" not in approved_at_utc:
        raise ValueError("approval timestamp must be UTC RFC 3339 text")
    payload: dict[str, object] = {
        "schema_version": MODULE_A_APPROVAL_SCHEMA,
        "module_id": MODULE_A_ID,
        "approved_files": _spec_file_rows(approved_spec_paths),
        "fresh_review_receipt": {
            "internal_sha256": fresh_review_internal_sha256,
            "file_sha256": fresh_review_file_sha256,
        },
        "approval_scope": MODULE_A_APPROVAL_SCOPE,
        "approval_statement_sha256": _sha256_bytes(approval_statement.encode("utf-8")),
        "approved_at_utc": approved_at_utc,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def verify_module_a_spec_approval(
    *,
    approval_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    approved_spec_paths: Sequence[Path],
    expected_fresh_review_internal_sha256: str,
    expected_fresh_review_file_sha256: str,
    expected_approval_statement_sha256: str,
) -> dict[str, object]:
    """Verify approval using only caller-supplied expected receipts."""
    expected_values = (
        expected_artifact_sha256,
        expected_file_sha256,
        expected_fresh_review_internal_sha256,
        expected_fresh_review_file_sha256,
        expected_approval_statement_sha256,
    )
    if not all(_is_sha256(value) for value in expected_values):
        raise ValueError("external approval artifact SHA-256 receipt is invalid")
    payload, encoded = _read_bounded_json(Path(approval_path), max_bytes=_MAX_APPROVAL_BYTES)
    if _sha256_bytes(encoded) != expected_file_sha256:
        raise ValueError("external approval artifact file SHA-256 does not match")
    if set(payload) != _APPROVAL_FIELDS:
        raise ValueError("approval artifact fields are invalid")
    if payload.get("schema_version") != MODULE_A_APPROVAL_SCHEMA:
        raise ValueError("approval artifact schema is invalid")
    if payload.get("module_id") != MODULE_A_ID:
        raise ValueError("approval artifact module is invalid")
    if payload.get("approval_scope") != MODULE_A_APPROVAL_SCOPE:
        raise ValueError("approval artifact scope is invalid")
    if payload.get("artifact_sha256") != expected_artifact_sha256:
        raise ValueError("external approval artifact SHA-256 does not match")
    if payload.get("artifact_sha256") != _canonical_sha256(payload):
        raise ValueError("approval artifact canonical SHA-256 does not match")
    if payload.get("approval_statement_sha256") != expected_approval_statement_sha256:
        raise ValueError("approval statement SHA-256 does not match")
    approved_at_utc = payload.get("approved_at_utc")
    if not isinstance(approved_at_utc, str) or not approved_at_utc.endswith("Z") or "T" not in approved_at_utc:
        raise ValueError("approval timestamp is invalid")
    fresh_review = payload.get("fresh_review_receipt")
    if not isinstance(fresh_review, dict) or set(fresh_review) != _FRESH_REVIEW_FIELDS:
        raise ValueError("fresh review receipt fields are invalid")
    if fresh_review != {
        "internal_sha256": expected_fresh_review_internal_sha256,
        "file_sha256": expected_fresh_review_file_sha256,
    }:
        raise ValueError("fresh review receipt does not match external values")
    approved_files = payload.get("approved_files")
    if not isinstance(approved_files, list) or any(
        not isinstance(row, dict) or set(row) != _APPROVED_FILE_FIELDS for row in approved_files
    ):
        raise ValueError("approved specification receipt fields are invalid")
    if approved_files != _spec_file_rows(approved_spec_paths):
        raise ValueError("approved specification bytes do not match the approval")
    return copy.deepcopy(payload)


def combine_tiled_swin_embeddings(
    tile_embeddings: Sequence[Sequence[float]],
) -> list[float]:
    """Combine four ordered 768-d Swin tiles using the frozen mean/delta rule."""
    if len(tile_embeddings) != TILED_SWIN_TILE_COUNT:
        raise ValueError("exactly four tile embeddings are required")
    matrix = np.asarray(tile_embeddings, dtype=np.float32)
    if matrix.shape != (TILED_SWIN_TILE_COUNT, TILED_SWIN_DIMENSION):
        raise ValueError("tile embeddings must have shape (4, 768)")
    if not np.isfinite(matrix).all():
        raise ValueError("tile embeddings must be finite")
    overall = np.mean(matrix, axis=0, dtype=np.float32)
    early = np.mean(matrix[:2], axis=0, dtype=np.float32)
    late = np.mean(matrix[2:], axis=0, dtype=np.float32)
    combined = np.concatenate((overall, late - early)).astype(np.float32, copy=False)
    if combined.shape != (TILED_SWIN_OUTPUT_DIMENSION,) or not np.isfinite(combined).all():
        raise ValueError("combined tiled embedding is invalid")
    return [float(value) for value in combined]


def build_temporal_feature_examples(
    *,
    ordered_rows: Sequence[Mapping[str, object]],
    parent_review_plan: Mapping[str, object],
    harwood_review_plan: Mapping[str, object],
) -> list[dict[str, object]]:
    """Project label-hidden 4x16 sampling from authoritative review indexes."""
    plan_examples: dict[tuple[str, str], tuple[str, list[int]]] = {}
    for plan in (parent_review_plan, harwood_review_plan):
        plan_sha256 = plan.get("artifact_sha256")
        examples = plan.get("examples")
        if not _is_sha256(plan_sha256) or not isinstance(examples, list):
            raise ValueError("review plan structure is invalid")
        for example in examples:
            if not isinstance(example, Mapping):
                raise ValueError("review plan example is invalid")
            source_sha256 = example.get("source_video_sha256")
            review_id = example.get("review_id")
            frame_indexes = example.get("frame_indexes")
            if (
                not _is_sha256(source_sha256)
                or not isinstance(review_id, str)
                or not review_id
                or not isinstance(frame_indexes, list)
                or len(frame_indexes) != 64
                or any(type(value) is not int or value < 0 for value in frame_indexes)
                or any(left >= right for left, right in zip(frame_indexes, frame_indexes[1:]))
            ):
                raise ValueError("review plan frame indexes are invalid")
            key = (source_sha256, review_id)
            if key in plan_examples:
                raise ValueError("review plan key is duplicated")
            plan_examples[key] = (plan_sha256, list(frame_indexes))
    output: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for row in ordered_rows:
        source_sha256 = row.get("source_video_sha256")
        bundle_sha256 = row.get("candidate_bundle_sha256")
        event_id = row.get("event_id")
        if (
            not _is_sha256(source_sha256)
            or not _is_sha256(bundle_sha256)
            or not isinstance(event_id, str)
            or not event_id
        ):
            raise ValueError("ordered temporal row key is invalid")
        key = (source_sha256, bundle_sha256, event_id)
        if key in seen_keys:
            raise ValueError("ordered temporal row key is duplicated")
        seen_keys.add(key)
        plan_match = plan_examples.get((source_sha256, event_id))
        if plan_match is None:
            raise ValueError("ordered temporal row is missing from both review plans")
        plan_sha256, frame_indexes = plan_match
        output.append(
            {
                "source_video_sha256": source_sha256,
                "candidate_bundle_sha256": bundle_sha256,
                "event_id": event_id,
                "review_plan_artifact_sha256": plan_sha256,
                "frame_indexes": frame_indexes,
                "tile_frame_indexes": [frame_indexes[offset : offset + 16] for offset in range(0, 64, 16)],
            }
        )
    return output


def _binary_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, object]:
    raw_truth = np.asarray(labels)
    if raw_truth.dtype != np.bool_:
        raise ValueError("final evaluator labels must be exact booleans")
    truth = raw_truth.astype(np.bool_, copy=False)
    predicted = np.asarray(decisions, dtype=np.bool_)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("metric inputs must be aligned one-dimensional arrays")
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
        "f1": float(f1),
    }


def _fit_probability_model(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[StandardScaler, LogisticRegression]:
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    if matrix.ndim != 2 or truth.ndim != 1 or matrix.shape[0] != truth.shape[0]:
        raise ValueError("training features and labels are not aligned")
    if not np.isfinite(matrix).all() or np.unique(truth).size != 2:
        raise ValueError("every fit requires finite features and both classes")
    with threadpool_limits(limits=1):
        scaler = StandardScaler()
        observe_temporal_environment_contract()
        transformed = scaler.fit_transform(matrix)
        observe_temporal_environment_contract()
        classifier = LogisticRegression(
            C=REGULARIZATION_C,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        )
        observe_temporal_environment_contract()
        classifier.fit(transformed, truth.astype(np.int64))
        observe_temporal_environment_contract()
    return scaler, classifier


def _probabilities(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("prediction features must be a finite matrix")
    values = classifier.predict_proba(scaler.transform(matrix))[:, 1]
    if not np.isfinite(values).all():
        raise ValueError("model probabilities must be finite")
    return values


def _model_state(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    labels: np.ndarray,
) -> dict[str, object]:
    classes = [int(value) for value in classifier.classes_]
    sample_count = int(scaler.n_samples_seen_)
    truth = np.asarray(labels, dtype=np.bool_)
    counts = [int(np.count_nonzero(~truth)), int(np.count_nonzero(truth))]
    if sum(counts) != sample_count or any(count <= 0 for count in counts):
        raise ValueError("model-state labels do not match the fitted sample count")
    return {
        "feature_dimension": int(scaler.mean_.shape[0]),
        "sample_count": sample_count,
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_variance": [float(value) for value in scaler.var_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "scaler_n_samples_seen": sample_count,
        "scaler_parameters": copy.deepcopy(scaler.get_params(deep=False)),
        "classes": classes,
        "effective_balanced_class_weights": [sample_count / (2 * count) for count in counts],
        "coefficients": [[float(value) for value in row] for row in classifier.coef_],
        "intercept": [float(value) for value in classifier.intercept_],
        "iteration_count": [int(value) for value in classifier.n_iter_],
        "classifier_parameters": copy.deepcopy(classifier.get_params(deep=False)),
        "regularization_c": REGULARIZATION_C,
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": 0,
        "solver": "lbfgs",
        "library_versions": {
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
        },
    }


def screen_temporal_outer_fold(
    *,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    game_ids: Sequence[str],
    held_game_id: str,
    representation_name: str,
) -> dict[str, object]:
    """Run one frozen nested outer-game-held diagnostic fold."""
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    groups = tuple(game_ids)
    if (
        matrix.ndim != 2
        or truth.ndim != 1
        or matrix.shape[0] != truth.shape[0]
        or len(groups) != truth.shape[0]
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("fold inputs are not aligned finite arrays")
    game_order = tuple(dict.fromkeys(groups))
    if held_game_id not in game_order or len(game_order) != 4:
        raise ValueError("outer fold requires exactly four games")
    fit_game_ids = tuple(game for game in game_order if game != held_game_id)
    inner_probabilities = np.empty(truth.shape[0], dtype=np.float64)
    inner_mask = np.zeros(truth.shape[0], dtype=np.bool_)
    for inner_held in fit_game_ids:
        train_mask = np.asarray(
            [game in fit_game_ids and game != inner_held for game in groups],
            dtype=np.bool_,
        )
        validation_mask = np.asarray([game == inner_held for game in groups], dtype=np.bool_)
        if not train_mask.any() or not validation_mask.any():
            raise ValueError("inner LOGO split is empty")
        scaler, classifier = _fit_probability_model(matrix[train_mask], truth[train_mask])
        inner_probabilities[validation_mask] = _probabilities(scaler, classifier, matrix[validation_mask])
        inner_mask |= validation_mask
    expected_inner_mask = np.asarray([game in fit_game_ids for game in groups], dtype=np.bool_)
    if not np.array_equal(inner_mask, expected_inner_mask):
        raise ValueError("inner LOGO predictions do not exactly cover outer training")
    threshold_candidates: list[dict[str, object]] = []
    for threshold in THRESHOLD_GRID:
        metrics = _binary_metrics(truth[inner_mask], inner_probabilities[inner_mask] >= threshold)
        threshold_candidates.append({"threshold": threshold, **metrics})
    selected = max(
        threshold_candidates,
        key=lambda row: (
            row["balanced_accuracy"],
            row["f1"],
            row["precision"],
            row["recall"],
            row["threshold"],
        ),
    )
    selected_threshold = float(selected["threshold"])
    outer_train_mask = expected_inner_mask
    held_mask = np.asarray([game == held_game_id for game in groups], dtype=np.bool_)
    scaler, classifier = _fit_probability_model(matrix[outer_train_mask], truth[outer_train_mask])
    held_probabilities = _probabilities(scaler, classifier, matrix[held_mask])
    held_decisions = held_probabilities >= selected_threshold
    held_indexes = np.flatnonzero(held_mask)
    return {
        "representation_name": representation_name,
        "held_game_id": held_game_id,
        "fit_game_ids": list(fit_game_ids),
        "inner_selection_game_ids": list(fit_game_ids),
        "inner_oof_predictions": [
            {
                "row_index": int(index),
                "probability": float(inner_probabilities[index]),
            }
            for index in np.flatnonzero(inner_mask)
        ],
        "threshold_candidates": threshold_candidates,
        "selected_threshold": selected_threshold,
        "outer_fit_state": _model_state(scaler, classifier, truth[outer_train_mask]),
        "held_predictions": [
            {
                "row_index": int(index),
                "probability": float(probability),
                "decision": bool(decision),
                "truth": bool(truth[index]),
            }
            for index, probability, decision in zip(held_indexes, held_probabilities, held_decisions, strict=True)
        ],
        "held_metrics": _binary_metrics(truth[held_mask], held_decisions),
    }


def _aggregate_prediction_metrics(
    *,
    predictions: Sequence[Mapping[str, object]],
    game_ids: Sequence[str],
) -> dict[str, object]:
    ordered = sorted(predictions, key=lambda row: row["row_index"])
    truth = np.asarray([row["truth"] for row in ordered], dtype=np.bool_)
    decisions = np.asarray([row["decision"] for row in ordered], dtype=np.bool_)
    pooled = _binary_metrics(truth, decisions)
    per_game: list[dict[str, object]] = []
    for game_id in TEMPORAL_EVALUATION_PROTOCOL["game_order"]:
        mask = np.asarray([game_ids[row["row_index"]] == game_id for row in ordered], dtype=np.bool_)
        per_game.append({"game_id": game_id, **_binary_metrics(truth[mask], decisions[mask])})
    families = {"hctv": ("hazen", "randolph", "harwood"), "vtv": ("vtv",)}
    per_family: list[dict[str, object]] = []
    for family, games in families.items():
        mask = np.asarray([game_ids[row["row_index"]] in games for row in ordered], dtype=np.bool_)
        per_family.append(
            {
                "production_family": family,
                "descriptive_only": True,
                **_binary_metrics(truth[mask], decisions[mask]),
            }
        )
    return {"pooled": pooled, "per_game": per_game, "per_production_family": per_family}


def build_temporal_retrospective(
    *,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    game_ids: Sequence[str],
    ordered_keys: Sequence[Mapping[str, object]],
    input_receipts: Mapping[str, object],
    archived_task0257_comparator: Mapping[str, object],
) -> dict[str, object]:
    """Build the four-game nested retrospective diagnostic and frozen bound decision."""
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    groups = tuple(game_ids)
    if (
        matrix.ndim != 2
        or matrix.shape != (45, TILED_SWIN_OUTPUT_DIMENSION)
        or truth.shape != (45,)
        or len(groups) != 45
        or len(ordered_keys) != 45
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("temporal retrospective inputs are invalid")
    expected_groups = tuple(
        game
        for game, count in zip(
            TEMPORAL_EVALUATION_PROTOCOL["game_order"],
            TEMPORAL_EVALUATION_PROTOCOL["game_row_counts"],
            strict=True,
        )
        for _ in range(count)
    )
    if groups != expected_groups:
        raise ValueError("temporal retrospective game order is invalid")
    outer_folds: list[dict[str, object]] = []
    pooled_predictions: list[dict[str, object]] = []
    family_by_game = {"hazen": "hctv", "randolph": "hctv", "vtv": "vtv", "harwood": "hctv"}
    source_by_game: dict[str, str] = {}
    for index, (key, game_id) in enumerate(zip(ordered_keys, groups, strict=True)):
        if not isinstance(key, Mapping) or set(key) != _EMBEDDING_KEY_FIELDS:
            raise ValueError("retrospective ordered key is invalid")
        source = key.get("source_video_sha256")
        if not _is_sha256(source):
            raise ValueError("retrospective source binding is invalid")
        source_by_game.setdefault(game_id, source)
        if source_by_game[game_id] != source:
            raise ValueError("retrospective game contains multiple sources")
        if index and key == ordered_keys[index - 1]:
            raise ValueError("retrospective ordered key is duplicated")
    for held_game_id in TEMPORAL_EVALUATION_PROTOCOL["game_order"]:
        fold = screen_temporal_outer_fold(
            feature_matrix=matrix,
            labels=truth,
            game_ids=groups,
            held_game_id=held_game_id,
            representation_name=TEMPORAL_REPRESENTATION_CONTRACT["name"],
        )
        fold["held_source_video_sha256"] = source_by_game[held_game_id]
        fold["held_production_family"] = family_by_game[held_game_id]
        for row in fold["inner_oof_predictions"]:
            row["key"] = copy.deepcopy(dict(ordered_keys[row["row_index"]]))
        for row in fold["held_predictions"]:
            row["key"] = copy.deepcopy(dict(ordered_keys[row["row_index"]]))
            pooled_predictions.append(copy.deepcopy(row))
        outer_folds.append(fold)
    observed_metrics = _aggregate_prediction_metrics(predictions=pooled_predictions, game_ids=groups)
    bounds = {
        "hazen": {"maximum_errors": 0, "maximum_fp": None, "maximum_fn": None},
        "randolph": {"maximum_errors": 1, "maximum_fp": None, "maximum_fn": None},
        "vtv": {"maximum_errors": 2, "maximum_fp": None, "maximum_fn": None},
        "harwood": {"maximum_errors": 3, "maximum_fp": 2, "maximum_fn": 1},
    }
    error_rows: list[dict[str, object]] = []
    for metrics in observed_metrics["per_game"]:
        game_id = metrics["game_id"]
        bound = bounds[game_id]
        errors = metrics["fp"] + metrics["fn"]
        passed = errors <= bound["maximum_errors"]
        if bound["maximum_fp"] is not None:
            passed = passed and metrics["fp"] <= bound["maximum_fp"]
        if bound["maximum_fn"] is not None:
            passed = passed and metrics["fn"] <= bound["maximum_fn"]
        error_rows.append(
            {
                "game_id": game_id,
                **bound,
                "observed_fp": metrics["fp"],
                "observed_fn": metrics["fn"],
                "observed_errors": errors,
                "passed": bool(passed),
            }
        )
    within_bounds = all(row["passed"] for row in error_rows) and sum(row["observed_errors"] for row in error_rows) <= 6
    artifact: dict[str, object] = {
        "schema_version": TEMPORAL_RETROSPECTIVE_SCHEMA,
        "module_id": MODULE_A_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "input_receipts": copy.deepcopy(dict(input_receipts)),
        "selection_protocol": "nested_outer_game_held_inner_logo_v1",
        "source_selection_limitation": {
            "parent_rows": "22_prior-label-stratified_development_rows",
            "harwood_rows": "23_geometry-only_label-hidden_development_rows",
            "production_family_count": 2,
            "independent_or_formal": False,
        },
        "outer_folds": outer_folds,
        "observed_metrics": observed_metrics,
        "archived_task0257_comparator": copy.deepcopy(dict(archived_task0257_comparator)),
        "error_bound_result": {"per_game": error_rows, "pooled_maximum_errors": 6, "passed": within_bounds},
        "temporal_hypothesis_decision": (
            "within_frozen_error_bounds" if within_bounds else "temporal-hypothesis-rejected"
        ),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def _logo_variant(
    *,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    game_ids: tuple[str, ...],
    representation_name: str,
) -> dict[str, object]:
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    if (
        matrix.ndim != 2
        or truth.ndim != 1
        or matrix.shape[0] != truth.shape[0]
        or len(game_ids) != truth.shape[0]
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("LOGO inputs are not aligned finite arrays")
    game_order = tuple(dict.fromkeys(game_ids))
    if len(game_order) != 4:
        raise ValueError("final evaluator requires exactly four games")
    oof_probabilities = np.empty(truth.shape[0], dtype=np.float64)
    coverage = np.zeros(truth.shape[0], dtype=np.bool_)
    folds: list[dict[str, object]] = []
    for held_game_id in game_order:
        held_mask = np.asarray([game == held_game_id for game in game_ids], dtype=np.bool_)
        train_mask = ~held_mask
        scaler, classifier = _fit_probability_model(matrix[train_mask], truth[train_mask])
        probabilities = _probabilities(scaler, classifier, matrix[held_mask])
        oof_probabilities[held_mask] = probabilities
        coverage |= held_mask
        folds.append(
            {
                "held_game_id": held_game_id,
                "fit_game_ids": [game for game in game_order if game != held_game_id],
                "fit_state": _model_state(scaler, classifier, truth[train_mask]),
                "predictions": [
                    {
                        "row_index": int(index),
                        "probability": float(probability),
                    }
                    for index, probability in zip(np.flatnonzero(held_mask), probabilities, strict=True)
                ],
            }
        )
    if not coverage.all():
        raise ValueError("LOGO predictions do not cover every row exactly once")
    threshold_candidates: list[dict[str, object]] = []
    for threshold in THRESHOLD_GRID:
        metrics = _binary_metrics(truth, oof_probabilities >= threshold)
        threshold_candidates.append({"threshold": threshold, **metrics})
    selected = max(
        threshold_candidates,
        key=lambda row: (
            row["balanced_accuracy"],
            row["f1"],
            row["precision"],
            row["recall"],
            row["threshold"],
        ),
    )
    selected_threshold = float(selected["threshold"])
    return {
        "representation_name": representation_name,
        "folds": folds,
        "threshold_candidates": threshold_candidates,
        "selected_threshold": selected_threshold,
        "selected_metrics": copy.deepcopy(selected),
        "oof_predictions": [
            {
                "row_index": index,
                "probability": float(probability),
                "decision": bool(probability >= selected_threshold),
            }
            for index, probability in enumerate(oof_probabilities)
        ],
    }


def build_final_evaluator(
    *,
    feature_matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    game_ids: Sequence[str],
    representation_order: Sequence[str],
    evaluator_kind: str,
) -> dict[str, object]:
    """Build a diagnostic final evaluator from fresh four-game LOGO OOF."""
    if evaluator_kind not in {"baseline", "candidate"}:
        raise ValueError("evaluator kind is invalid")
    ordered_names = tuple(representation_order)
    if not ordered_names or set(feature_matrices) != set(ordered_names):
        raise ValueError("representation matrices must exactly match their frozen order")
    if evaluator_kind == "baseline" and ordered_names != (
        "swin3d_t",
        "mvit_v2_s+swin3d_t",
    ):
        raise ValueError("baseline representation order is invalid")
    candidate_name = TEMPORAL_REPRESENTATION_CONTRACT["name"]
    if evaluator_kind == "candidate" and ordered_names != (candidate_name,):
        raise ValueError("candidate representation order is invalid")
    raw_truth = np.asarray(labels)
    if raw_truth.dtype != np.bool_:
        raise ValueError("final evaluator labels must be exact booleans")
    truth = raw_truth.astype(np.bool_, copy=False)
    groups = tuple(game_ids)
    expected_groups = tuple(
        game
        for game, count in zip(
            TEMPORAL_EVALUATION_PROTOCOL["game_order"],
            TEMPORAL_EVALUATION_PROTOCOL["game_row_counts"],
            strict=True,
        )
        for _ in range(count)
    )
    if groups != expected_groups or truth.shape != (45,):
        raise ValueError("final evaluator row and game contract is invalid")
    expected_dimensions = {
        "swin3d_t": 768,
        "mvit_v2_s+swin3d_t": 1536,
        candidate_name: 1536,
    }
    for name in ordered_names:
        matrix = np.asarray(feature_matrices[name])
        if matrix.shape != (45, expected_dimensions[name]) or not np.issubdtype(matrix.dtype, np.floating):
            raise ValueError("final evaluator feature dimension or dtype is invalid")
    variants = [
        _logo_variant(
            feature_matrix=feature_matrices[name],
            labels=truth,
            game_ids=groups,
            representation_name=name,
        )
        for name in ordered_names
    ]
    selected = max(
        enumerate(variants),
        key=lambda item: (
            item[1]["selected_metrics"]["balanced_accuracy"],
            item[1]["selected_metrics"]["f1"],
            item[1]["selected_metrics"]["precision"],
            item[1]["selected_metrics"]["recall"],
            -item[0],
            item[1]["selected_threshold"],
        ),
    )[1]
    selected_name = str(selected["representation_name"])
    selected_matrix = np.asarray(feature_matrices[selected_name], dtype=np.float64)
    scaler, classifier = _fit_probability_model(selected_matrix, truth)
    return {
        "evaluator_kind": evaluator_kind,
        "logo_selection": variants,
        "selected_evaluator": copy.deepcopy(selected),
        "all_45_refit": {
            "representation_name": selected_name,
            "row_count": int(truth.shape[0]),
            "fit_state": _model_state(scaler, classifier, truth),
        },
    }


def build_sealed_final_evaluator(
    *,
    feature_matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    game_ids: Sequence[str],
    representation_order: Sequence[str],
    evaluator_role: str,
    ordered_training_rows: Sequence[Mapping[str, object]],
    input_receipts: Mapping[str, object],
) -> dict[str, object]:
    """Seal one independently replayable baseline or candidate evaluator."""
    raw_truth = np.asarray(labels)
    if raw_truth.dtype != np.bool_:
        raise ValueError("final evaluator labels must be exact booleans")
    truth = raw_truth.astype(np.bool_, copy=False)
    if truth.shape != (45,) or len(ordered_training_rows) != 45:
        raise ValueError("final evaluator requires exactly 45 ordered labels")
    normalized_rows: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for index, (row, target) in enumerate(zip(ordered_training_rows, truth, strict=True)):
        if not isinstance(row, Mapping) or set(row) != _EMBEDDING_KEY_FIELDS:
            raise ValueError("final evaluator training key fields are invalid")
        key = (
            row.get("source_video_sha256"),
            row.get("candidate_bundle_sha256"),
            row.get("event_id"),
        )
        if not _is_sha256(key[0]) or not _is_sha256(key[1]) or not isinstance(key[2], str) or not key[2]:
            raise ValueError("final evaluator training key is invalid")
        if key in seen_keys:
            raise ValueError("final evaluator training key is duplicated")
        seen_keys.add(key)
        normalized_rows.append(
            {
                "ordinal": index,
                **copy.deepcopy(dict(row)),
                "event_present": bool(target),
            }
        )
    computed = build_final_evaluator(
        feature_matrices=feature_matrices,
        labels=truth,
        game_ids=game_ids,
        representation_order=representation_order,
        evaluator_kind=evaluator_role,
    )
    selected = computed["selected_evaluator"]
    artifact: dict[str, object] = {
        "schema_version": FINAL_EVALUATOR_SCHEMA,
        "module_id": MODULE_A_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "evaluator_role": evaluator_role,
        "input_receipts": copy.deepcopy(dict(input_receipts)),
        "representation_candidates": list(representation_order),
        "ordered_training_rows": normalized_rows,
        "logo_selection": computed["logo_selection"],
        "selected_evaluator": {
            "representation_name": selected["representation_name"],
            "threshold": selected["selected_threshold"],
            "metrics": copy.deepcopy(selected["selected_metrics"]),
        },
        "all_45_refit": computed["all_45_refit"],
        "conditional_downstream": {
            "module_b_reference_allowed_only_after_mechanical_pass": True,
            "module_b_spec_and_user_approval_still_required": True,
            "runtime_or_training_authority_granted": False,
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_final_evaluator_computation(
    payload: Mapping[str, object],
    *,
    feature_matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    game_ids: Sequence[str],
    ordered_training_rows: Sequence[Mapping[str, object]],
    input_receipts: Mapping[str, object],
) -> dict[str, object]:
    """Freshly rerun every LOGO fit, selection and all-row refit."""
    if payload.get("schema_version") != FINAL_EVALUATOR_SCHEMA:
        raise ValueError("final evaluator schema is invalid")
    role = payload.get("evaluator_role")
    if role == "baseline":
        order = ("swin3d_t", "mvit_v2_s+swin3d_t")
    elif role == "candidate":
        order = (TEMPORAL_REPRESENTATION_CONTRACT["name"],)
    else:
        raise ValueError("final evaluator role is invalid")
    expected = build_sealed_final_evaluator(
        feature_matrices=feature_matrices,
        labels=labels,
        game_ids=game_ids,
        representation_order=order,
        evaluator_role=role,
        ordered_training_rows=ordered_training_rows,
        input_receipts=input_receipts,
    )
    if dict(payload) != expected:
        raise ValueError("final evaluator does not reproduce its frozen computation")
    return copy.deepcopy(expected)


def _aligned_temporal_feature_inputs(
    *,
    inputs: VerifiedTask0257TemporalInputs,
    tiled_embeddings: FreshlyProducedTiledSwinEmbeddings | VerifiedTiledSwinEmbeddings,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...], list[dict[str, object]]]:
    _require_verified_task0257_inputs(inputs)
    if type(tiled_embeddings) is FreshlyProducedTiledSwinEmbeddings:
        _require_fresh_embedding_capability(tiled_embeddings)
        tiled_payload = tiled_embeddings._payload
    elif type(tiled_embeddings) is VerifiedTiledSwinEmbeddings:
        _require_registered_capability(
            _REGISTERED_EMBEDDING_CAPABILITIES,
            tiled_embeddings,
            "verified tiled embedding capability is invalid",
        )
        tiled_payload = tiled_embeddings._payload
    else:
        raise TypeError("a transaction-local or externally verified tiled embedding capability is required")
    labels: list[bool] = []
    ordered_keys: list[dict[str, object]] = []
    for row in inputs.ordered_rows:
        target = row.get("event_present")
        key = {
            "source_video_sha256": row.get("source_video_sha256"),
            "candidate_bundle_sha256": row.get("candidate_bundle_sha256"),
            "event_id": row.get("event_id"),
        }
        if (
            type(target) is not bool
            or not _is_sha256(key["source_video_sha256"])
            or not _is_sha256(key["candidate_bundle_sha256"])
            or not isinstance(key["event_id"], str)
        ):
            raise ValueError("verified TASK-0257 training row is invalid")
        labels.append(target)
        ordered_keys.append(key)
    source_games = {row["source_video_sha256"]: row["game_id"] for row in inputs.source_groups["games"]}
    game_ids = tuple(source_games[key["source_video_sha256"]] for key in ordered_keys)
    old_matrices: dict[str, np.ndarray] = {}
    expected_backbones = (
        "torchvision/mvit_v2_s/kinetics400_v1",
        "torchvision/swin3d_t/kinetics400_v1",
    )
    for artifact, expected_backbone in zip(inputs.old_embeddings, expected_backbones, strict=True):
        if artifact.get("backbone") != expected_backbone or artifact.get("embedding_dimension") != 768:
            raise ValueError("TASK-0257 embedding backbone contract is invalid")
        examples = artifact.get("examples")
        if not isinstance(examples, list) or len(examples) != 45:
            raise ValueError("TASK-0257 embedding row coverage is invalid")
        vectors: list[list[float]] = []
        for index, (row, expected_key, expected_target) in enumerate(zip(examples, ordered_keys, labels, strict=True)):
            if not isinstance(row, Mapping):
                raise ValueError("TASK-0257 embedding row is invalid")
            actual_key = {field: row.get(field) for field in _EMBEDDING_KEY_FIELDS}
            values = row.get("embedding")
            if (
                actual_key != expected_key
                or row.get("event_present") is not expected_target
                or not isinstance(values, list)
                or len(values) != 768
                or any(
                    type(value) not in (int, float) or isinstance(value, bool) or not np.isfinite(value)
                    for value in values
                )
            ):
                raise ValueError(f"TASK-0257 embedding row {index} is not aligned")
            vectors.append([float(value) for value in values])
        old_matrices[expected_backbone] = np.asarray(vectors, dtype=np.float64)
    tiled_rows = tiled_payload.get("examples")
    if not isinstance(tiled_rows, list) or len(tiled_rows) != 45:
        raise ValueError("tiled embedding row coverage is invalid")
    tiled_values: list[list[float]] = []
    for row, expected_key in zip(tiled_rows, ordered_keys, strict=True):
        if not isinstance(row, Mapping) or row.get("key") != expected_key:
            raise ValueError("tiled embedding row is not aligned")
        values = row.get("model_input")
        if not isinstance(values, list) or len(values) != 1536:
            raise ValueError("tiled model input is invalid")
        tiled_values.append([float(value) for value in values])
    mvit = old_matrices[expected_backbones[0]]
    swin = old_matrices[expected_backbones[1]]
    return (
        {
            "swin3d_t": swin,
            "mvit_v2_s+swin3d_t": np.concatenate((mvit, swin), axis=1),
            TEMPORAL_REPRESENTATION_CONTRACT["name"]: np.asarray(tiled_values, dtype=np.float64),
        },
        np.asarray(labels, dtype=np.bool_),
        game_ids,
        ordered_keys,
    )


def _build_vru_causal_temporal_final_generation(
    *,
    inputs: VerifiedTask0257TemporalInputs,
    plan: VerifiedTemporalFeaturePlan,
    tiled_embeddings: FreshlyProducedTiledSwinEmbeddings,
) -> ModuleAFinalGeneration:
    """Build the four mutually bound terminal records in one transaction."""
    _require_verified_plan(plan)
    _require_fresh_embedding_capability(tiled_embeddings)
    _verify_tiled_embedding_payload(tiled_embeddings._payload, plan=plan)
    features, labels, game_ids, ordered_keys = _aligned_temporal_feature_inputs(
        inputs=inputs,
        tiled_embeddings=tiled_embeddings,
    )
    computation_input_receipts = {
        "plan": {
            "artifact_sha256": plan._artifact_sha256,
            "file_sha256": plan._file_sha256,
        },
        "task0257": copy.deepcopy(plan._payload["task0257_receipts"]),
        "tiled_embeddings": {
            "artifact_sha256": tiled_embeddings._payload["artifact_sha256"],
            "file_sha256": tiled_embeddings._file_sha256,
        },
    }
    archived_comparator = {
        "receipt": asdict(inputs.expected_receipts.old_nested_probe),
        "selection_protocol": inputs.replayed_nested_probe.get("selection_protocol"),
        "row_count": inputs.replayed_nested_probe.get("row_count"),
        "observed_metrics": copy.deepcopy(inputs.replayed_nested_probe.get("observed_metrics")),
        "excluded_from_candidate_selection": True,
    }
    retrospective = build_temporal_retrospective(
        feature_matrix=features[TEMPORAL_REPRESENTATION_CONTRACT["name"]],
        labels=labels,
        game_ids=game_ids,
        ordered_keys=ordered_keys,
        input_receipts=computation_input_receipts,
        archived_task0257_comparator=archived_comparator,
    )
    baseline = build_sealed_final_evaluator(
        feature_matrices={
            "swin3d_t": features["swin3d_t"],
            "mvit_v2_s+swin3d_t": features["mvit_v2_s+swin3d_t"],
        },
        labels=labels,
        game_ids=game_ids,
        representation_order=("swin3d_t", "mvit_v2_s+swin3d_t"),
        evaluator_role="baseline",
        ordered_training_rows=ordered_keys,
        input_receipts=computation_input_receipts,
    )
    candidate = build_sealed_final_evaluator(
        feature_matrices={TEMPORAL_REPRESENTATION_CONTRACT["name"]: features[TEMPORAL_REPRESENTATION_CONTRACT["name"]]},
        labels=labels,
        game_ids=game_ids,
        representation_order=(TEMPORAL_REPRESENTATION_CONTRACT["name"],),
        evaluator_role="candidate",
        ordered_training_rows=ordered_keys,
        input_receipts=computation_input_receipts,
    )
    resource_summary = copy.deepcopy(tiled_embeddings._resource_summary)
    resource_passed = (
        resource_summary.get("all_sustained_limits_passed") is True
        and resource_summary.get("process_tree_reaped") is True
    )
    hypothesis_passed = retrospective["error_bound_result"]["passed"] is True
    checks = [
        {"check": "task0258_plan_receipts", "passed": True},
        {"check": "task0257_full_replay", "passed": True},
        {"check": "tiled_embedding_formula_and_coverage", "passed": True},
        {"check": "retrospective_recomputed", "passed": True},
        {"check": "baseline_final_evaluator_recomputed", "passed": True},
        {"check": "candidate_final_evaluator_recomputed", "passed": True},
        {"check": "resource_and_process_contract", "passed": resource_passed},
        {"check": "two_empty_state_receipt_bound_replays", "passed": False},
        {"check": "held_label_invariance", "passed": False},
        {"check": "per_fit_backend_pre_and_post", "passed": False},
        {"check": "disk_budget_revalidation", "passed": False},
        {"check": "atomic_publication_and_no_residue", "passed": False},
        {"check": "complete_resume_chain_external_receipts", "passed": False},
    ]
    mandatory_evidence_passed = all(row["passed"] is True for row in checks)
    if not resource_passed:
        decision = "mechanical_failure"
        stop_reason = "resource_breach"
    elif not mandatory_evidence_passed:
        decision = "mechanical_failure"
        stop_reason = "unverified_mechanical_evidence"
    elif hypothesis_passed:
        decision = "mechanical_pass"
        stop_reason = None
    else:
        decision = "temporal-hypothesis-rejected"
        stop_reason = "temporal-hypothesis-rejected"
    mechanical_gate: dict[str, object] = {
        "schema_version": MECHANICAL_GATE_SCHEMA,
        "module_id": MODULE_A_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "input_receipts": {
            "plan": {
                "provider": "temporal_feature_plan",
                "verification_state": "verified",
                "receipt": copy.deepcopy(computation_input_receipts["plan"]),
            },
            "task0257": {
                "provider": "task0257_temporal_inputs",
                "verification_state": "verified",
                "receipt": copy.deepcopy(computation_input_receipts["task0257"]),
            },
            "tiled_embeddings": {
                "provider": "tiled_swin_embeddings",
                "verification_state": "verified",
                "receipt": copy.deepcopy(computation_input_receipts["tiled_embeddings"]),
            },
            "baseline_final_evaluator": {
                "provider": "baseline_final_evaluator",
                "verification_state": "verified",
                "receipt": {
                    "artifact_sha256": baseline["artifact_sha256"],
                    "file_sha256": _sha256_bytes(_canonical_json_bytes(baseline)),
                },
            },
            "candidate_final_evaluator": {
                "provider": "candidate_final_evaluator",
                "verification_state": "verified",
                "receipt": {
                    "artifact_sha256": candidate["artifact_sha256"],
                    "file_sha256": _sha256_bytes(_canonical_json_bytes(candidate)),
                },
            },
        },
        "attempt_chain": copy.deepcopy(tiled_embeddings._payload["attempt_chain"]),
        "ordered_check_results": checks,
        "error_bound_result": copy.deepcopy(retrospective["error_bound_result"]),
        "resource_summary": resource_summary,
        "publication_summary": {
            "final_generation_published": True,
            "atomic_directory_publication_required": True,
        },
        "decision": decision,
        "stop_reason": stop_reason,
        "conditional_downstream": {
            "module_b_permitted_only_for_mechanical_pass": decision == "mechanical_pass",
            "module_b_spec_and_user_approval_still_required": True,
            "formal_or_runtime_authority_granted": False,
        },
    }
    mechanical_gate["artifact_sha256"] = _canonical_sha256(mechanical_gate)
    return ModuleAFinalGeneration(
        retrospective=retrospective,
        baseline_evaluator=baseline,
        candidate_evaluator=candidate,
        mechanical_gate=mechanical_gate,
    )


_FINAL_GENERATION_FILE_KEYS = (
    "terminal_attempt/resource_guard.jsonl",
    "terminal_attempt/attempt_record.json",
    "tiled_swin_embeddings.json",
    "temporal_retrospective.json",
    "baseline_final_evaluator.json",
    "candidate_final_evaluator.json",
    "mechanical_gate.json",
)
_FINAL_GENERATION_ARTIFACT_KEYS = tuple(
    key for key in _FINAL_GENERATION_FILE_KEYS if not key.endswith("resource_guard.jsonl")
)
FINAL_GENERATION_RECEIPT_REGISTRY_SCHEMA = "agu.vru-causal-temporal-final-generation-receipts.v1"
_MECHANICAL_RECEIPT_SLOT_NAMES = (
    "plan",
    "task0257",
    "tiled_embeddings",
    "baseline_final_evaluator",
    "candidate_final_evaluator",
)
_MANDATORY_MECHANICAL_EVIDENCE_CHECKS = (
    "two_empty_state_receipt_bound_replays",
    "held_label_invariance",
    "per_fit_backend_pre_and_post",
    "disk_budget_revalidation",
    "atomic_publication_and_no_residue",
    "complete_resume_chain_external_receipts",
)


def _verify_mechanical_receipt_slots(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(_MECHANICAL_RECEIPT_SLOT_NAMES):
        raise ValueError("mechanical gate provider-slot coverage is invalid")
    result: dict[str, object] = {}
    for name in _MECHANICAL_RECEIPT_SLOT_NAMES:
        slot = value[name]
        if not isinstance(slot, Mapping) or set(slot) != {"provider", "verification_state", "receipt"}:
            raise ValueError("mechanical gate provider slot fields are invalid")
        state = slot.get("verification_state")
        receipt = slot.get("receipt")
        if (
            slot.get("provider")
            != {
                "plan": "temporal_feature_plan",
                "task0257": "task0257_temporal_inputs",
                "tiled_embeddings": "tiled_swin_embeddings",
                "baseline_final_evaluator": "baseline_final_evaluator",
                "candidate_final_evaluator": "candidate_final_evaluator",
            }[name]
            or state not in {"verified", "failed", "not_reached"}
            or (receipt is None) != (state != "verified")
        ):
            raise ValueError("mechanical gate provider slot value is invalid")
        result[name] = copy.deepcopy(dict(slot))
    return result


def _require_verified_tiled_without_plan(embeddings: VerifiedTiledSwinEmbeddings) -> dict[str, object]:
    if type(embeddings) is not VerifiedTiledSwinEmbeddings:
        raise TypeError("verified tiled Swin embeddings are required")
    _require_registered_capability(
        _REGISTERED_EMBEDDING_CAPABILITIES,
        embeddings,
        "verified tiled Swin embeddings are required",
    )
    metadata = embeddings._path.stat(follow_symlinks=False)
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ) != (
        embeddings._device,
        embeddings._inode,
        embeddings._size_bytes,
        embeddings._mtime_ns,
    ):
        raise ValueError("verified tiled embedding identity changed")
    payload, encoded = _read_bounded_json(embeddings._path, max_bytes=67_108_864)
    if (
        _sha256_bytes(encoded) != embeddings._file_sha256
        or payload.get("artifact_sha256") != embeddings._artifact_sha256
        or payload != embeddings._payload
    ):
        raise ValueError("verified tiled embedding bytes changed")
    return copy.deepcopy(payload)


def verify_vru_causal_final_evaluator(
    *,
    evaluator_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    verified_inputs: VerifiedTask0257TemporalInputs,
    tiled_embeddings: VerifiedTiledSwinEmbeddings,
) -> dict[str, object]:
    """Replay one separately addressed final evaluator from external receipts."""
    _require_verified_task0257_inputs(verified_inputs)
    tiled_payload = _require_verified_tiled_without_plan(tiled_embeddings)
    if not _is_sha256(expected_artifact_sha256) or not _is_sha256(expected_file_sha256):
        raise ValueError("final evaluator expected SHA-256 is invalid")
    raw_path = Path(evaluator_path)
    if raw_path.is_symlink():
        raise ValueError("final evaluator path must not be a symlink")
    path = raw_path.resolve(strict=True)
    payload, encoded = _read_bounded_json(path, max_bytes=67_108_864)
    if _sha256_bytes(encoded) != expected_file_sha256 or payload.get("artifact_sha256") != expected_artifact_sha256:
        raise ValueError("final evaluator external receipt does not match")
    if encoded != _canonical_json_bytes(payload):
        raise ValueError("final evaluator bytes are not canonical JSON plus LF")
    features, labels, game_ids, ordered_keys = _aligned_temporal_feature_inputs(
        inputs=verified_inputs,
        tiled_embeddings=tiled_embeddings,
    )
    input_receipts = {
        "plan": copy.deepcopy(tiled_payload["plan_receipt"]),
        "task0257": copy.deepcopy(tiled_payload["task0257_input_receipts"]),
        "tiled_embeddings": {
            "artifact_sha256": tiled_embeddings._artifact_sha256,
            "file_sha256": tiled_embeddings._file_sha256,
        },
    }
    role = payload.get("evaluator_role")
    if role == "baseline":
        matrices = {
            "swin3d_t": features["swin3d_t"],
            "mvit_v2_s+swin3d_t": features["mvit_v2_s+swin3d_t"],
        }
    elif role == "candidate":
        name = TEMPORAL_REPRESENTATION_CONTRACT["name"]
        matrices = {name: features[name]}
    else:
        raise ValueError("final evaluator role is invalid")
    return verify_final_evaluator_computation(
        payload,
        feature_matrices=matrices,
        labels=labels,
        game_ids=game_ids,
        ordered_training_rows=ordered_keys,
        input_receipts=input_receipts,
    )


def seal_module_a_final_receipt_registry(*, generation_dir: Path) -> dict[str, object]:
    """Freeze the published generation's seven file and six artifact receipts."""
    directory = Path(generation_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("final generation directory is invalid")
    file_receipts: dict[str, str] = {}
    artifact_receipts: dict[str, str] = {}
    for key in _FINAL_GENERATION_FILE_KEYS:
        path = directory / key
        if path.is_symlink() or not path.is_file():
            raise ValueError("final generation member is missing")
        if key.endswith("resource_guard.jsonl"):
            encoded = _read_bounded_bytes(
                path,
                max_bytes=TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"],
            )
        else:
            payload, encoded = _read_bounded_json(path, max_bytes=67_108_864)
            if encoded != _canonical_json_bytes(payload):
                raise ValueError("final generation member encoding is not canonical")
            artifact_sha256 = payload.get("artifact_sha256")
            if not _is_sha256(artifact_sha256):
                raise ValueError("final generation member artifact SHA-256 is invalid")
            artifact_receipts[key] = artifact_sha256
        file_receipts[key] = _sha256_bytes(encoded)
    registry: dict[str, object] = {
        "schema_version": FINAL_GENERATION_RECEIPT_REGISTRY_SCHEMA,
        "module_id": MODULE_A_ID,
        "purpose": "external_postpublication_receipt_registry",
        "runtime_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "generation_name": "final_v1",
        "file_receipts": file_receipts,
        "artifact_receipts": artifact_receipts,
    }
    registry["artifact_sha256"] = _canonical_sha256(registry)
    return registry


def verify_module_a_final_receipt_registry(
    payload: Mapping[str, object],
    *,
    expected_artifact_sha256: str,
) -> dict[str, object]:
    expected_fields = {
        "schema_version",
        "module_id",
        "purpose",
        "runtime_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
        "generation_name",
        "file_receipts",
        "artifact_receipts",
        "artifact_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("final receipt registry fields are invalid")
    if (
        payload.get("schema_version") != FINAL_GENERATION_RECEIPT_REGISTRY_SCHEMA
        or payload.get("module_id") != MODULE_A_ID
        or payload.get("purpose") != "external_postpublication_receipt_registry"
        or payload.get("generation_name") != "final_v1"
        or any(
            payload.get(field) is not False
            for field in (
                "runtime_consumable",
                "formal_evaluation_eligible",
                "promotion_eligible",
                "promoted",
            )
        )
        or not _is_sha256(expected_artifact_sha256)
        or payload.get("artifact_sha256") != expected_artifact_sha256
        or payload.get("artifact_sha256") != _canonical_sha256(payload)
    ):
        raise ValueError("final receipt registry identity is invalid")
    file_receipts = payload.get("file_receipts")
    artifact_receipts = payload.get("artifact_receipts")
    if (
        not isinstance(file_receipts, Mapping)
        or set(file_receipts) != set(_FINAL_GENERATION_FILE_KEYS)
        or not isinstance(artifact_receipts, Mapping)
        or set(artifact_receipts) != set(_FINAL_GENERATION_ARTIFACT_KEYS)
        or any(not _is_sha256(value) for value in file_receipts.values())
        or any(not _is_sha256(value) for value in artifact_receipts.values())
    ):
        raise ValueError("final receipt registry coverage is invalid")
    return copy.deepcopy(dict(payload))


def verify_vru_causal_temporal_terminal_failure(
    *,
    generation_dir: Path,
    expected_file_receipts: Mapping[str, str],
    expected_artifact_receipts: Mapping[str, str],
    verified_inputs: VerifiedTask0257TemporalInputs,
    plan: VerifiedTemporalFeaturePlan,
    expected_prior_attempt_receipts: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Replay an externally receipted terminal-failure generation fail closed."""
    _require_verified_task0257_inputs(verified_inputs)
    _require_verified_plan(plan)
    directory = Path(generation_dir)
    if (
        directory.name != "terminal_failure_v1"
        or directory.is_symlink()
        or not directory.is_dir()
        or (directory.parent / "final_v1").exists()
        or any(directory.parent.glob(".task0258-*"))
    ):
        raise ValueError("terminal failure generation state is invalid")
    expected_files = {
        "terminal_attempt/resource_guard.jsonl",
        "terminal_attempt/attempt_record.json",
        "mechanical_failure.json",
    }
    expected_artifacts = {
        "terminal_attempt/attempt_record.json",
        "mechanical_failure.json",
    }
    if set(expected_file_receipts) != expected_files or set(expected_artifact_receipts) != expected_artifacts:
        raise ValueError("terminal failure external receipt coverage is invalid")
    if any(not _is_sha256(value) for value in expected_file_receipts.values()) or any(
        not _is_sha256(value) for value in expected_artifact_receipts.values()
    ):
        raise ValueError("terminal failure external receipt value is invalid")
    if {path.name for path in directory.iterdir()} != {"terminal_attempt", "mechanical_failure.json"}:
        raise ValueError("terminal failure member coverage is invalid")
    terminal_attempt = directory / "terminal_attempt"
    if (
        terminal_attempt.is_symlink()
        or not terminal_attempt.is_dir()
        or {path.name for path in terminal_attempt.iterdir()} != {"resource_guard.jsonl", "attempt_record.json"}
    ):
        raise ValueError("terminal failure attempt coverage is invalid")
    payloads: dict[str, dict[str, object]] = {}
    encoded_by_key: dict[str, bytes] = {}
    for key in expected_files:
        path = directory / key
        if path.is_symlink() or not path.is_file():
            raise ValueError("terminal failure member is not a regular file")
        if key.endswith(".jsonl"):
            encoded = _read_bounded_bytes(path, max_bytes=TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"])
        else:
            payload, encoded = _read_bounded_json(path, max_bytes=67_108_864)
            payloads[key] = payload
            if payload.get("artifact_sha256") != expected_artifact_receipts[key] or encoded != _canonical_json_bytes(
                payload
            ):
                raise ValueError("terminal failure artifact receipt or encoding is invalid")
        if _sha256_bytes(encoded) != expected_file_receipts[key]:
            raise ValueError("terminal failure file receipt does not match")
        encoded_by_key[key] = encoded
    attempt = verify_tiled_swin_attempt(payloads["terminal_attempt/attempt_record.json"])
    if attempt.get("disposition") != "terminal_failure":
        raise ValueError("terminal failure attempt disposition is invalid")
    resource_bytes = encoded_by_key["terminal_attempt/resource_guard.jsonl"]
    resource_receipt = {
        "file_sha256": _sha256_bytes(resource_bytes),
        "filename": "resource_guard.jsonl",
        "size_bytes": len(resource_bytes),
    }
    if attempt.get("resource_log_receipt") != resource_receipt:
        raise ValueError("terminal failure resource receipt does not match")
    chain = payloads["mechanical_failure.json"].get("attempt_chain")
    _verify_attempt_chain_receipts(chain)
    if not isinstance(chain, list) or len(chain) != attempt["attempt_ordinal"]:
        raise ValueError("terminal failure attempt chain length is invalid")
    prior_state = _replay_published_prior_attempts(
        output_root=directory.parent,
        attempt_chain=chain[:-1],
        plan=plan,
    )
    if len(expected_prior_attempt_receipts) != len(chain) - 1:
        raise ValueError("terminal failure prior-attempt external receipts are incomplete")
    for chain_row, external_receipt in zip(chain[:-1], expected_prior_attempt_receipts, strict=True):
        if chain_row.get("attempt_record") != external_receipt:
            raise ValueError("terminal failure prior-attempt external receipt does not match")
    terminal_row = chain[-1]
    if (
        attempt["prior_attempt_receipt"] != (chain[-2]["attempt_record"] if len(chain) > 1 else None)
        or attempt["resume_input_cas"] != prior_state["latest_resume_cas"]
        or attempt["started_prefix_count"] != prior_state["completed_prefix_count"]
        or terminal_row["attempt_record"]["internal_sha256"] != attempt["artifact_sha256"]
        or terminal_row["attempt_record"]["file_sha256"]
        != expected_file_receipts["terminal_attempt/attempt_record.json"]
        or terminal_row["resource_log"] != resource_receipt
        or terminal_row["resume_output_cas"] is not None
    ):
        raise ValueError("terminal failure attempt chain does not bind terminal members")
    terminal_sample_count, _ending_consecutive = verify_resource_log_bytes(
        resource_bytes,
        expected_attempt_ordinal=attempt["attempt_ordinal"],
        prior_cumulative_samples=prior_state["cumulative_resource_samples"],
        prior_consecutive_breach_count=prior_state["ending_consecutive_breach_count"],
    )
    if (
        attempt["cumulative_resource_samples"] != prior_state["cumulative_resource_samples"] + terminal_sample_count
        or attempt["cumulative_resource_log_bytes"]
        != prior_state["cumulative_resource_log_bytes"] + len(resource_bytes)
        or attempt["cumulative_active_runtime_nanoseconds"] < prior_state["cumulative_active_runtime_nanoseconds"]
    ):
        raise ValueError("terminal failure cumulative counters are invalid")
    failure = payloads["mechanical_failure.json"]
    expected_fields = _COMMON_ARTIFACT_FIELDS | {
        "input_receipts",
        "attempt_chain",
        "ordered_check_results",
        "error_bound_result",
        "resource_summary",
        "publication_summary",
        "decision",
        "stop_reason",
        "conditional_downstream",
        "artifact_sha256",
    }
    if (
        set(failure) != expected_fields
        or failure.get("schema_version") != MECHANICAL_GATE_SCHEMA
        or failure.get("module_id") != MODULE_A_ID
        or failure.get("purpose") != "development_diagnostic_only"
        or any(
            failure.get(field) is not False
            for field in (
                "runtime_consumable",
                "training_consumable",
                "formal_evaluation_eligible",
                "promotion_eligible",
                "promoted",
            )
        )
        or failure.get("decision") != "mechanical_failure"
        or failure.get("error_bound_result") is not None
        or failure.get("artifact_sha256") != _canonical_sha256(failure)
    ):
        raise ValueError("terminal failure mechanical record is invalid")
    slots = _verify_mechanical_receipt_slots(failure.get("input_receipts"))
    if (
        slots["plan"]["receipt"] != {"artifact_sha256": plan._artifact_sha256, "file_sha256": plan._file_sha256}
        or slots["task0257"]["receipt"] != plan._payload["task0257_receipts"]
        or any(slots[name]["verification_state"] != "not_reached" for name in _MECHANICAL_RECEIPT_SLOT_NAMES[2:])
        or failure.get("attempt_chain") != chain
        or failure.get("stop_reason") != attempt.get("stop_reason")
        or failure.get("publication_summary")
        != {"final_generation_published": False, "atomic_directory_publication_required": True}
    ):
        raise ValueError("terminal failure bindings are invalid")
    return copy.deepcopy(failure)


def verify_vru_causal_temporal_final_generation(
    *,
    generation_dir: Path,
    expected_file_receipts: Mapping[str, str],
    expected_artifact_receipts: Mapping[str, str],
    verified_inputs: VerifiedTask0257TemporalInputs,
    plan: VerifiedTemporalFeaturePlan,
    tiled_embeddings: VerifiedTiledSwinEmbeddings,
) -> ModuleAFinalGeneration:
    """Replay a published Module-A terminal generation from external receipts."""
    _require_verified_task0257_inputs(verified_inputs)
    _require_verified_plan(plan)
    verified_tiled_payload = _require_verified_tiled_swin_embeddings(tiled_embeddings, plan=plan)
    directory = Path(generation_dir)
    if (
        directory.name != "final_v1"
        or directory.is_symlink()
        or not directory.is_dir()
        or (directory.parent / "terminal_failure_v1").exists()
        or any(directory.parent.glob(".task0258-*"))
    ):
        raise ValueError("final generation directory is invalid")
    if set(expected_file_receipts) != set(_FINAL_GENERATION_FILE_KEYS) or set(expected_artifact_receipts) != set(
        _FINAL_GENERATION_ARTIFACT_KEYS
    ):
        raise ValueError("final generation external receipt coverage is invalid")
    if any(not _is_sha256(value) for value in expected_file_receipts.values()) or any(
        not _is_sha256(value) for value in expected_artifact_receipts.values()
    ):
        raise ValueError("final generation external receipt values are invalid")
    if {path.name for path in directory.iterdir()} != {
        "terminal_attempt",
        "tiled_swin_embeddings.json",
        "temporal_retrospective.json",
        "baseline_final_evaluator.json",
        "candidate_final_evaluator.json",
        "mechanical_gate.json",
    }:
        raise ValueError("final generation member coverage is invalid")
    terminal_attempt = directory / "terminal_attempt"
    if (
        terminal_attempt.is_symlink()
        or not terminal_attempt.is_dir()
        or {path.name for path in terminal_attempt.iterdir()} != {"resource_guard.jsonl", "attempt_record.json"}
    ):
        raise ValueError("final terminal-attempt coverage is invalid")
    payloads: dict[str, dict[str, object]] = {}
    for key in _FINAL_GENERATION_FILE_KEYS:
        path = directory / key
        if path.is_symlink() or not path.is_file():
            raise ValueError("final generation member is not a regular file")
        if key.endswith("resource_guard.jsonl"):
            encoded = _read_bounded_bytes(
                path,
                max_bytes=TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"],
            )
        else:
            payload, encoded = _read_bounded_json(path, max_bytes=67_108_864)
            payloads[key] = payload
            if payload.get("artifact_sha256") != expected_artifact_receipts[key] or encoded != _canonical_json_bytes(
                payload
            ):
                raise ValueError("final generation artifact receipt does not match")
        if _sha256_bytes(encoded) != expected_file_receipts[key]:
            raise ValueError("final generation file receipt does not match")
    attempt = verify_tiled_swin_attempt(payloads["terminal_attempt/attempt_record.json"])
    if attempt.get("disposition") != "completed" or attempt.get("completed_prefix_count") != 45:
        raise ValueError("final generation does not contain a completed attempt")
    resource_path = terminal_attempt / "resource_guard.jsonl"
    resource_bytes = _read_bounded_bytes(
        resource_path,
        max_bytes=TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"],
    )
    resource_receipt = {
        "file_sha256": _sha256_bytes(resource_bytes),
        "filename": "resource_guard.jsonl",
        "size_bytes": len(resource_bytes),
    }
    if attempt.get("resource_log_receipt") != resource_receipt:
        raise ValueError("completed attempt resource receipt does not match")
    current_sample_count = resource_bytes.count(b"\n")
    prior_cumulative_samples = attempt["cumulative_resource_samples"] - current_sample_count
    if prior_cumulative_samples < 0:
        raise ValueError("completed attempt resource sample count is invalid")
    verified_sample_count, _ending_consecutive = verify_resource_log_bytes(
        resource_bytes,
        expected_attempt_ordinal=attempt["attempt_ordinal"],
        prior_cumulative_samples=prior_cumulative_samples,
        prior_consecutive_breach_count=None,
    )
    if verified_sample_count != current_sample_count:
        raise ValueError("completed attempt resource log count is invalid")
    if tiled_embeddings._path != (directory / "tiled_swin_embeddings.json").resolve(strict=True):
        raise ValueError("verified tiled embeddings are not the final generation member")
    if verified_tiled_payload != payloads["tiled_swin_embeddings.json"]:
        raise ValueError("verified tiled embedding bytes changed")
    chain = verified_tiled_payload["attempt_chain"]
    _verify_attempt_chain_receipts(chain)
    prior_state = _replay_published_prior_attempts(
        output_root=directory.parent,
        attempt_chain=chain[:-1],
        plan=plan,
    )
    if (
        attempt["prior_attempt_receipt"] != (chain[-2]["attempt_record"] if len(chain) > 1 else None)
        or attempt["resume_input_cas"] != prior_state["latest_resume_cas"]
        or attempt["started_prefix_count"] != prior_state["completed_prefix_count"]
        or chain[-1]["attempt_record"]["internal_sha256"] != attempt["artifact_sha256"]
        or chain[-1]["attempt_record"]["file_sha256"] != expected_file_receipts["terminal_attempt/attempt_record.json"]
        or chain[-1]["resource_log"] != resource_receipt
        or chain[-1]["resume_output_cas"] is not None
        or attempt["attempt_ordinal"] != len(chain)
    ):
        raise ValueError("completed attempt chain does not bind terminal members")
    verified_sample_count, _ending_consecutive = verify_resource_log_bytes(
        resource_bytes,
        expected_attempt_ordinal=attempt["attempt_ordinal"],
        prior_cumulative_samples=prior_state["cumulative_resource_samples"],
        prior_consecutive_breach_count=prior_state["ending_consecutive_breach_count"],
    )
    if (
        verified_sample_count != current_sample_count
        or attempt["cumulative_resource_samples"] != prior_state["cumulative_resource_samples"] + current_sample_count
        or attempt["cumulative_resource_log_bytes"]
        != prior_state["cumulative_resource_log_bytes"] + len(resource_bytes)
        or attempt["cumulative_active_runtime_nanoseconds"] < prior_state["cumulative_active_runtime_nanoseconds"]
    ):
        raise ValueError("completed attempt cumulative counters are invalid")
    gate = payloads["mechanical_gate.json"]
    _verify_mechanical_receipt_slots(gate.get("input_receipts"))
    check_rows = gate.get("ordered_check_results")
    if not isinstance(check_rows, list) or any(
        not isinstance(row, Mapping)
        or set(row) != {"check", "passed"}
        or not isinstance(row.get("check"), str)
        or type(row.get("passed")) is not bool
        for row in check_rows
    ):
        raise ValueError("final generation mechanical checks are invalid")
    passed_by_name = {row["check"]: row["passed"] for row in check_rows}
    if gate.get("decision") == "mechanical_failure" or any(
        passed_by_name.get(name) is not True for name in _MANDATORY_MECHANICAL_EVIDENCE_CHECKS
    ):
        raise ValueError("final generation lacks mandatory mechanical evidence")
    resource_summary = gate.get("resource_summary")
    if (
        not isinstance(resource_summary, Mapping)
        or resource_summary.get("attempt_count") != len(chain)
        or resource_summary.get("cumulative_active_runtime_nanoseconds")
        != attempt["cumulative_active_runtime_nanoseconds"]
        or resource_summary.get("cumulative_resource_samples") != attempt["cumulative_resource_samples"]
        or resource_summary.get("cumulative_resource_log_bytes") != attempt["cumulative_resource_log_bytes"]
        or resource_summary.get("all_sustained_limits_passed") is not True
        or resource_summary.get("process_tree_reaped") is not True
    ):
        raise ValueError("final generation resource summary is invalid")
    fresh = FreshlyProducedTiledSwinEmbeddings(_FRESH_EMBEDDING_TOKEN)
    fresh._token = _FRESH_EMBEDDING_TOKEN
    fresh._payload = copy.deepcopy(verified_tiled_payload)
    fresh._file_sha256 = expected_file_receipts["tiled_swin_embeddings.json"]
    fresh._resource_summary = copy.deepcopy(dict(resource_summary))
    _register_fresh_embedding_capability(fresh)
    expected = _build_vru_causal_temporal_final_generation(
        inputs=verified_inputs,
        plan=plan,
        tiled_embeddings=fresh,
    )
    if (
        payloads["temporal_retrospective.json"] != expected.retrospective
        or payloads["baseline_final_evaluator.json"] != expected.baseline_evaluator
        or payloads["candidate_final_evaluator.json"] != expected.candidate_evaluator
        or gate != expected.mechanical_gate
    ):
        raise ValueError("final generation does not reproduce its frozen computation")
    return expected
