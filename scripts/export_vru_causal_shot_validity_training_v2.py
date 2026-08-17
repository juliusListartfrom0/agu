#!/usr/bin/env python3
"""Export the additive Harwood VRU causal training chain."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_training import encode_json_artifact  # noqa: E402
from app.analysis.vru_causal_training_v2 import (  # noqa: E402
    FROZEN_PARENT_CHILD_SHA256,
    HARWOOD_CANDIDATE_FILENAME,
    HARWOOD_LABEL_FILENAME,
    ContinuousCausalTrainingChain,
    build_vru_causal_shot_validity_training_extension_export,
)

_MAX_JSON_BYTES = 2 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-export", type=Path, required=True)
    parser.add_argument("--expected-parent-artifact-sha256", required=True)
    parser.add_argument("--expected-parent-file-sha256", required=True)
    parser.add_argument("--parent-asset-root", type=Path, required=True)
    parser.add_argument("--harwood-selection", type=Path, required=True)
    parser.add_argument("--expected-harwood-selection-artifact-sha256", required=True)
    parser.add_argument("--harwood-review-plan", type=Path, required=True)
    parser.add_argument("--expected-harwood-review-plan-artifact-sha256", required=True)
    parser.add_argument("--harwood-sealed-review", type=Path, required=True)
    parser.add_argument("--expected-harwood-sealed-review-artifact-sha256", required=True)
    parser.add_argument("--harwood-raw-frame-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-harwood-raw-frame-manifest-artifact-sha256",
        required=True,
    )
    parser.add_argument("--harwood-source-manifest", type=Path, required=True)
    parser.add_argument("--source-groups", type=Path, required=True)
    parser.add_argument("--expected-source-groups-artifact-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "v2 index path inside a dedicated immutable generation directory; "
            "the directory is atomically published with both Harwood children"
        ),
    )
    return parser.parse_args()


def _preflight_paths(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], Path, dict[str, Path], tuple[Path, ...]]:
    """Resolve inputs only after output aliases are rejected without content reads."""

    explicit_lexical = {
        "parent_export": _absolute_path(args.parent_export),
        "parent_asset_root": _absolute_path(args.parent_asset_root),
        "harwood_selection": _absolute_path(args.harwood_selection),
        "harwood_review_plan": _absolute_path(args.harwood_review_plan),
        "harwood_sealed_review": _absolute_path(args.harwood_sealed_review),
        "harwood_raw_frame_manifest": _absolute_path(args.harwood_raw_frame_manifest),
        "harwood_source_manifest": _absolute_path(args.harwood_source_manifest),
        "source_groups": _absolute_path(args.source_groups),
    }
    output_path = _absolute_path(args.output)
    _safe_basename(output_path.name, "v2 index filename")
    child_paths = {
        filename: output_path.parent / filename for filename in (HARWOOD_CANDIDATE_FILENAME, HARWOOD_LABEL_FILENAME)
    }
    outputs = (output_path, *child_paths.values())
    parent_root = explicit_lexical["parent_asset_root"]
    phase_one_protected = (
        *explicit_lexical.values(),
        *(parent_root / filename for filename in FROZEN_PARENT_CHILD_SHA256),
    )

    # Phase one is deliberately metadata-only. This must precede every JSON read,
    # source resolver, and raw-frame discovery operation.
    _require_generation_paths_disjoint(
        target_dir=output_path.parent,
        outputs=outputs,
        protected_inputs=phase_one_protected,
    )
    _require_existing_generation_shape(
        target_dir=output_path.parent,
        expected_names={path.name for path in outputs},
    )

    explicit = {name: path.resolve() for name, path in explicit_lexical.items()}
    resolved_parent_root = explicit["parent_asset_root"]
    if not resolved_parent_root.is_dir():
        raise ValueError("parent asset root must be an existing directory")
    protected: list[Path] = [*explicit.values()]
    protected.extend((resolved_parent_root / filename).resolve() for filename in FROZEN_PARENT_CHILD_SHA256)

    # Phase two uses only this CLI's capped, duplicate-key-rejecting parser.
    source_video = _resolve_source_video_input(explicit["harwood_source_manifest"])
    raw_root, jpeg_paths = _discover_raw_frame_inputs(explicit["harwood_raw_frame_manifest"])
    protected.extend((source_video, raw_root, *jpeg_paths))
    _require_generation_paths_disjoint(
        target_dir=output_path.parent,
        outputs=outputs,
        protected_inputs=protected,
    )
    return explicit, output_path, child_paths, tuple(protected)


def _resolve_source_video_input(manifest_path: Path) -> Path:
    payload = _read_json_object(manifest_path, "source manifest")
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("source manifest videos must be a list")
    matches = [row for row in videos if isinstance(row, Mapping) and row.get("location") == "harwood"]
    if len(matches) != 1:
        raise ValueError("source manifest must contain exactly one Harwood video")
    relative = _safe_relative_path(matches[0].get("path"), "Harwood source-video path")
    manifest_root = manifest_path.parent.resolve()
    unresolved = manifest_root / relative
    if unresolved.is_symlink():
        raise ValueError("Harwood source video cannot be a symlink")
    path = unresolved.resolve()
    if not path.is_relative_to(manifest_root) or not path.is_file():
        raise ValueError("Harwood source video is missing or outside its manifest")
    return path


def _discover_raw_frame_inputs(manifest_path: Path) -> tuple[Path, tuple[Path, ...]]:
    payload = _read_json_object(manifest_path, "raw-frame manifest")
    root_name = _safe_relative_path(payload.get("frame_root"), "frame root")
    if len(root_name.parts) != 1:
        raise ValueError("frame root must be a single safe directory")
    unresolved_root = manifest_path.parent.resolve() / root_name
    if unresolved_root.is_symlink():
        raise ValueError("raw-frame root cannot be a symlink")
    root = unresolved_root.resolve()
    if not root.is_dir() or not root.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("raw-frame root is missing or outside its manifest")
    rows = payload.get("frames")
    if not isinstance(rows, list) or payload.get("frame_count") != 1536 or len(rows) != 1536:
        raise ValueError("raw-frame manifest must expand exactly 1,536 JPEGs")
    paths: list[Path] = []
    path_keys: set[tuple[str, ...]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("raw-frame manifest rows must be objects")
        relative = _safe_relative_path(row.get("relative_path"), "review JPEG path")
        unresolved = root / relative
        if unresolved.is_symlink():
            raise ValueError("review JPEG cannot be a symlink")
        path = unresolved.resolve()
        key = _casefold_parts(path)
        if not path.is_relative_to(root) or not path.is_file() or key in path_keys:
            raise ValueError("review JPEG is missing or outside the frame root")
        paths.append(path)
        path_keys.add(key)
    return root, tuple(paths)


def _require_generation_paths_disjoint(
    *,
    target_dir: Path,
    outputs: Sequence[Path],
    protected_inputs: Sequence[Path],
) -> None:
    target = _absolute_path(target_dir)
    output_paths = tuple(_absolute_path(path) for path in outputs)
    if not output_paths or any(path.parent != target for path in output_paths):
        raise ValueError("all v2 artifacts must be direct children of the target directory")
    output_keys = [_casefold_parts(path) for path in output_paths]
    if len(output_keys) != len(set(output_keys)):
        raise ValueError("output artifact paths must be casefold-unique")
    if target.is_symlink() or _path_has_symlink_component(target.parent):
        raise ValueError("output target directory must not use symlinks")
    if os.path.lexists(target) and not target.is_dir():
        raise ValueError("output target must be a real directory or absent")
    if any(path.is_symlink() for path in output_paths):
        raise ValueError("output artifact must not be a symlink")

    target_variants = _path_alias_variants(target)
    for protected in protected_inputs:
        protected_path = _absolute_path(protected)
        protected_variants = _path_alias_variants(protected_path)
        if any(_paths_related_casefold(left, right) for left in target_variants for right in protected_variants):
            raise ValueError("output target aliases or nests protected input")

    protected_identities = {
        identity for path in protected_inputs if (identity := _existing_identity(_absolute_path(path))) is not None
    }
    output_identities: set[tuple[int, int]] = set()
    for output in output_paths:
        identity = _existing_identity(output)
        if identity is None:
            continue
        if identity in protected_identities or identity in output_identities:
            raise ValueError("output artifact is hardlink-aliased")
        output_identities.add(identity)


def _write_generation_atomic(
    *,
    output_path: Path,
    child_paths: Mapping[str, Path],
    chain: ContinuousCausalTrainingChain,
    protected_inputs: Sequence[Path],
) -> None:
    outputs = (output_path, *child_paths.values())
    _require_generation_paths_disjoint(
        target_dir=output_path.parent,
        outputs=outputs,
        protected_inputs=protected_inputs,
    )
    expected_names = set(chain.extension_candidate_bundles) | set(chain.extension_label_files)
    if set(child_paths) != expected_names:
        raise ValueError("resolved v2 child output set does not match the chain")
    expected_child_paths = {filename: output_path.parent / filename for filename in expected_names}
    if {filename: _absolute_path(path) for filename, path in child_paths.items()} != {
        filename: _absolute_path(path) for filename, path in expected_child_paths.items()
    }:
        raise ValueError("v2 children must use their sealed basenames in the target directory")
    payload_by_name: dict[str, Mapping[str, Any]] = {
        filename: payload
        for filename, payload in {
            **chain.extension_candidate_bundles,
            **chain.extension_label_files,
        }.items()
    }
    if output_path.name in payload_by_name:
        raise ValueError("v2 index filename collides with an extension child")
    payload_by_name[output_path.name] = chain.index
    extension = chain.index["extension"]
    refs = {
        HARWOOD_CANDIDATE_FILENAME: extension["candidate_bundle"],
        HARWOOD_LABEL_FILENAME: extension["labels"],
    }
    encoded_by_name = {name: encode_json_artifact(payload) for name, payload in payload_by_name.items()}
    for name, reference in refs.items():
        encoded = encoded_by_name[name]
        if len(encoded) != reference["size_bytes"] or _sha256_bytes(encoded) != reference["sha256"]:
            raise ValueError("v2 child bytes do not match the sealed index")
    _publish_generation(
        target_dir=output_path.parent,
        encoded_by_name=encoded_by_name,
    )


def _publish_generation(*, target_dir: Path, encoded_by_name: Mapping[str, bytes]) -> bool:
    """Publish an immutable three-file generation with one directory rename."""

    target = _absolute_path(target_dir)
    _safe_basename(target.name, "generation directory")
    if len(encoded_by_name) != 3:
        raise ValueError("v2 generation must contain exactly three artifacts")
    names = tuple(encoded_by_name)
    if any(_safe_basename(name, "generation artifact") != name for name in names):
        raise ValueError("generation artifact names must be safe basenames")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("generation artifact names must be casefold-unique")
    if any(not isinstance(encoded, bytes) for encoded in encoded_by_name.values()):
        raise TypeError("generation artifact contents must be bytes")

    parent = target.parent
    if target.is_symlink() or _path_has_symlink_component(parent):
        raise ValueError("generation target must not use symlinks")
    if not parent.is_dir():
        raise ValueError("generation target parent must be an existing directory")
    if os.path.lexists(target):
        _verify_existing_generation(
            target_dir=target,
            encoded_by_name=encoded_by_name,
        )
        return False

    stage = Path(
        tempfile.mkdtemp(
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".stage",
        )
    )
    published = False
    try:
        for position, name in enumerate(sorted(names)):
            encoded = encoded_by_name[name]
            staged_file = stage / name
            with staged_file.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            status = staged_file.lstat()
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_size != len(encoded)
                or staged_file.read_bytes() != encoded
            ):
                raise ValueError("staged v2 artifact bytes failed verification")
            if position == 0:
                _publication_checkpoint("during_stage_write")

        if {path.name for path in stage.iterdir()} != set(names):
            raise ValueError("staged v2 generation contains unexpected artifacts")
        _fsync_directory(stage)
        _publication_checkpoint("before_rename")
        if os.path.lexists(target):
            _verify_existing_generation(
                target_dir=target,
                encoded_by_name=encoded_by_name,
            )
            return False
        os.replace(stage, target)
        published = True
        _publication_checkpoint("after_rename_before_parent_fsync")
        _fsync_directory(parent)
        return True
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def _verify_existing_generation(*, target_dir: Path, encoded_by_name: Mapping[str, bytes]) -> None:
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise ValueError("immutable generation target must be a real non-symlink directory")
    entries = tuple(target_dir.iterdir())
    if {entry.name for entry in entries} != set(encoded_by_name):
        raise FileExistsError("immutable generation target exists with partial, extra, or different artifacts")
    for entry in entries:
        status = entry.lstat()
        expected = encoded_by_name[entry.name]
        if (
            entry.is_symlink()
            or not stat.S_ISREG(status.st_mode)
            or status.st_size != len(expected)
            or entry.read_bytes() != expected
        ):
            raise FileExistsError("immutable generation target exists with different artifact bytes")


def _require_existing_generation_shape(*, target_dir: Path, expected_names: set[str]) -> None:
    if not os.path.lexists(target_dir):
        return
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise ValueError("output target must be a real non-symlink directory")
    entries = tuple(target_dir.iterdir())
    if {entry.name for entry in entries} != expected_names:
        raise FileExistsError("immutable generation target is partial or contains extra artifacts")
    if any(entry.is_symlink() or not stat.S_ISREG(entry.lstat().st_mode) for entry in entries):
        raise FileExistsError("immutable generation target must contain exactly three regular files")


def _publication_checkpoint(_name: str) -> None:
    """Test-only crash injection point; production intentionally does nothing."""


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _paths_related_casefold(left: Path, right: Path) -> bool:
    left_parts = _casefold_parts(left)
    right_parts = _casefold_parts(right)
    shortest = min(len(left_parts), len(right_parts))
    return left_parts[:shortest] == right_parts[:shortest]


def _casefold_parts(path: Path) -> tuple[str, ...]:
    return tuple(part.casefold() for part in path.parts)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_alias_variants(path: Path) -> tuple[Path, ...]:
    absolute = _absolute_path(path)
    resolved = absolute.resolve(strict=False)
    return (absolute,) if resolved == absolute else (absolute, resolved)


def _path_has_symlink_component(path: Path) -> bool:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                return True
        except FileNotFoundError:
            return False
    return False


def _existing_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a path-safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a path-safe relative path")
    return path


def _safe_basename(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or Path(value).is_absolute()
        or len(Path(value).parts) != 1
        or value in {".", ".."}
    ):
        raise ValueError(f"{field} must be a path-safe basename")
    return value


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            encoded = handle.read(_MAX_JSON_BYTES + 1)
        if len(encoded) > _MAX_JSON_BYTES:
            raise ValueError(f"{description} exceeds the JSON size limit")
        payload = json.loads(encoded, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def main() -> int:
    args = parse_args()
    inputs, output_path, child_paths, protected = _preflight_paths(args)
    chain = build_vru_causal_shot_validity_training_extension_export(
        parent_export_path=inputs["parent_export"],
        expected_parent_artifact_sha256=args.expected_parent_artifact_sha256,
        expected_parent_file_sha256=args.expected_parent_file_sha256,
        parent_asset_root=inputs["parent_asset_root"],
        harwood_selection_path=inputs["harwood_selection"],
        expected_harwood_selection_artifact_sha256=(args.expected_harwood_selection_artifact_sha256),
        harwood_review_plan_path=inputs["harwood_review_plan"],
        expected_harwood_review_plan_artifact_sha256=(args.expected_harwood_review_plan_artifact_sha256),
        harwood_sealed_review_path=inputs["harwood_sealed_review"],
        expected_harwood_sealed_review_artifact_sha256=(args.expected_harwood_sealed_review_artifact_sha256),
        harwood_raw_frame_manifest_path=inputs["harwood_raw_frame_manifest"],
        expected_harwood_raw_frame_manifest_artifact_sha256=(args.expected_harwood_raw_frame_manifest_artifact_sha256),
        harwood_source_manifest_path=inputs["harwood_source_manifest"],
        source_groups_path=inputs["source_groups"],
        expected_source_groups_artifact_sha256=(args.expected_source_groups_artifact_sha256),
    )
    _write_generation_atomic(
        output_path=output_path,
        child_paths=child_paths,
        chain=chain,
        protected_inputs=(*protected, *chain.validated_input_paths),
    )
    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "artifact_sha256": chain.index["artifact_sha256"],
                "examples": 45,
                "positive": 17,
                "negative": 28,
                "excluded_uncertain": 3,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
