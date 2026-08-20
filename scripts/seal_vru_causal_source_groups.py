#!/usr/bin/env python3
"""Seal the exact TASK-0257 game-to-production-family manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_source_groups import (  # noqa: E402
    seal_vru_causal_source_groups,
)


class _DuplicateJsonKeyError(ValueError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve_disjoint_paths(spec: Path, output: Path) -> tuple[Path, Path]:
    lexical_spec = _absolute_path(spec)
    lexical_output = _absolute_path(output)
    resolved_spec = spec.resolve()
    resolved_output = output.resolve()
    if (
        _paths_overlap(lexical_spec, lexical_output)
        or _paths_overlap_casefold(lexical_spec, lexical_output)
        or _paths_overlap(resolved_spec, resolved_output)
        or _paths_overlap_casefold(resolved_spec, resolved_output)
        or (
            (spec_identity := _existing_path_identity(resolved_spec)) is not None
            and spec_identity == _existing_path_identity(resolved_output)
        )
    ):
        raise ValueError("spec and output paths must be disjoint; aliases and ancestor/descendant paths are forbidden")
    return resolved_spec, resolved_output


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _paths_overlap_casefold(first: Path, second: Path) -> bool:
    first_parts = tuple(part.casefold() for part in first.parts)
    second_parts = tuple(part.casefold() for part in second.parts)
    shortest = min(len(first_parts), len(second_parts))
    return first_parts[:shortest] == second_parts[:shortest]


def _existing_path_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateJsonKeyError as exc:
        raise ValueError(str(exc)) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source-group JSON spec: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("source-group JSON spec must be an object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path, output_path = _resolve_disjoint_paths(args.spec, args.output)
    spec = _read_json_object(spec_path)
    artifact = seal_vru_causal_source_groups(spec)
    # Recheck identity after the read/seal interval before changing output state.
    spec_path, output_path = _resolve_disjoint_paths(spec_path, output_path)
    _write_json_atomic(output_path, artifact)
    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "artifact_sha256": artifact["artifact_sha256"],
                "game_group_count": artifact["game_group_count"],
                "production_family_count": artifact["production_family_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
