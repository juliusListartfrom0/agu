#!/usr/bin/env python3
"""Fuse frozen independent VLM decisions with stored auxiliary OOF gates.

This command is label-free: it does not accept annotation paths and never
selects a threshold.  It only applies the per-game thresholds already sealed
in a training-only auxiliary screen.  Auxiliary coverage mismatches fail
closed.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.analysis.independent_shot_vlm_evidence_gate import (
    fuse_vlm_with_frozen_auxiliary,
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _require_output_disjoint_from_inputs(
    output_path: Path,
    *,
    input_paths: tuple[Path, ...],
) -> tuple[Path, tuple[Path, ...]]:
    resolved_output = output_path.resolve()
    resolved_inputs = tuple(path.resolve() for path in input_paths)
    if resolved_output in set(resolved_inputs):
        raise ValueError("output path must not alias a plan, VLM, or auxiliary input path")
    return resolved_output, resolved_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument(
        "--expected-auxiliary-sha256",
        required=True,
        help="SHA-256 frozen independently of the auxiliary JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_path, input_paths = _require_output_disjoint_from_inputs(
        args.output,
        input_paths=(args.plan, args.vlm_predictions, args.auxiliary),
    )
    plan_path, vlm_predictions_path, auxiliary_path = input_paths

    artifact = fuse_vlm_with_frozen_auxiliary(
        plan=json.loads(plan_path.read_text(encoding="utf-8")),
        vlm_predictions=json.loads(
            vlm_predictions_path.read_text(encoding="utf-8")
        ),
        auxiliary_artifact=json.loads(
            auxiliary_path.read_text(encoding="utf-8")
        ),
        expected_auxiliary_artifact_sha256=args.expected_auxiliary_sha256,
    )
    _write_json_atomic(output_path, artifact)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "artifact_sha256": artifact["artifact_sha256"],
                "coverage": artifact["coverage"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
