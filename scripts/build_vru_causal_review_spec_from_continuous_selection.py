#!/usr/bin/env python3
"""Build a sealed VRU review plan from one frozen continuous-game selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.continuous_causal_selection import (  # noqa: E402
    resolve_continuous_source_video_path,
    verify_continuous_causal_selection_receipt,
)
from scripts.build_vru_causal_review import (  # noqa: E402
    build_plan_from_verified_continuous_spec,
)

SPEC_SCHEMA = "agu.vru-causal-review-spec.v1"
PURPOSE = "offline_single_continuous_game_manual_causal_review_closure"


def build_spec_from_continuous_selection(
    *,
    selection_path: Path,
    expected_selection_artifact_sha256: str,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Verify all receipts and materialise the frozen 8 FPS frame geometry."""

    selection = verify_continuous_causal_selection_receipt(
        _read_mapping(selection_path, "continuous causal selection"),
        expected_artifact_sha256=expected_selection_artifact_sha256,
        source_manifest_path=source_manifest_path,
    )
    return _spec_from_verified_selection(
        selection,
        expected_selection_artifact_sha256=expected_selection_artifact_sha256,
    )


def build_review_plan_from_continuous_selection(
    *,
    selection_path: Path,
    expected_selection_artifact_sha256: str,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Verify the frozen selection and atomically return its sealed frame plan."""

    selection = verify_continuous_causal_selection_receipt(
        _read_mapping(selection_path, "continuous causal selection"),
        expected_artifact_sha256=expected_selection_artifact_sha256,
        source_manifest_path=source_manifest_path,
    )
    spec = _spec_from_verified_selection(
        selection,
        expected_selection_artifact_sha256=expected_selection_artifact_sha256,
    )
    return build_plan_from_verified_continuous_spec(
        manifest_path=source_manifest_path,
        spec=spec,
        source_selection_artifact_sha256=selection["artifact_sha256"],
        expected_source_manifest_sha256=selection["source_manifest_sha256"],
        source_clip_id=selection["source"]["clip_id"],
    )


def _spec_from_verified_selection(
    selection: Mapping[str, Any],
    *,
    expected_selection_artifact_sha256: str,
) -> dict[str, Any]:
    manifest_sha256 = selection["source_manifest_sha256"]

    policy = selection["selection"]
    if (
        policy["sample_count"] != 64
        or policy["sample_rate_hz"] != 8.0
        or policy["window_seconds"] != 8.0
        or policy["interval_semantics"] != "half_open"
    ):
        raise ValueError("continuous selection must use frozen 64-frame 8 FPS geometry")

    examples = [
        {
            "review_id": window["review_id"],
            "frame_indexes": _sample_frame_indexes(window),
        }
        for window in selection["windows"]
    ]
    return {
        "schema_version": SPEC_SCHEMA,
        "purpose": PURPOSE,
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "radius_seconds": 4.0,
        "sample_period_seconds": 0.125,
        "sample_count": 64,
        "interval_semantics": "half_open",
        "source_selection_artifact_sha256": selection["artifact_sha256"],
        "expected_selection_artifact_sha256": expected_selection_artifact_sha256,
        "source_manifest_sha256": manifest_sha256,
        "examples": examples,
    }


def _sample_frame_indexes(window: Mapping[str, Any]) -> list[int]:
    fps = float(window["source_fps"])
    start_seconds = float(window["start_seconds"])
    end_seconds = float(window["end_seconds"])
    indexes = [round((start_seconds + index / 8.0) * fps) for index in range(64)]
    if (
        indexes != sorted(set(indexes))
        or indexes[0] < 0
        or indexes[-1] >= int(window["frame_count"])
        or indexes[32] != int(window["anchor_frame"])
        or indexes[-1] >= round(end_seconds * fps)
    ):
        raise ValueError("continuous samples must be unique, bounded, and half-open")
    return indexes


def _read_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-artifact-sha256",
        required=True,
        help="externally recorded SHA-256 for the sealed selection",
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _freeze_paths(
    selection: Path,
    source_manifest: Path,
    output: Path,
) -> tuple[Path, Path, Path]:
    resolved_selection = selection.resolve(strict=True)
    resolved_manifest = source_manifest.resolve(strict=True)
    selected_video = _resolve_selected_video_path(
        selection_path=resolved_selection,
        manifest_path=resolved_manifest,
    )
    protected_inputs = (resolved_selection, resolved_manifest, selected_video)
    resolved_output = output.resolve(strict=False)
    output_casefold = resolved_output.as_posix().casefold()
    if output_casefold in {path.as_posix().casefold() for path in protected_inputs} or (
        resolved_output.exists() and any(os.path.samefile(resolved_output, path) for path in protected_inputs)
    ):
        raise ValueError("causal review plan output must not alias an input")
    return resolved_selection, resolved_manifest, resolved_output


def _resolve_selected_video_path(
    *,
    selection_path: Path,
    manifest_path: Path,
) -> Path:
    selection = _read_mapping(selection_path, "continuous causal selection")
    source = selection.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
        raise ValueError("continuous causal selection source is invalid")
    return resolve_continuous_source_video_path(
        manifest_path,
        source_id=source["source_id"],
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    selection, source_manifest, output = _freeze_paths(
        args.selection,
        args.source_manifest,
        args.output,
    )
    plan = build_review_plan_from_continuous_selection(
        selection_path=selection,
        expected_selection_artifact_sha256=(args.expected_selection_artifact_sha256),
        source_manifest_path=source_manifest,
    )
    _atomic_write_json(output, plan)
    print(
        json.dumps(
            {
                "artifact_sha256": plan["artifact_sha256"],
                "examples": len(plan["examples"]),
                "output": output.as_posix(),
                "source_manifest_sha256": plan["source_manifest_sha256"],
                "source_selection_artifact_sha256": plan["source_selection_artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
