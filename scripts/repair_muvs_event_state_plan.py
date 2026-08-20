#!/usr/bin/env python3
"""Replace unmaterializable MUVS samples without consulting hidden labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvs_event_state import (  # noqa: E402
    MUVS_VIDEO_BASE_URL,
    _read_annotations,
    seal_muvs_event_state_plan,
    verify_muvs_event_state_plan,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(row: dict[str, Any]) -> tuple[str, str, int, tuple[float, ...]]:
    return (
        str(row["event_id"]),
        str(row["camera_id"]),
        int(row["period"]),
        tuple(float(value) for value in row["frame_times_sec"]),
    )


def _parse_repair(value: str) -> tuple[str, float]:
    sample_id, separator, raw_duration = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("repair must be SAMPLE_ID=DURATION_SEC")
    try:
        duration = float(raw_duration)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("repair duration must be numeric") from exc
    if not sample_id or duration <= 0.0:
        raise argparse.ArgumentTypeError("repair sample and duration must be valid")
    return sample_id, duration


def _parse_explicit_replacement(
    value: str,
) -> tuple[str, int, int, str, float]:
    sample_id, separator, raw = value.partition("=")
    parts = raw.split(",")
    if not separator or len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "explicit replacement must be "
            "SAMPLE_ID=PERIOD,FRAGMENT,CAMERA,DURATION_SEC"
        )
    try:
        period = int(parts[0])
        fragment = int(parts[1])
        duration = float(parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "explicit replacement numbers must be valid"
        ) from exc
    camera = parts[2]
    if (
        not sample_id
        or period < 1
        or fragment < 1
        or not camera.startswith("cam_")
        or duration <= 2.0
    ):
        raise argparse.ArgumentTypeError(
            "explicit replacement values must be valid"
        )
    return sample_id, period, fragment, camera, duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument(
        "--repair",
        action="append",
        type=_parse_repair,
        required=True,
        help="Unmaterializable sample and probed source duration; may be repeated.",
    )
    parser.add_argument(
        "--exclude-plan",
        action="append",
        type=Path,
        default=[],
        help="Prior plan whose samples must not be selected; may be repeated.",
    )
    parser.add_argument(
        "--explicit-replacement",
        action="append",
        type=_parse_explicit_replacement,
        default=[],
        help=(
            "Audited fallback for a source period with no usable row: "
            "SAMPLE_ID=PERIOD,FRAGMENT,CAMERA,DURATION_SEC"
        ),
    )
    parser.add_argument("--end-margin-sec", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.end_margin_sec < 0:
        raise ValueError("MUVS end margin must be non-negative")
    original = verify_muvs_event_state_plan(_read_json(args.plan))
    repairs = dict(args.repair)
    if len(repairs) != len(args.repair):
        raise ValueError("MUVS repair sample ids must be unique")
    explicit_replacements = {
        sample_id: (period, fragment, camera, duration)
        for sample_id, period, fragment, camera, duration in (
            args.explicit_replacement
        )
    }
    if len(explicit_replacements) != len(args.explicit_replacement):
        raise ValueError("MUVS explicit replacement sample ids must be unique")
    unknown_explicit = sorted(set(explicit_replacements) - set(repairs))
    if unknown_explicit:
        raise ValueError(
            "MUVS explicit replacements require matching repairs: "
            + ", ".join(unknown_explicit)
        )

    samples = [dict(row) for row in original["samples"]]
    by_id = {str(row["sample_id"]): row for row in samples}
    missing = sorted(set(repairs) - set(by_id))
    if missing:
        raise ValueError(f"MUVS repair samples are absent: {', '.join(missing)}")

    used = {_signature(row) for row in samples}
    excluded_plan_hashes = []
    for path in args.exclude_plan:
        excluded = verify_muvs_event_state_plan(_read_json(path))
        excluded_plan_hashes.append(excluded["artifact_sha256"])
        used.update(_signature(row) for row in excluded["samples"])

    audit_rows = []
    for sample_id, duration_sec in args.repair:
        target = by_id[sample_id]
        expected_end = float(target["frame_times_sec"][1])
        if expected_end <= duration_sec:
            raise ValueError(f"MUVS repair target is already materializable: {sample_id}")
        annotation_path = (
            args.data_root
            / "annotations"
            / f"{target['event_id']}_ANNOTATIONS.csv"
        )
        fragments_path = (
            args.data_root
            / "fragments"
            / str(target["split"])
            / f"{target['event_id']}_FRAGMENTS.csv"
        )
        if (
            _sha256(annotation_path) != target["annotation_sha256"]
            or _sha256(fragments_path) != target["fragments_sha256"]
        ):
            raise ValueError(f"MUVS source hash drift for repair target: {sample_id}")
        rows = _read_annotations(
            annotation_path,
            expected_event=str(target["event_id"]),
            expected_split=str(target["split"]),
        )

        target_index = None
        candidates = []
        for index, row in enumerate(rows):
            camera_id = str(row["camera_selected"])
            period = int(row["period"])
            fragment_number = int(row["fragment_number"])
            camera_offset = float(row["offsets"][camera_id])
            frame_times = [
                round((fragment_number - 1) * 3.0 + camera_offset + 1.0, 6),
                round((fragment_number - 1) * 3.0 + camera_offset + 2.0, 6),
            ]
            if (
                camera_id == target["camera_id"]
                and period == target["period"]
                and fragment_number == target["fragment_number"]
                and frame_times == target["frame_times_sec"]
            ):
                target_index = index
            video_url = (
                f"{MUVS_VIDEO_BASE_URL}/basketball/{target['event_id']}/"
                f"{camera_id}/VIDEOS/P{period}/video.mp4"
            )
            candidate = {
                "sample_id": sample_id,
                "split": target["split"],
                "event_id": target["event_id"],
                "period": period,
                "fragment_number": fragment_number,
                "camera_id": camera_id,
                "camera_offset_sec": camera_offset,
                "frame_times_sec": frame_times,
                "video_url": video_url,
                "annotation_sha256": target["annotation_sha256"],
                "fragments_sha256": target["fragments_sha256"],
            }
            if (
                camera_id == target["camera_id"]
                and period == target["period"]
                and frame_times[1] <= duration_sec - args.end_margin_sec
                and _signature(candidate) not in used
            ):
                candidates.append((index, candidate))
        if target_index is None:
            raise ValueError(f"MUVS repair target row is absent: {sample_id}")
        explicit = explicit_replacements.get(sample_id)
        if explicit is None:
            preceding = [
                (index, candidate)
                for index, candidate in candidates
                if index < target_index
            ]
            if not preceding:
                raise ValueError(
                    f"MUVS repair has no preceding source candidate: {sample_id}"
                )
            replacement_index, replacement = max(
                preceding,
                key=lambda item: item[0],
            )
            replacement_duration = duration_sec
            selection_rule = (
                "nearest preceding unused annotation row from the same "
                "event, camera, and period whose second frame is within "
                "the probed source duration"
            )
        else:
            replacement_period, replacement_fragment, replacement_camera, (
                replacement_duration
            ) = explicit
            explicit_candidates = []
            for index, row in enumerate(rows):
                camera_id = str(row["camera_selected"])
                period = int(row["period"])
                fragment_number = int(row["fragment_number"])
                if (
                    period != replacement_period
                    or fragment_number != replacement_fragment
                    or camera_id != replacement_camera
                ):
                    continue
                camera_offset = float(row["offsets"][camera_id])
                frame_times = [
                    round(
                        (fragment_number - 1) * 3.0
                        + camera_offset
                        + 1.0,
                        6,
                    ),
                    round(
                        (fragment_number - 1) * 3.0
                        + camera_offset
                        + 2.0,
                        6,
                    ),
                ]
                explicit_candidate = {
                    "sample_id": sample_id,
                    "split": target["split"],
                    "event_id": target["event_id"],
                    "period": period,
                    "fragment_number": fragment_number,
                    "camera_id": camera_id,
                    "camera_offset_sec": camera_offset,
                    "frame_times_sec": frame_times,
                    "video_url": (
                        f"{MUVS_VIDEO_BASE_URL}/basketball/"
                        f"{target['event_id']}/{camera_id}/VIDEOS/P"
                        f"{period}/video.mp4"
                    ),
                    "annotation_sha256": target["annotation_sha256"],
                    "fragments_sha256": target["fragments_sha256"],
                }
                if (
                    index < target_index
                    and frame_times[1]
                    <= replacement_duration - args.end_margin_sec
                    and _signature(explicit_candidate) not in used
                ):
                    explicit_candidates.append((index, explicit_candidate))
            if len(explicit_candidates) != 1:
                raise ValueError(
                    "MUVS explicit replacement must resolve one preceding "
                    f"unused materializable source row: {sample_id}"
                )
            replacement_index, replacement = explicit_candidates[0]
            selection_rule = (
                "explicit preceding unused annotation row from the same "
                "event, validated against the separately probed replacement "
                "video duration because the target period has no usable row"
            )
        used.add(_signature(replacement))
        sample_offset = int(sample_id.rsplit("-", 1)[1]) - 1
        samples[sample_offset] = replacement
        audit_rows.append(
            {
                "sample_id": sample_id,
                "reason": "planned_frame_time_exceeds_probed_source_duration",
                "source_duration_sec": duration_sec,
                "replacement_source_duration_sec": replacement_duration,
                "original_annotation_row_index": target_index,
                "replacement_annotation_row_index": replacement_index,
                "original_signature": list(_signature(target)),
                "replacement_signature": list(_signature(replacement)),
                "selection_rule": selection_rule,
            }
        )

    repaired_payload = dict(original)
    repaired_payload["samples"] = samples
    repaired = seal_muvs_event_state_plan(repaired_payload)
    audit = {
        "schema_version": "agu.muvs-event-state-plan-repair.v1",
        "purpose": "label_hidden_source_materialization_repair",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "input_plan_sha256": original["artifact_sha256"],
        "output_plan_sha256": repaired["artifact_sha256"],
        "excluded_plan_sha256": sorted(excluded_plan_hashes),
        "repairs": audit_rows,
    }
    audit["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            audit,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(repaired, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit_output": str(args.audit_output),
                "repair_count": len(audit_rows),
                "artifact_sha256": repaired["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
