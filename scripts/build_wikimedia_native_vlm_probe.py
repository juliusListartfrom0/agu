#!/usr/bin/env python3
"""Build a label-free native-video VLM probe from sealed Wikimedia reviews.

The source review plans are label-hidden, while the sealed reviews are used
only to produce a separate, post-inference truth artifact.  Candidate bundle
IDs are deterministic hashes of the raw review window identity; they are not
runtime candidate bundles and the resulting plan is development-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import seal_independent_shot_vlm_plan  # noqa: E402

SOURCE_BY_ID = {
    "hazen": "54dedaee0b04a0bab210b6fe730f857edaf193ec5c28a03a84bf005aa64f5bf9",
    "randolph": "f234d5bfadd0b182c9a15a6d7b206335fcb371e9971cd2bf67313f544dc6a102",
    "vtv": "e5cdcdfd1e49f5a71765d4cece56a4810a563938fca4d407836ea6c12409fbec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, action="append", required=True)
    parser.add_argument("--per-source-positive", type=int, default=1)
    parser.add_argument("--per-source-negative", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(version < 1 for version in args.version):
        raise ValueError("review versions must be positive")
    if args.per_source_positive <= 0 or args.per_source_negative <= 0:
        raise ValueError("per-source class quotas must be positive")

    review_rows: dict[str, dict[str, object]] = {}
    review_labels: dict[str, str] = {}
    for version in sorted(set(args.version)):
        base = (
            ROOT
            / "analysis_outputs"
            / "public_research"
            / f"wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v{version}"
        )
        plan = json.loads((base / "review_plan.json").read_text(encoding="utf-8"))
        sealed = json.loads(
            (base / "review_sealed.json").read_text(encoding="utf-8")
        )
        for row in plan["examples"]:
            review_rows[str(row["review_id"])] = row
        for row in sealed["reviews"]:
            review_labels[str(row["review_id"])] = str(row["shot_sequence"])

    candidates: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {"shot": [], "not_a_shot": []}
    )
    for review_id, label in review_labels.items():
        if label not in {"shot", "not_a_shot"}:
            continue
        row = review_rows.get(review_id)
        if row is None:
            raise ValueError(f"sealed review is missing from its review plan: {review_id}")
        source_sha = str(row["source_video_sha256"])
        source_id = _source_id(source_sha)
        candidate = _candidate_row(row)
        candidates[source_id][label].append(candidate)

    selected: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    for source_id in sorted(SOURCE_BY_ID):
        for label, quota in (
            ("shot", args.per_source_positive),
            ("not_a_shot", args.per_source_negative),
        ):
            choices = sorted(
                candidates[source_id][label],
                key=lambda row: _stable_rank(row),
            )
            if len(choices) < quota:
                raise ValueError(
                    f"source {source_id} has {len(choices)} {label} rows; need {quota}"
                )
            for row in choices[:quota]:
                selected.append(dict(row))
                truth_rows.append(
                    {
                        **row,
                        "event_present": label == "shot",
                    }
                )

    selected.sort(
        key=lambda row: (
            str(row["source_video_sha256"]),
            int(row["start_frame"]),
            str(row["event_id"]),
        )
    )
    truth_rows.sort(
        key=lambda row: (
            str(row["source_video_sha256"]),
            int(row["start_frame"]),
            str(row["event_id"]),
        )
    )
    plan = seal_independent_shot_vlm_plan(
        {
            "selection": {
                "method": "sealed_review_label_stratified_native_probe",
                "source_review_versions": sorted(set(args.version)),
                "source_kind": "three_retained_continuous_originals",
                "labels_or_review_notes_exposed_to_model": False,
                "pilot_only": True,
                "example_count": len(selected),
            },
            "input_contract": {
                "raw_video_clips_only": True,
                "labels_or_review_notes_exposed_to_model": False,
                "native_temporal_position_encoding": True,
                "sample_fps": 2.0,
                "max_pixels": 151200,
            },
            "examples": selected,
        }
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    truth = {
        "schema_version": "agu.wikimedia-native-vlm-probe-truth.v1",
        "purpose": "offline_post_inference_diagnostic_only",
        "source_plan_sha256": plan["plan_sha256"],
        "labels_read_after_plan_sealed": True,
        "rows": truth_rows,
    }
    truth["artifact_sha256"] = _canonical_sha256(truth, "artifact_sha256")
    (output_dir / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "agu.wikimedia-native-vlm-probe-build.v1",
        "source_plan": str(output_dir / "source_plan.json"),
        "source_plan_sha256": plan["plan_sha256"],
        "truth_artifact": str(output_dir / "truth.json"),
        "truth_artifact_sha256": truth["artifact_sha256"],
        "source_review_versions": sorted(set(args.version)),
        "example_count": len(selected),
        "per_source": {
            source_id: {
                "positive": sum(
                    row["event_present"]
                    for row in truth_rows
                    if _source_id(str(row["source_video_sha256"])) == source_id
                ),
                "negative": sum(
                    not row["event_present"]
                    for row in truth_rows
                    if _source_id(str(row["source_video_sha256"])) == source_id
                ),
            }
            for source_id in sorted(SOURCE_BY_ID)
        },
    }
    summary["artifact_sha256"] = _canonical_sha256(summary, "artifact_sha256")
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _candidate_row(row: dict[str, object]) -> dict[str, object]:
    frame_indexes = [int(value) for value in row["frame_indexes"]]
    source_sha = str(row["source_video_sha256"])
    review_id = str(row["review_id"])
    identity = {
        "source_video_sha256": source_sha,
        "review_id": review_id,
        "source_fps": float(row["source_fps"]),
        "frame_indexes": frame_indexes,
    }
    candidate_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_video_sha256": source_sha,
        "source_video_filename": str(row["source_video_filename"]),
        "candidate_bundle_sha256": candidate_sha,
        "event_id": review_id,
        "start_frame": min(frame_indexes),
        "end_frame": max(frame_indexes),
        "source_fps": float(row["source_fps"]),
    }


def _stable_rank(row: dict[str, object]) -> str:
    return hashlib.sha256(
        f"{row['source_video_sha256']}:{row['candidate_bundle_sha256']}:{row['event_id']}".encode(
            "utf-8"
        )
    ).hexdigest()


def _source_id(source_sha: str) -> str:
    for source_id, expected_sha in SOURCE_BY_ID.items():
        if source_sha == expected_sha:
            return source_id
    raise ValueError(f"unknown retained source video SHA: {source_sha}")


def _canonical_sha256(payload: dict[str, object], hash_field: str) -> str:
    normalized = dict(payload)
    normalized.pop(hash_field, None)
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
