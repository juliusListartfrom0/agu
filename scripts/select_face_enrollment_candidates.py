#!/usr/bin/env python3
"""Create a hash-sealed review subset from a face-candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--cluster-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_face_enrollment_candidates(
    payload: Mapping[str, Any],
    cluster_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate a sealed source and seal an explicitly selected review subset."""

    if payload.get("schema_version") != "agu.face-enrollment-candidates.v1":
        raise ValueError("unsupported face enrollment candidate schema")
    if payload.get("benchmark_disjoint") is not True:
        raise ValueError("enrollment candidates must be benchmark-disjoint")
    source_sha256 = str(payload.get("manifest_sha256") or "")
    expected_sha256 = _canonical_sha256({**dict(payload), "manifest_sha256": ""})
    if not source_sha256 or source_sha256 != expected_sha256:
        raise ValueError("face enrollment candidate manifest hash mismatch")

    normalized_ids = [str(value).strip() for value in cluster_ids]
    if not normalized_ids or any(not value for value in normalized_ids):
        raise ValueError("selected cluster IDs must be non-empty")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("selected cluster IDs must be unique")
    available = {str(item.get("cluster_id") or ""): item for item in payload.get("clusters") or []}
    missing = sorted(set(normalized_ids) - set(available))
    if missing:
        raise ValueError(f"unknown face enrollment clusters: {missing}")

    subset = {
        "schema_version": "agu.face-enrollment-candidates.v1",
        "benchmark_disjoint": True,
        "producer": "agu_candidate_review_subset",
        "model_id": payload.get("model_id"),
        "sources": list(payload.get("sources") or []),
        "config": {
            "selection": "explicit_cluster_ids_for_human_or_codex_review",
            "source_candidate_manifest_sha256": source_sha256,
            "source_config": dict(payload.get("config") or {}),
        },
        "clusters": [dict(available[cluster_id]) for cluster_id in normalized_ids],
        "manifest_sha256": "",
    }
    subset["manifest_sha256"] = _canonical_sha256(subset)
    return subset


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    subset = select_face_enrollment_candidates(payload, args.cluster_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_sha256": subset["manifest_sha256"],
                "source_candidate_manifest_sha256": subset["config"]["source_candidate_manifest_sha256"],
                "cluster_count": len(subset["clusters"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
