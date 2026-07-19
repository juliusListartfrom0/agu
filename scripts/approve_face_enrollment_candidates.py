#!/usr/bin/env python3
"""Convert exhaustive face-cluster review decisions into an annotated face list."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def approve_candidates(
    candidate_payload: Mapping[str, Any],
    decision_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if candidate_payload.get("schema_version") != "agu.face-enrollment-candidates.v1":
        raise ValueError("unsupported face enrollment candidate schema")
    if candidate_payload.get("benchmark_disjoint") is not True:
        raise ValueError("enrollment candidates must be benchmark-disjoint")
    candidate_sha256 = str(candidate_payload.get("manifest_sha256") or "")
    expected = _canonical_sha256({**dict(candidate_payload), "manifest_sha256": ""})
    if not candidate_sha256 or candidate_sha256 != expected:
        raise ValueError("face enrollment candidate manifest hash mismatch")
    if decision_payload.get("schema_version") != "agu.face-enrollment-review.v1":
        raise ValueError("unsupported face enrollment review schema")
    if decision_payload.get("candidate_manifest_sha256") != candidate_sha256:
        raise ValueError("face enrollment decisions do not match candidate manifest")
    reviewer = str(decision_payload.get("reviewer") or "").strip()
    if not reviewer:
        raise ValueError("face enrollment reviewer is required")

    clusters = {str(item.get("cluster_id") or ""): item for item in candidate_payload.get("clusters") or []}
    if "" in clusters or not clusters:
        raise ValueError("candidate cluster IDs must be non-empty")
    consumed: set[str] = set()
    persons = []
    person_ids: set[str] = set()
    for raw_person in decision_payload.get("persons") or []:
        person_id = str(raw_person.get("person_id") or "").strip()
        if not person_id or person_id in person_ids:
            raise ValueError("approved person IDs must be non-empty and unique")
        cluster_ids = [str(value) for value in raw_person.get("cluster_ids") or []]
        if not cluster_ids:
            raise ValueError(f"approved person has no clusters: {person_id}")
        _claim_clusters(cluster_ids, clusters=clusters, consumed=consumed)
        samples = [
            {"path": str(sample["path"])}
            for cluster_id in cluster_ids
            for sample in clusters[cluster_id].get("samples") or []
        ]
        if len(samples) < 2:
            raise ValueError(f"approved person has fewer than two samples: {person_id}")
        persons.append(
            {
                "person_id": person_id,
                "team_id": raw_person.get("team_id"),
                "source_cluster_ids": cluster_ids,
                "samples": samples,
            }
        )
        person_ids.add(person_id)
    rejected = [str(value) for value in decision_payload.get("rejected_cluster_ids") or []]
    _claim_clusters(rejected, clusters=clusters, consumed=consumed)
    missing = sorted(set(clusters) - consumed)
    if missing:
        raise ValueError(f"face enrollment review is not exhaustive: {missing}")
    return {
        "schema_version": "agu.annotated-face-list.v1",
        "benchmark_disjoint": True,
        "annotation_producer": reviewer,
        "source_candidate_manifest_sha256": candidate_sha256,
        "persons": persons,
    }


def _claim_clusters(
    cluster_ids: list[str],
    *,
    clusters: Mapping[str, Any],
    consumed: set[str],
) -> None:
    for cluster_id in cluster_ids:
        if cluster_id not in clusters:
            raise ValueError(f"unknown face enrollment cluster: {cluster_id}")
        if cluster_id in consumed:
            raise ValueError(f"face enrollment cluster reviewed more than once: {cluster_id}")
        consumed.add(cluster_id)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    approved = approve_candidates(candidates, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(approved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_candidate_manifest_sha256": approved["source_candidate_manifest_sha256"],
                "person_count": len(approved["persons"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
