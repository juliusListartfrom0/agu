#!/usr/bin/env python3
"""Merge reviewed face lists with exhaustive cross-list identity decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_annotated_face_lists(
    manifests: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Any],
) -> dict[str, Any]:
    if len(manifests) < 2:
        raise ValueError("at least two annotated face lists are required")
    source_hashes = []
    available: dict[str, Mapping[str, Any]] = {}
    for source_index, manifest in enumerate(manifests, start=1):
        if manifest.get("schema_version") != "agu.annotated-face-list.v1":
            raise ValueError("unsupported annotated face list schema")
        if manifest.get("benchmark_disjoint") is not True:
            raise ValueError("all annotated face lists must be benchmark-disjoint")
        source_hash = str(manifest.get("source_candidate_manifest_sha256") or "")
        if not source_hash:
            raise ValueError("annotated face list source hash is required")
        source_hashes.append(source_hash)
        source_id = f"source_{source_index:03d}"
        for person in manifest.get("persons") or []:
            person_id = str(person.get("person_id") or "").strip()
            key = f"{source_id}:{person_id}"
            if not person_id or key in available:
                raise ValueError("source person IDs must be non-empty and unique")
            available[key] = person

    if decisions.get("schema_version") != "agu.annotated-face-merge.v1":
        raise ValueError("unsupported annotated face merge decision schema")
    if list(decisions.get("source_candidate_manifest_sha256s") or []) != source_hashes:
        raise ValueError("face merge decisions do not match source manifests")
    reviewer = str(decisions.get("reviewer") or "").strip()
    if not reviewer:
        raise ValueError("face merge reviewer is required")

    consumed: set[str] = set()
    merged_people = []
    output_ids: set[str] = set()
    for raw_person in decisions.get("persons") or []:
        person_id = str(raw_person.get("person_id") or "").strip()
        if not person_id or person_id in output_ids:
            raise ValueError("merged person IDs must be non-empty and unique")
        source_person_ids = [str(value) for value in raw_person.get("source_person_ids") or []]
        if not source_person_ids:
            raise ValueError(f"merged person has no source identities: {person_id}")
        for source_person_id in source_person_ids:
            if source_person_id not in available:
                raise ValueError(f"unknown source face identity: {source_person_id}")
            if source_person_id in consumed:
                raise ValueError(f"source face identity merged more than once: {source_person_id}")
            consumed.add(source_person_id)
        source_people = [available[value] for value in source_person_ids]
        team_ids = {str(item["team_id"]) for item in source_people if item.get("team_id")}
        if len(team_ids) > 1:
            raise ValueError(f"conflicting team IDs for merged person: {person_id}")
        samples = []
        for source_person_id, item in zip(source_person_ids, source_people):
            for sample in item.get("samples") or []:
                normalized_sample = {"path": sample} if isinstance(sample, str) else dict(sample)
                normalized_sample.setdefault("prototype_source_id", source_person_id)
                samples.append(normalized_sample)
        if len(samples) < 2:
            raise ValueError(f"merged person has fewer than two samples: {person_id}")
        merged_people.append(
            {
                "person_id": person_id,
                "team_id": next(iter(team_ids), None),
                "source_person_ids": source_person_ids,
                "samples": samples,
            }
        )
        output_ids.add(person_id)

    missing = sorted(set(available) - consumed)
    if missing:
        raise ValueError(f"face merge review is not exhaustive: {missing}")
    combined_source_hash = hashlib.sha256("|".join(source_hashes).encode()).hexdigest()
    return {
        "schema_version": "agu.annotated-face-list.v1",
        "benchmark_disjoint": True,
        "annotation_producer": reviewer,
        "source_candidate_manifest_sha256": combined_source_hash,
        "source_candidate_manifest_sha256s": source_hashes,
        "persons": merged_people,
    }


def main() -> int:
    args = parse_args()
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in args.manifest]
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    merged = merge_annotated_face_lists(manifests, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"person_count": len(merged["persons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
