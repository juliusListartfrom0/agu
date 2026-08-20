"""Bind enrollment face samples to same-frame player tracks and team colors."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def attach_face_player_context(
    candidate_payload: Mapping[str, Any],
    perception_payloads: Sequence[Mapping[str, Any]],
    *,
    perception_sha256s: Sequence[str],
    team_id_map: Mapping[str, str],
    minimum_team_observations: int = 2,
    minimum_team_consensus: float = 0.75,
) -> dict[str, Any]:
    """Associate each face sample with the smallest containing player box."""

    candidate_sha256 = _validate_candidate_payload(candidate_payload)
    if len(perception_payloads) != len(perception_sha256s):
        raise ValueError("perception payloads and hashes must have the same length")
    if minimum_team_observations < 1:
        raise ValueError("minimum team observations must be positive")
    if not 0.5 <= minimum_team_consensus <= 1.0:
        raise ValueError("minimum team consensus must be in [0.5, 1.0]")
    normalized_team_map = {
        str(raw_id).strip(): str(team_id).strip()
        for raw_id, team_id in team_id_map.items()
    }
    if any(not raw_id or not team_id for raw_id, team_id in normalized_team_map.items()):
        raise ValueError("team ID mappings must be non-empty")

    candidate_sources = {
        (str(source.get("filename") or ""), str(source.get("sha256") or "")): source
        for source in candidate_payload.get("sources") or []
    }
    if (
        not candidate_sources
        or ("", "") in candidate_sources
        or len(candidate_sources) != len(candidate_payload.get("sources") or [])
    ):
        raise ValueError("candidate sources must have unique filename and hash bindings")

    detections_by_source_frame: dict[
        tuple[str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    perception_sources = []
    bound_source_ids: set[str] = set()
    for payload, payload_sha256 in zip(perception_payloads, perception_sha256s):
        if payload.get("schema_version") != "agu.official-perception.v1":
            raise ValueError("unsupported official perception schema")
        raw_video = payload.get("raw_video") or {}
        binding = (
            str(raw_video.get("filename") or ""),
            str(raw_video.get("sha256") or ""),
        )
        candidate_source = candidate_sources.get(binding)
        if candidate_source is None:
            raise ValueError("official perception does not bind a candidate source")
        source_video_id = str(candidate_source.get("source_video_id") or "")
        if not source_video_id or source_video_id in bound_source_ids:
            raise ValueError("official perceptions must uniquely bind candidate sources")
        bound_source_ids.add(source_video_id)
        perception_sources.append(
            {
                "source_video_id": source_video_id,
                "filename": binding[0],
                "raw_video_sha256": binding[1],
                "perception_sha256": str(payload_sha256),
            }
        )
        for detection in payload.get("detections") or []:
            if detection.get("object_type") != "player":
                continue
            frame = detection.get("frame")
            if frame is None:
                continue
            detections_by_source_frame[(source_video_id, int(frame))].append(
                detection
            )
    expected_source_ids = {
        str(source.get("source_video_id") or "")
        for source in candidate_payload.get("sources") or []
    }
    if bound_source_ids != expected_source_ids:
        raise ValueError("official perceptions must bind every candidate source")

    cluster_contexts = []
    for cluster in candidate_payload.get("clusters") or []:
        cluster_id = str(cluster.get("cluster_id") or "")
        if not cluster_id:
            raise ValueError("candidate cluster IDs must be non-empty")
        sample_contexts = []
        team_votes: Counter[str] = Counter()
        track_ids: set[str] = set()
        for sample in cluster.get("samples") or []:
            context = _match_sample(
                sample,
                detections_by_source_frame=detections_by_source_frame,
                team_id_map=normalized_team_map,
            )
            if context is None:
                continue
            sample_contexts.append(context)
            if context["team_id"] is not None:
                team_votes[str(context["team_id"])] += 1
            if context["track_id"] is not None:
                track_ids.add(str(context["track_id"]))
        dominant_team_id, dominant_team_fraction = _dominant_team(
            team_votes,
            minimum_observations=minimum_team_observations,
            minimum_consensus=minimum_team_consensus,
        )
        cluster_contexts.append(
            {
                "cluster_id": cluster_id,
                "sample_count": len(cluster.get("samples") or []),
                "matched_sample_count": len(sample_contexts),
                "team_vote_counts": dict(sorted(team_votes.items())),
                "dominant_team_id": dominant_team_id,
                "dominant_team_fraction": dominant_team_fraction,
                "track_ids": sorted(track_ids),
                "samples": sample_contexts,
            }
        )

    output = {
        "schema_version": "agu.face-player-context.v1",
        "benchmark_disjoint": True,
        "runtime_consumable": False,
        "producer": "agu_same_frame_face_player_context",
        "source_candidate_manifest_sha256": candidate_sha256,
        "perception_sources": perception_sources,
        "team_id_map": dict(sorted(normalized_team_map.items())),
        "minimum_team_observations": minimum_team_observations,
        "minimum_team_consensus": minimum_team_consensus,
        "clusters": cluster_contexts,
        "manifest_sha256": "",
    }
    output["manifest_sha256"] = _canonical_sha256(output)
    return output


def select_team_consistent_candidates(
    candidate_payload: Mapping[str, Any],
    context_payload: Mapping[str, Any],
    *,
    team_id: str,
) -> dict[str, Any]:
    """Seal a candidate subset whose player context agrees on one team."""

    candidate_sha256 = _validate_candidate_payload(candidate_payload)
    context_sha256 = _validate_context_payload(context_payload)
    if context_payload.get("source_candidate_manifest_sha256") != candidate_sha256:
        raise ValueError("face player context does not match candidate manifest")
    normalized_team_id = str(team_id).strip()
    if not normalized_team_id:
        raise ValueError("team ID must be non-empty")
    contexts = {
        str(row.get("cluster_id") or ""): row
        for row in context_payload.get("clusters") or []
    }
    selected = [
        dict(cluster)
        for cluster in candidate_payload.get("clusters") or []
        if contexts.get(str(cluster.get("cluster_id") or ""), {}).get(
            "dominant_team_id"
        )
        == normalized_team_id
    ]
    output = {
        "schema_version": "agu.face-enrollment-candidates.v1",
        "benchmark_disjoint": True,
        "producer": "agu_team_consistent_face_candidate_subset",
        "model_id": candidate_payload.get("model_id"),
        "sources": list(candidate_payload.get("sources") or []),
        "config": {
            "selection": "same_frame_player_context_team_consensus",
            "source_candidate_manifest_sha256": candidate_sha256,
            "source_config": dict(candidate_payload.get("config") or {}),
            "team_consistent_selection": {
                "team_id": normalized_team_id,
                "source_context_manifest_sha256": context_sha256,
                "minimum_team_observations": context_payload.get(
                    "minimum_team_observations"
                ),
                "minimum_team_consensus": context_payload.get(
                    "minimum_team_consensus"
                ),
            },
        },
        "clusters": selected,
        "manifest_sha256": "",
    }
    output["manifest_sha256"] = _canonical_sha256(output)
    return output


def _match_sample(
    sample: Mapping[str, Any],
    *,
    detections_by_source_frame: Mapping[
        tuple[str, int], Sequence[Mapping[str, Any]]
    ],
    team_id_map: Mapping[str, str],
) -> dict[str, Any] | None:
    source_video_id = str(sample.get("source_video_id") or "")
    frame = sample.get("frame")
    bbox = sample.get("bbox")
    if not source_video_id or frame is None or not isinstance(bbox, Sequence) or len(bbox) != 4:
        return None
    x, y, width, height = (float(value) for value in bbox)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    containing = []
    for detection in detections_by_source_frame.get(
        (source_video_id, int(frame)), ()
    ):
        player_bbox = detection.get("bbox") or {}
        x1 = float(player_bbox.get("x1", 0.0))
        y1 = float(player_bbox.get("y1", 0.0))
        x2 = float(player_bbox.get("x2", 0.0))
        y2 = float(player_bbox.get("y2", 0.0))
        if x1 <= center_x <= x2 and y1 <= center_y <= y2 and x2 > x1 and y2 > y1:
            containing.append(((x2 - x1) * (y2 - y1), -float(detection.get("confidence") or 0.0), detection))
    if not containing:
        return None
    _, _, detection = min(containing, key=lambda row: (row[0], row[1]))
    raw_team_id = str(detection.get("team_id") or "")
    track_id = detection.get("track_id")
    return {
        "path": str(sample.get("path") or ""),
        "source_video_id": source_video_id,
        "frame": int(frame),
        "detection_id": detection.get("detection_id"),
        "track_id": None if track_id is None else str(track_id),
        "raw_team_id": raw_team_id or None,
        "team_id": team_id_map.get(raw_team_id),
        "player_confidence": float(detection.get("confidence") or 0.0),
        "player_bbox": dict(detection.get("bbox") or {}),
    }


def _dominant_team(
    votes: Counter[str],
    *,
    minimum_observations: int,
    minimum_consensus: float,
) -> tuple[str | None, float | None]:
    total = sum(votes.values())
    if total == 0:
        return None, None
    team_id, count = max(votes.items(), key=lambda row: (row[1], row[0]))
    fraction = count / total
    if total < minimum_observations or fraction < minimum_consensus:
        return None, fraction
    return team_id, fraction


def _validate_candidate_payload(payload: Mapping[str, Any]) -> str:
    if payload.get("schema_version") != "agu.face-enrollment-candidates.v1":
        raise ValueError("unsupported face enrollment candidate schema")
    if payload.get("benchmark_disjoint") is not True:
        raise ValueError("face enrollment candidates must be benchmark-disjoint")
    manifest_sha256 = str(payload.get("manifest_sha256") or "")
    expected = _canonical_sha256({**dict(payload), "manifest_sha256": ""})
    if not manifest_sha256 or manifest_sha256 != expected:
        raise ValueError("face enrollment candidate manifest hash mismatch")
    return manifest_sha256


def _validate_context_payload(payload: Mapping[str, Any]) -> str:
    if payload.get("schema_version") != "agu.face-player-context.v1":
        raise ValueError("unsupported face player context schema")
    if payload.get("benchmark_disjoint") is not True:
        raise ValueError("face player context must be benchmark-disjoint")
    manifest_sha256 = str(payload.get("manifest_sha256") or "")
    expected = _canonical_sha256({**dict(payload), "manifest_sha256": ""})
    if not manifest_sha256 or manifest_sha256 != expected:
        raise ValueError("face player context manifest hash mismatch")
    return manifest_sha256


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
