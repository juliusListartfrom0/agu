"""License-aware adapters for open basketball action annotations.

The normalized catalogs are training inputs, never runtime predictions.  Media
rights are tracked separately from annotation licenses because an open metadata
repository does not automatically grant redistribution rights for linked video.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

OPEN_ACTION_DATASET_SCHEMA = "agu.open-action-dataset.v1"
BARD_DATASET_ID = "bard-2026"
BARD_ANNOTATION_LICENSE = "CC-BY-4.0"
BARD_SOURCE_REPOSITORY = "https://github.com/GabrieleGiudic/BARD"

BARD_ACTION_TYPES = {
    "2PT Shot": "field_goal_attempt",
    "3PT Shot": "field_goal_attempt",
    "Free Throw": "free_throw_attempt",
    "Rebound": "rebound",
    "Steal": "steal",
    "Turnover": "turnover",
    "Block": "block",
    "Foul": "foul",
    "Violation": "violation",
}


def import_bard_annotations(
    annotation_path: str | Path,
    *,
    repository_revision: str,
) -> dict[str, Any]:
    """Normalize BARD's semicolon CSV without downloading linked NBA media."""

    path = Path(annotation_path)
    if not repository_revision.strip():
        raise ValueError("BARD import requires an immutable repository revision")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames != ["urls", "actions", "numerosity"]:
            raise ValueError("unsupported BARD CSV columns")
        for row_index, row in enumerate(reader, start=2):
            actions = _parse_bard_actions(row.get("actions"), row_index=row_index)
            expected = int(row.get("numerosity") or 0)
            if expected != len(actions):
                raise ValueError(f"BARD row {row_index} numerosity mismatch")
            clip_ref = str(row.get("urls") or "").strip()
            if not clip_ref:
                raise ValueError(f"BARD row {row_index} has no clip reference")
            records.append(
                {
                    "record_id": f"bard-{row_index - 1:05d}",
                    "clip_reference": clip_ref,
                    "source_locator": _bard_source_locator(clip_ref),
                    "actions": [_normalize_bard_action(item) for item in actions],
                }
            )
    if not records:
        raise ValueError("BARD annotation file is empty")
    payload: dict[str, Any] = {
        "schema_version": OPEN_ACTION_DATASET_SCHEMA,
        "dataset_id": BARD_DATASET_ID,
        "purpose": "model_pretraining_only",
        "runtime_consumable": False,
        "annotation_license": BARD_ANNOTATION_LICENSE,
        "attribution_required": True,
        "source_repository": BARD_SOURCE_REPOSITORY,
        "repository_revision": repository_revision,
        "annotation_filename": path.name,
        "annotation_sha256": _file_sha256(path),
        "media_included": False,
        "media_rights_verified": False,
        "media_policy": "external_authorized_media_only",
        "weak_temporal_alignment": True,
        "actor_boxes_included": False,
        "records": records,
    }
    payload["catalog_sha256"] = _json_sha256(payload)
    return payload


def verify_open_action_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    catalog = dict(payload)
    claimed_hash = str(catalog.pop("catalog_sha256", ""))
    if catalog.get("schema_version") != OPEN_ACTION_DATASET_SCHEMA:
        raise ValueError("unsupported open action dataset schema")
    if catalog.get("runtime_consumable") is not False:
        raise ValueError("open action annotations must not be runtime-consumable")
    if catalog.get("media_rights_verified") is not False:
        raise ValueError("BARD adapter must not claim rights to externally linked media")
    if _json_sha256(catalog) != claimed_hash:
        raise ValueError("open action dataset catalog hash mismatch")
    catalog["catalog_sha256"] = claimed_hash
    return catalog


def _parse_bard_actions(value: Any, *, row_index: int) -> list[Mapping[str, Any]]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"BARD row {row_index} has malformed actions") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, Mapping) for item in parsed):
        raise ValueError(f"BARD row {row_index} actions must be a list of objects")
    return parsed


def _normalize_bard_action(item: Mapping[str, Any]) -> dict[str, Any]:
    source_action = str(item.get("action") or "")
    event_type = BARD_ACTION_TYPES.get(source_action)
    if event_type is None:
        raise ValueError(f"unsupported BARD action: {source_action}")
    shot_value = 2 if source_action == "2PT Shot" else 3 if source_action == "3PT Shot" else None
    return {
        "event_type": event_type,
        "source_action": source_action,
        "jersey_number": str(item.get("player") or ""),
        "jersey_color": str(item.get("color") or item.get("jersey_color") or ""),
        "shot_value": shot_value,
        "made": item.get("result"),
        "assisted": item.get("assisted"),
        "related_jersey_number": (
            str(item["other_player"]) if item.get("other_player") is not None else None
        ),
    }


def _bard_source_locator(clip_ref: str) -> dict[str, str]:
    parsed = urlparse(clip_ref)
    query = parse_qs(parsed.query)
    return {
        "game_id": str((query.get("GameID") or [""])[0]),
        "event_id": str((query.get("GameEventID") or [""])[0]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
