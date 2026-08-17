"""Strict, development-only game and production-family grouping artifact."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

SOURCE_GROUP_SCHEMA = "agu.vru-causal-source-groups.v1"
SOURCE_GROUP_SPEC_SCHEMA = "agu.vru-causal-source-groups-spec.v1"

_PURPOSE = "development_diagnostic_only"
_GAME_ORDER = ("hazen", "randolph", "vtv", "harwood")
_GAME_FAMILIES = {
    "hazen": "hctv",
    "randolph": "hctv",
    "vtv": "vtv",
    "harwood": "hctv",
}
_PRODUCTION_FAMILIES = (
    ("hctv", ("hazen", "randolph", "harwood")),
    ("vtv", ("vtv",)),
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPEC_FIELDS = {"schema_version", "games"}
_MANIFEST_FIELDS = {
    "artifact_sha256",
    "formal_evaluation_eligible",
    "game_group_count",
    "games",
    "production_families",
    "production_family_count",
    "promoted",
    "promotion_eligible",
    "purpose",
    "runtime_consumable",
    "schema_version",
}
_GAME_FIELDS = {
    "game_id",
    "production_family",
    "source_video_sha256",
}
_FAMILY_FIELDS = {"game_ids", "production_family"}


def seal_vru_causal_source_groups(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one exact four-game/two-family development grouping spec."""

    if not isinstance(spec, Mapping) or set(spec) != _SPEC_FIELDS:
        raise ValueError("source-group spec fields are not canonical")
    if spec.get("schema_version") != SOURCE_GROUP_SPEC_SCHEMA:
        raise ValueError("unsupported source-group spec schema")
    games = _canonical_games(spec.get("games"), context="source-group spec")
    payload: dict[str, Any] = {
        "schema_version": SOURCE_GROUP_SCHEMA,
        "purpose": _PURPOSE,
        "runtime_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "game_group_count": len(_GAME_ORDER),
        "production_family_count": len(_PRODUCTION_FAMILIES),
        "games": games,
        "production_families": _canonical_family_rows(),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return verify_vru_causal_source_groups(payload)


def verify_vru_causal_source_groups(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify and return a fresh canonical source-group manifest."""

    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("source-group manifest fields are not canonical")
    if payload.get("schema_version") != SOURCE_GROUP_SCHEMA:
        raise ValueError("unsupported source-group manifest schema")
    if payload.get("purpose") != _PURPOSE:
        raise ValueError("source-group manifest purpose is invalid")
    if any(
        payload.get(field) is not False
        for field in (
            "runtime_consumable",
            "formal_evaluation_eligible",
            "promotion_eligible",
            "promoted",
        )
    ):
        raise ValueError("source-group eligibility flags must remain false")
    _require_exact_count(
        payload.get("game_group_count"),
        len(_GAME_ORDER),
        "game group count",
    )
    _require_exact_count(
        payload.get("production_family_count"),
        len(_PRODUCTION_FAMILIES),
        "production family count",
    )
    games = _canonical_games(payload.get("games"), context="source-group manifest")
    family_rows = _canonical_families(payload.get("production_families"))
    claimed = _require_sha256(
        payload.get("artifact_sha256"),
        "source-group artifact",
    )
    canonical: dict[str, Any] = {
        "schema_version": SOURCE_GROUP_SCHEMA,
        "purpose": _PURPOSE,
        "runtime_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "game_group_count": len(_GAME_ORDER),
        "production_family_count": len(_PRODUCTION_FAMILIES),
        "games": games,
        "production_families": family_rows,
    }
    if claimed != _canonical_sha256(canonical):
        raise ValueError("source-group manifest hash mismatch")
    if expected_artifact_sha256 is not None:
        expected = _require_sha256(
            expected_artifact_sha256,
            "expected source-group artifact",
        )
        if claimed != expected:
            raise ValueError("expected source-group artifact SHA-256 does not match")
    canonical["artifact_sha256"] = claimed
    return canonical


def _canonical_games(value: Any, *, context: str) -> list[dict[str, str]]:
    if type(value) is not list:
        raise ValueError(f"{context} games must be a canonical list")
    if len(value) != len(_GAME_ORDER):
        raise ValueError(f"{context} must contain exactly four games")
    rows: list[dict[str, str]] = []
    seen_game_ids: set[str] = set()
    seen_source_sha256s: set[str] = set()
    for position, raw_row in enumerate(value):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _GAME_FIELDS:
            raise ValueError(f"{context} game fields are not canonical")
        game_id = raw_row.get("game_id")
        expected_game_id = _GAME_ORDER[position]
        if not isinstance(game_id, str) or game_id != expected_game_id:
            raise ValueError(f"{context} game order or IDs are invalid")
        if game_id in seen_game_ids:
            raise ValueError(f"{context} game IDs must be unique")
        source_sha256 = _require_sha256(
            raw_row.get("source_video_sha256"),
            f"{context} source video",
        )
        if source_sha256 in seen_source_sha256s:
            raise ValueError(f"{context} source video SHA-256 values must be unique")
        production_family = raw_row.get("production_family")
        if not isinstance(production_family, str) or production_family != _GAME_FAMILIES[game_id]:
            raise ValueError(f"{context} production family mapping is invalid")
        rows.append(
            {
                "game_id": game_id,
                "source_video_sha256": source_sha256,
                "production_family": production_family,
            }
        )
        seen_game_ids.add(game_id)
        seen_source_sha256s.add(source_sha256)
    return rows


def _canonical_families(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise ValueError("production families must be a canonical list")
    if len(value) != len(_PRODUCTION_FAMILIES):
        raise ValueError("source groups must contain exactly two production families")
    rows: list[dict[str, Any]] = []
    for raw_row, (expected_family, expected_game_ids) in zip(
        value,
        _PRODUCTION_FAMILIES,
        strict=True,
    ):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _FAMILY_FIELDS:
            raise ValueError("production family fields are not canonical")
        family = raw_row.get("production_family")
        game_ids = raw_row.get("game_ids")
        if family != expected_family or type(game_ids) is not list:
            raise ValueError("production family order or membership is invalid")
        if game_ids != list(expected_game_ids) or any(not isinstance(game_id, str) for game_id in game_ids):
            raise ValueError("production family game membership is invalid")
        rows.append(
            {
                "production_family": expected_family,
                "game_ids": list(expected_game_ids),
            }
        )
    return rows


def _canonical_family_rows() -> list[dict[str, Any]]:
    return [
        {
            "production_family": family,
            "game_ids": list(game_ids),
        }
        for family, game_ids in _PRODUCTION_FAMILIES
    ]


def _require_exact_count(value: Any, expected: int, field: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{field} must be the exact integer {expected}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} SHA-256 must be a lowercase hex digest")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
