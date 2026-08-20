"""Build identity-only rosters from benchmark-disjoint NBA game records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

NBA_HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/{person_id}.png"


@dataclass(frozen=True)
class IdentityRosterSource:
    """One official game record used only to identify a team's players."""

    path: Path
    team_tricode: str
    team_id: str


def build_disjoint_identity_roster(
    sources: Iterable[IdentityRosterSource],
    *,
    forbidden_game_ids: set[str],
    supplemental_identity_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Strip statistics and seal player identities from disjoint game records."""

    normalized_forbidden = {
        str(game_id).strip() for game_id in forbidden_game_ids if str(game_id).strip()
    }
    if not normalized_forbidden:
        raise ValueError("at least one forbidden benchmark game ID is required")

    source_entries: list[dict[str, Any]] = []
    supplemental_source_entries: list[dict[str, Any]] = []
    players_by_id: dict[str, dict[str, str]] = {}
    visual_identity_scope: dict[str, Any] | None = None
    for source in sources:
        source_bytes = source.path.read_bytes()
        rows = [
            json.loads(line)
            for line in source_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"identity roster source is empty: {source.path}")
        game_ids = {str(row.get("game_id") or "").strip() for row in rows}
        game_ids.discard("")
        if len(game_ids) != 1:
            raise ValueError(f"identity roster source must contain exactly one game: {source.path}")
        game_id = next(iter(game_ids))
        if game_id in normalized_forbidden:
            raise ValueError(f"identity roster source is a forbidden benchmark game: {game_id}")

        team_tricode = source.team_tricode.strip().upper()
        team_id = source.team_id.strip()
        if not team_tricode or not team_id:
            raise ValueError("identity roster source team ID and tricode are required")
        selected = [
            row
            for row in rows
            if row.get("row_type") == "player"
            and str(row.get("team_tricode") or "").strip().upper() == team_tricode
        ]
        if not selected:
            raise ValueError(f"identity roster source has no players for {team_tricode}")
        for row in selected:
            if str(row.get("team_id") or "").strip() != team_id:
                raise ValueError(f"identity roster team ID mismatch for {team_tricode}")
            person_id = str(row.get("person_id") or "").strip()
            first_name = str(row.get("first_name") or "").strip()
            family_name = str(row.get("family_name") or "").strip()
            display_name = " ".join(part for part in (first_name, family_name) if part)
            if not person_id or not display_name:
                raise ValueError("every identity roster player requires an ID and name")
            player = {
                "person_id": person_id,
                "display_name": display_name,
                "team_id": team_id,
                "team_tricode": team_tricode,
                "jersey_number": str(row.get("jersey_num") or "").strip(),
                "player_slug": str(row.get("player_slug") or "").strip(),
                "portrait_url": NBA_HEADSHOT_URL.format(person_id=person_id),
            }
            existing = players_by_id.get(person_id)
            if existing is not None and existing != player:
                raise ValueError(f"conflicting identity roster entry for {person_id}")
            players_by_id[person_id] = player

        first_row = rows[0]
        source_entries.append(
            {
                "filename": source.path.name,
                "file_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "game_id": game_id,
                "game_date": str(first_row.get("game_date") or "").strip(),
                "team_id": team_id,
                "team_tricode": team_tricode,
                "source_provider": str(first_row.get("source") or "nba.com").strip(),
            }
        )

    for supplemental_path in supplemental_identity_paths:
        source_bytes = supplemental_path.read_bytes()
        supplement = json.loads(source_bytes)
        if supplement.get("schema_version") != "agu.face-roster-supplement.v1":
            raise ValueError("unsupported identity roster supplement schema")
        if (
            supplement.get("benchmark_disjoint") is not True
            or supplement.get("identity_only") is not True
            or supplement.get("runtime_consumable") is not False
        ):
            raise ValueError("identity roster supplements must be disjoint and identity-only")
        if _contains_key(supplement, "statistics"):
            raise ValueError("identity roster supplements must not contain statistics")
        evidence_sources = list(supplement.get("evidence_sources") or [])
        if not evidence_sources:
            raise ValueError("identity roster supplement evidence sources are required")
        for evidence in evidence_sources:
            if not str(evidence.get("source_url") or "").startswith("https://"):
                raise ValueError("identity roster supplement source URLs must use HTTPS")
        supplemental_players = list(supplement.get("players") or [])
        if not supplemental_players:
            raise ValueError("identity roster supplement players are required")
        for raw_player in supplemental_players:
            player = _normalize_supplemental_player(raw_player)
            person_id = player["person_id"]
            existing = players_by_id.get(person_id)
            if existing is not None and existing != player:
                raise ValueError(f"conflicting identity roster entry for {person_id}")
            players_by_id[person_id] = player
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        supplemental_source_entries.append(
            {
                "filename": supplemental_path.name,
                "file_sha256": source_sha256,
                "evidence_sources": evidence_sources,
            }
        )
        raw_scope = supplement.get("visual_identity_scope")
        if raw_scope is not None:
            if visual_identity_scope is not None:
                raise ValueError("only one visual identity scope is supported")
            visual_identity_scope = {
                **dict(raw_scope),
                "source_file_sha256": source_sha256,
            }

    if not source_entries:
        raise ValueError("at least one identity roster source is required")
    payload: dict[str, Any] = {
        "schema_version": "agu.face-roster.v1",
        "benchmark_disjoint": True,
        "identity_only": True,
        "runtime_consumable": False,
        "answer_fields_removed": True,
        "portrait_rights": {
            "provider": "NBA",
            "license": None,
            "redistribution_permitted": False,
            "usage": "local_identity_enrollment_only",
        },
        "sources": sorted(source_entries, key=lambda item: item["team_tricode"]),
        "players": sorted(
            players_by_id.values(),
            key=lambda item: (item["team_tricode"], item["display_name"], item["person_id"]),
        ),
        "roster_sha256": "",
    }
    if supplemental_source_entries:
        payload["supplemental_sources"] = supplemental_source_entries
    if visual_identity_scope is not None:
        payload["visual_identity_scope"] = visual_identity_scope
        _validate_visual_identity_scope(payload)
    payload["roster_sha256"] = _canonical_sha256(payload)
    return verify_disjoint_identity_roster(payload)


def verify_disjoint_identity_roster(payload: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if an identity roster is unsealed or contains answer fields."""

    if payload.get("schema_version") != "agu.face-roster.v1":
        raise ValueError("unsupported identity roster schema")
    if payload.get("benchmark_disjoint") is not True:
        raise ValueError("identity roster must be benchmark-disjoint")
    if payload.get("identity_only") is not True or payload.get("runtime_consumable") is not False:
        raise ValueError("identity roster must remain identity-only and non-runtime")
    if _contains_key(payload, "statistics"):
        raise ValueError("identity roster must not contain statistics")
    if payload.get("visual_identity_scope") is not None:
        _validate_visual_identity_scope(payload)
    expected = str(payload.get("roster_sha256") or "")
    candidate = {**payload, "roster_sha256": ""}
    if not expected or expected != _canonical_sha256(candidate):
        raise ValueError("identity roster hash mismatch")
    return payload


def identity_coverage_person_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return the roster identities that can be visually registered from enrollment games."""

    verify_disjoint_identity_roster(payload)
    scope = payload.get("visual_identity_scope")
    if scope is None:
        return tuple(
            sorted(str(player["person_id"]) for player in payload.get("players") or [])
        )
    required = tuple(sorted(str(value) for value in scope["required_person_ids"]))
    if not required:
        raise ValueError("visual identity scope requires at least one required person")
    return required


def download_identity_portraits(
    roster: dict[str, Any],
    *,
    output_dir: Path,
    fetch_bytes: Callable[[str], bytes],
    duplicate_policy: Literal["reject", "exclude"] = "reject",
) -> dict[str, Any]:
    """Download sealed NBA identity references for a face-gallery input manifest."""

    verify_disjoint_identity_roster(roster)
    if duplicate_policy not in {"reject", "exclude"}:
        raise ValueError("duplicate portrait policy must be reject or exclude")
    downloaded: list[tuple[dict[str, Any], bytes, str, str]] = []
    people_by_content_sha256: dict[str, list[str]] = {}
    for player in roster.get("players") or []:
        person_id = str(player.get("person_id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", person_id):
            raise ValueError(f"unsafe portrait person ID: {person_id!r}")
        url = str(player.get("portrait_url") or "").strip()
        if not url.startswith("https://cdn.nba.com/headshots/"):
            raise ValueError(f"unsupported identity portrait URL for {person_id}")
        content = fetch_bytes(url)
        extension = _image_extension(content)
        if extension is None:
            raise ValueError(f"portrait response for {person_id} is not a supported image")
        content_sha256 = hashlib.sha256(content).hexdigest()
        people_by_content_sha256.setdefault(content_sha256, []).append(person_id)
        downloaded.append((player, content, extension, content_sha256))

    duplicate_hashes = {
        content_sha256
        for content_sha256, person_ids in people_by_content_sha256.items()
        if len(person_ids) > 1
    }
    if duplicate_hashes and duplicate_policy == "reject":
        duplicate_people = sorted(
            person_id
            for content_sha256 in duplicate_hashes
            for person_id in people_by_content_sha256[content_sha256]
        )
        raise ValueError(
            "identity portrait content reused for multiple people: "
            + ", ".join(duplicate_people)
        )
    excluded = [
        {
            "person_id": str(player["person_id"]),
            "content_sha256": content_sha256,
            "reason": "duplicate_content_across_person_ids",
        }
        for player, _content, _extension, content_sha256 in downloaded
        if content_sha256 in duplicate_hashes
    ]
    downloaded = [
        item for item in downloaded if item[3] not in duplicate_hashes
    ]
    if not downloaded:
        raise ValueError("identity roster has no portrait candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    persons: list[dict[str, Any]] = []
    for player, content, extension, content_sha256 in downloaded:
        person_id = str(player["person_id"])
        url = str(player["portrait_url"])
        path = (output_dir / f"{person_id}{extension}").resolve()
        path.write_bytes(content)
        persons.append(
            {
                "person_id": person_id,
                "display_name": str(player.get("display_name") or "").strip(),
                "team_id": str(player.get("team_id") or "").strip(),
                "team_tricode": str(player.get("team_tricode") or "").strip(),
                "jersey_number": str(player.get("jersey_number") or "").strip(),
                "samples": [
                    {
                        "path": str(path),
                        "sha256": content_sha256,
                        "source_url": url,
                    }
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "agu.annotated-face-list.v1",
        "benchmark_disjoint": True,
        "identity_only": True,
        "runtime_consumable": False,
        "annotation_producer": "agu_official_nba_portrait_identity_v1",
        "source_roster_sha256": roster["roster_sha256"],
        "portrait_rights": roster["portrait_rights"],
        "excluded_duplicate_portraits": excluded,
        "persons": persons,
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def build_single_face_reference_manifest(
    annotated: dict[str, Any],
    *,
    model_id: str,
) -> dict[str, Any]:
    """Convert verified single portraits into an offline-only ranking manifest."""

    if annotated.get("schema_version") != "agu.annotated-face-list.v1":
        raise ValueError("unsupported annotated portrait schema")
    if (
        annotated.get("benchmark_disjoint") is not True
        or annotated.get("identity_only") is not True
        or annotated.get("runtime_consumable") is not False
    ):
        raise ValueError("annotated portraits must be disjoint, identity-only, and offline")
    expected_manifest_sha256 = str(annotated.get("manifest_sha256") or "")
    if expected_manifest_sha256 != _canonical_sha256(
        {**annotated, "manifest_sha256": ""}
    ):
        raise ValueError("annotated portrait manifest hash mismatch")
    normalized_model_id = model_id.strip()
    if not normalized_model_id:
        raise ValueError("single face reference model ID is required")

    references = []
    for person in annotated.get("persons") or []:
        person_id = str(person.get("person_id") or "").strip()
        samples = list(person.get("samples") or [])
        if not person_id or len(samples) != 1:
            raise ValueError("every single face reference requires exactly one portrait")
        sample = samples[0]
        path = Path(str(sample.get("path") or ""))
        expected_sha256 = str(sample.get("sha256") or "")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"single face reference hash mismatch for {person_id}")
        references.append(
            {
                "person_id": person_id,
                "team_id": str(person.get("team_id") or "").strip(),
                "path": str(path),
                "sha256": expected_sha256,
                "source_url": str(sample.get("source_url") or "").strip(),
            }
        )
    if not references:
        raise ValueError("single face reference manifest cannot be empty")
    payload: dict[str, Any] = {
        "schema_version": "agu.single-face-reference-list.v1",
        "benchmark_disjoint": True,
        "runtime_consumable": False,
        "redistribution_permitted": False,
        "retention_policy": "delete_image_bytes_after_offline_ranking",
        "annotation_producer": "agu_official_nba_portrait_identity_v1",
        "model_id": normalized_model_id,
        "references": references,
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _normalize_supplemental_player(raw_player: dict[str, Any]) -> dict[str, str]:
    person_id = str(raw_player.get("person_id") or "").strip()
    display_name = str(raw_player.get("display_name") or "").strip()
    team_id = str(raw_player.get("team_id") or "").strip()
    team_tricode = str(raw_player.get("team_tricode") or "").strip().upper()
    if not person_id or not display_name or not team_id or not team_tricode:
        raise ValueError("supplemental identity players require ID, name, and team")
    return {
        "person_id": person_id,
        "display_name": display_name,
        "team_id": team_id,
        "team_tricode": team_tricode,
        "jersey_number": str(raw_player.get("jersey_number") or "").strip(),
        "player_slug": str(raw_player.get("player_slug") or "").strip(),
        "portrait_url": NBA_HEADSHOT_URL.format(person_id=person_id),
    }


def _validate_visual_identity_scope(payload: dict[str, Any]) -> None:
    scope = payload.get("visual_identity_scope")
    if not isinstance(scope, dict):
        raise ValueError("visual identity scope must be an object")
    if scope.get("basis") != "benchmark_disjoint_enrollment_game_participation":
        raise ValueError("unsupported visual identity scope basis")
    source_game_ids = {
        str(value).strip() for value in scope.get("source_game_ids") or [] if str(value).strip()
    }
    if not source_game_ids:
        raise ValueError("visual identity scope source games are required")
    required = {
        str(value).strip()
        for value in scope.get("required_person_ids") or []
        if str(value).strip()
    }
    not_required_rows = list(scope.get("not_required_persons") or [])
    not_required: set[str] = set()
    for row in not_required_rows:
        person_id = str(row.get("person_id") or "").strip()
        reason = str(row.get("reason") or "").strip()
        if not person_id or not reason:
            raise ValueError("visual identity exemptions require person ID and reason")
        if person_id in not_required:
            raise ValueError("visual identity exemptions must be unique")
        not_required.add(person_id)
    if required & not_required:
        raise ValueError("visual identity required and exempt identities overlap")
    roster_ids = {
        str(player.get("person_id") or "").strip()
        for player in payload.get("players") or []
    }
    if required | not_required != roster_ids:
        raise ValueError("visual identity scope must partition the complete roster")
    source_file_sha256 = str(scope.get("source_file_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_file_sha256):
        raise ValueError("visual identity scope source hash is required")


def _image_extension(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return None


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
