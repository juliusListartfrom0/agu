from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from app.analysis.vru_causal_source_groups import (
    SOURCE_GROUP_SCHEMA,
    SOURCE_GROUP_SPEC_SCHEMA,
    seal_vru_causal_source_groups,
    verify_vru_causal_source_groups,
)
from scripts import seal_vru_causal_source_groups as seal_cli

GAME_SHA256S = {
    "hazen": "1" * 64,
    "randolph": "2" * 64,
    "vtv": "3" * 64,
    "harwood": "4" * 64,
}
GAME_FAMILIES = {
    "hazen": "hctv",
    "randolph": "hctv",
    "vtv": "vtv",
    "harwood": "hctv",
}
GAME_ORDER = ("hazen", "randolph", "vtv", "harwood")


def _spec() -> dict[str, object]:
    return {
        "schema_version": SOURCE_GROUP_SPEC_SCHEMA,
        "games": [
            {
                "game_id": game_id,
                "source_video_sha256": GAME_SHA256S[game_id],
                "production_family": GAME_FAMILIES[game_id],
            }
            for game_id in GAME_ORDER
        ],
    }


def _canonical_sha256(payload: dict[str, object]) -> str:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("artifact_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reseal(payload: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _canonical_sha256(result)
    return result


def _write_spec(path: Path, payload: object | None = None) -> None:
    path.write_text(
        json.dumps(_spec() if payload is None else payload, indent=2) + "\n",
        encoding="utf-8",
    )


def test_seal_and_verify_return_the_exact_canonical_contract() -> None:
    sealed = seal_vru_causal_source_groups(_spec())

    assert sealed == {
        "schema_version": SOURCE_GROUP_SCHEMA,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "game_group_count": 4,
        "production_family_count": 2,
        "games": [
            {
                "game_id": game_id,
                "source_video_sha256": GAME_SHA256S[game_id],
                "production_family": GAME_FAMILIES[game_id],
            }
            for game_id in GAME_ORDER
        ],
        "production_families": [
            {
                "production_family": "hctv",
                "game_ids": ["hazen", "randolph", "harwood"],
            },
            {"production_family": "vtv", "game_ids": ["vtv"]},
        ],
        "artifact_sha256": sealed["artifact_sha256"],
    }
    assert sealed["artifact_sha256"] == _canonical_sha256(sealed)
    assert (
        verify_vru_causal_source_groups(
            sealed,
            expected_artifact_sha256=str(sealed["artifact_sha256"]),
        )
        == sealed
    )


def test_sealing_is_deterministic_and_does_not_alias_mutable_input() -> None:
    spec = _spec()
    first = seal_vru_causal_source_groups(spec)
    second = seal_vru_causal_source_groups(copy.deepcopy(spec))

    assert first == second
    assert first is not second
    assert first["games"] is not spec["games"]
    spec["games"][0]["source_video_sha256"] = "f" * 64  # type: ignore[index]
    assert first["games"][0]["source_video_sha256"] == "1" * 64  # type: ignore[index]


@pytest.mark.parametrize(
    ("location", "field"),
    (
        ("top", "purpose"),
        ("top", "artifact_sha256"),
        ("game", "production_family"),
        ("family", "game_ids"),
    ),
)
@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_verifier_rejects_missing_or_unknown_fields(
    location: str,
    field: str,
    mutation: str,
) -> None:
    payload = seal_vru_causal_source_groups(_spec())
    if location == "top":
        target = payload
    elif location == "game":
        target = payload["games"][0]  # type: ignore[index]
    else:
        target = payload["production_families"][0]  # type: ignore[index]
    if mutation == "missing":
        target.pop(field)
    else:
        target["unknown_field"] = "not allowed"
    if field != "artifact_sha256":
        payload = _reseal(payload)

    with pytest.raises(ValueError, match="fields|canonical|artifact"):
        verify_vru_causal_source_groups(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("game_group_count", True),
        ("production_family_count", 2.0),
        ("games", {"not": "a list"}),
        ("production_families", "hctv,vtv"),
        ("runtime_consumable", 0),
        ("formal_evaluation_eligible", None),
        ("promotion_eligible", "false"),
        ("promoted", []),
    ),
)
def test_verifier_rejects_type_confusion(field: str, value: object) -> None:
    payload = seal_vru_causal_source_groups(_spec())
    payload[field] = value
    payload = _reseal(payload)

    with pytest.raises(ValueError, match="invalid|integer|list|false"):
        verify_vru_causal_source_groups(payload)


def test_verifier_rejects_internal_or_external_sha_mismatch() -> None:
    payload = seal_vru_causal_source_groups(_spec())

    payload["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_vru_causal_source_groups(payload)

    payload = seal_vru_causal_source_groups(_spec())
    with pytest.raises(ValueError, match="expected.*does not match"):
        verify_vru_causal_source_groups(
            payload,
            expected_artifact_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        "third_family",
        "wrong_known_family",
        "duplicate_game_id",
        "duplicate_source_sha",
        "missing_game",
        "extra_game",
        "game_order",
        "family_order",
        "family_membership",
        "wrong_game_count",
        "wrong_family_count",
    ),
)
def test_self_resealing_cannot_change_the_frozen_group_semantics(
    mutate: str,
) -> None:
    payload = seal_vru_causal_source_groups(_spec())
    games = payload["games"]
    families = payload["production_families"]
    if mutate == "third_family":
        games[3]["production_family"] = "third"  # type: ignore[index]
    elif mutate == "wrong_known_family":
        games[3]["production_family"] = "vtv"  # type: ignore[index]
    elif mutate == "duplicate_game_id":
        games[3]["game_id"] = "hazen"  # type: ignore[index]
    elif mutate == "duplicate_source_sha":
        games[3]["source_video_sha256"] = games[0]["source_video_sha256"]  # type: ignore[index]
    elif mutate == "missing_game":
        games.pop()  # type: ignore[union-attr]
        payload["game_group_count"] = 3
    elif mutate == "extra_game":
        games.append(  # type: ignore[union-attr]
            {
                "game_id": "extra",
                "source_video_sha256": "e" * 64,
                "production_family": "hctv",
            }
        )
        payload["game_group_count"] = 5
    elif mutate == "game_order":
        games[0], games[1] = games[1], games[0]  # type: ignore[index]
    elif mutate == "family_order":
        families[0], families[1] = families[1], families[0]  # type: ignore[index]
    elif mutate == "family_membership":
        families[0]["game_ids"] = ["hazen", "randolph"]  # type: ignore[index]
    elif mutate == "wrong_game_count":
        payload["game_group_count"] = 5
    else:
        payload["production_family_count"] = 3
    payload = _reseal(payload)

    with pytest.raises(ValueError, match="game|famil|source|count|order"):
        verify_vru_causal_source_groups(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_top_field",
        "unknown_game_field",
        "wrong_spec_schema",
        "wrong_order",
        "duplicate_game",
        "duplicate_sha",
        "third_family",
        "bool_sha",
        "games_mapping",
    ),
)
def test_sealer_rejects_noncanonical_specs(mutation: str) -> None:
    spec = _spec()
    games = spec["games"]
    if mutation == "unknown_top_field":
        spec["unknown"] = True
    elif mutation == "unknown_game_field":
        games[0]["unknown"] = True  # type: ignore[index]
    elif mutation == "wrong_spec_schema":
        spec["schema_version"] = SOURCE_GROUP_SCHEMA
    elif mutation == "wrong_order":
        games[0], games[1] = games[1], games[0]  # type: ignore[index]
    elif mutation == "duplicate_game":
        games[3]["game_id"] = "hazen"  # type: ignore[index]
    elif mutation == "duplicate_sha":
        games[3]["source_video_sha256"] = "1" * 64  # type: ignore[index]
    elif mutation == "third_family":
        games[3]["production_family"] = "other"  # type: ignore[index]
    elif mutation == "bool_sha":
        games[0]["source_video_sha256"] = True  # type: ignore[index]
    else:
        spec["games"] = {"hazen": "1" * 64}

    with pytest.raises(ValueError, match="spec|game|source|family|fields|list"):
        seal_vru_causal_source_groups(spec)


def test_cli_generates_a_deterministic_sealed_manifest(tmp_path: Path) -> None:
    spec_path = tmp_path / "source-groups-spec.json"
    output_path = tmp_path / "source-groups.json"
    _write_spec(spec_path)
    argv = [
        "--spec",
        str(spec_path),
        "--output",
        str(output_path),
    ]

    assert seal_cli.main(argv) == 0
    first_bytes = output_path.read_bytes()
    verified = verify_vru_causal_source_groups(
        json.loads(first_bytes),
    )
    assert verified["game_group_count"] == 4
    assert verified["production_family_count"] == 2
    assert seal_cli.main(argv) == 0
    assert output_path.read_bytes() == first_bytes


@pytest.mark.parametrize(
    "bad_json",
    (
        '["not", "an", "object"]',
        '{"schema_version":"agu.vru-causal-source-groups-spec.v1",'
        '"schema_version":"agu.vru-causal-source-groups-spec.v1","games":[]}',
    ),
)
def test_cli_rejects_nonobject_or_duplicate_key_json_specs(
    tmp_path: Path,
    bad_json: str,
) -> None:
    spec_path = tmp_path / "bad.json"
    output_path = tmp_path / "out.json"
    spec_path.write_text(bad_json, encoding="utf-8")

    with pytest.raises(ValueError, match="object|duplicate"):
        seal_cli.main(["--spec", str(spec_path), "--output", str(output_path)])
    assert not output_path.exists()


@pytest.mark.parametrize(
    "alias_kind",
    (
        "direct",
        "output_symlink",
        "input_symlink",
        "hardlink",
        "casefold",
        "descendant",
        "ancestor_symlink",
    ),
)
def test_cli_rejects_input_output_aliases_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    output_path = tmp_path / "output.json"
    if alias_kind == "direct":
        output_path = spec_path
    elif alias_kind == "output_symlink":
        output_path.symlink_to(spec_path)
    elif alias_kind == "input_symlink":
        physical = tmp_path / "physical.json"
        spec_path.replace(physical)
        spec_path.symlink_to(physical)
        output_path = physical
    elif alias_kind == "hardlink":
        os.link(spec_path, output_path)
    elif alias_kind == "casefold":
        output_path = spec_path.with_name("SPEC.JSON")
    elif alias_kind == "descendant":
        output_path = spec_path / "output.json"
    else:
        nested = tmp_path / "nested"
        nested.mkdir()
        alias = tmp_path / "nested-alias"
        alias.symlink_to(nested, target_is_directory=True)
        spec_path = nested
        output_path = alias / "output.json"

    def reject_read(_path: Path) -> dict[str, object]:
        raise AssertionError("path aliases must fail before reading the spec")

    monkeypatch.setattr(seal_cli, "_read_json_object", reject_read)

    with pytest.raises(ValueError, match="alias|ancestor|descendant|disjoint"):
        seal_cli.main(["--spec", str(spec_path), "--output", str(output_path)])


def test_atomic_writer_uses_random_same_directory_temp_fsync_and_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "groups.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    original_replace = Path.replace
    events: list[str] = []
    temporary_names: list[str] = []

    def observe_fsync(file_descriptor: int) -> None:
        assert file_descriptor >= 0
        events.append("fsync")

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.name.startswith(f".{destination.name}.")
        assert source.suffix == ".tmp"
        assert source != destination
        temporary_names.append(source.name)
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        events.append("replace")
        return original_replace(source, target)

    monkeypatch.setattr(seal_cli.os, "fsync", observe_fsync)
    monkeypatch.setattr(Path, "replace", observe_replace)

    seal_cli._write_json_atomic(destination, {"new": True})

    assert events == ["fsync", "replace"]
    assert len(temporary_names) == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writer_cleans_random_temp_and_preserves_destination_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "groups.json"
    destination.write_text("old artifact\n", encoding="utf-8")

    def reject_replace(_source: Path, _target: Path) -> Path:
        raise OSError("injected replace failure")

    monkeypatch.setattr(Path, "replace", reject_replace)

    with pytest.raises(OSError, match="injected"):
        seal_cli._write_json_atomic(destination, {"new": True})

    assert destination.read_text(encoding="utf-8") == "old artifact\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_help_uses_the_canonical_python_entrypoint_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["seal_vru_causal_source_groups.py", "--help"],
    )
    with pytest.raises(SystemExit) as exc_info:
        seal_cli.main()
    assert exc_info.value.code == 0
