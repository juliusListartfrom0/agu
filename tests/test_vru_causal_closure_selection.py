from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import scripts.build_vru_causal_closure_selection as closure_cli
from scripts.build_vru_causal_closure_selection import (
    build_causal_closure_selection,
    verify_causal_closure_selection,
)

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = ROOT / "analysis_outputs" / "public_research"
SOURCE_MANIFEST = (
    ANNOTATION_ROOT
    / "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v35_rv"
    / "source_manifest.json"
)

EXPECTED_CENTERS = {
    "hazen": [
        331.998333,
        461.994867,
        480.012867,
        1682.013667,
        2101.9999,
        2399.9976,
        3282.012067,
        3491.9885,
    ],
    "randolph": [1370.0, 1520.0, 1612.0, 2930.0, 3162.0, 4000.0, 4582.0, 5420.0],
    "vtv": [632.0, 1382.0, 1822.0, 4802.0, 4842.0, 6492.0, 7232.0, 9362.0],
}
EXPECTED_ARTIFACT_SHA256 = "dfe9798aef33dfa993ced6a33192d30cd3e1d97c51ed3df7837008b732bcd38d"

FORBIDDEN_FIELDS = {
    "confidence",
    "event_present",
    "evidence",
    "ground_truth",
    "label",
    "labels",
    "notes",
    "outcome",
    "prior_disposition",
    "release_frame",
    "release_position",
    "review_note",
    "rim_frame",
    "rim_position",
    "shot_sequence",
    "target",
    "training_eligible",
}


def _build() -> dict[str, Any]:
    return build_causal_closure_selection(
        annotation_root=ANNOTATION_ROOT,
        source_manifest_path=SOURCE_MANIFEST,
        max_version=41,
    )


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(map(_all_keys, value.values())))
    if isinstance(value, list):
        return set().union(*(map(_all_keys, value)))
    return set()


def test_selection_freezes_the_audited_24_centers_without_label_fields() -> None:
    artifact = _build()

    assert artifact["runtime_consumable"] is False
    assert artifact["training_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    assert artifact["labels_hidden_from_reviewer"] is True
    assert artifact["prior_labels_used_for_stratified_selection"] is True
    assert artifact["prior_labels_used_as_final_8s_targets"] is False
    assert artifact["formal_evaluation_eligible"] is False
    assert (
        artifact["selection_provenance_verification"]
        == "structural_only_without_source_inputs"
    )
    assert len(artifact["windows"]) == 24
    assert not FORBIDDEN_FIELDS.intersection(_all_keys(artifact))

    observed = {
        source_id: [
            row["center_seconds"]
            for row in artifact["windows"]
            if row["source_id"] == source_id
        ]
        for source_id in EXPECTED_CENTERS
    }
    assert observed == EXPECTED_CENTERS
    assert [row["review_id"] for row in artifact["windows"]] == [
        f"closure-{source_id}-{index:04d}"
        for source_id in ("hazen", "randolph", "vtv")
        for index in range(1, 9)
    ]


def test_selection_records_non_identifying_dedup_counts_and_excludes_smoke() -> None:
    artifact = _build()

    assert artifact["audit"]["included_input_count"] == 39
    assert artifact["audit"]["excluded_input_ids"] == ["v22_smoke"]
    assert artifact["audit"]["observation_count"] == 942
    assert artifact["audit"]["physical_window_count"] == 918
    assert artifact["audit"]["collision_group_count"] == 18
    assert artifact["audit"]["conflicting_physical_window_count"] == 3
    assert artifact["audit"]["candidate_physical_window_count"] == 182
    assert artifact["audit"]["selectable_physical_window_count"] == 182
    assert "conflicts" not in artifact["audit"]
    assert "distinct_observation_sha256s" not in _all_keys(artifact)


def test_selection_honors_source_prior_and_temporal_quotas_without_overlap() -> None:
    artifact = _build()

    assert artifact["selection"]["prior_pool_quota_pairs"] == {
        "hazen": [4, 4],
        "randolph": [3, 5],
        "vtv": [4, 4],
    }
    assert artifact["selection"]["temporal_bucket_quotas"] == {
        source_id: {"early": 3, "middle": 2, "late": 3}
        for source_id in EXPECTED_CENTERS
    }
    assert artifact["selection"]["interval_semantics"] == "half_open"
    assert artifact["selection"]["window_seconds"] == 8.0
    assert artifact["selection"]["sample_rate_hz"] == 8.0
    assert artifact["selection"]["sample_count"] == 64
    assert artifact["selection"]["ranking_direction"] == "ascending_sha256"
    assert artifact["selection"]["physical_window_supersession"] == (
        "highest_formal_version_then_input_id_then_review_id"
    )

    for source_id in EXPECTED_CENTERS:
        rows = [row for row in artifact["windows"] if row["source_id"] == source_id]
        assert Counter(row["temporal_bucket"] for row in rows) == {
            "early": 3,
            "middle": 2,
            "late": 3,
        }
        for left, right in zip(rows, rows[1:], strict=False):
            assert left["end_seconds"] <= right["start_seconds"]


def test_selection_is_deterministic_and_canonically_sealed() -> None:
    first = _build()
    second = _build()
    assert first == second
    assert first["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert verify_causal_closure_selection(first) == first

    unsigned = copy.deepcopy(first)
    claimed = unsigned.pop("artifact_sha256")
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == claimed


def test_selection_verifier_rejects_a_reintroduced_label_field() -> None:
    artifact = _build()
    artifact["windows"][0]["outcome"] = "made"

    with pytest.raises(ValueError, match="label-bearing"):
        verify_causal_closure_selection(artifact)


@pytest.mark.parametrize("location", ("top", "audit", "window"))
def test_selection_verifier_rejects_noncanonical_hidden_payloads(
    location: str,
) -> None:
    artifact = _build()
    if location == "top":
        artifact["prior_answers"] = {"closure-hazen-0001": "shot"}
    elif location == "audit":
        artifact["audit"]["prior_answers"] = ["shot"]
    else:
        artifact["windows"][0]["prior_answers"] = ["shot"]
    artifact["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "artifact_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="fields|canonical"):
        verify_causal_closure_selection(artifact)


def test_selection_verifier_rejects_duplicate_physical_windows() -> None:
    artifact = _build()
    first = artifact["windows"][0]
    second = artifact["windows"][1]
    for field in (
        "anchor_frame",
        "center_seconds",
        "end_seconds",
        "ranking_sha256",
        "source_video_sha256",
        "start_seconds",
    ):
        second[field] = first[field]
    artifact["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "artifact_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="globally unique"):
        verify_causal_closure_selection(artifact)


@pytest.mark.parametrize("field", ("clip_id", "source_video_filename"))
def test_selection_verifier_rejects_nested_payload_in_scalar_window_field(
    field: str,
) -> None:
    artifact = _build()
    artifact["windows"][0][field] = {
        "prior_answers": {"closure-hazen-0001": "shot/made"}
    }
    artifact["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "artifact_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="clip ID|filename|scalar"):
        verify_causal_closure_selection(artifact)


def test_selection_verifier_rejects_nested_payload_in_input_id() -> None:
    artifact = _build()
    artifact["inputs"][0]["input_id"] = {
        "prior_answers": {"result": "shot/made"}
    }
    artifact["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "artifact_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="input ID|provenance"):
        verify_causal_closure_selection(artifact)


@pytest.mark.parametrize(
    "input_name", ("source_manifest", "review_plan", "sealed_review")
)
@pytest.mark.parametrize("alias_kind", ("direct", "output_symlink"))
def test_selection_cli_rejects_output_input_alias_before_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    annotation_root = tmp_path / "annotations"
    annotation_directory = (
        annotation_root
        / "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v3"
    )
    annotation_directory.mkdir(parents=True)
    input_paths = {
        "source_manifest": tmp_path / "source-manifest.json",
        "review_plan": annotation_directory / "review_plan.json",
        "sealed_review": annotation_directory / "review_sealed.json",
    }
    for name, path in input_paths.items():
        path.write_text(f"{name} evidence\n", encoding="utf-8")
    output = input_paths[input_name]
    if alias_kind == "output_symlink":
        output = tmp_path / f"{input_name}-output-alias.json"
        output.symlink_to(input_paths[input_name])
    snapshots = {name: path.read_bytes() for name, path in input_paths.items()}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_vru_causal_closure_selection.py",
            "--annotation-root",
            str(annotation_root),
            "--source-manifest",
            str(input_paths["source_manifest"]),
            "--max-version",
            "41",
            "--output",
            str(output),
        ],
    )

    def reject_computation(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("output/input alias must be rejected before computation")

    monkeypatch.setattr(
        closure_cli,
        "build_causal_closure_selection",
        reject_computation,
    )

    with pytest.raises(ValueError, match="output.*input"):
        closure_cli.main()

    assert {
        name: path.read_bytes() for name, path in input_paths.items()
    } == snapshots


def test_selection_json_writer_stages_and_fsyncs_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "selection.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    writer = getattr(closure_cli, "_write_json_atomic", None)
    assert callable(writer)
    original_replace = Path.replace
    replace_observations: list[Path] = []
    fsync_calls: list[int] = []

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.suffix == ".tmp"
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": True}
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        replace_observations.append(source)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    monkeypatch.setattr(
        closure_cli.os,
        "fsync",
        lambda file_descriptor: fsync_calls.append(file_descriptor),
    )

    writer(destination, {"new": True})

    assert replace_observations
    assert fsync_calls
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_selection_cli_uses_frozen_resolved_paths_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    first_destination = tmp_path / "first-destination"
    second_destination = tmp_path / "second-destination"
    first_destination.mkdir()
    second_destination.mkdir()
    output_parent = tmp_path / "output-parent"
    output_parent.symlink_to(first_destination, target_is_directory=True)
    output = output_parent / "selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_vru_causal_closure_selection.py",
            "--annotation-root",
            str(annotation_root),
            "--source-manifest",
            str(source_manifest),
            "--max-version",
            "41",
            "--output",
            str(output),
        ],
    )

    def redirect_after_validation(
        *,
        annotation_root: Path,
        source_manifest_path: Path,
        max_version: int,
    ) -> dict[str, object]:
        assert annotation_root == annotation_root.resolve()
        assert source_manifest_path == source_manifest_path.resolve()
        assert max_version == 41
        output_parent.unlink()
        output_parent.symlink_to(second_destination, target_is_directory=True)
        return {
            "artifact_sha256": "a" * 64,
            "windows": [],
        }

    monkeypatch.setattr(
        closure_cli,
        "build_causal_closure_selection",
        redirect_after_validation,
    )

    assert closure_cli.main() == 0

    assert json.loads(
        (first_destination / "selection.json").read_text(encoding="utf-8")
    )["artifact_sha256"] == "a" * 64
    assert not (second_destination / "selection.json").exists()
