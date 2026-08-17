from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import app.analysis.continuous_causal_selection as selection_module
from app.analysis.continuous_causal_selection import (
    RANKING_NAMESPACE,
    SOURCE_SELECTION_PROTOCOL,
    build_continuous_causal_selection,
    neutral_continuous_review_id,
    verify_continuous_causal_selection,
    verify_continuous_causal_selection_receipt,
)


@pytest.fixture(autouse=True)
def _matching_media_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        selection_module,
        "_probe_video_geometry",
        lambda _path: {
            "fps": 30.0,
            "frame_count": 123_753,
            "duration_seconds": 4_125.088,
        },
        raising=False,
    )


def _write_manifest(tmp_path: Path) -> Path:
    video = tmp_path / "harwood.webm"
    video_bytes = b"fixture continuous source video\n"
    video.write_bytes(video_bytes)
    path = tmp_path / "source_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "agu.vru-basketball-source-manifest.v1",
                "purpose": "offline_harwood_causal_review",
                "license": "CC-BY-4.0-per-file",
                "source_urls": ["https://commons.wikimedia.org/wiki/File:Harwood.webm"],
                "selection": SOURCE_SELECTION_PROTOCOL,
                "runtime_consumable": False,
                "codex_runtime_answer_used": False,
                "provenance": {
                    "provider": "Wikimedia Commons",
                    "publisher": "Hardwick Community Television (HCTV)",
                    "publisher_channel_id": "hctv",
                    "source_page_url": "https://commons.wikimedia.org/wiki/File:Harwood.webm",
                    "original_source_url": "https://www.youtube.com/watch?v=fixture",
                    "original_source_id": "fixture",
                    "title": "Boys Varsity Basketball v. Harwood",
                    "event_date": "2026-01-22",
                    "license_spdx": "CC-BY-4.0",
                    "attribution": "Hardwick Community Television (HCTV)",
                    "license_review_warning": True,
                    "production_family": "HCTV",
                },
                "videos": [
                    {
                        "location": "harwood",
                        "clip_id": "harwood_fullgame_2026",
                        "path": video.name,
                        "size_bytes": len(video_bytes),
                        "sha256": hashlib.sha256(video_bytes).hexdigest(),
                        "fps": 30.0,
                        "frame_count": 123_753,
                        "duration_seconds": 4_125.088,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _build(tmp_path: Path) -> dict[str, Any]:
    return build_continuous_causal_selection(
        source_manifest_path=_write_manifest(tmp_path),
        source_id="harwood",
    )


def _reseal(artifact: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    artifact["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value))
    return set()


def test_selection_freezes_exact_label_hidden_temporal_quota(tmp_path: Path) -> None:
    first = _build(tmp_path)
    second = build_continuous_causal_selection(
        source_manifest_path=tmp_path / "source_manifest.json",
        source_id="harwood",
    )

    assert first == second
    assert verify_continuous_causal_selection(first) == first
    assert first["runtime_consumable"] is False
    assert first["training_consumable"] is False
    assert first["formal_evaluation_eligible"] is False
    assert first["codex_runtime_answer_used"] is False
    assert first["labels_hidden_from_reviewer"] is True
    assert first["selection_provenance_verification"] == "exact_geometry_replay_v1"
    assert first["source"]["provenance"]["license_review_warning"] is True
    assert first["source"]["provenance"]["production_family"] == "HCTV"
    assert len(first["windows"]) == 24
    assert Counter(row["temporal_bucket"] for row in first["windows"]) == {
        "early": 8,
        "middle": 8,
        "late": 8,
    }
    assert [row["review_id"] for row in first["windows"]] == [
        neutral_continuous_review_id(
            first["source"]["source_video_sha256"],
            index,
        )
        for index in range(1, 25)
    ]
    assert not {
        "answer",
        "confidence",
        "event_present",
        "label",
        "labels",
        "model_score",
        "notes",
        "outcome",
        "pbp",
        "prior_answers",
        "probability",
        "release_frame",
        "review_state",
        "shot_sequence",
        "target",
    }.intersection(_all_keys(first))


def test_selection_uses_frozen_grid_hash_ranking_and_half_open_samples(
    tmp_path: Path,
) -> None:
    artifact = _build(tmp_path)
    policy = artifact["selection"]

    assert policy == {
        "ranking_namespace": RANKING_NAMESPACE,
        "ranking_direction": "ascending_sha256",
        "candidate_grid_spacing_seconds": 30.0,
        "window_seconds": 8.0,
        "sample_rate_hz": 8.0,
        "sample_count": 64,
        "interval_semantics": "half_open",
        "temporal_bucket_quotas": {"early": 8, "middle": 8, "late": 8},
    }
    rows = artifact["windows"]
    assert rows == sorted(rows, key=lambda row: row["center_seconds"])
    for left, right in zip(rows, rows[1:], strict=False):
        assert left["end_seconds"] <= right["start_seconds"]
    for row in rows:
        indexes = [round((row["start_seconds"] + index / 8.0) * row["source_fps"]) for index in range(64)]
        assert indexes == sorted(set(indexes))
        assert indexes[32] == row["anchor_frame"]
        assert indexes[-1] < round(row["end_seconds"] * row["source_fps"])
        expected_rank = hashlib.sha256(
            f"{RANKING_NAMESPACE}\0{artifact['source']['source_video_sha256']}\0{row['anchor_frame']}".encode()
        ).hexdigest()
        assert row["ranking_sha256"] == expected_rank

    candidates = []
    for center_seconds in range(30, 4_111, 30):
        anchor = round(center_seconds * 30.0)
        fraction = center_seconds / 4_125.088
        bucket = "early" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "late"
        rank = hashlib.sha256(
            f"{RANKING_NAMESPACE}\0{artifact['source']['source_video_sha256']}\0{anchor}".encode()
        ).hexdigest()
        candidates.append((bucket, rank, anchor))
    oracle = sorted(
        anchor
        for bucket in ("early", "middle", "late")
        for _, _, anchor in sorted(
            (row for row in candidates if row[0] == bucket),
            key=lambda row: row[1],
        )[:8]
    )
    assert [row["anchor_frame"] for row in rows] == oracle


@pytest.mark.parametrize(
    ("location", "key", "value"),
    [
        ("top", "prior_answers", {"x": "shot"}),
        ("source", "pbp", ["made"]),
        ("selection", "model_score", 0.9),
        ("window", "outcome", "made"),
    ],
)
def test_verifier_rejects_resealed_noncanonical_or_label_bearing_fields(
    tmp_path: Path,
    location: str,
    key: str,
    value: Any,
) -> None:
    artifact = _build(tmp_path)
    if location == "top":
        target = artifact
    elif location == "window":
        target = artifact["windows"][0]
    else:
        target = artifact[location]
    target[key] = value
    _reseal(artifact)

    with pytest.raises(ValueError, match="fields|canonical"):
        verify_continuous_causal_selection(artifact)


def test_verifier_replays_selection_and_rejects_resealed_window_drift(
    tmp_path: Path,
) -> None:
    artifact = _build(tmp_path)
    artifact["windows"][0]["anchor_frame"] += 1
    artifact["windows"][0]["center_seconds"] = round(
        artifact["windows"][0]["anchor_frame"] / 30.0,
        6,
    )
    artifact["windows"][0]["start_seconds"] = round(
        artifact["windows"][0]["center_seconds"] - 4.0,
        6,
    )
    artifact["windows"][0]["end_seconds"] = round(
        artifact["windows"][0]["center_seconds"] + 4.0,
        6,
    )
    artifact["windows"][0]["ranking_sha256"] = hashlib.sha256(b"forged").hexdigest()
    _reseal(artifact)

    with pytest.raises(ValueError, match="replay"):
        verify_continuous_causal_selection(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_fps", True),
        ("frame_count", 123_753.0),
        ("source_video_sha256", 1),
        ("clip_id", {"prior_answers": "shot"}),
    ],
)
def test_verifier_rejects_noncanonical_source_scalar_types(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    artifact = _build(tmp_path)
    artifact["source"][field] = value
    _reseal(artifact)

    with pytest.raises(ValueError):
        verify_continuous_causal_selection(artifact)


def test_builder_rejects_source_manifest_extra_fields_and_label_channels(
    tmp_path: Path,
) -> None:
    path = _write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["videos"][0]["outcome"] = "made"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="video fields"):
        build_continuous_causal_selection(
            source_manifest_path=path,
            source_id="harwood",
        )


def test_builder_rejects_unselected_videos_and_free_text_selection_protocol(
    tmp_path: Path,
) -> None:
    path = _write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    unselected = copy.deepcopy(payload["videos"][0])
    unselected["location"] = "leaky"
    unselected["clip_id"] = "leaky_clip"
    unselected["outcome"] = "made"
    payload["videos"].append(unselected)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one video"):
        build_continuous_causal_selection(
            source_manifest_path=path,
            source_id="harwood",
        )

    payload["videos"] = payload["videos"][:1]
    payload["selection"] = "selected after consulting model_score and PBP"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selection protocol"):
        build_continuous_causal_selection(
            source_manifest_path=path,
            source_id="harwood",
        )


def test_builder_verifies_actual_source_video_size_and_sha(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    (tmp_path / "harwood.webm").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="size|SHA-256"):
        build_continuous_causal_selection(
            source_manifest_path=path,
            source_id="harwood",
        )


def test_builder_rejects_manifest_geometry_that_disagrees_with_media_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_manifest(tmp_path)
    monkeypatch.setattr(
        selection_module,
        "_probe_video_geometry",
        lambda _path: {
            "fps": 60.0,
            "frame_count": 247_504,
            "duration_seconds": 4_125.088,
        },
        raising=False,
    )

    with pytest.raises(ValueError, match="probe.*FPS|FPS.*probe"):
        build_continuous_causal_selection(
            source_manifest_path=path,
            source_id="harwood",
        )


@pytest.mark.parametrize("escaped_path", ("../harwood.webm", "/tmp/harwood.webm"))
def test_builder_rejects_video_paths_outside_manifest_root(
    tmp_path: Path,
    escaped_path: str,
) -> None:
    path = _write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["videos"][0]["path"] = escaped_path
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="relative|manifest root"):
        build_continuous_causal_selection(
            source_manifest_path=path,
            source_id="harwood",
        )


def test_builder_rejects_source_symlink_that_escapes_manifest_root(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = _write_manifest(bundle)
    outside = tmp_path / "outside.webm"
    outside.write_bytes((bundle / "harwood.webm").read_bytes())
    (bundle / "harwood.webm").unlink()
    (bundle / "harwood.webm").symlink_to(outside)

    with pytest.raises(ValueError, match="manifest root"):
        build_continuous_causal_selection(
            source_manifest_path=manifest,
            source_id="harwood",
        )


def test_reviewer_ids_are_opaque_and_do_not_copy_source_identifiers(
    tmp_path: Path,
) -> None:
    path = _write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["videos"][0]["location"] = "shot"
    payload["videos"][0]["clip_id"] = "model_score_099"
    path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = build_continuous_causal_selection(
        source_manifest_path=path,
        source_id="shot",
    )

    review_ids = [row["review_id"] for row in artifact["windows"]]
    assert review_ids[0].startswith("closure-")
    assert all("shot" not in review_id for review_id in review_ids)
    assert all("model_score" not in review_id for review_id in review_ids)


@pytest.mark.parametrize(
    "unsafe",
    (
        "..",
        "raw/..",
        "..\\source.webm",
        "C:\\source.webm",
        "C:/source.webm",
        "bad\nname.webm",
    ),
)
def test_builder_rejects_cross_platform_unsafe_video_paths(
    tmp_path: Path,
    unsafe: str,
) -> None:
    path = _write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["videos"][0]["path"] = unsafe
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="path|filename"):
        build_continuous_causal_selection(
            source_manifest_path=path,
            source_id="harwood",
        )


def test_expected_artifact_receipt_is_fail_closed(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    artifact = build_continuous_causal_selection(
        source_manifest_path=manifest,
        source_id="harwood",
    )
    assert (
        verify_continuous_causal_selection_receipt(
            artifact,
            expected_artifact_sha256=artifact["artifact_sha256"],
            source_manifest_path=manifest,
        )
        == artifact
    )
    with pytest.raises(ValueError, match="expected artifact"):
        verify_continuous_causal_selection_receipt(
            artifact,
            expected_artifact_sha256="f" * 64,
            source_manifest_path=manifest,
        )
