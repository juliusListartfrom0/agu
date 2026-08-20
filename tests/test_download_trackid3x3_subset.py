from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "download_trackid3x3_subset.py"
    spec = importlib.util.spec_from_file_location("download_trackid3x3_subset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    file_id: str,
    name: str,
    mime_type: str,
    *,
    size_bytes: int | None = None,
) -> list[object]:
    values: list[object] = [None] * 14
    values[0] = file_id
    values[1] = ["parent"]
    values[2] = name
    values[3] = mime_type
    values[13] = size_bytes
    return values


def _folder_html(records: list[list[object]]) -> str:
    payload = [records, None, None, None, [], 0]
    return (
        "<script>"
        "window['_DRIVE_ivd'] = "
        f"{json.dumps(json.dumps(payload))};"
        "</script>"
    )


def test_parse_drive_folder_records_extracts_ids_names_types_and_sizes() -> None:
    module = _module()
    records = module.parse_drive_folder_records(
        _folder_html(
            [
                _record(
                    "folder-id",
                    "Indoor",
                    "application/vnd.google-apps.folder",
                ),
                _record("video-id", "clip.mp4", "video/mp4", size_bytes=1234),
            ]
        )
    )

    assert records == [
        {
            "file_id": "folder-id",
            "name": "Indoor",
            "mime_type": "application/vnd.google-apps.folder",
            "size_bytes": None,
        },
        {
            "file_id": "video-id",
            "name": "clip.mp4",
            "mime_type": "video/mp4",
            "size_bytes": 1234,
        },
    ]


def test_build_subset_manifest_walks_exact_folder_path_and_rejects_non_video() -> None:
    module = _module()
    pages = {
        "root": _folder_html(
            [_record("videos-id", "videos", "application/vnd.google-apps.folder")]
        ),
        "videos-id": _folder_html(
            [_record("indoor-id", "Indoor", "application/vnd.google-apps.folder")]
        ),
        "indoor-id": _folder_html(
            [_record("raw-id", "raw", "application/vnd.google-apps.folder")]
        ),
        "raw-id": _folder_html(
            [
                _record("a", "a.mp4", "video/mp4", size_bytes=10),
                _record("b", "b.mp4", "video/mp4", size_bytes=20),
            ]
        ),
    }

    manifest = module.build_subset_manifest(
        root_folder_id="root",
        subset_path=("videos", "Indoor", "raw"),
        repository_revision="abc123",
        fetch_html=lambda folder_id: pages[folder_id],
    )

    assert manifest["schema_version"] == "agu.trackid3x3-drive-manifest.v1"
    assert manifest["subset_path"] == "videos/Indoor/raw"
    assert manifest["file_count"] == 2
    assert manifest["declared_size_bytes"] == 30
    assert [item["name"] for item in manifest["files"]] == ["a.mp4", "b.mp4"]
    assert len(manifest["manifest_sha256"]) == 64

    pages["raw-id"] = _folder_html(
        [_record("notes", "notes.txt", "text/plain", size_bytes=3)]
    )
    with pytest.raises(ValueError, match="non-video"):
        module.build_subset_manifest(
            root_folder_id="root",
            subset_path=("videos", "Indoor", "raw"),
            repository_revision="abc123",
            fetch_html=lambda folder_id: pages[folder_id],
        )


def test_build_subset_manifest_rejects_ambiguous_or_unsafe_names() -> None:
    module = _module()
    pages = {
        "root": _folder_html(
            [
                _record("first", "videos", "application/vnd.google-apps.folder"),
                _record("second", "videos", "application/vnd.google-apps.folder"),
            ]
        )
    }
    with pytest.raises(ValueError, match="exactly one"):
        module.build_subset_manifest(
            root_folder_id="root",
            subset_path=("videos",),
            repository_revision="abc123",
            fetch_html=lambda folder_id: pages[folder_id],
        )

    pages = {
        "root": _folder_html(
            [_record("raw", "raw", "application/vnd.google-apps.folder")]
        ),
        "raw": _folder_html(
            [_record("bad", "../escape.mp4", "video/mp4", size_bytes=10)]
        ),
    }
    with pytest.raises(ValueError, match="unsafe"):
        module.build_subset_manifest(
            root_folder_id="root",
            subset_path=("raw",),
            repository_revision="abc123",
            fetch_html=lambda folder_id: pages[folder_id],
        )


def test_refresh_manifest_preserves_matching_completed_file_seals() -> None:
    module = _module()
    current = {
        "schema_version": module.SCHEMA_VERSION,
        "repository_revision": "revision-1",
        "subset_folder_id": "folder-1",
        "files": [
            {
                "file_id": "video-1",
                "name": "clip.mp4",
                "declared_size_bytes": 10,
                "sha256": None,
            }
        ],
    }
    previous = module.seal_manifest({
        **current,
        "download_complete": True,
        "files": [
            {
                **current["files"][0],
                "sha256": "a" * 64,
                "local_size_bytes": 10,
            }
        ],
    })

    refreshed = module.preserve_completed_file_seals(current, previous)

    assert refreshed["download_complete"] is True
    assert refreshed["files"][0]["sha256"] == "a" * 64
    assert refreshed["files"][0]["local_size_bytes"] == 10

    previous["files"][0]["file_id"] = "different-video"
    previous = module.seal_manifest(previous)
    with pytest.raises(ValueError, match="does not match"):
        module.preserve_completed_file_seals(current, previous)


def test_manifest_verification_and_existing_download_hash_are_fail_closed(
    tmp_path: Path,
) -> None:
    module = _module()
    content = b"local-video"
    expected_sha = hashlib.sha256(b"different-video").hexdigest()
    manifest = module.seal_manifest(
        {
            "schema_version": module.SCHEMA_VERSION,
            "repository_revision": "revision-1",
            "subset_folder_id": "folder-1",
            "file_count": 1,
            "files": [
                {
                    "file_id": "video-1",
                    "name": "clip.mp4",
                    "declared_size_bytes": len(content),
                    "sha256": expected_sha,
                }
            ],
        }
    )
    output_dir = tmp_path / "media"
    output_dir.mkdir()
    (output_dir / "clip.mp4").write_bytes(content)

    with pytest.raises(ValueError, match="hash mismatch"):
        module.download_manifest_files(
            manifest,
            output_dir=output_dir,
            manifest_path=tmp_path / "manifest.json",
            reserve_bytes=0,
        )

    manifest["file_count"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        module.verify_manifest(manifest)


def test_copy_response_bounded_stops_after_declared_size() -> None:
    module = _module()
    destination = io.BytesIO()

    with pytest.raises(RuntimeError, match="exceeded"):
        module.copy_response_bounded(
            io.BytesIO(b"too-many-bytes"),
            destination,
            maximum_bytes=4,
        )


def test_completed_partial_file_is_promoted_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    target = tmp_path / "clip.mp4"
    part = tmp_path / "clip.mp4.part"
    part.write_bytes(b"complete")
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network should not be used"),
    )

    module._download_drive_file(
        file_id="video-1",
        target=target,
        expected_size=8,
    )

    assert target.read_bytes() == b"complete"
    assert not part.exists()
