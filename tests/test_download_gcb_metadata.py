from __future__ import annotations

import pytest

from scripts.download_gcb_metadata import (
    resolve_source_url,
    seal_download_manifest,
    verify_download_manifest,
)


def test_gcb_resolve_source_url_rejects_path_traversal() -> None:
    with pytest.raises(ValueError):
        resolve_source_url("rev-1", "../secret.jsonl")


def test_gcb_download_manifest_is_self_verifying() -> None:
    manifest = seal_download_manifest(
        {
            "source_url": "https://huggingface.co/datasets/A4Blind/GCB",
            "source_revision": "rev-1",
            "files": [{"path": "metadata.jsonl", "bytes": 1, "sha256": "a" * 64}],
            "total_bytes": 1,
            "row_count": 1,
            "media_downloaded": False,
        }
    )

    assert manifest["schema_version"] == "agu.gcb-metadata-download.v1"
    assert verify_download_manifest(manifest)["manifest_sha256"] == manifest["manifest_sha256"]
