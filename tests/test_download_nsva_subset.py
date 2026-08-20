from __future__ import annotations

import pytest

from scripts.download_nsva_subset import (
    resolve_source_url,
    seal_download_manifest,
    verify_download_manifest,
)


def test_nsva_resolve_source_url_rejects_path_traversal() -> None:
    with pytest.raises(ValueError):
        resolve_source_url("rev-1", "../secret.parquet")


def test_nsva_download_manifest_is_self_verifying() -> None:
    manifest = seal_download_manifest(
        {
            "source_url": "https://huggingface.co/datasets/sportsvision/nsva_subset",
            "source_revision": "rev-1",
            "files": [{"path": "data/train.parquet", "bytes": 1, "sha256": "a" * 64}],
            "total_bytes": 1,
            "row_count": 1,
        }
    )

    assert manifest["schema_version"] == "agu.nsva-subset-download.v1"
    assert verify_download_manifest(manifest)["manifest_sha256"] == manifest["manifest_sha256"]
