from __future__ import annotations

import pytest

from app.analysis.hf_dataset_sweep import (
    HF_DATASET_SWEEP_SCHEMA,
    build_hf_basketball_sweep,
    verify_hf_basketball_sweep,
)


def test_sweep_is_deterministic_payload_free_and_deduplicated() -> None:
    audit = build_hf_basketball_sweep(
        [
            {"id": "new/one", "lastModified": "2026-01-02", "downloads": 2},
            {"id": "new/one", "lastModified": "2026-01-01", "downloads": 1},
            {"id": "old/two", "lastModified": "2026-01-03", "likes": 3},
        ],
        generated_on="2026-08-08",
        catalog_dataset_ids=["old/two"],
        triage_overrides={"new/one": "duplicate"},
    )

    assert audit["schema_version"] == HF_DATASET_SWEEP_SCHEMA
    assert audit["result_count"] == 2
    assert audit["catalog_match_count"] == 1
    assert audit["payload_downloads_performed"] == 0
    assert audit["triage_counts"] == {"duplicate": 1, "unreviewed_metadata_only": 1}
    assert all(item["payload_downloaded"] is False for item in audit["records"])
    assert verify_hf_basketball_sweep(audit)["audit_sha256"] == audit["audit_sha256"]


def test_sweep_verifier_rejects_payload_downloads() -> None:
    audit = build_hf_basketball_sweep([], generated_on="2026-08-08")
    audit["payload_downloads_performed"] = 1
    with pytest.raises(ValueError, match="must not download"):
        verify_hf_basketball_sweep(audit)
