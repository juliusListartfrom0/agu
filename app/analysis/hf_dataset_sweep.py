"""Bounded Hugging Face basketball-dataset directory sweeps.

The sweep is a discovery aid only.  It reads the Hub search index and never
downloads dataset payloads.  Candidate payload acquisition remains behind the
separate continuous-causal source gate.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HF_DATASET_SWEEP_SCHEMA = "agu.hf-basketball-directory-sweep.v1"
HF_DATASET_SEARCH_URL = "https://huggingface.co/api/datasets"


def fetch_hf_basketball_search(*, limit: int = 100, timeout_seconds: float = 30.0) -> list[dict[str, Any]]:
    """Fetch only the bounded Hub search index for ``basketball`` datasets."""

    if not 1 <= int(limit) <= 100:
        raise ValueError("limit must be in [1, 100]")
    query = urlencode(
        {
            "search": "basketball",
            "sort": "lastModified",
            "direction": "-1",
            "limit": int(limit),
        }
    )
    request = Request(
        f"{HF_DATASET_SEARCH_URL}?{query}",
        headers={"Accept": "application/json", "User-Agent": "agu-hf-directory-sweep/1"},
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Hugging Face dataset search response must be a list")
    return [dict(row) for row in payload if isinstance(row, Mapping) and str(row.get("id") or "").strip()]


def build_hf_basketball_sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    generated_on: str,
    catalog_dataset_ids: Iterable[str] = (),
    triage_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, payload-free discovery audit."""

    catalog_ids = {str(item).strip().lower() for item in catalog_dataset_ids if str(item).strip()}
    overrides = {str(key).strip().lower(): str(value) for key, value in (triage_overrides or {}).items()}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        dataset_id = str(row.get("id") or "").strip()
        if not dataset_id:
            continue
        key = dataset_id.lower()
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "dataset_id": dataset_id,
                "last_modified": row.get("lastModified"),
                "sha": row.get("sha"),
                "downloads": int(row.get("downloads") or 0),
                "likes": int(row.get("likes") or 0),
                "private": bool(row.get("private")),
                "gated": row.get("gated") or False,
                "tags": [str(tag) for tag in (row.get("tags") or [])],
                "catalog_match": key in catalog_ids,
                "triage": overrides.get(key, "unreviewed_metadata_only"),
                "payload_downloaded": False,
            }
        )
    records.sort(key=lambda item: (str(item.get("last_modified") or ""), item["dataset_id"]), reverse=True)
    triage_counts = Counter(str(item["triage"]) for item in records)
    audit: dict[str, Any] = {
        "schema_version": HF_DATASET_SWEEP_SCHEMA,
        "generated_on": str(generated_on),
        "purpose": "bounded_huggingface_basketball_directory_discovery_without_payload_download",
        "query": {
            "endpoint": HF_DATASET_SEARCH_URL,
            "search": "basketball",
            "sort": "lastModified",
            "direction": "-1",
            "limit": len(records),
        },
        "result_count": len(records),
        "catalog_match_count": sum(bool(item["catalog_match"]) for item in records),
        "triage_counts": dict(sorted(triage_counts.items())),
        "payload_downloads_performed": 0,
        "download_policy": "directory discovery does not authorize payload download; use the five-check causal gate",
        "records": records,
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    return audit


def verify_hf_basketball_sweep(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a sealed sweep and retain its immutable audit hash."""

    artifact = dict(payload)
    claimed = str(artifact.pop("audit_sha256", ""))
    if artifact.get("schema_version") != HF_DATASET_SWEEP_SCHEMA:
        raise ValueError("invalid HF basketball sweep schema")
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError("HF basketball sweep records must be a list")
    if int(artifact.get("payload_downloads_performed", -1)) != 0:
        raise ValueError("HF basketball directory sweep must not download payloads")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("HF basketball sweep hash mismatch")
    artifact["audit_sha256"] = claimed
    return artifact


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
