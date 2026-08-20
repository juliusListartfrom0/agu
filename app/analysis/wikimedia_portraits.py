"""License-aware Wikimedia Commons portrait candidates for face enrollment."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "agu.wikimedia-face-candidates.v1"
ALLOWED_LICENSE_PREFIXES = ("cc by", "cc-by", "cc0", "public domain", "pd-")


def parse_commons_search_results(
    payload: Mapping[str, Any],
    *,
    person_id: str,
    display_name: str,
) -> list[dict[str, Any]]:
    """Return reusable, still-unverified image candidates from an API response."""

    candidates = []
    for page in (payload.get("query") or {}).get("pages") or []:
        info_items = page.get("imageinfo") or []
        if not info_items:
            continue
        info = info_items[0]
        metadata = info.get("extmetadata") or {}
        license_name = _metadata_value(metadata, "LicenseShortName") or _metadata_value(
            metadata, "UsageTerms"
        )
        mime = str(info.get("mime") or "").lower()
        if not _license_allowed(license_name) or mime not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        image_url = str(info.get("thumburl") or info.get("url") or "").strip()
        source_page = str(info.get("descriptionurl") or "").strip()
        if not image_url or not source_page:
            continue
        candidates.append(
            {
                "person_id": person_id,
                "display_name": display_name,
                "identity_verified": False,
                "commons_page_id": int(page.get("pageid") or 0),
                "commons_title": str(page.get("title") or ""),
                "source_page": source_page,
                "image_url": image_url,
                "mime": mime,
                "license": license_name,
                "license_url": _metadata_value(metadata, "LicenseUrl"),
                "artist": _metadata_value(metadata, "Artist"),
                "credit": _metadata_value(metadata, "Credit"),
                "description": _metadata_value(metadata, "ImageDescription"),
                "attribution_required": _metadata_value(metadata, "AttributionRequired"),
                "restrictions": _metadata_value(metadata, "Restrictions"),
            }
        )
    return sorted(candidates, key=lambda item: (item["commons_page_id"], item["commons_title"]))


def seal_portrait_candidate_payload(
    *,
    roster_sha256: str,
    gallery_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runtime_consumable": False,
        "benchmark_disjoint": True,
        "identity_verified": False,
        "roster_sha256": roster_sha256,
        "gallery_sha256": gallery_sha256,
        "candidates": [dict(item) for item in candidates],
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def write_downloaded_candidate(
    candidate: Mapping[str, Any],
    content: bytes,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    if not content:
        raise ValueError("downloaded portrait candidate is empty")
    extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(
        str(candidate.get("mime") or "").lower()
    )
    if extension is None:
        raise ValueError("unsupported portrait candidate MIME type")
    if not _content_matches_mime(content, str(candidate.get("mime") or "").lower()):
        raise ValueError("downloaded portrait bytes do not match the declared MIME type")
    page_id = int(candidate.get("commons_page_id") or 0)
    person_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate.get("person_id") or ""))
    if page_id <= 0 or not person_id:
        raise ValueError("portrait candidate requires person_id and Commons page ID")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{person_id}-{page_id}{extension}"
    path.write_bytes(content)
    return {
        **dict(candidate),
        "local_path": str(path.resolve()),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _license_allowed(value: str) -> bool:
    normalized = value.strip().lower()
    return any(normalized.startswith(prefix) for prefix in ALLOWED_LICENSE_PREFIXES)


def _metadata_value(metadata: Mapping[str, Any], key: str) -> str:
    raw = metadata.get(key)
    if isinstance(raw, Mapping):
        raw = raw.get("value")
    value = html.unescape(str(raw or ""))
    return re.sub(r"<[^>]+>", " ", value).replace("\xa0", " ").strip()


def _content_matches_mime(content: bytes, mime: str) -> bool:
    if mime == "image/jpeg":
        return content.startswith(b"\xff\xd8\xff")
    if mime == "image/png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/webp":
        return len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP"
    return False
