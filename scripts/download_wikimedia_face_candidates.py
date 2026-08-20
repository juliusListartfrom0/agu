#!/usr/bin/env python3
"""Download license-recorded, identity-unverified Commons face candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_gallery import load_face_gallery  # noqa: E402
from app.analysis.wikimedia_portraits import (  # noqa: E402
    canonical_sha256,
    parse_commons_search_results,
    seal_portrait_candidate_payload,
    write_downloaded_candidate,
)

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AGU-open-research/1.0 (https://github.com/julius19910613/agu; face enrollment candidates)"
REQUEST_INTERVAL_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maximum-per-person", type=int, default=3)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.maximum_per_person <= 0 or args.retries <= 0:
        raise ValueError("maximum-per-person and retries must be positive")
    roster_bytes = args.roster.read_bytes()
    roster = json.loads(roster_bytes)
    if roster.get("schema_version") != "agu.face-roster.v1":
        raise ValueError("unsupported face roster schema")
    gallery = load_face_gallery(args.gallery)
    enrolled = {entry.person_id for entry in gallery.entries}
    candidates = []
    for player in roster.get("players") or []:
        person_id = str(player.get("person_id") or "").strip()
        display_name = str(player.get("display_name") or player.get("name") or "").strip()
        if not person_id or not display_name:
            raise ValueError("every face roster player requires person_id and display_name")
        if person_id in enrolled:
            continue
        payload = _request_json(_search_url(display_name, args.maximum_per_person), retries=args.retries)
        for candidate in parse_commons_search_results(
            payload,
            person_id=person_id,
            display_name=display_name,
        )[: args.maximum_per_person]:
            content = _request_bytes(str(candidate["image_url"]), retries=args.retries)
            candidates.append(write_downloaded_candidate(candidate, content, output_dir=args.output_dir))
            time.sleep(REQUEST_INTERVAL_SECONDS)
        time.sleep(REQUEST_INTERVAL_SECONDS)
    manifest = seal_portrait_candidate_payload(
        roster_sha256=canonical_sha256(roster),
        gallery_sha256=gallery.gallery_sha256,
        candidates=candidates,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "missing_person_count": len({item["person_id"] for item in candidates}),
                "candidate_count": len(candidates),
            },
            indent=2,
        )
    )
    return 0


def _search_url(display_name: str, limit: int) -> str:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": f'intitle:"{display_name}" filetype:bitmap',
        "gsrlimit": str(limit * 3),
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "640",
        "iiextmetadatalanguage": "en",
        "iiextmetadatafilter": (
            "LicenseShortName|LicenseUrl|UsageTerms|Artist|Credit|ImageDescription|"
            "AttributionRequired|Restrictions"
        ),
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def _request_json(url: str, *, retries: int) -> dict:
    return json.loads(_request_bytes(url, retries=retries))


def _request_bytes(url: str, *, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - exercised against the network
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Commons request failed after {retries} attempts: {url}") from last_error


if __name__ == "__main__":
    raise SystemExit(main())
