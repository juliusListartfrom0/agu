from __future__ import annotations

import pytest

from app.analysis.wikimedia_portraits import (
    parse_commons_search_results,
    seal_portrait_candidate_payload,
    write_downloaded_candidate,
)


def test_commons_candidates_require_reusable_license_and_remain_unverified() -> None:
    payload = {
        "query": {
            "pages": [
                {
                    "pageid": 12,
                    "title": "File:Player.jpg",
                    "imageinfo": [
                        {
                            "mime": "image/jpeg",
                            "thumburl": "https://upload.wikimedia.org/player.jpg",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Player.jpg",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                                "Artist": {"value": "<b>Photographer</b>"},
                            },
                        }
                    ],
                },
                {
                    "pageid": 13,
                    "title": "File:Unknown.jpg",
                    "imageinfo": [
                        {
                            "mime": "image/jpeg",
                            "url": "https://example.invalid/unknown.jpg",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Unknown.jpg",
                            "extmetadata": {"LicenseShortName": {"value": "unknown"}},
                        }
                    ],
                },
            ]
        }
    }

    candidates = parse_commons_search_results(payload, person_id="708", display_name="Kevin Garnett")

    assert len(candidates) == 1
    assert candidates[0]["identity_verified"] is False
    assert candidates[0]["artist"] == "Photographer"


def test_commons_manifest_is_non_runtime_and_hash_sealed() -> None:
    manifest = seal_portrait_candidate_payload(
        roster_sha256="roster",
        gallery_sha256="gallery",
        candidates=[{"person_id": "708", "identity_verified": False}],
    )

    assert manifest["runtime_consumable"] is False
    assert manifest["benchmark_disjoint"] is True
    assert manifest["identity_verified"] is False
    assert len(manifest["manifest_sha256"]) == 64


def test_downloaded_candidate_rejects_non_image_response(tmp_path) -> None:
    with pytest.raises(ValueError, match="declared MIME"):
        write_downloaded_candidate(
            {"person_id": "708", "commons_page_id": 1, "mime": "image/jpeg"},
            b"<html>rate limited</html>",
            output_dir=tmp_path,
        )
