from __future__ import annotations

from PIL import Image

from scripts.screen_e_bard_role_transfer import _sheet_crop, _verify_review


def test_sheet_crop_removes_annotation_border_from_first_panel() -> None:
    sheet = Image.new("RGB", (1920, 1080), (0, 0, 0))
    sheet.putpixel((318 + 10, 148 + 10), (231, 17, 19))

    crop = _sheet_crop(sheet, 1)

    assert crop.size == (154, 114)
    assert crop.getpixel((10, 10)) == (231, 17, 19)


def test_verify_review_requires_hash_bound_sealed_schema() -> None:
    plan = {
        "schema_version": "agu.broadcast-ball-offline-review-plan.v1",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "artifact_sha256": "plan-hash",
        "candidates": [{"candidate_id": "c-1"}],
    }
    review = {
        "schema_version": "agu.broadcast-ball-offline-review.v1",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "plan_sha256": "plan-hash",
        "decisions": [{"candidate_id": "c-1", "decision": "uncertain"}],
    }

    assert _verify_review(plan, review) == {"c-1": "uncertain"}
