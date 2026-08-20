from pathlib import Path

import pytest

from scripts.materialize_e_bard_muvy_ball_dataset import _verify_muvy_manifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_root",
    [
        "dataset/public_sources/muvy_ball_yolo_v1",
        "dataset/public_sources/muvy_ball_yolo_v2",
    ],
)
def test_verify_muvy_manifest_accepts_sealed_review_revisions(
    relative_root: str,
) -> None:
    manifest, manifest_sha = _verify_muvy_manifest(ROOT / relative_root)

    assert manifest["runtime_consumable"] is False
    assert manifest["codex_runtime_answer_used"] is False
    assert manifest["image_count"] == len(manifest["examples"])
    assert manifest["box_count"] == sum(
        len(example["labels"]) for example in manifest["examples"]
    )
    assert len(manifest_sha) == 64
