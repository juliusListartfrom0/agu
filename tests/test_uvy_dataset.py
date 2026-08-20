from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.uvy_dataset import (
    UVY_AUDIT_SCHEMA,
    UVY_MANIFEST_SCHEMA,
    build_uvy_basketball_audit,
    verify_uvy_basketball_audit,
)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "uvy"
    sequence = root / "UVY" / "basketball_V01"
    image_dir = sequence / "img1"
    gt_dir = sequence / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir()
    (sequence / "video_info.txt").write_text(
        "videoName=basketball_V01\nimgDir=img1\nFPS=30\nnumFrames=4\nimWidth=1280\nimHeight=720\n",
        encoding="utf-8",
    )
    (gt_dir / "labels.txt").write_text("player\nsports ball\ngoal\n", encoding="utf-8")
    (gt_dir / "gt.txt").write_text(
        "1,1,10,20,30,40,1,1,1\n"
        "1,2,40,50,5,5,1,2,1\n"
        "2,1,11,20,30,40,1,1,1\n"
        "3,3,70,80,10,10,1,3,0.5\n",
        encoding="utf-8",
    )
    for frame in range(1, 5):
        (image_dir / f"{frame:06d}.jpg").write_bytes(b"jpeg-fixture")
    manifest: dict[str, object] = {
        "schema_version": UVY_MANIFEST_SCHEMA,
        "source_archive": {"name": "UVY.zip", "bytes": 10, "md5": "a" * 32},
        "manifest_sha256": "b" * 64,
        "sequences": [
            {
                "sequence": "basketball_V01",
                "entry_count": 7,
                "image_file_count": 4,
            }
        ],
    }
    return root, manifest


def test_uvy_audit_keeps_detector_auxiliary_separate_from_causal_truth(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    audit = build_uvy_basketball_audit(
        root,
        manifest,
        source_url="https://zenodo.org/api/records/21303900/files/UVY.zip/content",
        source_record_url="https://zenodo.org/records/21303900",
        source_revision="10.5281/zenodo.21303900",
        source_license="CC-BY-4.0",
    )

    assert audit["schema_version"] == UVY_AUDIT_SCHEMA
    assert audit["image_frame_count"] == 4
    assert audit["box_count"] == 4
    assert audit["class_counts"] == {"goal": 1, "player": 2, "sports ball": 1}
    assert audit["training_media_eligible"] is True
    assert audit["training_scope"] == "auxiliary_detector_and_hard_negative_only"
    assert audit["runtime_consumable"] is False
    assert audit["causal_truth_eligible"] is False
    assert audit["shot_outcome_label_count"] == 0
    assert verify_uvy_basketball_audit(audit)["artifact_sha256"] == audit["artifact_sha256"]


def test_uvy_audit_records_metadata_frame_mismatch(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    info = root / "UVY" / "basketball_V01" / "video_info.txt"
    info.write_text(info.read_text(encoding="utf-8").replace("numFrames=4", "numFrames=5"), encoding="utf-8")
    audit = build_uvy_basketball_audit(
        root,
        manifest,
        source_url="source",
        source_record_url="record",
        source_revision="rev",
        source_license="CC-BY-4.0",
    )
    sequence = audit["sequences"][0]
    assert sequence["metadata_frame_count_mismatch"] is True
    assert sequence["image_frame_count"] == 4


def test_uvy_audit_rejects_wrong_license_or_hash(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    audit = build_uvy_basketball_audit(
        root,
        manifest,
        source_url="source",
        source_record_url="record",
        source_revision="rev",
        source_license="CC-BY-4.0",
    )
    bad_license = dict(audit)
    bad_license["source_license"] = "unknown"
    with pytest.raises(ValueError, match="license"):
        verify_uvy_basketball_audit(bad_license)
    bad_hash = dict(audit)
    bad_hash["box_count"] = 99
    with pytest.raises(ValueError, match="hash"):
        verify_uvy_basketball_audit(bad_hash)


def test_uvy_audit_payload_is_json_serializable(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    audit = build_uvy_basketball_audit(
        root,
        manifest,
        source_url="source",
        source_record_url="record",
        source_revision="rev",
        source_license="CC-BY-4.0",
    )
    assert json.loads(json.dumps(audit, ensure_ascii=False))["schema_version"] == UVY_AUDIT_SCHEMA
