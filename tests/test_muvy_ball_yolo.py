from __future__ import annotations

import pytest

from app.analysis.muvy_ball_review import (
    seal_muvy_ball_review,
    seal_muvy_ball_review_plan,
)
from app.analysis.muvy_ball_yolo import (
    build_muvy_ball_yolo_data_yaml,
    build_muvy_yolo_examples,
)


def _candidate(index: int, event: str, frame_id: int) -> dict:
    return {
        "candidate_id": f"muvy-ball-{index:04d}",
        "video_path": f"{event}/cam/video.mp4",
        "video_sha256": f"{index + 1:x}" * 64,
        "annotation_sha256": f"{index + 4:x}" * 64,
        "frame_id": frame_id,
        "frame_index": frame_id - 1,
        "frame_width": 1000,
        "frame_height": 500,
        "frame_sha256": f"{index + 7:x}" * 64,
        "confidence": 0.1,
        "bbox": [10.0, 5.0, 30.0, 15.0],
    }


def test_build_muvy_yolo_examples_filters_and_splits_by_event() -> None:
    plan = seal_muvy_ball_review_plan(
        {
            "source_manifest_sha256": "a" * 64,
            "source_record_url": "https://zenodo.org/records/13883315",
            "source_license": "CC-BY-4.0",
            "candidates": [
                _candidate(1, "train_event", 1),
                _candidate(2, "train_event", 2),
                _candidate(3, "val_event", 1),
            ],
        }
    )
    review = seal_muvy_ball_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "codex_offline_source_annotation",
            "decisions": [
                {
                    "candidate_id": "muvy-ball-0001",
                    "decision": "valid_ball",
                    "notes": "visible",
                },
                {
                    "candidate_id": "muvy-ball-0002",
                    "decision": "false_positive",
                    "notes": "not a ball",
                },
                {
                    "candidate_id": "muvy-ball-0003",
                    "decision": "valid_ball",
                    "notes": "visible",
                },
            ],
        },
        plan=plan,
    )

    examples = build_muvy_yolo_examples(
        plan,
        review,
        validation_events={"val_event"},
    )

    assert [example["split"] for example in examples] == ["train", "val"]
    assert examples[0]["labels"] == [[0, 0.02, 0.02, 0.02, 0.02]]
    assert examples[1]["event"] == "val_event"
    with pytest.raises(ValueError, match="validation event"):
        build_muvy_yolo_examples(
            plan,
            review,
            validation_events={"missing"},
        )


def test_build_muvy_ball_yolo_data_yaml_uses_absolute_dataset_path(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset"

    payload = build_muvy_ball_yolo_data_yaml(dataset_path)

    assert f"path: {dataset_path.resolve().as_posix()}\n" in payload
    assert "train: images/train\n" in payload
    assert "val: images/val\n" in payload
    assert "  0: basketball\n" in payload
    assert "  1: hoop\n" in payload
    assert "  2: player\n" in payload
    assert "  3: referee\n" in payload
