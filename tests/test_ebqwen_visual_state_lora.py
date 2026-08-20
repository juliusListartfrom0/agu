from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from app.analysis.ebqwen_visual_state_lora import (
    contact_sheet_row_bounds,
    seal_lora_dataset_manifest,
    select_lora_examples,
    verify_lora_dataset_manifest,
)
from app.analysis.visual_state_temporal_head import (
    VisualStateTemporalClassifier,
    load_visual_state_records,
    prepare_contact_sheet_tensor,
    split_contact_sheet_panels,
)


def test_select_lora_examples_keeps_resolved_rows_in_plan_order() -> None:
    plan_examples = [
        {
            "review_id": "visual-state-0001",
            "source_video_sha256": "1" * 64,
            "event_id": "event-1",
        },
        {
            "review_id": "visual-state-0002",
            "source_video_sha256": "2" * 64,
            "event_id": "event-2",
        },
        {
            "review_id": "visual-state-0003",
            "source_video_sha256": "1" * 64,
            "event_id": "event-3",
        },
    ]
    correction_decisions = [
        {
            "review_id": "visual-state-0001",
            "source_video_sha256": "1" * 64,
            "event_id": "event-1",
            "corrected_state": "free_throw",
        },
        {
            "review_id": "visual-state-0002",
            "source_video_sha256": "2" * 64,
            "event_id": "event-2",
            "corrected_state": None,
        },
        {
            "review_id": "visual-state-0003",
            "source_video_sha256": "1" * 64,
            "event_id": "event-3",
            "corrected_state": "field_goal",
        },
    ]
    sheet_records = [
        {
            "sheet": "sheets/source-01-sheet-01.jpg",
            "sheet_sha256": "a" * 64,
            "review_ids": [
                "visual-state-0001",
                "visual-state-0002",
                "visual-state-0003",
            ],
        }
    ]

    examples = select_lora_examples(
        plan_examples=plan_examples,
        correction_decisions=correction_decisions,
        sheet_records=sheet_records,
        sealed_blind_video_sha256s={"3" * 64},
    )

    assert [row["review_id"] for row in examples] == [
        "visual-state-0001",
        "visual-state-0003",
    ]
    assert [row["label"] for row in examples] == [
        "FREE_THROW",
        "LIVE_FIELD_GOAL",
    ]
    assert [row["sheet_row_index"] for row in examples] == [0, 2]


def test_select_lora_examples_rejects_blind_or_mismatched_rows() -> None:
    plan_examples = [
        {
            "review_id": "visual-state-0001",
            "source_video_sha256": "1" * 64,
            "event_id": "event-1",
        }
    ]
    decisions = [
        {
            "review_id": "visual-state-0001",
            "source_video_sha256": "1" * 64,
            "event_id": "different-event",
            "corrected_state": "free_throw",
        }
    ]
    sheets = [
        {
            "sheet": "sheets/source-01-sheet-01.jpg",
            "sheet_sha256": "a" * 64,
            "review_ids": ["visual-state-0001"],
        }
    ]

    with pytest.raises(ValueError, match="correction"):
        select_lora_examples(
            plan_examples=plan_examples,
            correction_decisions=decisions,
            sheet_records=sheets,
            sealed_blind_video_sha256s=set(),
        )

    decisions[0]["event_id"] = "event-1"
    with pytest.raises(ValueError, match="blind"):
        select_lora_examples(
            plan_examples=plan_examples,
            correction_decisions=decisions,
            sheet_records=sheets,
            sealed_blind_video_sha256s={"1" * 64},
        )


def test_contact_sheet_row_bounds_cover_every_pixel_once() -> None:
    assert [
        contact_sheet_row_bounds(
            image_height=101,
            row_count=3,
            row_index=index,
        )
        for index in range(3)
    ] == [(0, 34), (34, 67), (67, 101)]

    with pytest.raises(ValueError, match="row"):
        contact_sheet_row_bounds(image_height=100, row_count=3, row_index=3)


def test_lora_dataset_manifest_is_training_only_and_hash_bound() -> None:
    manifest = seal_lora_dataset_manifest(
        {
            "purpose": "test",
            "review_plan_sha256": "1" * 64,
            "sheet_manifest_sha256": "2" * 64,
            "label_corrections_sha256": "3" * 64,
            "sealed_blind_video_sha256s": ["4" * 64],
            "examples": [
                {
                    "review_id": "visual-state-0001",
                    "source_video_sha256": "5" * 64,
                    "event_id": "event-1",
                    "image": "images/visual-state-0001.jpg",
                    "image_sha256": "6" * 64,
                    "label": "FREE_THROW",
                }
            ],
        }
    )

    assert manifest["runtime_consumable"] is False
    assert manifest["codex_runtime_answer_used"] is False
    verify_lora_dataset_manifest(manifest)

    manifest["examples"][0]["label"] = "LIVE_FIELD_GOAL"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_lora_dataset_manifest(manifest)


def test_split_contact_sheet_panels_removes_header_and_preserves_order() -> None:
    sheet = np.zeros((20, 50, 3), dtype=np.uint8)
    for index in range(5):
        sheet[4:, index * 10 : (index + 1) * 10] = index + 1

    panels = split_contact_sheet_panels(sheet, header_height=4)

    assert panels.shape == (5, 16, 10, 3)
    assert [int(panel.mean()) for panel in panels] == [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="contact sheet"):
        split_contact_sheet_panels(sheet[:, :-1], header_height=4)


def test_visual_state_temporal_classifier_pools_five_ordered_frames() -> None:
    encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    model = VisualStateTemporalClassifier(encoder, embedding_dimension=3)

    logits = model(torch.ones(2, 5, 3, 8, 8))

    assert logits.shape == (2, 2)
    with pytest.raises(ValueError, match="five"):
        model(torch.ones(2, 4, 3, 8, 8))


def test_load_visual_state_records_verifies_training_image_hash(tmp_path) -> None:
    image = tmp_path / "images" / "sample.jpg"
    image.parent.mkdir()
    image.write_bytes(b"training-image")
    import hashlib

    manifest = seal_lora_dataset_manifest(
        {
            "review_plan_sha256": "1" * 64,
            "sheet_manifest_sha256": "2" * 64,
            "label_corrections_sha256": "3" * 64,
            "sealed_blind_video_sha256s": ["4" * 64],
            "examples": [
                {
                    "review_id": "visual-state-0001",
                    "source_video_sha256": "5" * 64,
                    "event_id": "event-1",
                    "image": "images/sample.jpg",
                    "image_sha256": hashlib.sha256(
                        b"training-image"
                    ).hexdigest(),
                    "label": "FREE_THROW",
                }
            ],
        }
    )

    records = load_visual_state_records(manifest, root=tmp_path)
    assert records[0].target == 1

    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_visual_state_records(manifest, root=tmp_path)


def test_prepare_contact_sheet_tensor_preserves_five_frame_shape() -> None:
    sheet = np.zeros((40, 100, 3), dtype=np.uint8)
    sheet[4:] = 127

    tensor = prepare_contact_sheet_tensor(
        sheet,
        header_height=4,
        output_size=32,
        training=False,
    )

    assert tensor.shape == (5, 3, 32, 32)
    assert torch.isfinite(tensor).all()
