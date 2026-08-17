from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from app.analysis.ball_candidate_review import seal_artifact
from app.analysis.deim_basketball_detector import (
    DEIM_SCREEN_THRESHOLDS,
    DeimBasketballDetector,
    build_deim_external_screen,
    decode_deim_ball_outputs,
    preprocess_deim_frame,
)
from scripts import screen_deim_basketball_detector_external as deim_cli


@dataclass(frozen=True)
class _Input:
    name: str


class _Session:
    def __init__(self) -> None:
        self.feed: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[_Input]:
        return [_Input("images"), _Input("orig_target_sizes")]

    def get_outputs(self) -> list[_Input]:
        return [_Input("labels"), _Input("boxes"), _Input("scores")]

    def run(
        self,
        output_names: None,
        feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        assert output_names is None
        self.feed = feed
        return [
            np.asarray([[0, 1, 9, 3]], dtype=np.int64),
            np.asarray(
                [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]],
                dtype=np.float32,
            ),
            np.asarray([[0.9, 0.4, 0.99, 0.8]], dtype=np.float32),
        ]


def test_preprocess_deim_frame_matches_official_bgr_to_rgb_contract() -> None:
    frame = np.asarray(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )

    tensor = preprocess_deim_frame(frame, input_size=2)

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(
        tensor[0, 0],
        [[20 / 255, 50 / 255], [80 / 255, 110 / 255]],
    )
    np.testing.assert_allclose(
        tensor[0, 1],
        [[10 / 255, 40 / 255], [70 / 255, 100 / 255]],
    )
    np.testing.assert_allclose(
        tensor[0, 2],
        [[0, 30 / 255], [60 / 255, 90 / 255]],
    )


def test_detector_supplies_original_size_and_keeps_both_ball_classes(tmp_path) -> None:
    session = _Session()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test-model")
    detector = DeimBasketballDetector(
        model_path,
        session_factory=lambda _path, _providers: session,
    )
    frame = np.zeros((4, 7, 3), dtype=np.uint8)

    predictions = detector.predict(frame, confidence_floor=0.25)

    assert session.feed is not None
    assert session.feed["images"].shape == (1, 3, 640, 640)
    assert session.feed["images"].dtype == np.float32
    assert session.feed["orig_target_sizes"].tolist() == [[7, 4]]
    assert session.feed["orig_target_sizes"].dtype == np.int64
    assert predictions == [
        {
            "class_id": 0,
            "class_name": "ball",
            "confidence": pytest.approx(0.9),
            "bbox_xyxy": pytest.approx([1.0, 2.0, 3.0, 4.0]),
        },
        {
            "class_id": 1,
            "class_name": "ball-in-basket",
            "confidence": pytest.approx(0.4),
            "bbox_xyxy": pytest.approx([5.0, 6.0, 7.0, 8.0]),
        },
    ]


def test_decoder_is_strict_about_threshold_shapes_and_finite_values() -> None:
    outputs = [
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([[[0, 0, 1, 1], [1, 1, 2, 2]]], dtype=np.float32),
        np.asarray([[0.25, 0.2501]], dtype=np.float32),
    ]
    assert [row["class_id"] for row in decode_deim_ball_outputs(outputs, confidence_floor=0.25)] == [1]

    with pytest.raises(ValueError, match="confidence floor"):
        decode_deim_ball_outputs(outputs, confidence_floor=True)
    with pytest.raises(ValueError, match="three outputs"):
        decode_deim_ball_outputs(outputs[:2], confidence_floor=0.25)
    invalid = [*outputs]
    invalid[2] = np.asarray([[np.nan, 0.4]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        decode_deim_ball_outputs(invalid, confidence_floor=0.25)
    invalid = [*outputs]
    invalid[1] = np.asarray([[[0, 0, 1, 1]]], dtype=np.float32)
    with pytest.raises(ValueError, match="align"):
        decode_deim_ball_outputs(invalid, confidence_floor=0.25)


def test_external_screen_uses_only_predeclared_thresholds_and_never_promotes() -> None:
    plan = seal_artifact(
        {
            "schema_version": "agu.broadcast-ball-offline-review-plan.v1",
            "runtime_consumable": False,
            "raw_video_sha256": "a" * 64,
            "candidates": [
                {"candidate_id": "a", "frame": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
                {"candidate_id": "b", "frame": 2, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
                {"candidate_id": "c", "frame": 3, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            ],
        }
    )
    review = seal_artifact(
        {
            "schema_version": "agu.broadcast-ball-offline-review.v1",
            "runtime_consumable": False,
            "plan_sha256": plan["artifact_sha256"],
            "decisions": [
                {"candidate_id": "a", "decision": "valid_ball"},
                {"candidate_id": "b", "decision": "valid_ball"},
                {"candidate_id": "c", "decision": "false_positive"},
            ],
        }
    )
    predictions = {
        "a": [
            {
                "class_id": 0,
                "class_name": "ball",
                "confidence": 0.6,
                "bbox_xyxy": [0, 0, 10, 10],
            },
            {
                "class_id": 0,
                "class_name": "ball",
                "confidence": 0.2,
                "bbox_xyxy": [20, 20, 30, 30],
            },
        ],
        "b": [
            {
                "class_id": 1,
                "class_name": "ball-in-basket",
                "confidence": 0.2,
                "bbox_xyxy": [0, 0, 10, 10],
            }
        ],
        "c": [
            {
                "class_id": 0,
                "class_name": "ball",
                "confidence": 0.4,
                "bbox_xyxy": [20, 20, 30, 30],
            }
        ],
    }

    artifact = build_deim_external_screen(
        plan,
        review,
        predictions,
        model_sha256="b" * 64,
        raw_video_sha256="a" * 64,
        iou_threshold=0.25,
    )

    assert artifact["thresholds"] == list(DEIM_SCREEN_THRESHOLDS) == [0.1, 0.25, 0.5]
    assert [(row["tp"], row["fp"], row["fn"]) for row in artifact["metrics_by_threshold"]] == [
        (2, 2, 0),
        (1, 1, 1),
        (1, 0, 1),
    ]
    assert artifact["runtime_consumable"] is False
    assert artifact["formal_evaluation_eligible"] is False
    assert artifact["promotion_eligible"] is False
    assert artifact["promoted"] is False
    assert artifact["meets_followup_gate"] is False
    assert artifact["artifact_sha256"]


def test_external_screen_rejects_noncanonical_prediction_and_target_rows() -> None:
    plan = seal_artifact(
        {
            "schema_version": "agu.broadcast-ball-offline-review-plan.v1",
            "runtime_consumable": False,
            "raw_video_sha256": "a" * 64,
            "candidates": [
                {
                    "candidate_id": "a",
                    "frame": 1,
                    "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                }
            ],
        }
    )
    review = seal_artifact(
        {
            "schema_version": "agu.broadcast-ball-offline-review.v1",
            "runtime_consumable": False,
            "plan_sha256": plan["artifact_sha256"],
            "decisions": [{"candidate_id": "a", "decision": "valid_ball"}],
        }
    )
    prediction = {
        "class_id": 0,
        "class_name": "ball",
        "confidence": 0.9,
        "bbox_xyxy": [0, 0, 10, 10],
        "ground_truth": True,
    }

    with pytest.raises(ValueError, match="exact fields"):
        build_deim_external_screen(
            plan,
            review,
            {"a": [prediction]},
            model_sha256="b" * 64,
            raw_video_sha256="a" * 64,
        )

    invalid_plan = dict(plan)
    invalid_plan["candidates"] = [dict(plan["candidates"][0])]
    invalid_plan["candidates"][0]["bbox"] = {"x1": 0, "y1": 0, "x2": float("nan"), "y2": 10}
    invalid_plan = seal_artifact(invalid_plan)
    invalid_review = seal_artifact(
        {
            **review,
            "plan_sha256": invalid_plan["artifact_sha256"],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        build_deim_external_screen(
            invalid_plan,
            invalid_review,
            {"a": [{key: value for key, value in prediction.items() if key != "ground_truth"}]},
            model_sha256="b" * 64,
            raw_video_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2, 4), dtype=np.uint8),
    ],
)
def test_preprocess_rejects_noncanonical_frames(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8 HWC BGR"):
        preprocess_deim_frame(frame)


def test_cli_rejects_output_alias_before_any_input_read(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.onnx"
    plan = tmp_path / "plan.json"
    review = tmp_path / "review.json"
    video = tmp_path / "video.mp4"
    for path in (model, plan, review, video):
        path.write_bytes(b"input")
    monkeypatch.setattr(deim_cli.Path, "read_bytes", lambda _self: pytest.fail("input read"))

    with pytest.raises(ValueError, match="output path must not alias"):
        deim_cli._resolve_disjoint_paths(
            output_path=model,
            input_paths=(model, plan, review, video),
        )


def test_cli_atomic_writer_publishes_canonical_json(tmp_path) -> None:
    output = tmp_path / "screen.json"

    deim_cli._write_json_atomic(output, {"value": 1})

    assert output.read_text(encoding="utf-8") == '{\n  "value": 1\n}\n'
