from __future__ import annotations

import numpy as np
import pytest

from app.analysis.rfdetr_query_verifier import (
    expand_detection_ids_to_tracks,
    match_candidates_to_queries,
    query_feature_vector,
    track_query_features,
)


def test_match_candidates_to_queries_reconstructs_adapter_object_index() -> None:
    logits = np.asarray(
        [
            [2.0, -3.0],
            [1.0, -4.0],
        ],
        dtype=np.float32,
    )
    pred_boxes = np.asarray(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.25, 0.25, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    candidates = [
        {
            "detection_id": "transformers_rfdetr:10:0",
            "frame": 10,
            "confidence": float(1.0 / (1.0 + np.exp(-2.0))),
            "bbox": {"x1": 40.0, "y1": 40.0, "x2": 60.0, "y2": 60.0},
        }
    ]

    matches = match_candidates_to_queries(
        logits,
        pred_boxes,
        candidates,
        image_width=100,
        image_height=100,
        confidence_threshold=0.1,
        ball_label_id=0,
    )

    assert matches == {
        "transformers_rfdetr:10:0": {
            "query_index": 0,
            "label_id": 0,
            "score": pytest.approx(candidates[0]["confidence"]),
        }
    }


def test_match_candidates_to_queries_rejects_non_reproducible_box() -> None:
    with pytest.raises(ValueError, match="bbox"):
        match_candidates_to_queries(
            np.asarray([[2.0]], dtype=np.float32),
            np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            [
                {
                    "detection_id": "transformers_rfdetr:10:0",
                    "frame": 10,
                    "confidence": float(1.0 / (1.0 + np.exp(-2.0))),
                    "bbox": {
                        "x1": 0.0,
                        "y1": 0.0,
                        "x2": 1.0,
                        "y2": 1.0,
                    },
                }
            ],
            image_width=100,
            image_height=100,
            confidence_threshold=0.1,
            ball_label_id=0,
        )


def test_query_feature_vector_combines_context_logits_and_box() -> None:
    vector = query_feature_vector(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        np.asarray(
            [[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]],
            dtype=np.float32,
        ),
        query_index=1,
    )

    assert vector.dtype == np.float64
    assert vector.tolist() == pytest.approx(
        [3.0, 4.0, 0.3, 0.4, 0.25, 0.25, 0.1, 0.1]
    )


def _track_perception() -> dict[str, object]:
    return {
        "detections": [
            {
                "detection_id": "ball:10:0",
                "frame": 10,
                "bbox": {"x1": 10.0, "y1": 10.0, "x2": 14.0, "y2": 14.0},
            },
            {
                "detection_id": "ball:13:0",
                "frame": 13,
                "bbox": {"x1": 16.0, "y1": 10.0, "x2": 20.0, "y2": 14.0},
            },
            {
                "detection_id": "ball:20:0",
                "frame": 20,
                "bbox": {"x1": 1.0, "y1": 1.0, "x2": 3.0, "y2": 3.0},
            },
        ],
        "ball_tracks": [
            {
                "points": [
                    {
                        "frame": 10,
                        "center": {"x": 12.0, "y": 12.0},
                        "visible": True,
                        "predicted": False,
                    },
                    {
                        "frame": 13,
                        "center": {"x": 18.0, "y": 12.0},
                        "visible": True,
                        "predicted": False,
                    },
                ]
            }
        ],
    }


def test_expand_detection_ids_to_tracks_includes_visible_members() -> None:
    expanded = expand_detection_ids_to_tracks(
        _track_perception(),
        {"ball:10:0", "ball:20:0"},
    )

    assert expanded == {"ball:10:0", "ball:13:0", "ball:20:0"}


def test_track_query_features_summarizes_sequence_and_singleton() -> None:
    perception = _track_perception()
    feature_by_id = {
        "ball:10:0": np.asarray(
            [1.0, 0.0, 0.2, -0.1, -0.3, 0.5, 0.5, 0.1, 0.1]
        ),
        "ball:13:0": np.asarray(
            [0.0, 1.0, 0.6, -0.2, -0.4, 0.6, 0.5, 0.1, 0.1]
        ),
        "ball:20:0": np.asarray(
            [1.0, 1.0, -0.2, -0.3, -0.4, 0.1, 0.1, 0.05, 0.05]
        ),
    }

    names, output = track_query_features(
        perception,
        feature_by_id,
        context_width=2,
        label_width=3,
    )

    assert len(names) == 36
    assert output["ball:10:0"].shape == (36,)
    assert output["ball:10:0"][:3].tolist() == [1.0, 2.0, 3.0]
    assert output["ball:20:0"][:3].tolist() == [0.0, 1.0, 0.0]
    assert np.isfinite(np.stack(list(output.values()))).all()
