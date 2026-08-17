from __future__ import annotations

from app.analysis.ball_candidate_review import select_stratified_candidates


def test_select_stratified_candidates_is_deterministic_and_spans_each_band() -> None:
    detections = [
        {
            "detection_id": f"d:{index}",
            "frame": index,
            "confidence": confidence,
        }
        for index, confidence in enumerate(
            [0.11, 0.12, 0.13, 0.14, 0.21, 0.22, 0.23, 0.24]
        )
    ]

    selected = select_stratified_candidates(
        detections,
        bands=((0.1, 0.2), (0.2, 0.3)),
        samples_per_band=2,
    )

    assert [row["detection_id"] for row in selected] == ["d:0", "d:3", "d:4", "d:7"]
    assert [row["confidence_band"] for row in selected] == [
        [0.1, 0.2],
        [0.1, 0.2],
        [0.2, 0.3],
        [0.2, 0.3],
    ]


def test_select_stratified_candidates_rejects_underfilled_band() -> None:
    detections = [{"detection_id": "d:1", "frame": 1, "confidence": 0.15}]

    try:
        select_stratified_candidates(
            detections,
            bands=((0.1, 0.2),),
            samples_per_band=2,
        )
    except ValueError as exc:
        assert "underfilled" in str(exc)
    else:
        raise AssertionError("underfilled review band was accepted")
