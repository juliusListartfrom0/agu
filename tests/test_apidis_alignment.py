from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.apidis_alignment import (
    events_in_clip,
    frame_index_for_local_timestamp,
    parse_apidis_epoch_seconds,
    parse_ball_position_rows,
)


def test_parse_apidis_time_and_map_to_25_fps_clip() -> None:
    assert parse_apidis_epoch_seconds("1,207,759,620.500000s") == 1207759620.5
    assert frame_index_for_local_timestamp("184700.000", "184700", fps=25.0, frame_count=1500) == 0
    assert frame_index_for_local_timestamp("184759.960", "184700", fps=25.0, frame_count=1500) == 1499
    assert frame_index_for_local_timestamp("184800.000", "184700", fps=25.0, frame_count=1500) is None


def test_ball_rows_keep_source_coordinates_and_skip_comments(tmp_path: Path) -> None:
    path = tmp_path / "camera.ballposition.txt"
    path.write_text(
        "# header\n184722.343  1574.9836  577.9864\n"
        "184722.390  1557.9723  565.9770\n",
        encoding="utf-8",
    )
    rows = parse_ball_position_rows(path)
    assert rows == [
        {"timestamp": "184722.343", "x": 1574.9836, "y": 577.9864},
        {"timestamp": "184722.390", "x": 1557.9723, "y": 565.977},
    ]


def test_events_in_clip_extract_nested_action_types(tmp_path: Path) -> None:
    path = tmp_path / "quarter.events.xml"
    path.write_text(
        """<?xml version='1.0'?>
<Video Start-time='1,207,759,560.000000s' End-time='1,207,760,669.650000s'>
  <Clock-event Timestamp='1,207,759,620.400000s' Player-no='7' Team-color='Team B'>
    <Throw Score='2' Layup='YES'/>
    <Rebound/>
  </Clock-event>
  <Clock-event Timestamp='1,207,759,700.000000s'>
    <Foul/>
  </Clock-event>
</Video>
""",
        encoding="utf-8",
    )
    rows = events_in_clip(
        [path],
        clip_start_epoch=1207759620.0,
        clip_duration_seconds=60.0,
        fps=25.0,
        frame_count=1500,
    )
    assert len(rows) == 1
    assert rows[0]["frame"] == 10
    assert rows[0]["action_types"] == ["Rebound", "Throw"]
    assert rows[0]["attributes"] == {"Player-no": "7", "Team-color": "Team B"}


def test_invalid_ball_row_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("184722.343 nope 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid ball annotation"):
        parse_ball_position_rows(path)
