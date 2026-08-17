from __future__ import annotations

from pathlib import Path

from scripts.prepare_bard_visual_state_subset import load_benchmark_rows


def test_load_benchmark_rows_normalizes_official_relative_paths(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "files,actions_name,number_actions\n"
        '"../validation/2025/multi/bos-vs-ind-0022300507_364.mp4",'
        "\"[{'action': 'Free Throw'}]\",1\n",
        encoding="utf-8",
    )

    rows = load_benchmark_rows([benchmark])

    assert rows == [
        {
            "path": "validation/2025/multi/bos-vs-ind-0022300507_364.mp4",
            "actions": ["Free Throw"],
        }
    ]
