from __future__ import annotations

import json
import sys

from scripts import screen_shot_overlay_fusion


def test_screen_cli_reads_promotion_from_variant_gate(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        screen_shot_overlay_fusion,
        "screen_shot_overlay_fusion",
        lambda **_: {
            "artifact_sha256": "a" * 64,
            "best_variant": {
                "name": "base",
                "gate": {"promoted": False},
            },
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_shot_overlay_fusion.py",
            "--scene-embeddings",
            str(source),
            "--broadcast-state",
            str(source),
            "--overlay-state",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert screen_shot_overlay_fusion.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["promotion_eligible"] is False
