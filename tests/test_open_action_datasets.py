from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.open_action_datasets import (
    import_bard_annotations,
    verify_open_action_dataset,
)


def _write_bard_csv(path: Path, *, action: str = "2PT Shot") -> None:
    path.write_text(
        "urls;actions;numerosity\n"
        "https://www.nba.com/stats/events/?GameEventID=7&GameID=002;"
        f"[{{'player': '8', 'action': '{action}', 'result': True, "
        "'assisted': True, 'other_player': '3', 'color': 'black'}];1\n",
        encoding="utf-8",
    )


def test_bard_import_is_training_only_and_does_not_claim_media_rights(tmp_path: Path) -> None:
    source = tmp_path / "dataset.csv"
    _write_bard_csv(source)

    catalog = import_bard_annotations(source, repository_revision="abc123")
    verified = verify_open_action_dataset(json.loads(json.dumps(catalog)))

    assert verified["runtime_consumable"] is False
    assert verified["media_included"] is False
    assert verified["media_rights_verified"] is False
    assert verified["records"][0]["source_locator"] == {
        "game_id": "002",
        "event_id": "7",
    }
    assert verified["records"][0]["actions"][0] == {
        "event_type": "field_goal_attempt",
        "source_action": "2PT Shot",
        "jersey_number": "8",
        "jersey_color": "black",
        "shot_value": 2,
        "made": True,
        "assisted": True,
        "related_jersey_number": "3",
    }


def test_bard_import_rejects_unknown_action_and_tampering(tmp_path: Path) -> None:
    source = tmp_path / "dataset.csv"
    _write_bard_csv(source, action="Unknown")
    with pytest.raises(ValueError, match="unsupported BARD action"):
        import_bard_annotations(source, repository_revision="abc123")

    _write_bard_csv(source)
    catalog = import_bard_annotations(source, repository_revision="abc123")
    catalog["records"][0]["actions"][0]["jersey_number"] = "99"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_open_action_dataset(catalog)
