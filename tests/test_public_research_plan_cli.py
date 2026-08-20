from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "build_public_research_plan.py"
    spec = importlib.util.spec_from_file_location("build_public_research_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_media_overrides_requires_slug_mapping(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "overrides.json"
    payload = {
        "target-a": {
            "provider": "internet_archive",
            "identifier": "archive-item",
            "filename": "game.ogv",
            "url": "https://archive.org/download/archive-item/game.ogv",
            "declared_license": None,
            "match_basis": "date_teams_and_game_number",
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert module._load_media_overrides(path) == payload

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        module._load_media_overrides(path)
