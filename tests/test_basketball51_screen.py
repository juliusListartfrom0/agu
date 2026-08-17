from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from app.analysis.basketball51 import seal_basketball51_embedding_artifact


def _module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "screen_basketball51_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "screen_basketball51_embeddings",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_screen_uses_source_group_folds_and_stays_training_only(
    tmp_path: Path,
) -> None:
    labels = ("2p0", "2p1", "3p0", "3p1", "ft0", "ft1", "mp0", "mp1")
    examples = []
    for group_index in range(12):
        for label_index, label in enumerate(labels):
            embedding = [0.0] * len(labels)
            embedding[label_index] = 10.0
            examples.append(
                {
                    "relative_path": f"{label}/{label}_v{group_index:03d}.mp4",
                    "source_group": f"v{group_index:03d}",
                    "label": label,
                    "shot_type": {
                        "2p": "two_point",
                        "3p": "three_point",
                        "ft": "free_throw",
                        "mp": "mid_range",
                    }[label[:2]],
                    "outcome": "made" if label.endswith("1") else "missed",
                    "embedding": embedding,
                }
            )
    artifact = seal_basketball51_embedding_artifact(
        {
            "purpose": "backbone_screening_training_only",
            "subset_manifest_sha256": "subset",
            "backbone": "fixture/backbone",
            "backbone_sha256": "backbone",
            "embedding_dimension": len(labels),
            "clip_frames": 16,
            "examples": examples,
        }
    )
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    result = _module().screen_basketball51_embeddings(
        path,
        fold_count=4,
        regularization_values=(0.01, 0.1),
    )

    assert result["runtime_consumable"] is False
    assert result["promotion_eligible"] is False
    assert result["source_group_count"] == 12
    assert result["tasks"]["field_goal_vs_free_throw"]["best"]["worst_fold_f1"] > 0.99
    assert result["tasks"]["made_vs_missed"]["best"]["worst_fold_f1"] > 0.99
    assert result["tasks"]["eight_class"]["best"]["worst_fold_macro_f1"] > 0.99
    assert len(result["artifact_sha256"]) == 64
