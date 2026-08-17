from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.extract_bard_visual_state_embeddings import (
    verify_acquisition_artifact,
)


def _acquisition() -> dict:
    payload = {
        "schema_version": "agu.bard-visual-state-acquisition.v1",
        "purpose": "offline_broadcast_visual_state_pretraining",
        "runtime_consumable": False,
        "truth_used_for_training_only": True,
        "codex_runtime_answer_used": False,
        "selection_artifact_sha256": "1" * 64,
        "source_revision": "2" * 40,
        "license": "CC-BY-4.0",
        "sealed_blind_video_sha256s": ["3" * 64],
        "examples": [{"path": "validation/2025/multi/example.mp4"}],
        "summary": {"examples": 1},
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_verify_bard_acquisition_rejects_tampering() -> None:
    acquisition = _acquisition()
    assert verify_acquisition_artifact(acquisition)["artifact_sha256"]

    tampered = copy.deepcopy(acquisition)
    tampered["summary"]["examples"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_acquisition_artifact(tampered)
