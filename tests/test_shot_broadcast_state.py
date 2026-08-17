from __future__ import annotations

import copy

import pytest

from app.analysis.broadcast_clock import (
    BROADCAST_CLOCK_EVIDENCE_SCHEMA,
    seal_broadcast_clock_artifact,
)
from app.analysis.shot_broadcast_state import (
    FEATURE_NAMES,
    build_broadcast_state_evidence_artifact,
    extract_broadcast_state_features,
    seal_broadcast_state_evidence_artifact,
    verify_broadcast_state_evidence_artifact,
)
from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    seal_scene_embedding_artifact,
)


def _clock_event(
    event_id: str,
    clock_seconds: list[int | None],
) -> dict[str, object]:
    samples = []
    for index, (offset, seconds) in enumerate(
        zip((-6.0, -3.0, 0.0, 3.0, 6.0), clock_seconds, strict=True)
    ):
        frame = 820 + index * 90
        samples.append(
            {
                "frame": frame,
                "offset_seconds": offset,
                "read": (
                    None
                    if seconds is None
                    else {
                        "frame": frame,
                        "period": 1,
                        "clock_seconds": seconds,
                        "confidence": 0.9,
                        "raw_text": f"1ST{seconds // 60}:{seconds % 60:02d}",
                        "source": "rapidocr_broadcast_clock_v1",
                    }
                ),
            }
        )
    return {
        "event_id": event_id,
        "anchor_frame": 1000,
        "samples": samples,
        "pre_anchor_read_count": sum(
            sample["read"] is not None
            for sample in samples
            if float(sample["offset_seconds"]) <= -3.0
        ),
        "post_anchor_missing_count": sum(
            sample["read"] is None
            for sample in samples
            if float(sample["offset_seconds"]) >= 0.0
        ),
        "frozen_period": None,
        "frozen_clock_seconds": None,
        "broadcast_state": "unknown",
        "method": "pre_anchor_frozen_clock_disappearance_v1",
    }


def _scene() -> dict[str, object]:
    return seal_scene_embedding_artifact(
        {
            "purpose": "test",
            "producer": "test",
            "training_manifest_sha256": "a" * 64,
            "training_annotation_sha256": ["b" * 64],
            "source_video_sha256": ["d" * 64],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": "e" * 64,
            "backbone_license": "BSD-3-Clause",
            "backbone_weights_url": "https://example.invalid/model",
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "phase_fractions": [0.15, 0.5, 0.85],
            "examples": [
                {
                    "source_video_sha256": "d" * 64,
                    "candidate_bundle_sha256": "c" * 64,
                    "event_id": "event-1",
                    "event_present": True,
                    "phase_embeddings": [
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                    ],
                }
            ],
        }
    )


def _clock_artifact(
    *events: dict[str, object],
) -> dict[str, object]:
    return seal_broadcast_clock_artifact(
        {
            "schema_version": BROADCAST_CLOCK_EVIDENCE_SCHEMA,
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": "d" * 64,
            "candidate_bundle_sha256": "c" * 64,
            "events": list(events),
        }
    )


def test_broadcast_state_features_distinguish_running_frozen_and_missing_clock() -> None:
    running = extract_broadcast_state_features(
        _clock_event("running", [600, 597, 594, 591, 588])
    )
    frozen = extract_broadcast_state_features(
        _clock_event("frozen", [600, 600, 600, 600, 600])
    )
    trailing_missing = extract_broadcast_state_features(
        _clock_event("missing", [600, 597, 594, None, None])
    )

    assert tuple(running) == FEATURE_NAMES
    assert running["read_fraction"] == 1.0
    assert running["clock_elapsed_seconds"] == 12.0
    assert running["clock_elapsed_to_offset_ratio"] == pytest.approx(1.0)
    assert running["frozen_clock_flag"] == 0.0
    assert frozen["dominant_clock_fraction"] == 1.0
    assert frozen["frozen_clock_flag"] == 1.0
    assert trailing_missing["trailing_missing_fraction"] == pytest.approx(0.4)
    assert trailing_missing["disappearance_after_read_flag"] == 1.0
    assert all(
        value == pytest.approx(float(value))
        for feature_map in (running, frozen, trailing_missing)
        for value in feature_map.values()
    )


def test_broadcast_state_artifact_is_label_free_and_tamper_evident() -> None:
    artifact = seal_broadcast_state_evidence_artifact(
        {
            "source_scene_artifact_sha256": "a" * 64,
            "source_clock_artifact_sha256s": ["b" * 64],
            "examples": [
                {
                    "source_video_sha256": "d" * 64,
                    "candidate_bundle_sha256": "c" * 64,
                    "event_id": "event-1",
                    "features": [0.0] * len(FEATURE_NAMES),
                }
            ],
        }
    )

    assert artifact["schema_version"] == "agu.shot-broadcast-state-evidence.v1"
    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    assert "event_present" not in str(artifact)
    assert verify_broadcast_state_evidence_artifact(artifact) == artifact

    leaked = copy.deepcopy(artifact)
    leaked["examples"][0]["event_present"] = True
    with pytest.raises(ValueError, match="labels"):
        seal_broadcast_state_evidence_artifact(leaked)

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["features"][0] = 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_broadcast_state_evidence_artifact(tampered)


def test_builder_requires_exact_unique_scene_clock_coverage_and_binding() -> None:
    scene = _scene()
    clock = _clock_artifact(_clock_event("event-1", [600, 597, 594, 591, 588]))

    artifact = build_broadcast_state_evidence_artifact(
        scene_artifact=scene,
        clock_artifacts=[clock],
    )

    assert artifact["source_scene_artifact_sha256"] == scene["artifact_sha256"]
    assert artifact["source_clock_artifact_sha256s"] == [clock["artifact_sha256"]]
    assert len(artifact["examples"]) == 1
    assert "event_present" not in str(artifact)

    with pytest.raises(ValueError, match="exactly cover"):
        build_broadcast_state_evidence_artifact(
            scene_artifact=scene,
            clock_artifacts=[
                _clock_artifact(
                    _clock_event("event-2", [600, 597, 594, 591, 588])
                )
            ],
        )

    with pytest.raises(ValueError, match="duplicated"):
        build_broadcast_state_evidence_artifact(
            scene_artifact=scene,
            clock_artifacts=[clock, clock],
        )
