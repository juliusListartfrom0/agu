from __future__ import annotations

import numpy as np
import pytest
import torch

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from scripts.run_replay_transition_evidence import (
    candidate_anchor,
    event_ids_from_review_plan,
    predict_transition_probabilities,
)


class _FakeTransNet(torch.nn.Module):
    def forward(self, frames: torch.Tensor):
        logits = torch.full((1, 100, 1), -10.0)
        logits[:, 50, :] = 10.0
        return logits, {"many_hot": logits}


def test_predict_transition_probabilities_preserves_input_length() -> None:
    frames = np.zeros((137, 27, 48, 3), dtype=np.uint8)

    values = predict_transition_probabilities(_FakeTransNet(), frames)

    assert values.shape == (137,)
    assert int((values > 0.5).sum()) == 3


def test_predict_transition_probabilities_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="frames"):
        predict_transition_probabilities(
            _FakeTransNet(),
            np.zeros((20, 48, 27, 3), dtype=np.uint8),
        )


def test_candidate_anchor_prefers_highest_hit_cluster() -> None:
    event = GameEventResponse(
        event_id="shot-1",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video-1",
        start_frame=10,
        end_frame=50,
        evidence=[
            EventEvidenceResponse(
                evidence_id="weak",
                kind="ball_rim_proximity_cluster",
                source_video_id="video-1",
                start_frame=20,
                end_frame=20,
                confidence=0.2,
                details={"hit_count": 1, "review_anchor_frame": 20},
            ),
            EventEvidenceResponse(
                evidence_id="strong",
                kind="ball_rim_proximity_cluster",
                source_video_id="video-1",
                start_frame=40,
                end_frame=40,
                confidence=0.3,
                details={"hit_count": 4, "review_anchor_frame": 42},
            ),
        ],
    )

    assert candidate_anchor(event) == 42


def test_event_ids_from_review_plan_selects_only_exact_source_binding() -> None:
    plan = {
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "sealed_blind_video_sha256s": ["blind-sha"],
        "examples": [
            {
                "source_video_sha256": "video-sha",
                "candidate_bundle_sha256": "bundle-sha",
                "event_id": "shot-1",
            },
            {
                "source_video_sha256": "other-video",
                "candidate_bundle_sha256": "other-bundle",
                "event_id": "shot-2",
            },
        ],
    }

    assert event_ids_from_review_plan(
        plan,
        raw_video_sha256="video-sha",
        candidate_bundle_sha256="bundle-sha",
    ) == {"shot-1"}

    with pytest.raises(ValueError, match="sealed blind"):
        event_ids_from_review_plan(
            plan,
            raw_video_sha256="blind-sha",
            candidate_bundle_sha256="bundle-sha",
        )
