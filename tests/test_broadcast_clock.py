from __future__ import annotations

import copy

import pytest

from app.analysis.broadcast_clock import (
    BROADCAST_CLOCK_EVIDENCE_SCHEMA,
    BroadcastClockRead,
    BroadcastClockSample,
    evaluate_broadcast_clock_proxy,
    parse_broadcast_clock,
    seal_broadcast_clock_artifact,
    summarize_clock_disappearance,
    verify_broadcast_clock_artifact,
)


def _ocr(text: str, confidence: float, x: float) -> list[object]:
    return [
        [[x, 20], [x + 40, 20], [x + 40, 35], [x, 35]],
        text,
        confidence,
    ]


def _read(frame: int, period: int = 1, seconds: int = 611) -> BroadcastClockRead:
    return BroadcastClockRead(
        frame=frame,
        period=period,
        clock_seconds=seconds,
        confidence=0.91,
        raw_text="1ST10:11",
    )


def _sample(offset: float, read: BroadcastClockRead | None) -> BroadcastClockSample:
    return BroadcastClockSample(
        frame=round(1000 + 30 * offset),
        offset_seconds=offset,
        read=read,
    )


def test_parser_handles_separate_period_and_clock_tokens() -> None:
    result = parse_broadcast_clock(
        [_ocr("ATL", 0.99, 20), _ocr("1ST", 0.97, 100), _ocr("10:09", 0.98, 150)],
        frame=120,
        image_height=100,
    )

    assert result is not None
    assert (result.period, result.clock_seconds) == (1, 609)
    assert result.confidence == 0.97


def test_parser_handles_ocr_period_confusion_and_ignores_shot_clock_suffix() -> None:
    result = parse_broadcast_clock(
        [_ocr("1sT11:51:17", 0.91, 100)],
        frame=120,
        image_height=100,
    )

    assert result is not None
    assert (result.period, result.clock_seconds) == (1, 711)


def test_parser_requires_period_and_rejects_invalid_clock() -> None:
    assert (
        parse_broadcast_clock(
            [_ocr("10:09", 0.99, 100)],
            frame=120,
            image_height=100,
        )
        is None
    )
    assert (
        parse_broadcast_clock(
            [_ocr("1ST13:70", 0.99, 100)],
            frame=120,
            image_height=100,
        )
        is None
    )


def test_clock_disappearance_flags_only_frozen_pre_anchor_pattern() -> None:
    evidence = summarize_clock_disappearance(
        event_id="shot-1",
        anchor_frame=1000,
        samples=[
            _sample(-6, _read(820)),
            _sample(-3, _read(910)),
            _sample(0, None),
            _sample(3, None),
            _sample(6, None),
        ],
    )

    assert evidence.broadcast_state == "replay"
    assert (evidence.frozen_period, evidence.frozen_clock_seconds) == (1, 611)


@pytest.mark.parametrize(
    "samples",
    [
        [
            _sample(-6, _read(820, seconds=611)),
            _sample(-3, _read(910, seconds=608)),
            _sample(0, None),
            _sample(3, None),
            _sample(6, None),
        ],
        [
            _sample(-6, None),
            _sample(-3, None),
            _sample(0, None),
            _sample(3, _read(1090)),
            _sample(6, _read(1180)),
        ],
        [
            _sample(-6, _read(820)),
            _sample(-3, _read(910)),
            _sample(0, None),
            _sample(3, _read(1090)),
            _sample(6, None),
        ],
    ],
)
def test_clock_disappearance_fails_closed_on_ordinary_or_incomplete_patterns(
    samples: list[BroadcastClockSample],
) -> None:
    evidence = summarize_clock_disappearance(
        event_id="shot-1",
        anchor_frame=1000,
        samples=samples,
    )

    assert evidence.broadcast_state == "unknown"


def test_sealed_artifact_detects_tampering() -> None:
    evidence = summarize_clock_disappearance(
        event_id="shot-1",
        anchor_frame=1000,
        samples=[
            _sample(-6, _read(820)),
            _sample(-3, _read(910)),
            _sample(0, None),
            _sample(3, None),
            _sample(6, None),
        ],
    )
    artifact = seal_broadcast_clock_artifact(
        {
            "schema_version": BROADCAST_CLOCK_EVIDENCE_SCHEMA,
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": "video",
            "candidate_bundle_sha256": "bundle",
            "events": [
                {
                    **evidence.__dict__,
                    "samples": [
                        {
                            **sample.__dict__,
                            "read": sample.read.__dict__ if sample.read else None,
                        }
                        for sample in evidence.samples
                    ],
                }
            ],
        }
    )
    assert verify_broadcast_clock_artifact(artifact) == artifact
    tampered = copy.deepcopy(artifact)
    tampered["events"][0]["broadcast_state"] = "unknown"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_broadcast_clock_artifact(tampered)


def test_evaluator_requires_bound_offline_replay_labels() -> None:
    evidence = summarize_clock_disappearance(
        event_id="shot-1",
        anchor_frame=1000,
        samples=[
            _sample(-6, _read(820)),
            _sample(-3, _read(910)),
            _sample(0, None),
            _sample(3, None),
            _sample(6, None),
        ],
    )
    artifact = seal_broadcast_clock_artifact(
        {
            "schema_version": BROADCAST_CLOCK_EVIDENCE_SCHEMA,
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": "video",
            "candidate_bundle_sha256": "bundle",
            "events": [
                {
                    **evidence.__dict__,
                    "samples": [
                        {
                            **sample.__dict__,
                            "read": sample.read.__dict__ if sample.read else None,
                        }
                        for sample in evidence.samples
                    ],
                }
            ],
        }
    )
    metrics = evaluate_broadcast_clock_proxy(
        labels=[
            {
                "schema_version": "agu.shot-validity-labels.v1",
                "runtime_consumable": False,
                "source_video_sha256": "video",
                "candidate_bundle_sha256": "bundle",
                "examples": [
                    {
                        "event_id": "shot-1",
                        "event_present": False,
                        "review_note": "replay window",
                    }
                ],
            }
        ],
        evidence=[artifact],
        minimum_replay_predictions=1,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["promotion_eligible"] is True
