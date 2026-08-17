from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_causal_review_spec_from_times import build_spec


def test_build_spec_supports_versioned_review_tags(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"a","clip_id":"clip-a","fps":10,"duration_seconds":100}]}',
        encoding="utf-8",
    )
    artifact = build_spec(
        manifest_path=manifest,
        candidates=["a=20"],
        radius_seconds=4.0,
        sample_period_seconds=0.2,
        review_tag="v8",
    )
    assert artifact["purpose"] == "offline_continuous_game_manual_causal_review_v8"
    assert artifact["examples"][0]["review_id"] == "a-20s-v8"


def test_build_spec_rejects_unsafe_review_tag(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"a","clip_id":"clip-a","fps":10,"duration_seconds":100}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="review_tag"):
        build_spec(
            manifest_path=manifest,
            candidates=["a=20"],
            radius_seconds=4.0,
            sample_period_seconds=0.2,
            review_tag="../labels",
        )


@pytest.mark.parametrize("fps", [30000 / 1001, 30.0, 60.0])
def test_build_spec_supports_strict_64_frame_half_open_sampling(
    tmp_path: Path,
    fps: float,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        (
            '{"schema_version":"agu.vru-basketball-source-manifest.v1",'
            f'"videos":[{{"location":"a","clip_id":"clip-a","fps":{fps},'
            '"duration_seconds":100}]}'
        ),
        encoding="utf-8",
    )

    artifact = build_spec(
        manifest_path=manifest,
        candidates=["a=20"],
        radius_seconds=4.0,
        sample_period_seconds=0.125,
        sample_count=64,
        review_tag="closure_v1",
    )

    expected = [round((16.0 + index * 0.125) * fps) for index in range(64)]
    assert artifact["sample_count"] == 64
    assert artifact["interval_semantics"] == "half_open"
    assert artifact["examples"][0]["frame_indexes"] == expected
    assert len(artifact["examples"][0]["frame_indexes"]) == 64
    assert artifact["examples"][0]["frame_indexes"][-1] < round(24.0 * fps)


def test_build_spec_keeps_legacy_inclusive_sampling_when_count_is_omitted(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"a","clip_id":"clip-a","fps":30,"duration_seconds":100}]}',
        encoding="utf-8",
    )

    artifact = build_spec(
        manifest_path=manifest,
        candidates=["a=20"],
        radius_seconds=4.0,
        sample_period_seconds=0.2,
        review_tag="v8",
    )

    assert "sample_count" not in artifact
    assert "interval_semantics" not in artifact
    assert len(artifact["examples"][0]["frame_indexes"]) == 41
    assert artifact["examples"][0]["frame_indexes"][-1] == round(24.0 * 30.0)


def test_build_spec_rejects_inconsistent_strict_sample_count(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"a","clip_id":"clip-a","fps":30,"duration_seconds":100}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_count"):
        build_spec(
            manifest_path=manifest,
            candidates=["a=20"],
            radius_seconds=4.0,
            sample_period_seconds=0.2,
            sample_count=64,
            review_tag="closure_v1",
        )


def test_build_spec_rejects_quantized_frame_on_half_open_right_boundary(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"a","clip_id":"clip-a","fps":7.875,"duration_seconds":100}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="half-open|right boundary"):
        build_spec(
            manifest_path=manifest,
            candidates=["a=10.03"],
            radius_seconds=4.0,
            sample_period_seconds=0.125,
            sample_count=64,
            review_tag="closure_v1",
        )


def test_build_spec_rejects_candidates_that_alias_after_quantization(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"a","clip_id":"clip-a","fps":30,"duration_seconds":100}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="review ID|physical frame"):
        build_spec(
            manifest_path=manifest,
            candidates=["a=20.0000001", "a=20.0000002"],
            radius_seconds=4.0,
            sample_period_seconds=0.125,
            sample_count=64,
            review_tag="closure_v1",
        )


@pytest.mark.parametrize("sample_count", (1, 2))
def test_build_spec_requires_at_least_three_strict_samples(
    tmp_path: Path,
    sample_count: int,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"a","clip_id":"clip-a","fps":30,"duration_seconds":100}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least three"):
        build_spec(
            manifest_path=manifest,
            candidates=["a=20"],
            radius_seconds=1.0,
            sample_period_seconds=2.0 / sample_count,
            sample_count=sample_count,
            review_tag="closure_v1",
        )


def test_build_spec_rejects_legacy_period_that_overshoots_window(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"a","clip_id":"clip-a","fps":30,"duration_seconds":10}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evenly cover|review span"):
        build_spec(
            manifest_path=manifest,
            candidates=["a=6"],
            radius_seconds=4.0,
            sample_period_seconds=3.0,
            review_tag="v8",
        )


def test_build_spec_rejects_unsafe_manifest_source_id(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"agu.vru-basketball-source-manifest.v1","videos":[{"location":"../escape","clip_id":"clip-a","fps":30,"duration_seconds":100}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source ID"):
        build_spec(
            manifest_path=manifest,
            candidates=["../escape=20"],
            radius_seconds=4.0,
            sample_period_seconds=0.2,
            review_tag="v8",
        )
