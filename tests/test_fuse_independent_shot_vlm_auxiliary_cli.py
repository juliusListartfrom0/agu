from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import fuse_independent_shot_vlm_auxiliary as fusion_cli
from scripts.fuse_independent_shot_vlm_auxiliary import _write_json_atomic


def test_atomic_writer_does_not_follow_predictable_temporary_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fused.json"
    protected = tmp_path / "protected.txt"
    protected.write_text("do-not-overwrite\n", encoding="utf-8")
    legacy_temporary = output.with_suffix(output.suffix + ".tmp")
    legacy_temporary.symlink_to(protected)

    _write_json_atomic(output, {"result": "ok"})

    assert protected.read_text(encoding="utf-8") == "do-not-overwrite\n"
    assert json.loads(output.read_text(encoding="utf-8")) == {"result": "ok"}
    assert legacy_temporary.is_symlink()


@pytest.mark.parametrize("input_name", ("plan", "vlm", "auxiliary"))
@pytest.mark.parametrize(
    "alias_kind",
    (
        "direct",
        "output_symlink",
        "input_symlink",
        "output_parent_symlink",
        "input_parent_symlink",
    ),
)
def test_cli_rejects_output_input_alias_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    input_paths = {
        "plan": tmp_path / "plan.json",
        "vlm": tmp_path / "vlm.json",
        "auxiliary": tmp_path / "auxiliary.json",
    }
    for name, path in input_paths.items():
        path.write_text(f"{name} evidence\n", encoding="utf-8")
    output = input_paths[input_name]
    if alias_kind == "output_symlink":
        output = tmp_path / f"{input_name}-output-alias.json"
        output.symlink_to(input_paths[input_name])
    elif alias_kind == "input_symlink":
        physical_input = tmp_path / f"{input_name}-physical.json"
        input_paths[input_name].replace(physical_input)
        input_paths[input_name].symlink_to(physical_input)
        output = physical_input
    elif alias_kind == "output_parent_symlink":
        output_parent = tmp_path / "output-parent-alias"
        output_parent.symlink_to(tmp_path, target_is_directory=True)
        output = output_parent / input_paths[input_name].name
    elif alias_kind == "input_parent_symlink":
        physical_parent = tmp_path / "physical-input-parent"
        physical_parent.mkdir()
        physical_input = physical_parent / input_paths[input_name].name
        input_paths[input_name].replace(physical_input)
        input_parent = tmp_path / "input-parent-alias"
        input_parent.symlink_to(physical_parent, target_is_directory=True)
        input_paths[input_name] = input_parent / physical_input.name
        output = physical_input
    snapshots = {name: path.read_bytes() for name, path in input_paths.items()}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fuse_independent_shot_vlm_auxiliary.py",
            "--plan",
            str(input_paths["plan"]),
            "--vlm-predictions",
            str(input_paths["vlm"]),
            "--auxiliary",
            str(input_paths["auxiliary"]),
            "--expected-auxiliary-sha256",
            "a" * 64,
            "--output",
            str(output),
        ],
    )

    def reject_read(_path: Path, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("output/input alias must be rejected before JSON reading")

    monkeypatch.setattr(Path, "read_text", reject_read)

    with pytest.raises(ValueError, match="output.*input"):
        fusion_cli.main()

    assert {
        name: path.read_bytes() for name, path in input_paths.items()
    } == snapshots


def test_cli_uses_frozen_resolved_output_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths = {
        "plan": tmp_path / "plan.json",
        "vlm": tmp_path / "vlm.json",
        "auxiliary": tmp_path / "auxiliary.json",
    }
    for path in input_paths.values():
        path.write_text("{}\n", encoding="utf-8")
    first_destination = tmp_path / "first-destination"
    second_destination = tmp_path / "second-destination"
    first_destination.mkdir()
    second_destination.mkdir()
    output_parent = tmp_path / "output-parent"
    output_parent.symlink_to(first_destination, target_is_directory=True)
    output = output_parent / "fused.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fuse_independent_shot_vlm_auxiliary.py",
            "--plan",
            str(input_paths["plan"]),
            "--vlm-predictions",
            str(input_paths["vlm"]),
            "--auxiliary",
            str(input_paths["auxiliary"]),
            "--expected-auxiliary-sha256",
            "a" * 64,
            "--output",
            str(output),
        ],
    )

    def redirect_after_validation(**_kwargs: object) -> dict[str, object]:
        output_parent.unlink()
        output_parent.symlink_to(second_destination, target_is_directory=True)
        return {
            "artifact_sha256": "b" * 64,
            "coverage": {"event_count": 0},
        }

    monkeypatch.setattr(
        fusion_cli,
        "fuse_vlm_with_frozen_auxiliary",
        redirect_after_validation,
    )

    assert fusion_cli.main() == 0

    assert json.loads(
        (first_destination / "fused.json").read_text(encoding="utf-8")
    )["artifact_sha256"] == "b" * 64
    assert not (second_destination / "fused.json").exists()
