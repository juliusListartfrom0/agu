from pathlib import Path

from scripts import verify_harness


def test_harness_prefers_the_canonical_dot_venv() -> None:
    python = verify_harness.python_with_pytest()

    assert python is not None
    assert Path(python).parent.parent.name == ".venv"


def test_harness_rejects_legacy_virtualenv_commands_outside_markdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    canonical_python = tmp_path / ".venv/bin/python"
    canonical_python.parent.mkdir(parents=True)
    canonical_python.touch()
    vscode = tmp_path / ".vscode"
    vscode.mkdir()
    (vscode / "settings.json").write_text(
        '{"python.defaultInterpreterPath":"${workspaceFolder}/venv/bin/python"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_harness, "ROOT", tmp_path)
    result = verify_harness.CheckResult()

    verify_harness.check_canonical_virtualenv(result)

    assert any(
        "Legacy virtual-environment command remains in .vscode/settings.json"
        in failure
        for failure in result.failures
    )
