from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_vru_causal_temporal_feature_plan import _sealed_feature_plan

from app.analysis import vru_causal_temporal_retrospective as module
from scripts import extract_vru_causal_tiled_swin_embeddings as extractor_cli
from scripts import screen_vru_causal_temporal_retrospective as screen_cli
from scripts import seal_vru_causal_temporal_feature_plan as plan_cli


def test_disk_budget_groups_targets_by_filesystem_without_creating_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_disk_usage(path: Path) -> SimpleNamespace:
        calls.append(Path(path))
        return SimpleNamespace(free=6 * 1024**3)

    monkeypatch.setattr(module.shutil, "disk_usage", fake_disk_usage)
    first = tmp_path / "missing" / "a.json"
    second = tmp_path / "other" / "b.json"

    module.verify_disk_write_budget(
        output_paths=(first, second),
        worst_case_new_bytes=512 * 1024**2,
    )

    assert len(calls) == 1
    assert not first.parent.exists()
    assert not second.parent.exists()


def test_disk_budget_fails_when_worst_case_would_cross_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=4 * 1024**3),
    )
    with pytest.raises(ValueError, match="reserve"):
        module.verify_disk_write_budget(
            output_paths=(tmp_path / "future" / "artifact.json",),
            worst_case_new_bytes=1024**3,
        )


@pytest.mark.parametrize(
    ("worst_case_new_bytes", "reserve_bytes"),
    (
        (True, 3_758_096_384),
        (1, True),
        (1, 3_758_096_383),
        (0, 3_758_096_384),
    ),
)
def test_disk_budget_rejects_noncanonical_or_weakened_values(
    tmp_path: Path,
    worst_case_new_bytes: int,
    reserve_bytes: int,
) -> None:
    with pytest.raises(ValueError):
        module.verify_disk_write_budget(
            output_paths=(tmp_path / "artifact.json",),
            worst_case_new_bytes=worst_case_new_bytes,
            reserve_bytes=reserve_bytes,
        )


def test_observed_environment_matches_the_approved_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in module.TEMPORAL_ENVIRONMENT_CONTRACT["process_environment"].items():
        monkeypatch.setenv(name, value)

    assert module.observe_temporal_environment_contract() == module.TEMPORAL_ENVIRONMENT_CONTRACT


def test_observed_environment_is_independent_of_backend_load_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threadpoolctl

    for name, value in module.TEMPORAL_ENVIRONMENT_CONTRACT["process_environment"].items():
        monkeypatch.setenv(name, value)
    observe_backends = threadpoolctl.threadpool_info
    monkeypatch.setattr(
        threadpoolctl,
        "threadpool_info",
        lambda: list(reversed(observe_backends())),
    )

    assert module.observe_temporal_environment_contract() == module.TEMPORAL_ENVIRONMENT_CONTRACT


def test_plan_cli_rejects_explicit_output_alias_before_any_json_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    specs = tuple(tmp_path / name for name in ("requirement.md", "solution.md", "gate-review.md"))
    for path in specs:
        path.write_text("spec\n", encoding="utf-8")
    monkeypatch.setattr(plan_cli, "_read_contract", lambda *_args, **_kwargs: pytest.fail("JSON read occurred"))

    with pytest.raises(ValueError, match="output aliases an input"):
        plan_cli._preflight_explicit_paths(
            approval_path=approval,
            contract_path=contract,
            spec_paths=specs,
            output_path=approval,
        )


def test_plan_cli_rejects_symlink_or_hardlink_output_alias(tmp_path: Path) -> None:
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    specs = tuple(tmp_path / name for name in ("requirement.md", "solution.md", "gate-review.md"))
    for path in specs:
        path.write_text("spec\n", encoding="utf-8")
    outputs = (tmp_path / "approval-link.json", tmp_path / "approval-hardlink.json")
    outputs[0].symlink_to(approval)
    outputs[1].hardlink_to(approval)

    for output in outputs:
        with pytest.raises(ValueError, match="output aliases an input"):
            plan_cli._preflight_explicit_paths(
                approval_path=approval,
                contract_path=contract,
                spec_paths=specs,
                output_path=output,
            )


def test_plan_cli_atomic_writer_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    output.write_text("old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        plan_cli._write_json_atomic(output, {"value": 1})
    assert output.read_text(encoding="utf-8") == "old\n"


def test_plan_and_terminal_member_writers_use_compact_sorted_canonical_json(
    tmp_path: Path,
) -> None:
    payload = {"z": [3, 2, 1], "a": "\u65f6\u95f4"}
    expected = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output = tmp_path / "plan.json"

    plan_cli._write_json_atomic(output, payload)

    assert {
        "plan_writer": output.read_bytes(),
        "terminal_member_encoder": extractor_cli._canonical_json_bytes(payload),
    } == {
        "plan_writer": expected,
        "terminal_member_encoder": expected,
    }


def test_plan_cli_rejects_output_alias_with_contract_leaf(tmp_path: Path) -> None:
    leaf = tmp_path / "video.mp4"
    leaf.write_bytes(b"video")
    paths = {field: str(tmp_path / f"{field}.json") for field in module.Task0257InputPaths.__dataclass_fields__}
    for field in (
        "parent_review_jpegs",
        "parent_candidate_children",
        "parent_label_children",
        "old_embedding_files",
        "harwood_review_jpegs",
        "source_videos",
        "checkpoints",
    ):
        paths[field] = [str(leaf)]
    paths["source_videos"] = [str(leaf)] * 4
    paths["checkpoints"] = [str(leaf)] * 2
    contract = {"paths": paths, "expected_receipts": {}}

    with pytest.raises(ValueError, match="output aliases an input"):
        plan_cli._preflight_contract_paths(contract, output_path=leaf)


def test_plan_cli_lock_is_a_non_writing_flock_on_an_existing_directory(tmp_path: Path) -> None:
    entries_before = tuple(tmp_path.iterdir())
    with plan_cli._exclusive_lock(tmp_path):
        assert tuple(tmp_path.iterdir()) == entries_before
        with pytest.raises(ValueError, match="lock already exists"):
            with plan_cli._exclusive_lock(tmp_path):
                pass
    assert tuple(tmp_path.iterdir()) == entries_before


@pytest.mark.parametrize(
    "relationship",
    ("output_below_input", "output_above_input", "symlink_ancestor", "casefold_ancestor"),
)
def test_plan_cli_rejects_ancestor_descendant_and_casefold_aliases_before_reads_or_writes(
    tmp_path: Path,
    relationship: str,
) -> None:
    immutable = tmp_path / "ImmutableInput"
    immutable.mkdir()
    approval: Path
    if relationship == "output_below_input":
        approval = immutable
        output = immutable / "temporal_feature_plan.json"
    elif relationship == "output_above_input":
        approval = immutable / "approval.json"
        approval.write_text("{}\n", encoding="utf-8")
        output = immutable
    elif relationship == "symlink_ancestor":
        alias = tmp_path / "input-link"
        alias.symlink_to(immutable, target_is_directory=True)
        approval = immutable
        output = alias / "temporal_feature_plan.json"
    else:
        approval = immutable
        output = tmp_path / "immutableinput" / "temporal_feature_plan.json"
    before = tuple(sorted(str(path) for path in tmp_path.rglob("*")))

    with pytest.raises(ValueError, match="output aliases an input"):
        plan_cli._preflight_explicit_paths(
            approval_path=approval,
            contract_path=tmp_path / "contract.json",
            spec_paths=tuple(tmp_path / name for name in ("requirement.md", "solution.md", "gate-review.md")),
            output_path=output,
        )

    assert tuple(sorted(str(path) for path in tmp_path.rglob("*"))) == before


def test_public_outer_fold_guards_every_fit_and_uses_the_exact_estimator_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threadpoolctl

    events: list[tuple[str, int]] = []
    constructor_kwargs: list[dict[str, object]] = []
    state = {"limit_depth": 0}
    real_scaler = module.StandardScaler
    real_logistic = module.LogisticRegression

    @contextmanager
    def guarded_limits(*, limits: int):
        assert limits == 1
        state["limit_depth"] += 1
        try:
            yield
        finally:
            state["limit_depth"] -= 1

    def observe_environment() -> dict[str, object]:
        events.append(("environment", state["limit_depth"]))
        return module.TEMPORAL_ENVIRONMENT_CONTRACT

    def scaler_constructor():
        scaler = real_scaler()
        real_fit_transform = scaler.fit_transform

        def fit_transform(feature_matrix, labels=None, **kwargs):
            events.append(("scaler_fit", state["limit_depth"]))
            return real_fit_transform(feature_matrix, labels, **kwargs)

        scaler.fit_transform = fit_transform
        return scaler

    def logistic_constructor(**kwargs):
        constructor_kwargs.append(dict(kwargs))
        classifier = real_logistic(**kwargs)
        real_fit = classifier.fit

        def fit(feature_matrix, labels, **fit_kwargs):
            events.append(("logistic_fit", state["limit_depth"]))
            return real_fit(feature_matrix, labels, **fit_kwargs)

        classifier.fit = fit
        return classifier

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", guarded_limits)
    monkeypatch.setattr(module, "threadpool_limits", guarded_limits, raising=False)
    monkeypatch.setattr(module, "observe_temporal_environment_contract", observe_environment)
    monkeypatch.setattr(module, "StandardScaler", scaler_constructor)
    monkeypatch.setattr(module, "LogisticRegression", logistic_constructor)

    module.screen_temporal_outer_fold(
        feature_matrix=np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0], [5.0, 1.0], [6.0, 0.0], [7.0, 1.0]],
            dtype=np.float64,
        ),
        labels=np.asarray([False, True] * 4, dtype=np.bool_),
        game_ids=("hazen", "hazen", "randolph", "randolph", "vtv", "vtv", "harwood", "harwood"),
        held_game_id="harwood",
        representation_name=module.TEMPORAL_REPRESENTATION_CONTRACT["name"],
    )

    violations: list[str] = []
    fit_indexes = [index for index, (event, _depth) in enumerate(events) if event.endswith("_fit")]
    for index in fit_indexes:
        event, depth = events[index]
        if depth != 1:
            violations.append(f"{event} ran outside threadpool_limits(1)")
        if index == 0 or events[index - 1] != ("environment", 1):
            violations.append(f"{event} lacked an immediate pre-fit frozen-environment check")
        if index + 1 >= len(events) or events[index + 1] != ("environment", 1):
            violations.append(f"{event} lacked an immediate post-fit frozen-environment check")
    expected_constructor = {
        "C": 0.01,
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": 0,
    }
    if len(fit_indexes) != 8:
        violations.append(f"expected 8 guarded scaler/logistic fits, observed {len(fit_indexes)}")
    for kwargs in constructor_kwargs:
        if kwargs != expected_constructor:
            violations.append(f"logistic constructor drifted: {kwargs!r}")
    assert not violations, violations


def test_candidate_evaluator_uses_the_exact_registered_representation_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    representation = module.TEMPORAL_REPRESENTATION_CONTRACT["name"]
    monkeypatch.setattr(
        module,
        "_logo_variant",
        lambda **kwargs: {
            "representation_name": kwargs["representation_name"],
            "selected_threshold": 0.5,
            "selected_metrics": {
                "balanced_accuracy": 1.0,
                "f1": 1.0,
                "precision": 1.0,
                "recall": 1.0,
            },
        },
    )
    monkeypatch.setattr(module, "_fit_probability_model", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(module, "_model_state", lambda *_args, **_kwargs: {})
    game_ids = tuple(
        game
        for game, count in zip(
            module.TEMPORAL_EVALUATION_PROTOCOL["game_order"],
            module.TEMPORAL_EVALUATION_PROTOCOL["game_row_counts"],
            strict=True,
        )
        for _ in range(count)
    )

    evaluator = module.build_final_evaluator(
        feature_matrices={representation: np.zeros((45, 1536), dtype=np.float64)},
        labels=np.asarray([index % 2 == 0 for index in range(45)], dtype=np.bool_),
        game_ids=game_ids,
        representation_order=(representation,),
        evaluator_kind="candidate",
    )

    assert evaluator["selected_evaluator"]["representation_name"] == representation


def test_resource_sample_uses_frozen_schedule_and_sustained_breach_count() -> None:
    first = module.build_resource_sample(
        attempt_ordinal=1,
        attempt_sample_ordinal=1,
        cumulative_sample_ordinal=1,
        observed_attempt_active_nanoseconds=2_000_000_000,
        system_memory_percent=91.0,
        available_memory_bytes=3 * 1024**3,
        free_swap_bytes=1024**3,
        system_cpu_percent=50.0,
        prior_consecutive_breach_count=0,
    )
    second = module.build_resource_sample(
        attempt_ordinal=1,
        attempt_sample_ordinal=2,
        cumulative_sample_ordinal=2,
        observed_attempt_active_nanoseconds=4_000_000_000,
        system_memory_percent=91.0,
        available_memory_bytes=3 * 1024**3,
        free_swap_bytes=1024**3,
        system_cpu_percent=50.0,
        prior_consecutive_breach_count=first["consecutive_breach_count"],
    )

    assert first["scheduled_cumulative_active_seconds"] == 2
    assert first["breached_limits"] == ["memory_percent"]
    assert second["consecutive_breach_count"] == 2
    assert module.encode_resource_sample_line(second).endswith(b"\n")


def test_resource_log_verifier_rejects_resealed_schedule_drift() -> None:
    sample = module.build_resource_sample(
        attempt_ordinal=1,
        attempt_sample_ordinal=1,
        cumulative_sample_ordinal=1,
        observed_attempt_active_nanoseconds=2_000_000_000,
        system_memory_percent=50.0,
        available_memory_bytes=8_000_000_000,
        free_swap_bytes=8_000_000_000,
        system_cpu_percent=25.0,
        prior_consecutive_breach_count=0,
    )
    encoded = module.encode_resource_sample_line(sample)
    assert module.verify_resource_log_bytes(
        encoded,
        expected_attempt_ordinal=1,
        prior_cumulative_samples=0,
        prior_consecutive_breach_count=0,
    ) == (1, 0)

    drifted = {**sample, "scheduled_cumulative_active_seconds": 4}
    with pytest.raises(ValueError, match="schedule or canonical"):
        module.verify_resource_log_bytes(
            module.encode_resource_sample_line(drifted),
            expected_attempt_ordinal=1,
            prior_cumulative_samples=0,
            prior_consecutive_breach_count=0,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"cumulative_sample_ordinal": 21_601},
        {"attempt_ordinal": True},
        {"system_cpu_percent": float("nan")},
        {"available_memory_bytes": -1},
    ),
)
def test_resource_sample_rejects_noncanonical_or_over_cap_values(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "attempt_ordinal": 1,
        "attempt_sample_ordinal": 1,
        "cumulative_sample_ordinal": 1,
        "observed_attempt_active_nanoseconds": 0,
        "system_memory_percent": 50.0,
        "available_memory_bytes": 3 * 1024**3,
        "free_swap_bytes": 1024**3,
        "system_cpu_percent": 50.0,
        "prior_consecutive_breach_count": 0,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        module.build_resource_sample(**values)


def _completed_attempt_body() -> dict[str, object]:
    plan = _sealed_feature_plan()
    return {
        "attempt_ordinal": 1,
        "prior_attempt_receipt": None,
        "plan_receipt": {"artifact_sha256": "1" * 64, "file_sha256": "2" * 64},
        "task0257_input_receipts": plan["task0257_receipts"],
        "started_prefix_count": 0,
        "completed_prefix_count": 45,
        "new_rows_verified": 45,
        "cumulative_active_runtime_nanoseconds": 1,
        "cumulative_resource_samples": 0,
        "cumulative_resource_log_bytes": 0,
        "resource_log_receipt": {
            "file_sha256": "3" * 64,
            "filename": "resource.jsonl",
            "size_bytes": 0,
        },
        "resume_input_cas": None,
        "resume_output_cas": None,
        "received_signal": None,
        "disposition": "completed",
        "stop_reason": None,
    }


def test_attempt_state_machine_accepts_clean_completion_and_rejects_false_completion() -> None:
    attempt = module.seal_tiled_swin_attempt(_completed_attempt_body())
    assert module.verify_tiled_swin_attempt(attempt) == attempt

    body = _completed_attempt_body()
    body["completed_prefix_count"] = 44
    body["new_rows_verified"] = 44
    with pytest.raises(ValueError, match="completed attempt"):
        module.seal_tiled_swin_attempt(body)

    body = _completed_attempt_body()
    body["resume_input_cas"] = {
        "device": 1,
        "inode": 2,
        "size_bytes": 3,
        "internal_sha256": "4" * 64,
        "file_sha256": "5" * 64,
    }
    with pytest.raises(ValueError, match="attempt one"):
        module.seal_tiled_swin_attempt(body)

    body = _completed_attempt_body()
    body["attempt_ordinal"] = 2
    body["prior_attempt_receipt"] = {
        "schema_version": module.TILED_SWIN_ATTEMPT_SCHEMA,
        "internal_sha256_field": "artifact_sha256",
        "internal_sha256": "6" * 64,
        "file_sha256": "7" * 64,
        "filename": "attempt_record.json",
        "size_bytes": 100,
    }
    with pytest.raises(ValueError, match="resume input"):
        module.seal_tiled_swin_attempt(body)


def test_resume_contract_is_an_exact_verified_plan_prefix(tmp_path: Path) -> None:
    from test_vru_causal_tiled_swin_embeddings import _verified_plan

    plan, _artifact_sha256, _file_sha256, _payload = _verified_plan(tmp_path)
    tiles = [[[float(ordinal + tile)] * 768 for tile in range(4)] for ordinal in range(2)]
    resume = module.seal_tiled_swin_resume(
        {
            "plan_receipt": {
                "artifact_sha256": plan._artifact_sha256,
                "file_sha256": plan._file_sha256,
            },
            "task0257_input_receipts": plan._payload["task0257_receipts"],
            "representation": plan._payload["representation"],
            "producer_environment": plan._payload["environment_contract"],
            "checkpoint_receipt": plan._payload["task0257_receipts"]["checkpoints"][1],
            "source_video_receipts": plan._payload["task0257_receipts"]["source_videos"],
            "attempt_chain_receipts": [],
            "row_count": 45,
            "completed_count": 2,
            "completed_examples": tiles,
        },
        plan=plan,
    )

    assert module.verify_tiled_swin_resume(resume, plan=plan)["completed_count"] == 2

    drifted = dict(resume)
    drifted["completed_examples"] = [*tiles, tiles[0]]
    drifted["artifact_sha256"] = module._canonical_sha256(drifted)
    with pytest.raises(ValueError, match="resume prefix"):
        module.verify_tiled_swin_resume(drifted, plan=plan)


def test_terminal_directory_publisher_uses_exclusive_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / ".stage"
    stage.mkdir()
    (stage / "member.json").write_text("{}\n", encoding="utf-8")
    final = tmp_path / "final_v1"
    calls: list[tuple[Path, Path]] = []

    def fake_exclusive(source: Path, destination: Path) -> None:
        calls.append((source, destination))
        source.rename(destination)

    monkeypatch.setattr(extractor_cli, "_rename_exclusive", fake_exclusive)
    extractor_cli._publish_final_directory(stage, final)

    assert calls == [(stage, final)]
    assert (final / "member.json").read_text(encoding="utf-8") == "{}\n"


def test_exclusive_file_writer_handles_short_os_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = extractor_cli.os.write

    def short_write(descriptor: int, payload: bytes) -> int:
        return original_write(descriptor, payload[:3])

    monkeypatch.setattr(extractor_cli.os, "write", short_write)
    output = tmp_path / "member.bin"
    extractor_cli._write_file_fsync(output, b"abcdefghij")

    assert output.read_bytes() == b"abcdefghij"


def test_worker_reuses_a_strict_private_prefix_before_extracting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_vru_causal_tiled_swin_embeddings import _verified_plan

    plan, artifact_sha256, file_sha256, _payload = _verified_plan(tmp_path)
    prefix = [[[float(tile)] * 768 for tile in range(4)]]
    progress = tmp_path / "private-progress.json"
    progress.write_text(
        json.dumps(
            {
                "schema_version": "agu.vru-causal-tiled-swin-private-progress.v1",
                "completed_count": 1,
                "tile_embeddings": prefix,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    request = {
        "plan_path": str(plan._path),
        "plan_artifact_sha256": artifact_sha256,
        "plan_file_sha256": file_sha256,
        "source_video_paths": [str(tmp_path / f"video-{index}.mp4") for index in range(4)],
        "checkpoint_path": str(tmp_path / "weights.pth"),
        "progress_path": str(progress),
        "started_prefix_count": 1,
    }
    request_path = tmp_path / "request.json"
    encoded = (json.dumps(request, indent=2) + "\n").encode("utf-8")
    request_path.write_bytes(encoded)
    captured: dict[str, object] = {}

    def fake_extract(**kwargs):
        captured.update(kwargs)
        return prefix

    monkeypatch.setattr(extractor_cli, "_extract_tiled_swin_rows", fake_extract)
    assert (
        extractor_cli._worker_main(
            request_path,
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        == 0
    )
    assert captured["initial_rows"] == prefix


def test_worker_rejects_progress_outside_its_request_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_vru_causal_tiled_swin_embeddings import _verified_plan

    plan, artifact_sha256, file_sha256, _payload = _verified_plan(tmp_path)
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    request = {
        "plan_path": str(plan._path),
        "plan_artifact_sha256": artifact_sha256,
        "plan_file_sha256": file_sha256,
        "source_video_paths": [str(tmp_path / f"video-{index}.mp4") for index in range(4)],
        "checkpoint_path": str(tmp_path / "weights.pth"),
        "progress_path": str(outside / "progress.json"),
        "started_prefix_count": 0,
    }
    request_path = request_dir / "worker-request.json"
    encoded = (json.dumps(request, indent=2) + "\n").encode("utf-8")
    request_path.write_bytes(encoded)
    monkeypatch.setattr(
        extractor_cli,
        "_extract_tiled_swin_rows",
        lambda **_kwargs: pytest.fail("model extraction occurred"),
    )

    with pytest.raises(ValueError, match="private directory"):
        extractor_cli._worker_main(
            request_path,
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_recoverable_attempt_publication_replays_only_with_external_receipts(
    tmp_path: Path,
) -> None:
    from test_vru_causal_tiled_swin_embeddings import _verified_plan

    plan, _artifact_sha256, _file_sha256, _payload = _verified_plan(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    work_stage = tmp_path / "work"
    work_stage.mkdir()
    (work_stage / "resource_guard.jsonl").write_bytes(b"")
    rows = [[[float(tile)] * 768 for tile in range(4)]]
    prior = extractor_cli._PriorAttemptState((), (), None, None, 0)

    attempt_receipt, resume_cas = extractor_cli._publish_recoverable_attempt(
        work_stage=work_stage,
        output_root=output_root,
        plan=plan,
        prior_state=prior,
        worker_summary={
            "active_runtime_nanoseconds": 1,
            "sample_count": 0,
            "log_bytes": 0,
            "process_tree_reaped": True,
        },
        completed_rows=rows,
        signal_name="SIGTERM",
    )
    replayed = extractor_cli._load_prior_attempt_state(
        output_root=output_root,
        plan=plan,
        raw_expected_attempt_receipts=[json.dumps(attempt_receipt)],
        raw_expected_resume_cas=json.dumps(resume_cas),
    )

    independently_replayed = module._replay_published_prior_attempts(
        output_root=output_root,
        attempt_chain=replayed.attempt_chain,
        plan=plan,
    )

    assert replayed.latest_resume["completed_count"] == 1
    assert replayed.latest_resume_cas == resume_cas
    assert independently_replayed["latest_resume_cas"] == resume_cas
    assert independently_replayed["completed_prefix_count"] == 1
    with pytest.raises(ValueError, match="external value"):
        extractor_cli._load_prior_attempt_state(
            output_root=output_root,
            plan=plan,
            raw_expected_attempt_receipts=[json.dumps(attempt_receipt)],
            raw_expected_resume_cas=json.dumps({**resume_cas, "inode": resume_cas["inode"] + 1}),
        )

    (output_root / "attempts" / "attempt-0001" / "resume.json").unlink()
    with pytest.raises(ValueError, match="prior-attempt"):
        module._replay_published_prior_attempts(
            output_root=output_root,
            attempt_chain=replayed.attempt_chain,
            plan=plan,
        )


def test_completed_generation_failure_removes_private_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_vru_causal_tiled_swin_embeddings import _verified_plan

    plan, _artifact_sha256, _file_sha256, _payload = _verified_plan(tmp_path)
    work_stage = tmp_path / "work"
    work_stage.mkdir()
    (work_stage / "resource_guard.jsonl").write_bytes(b"")
    final_path = tmp_path / "final_v1"
    monkeypatch.setattr(
        extractor_cli,
        "_build_fresh_tiled_swin_embeddings",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("fit failed")),
    )

    with pytest.raises(ValueError, match="fit failed"):
        extractor_cli._publish_completed_generation(
            work_stage=work_stage,
            final_path=final_path,
            verified_inputs=object(),
            plan=plan,
            prior_state=extractor_cli._PriorAttemptState((), (), None, None, 0),
            worker_summary={
                "active_runtime_nanoseconds": 1,
                "sample_count": 0,
                "log_bytes": 0,
                "process_tree_reaped": True,
            },
            progress={"tile_embeddings": []},
            publication_revalidator=lambda: None,
        )

    assert not final_path.exists()
    assert not list(tmp_path.glob(".task0258-final.*"))


def test_completed_generation_never_publishes_a_mechanical_failure_as_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_vru_causal_tiled_swin_embeddings import _verified_plan

    plan, _artifact_sha256, _file_sha256, _payload = _verified_plan(tmp_path)
    work_stage = tmp_path / "work"
    work_stage.mkdir()
    (work_stage / "resource_guard.jsonl").write_bytes(b"")
    final_path = tmp_path / "final_v1"
    monkeypatch.setattr(extractor_cli, "_build_fresh_tiled_swin_embeddings", lambda **_kwargs: object())
    monkeypatch.setattr(
        extractor_cli,
        "_build_vru_causal_temporal_final_generation",
        lambda **_kwargs: SimpleNamespace(
            mechanical_gate={"decision": "mechanical_failure"},
        ),
    )

    with pytest.raises(ValueError, match="mechanical failure"):
        extractor_cli._publish_completed_generation(
            work_stage=work_stage,
            final_path=final_path,
            verified_inputs=object(),
            plan=plan,
            prior_state=extractor_cli._PriorAttemptState((), (), None, None, 0),
            worker_summary={
                "active_runtime_nanoseconds": 1,
                "sample_count": 0,
                "log_bytes": 0,
                "process_tree_reaped": True,
            },
            progress={"tile_embeddings": []},
            publication_revalidator=lambda: None,
        )

    assert not final_path.exists()
    assert not list(tmp_path.glob(".task0258-final.*"))


def test_extractor_cli_requires_exact_batch_and_fixed_output_root() -> None:
    parser = extractor_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--batch-size", "2"])


def test_extractor_output_state_is_rejected_before_full_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "vru_causal_temporal_retrospective_v1"
    output_root.mkdir()
    (output_root / "temporal_feature_plan.json").write_text("{}\n", encoding="utf-8")
    (output_root / "final_v1").mkdir()
    monkeypatch.setattr(
        extractor_cli,
        "_verify_preflight",
        lambda _args: pytest.fail("full receipt replay occurred"),
    )
    monkeypatch.setattr(
        extractor_cli,
        "build_parser",
        lambda: type(
            "Parser",
            (),
            {
                "parse_args": staticmethod(
                    lambda _argv: type(
                        "Args",
                        (),
                        {"batch_size": 1, "output_root": output_root},
                    )()
                )
            },
        )(),
    )

    with pytest.raises(ValueError, match="output state"):
        extractor_cli.main([])


def test_extractor_replays_an_externally_receipted_existing_final_as_a_read_only_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "vru_causal_temporal_retrospective_v1"
    output_root.mkdir()
    (output_root / "temporal_feature_plan.json").write_text("{}\n", encoding="utf-8")
    final = output_root / "final_v1"
    terminal_attempt = final / "terminal_attempt"
    terminal_attempt.mkdir(parents=True)
    for relative in (
        "terminal_attempt/resource_guard.jsonl",
        "terminal_attempt/attempt_record.json",
        "tiled_swin_embeddings.json",
        "temporal_retrospective.json",
        "baseline_final_evaluator.json",
        "candidate_final_evaluator.json",
        "mechanical_gate.json",
    ):
        (final / relative).write_text("{}\n", encoding="utf-8")
    receipt_registry = tmp_path / "final-receipts.json"
    registry_bytes = b'{"externally_frozen":true}\n'
    receipt_registry.write_bytes(registry_bytes)
    fake_plan = SimpleNamespace(_payload={}, _artifact_sha256="a" * 64, _file_sha256="b" * 64)
    file_keys = (
        "terminal_attempt/resource_guard.jsonl",
        "terminal_attempt/attempt_record.json",
        "tiled_swin_embeddings.json",
        "temporal_retrospective.json",
        "baseline_final_evaluator.json",
        "candidate_final_evaluator.json",
        "mechanical_gate.json",
    )
    artifact_keys = tuple(key for key in file_keys if not key.endswith(".jsonl"))
    verified_registry = {
        "file_receipts": {key: "c" * 64 for key in file_keys},
        "artifact_receipts": {key: "d" * 64 for key in artifact_keys},
    }
    args = SimpleNamespace(
        batch_size=1,
        output_root=output_root,
        receipt_registry=receipt_registry,
        expected_receipt_registry_artifact_sha256="e" * 64,
        expected_receipt_registry_file_sha256=hashlib.sha256(registry_bytes).hexdigest(),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        extractor_cli,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda _argv: args),
    )
    monkeypatch.setattr(
        extractor_cli,
        "_verify_preflight",
        lambda _args: (object(), fake_plan, object()),
    )
    monkeypatch.setattr(
        extractor_cli,
        "_read_bounded_json",
        lambda *_args, **_kwargs: ({"externally_frozen": True}, registry_bytes),
    )
    monkeypatch.setattr(
        extractor_cli,
        "verify_module_a_final_receipt_registry",
        lambda *_args, **_kwargs: calls.append("registry") or verified_registry,
        raising=False,
    )
    monkeypatch.setattr(
        extractor_cli,
        "load_verified_tiled_swin_embeddings",
        lambda *_args, **_kwargs: calls.append("embeddings") or object(),
        raising=False,
    )
    monkeypatch.setattr(
        extractor_cli,
        "verify_vru_causal_temporal_final_generation",
        lambda *_args, **_kwargs: calls.append("final") or object(),
        raising=False,
    )
    monkeypatch.setattr(
        extractor_cli.tempfile,
        "TemporaryDirectory",
        lambda *_args, **_kwargs: pytest.fail("read-only no-op created staging"),
    )
    monkeypatch.setattr(
        extractor_cli.tempfile,
        "mkdtemp",
        lambda *_args, **_kwargs: pytest.fail("read-only no-op created staging"),
    )
    before = {
        str(path.relative_to(output_root)): ("dir" if path.is_dir() else path.read_bytes())
        for path in output_root.rglob("*")
    }

    assert extractor_cli.main([]) == 0

    after = {
        str(path.relative_to(output_root)): ("dir" if path.is_dir() else path.read_bytes())
        for path in output_root.rglob("*")
    }
    assert calls == ["registry", "embeddings", "final"]
    assert after == before

    for malformed_name in ("dual-terminal", "staging-residue"):
        malformed = tmp_path / malformed_name / "vru_causal_temporal_retrospective_v1"
        malformed.mkdir(parents=True)
        (malformed / "temporal_feature_plan.json").write_text("{}\n", encoding="utf-8")
        if malformed_name == "dual-terminal":
            (malformed / "final_v1").mkdir()
            (malformed / "terminal_failure_v1").mkdir()
        else:
            (malformed / ".task0258-final.stale").mkdir()
        with pytest.raises(ValueError, match="output state"):
            extractor_cli._preflight_output_state(malformed)

    missing_prior = tmp_path / "missing-prior" / "vru_causal_temporal_retrospective_v1"
    (missing_prior / "attempts" / "attempt-0002").mkdir(parents=True)
    with pytest.raises(ValueError, match="prior-attempt"):
        extractor_cli._load_prior_attempt_state(
            output_root=missing_prior,
            plan=fake_plan,
            raw_expected_attempt_receipts=(),
            raw_expected_resume_cas=None,
        )


def test_terminal_failure_is_canonical_schema_bound_and_has_an_independent_verifier(
    tmp_path: Path,
) -> None:
    from test_vru_causal_temporal_feature_plan import _verified_task_fixture
    from test_vru_causal_tiled_swin_embeddings import _verified_plan

    plan, _artifact_sha256, _file_sha256, _payload = _verified_plan(tmp_path)
    work_stage = tmp_path / "work"
    work_stage.mkdir()
    (work_stage / "resource_guard.jsonl").write_bytes(b"")
    terminal_path = tmp_path / "terminal_failure_v1"
    failure = extractor_cli._GuardedWorkerFailure("disk_failure", "disk receipt drift")
    failure.summary = {
        "active_runtime_nanoseconds": 1,
        "sample_count": 0,
        "log_bytes": 0,
        "process_tree_reaped": True,
    }

    extractor_cli._publish_terminal_failure(
        work_stage=work_stage,
        terminal_path=terminal_path,
        plan=plan,
        prior_state=extractor_cli._PriorAttemptState((), (), None, None, 0),
        failure=failure,
    )

    failure_path = terminal_path / "mechanical_failure.json"
    encoded = failure_path.read_bytes()
    payload = json.loads(encoded)
    expected_fields = {
        "schema_version",
        "module_id",
        "purpose",
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
        "input_receipts",
        "attempt_chain",
        "ordered_check_results",
        "error_bound_result",
        "resource_summary",
        "publication_summary",
        "decision",
        "stop_reason",
        "conditional_downstream",
        "artifact_sha256",
    }
    violations: list[str] = []
    canonical = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if encoded != canonical:
        violations.append("terminal failure is not compact sorted canonical JSON plus LF")
    if set(payload) != expected_fields:
        violations.append("terminal failure top-level schema drifted")
    receipt_slots = payload.get("input_receipts")
    if (
        not isinstance(receipt_slots, dict)
        or not receipt_slots
        or any(
            not isinstance(slot, dict)
            or set(slot) != {"provider", "verification_state", "receipt"}
            or slot["verification_state"] not in {"verified", "failed", "not_reached"}
            or (slot["receipt"] is None) != (slot["verification_state"] != "verified")
            for slot in receipt_slots.values()
        )
    ):
        violations.append("terminal input receipts are not exact provider/state/receipt slots")
    if not callable(getattr(module, "verify_vru_causal_temporal_terminal_failure", None)):
        violations.append("terminal failure has no independent public verifier")
    assert not violations, violations

    file_receipts = {
        str(path.relative_to(terminal_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in terminal_path.rglob("*")
        if path.is_file()
    }
    artifact_receipts = {
        key: json.loads((terminal_path / key).read_text(encoding="utf-8"))["artifact_sha256"]
        for key in file_receipts
        if not key.endswith(".jsonl")
    }
    inputs = _verified_task_fixture(tmp_path / "verified-inputs")
    replayed = module.verify_vru_causal_temporal_terminal_failure(
        generation_dir=terminal_path,
        expected_file_receipts=file_receipts,
        expected_artifact_receipts=artifact_receipts,
        verified_inputs=inputs,
        plan=plan,
    )
    assert replayed["decision"] == "mechanical_failure"


@pytest.mark.parametrize("drift", ("input receipt drift", "disk reserve drift"))
def test_terminal_failure_revalidates_under_lock_and_stale_state_cannot_publish(
    tmp_path: Path,
    drift: str,
) -> None:
    from test_vru_causal_tiled_swin_embeddings import _verified_plan

    plan, _artifact_sha256, _file_sha256, _payload = _verified_plan(tmp_path)
    output_root = tmp_path / "vru_causal_temporal_retrospective_v1"
    output_root.mkdir()
    work_stage = tmp_path / "work"
    work_stage.mkdir()
    (work_stage / "resource_guard.jsonl").write_bytes(b"")
    terminal_path = output_root / "terminal_failure_v1"
    failure = extractor_cli._GuardedWorkerFailure("receipt_failure", drift)
    failure.summary = {
        "active_runtime_nanoseconds": 1,
        "sample_count": 0,
        "log_bytes": 0,
        "process_tree_reaped": True,
    }

    def reject_stale_publication() -> None:
        with pytest.raises(ValueError, match="already running"):
            with extractor_cli._exclusive_directory_lock(output_root):
                pass
        raise ValueError(drift)

    with extractor_cli._exclusive_directory_lock(output_root):
        with pytest.raises(ValueError, match=drift):
            extractor_cli._publish_terminal_failure(
                work_stage=work_stage,
                terminal_path=terminal_path,
                plan=plan,
                prior_state=extractor_cli._PriorAttemptState((), (), None, None, 0),
                failure=failure,
                publication_revalidator=reject_stale_publication,
            )

    assert not terminal_path.exists()
    assert not list(output_root.glob(".task0258-terminal.*"))


def test_finalization_disk_refusal_has_the_exact_terminal_reason() -> None:
    assert (
        extractor_cli._classify_finalization_failure(
            module.DiskWriteBudgetError("reserve crossed"),
        )
        == "disk_failure"
    )


def test_screen_cli_requires_both_external_receipts_for_every_control_artifact() -> None:
    actions = {action.dest: action.required for action in screen_cli.build_parser()._actions}
    assert actions["expected_plan_artifact_sha256"] is True
    assert actions["expected_plan_file_sha256"] is True
    assert actions["expected_receipt_registry_artifact_sha256"] is True
    assert actions["expected_receipt_registry_file_sha256"] is True


def test_extractor_guard_reaps_successful_and_failed_worker_processes(tmp_path: Path) -> None:
    success_log = tmp_path / "success.jsonl"
    success_log.write_bytes(b"")
    summary = extractor_cli._run_guarded_worker(
        command=(sys.executable, "-c", "pass"),
        log_path=success_log,
    )
    assert summary["process_tree_reaped"] is True

    failure_log = tmp_path / "failure.jsonl"
    failure_log.write_bytes(b"")
    with pytest.raises(extractor_cli._GuardedWorkerFailure) as captured:
        extractor_cli._run_guarded_worker(
            command=(sys.executable, "-c", "raise SystemExit(4)"),
            log_path=failure_log,
        )
    assert captured.value.stop_reason == "model_failure"
    assert captured.value.summary["sample_count"] == 0


def test_extractor_guard_rejects_a_symlink_log_without_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(b"preserve")
    log_path = tmp_path / "resource.jsonl"
    log_path.symlink_to(target)
    monkeypatch.setattr(
        extractor_cli.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker spawned"),
    )

    with pytest.raises(extractor_cli._GuardedWorkerFailure) as captured:
        extractor_cli._run_guarded_worker(
            command=(sys.executable, "-c", "pass"),
            log_path=log_path,
        )

    assert captured.value.stop_reason == "publication_failure"
    assert target.read_bytes() == b"preserve"


def test_extractor_guard_converts_sigterm_into_a_reaped_interruption(tmp_path: Path) -> None:
    script = f"""
import json
import os
import signal
import sys
import threading
from pathlib import Path
from scripts import extract_vru_causal_tiled_swin_embeddings as cli

log = Path({str(tmp_path / "signal.jsonl")!r})
log.write_bytes(b'')
threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
try:
    cli._run_guarded_worker(
        command=(sys.executable, '-c', 'import time; time.sleep(30)'),
        log_path=log,
    )
except cli._GuardedWorkerInterrupted as error:
    print(json.dumps({{'signal': error.signal_name, **error.summary}}))
    raise SystemExit(0)
raise SystemExit(9)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["signal"] == "SIGTERM"
    assert observed["process_tree_reaped"] is True
