from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.analysis import vru_causal_temporal_retrospective as temporal_module
from app.analysis.vru_causal_temporal_retrospective import (
    FileReceipt,
    StoredArtifactReceipt,
    Task0257ExpectedReceipts,
    Task0257InputPaths,
    VerifiedTask0257TemporalInputs,
    VerifiedTemporalFeaturePlan,
    build_temporal_feature_examples,
    combine_tiled_swin_embeddings,
    load_verified_temporal_feature_plan,
    seal_module_a_spec_approval,
    verify_module_a_spec_approval,
    verify_task0257_temporal_inputs,
)

SPEC_FILENAMES = ("requirement.md", "solution.md", "gate-review.md")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approval_record(
    tmp_path: Path,
) -> tuple[Path, tuple[Path, ...], str, str, str]:
    spec_paths = tuple(tmp_path / filename for filename in SPEC_FILENAMES)
    for index, path in enumerate(spec_paths):
        path.write_text(f"spec-{index}\n", encoding="utf-8")
    review_internal = "a" * 64
    review_file = "b" * 64
    statement = "批准 TASK-0258 Module A 规格：" + "；".join(f"{path.stem}={_file_sha256(path)}" for path in spec_paths)
    payload = seal_module_a_spec_approval(
        approved_spec_paths=spec_paths,
        fresh_review_internal_sha256=review_internal,
        fresh_review_file_sha256=review_file,
        approval_statement=statement,
        approved_at_utc="2026-08-16T12:00:00Z",
    )
    approval_path = tmp_path / "module_a_spec_approval.json"
    approval_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return approval_path, spec_paths, review_internal, review_file, statement


def test_module_a_spec_approval_requires_external_receipts_and_current_bytes(
    tmp_path: Path,
) -> None:
    approval_path, spec_paths, review_internal, review_file, statement = _approval_record(tmp_path)
    expected_file_sha256 = _file_sha256(approval_path)
    payload = json.loads(approval_path.read_text(encoding="utf-8"))

    verified = verify_module_a_spec_approval(
        approval_path=approval_path,
        expected_artifact_sha256=payload["artifact_sha256"],
        expected_file_sha256=expected_file_sha256,
        approved_spec_paths=spec_paths,
        expected_fresh_review_internal_sha256=review_internal,
        expected_fresh_review_file_sha256=review_file,
        expected_approval_statement_sha256=hashlib.sha256(statement.encode("utf-8")).hexdigest(),
    )
    assert verified["artifact_sha256"] == payload["artifact_sha256"]

    with pytest.raises(ValueError, match="external approval artifact"):
        verify_module_a_spec_approval(
            approval_path=approval_path,
            expected_artifact_sha256="c" * 64,
            expected_file_sha256=expected_file_sha256,
            approved_spec_paths=spec_paths,
            expected_fresh_review_internal_sha256=review_internal,
            expected_fresh_review_file_sha256=review_file,
            expected_approval_statement_sha256=hashlib.sha256(statement.encode("utf-8")).hexdigest(),
        )

    spec_paths[0].write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="approved specification"):
        verify_module_a_spec_approval(
            approval_path=approval_path,
            expected_artifact_sha256=payload["artifact_sha256"],
            expected_file_sha256=expected_file_sha256,
            approved_spec_paths=spec_paths,
            expected_fresh_review_internal_sha256=review_internal,
            expected_fresh_review_file_sha256=review_file,
            expected_approval_statement_sha256=hashlib.sha256(statement.encode("utf-8")).hexdigest(),
        )


def test_combine_tiled_swin_embeddings_uses_frozen_mean_delta_formula() -> None:
    tiles = [[float(tile + offset) for offset in range(768)] for tile in range(4)]
    combined = combine_tiled_swin_embeddings(tiles)

    assert len(combined) == 1536
    assert combined[0] == pytest.approx(1.5)
    assert combined[767] == pytest.approx(768.5)
    assert combined[768] == pytest.approx(2.0)
    assert combined[-1] == pytest.approx(2.0)


def test_temporal_examples_slice_authoritative_review_indexes_and_ignore_labels() -> None:
    source_sha = "1" * 64
    row = {
        "source_video_sha256": source_sha,
        "candidate_bundle_sha256": "2" * 64,
        "event_id": "closure-0001",
        "event_present": True,
        "start_frame": 999,
        "end_frame": 1000,
    }
    review_plan = {
        "artifact_sha256": "3" * 64,
        "examples": [
            {
                "review_id": "closure-0001",
                "source_video_sha256": source_sha,
                "frame_indexes": list(range(100, 164)),
            }
        ],
    }
    examples = build_temporal_feature_examples(
        ordered_rows=[row],
        parent_review_plan=review_plan,
        harwood_review_plan={"artifact_sha256": "4" * 64, "examples": []},
    )

    assert examples[0]["frame_indexes"] == list(range(100, 164))
    assert examples[0]["tile_frame_indexes"] == [
        list(range(100, 116)),
        list(range(116, 132)),
        list(range(132, 148)),
        list(range(148, 164)),
    ]
    assert "event_present" not in examples[0]
    assert "start_frame" not in examples[0]
    assert "end_frame" not in examples[0]

    changed = dict(row, event_present=False, start_frame=1, end_frame=9999)
    assert (
        build_temporal_feature_examples(
            ordered_rows=[changed],
            parent_review_plan=review_plan,
            harwood_review_plan={"artifact_sha256": "4" * 64, "examples": []},
        )
        == examples
    )


def _sealed_feature_plan() -> dict[str, object]:
    game_rows = (("hazen", "hctv", 8), ("randolph", "hctv", 7), ("vtv", "vtv", 7), ("harwood", "hctv", 23))
    examples: list[dict[str, object]] = []
    ordinal = 0
    for game_id, production_family, row_count in game_rows:
        for game_ordinal in range(row_count):
            examples.append(
                {
                    "ordinal": ordinal,
                    "game_id": game_id,
                    "production_family": production_family,
                    "source_video_sha256": f"{ordinal + 1:064x}",
                    "candidate_bundle_sha256": f"{ordinal + 101:064x}",
                    "event_id": f"{game_id}-{game_ordinal:04d}",
                    "review_plan_receipt": "4" * 64,
                    "tile_frame_indexes": [
                        list(range(ordinal * 64 + offset, ordinal * 64 + offset + 16)) for offset in range(0, 64, 16)
                    ],
                }
            )
            ordinal += 1

    def plain_receipt(index: int) -> dict[str, object]:
        return {
            "file_sha256": f"{index:064x}",
            "filename": f"file-{index}.bin",
            "size_bytes": index,
        }

    def stored_receipt(index: int) -> dict[str, object]:
        return {
            "schema_version": "fixture.v1",
            "internal_sha256_field": "artifact_sha256",
            "internal_sha256": f"{index + 10_000:064x}",
            **plain_receipt(index),
        }

    task0257_receipts: dict[str, object] = {}
    index = 1
    stored_sequences = {
        "parent_candidate_children": 3,
        "old_embedding_files": 2,
    }
    plain_sequences = {
        "parent_label_children": 3,
        "parent_review_jpegs": 1536,
        "harwood_review_jpegs": 1536,
        "source_videos": 4,
        "checkpoints": 2,
    }
    stored_scalars = {
        "parent_selection",
        "parent_review_plan",
        "parent_raw_frame_manifest",
        "parent_sealed_review",
        "parent_v1_export",
        "v2_export",
        "v2_candidate_child",
        "source_groups",
        "four_video_training_manifest",
        "old_nested_probe_plan",
        "old_nested_probe",
        "harwood_selection",
        "harwood_review_plan",
        "harwood_raw_frame_manifest",
        "harwood_sealed_review",
    }
    plain_scalars = {"parent_source_manifest", "v2_label_child", "harwood_source_manifest"}
    for field in Task0257ExpectedReceipts.__dataclass_fields__:
        if field in stored_sequences:
            count = stored_sequences[field]
            value = [stored_receipt(index + offset) for offset in range(count)]
            index += count
        elif field in plain_sequences:
            count = plain_sequences[field]
            value = [plain_receipt(index + offset) for offset in range(count)]
            index += count
        elif field in stored_scalars:
            value = stored_receipt(index)
            index += 1
        elif field in plain_scalars:
            value = plain_receipt(index)
            index += 1
        else:  # pragma: no cover - closed fixture mirrors the public dataclass
            raise AssertionError(field)
        task0257_receipts[field] = value
    task0257_receipts["checkpoints"][1]["file_sha256"] = temporal_module.TEMPORAL_REPRESENTATION_CONTRACT[
        "checkpoint_sha256"
    ]
    source_sha256s = [row["file_sha256"] for row in task0257_receipts["source_videos"]]
    game_bindings = {
        "hazen": (source_sha256s[0], "a" * 64),
        "randolph": (source_sha256s[1], "b" * 64),
        "vtv": (source_sha256s[2], "c" * 64),
        "harwood": (source_sha256s[3], "d" * 64),
    }
    for row in examples:
        row["source_video_sha256"], row["candidate_bundle_sha256"] = game_bindings[row["game_id"]]
        row["review_plan_receipt"] = task0257_receipts[
            "harwood_review_plan" if row["game_id"] == "harwood" else "parent_review_plan"
        ]["internal_sha256"]

    payload: dict[str, object] = {
        "schema_version": "agu.vru-causal-temporal-feature-plan.v1",
        "module_id": "existing-45-temporal-retrospective",
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "label_hidden": True,
        "representation": deepcopy(temporal_module.TEMPORAL_REPRESENTATION_CONTRACT),
        "task0257_receipts": task0257_receipts,
        "ordered_examples": examples,
        "evaluation_protocol": deepcopy(temporal_module.TEMPORAL_EVALUATION_PROTOCOL),
        "resource_policy": deepcopy(temporal_module.TEMPORAL_RESOURCE_POLICY),
        "environment_contract": deepcopy(temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def test_plan_schema_is_exactly_45_rows_and_rejects_contract_drift(tmp_path: Path) -> None:
    payload = _sealed_feature_plan()
    mutations = (
        lambda value: value["ordered_examples"].pop(),
        lambda value: value["representation"].update({"window_seconds": 7.0}),
        lambda value: value["evaluation_protocol"].update({"pca_enabled": True}),
        lambda value: value["resource_policy"].update({"batch_size": 2}),
        lambda value: value["environment_contract"].update({"device": "cpu"}),
        lambda value: value["ordered_examples"][0].update({"event_present": True}),
    )
    for index, mutate in enumerate(mutations):
        changed = deepcopy(payload)
        mutate(changed)
        changed["artifact_sha256"] = temporal_module._canonical_sha256(changed)
        encoded = (json.dumps(changed, indent=2) + "\n").encode("utf-8")
        plan_path = tmp_path / f"drift-{index}.json"
        plan_path.write_bytes(encoded)
        with pytest.raises(ValueError):
            load_verified_temporal_feature_plan(
                plan_path=plan_path,
                expected_artifact_sha256=changed["artifact_sha256"],
                expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
            )


def test_plan_sealer_requires_a_verified_task0257_capability() -> None:
    with pytest.raises(TypeError, match="verified TASK-0257"):
        temporal_module.seal_vru_causal_temporal_feature_plan(
            inputs={"ordered_rows": []},
            environment_contract=temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT,
        )


def _receipt_contract_from_mapping(mapping: dict[str, object]) -> Task0257ExpectedReceipts:
    stored_sequences = {"parent_candidate_children", "old_embedding_files"}
    file_sequences = {
        "parent_label_children",
        "parent_review_jpegs",
        "harwood_review_jpegs",
        "source_videos",
        "checkpoints",
    }
    file_scalars = {"parent_source_manifest", "v2_label_child", "harwood_source_manifest"}
    values: dict[str, object] = {}
    for field in Task0257ExpectedReceipts.__dataclass_fields__:
        value = mapping[field]
        if field in stored_sequences:
            values[field] = tuple(StoredArtifactReceipt(**row) for row in value)
        elif field in file_sequences:
            values[field] = tuple(FileReceipt(**row) for row in value)
        elif field in file_scalars:
            values[field] = FileReceipt(**value)
        else:
            values[field] = StoredArtifactReceipt(**value)
    return Task0257ExpectedReceipts(**values)


def _verified_task_fixture(
    tmp_path: Path,
    *,
    register: bool = True,
) -> VerifiedTask0257TemporalInputs:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = _sealed_feature_plan()
    leaf = tmp_path / "verified-leaf.bin"
    leaf.write_bytes(b"verified")
    metadata = leaf.stat()
    capability = VerifiedTask0257TemporalInputs(temporal_module._VERIFIED_TASK0257_TOKEN)
    capability._token = temporal_module._VERIFIED_TASK0257_TOKEN
    capability.expected_receipts = _receipt_contract_from_mapping(plan["task0257_receipts"])
    capability.validated_input_paths = (leaf,)
    capability.validated_leaf_identities = ((metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns),)
    capability.validated_leaf_receipts = (
        FileReceipt(
            file_sha256=_file_sha256(leaf),
            filename=leaf.name,
            size_bytes=metadata.st_size,
        ),
    )
    capability.ordered_rows = tuple(
        {
            "source_video_sha256": row["source_video_sha256"],
            "candidate_bundle_sha256": row["candidate_bundle_sha256"],
            "event_id": row["event_id"],
            "event_present": row["ordinal"] % 3 == 0,
        }
        for row in plan["ordered_examples"]
    )
    review_examples = {
        "parent": [],
        "harwood": [],
    }
    for row in plan["ordered_examples"]:
        review_examples["harwood" if row["game_id"] == "harwood" else "parent"].append(
            {
                "review_id": row["event_id"],
                "source_video_sha256": row["source_video_sha256"],
                "frame_indexes": [value for tile in row["tile_frame_indexes"] for value in tile],
            }
        )
    capability.parent_review_plan = {
        "artifact_sha256": plan["task0257_receipts"]["parent_review_plan"]["internal_sha256"],
        "examples": review_examples["parent"],
    }
    capability.harwood_review_plan = {
        "artifact_sha256": plan["task0257_receipts"]["harwood_review_plan"]["internal_sha256"],
        "examples": review_examples["harwood"],
    }
    capability.source_groups = {
        "games": [
            {
                "game_id": game_id,
                "source_video_sha256": next(
                    row["source_video_sha256"] for row in plan["ordered_examples"] if row["game_id"] == game_id
                ),
                "production_family": "vtv" if game_id == "vtv" else "hctv",
            }
            for game_id in ("hazen", "randolph", "vtv", "harwood")
        ]
    }
    capability.training_chain = None
    capability.old_embeddings = ()
    capability.replayed_nested_probe = {}
    if register:
        temporal_module._register_task0257_capability(capability)
    return capability


def test_plan_sealer_is_label_invariant_and_self_verifying(tmp_path: Path) -> None:
    inputs = _verified_task_fixture(tmp_path)
    first = temporal_module.seal_vru_causal_temporal_feature_plan(
        inputs=inputs,
        environment_contract=temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT,
    )
    changed_inputs = _verified_task_fixture(tmp_path / "changed", register=False)
    changed_inputs.ordered_rows = tuple(
        {**row, "event_present": not row["event_present"]} for row in changed_inputs.ordered_rows
    )
    temporal_module._register_task0257_capability(changed_inputs)
    second = temporal_module.seal_vru_causal_temporal_feature_plan(
        inputs=changed_inputs,
        environment_contract=temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT,
    )

    assert first == second
    assert len(first["ordered_examples"]) == 45
    assert all("event_present" not in row for row in first["ordered_examples"])


def test_plan_loader_requires_external_double_receipt_and_returns_opaque_token(
    tmp_path: Path,
) -> None:
    payload = _sealed_feature_plan()
    plan_path = tmp_path / "temporal_feature_plan.json"
    encoded = temporal_module._canonical_json_bytes(payload)
    plan_path.write_bytes(encoded)

    verified = load_verified_temporal_feature_plan(
        plan_path=plan_path,
        expected_artifact_sha256=payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    assert type(verified) is VerifiedTemporalFeaturePlan
    with pytest.raises(TypeError):
        VerifiedTemporalFeaturePlan()
    with pytest.raises(ValueError, match="file SHA-256"):
        load_verified_temporal_feature_plan(
            plan_path=plan_path,
            expected_artifact_sha256=payload["artifact_sha256"],
            expected_file_sha256="f" * 64,
        )


def test_verified_plan_rejects_mutated_local_snapshot_while_external_bytes_are_unchanged(
    tmp_path: Path,
) -> None:
    payload = _sealed_feature_plan()
    plan_path = tmp_path / "temporal_feature_plan.json"
    encoded = temporal_module._canonical_json_bytes(payload)
    plan_path.write_bytes(encoded)
    plan = load_verified_temporal_feature_plan(
        plan_path=plan_path,
        expected_artifact_sha256=payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
    )

    payload["ordered_examples"][0]["tile_frame_indexes"][0][0] += 100
    assert (
        plan._payload["ordered_examples"][0]["tile_frame_indexes"][0][0]
        != payload["ordered_examples"][0]["tile_frame_indexes"][0][0]
    )

    plan._payload["ordered_examples"][0]["tile_frame_indexes"][0][0] += 1
    tiles = [[[0.0] * 768 for _tile in range(4)] for _row in range(45)]
    attempt_chain = [
        {
            "attempt_ordinal": 1,
            "attempt_record": {
                "schema_version": "agu.vru-causal-tiled-swin-attempt.v1",
                "internal_sha256_field": "artifact_sha256",
                "internal_sha256": "1" * 64,
                "file_sha256": "2" * 64,
                "filename": "attempt_record.json",
                "size_bytes": 1,
            },
            "resource_log": {
                "file_sha256": "3" * 64,
                "filename": "resource_guard.jsonl",
                "size_bytes": 0,
            },
            "resume_output_cas": None,
        }
    ]

    with pytest.raises(ValueError, match="verified temporal feature plan"):
        temporal_module._build_fresh_tiled_swin_embeddings(
            plan=plan,
            tile_embeddings=tiles,
            attempt_chain=attempt_chain,
        )


def test_module_token_cannot_forge_a_task0257_capability(tmp_path: Path) -> None:
    source = _verified_task_fixture(tmp_path)
    forged = VerifiedTask0257TemporalInputs(temporal_module._VERIFIED_TASK0257_TOKEN)
    for field in VerifiedTask0257TemporalInputs.__slots__:
        if hasattr(source, field):
            setattr(forged, field, getattr(source, field))

    with pytest.raises(TypeError, match="verified TASK-0257"):
        temporal_module.seal_vru_causal_temporal_feature_plan(
            inputs=forged,
            environment_contract=temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT,
        )


def test_plan_loader_rejects_valid_but_noncanonical_json_bytes(tmp_path: Path) -> None:
    payload = _sealed_feature_plan()
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path = tmp_path / "noncanonical-plan.json"
    path.write_bytes(encoded)

    with pytest.raises(ValueError, match="canonical"):
        load_verified_temporal_feature_plan(
            plan_path=path,
            expected_artifact_sha256=payload["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_plan_loader_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path) -> None:
    for encoded in (
        b'{"schema_version":"x","schema_version":"y"}\n',
        b'{"schema_version":"x","value":NaN}\n',
    ):
        plan_path = tmp_path / f"plan-{hashlib.sha256(encoded).hexdigest()}.json"
        plan_path.write_bytes(encoded)
        with pytest.raises(ValueError):
            load_verified_temporal_feature_plan(
                plan_path=plan_path,
                expected_artifact_sha256="a" * 64,
                expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
            )


def test_plan_loader_rejects_symlink_input(tmp_path: Path) -> None:
    payload = _sealed_feature_plan()
    real_path = tmp_path / "real-plan.json"
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    real_path.write_bytes(encoded)
    symlink_path = tmp_path / "linked-plan.json"
    symlink_path.symlink_to(real_path)

    with pytest.raises(ValueError, match="symlink"):
        load_verified_temporal_feature_plan(
            plan_path=symlink_path,
            expected_artifact_sha256=payload["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_task0257_input_paths_reject_aliases_before_any_file_read(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    stored = StoredArtifactReceipt(
        schema_version="x",
        internal_sha256_field="artifact_sha256",
        internal_sha256="a" * 64,
        file_sha256="b" * 64,
        filename=missing.name,
        size_bytes=1,
    )
    plain = FileReceipt(file_sha256="b" * 64, filename=missing.name, size_bytes=1)
    paths = Task0257InputPaths(
        parent_selection=missing,
        parent_source_manifest=missing,
        parent_review_plan=missing,
        parent_raw_frame_manifest=missing,
        parent_review_jpegs=(missing,),
        parent_sealed_review=missing,
        parent_v1_export=missing,
        parent_candidate_children=(missing, missing, missing),
        parent_label_children=(missing, missing, missing),
        v2_export=missing,
        v2_candidate_child=missing,
        v2_label_child=missing,
        source_groups=missing,
        four_video_training_manifest=missing,
        old_embedding_files=(missing, missing),
        old_nested_probe_plan=missing,
        old_nested_probe=missing,
        harwood_source_manifest=missing,
        harwood_selection=missing,
        harwood_review_plan=missing,
        harwood_raw_frame_manifest=missing,
        harwood_review_jpegs=(missing,),
        harwood_sealed_review=missing,
        source_videos=(missing, missing, missing, missing),
        checkpoints=(missing, missing),
    )
    receipts = Task0257ExpectedReceipts(
        parent_selection=stored,
        parent_source_manifest=plain,
        parent_review_plan=stored,
        parent_raw_frame_manifest=stored,
        parent_review_jpegs=(plain,),
        parent_sealed_review=stored,
        parent_v1_export=stored,
        parent_candidate_children=(stored, stored, stored),
        parent_label_children=(plain, plain, plain),
        v2_export=stored,
        v2_candidate_child=stored,
        v2_label_child=plain,
        source_groups=stored,
        four_video_training_manifest=stored,
        old_embedding_files=(stored, stored),
        old_nested_probe_plan=stored,
        old_nested_probe=stored,
        harwood_source_manifest=plain,
        harwood_selection=stored,
        harwood_review_plan=stored,
        harwood_raw_frame_manifest=stored,
        harwood_review_jpegs=(plain,),
        harwood_sealed_review=stored,
        source_videos=(plain, plain, plain, plain),
        checkpoints=(plain, plain),
    )

    with pytest.raises(ValueError, match="unique leaf"):
        verify_task0257_temporal_inputs(paths=paths, expected_receipts=receipts)


def test_verified_task0257_inputs_cannot_be_forged() -> None:
    with pytest.raises(TypeError):
        VerifiedTask0257TemporalInputs(
            training_chain=object(),
            ordered_rows=(),
            parent_review_plan={},
            harwood_review_plan={},
            source_groups={},
            old_embeddings=({}, {}),
            replayed_nested_probe={},
            validated_input_paths=(),
        )


def test_leaf_receipt_rejects_self_resealed_drift_against_external_file_hash(
    tmp_path: Path,
) -> None:
    payload = {"schema_version": "example.v1", "value": 1}
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "artifact.json"
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    receipt = StoredArtifactReceipt(
        schema_version="example.v1",
        internal_sha256_field="artifact_sha256",
        internal_sha256=payload["artifact_sha256"],
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        filename=path.name,
        size_bytes=len(encoded),
    )

    verified_payload, _identity = temporal_module._verify_leaf_receipt(path, receipt)
    assert verified_payload == payload

    payload["value"] = 2
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "artifact_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file SHA-256"):
        temporal_module._verify_leaf_receipt(path, receipt)


@pytest.mark.parametrize(
    "tiles",
    (
        [[0.0] * 768] * 3,
        [[0.0] * 767] * 4,
        [[0.0] * 768, [0.0] * 768, [0.0] * 768, [float("nan")] * 768],
    ),
)
def test_combine_tiled_swin_embeddings_rejects_wrong_or_nonfinite_shape(
    tiles: list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        combine_tiled_swin_embeddings(tiles)
