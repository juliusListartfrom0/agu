from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_vru_causal_temporal_feature_plan import _sealed_feature_plan

from app.analysis import vru_causal_temporal_retrospective as temporal_module
from app.analysis.vru_causal_temporal_retrospective import (
    FreshlyProducedTiledSwinEmbeddings,
    VerifiedTiledSwinEmbeddings,
    load_verified_temporal_feature_plan,
    load_verified_tiled_swin_embeddings,
)


def _canonical_sha256(payload: dict[str, object]) -> str:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> tuple[str, str]:
    payload["artifact_sha256"] = _canonical_sha256(payload)
    encoded = temporal_module._canonical_json_bytes(payload)
    path.write_bytes(encoded)
    return payload["artifact_sha256"], hashlib.sha256(encoded).hexdigest()


def _verified_plan(tmp_path: Path):
    payload = _sealed_feature_plan()
    path = tmp_path / "plan.json"
    artifact_sha256, file_sha256 = _write_json(path, payload)
    return (
        load_verified_temporal_feature_plan(
            plan_path=path,
            expected_artifact_sha256=artifact_sha256,
            expected_file_sha256=file_sha256,
        ),
        artifact_sha256,
        file_sha256,
        payload,
    )


def _embedding_payload(plan_payload: dict[str, object]) -> dict[str, object]:
    tiles = [[float(tile + offset) for offset in range(768)] for tile in range(4)]
    overall = [float(offset + 1.5) for offset in range(768)]
    delta = [2.0] * 768
    return {
        "schema_version": "agu.vru-causal-tiled-swin-embeddings.v1",
        "module_id": "existing-45-temporal-retrospective",
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "plan_receipt": {"artifact_sha256": "a" * 64, "file_sha256": "b" * 64},
        "task0257_input_receipts": {"source_videos": ["2" * 64]},
        "representation": copy.deepcopy(plan_payload["representation"]),
        "producer_environment": {"device": "mps:0"},
        "attempt_chain": [
            {
                "attempt_ordinal": 1,
                "attempt_record": {
                    "schema_version": "agu.vru-causal-tiled-swin-attempt.v1",
                    "internal_sha256_field": "artifact_sha256",
                    "internal_sha256": "d" * 64,
                    "file_sha256": "e" * 64,
                    "filename": "attempt.json",
                    "size_bytes": 100,
                },
                "resource_log": {
                    "file_sha256": "f" * 64,
                    "filename": "resource.jsonl",
                    "size_bytes": 1,
                },
                "resume_output_cas": None,
            }
        ],
        "row_count": 45,
        "examples": [
            {
                "ordinal": row["ordinal"],
                "key": {
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                },
                "tile_frame_indexes": copy.deepcopy(row["tile_frame_indexes"]),
                "derivation_only_tile_embeddings": tiles,
                "model_input": overall + delta,
            }
            for row in plan_payload["ordered_examples"]
        ],
    }


def test_embedding_loader_recomputes_formula_and_returns_opaque_token(
    tmp_path: Path,
) -> None:
    plan, plan_artifact_sha256, plan_file_sha256, plan_payload = _verified_plan(tmp_path)
    payload = _embedding_payload(plan_payload)
    payload["plan_receipt"] = {
        "artifact_sha256": plan_artifact_sha256,
        "file_sha256": plan_file_sha256,
    }
    payload["task0257_input_receipts"] = copy.deepcopy(plan_payload["task0257_receipts"])
    payload["producer_environment"] = copy.deepcopy(plan_payload["environment_contract"])
    path = tmp_path / "embeddings.json"
    artifact_sha256, file_sha256 = _write_json(path, payload)

    verified = load_verified_tiled_swin_embeddings(
        embeddings_path=path,
        expected_artifact_sha256=artifact_sha256,
        expected_file_sha256=file_sha256,
        plan=plan,
    )
    assert type(verified) is VerifiedTiledSwinEmbeddings
    with pytest.raises(TypeError):
        VerifiedTiledSwinEmbeddings()


def test_embedding_loader_rejects_resealed_wrong_model_input(tmp_path: Path) -> None:
    plan, plan_artifact_sha256, plan_file_sha256, plan_payload = _verified_plan(tmp_path)
    payload = _embedding_payload(plan_payload)
    payload["plan_receipt"] = {
        "artifact_sha256": plan_artifact_sha256,
        "file_sha256": plan_file_sha256,
    }
    payload["task0257_input_receipts"] = copy.deepcopy(plan_payload["task0257_receipts"])
    payload["producer_environment"] = copy.deepcopy(plan_payload["environment_contract"])
    payload["examples"][0]["model_input"][0] += 1.0
    path = tmp_path / "embeddings.json"
    artifact_sha256, file_sha256 = _write_json(path, payload)

    with pytest.raises(ValueError, match="model input formula"):
        load_verified_tiled_swin_embeddings(
            embeddings_path=path,
            expected_artifact_sha256=artifact_sha256,
            expected_file_sha256=file_sha256,
            plan=plan,
        )


def test_embedding_loader_rejects_valid_but_noncanonical_json_bytes(tmp_path: Path) -> None:
    plan, plan_artifact_sha256, plan_file_sha256, plan_payload = _verified_plan(tmp_path)
    payload = _embedding_payload(plan_payload)
    payload["plan_receipt"] = {
        "artifact_sha256": plan_artifact_sha256,
        "file_sha256": plan_file_sha256,
    }
    payload["task0257_input_receipts"] = copy.deepcopy(plan_payload["task0257_receipts"])
    payload["producer_environment"] = copy.deepcopy(plan_payload["environment_contract"])
    payload["artifact_sha256"] = _canonical_sha256(payload)
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path = tmp_path / "noncanonical-embeddings.json"
    path.write_bytes(encoded)

    with pytest.raises(ValueError, match="canonical"):
        load_verified_tiled_swin_embeddings(
            embeddings_path=path,
            expected_artifact_sha256=payload["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
            plan=plan,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.update({"task0257_input_receipts": {"forged": True}}),
        lambda payload: payload.update({"producer_environment": {"device": "cpu"}}),
        lambda payload: payload["examples"].pop(),
        lambda payload: payload["examples"][0]["derivation_only_tile_embeddings"][0].__setitem__(0, True),
        lambda payload: payload["examples"][0]["model_input"].__setitem__(0, "1.5"),
    ),
)
def test_embedding_loader_rejects_contract_and_numeric_drift(
    tmp_path: Path,
    mutate,
) -> None:
    plan, plan_artifact_sha256, plan_file_sha256, plan_payload = _verified_plan(tmp_path)
    payload = _embedding_payload(plan_payload)
    payload["plan_receipt"] = {
        "artifact_sha256": plan_artifact_sha256,
        "file_sha256": plan_file_sha256,
    }
    payload["task0257_input_receipts"] = copy.deepcopy(plan_payload["task0257_receipts"])
    payload["producer_environment"] = copy.deepcopy(plan_payload["environment_contract"])
    mutate(payload)
    path = tmp_path / "drifted-embeddings.json"
    artifact_sha256, file_sha256 = _write_json(path, payload)

    with pytest.raises(ValueError):
        load_verified_tiled_swin_embeddings(
            embeddings_path=path,
            expected_artifact_sha256=artifact_sha256,
            expected_file_sha256=file_sha256,
            plan=plan,
        )


def test_private_fresh_embedding_capability_requires_exact_45_by_4_by_768(
    tmp_path: Path,
) -> None:
    plan, _artifact_sha256, _file_sha256, _payload = _verified_plan(tmp_path)
    tiles = [[[float(tile)] * 768 for tile in range(4)] for _ in range(45)]

    fresh = temporal_module._build_fresh_tiled_swin_embeddings(
        plan=plan,
        tile_embeddings=tiles,
        attempt_chain=_embedding_payload(_payload)["attempt_chain"],
    )

    assert type(fresh) is FreshlyProducedTiledSwinEmbeddings
    assert fresh._payload["row_count"] == 45
    with pytest.raises(TypeError):
        FreshlyProducedTiledSwinEmbeddings()

    tiles[0][0][0] = True
    with pytest.raises(ValueError, match="exact finite float"):
        temporal_module._build_fresh_tiled_swin_embeddings(
            plan=plan,
            tile_embeddings=tiles,
            attempt_chain=_embedding_payload(_payload)["attempt_chain"],
        )


def test_extractor_rehashes_every_source_after_decode_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    import torch

    for name, value in temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT["process_environment"].items():
        monkeypatch.setenv(name, value)
    plan, _artifact_sha256, _file_sha256, payload = _verified_plan(tmp_path)
    source_paths = tuple(tmp_path / receipt["filename"] for receipt in payload["task0257_receipts"]["source_videos"])
    for path in source_paths:
        path.write_bytes(b"video")
    checkpoint_receipt = payload["task0257_receipts"]["checkpoints"][1]
    checkpoint_path = tmp_path / checkpoint_receipt["filename"]
    checkpoint_path.write_bytes(b"x" * checkpoint_receipt["size_bytes"])
    calls: list[Path] = []
    identities = {
        path: (
            1,
            index + 1,
            payload["task0257_receipts"]["source_videos"][index]["size_bytes"],
            index + 10,
        )
        for index, path in enumerate(source_paths)
    }

    def fake_file_hash(path: Path):
        resolved = Path(path).resolve()
        calls.append(resolved)
        index = source_paths.index(resolved)
        return payload["task0257_receipts"]["source_videos"][index]["file_sha256"], identities[resolved]

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, _field: int) -> int:
            return cv2.CAP_FFMPEG

        def release(self) -> None:
            pass

    monkeypatch.setattr(temporal_module, "_file_sha256_no_follow", fake_file_hash)
    monkeypatch.setattr(
        temporal_module,
        "_load_swin_backbone_from_verified_descriptor",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args, **_kwargs: FakeCapture())
    monkeypatch.setattr(
        temporal_module,
        "_decode_exact_frame_tiles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("decode failed")),
    )
    monkeypatch.setattr(torch, "device", lambda *_args, **_kwargs: SimpleNamespace(type="mps"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)

    with pytest.raises(ValueError, match="decode failed"):
        temporal_module._extract_tiled_swin_rows(
            plan=plan,
            source_video_paths=source_paths,
            checkpoint_path=checkpoint_path,
        )

    assert calls == [*source_paths, *source_paths]


def test_decoder_independently_seeks_and_reads_each_authoritative_frame_index() -> None:
    import cv2

    frame_indexes = [list(range(offset, offset + 16)) for offset in range(0, 64, 16)]

    class FakeCapture:
        def __init__(self) -> None:
            self.position = 0
            self.seeks: list[tuple[int, int]] = []
            self.read_positions: list[int] = []

        def get(self, field: int) -> int:
            if field == cv2.CAP_PROP_FRAME_WIDTH:
                return 4
            if field == cv2.CAP_PROP_FRAME_HEIGHT:
                return 3
            raise AssertionError(f"unexpected capture property {field}")

        def set(self, field: int, value: int) -> bool:
            self.position = int(value)
            self.seeks.append((field, self.position))
            return True

        def read(self):
            self.read_positions.append(self.position)
            frame = np.full((3, 4, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, frame

    capture = FakeCapture()
    tiles = temporal_module._decode_exact_frame_tiles(capture, frame_indexes)
    flattened = [index for tile in frame_indexes for index in tile]

    assert capture.seeks == [(cv2.CAP_PROP_POS_FRAMES, index) for index in flattened]
    assert capture.read_positions == flattened
    assert all(tile.shape == (16, 3, 4, 3) and tile.dtype == np.uint8 and tile.flags.c_contiguous for tile in tiles)


@pytest.mark.parametrize(
    "invalid_frame",
    (
        pytest.param(np.zeros((3, 4, 3), dtype=np.uint16), id="non-uint8"),
        pytest.param(np.zeros((3, 4, 4), dtype=np.uint8), id="non-three-channel"),
        pytest.param(np.asfortranarray(np.zeros((3, 4, 3), dtype=np.uint8)), id="non-c-contiguous"),
        pytest.param(np.zeros((2, 4, 3), dtype=np.uint8), id="wrong-source-dimensions"),
    ),
)
def test_decoder_rejects_frames_outside_the_frozen_pixel_contract(
    invalid_frame: np.ndarray,
) -> None:
    import cv2

    class FakeCapture:
        def __init__(self) -> None:
            self.position = 0

        def get(self, field: int) -> int:
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 4,
                cv2.CAP_PROP_FRAME_HEIGHT: 3,
            }[field]

        def set(self, _field: int, value: int) -> bool:
            self.position = int(value)
            return True

        def read(self):
            self.position += 1
            return True, invalid_frame

    with pytest.raises(ValueError):
        temporal_module._decode_exact_frame_tiles(
            FakeCapture(),
            [list(range(offset, offset + 16)) for offset in range(0, 64, 16)],
        )


def test_swin_checkpoint_hash_and_torch_load_share_one_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    checkpoint = tmp_path / "swin.pth"
    checkpoint.write_bytes(b"verified-weights")
    observed: dict[str, object] = {}

    def fake_torch_load(handle, *, map_location: str, weights_only: bool):
        observed["is_path"] = isinstance(handle, (str, Path))
        observed["payload"] = handle.read()
        observed["map_location"] = map_location
        observed["weights_only"] = weights_only
        raise RuntimeError("stop after descriptor proof")

    monkeypatch.setattr(torch, "load", fake_torch_load)
    with pytest.raises(RuntimeError, match="descriptor proof"):
        temporal_module._load_swin_backbone_from_verified_descriptor(
            checkpoint,
            expected_file_sha256=hashlib.sha256(b"verified-weights").hexdigest(),
            expected_size_bytes=len(b"verified-weights"),
            device=SimpleNamespace(type="mps"),
        )

    assert observed == {
        "is_path": False,
        "payload": b"verified-weights",
        "map_location": "cpu",
        "weights_only": True,
    }
