#!/usr/bin/env python3
"""Extract all 24 causal review panels with the local MobileNet backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.causal_phase_dense_temporal import (  # noqa: E402
    crop_dense_causal_sheet_panels,
    seal_dense_causal_frame_embeddings,
)
from app.analysis.causal_shot_phase_review import (  # noqa: E402
    verify_causal_shot_phase_review_plan,
)
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    load_scene_backbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sheet-manifest", type=Path, required=True)
    parser.add_argument("--sheet-root", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def extract_dense_embeddings(
    *,
    plan_path: Path,
    sheet_manifest_path: Path,
    sheet_root: Path,
    checkpoint_path: Path,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    plan = verify_causal_shot_phase_review_plan(_read_json(plan_path))
    manifest = _verify_sheet_manifest(
        _read_json(sheet_manifest_path),
        plan_sha256=plan["artifact_sha256"],
    )
    device = _resolve_device(device_name)
    backbone, transform = load_scene_backbone(
        checkpoint_path,
        device=device,
    )
    planned = {
        str(row["phase_review_id"]): row for row in plan["examples"]
    }
    manifest_ids = [
        str(review_id)
        for record in manifest["records"]
        for review_id in record["phase_review_ids"]
    ]
    if len(manifest_ids) != len(set(manifest_ids)) or set(
        manifest_ids
    ) != set(planned):
        raise ValueError("sheet manifest must exactly cover the causal plan")

    values_by_review: dict[str, list[list[float] | None]] = {
        review_id: [None] * len(plan["anchor_offsets_seconds"])
        for review_id in planned
    }
    pending_tensors: list[torch.Tensor] = []
    pending_refs: list[tuple[str, int]] = []

    def flush() -> None:
        if not pending_tensors:
            return
        with torch.inference_mode():
            values = backbone(torch.stack(pending_tensors).to(device))
        rows = values.detach().cpu().to(torch.float32).tolist()
        for (review_id, position), embedding in zip(
            pending_refs,
            rows,
            strict=True,
        ):
            values_by_review[review_id][position] = embedding
        pending_tensors.clear()
        pending_refs.clear()

    root = sheet_root.resolve()
    for record_index, record in enumerate(manifest["records"], start=1):
        sheet_path = (root / str(record["sheet"])).resolve()
        if (
            not sheet_path.is_relative_to(root)
            or not sheet_path.is_file()
            or _file_sha256(sheet_path) != record["sheet_sha256"]
        ):
            raise ValueError("causal review sheet hash mismatch")
        image = cv2.imread(str(sheet_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode causal review sheet: {sheet_path.name}")
        review_ids = [str(value) for value in record["phase_review_ids"]]
        for example_index, review_id in enumerate(review_ids):
            plan_row = planned[review_id]
            if (
                plan_row["source_video_sha256"]
                != record["source_video_sha256"]
            ):
                raise ValueError("causal sheet source binding mismatch")
            panels = crop_dense_causal_sheet_panels(
                image,
                example_index=example_index,
                examples_in_sheet=len(review_ids),
                panel_width=int(manifest["rendering"]["panel_width"]),
                grid_columns=int(manifest["rendering"]["grid_columns"]),
                grid_rows_per_example=int(
                    manifest["rendering"]["grid_rows_per_example"]
                ),
            )
            for position, panel in enumerate(panels):
                rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)
                pending_tensors.append(transform(tensor))
                pending_refs.append((review_id, position))
                if len(pending_tensors) >= batch_size:
                    flush()
        print(
            json.dumps(
                {
                    "stage": "dense_causal_panel_embeddings",
                    "sheet": record_index,
                    "sheet_count": len(manifest["records"]),
                    "event_count": sum(
                        all(value is not None for value in rows)
                        for rows in values_by_review.values()
                    ),
                }
            ),
            flush=True,
        )
    flush()

    examples = []
    for row in plan["examples"]:
        review_id = str(row["phase_review_id"])
        embeddings = values_by_review[review_id]
        if any(value is None for value in embeddings):
            raise ValueError("dense causal embedding extraction is incomplete")
        examples.append(
            {
                "phase_review_id": review_id,
                "source_video_sha256": row["source_video_sha256"],
                "candidate_bundle_sha256": row[
                    "candidate_bundle_sha256"
                ],
                "event_id": row["event_id"],
                "frame_indexes": row["frame_indexes"],
                "embeddings": embeddings,
            }
        )
    return seal_dense_causal_frame_embeddings(
        {
            "purpose": "training_only_dense_causal_frame_embeddings",
            "codex_runtime_answer_used": False,
            "review_plan_sha256": plan["artifact_sha256"],
            "sheet_manifest_sha256": manifest["artifact_sha256"],
            "source_video_sha256s": plan["source_video_sha256s"],
            "sealed_blind_video_sha256s": plan[
                "sealed_blind_video_sha256s"
            ],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": _file_sha256(checkpoint_path),
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "anchor_offsets_seconds": plan["anchor_offsets_seconds"],
            "examples": examples,
        }
    )


def _verify_sheet_manifest(
    payload: Mapping[str, Any],
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if (
        artifact.get("schema_version")
        != "agu.causal-shot-phase-review-sheets.v1"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("labels_hidden_from_reviewer") is not True
        or artifact.get("plan_sha256") != plan_sha256
        or artifact.get("rendering", {}).get("raw_frames_only") is not True
        or artifact.get("rendering", {}).get(
            "labels_or_predictions_rendered"
        )
        is not False
        or not isinstance(artifact.get("records"), list)
        or not artifact["records"]
    ):
        raise ValueError("invalid causal review sheet manifest")
    if claimed != _canonical_sha256(artifact):
        raise ValueError("causal review sheet manifest hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(value)
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return device


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    artifact = extract_dense_embeddings(
        plan_path=args.plan,
        sheet_manifest_path=args.sheet_manifest,
        sheet_root=args.sheet_root,
        checkpoint_path=args.backbone_checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "example_count": len(artifact["examples"]),
                "embedding_dimension": artifact["embedding_dimension"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
