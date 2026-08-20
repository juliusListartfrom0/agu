"""TASK-0258 Amendment-001 v2 verification worker — extraction wiring.

Reuses the parent Module-A extraction contract (``_extract_tiled_swin_rows``)
for the second empty-state extraction. The read-isolation guarantee is
*structural* here: the verification run passes ``initial_rows=()`` (empty
prefix) and never loads the producer embedding, resume, or caller feature
matrix. The stronger kernel-audit enforcement is pinned by the read-isolation
policy/attestation schemas in ``task0258_v2_read_isolation``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.analysis.task0258_v2_verification import (
    ROLE,
    build_verification_embedding_payload,
    compute_verification_projection,
    normalize_float32_leaf,
)


def build_verification_examples(
    plan: object,
    tile_embeddings: Sequence[Sequence[Sequence[float]]],
) -> list[dict[str, object]]:
    """Construct the 45 plan-order example rows from extracted tiles.

    Mirrors the parent producer example construction (ordinal/key/
    tile_frame_indexes/derivation_only_tile_embeddings/model_input) with every
    float normalized to float32. ``plan`` is the parent's verified plan whose
    ``_payload["ordered_examples"]`` carries the frozen keys and frame indexes.
    """
    from app.analysis.vru_causal_temporal_retrospective import combine_tiled_swin_embeddings

    plan_examples = plan._payload["ordered_examples"]  # type: ignore[attr-defined]
    if len(tile_embeddings) != len(plan_examples):
        raise ValueError("verification rows do not exactly cover the plan")
    examples: list[dict[str, object]] = []
    for row, plan_row in zip(tile_embeddings, plan_examples, strict=True):
        normalized_tiles = normalize_float32_leaf([list(tile) for tile in row])
        examples.append(
            {
                "ordinal": plan_row["ordinal"],
                "key": {
                    "source_video_sha256": plan_row["source_video_sha256"],
                    "candidate_bundle_sha256": plan_row["candidate_bundle_sha256"],
                    "event_id": plan_row["event_id"],
                },
                "tile_frame_indexes": copy.deepcopy(plan_row["tile_frame_indexes"]),
                "derivation_only_tile_embeddings": normalized_tiles,
                "model_input": combine_tiled_swin_embeddings(normalized_tiles),
            }
        )
    return examples


def extract_verification_rows(
    *,
    plan: object,
    source_video_paths: Sequence[Path],
    checkpoint_path: Path,
) -> list[list[list[float]]]:
    """Empty-state re-extraction of all 45 rows (no producer prefix).

    Delegates to the parent ``_extract_tiled_swin_rows`` with ``initial_rows=()``,
    then float32-normalizes. Requires the frozen runtime (MPS/torch/cv2, the
    verified checkpoint, and the four SHA-bound source videos).
    """
    from app.analysis.vru_causal_temporal_retrospective import _extract_tiled_swin_rows

    rows = _extract_tiled_swin_rows(
        plan=plan,
        source_video_paths=source_video_paths,
        checkpoint_path=checkpoint_path,
        initial_rows=(),
    )
    return normalize_float32_leaf([list(tile) for tile in rows])


def run_verification_extraction(
    *,
    plan: object,
    source_video_paths: Sequence[Path],
    checkpoint_path: Path,
    representation: str,
    authorization_receipts: Mapping[str, object],
    run_identity_receipt: Mapping[str, object],
    history_head_receipt: Mapping[str, object],
    run_admission_receipt: Mapping[str, object],
    plan_receipt: Mapping[str, object],
    task0257_input_receipts: object,
    producer_environment: object,
    checkpoint_receipt: Mapping[str, object],
    source_video_receipts: object,
) -> tuple[dict[str, object], str]:
    """Run the empty-state verification extraction and seal the embedding.

    Returns ``(verification_embedding_payload, computational_projection_sha256)``.
    """
    rows = extract_verification_rows(
        plan=plan,
        source_video_paths=source_video_paths,
        checkpoint_path=checkpoint_path,
    )
    examples = build_verification_examples(plan, rows)
    payload = build_verification_embedding_payload(
        authorization_receipts=authorization_receipts,
        run_identity_receipt=run_identity_receipt,
        history_head_receipt=history_head_receipt,
        run_admission_receipt=run_admission_receipt,
        plan_receipt=plan_receipt,
        task0257_input_receipts=task0257_input_receipts,
        representation=representation,
        producer_environment=producer_environment,
        checkpoint_receipt=checkpoint_receipt,
        source_video_receipts=source_video_receipts,
        examples=examples,
    )
    projection = compute_verification_projection(representation, examples)
    return payload, projection


__all__ = [
    "ROLE",
    "build_verification_examples",
    "extract_verification_rows",
    "run_verification_extraction",
]
