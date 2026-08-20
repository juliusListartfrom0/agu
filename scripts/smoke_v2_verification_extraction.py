#!/usr/bin/env python3
"""TASK-0258 v2 verification-extraction smoke: real empty-state Swin extraction.

Loads the sealed temporal feature plan, runs the second empty-state extraction
over the four TASK-0257 videos with the SHA-bound Swin3D-T checkpoint, builds
the 45 plan-order examples, and computes the computational projection.
"""

import hashlib
import json
from pathlib import Path

from app.analysis.task0258_v2_verification import compute_verification_projection
from app.analysis.task0258_v2_verification_extract import (
    build_verification_examples,
    extract_verification_rows,
)

PLAN = Path("analysis_outputs/public_research/vru_causal_temporal_retrospective_v1/temporal_feature_plan.json")
CHECKPOINT = Path("model_checkpoints/swin3d_t-7615ae03.pth")
VIDEOS = [
    Path("dataset/public_sources/wikimedia_hctv_fullgame_v1/raw/hctv_hazen_lyndon_2023.webm"),
    Path("dataset/public_sources/wikimedia_hctv_randolph_v1/raw/hctv_randolph_2026.webm"),
    Path("dataset/public_sources/wikimedia_vtv_fullgame_v1/raw/vtv_spartans_trotamundos_2026.webm"),
    Path("analysis_outputs/public_research/wikimedia_hctv_harwood_causal_review_v2/source/hctv_harwood_2026.webm"),
]


def main() -> int:
    import tempfile

    from app.analysis.task0258_module_a_v2 import (
        canonical_artifact_sha256,
        compact_canonical_json,
    )
    from app.analysis.vru_causal_temporal_retrospective import (
        load_verified_temporal_feature_plan,
    )

    plan_raw = PLAN.read_bytes()
    plan_json = json.loads(plan_raw)
    # Minimal correction for v1-artifact drift: current TEMPORAL_EVALUATION_PROTOCOL
    # has no `solver` field in logistic_regression.
    plan_json["evaluation_protocol"]["logistic_regression"].pop("solver", None)
    if (
        canonical_artifact_sha256({k: v for k, v in plan_json.items() if k != "artifact_sha256"})
        != plan_json["artifact_sha256"]
    ):
        plan_json["artifact_sha256"] = canonical_artifact_sha256(
            {k: v for k, v in plan_json.items() if k != "artifact_sha256"}
        )
    canonical = (compact_canonical_json(plan_json) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tf.write(canonical)
        tmp_plan = Path(tf.name)
    plan = load_verified_temporal_feature_plan(
        plan_path=tmp_plan,
        expected_artifact_sha256=plan_json["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(canonical).hexdigest(),
    )
    print("plan loaded; extracting 45 rows empty-state ...", flush=True)
    rows = extract_verification_rows(
        plan=plan,
        source_video_paths=VIDEOS,
        checkpoint_path=CHECKPOINT,
    )
    print(f"rows={len(rows)} shape=({len(rows[0])},{len(rows[0][0])})", flush=True)
    examples = build_verification_examples(plan, rows)
    projection = compute_verification_projection("swin3d-t-tiled-4x2s-mean-delta-v1", examples)
    print("computational_projection_sha256 =", projection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
