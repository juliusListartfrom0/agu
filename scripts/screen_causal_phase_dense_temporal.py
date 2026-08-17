#!/usr/bin/env python3
"""Train and screen the 24-frame causal temporal head by held-out game."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.causal_phase_dense_temporal import (  # noqa: E402
    attach_dense_reason_features,
    build_dense_causal_temporal_examples,
    screen_dense_causal_temporal_examples,
)
from app.analysis.causal_phase_entity_relations import (  # noqa: E402
    attach_dense_entity_relation_features,
)
from app.analysis.causal_phase_geometry import (  # noqa: E402
    attach_dense_geometry_features,
)
from app.analysis.causal_shot_phase_review import (  # noqa: E402
    verify_causal_shot_phase_review,
    verify_causal_shot_phase_review_plan,
)
from app.analysis.pbp_visual_state_review import (  # noqa: E402
    verify_visual_state_label_corrections,
    verify_visual_state_review_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-corrections", type=Path, required=True)
    parser.add_argument("--followup-plan", type=Path, required=True)
    parser.add_argument("--followup-corrections", type=Path, required=True)
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--phase-review", type=Path, required=True)
    parser.add_argument("--reason-artifact", type=Path)
    parser.add_argument("--geometry-artifact", type=Path)
    parser.add_argument("--entity-relation-artifact", type=Path)
    parser.add_argument("--include-gru", action="store_true")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_screen(
    *,
    embeddings_path: Path,
    base_plan_path: Path,
    base_corrections_path: Path,
    followup_plan_path: Path,
    followup_corrections_path: Path,
    phase_plan_path: Path,
    phase_review_path: Path,
    reason_artifact_path: Path | None = None,
    geometry_artifact_path: Path | None = None,
    entity_relation_artifact_path: Path | None = None,
    include_gru: bool = False,
    include_gated: bool = False,
) -> dict[str, Any]:
    base_plan = verify_visual_state_review_plan(_read_json(base_plan_path))
    base_corrections = verify_visual_state_label_corrections(
        _read_json(base_corrections_path),
        plan=base_plan,
    )
    followup_plan = verify_visual_state_review_plan(
        _read_json(followup_plan_path)
    )
    followup_corrections = verify_visual_state_label_corrections(
        _read_json(followup_corrections_path),
        plan=followup_plan,
    )
    if (
        followup_plan.get("parent_plan_sha256")
        != base_plan["artifact_sha256"]
        or followup_plan.get("parent_corrections_sha256")
        != base_corrections["artifact_sha256"]
        or followup_plan.get("sealed_blind_video_sha256s")
        != base_plan["sealed_blind_video_sha256s"]
    ):
        raise ValueError("visual-state correction chain is invalid")
    phase_plan = verify_causal_shot_phase_review_plan(
        _read_json(phase_plan_path)
    )
    phase_review = verify_causal_shot_phase_review(
        _read_json(phase_review_path),
        plan=phase_plan,
    )
    rows, provenance = build_dense_causal_temporal_examples(
        embedding_artifact=_read_json(embeddings_path),
        base_decisions=base_corrections["decisions"],
        followup_decisions=followup_corrections["decisions"],
        phase_reviews=phase_review["reviews"],
    )
    if (
        set(provenance["sealed_blind_video_sha256s"])
        != set(phase_plan["sealed_blind_video_sha256s"])
        or provenance["review_plan_sha256"]
        != phase_plan["artifact_sha256"]
    ):
        raise ValueError("dense temporal phase provenance mismatch")
    if reason_artifact_path is not None:
        rows, reason_provenance = attach_dense_reason_features(
            rows,
            reason_artifact=_read_json(reason_artifact_path),
        )
        component_hashes = [
            provenance["embedding_artifact_sha256"],
            reason_provenance["reason_artifact_sha256"],
        ]
        provenance = {
            **provenance,
            "backbone": f"{provenance['backbone']}+shot_reason_v2",
            "backbone_sha256": _canonical_sha256(component_hashes),
            "embedding_dimension": (
                int(provenance["embedding_dimension"])
                + int(reason_provenance["reason_feature_dimension"])
            ),
            "component_artifact_sha256s": component_hashes,
        }
    if geometry_artifact_path is not None:
        rows, geometry_provenance = attach_dense_geometry_features(
            rows,
            geometry_artifact=_read_json(geometry_artifact_path),
        )
        component_hashes = list(
            provenance.get(
                "component_artifact_sha256s",
                [provenance["embedding_artifact_sha256"]],
            )
        )
        component_hashes.append(
            geometry_provenance["geometry_artifact_sha256"]
        )
        provenance = {
            **provenance,
            "backbone": f"{provenance['backbone']}+dense_geometry_v1",
            "backbone_sha256": _canonical_sha256(component_hashes),
            "embedding_dimension": (
                int(provenance["embedding_dimension"])
                + int(geometry_provenance["geometry_feature_dimension"])
            ),
            "component_artifact_sha256s": component_hashes,
        }
    if entity_relation_artifact_path is not None:
        rows, relation_provenance = attach_dense_entity_relation_features(
            rows,
            relation_artifact=_read_json(entity_relation_artifact_path),
        )
        component_hashes = list(
            provenance.get(
                "component_artifact_sha256s",
                [provenance["embedding_artifact_sha256"]],
            )
        )
        component_hashes.append(
            relation_provenance["entity_relation_artifact_sha256"]
        )
        provenance = {
            **provenance,
            **relation_provenance,
            "backbone": (
                f"{provenance['backbone']}+dense_entity_relations_v1"
            ),
            "backbone_sha256": _canonical_sha256(component_hashes),
            "embedding_dimension": (
                int(provenance["embedding_dimension"])
                + int(
                    relation_provenance[
                        "entity_relation_feature_dimension"
                    ]
                )
            ),
            "component_artifact_sha256s": component_hashes,
        }
    operators = ["conv"]
    if include_gru:
        operators.append("gru")
    if include_gated:
        operators.append("gated")
    return screen_dense_causal_temporal_examples(
        examples=rows,
        provenance={
            **provenance,
            "label_correction_artifact_sha256s": [
                base_corrections["artifact_sha256"],
                followup_corrections["artifact_sha256"],
            ],
            "phase_review_artifact_sha256": phase_review[
                "artifact_sha256"
            ],
        },
        configurations=[
            {
                "temporal_operator": operator,
                "hidden_dimension": hidden,
                "kernel_size": 3,
                "auxiliary_weight": auxiliary,
                "epochs": 60,
                "learning_rate": 0.01,
                "weight_decay": 0.001,
            }
            for operator in operators
            for hidden in (16, 32)
            for auxiliary in (0.0, 0.25)
        ],
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


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
    artifact = run_screen(
        embeddings_path=args.embeddings,
        base_plan_path=args.base_plan,
        base_corrections_path=args.base_corrections,
        followup_plan_path=args.followup_plan,
        followup_corrections_path=args.followup_corrections,
        phase_plan_path=args.phase_plan,
        phase_review_path=args.phase_review,
        reason_artifact_path=args.reason_artifact,
        geometry_artifact_path=args.geometry_artifact,
        entity_relation_artifact_path=args.entity_relation_artifact,
        include_gru=args.include_gru,
        include_gated=args.include_gated,
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
                "metrics": artifact["metrics"],
                "accepted": artifact["accepted"],
                "artifact_sha256": artifact["artifact_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
