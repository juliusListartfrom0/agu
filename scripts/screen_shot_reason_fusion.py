#!/usr/bin/env python3
"""Screen continuous shot-reason evidence with nested game-held isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ball_release_annotation import (  # noqa: E402
    verify_ball_release_plan,
    verify_ball_release_review,
)
from app.analysis.shot_reason_fusion import screen_shot_reason_fusion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-embeddings", type=Path, required=True)
    parser.add_argument("--video-embeddings", type=Path, action="append", default=[])
    parser.add_argument("--reason-evidence", type=Path, required=True)
    parser.add_argument(
        "--review-pair",
        nargs=2,
        metavar=("PLAN", "REVIEW"),
        action="append",
        required=True,
    )
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--regularization-c", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_rows = []
    review_artifact_sha256s = []
    for plan_path, review_path in args.review_pair:
        plan = verify_ball_release_plan(_read_json(Path(plan_path)))
        review = verify_ball_release_review(
            _read_json(Path(review_path)),
            plan=plan,
        )
        review_rows.extend(review["reviews"])
        review_artifact_sha256s.append(review["artifact_sha256"])
    result = screen_shot_reason_fusion(
        scene_artifact=_read_json(args.scene_embeddings),
        video_artifacts=[_read_json(path) for path in args.video_embeddings],
        reason_evidence_artifact=_read_json(args.reason_evidence),
        review_rows=review_rows,
        pca_components=args.pca_components,
        regularization_c=args.regularization_c,
    )
    result["review_artifact_sha256s"] = sorted(review_artifact_sha256s)
    result["artifact_sha256"] = _canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "best_variant": result["best_variant"]["name"],
                "best_gate": result["best_variant"]["gate"],
                "auxiliary_metrics": result["auxiliary_metrics"],
                "artifact_sha256": result["artifact_sha256"],
            }
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(payload: dict[str, object]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
