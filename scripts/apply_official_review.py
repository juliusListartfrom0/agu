#!/usr/bin/env python3
"""Apply hash-bound Codex/human decisions and reseal official event output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.review import finalize_reviewed_bundle  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse, ReviewDecisionResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-team-points", default="{}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = RawOnlyPredictionBundleResponse.model_validate_json(args.bundle.read_text(encoding="utf-8"))
    decisions = _load_decisions(args.decisions)
    expected = json.loads(args.expected_team_points)
    if not isinstance(expected, dict):
        raise ValueError("--expected-team-points must be a JSON object")
    reviewed, ledger, score = finalize_reviewed_bundle(
        bundle,
        decisions=decisions,
        raw_video_paths=args.video,
        config={"pipeline": "official_review_v1", "decision_count": len(decisions)},
        expected_team_points={str(key): int(value) for key, value in expected.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "reviewed_prediction_bundle.json", reviewed.model_dump(mode="json"))
    _write_json(args.output_dir / "official_box_score.json", score.model_dump(mode="json"))
    history = "".join(
        json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for event in ledger.all_revisions()
    )
    (args.output_dir / "event_revision_history.jsonl").write_text(history, encoding="utf-8")
    _write_json(
        args.output_dir / "review_audit.json",
        {
            "source_bundle_sha256": bundle.bundle_sha256,
            "reviewed_bundle_sha256": reviewed.bundle_sha256,
            "decision_count": len(decisions),
            "revision_count": len(ledger.all_revisions()),
            "latest_event_count": len(ledger.latest_events()),
            "box_score_status": score.status,
        },
    )
    print(json.dumps({"bundle_sha256": reviewed.bundle_sha256, "status": score.status}, indent=2))
    return 0


def _load_decisions(path: Path) -> list[ReviewDecisionResponse]:
    decisions: list[ReviewDecisionResponse] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decisions.append(ReviewDecisionResponse.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid review decision row {line_number}: {exc}") from exc
    return decisions


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
