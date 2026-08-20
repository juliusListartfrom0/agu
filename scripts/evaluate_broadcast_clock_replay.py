#!/usr/bin/env python3
"""Evaluate the offline broadcast-clock replay proxy on sealed labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.broadcast_clock import evaluate_broadcast_clock_proxy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--minimum-replay-predictions", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.labels
    ]
    evidence_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.evidence
    ]
    metrics = evaluate_broadcast_clock_proxy(
        labels=label_payloads,
        evidence=evidence_payloads,
        minimum_precision=args.minimum_precision,
        minimum_replay_predictions=args.minimum_replay_predictions,
    )
    artifact = {
        "schema_version": "agu.broadcast-clock-replay-evaluation.v1",
        "purpose": "offline_development_replay_proxy_evaluation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "label_artifact_sha256": [_file_sha256(path) for path in args.labels],
        "evidence_artifact_sha256": [
            str(payload["artifact_sha256"]) for payload in evidence_payloads
        ],
        "metrics": metrics,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_sha256": artifact["artifact_sha256"], **metrics}))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
