#!/usr/bin/env python3
"""Seal the DeepBall-Large development and independent window screens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.screen_deepball_large_windows import summarize_window_rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_hou_probe(probe: dict[str, Any], base_screen: dict[str, Any]) -> dict[str, Any]:
    event_rows: list[dict[str, Any]] = []
    detections = probe.get("detections") or []
    detection_frames = [
        int(row["frame"])
        for row in detections
        if row.get("detections")
    ]
    candidate_matches = base_screen["hou_sac_screen"]["interval"]["candidate_matches"]
    for event in candidate_matches:
        start_frame = int(event["start_frame"])
        end_frame = int(event["end_frame"])
        matched = [frame for frame in detection_frames if start_frame <= frame <= end_frame]
        event_present = bool(event["matched_shot_frames"])
        event_rows.append(
            {
                "event_id": str(event["event_id"]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "event_present": event_present,
                "prediction_present": bool(matched),
                "detection_frames": matched,
            }
        )
    return {
        "label_policy": "fixed predeclared HOU-SAC candidate windows; event presence is the existing PBP match field",
        "metrics": summarize_window_rows(event_rows),
        "rows": event_rows,
    }


def seal_screen(
    *,
    hou_probe_path: Path,
    hou_base_screen_path: Path,
    lakers_screen_path: Path,
    model_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    for path in (
        hou_probe_path,
        hou_base_screen_path,
        lakers_screen_path,
        model_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    hou_probe = json.loads(hou_probe_path.read_text(encoding="utf-8"))
    base_screen = json.loads(hou_base_screen_path.read_text(encoding="utf-8"))
    lakers_screen = json.loads(lakers_screen_path.read_text(encoding="utf-8"))
    hou = evaluate_hou_probe(hou_probe, base_screen)
    lakers = {
        "artifact_path": str(lakers_screen_path),
        "artifact_sha256": _file_sha256(lakers_screen_path),
        "metrics": lakers_screen["metrics"],
        "rows": lakers_screen["rows"],
        "label_policy": "fixed independently reviewed Lakers-Magic raw-video windows; prediction is any DeepBall-Large detection in the window",
    }
    payload: dict[str, Any] = {
        "schema_version": "agu.deepball-large-screen.v1",
        "purpose": "offline_deepball_large_cross_broadcast_detector_screen",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "model": {
            "backend": "wasb_deepball_large_basketball_v1",
            "checkpoint_path": str(model_path),
            "checkpoint_sha256": _file_sha256(model_path),
            "score_threshold": 0.5,
            "device": "cpu",
        },
        "inputs": {
            "hou_probe_path": str(hou_probe_path),
            "hou_probe_sha256": _file_sha256(hou_probe_path),
            "hou_base_screen_path": str(hou_base_screen_path),
            "hou_base_screen_sha256": _file_sha256(hou_base_screen_path),
            "lakers_screen_path": str(lakers_screen_path),
            "lakers_screen_sha256": _file_sha256(lakers_screen_path),
        },
        "development_screen": {
            "game": "HOU-SAC",
            "video_sha256": hou_probe["raw_video"]["sha256"],
            **hou,
        },
        "independent_screen": {
            "game": "Lakers-Magic",
            "video_sha256": lakers_screen["video_sha256"],
            **lakers,
        },
        "resource_logs": [
            "analysis_outputs/public_research/resource_deepball_large_hou_sac_v1.jsonl",
            "analysis_outputs/public_research/resource_deepball_large_lakers_magic_v1.jsonl",
        ],
        "decision": {
            "accepted": False,
            "reason": "The development-window screen reaches the dual floor, but the independently held Lakers-Magic screen has precision 0.30 and F1 0.4444; retain as offline baseline only and do not alter AGU detector/VLM/runtime defaults.",
        },
        "cleanup": {
            "temporary_probe_threshold07_retained": True,
            "checkpoint_retained": True,
            "checkpoint_retention_reason": "3.9 MiB official baseline remains useful for future offline detector fusion; it is not runtime-consumable or promotion-eligible.",
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hou-probe", type=Path, required=True)
    parser.add_argument("--hou-base-screen", type=Path, required=True)
    parser.add_argument("--lakers-screen", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = seal_screen(
        hou_probe_path=args.hou_probe,
        hou_base_screen_path=args.hou_base_screen,
        lakers_screen_path=args.lakers_screen,
        model_path=args.model,
        output_path=args.output,
    )
    print(json.dumps({"metrics": {
        "hou": artifact["development_screen"]["metrics"],
        "lakers": artifact["independent_screen"]["metrics"],
    }, "accepted": artifact["decision"]["accepted"], "artifact_sha256": artifact["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
