from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import scripts.attach_face_player_context as cli


def _seal(payload: dict) -> dict:
    payload = {**payload, "manifest_sha256": ""}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_cli_writes_context_and_team_subset(tmp_path: Path, monkeypatch) -> None:
    candidates = _seal(
        {
            "schema_version": "agu.face-enrollment-candidates.v1",
            "benchmark_disjoint": True,
            "model_id": "test",
            "sources": [
                {
                    "source_video_id": "enrollment_001",
                    "filename": "game.mp4",
                    "sha256": "video",
                }
            ],
            "config": {},
            "clusters": [
                {
                    "cluster_id": "face-001",
                    "samples": [
                        {
                            "path": "a.jpg",
                            "source_video_id": "enrollment_001",
                            "frame": 25,
                            "bbox": [10, 10, 10, 10],
                        },
                        {
                            "path": "b.jpg",
                            "source_video_id": "enrollment_001",
                            "frame": 50,
                            "bbox": [10, 10, 10, 10],
                        },
                    ],
                }
            ],
        }
    )
    perception = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {"filename": "game.mp4", "sha256": "video"},
        "detections": [
            {
                "frame": frame,
                "object_type": "player",
                "bbox": {"x1": 0, "y1": 0, "x2": 30, "y2": 50},
                "confidence": 0.9,
                "track_id": 7,
                "team_id": "raw-dark",
            }
            for frame in (25, 50)
        ],
    }
    candidates_path = tmp_path / "candidates.json"
    perception_path = tmp_path / "perception.json"
    context_path = tmp_path / "context.json"
    subset_path = tmp_path / "subset.json"
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
    perception_path.write_text(json.dumps(perception), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attach_face_player_context.py",
            "--candidates",
            str(candidates_path),
            "--perception",
            str(perception_path),
            "--team-map",
            "raw-dark=1610612753",
            "--output-context",
            str(context_path),
            "--team-id",
            "1610612753",
            "--output-candidates",
            str(subset_path),
        ],
    )

    assert cli.main() == 0
    context = json.loads(context_path.read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    assert context["clusters"][0]["dominant_team_id"] == "1610612753"
    assert [row["cluster_id"] for row in subset["clusters"]] == ["face-001"]
