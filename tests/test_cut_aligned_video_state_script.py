from __future__ import annotations

import copy
import json
from pathlib import Path

from app.analysis.cut_aligned_video_state import (
    seal_cut_aligned_video_embedding_artifact,
)
from scripts.extract_cut_aligned_visual_state_embeddings import run_extraction
from scripts.screen_cut_aligned_visual_state_embeddings import run_screen

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "100f7d9a07251fe133f3f1b7fc18a9b147997862e39a2ad5864eed5f96478a64"


def test_cut_aligned_extraction_binds_plan_transition_video_and_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "fH-RPTKY4zI.mp4"
    checkpoint_path = tmp_path / "mvit.pth"
    output_path = tmp_path / "embeddings.json"
    video_path.write_bytes(b"video")
    checkpoint_path.write_bytes(b"checkpoint")

    def fake_sha(path: Path) -> str:
        return SOURCE_SHA if path == video_path else "2" * 64

    def fake_extract(**kwargs):
        rows = copy.deepcopy(kwargs["planned_examples"])
        for example in rows:
            for segment in example["segments"]:
                segment["embedding"] = (
                    [1.0] * 768 if segment["available"] else None
                )
        return rows

    monkeypatch.setattr(
        "scripts.extract_cut_aligned_visual_state_embeddings.file_sha256",
        fake_sha,
    )
    monkeypatch.setattr(
        "scripts.extract_cut_aligned_visual_state_embeddings.load_video_backbone",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        "scripts.extract_cut_aligned_visual_state_embeddings."
        "extract_cut_aligned_video_embeddings",
        fake_extract,
    )

    artifact = run_extraction(
        review_plan_path=ROOT
        / "analysis_outputs/public_research/pbp_visual_state_codex_review_v1/plan.json",
        transition_path=ROOT
        / "analysis_outputs/public_research/atl_chi_benchmark/"
        "pbp_visual_state_transition_boundaries_v1.json",
        video_path=video_path,
        backbone_name="torchvision/mvit_v2_s/kinetics400_v1",
        backbone_checkpoint=checkpoint_path,
        output_path=output_path,
        clip_frames=16,
        batch_size=1,
        device_name="cpu",
    )

    assert artifact["complete"] is True
    assert len(artifact["examples"]) == 33
    assert artifact["raw_video_sha256"] == SOURCE_SHA
    assert json.loads(output_path.read_text())["artifact_sha256"] == artifact[
        "artifact_sha256"
    ]


def test_cut_aligned_screen_cli_verifies_real_review_chain(tmp_path: Path) -> None:
    base_plan_path = (
        ROOT
        / "analysis_outputs/public_research/pbp_visual_state_codex_review_v1/plan.json"
    )
    base_corrections_path = (
        ROOT
        / "analysis_outputs/public_research/pbp_visual_state_codex_review_v1/"
        "label_corrections.json"
    )
    followup_plan_path = (
        ROOT
        / "analysis_outputs/public_research/pbp_visual_state_codex_followup_v2/"
        "plan.json"
    )
    followup_corrections_path = (
        ROOT
        / "analysis_outputs/public_research/pbp_visual_state_codex_followup_v2/"
        "label_corrections.json"
    )
    plan = json.loads(base_plan_path.read_text())
    base = json.loads(base_corrections_path.read_text())
    followup = json.loads(followup_corrections_path.read_text())
    decisions = {
        (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        ): row["corrected_state"]
        for row in base["decisions"]
    }
    decisions.update(
        {
            (
                row["source_video_sha256"],
                row["candidate_bundle_sha256"],
                row["event_id"],
            ): row["corrected_state"]
            for row in followup["decisions"]
        }
    )

    paths = []
    for source_index, source in enumerate(plan["source_video_sha256s"]):
        source_rows = [
            row for row in plan["examples"] if row["source_video_sha256"] == source
        ]
        bundle = source_rows[0]["candidate_bundle_sha256"]
        examples = []
        for row in source_rows:
            key = (source, bundle, row["event_id"])
            label = decisions[key]
            sign = 2.0 if label == "free_throw" else -2.0
            examples.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "source_video_sha256",
                            "source_video_filename",
                            "candidate_bundle_sha256",
                            "event_id",
                            "anchor_frame",
                            "source_fps",
                            "frame_count",
                        )
                    },
                    "segments": [
                        {
                            "role": role,
                            "available": True,
                            "start_offset_seconds": float(role_index - 2),
                            "end_offset_seconds": float(role_index - 1),
                            "frame_indexes": list(range(16)),
                            "embedding": [sign + role_index * 0.01] * 4,
                        }
                        for role_index, role in enumerate(
                            ("previous", "anchor", "next")
                        )
                    ],
                }
            )
        artifact = seal_cut_aligned_video_embedding_artifact(
            {
                "purpose": "cut_aligned_visual_state_training_only",
                "runtime_consumable": False,
                "codex_runtime_answer_used": False,
                "transition_semantics_used": False,
                "truth_used_for_training_only": False,
                "review_plan_sha256": plan["artifact_sha256"],
                "transition_artifact_sha256": str(source_index + 5) * 64,
                "raw_video_sha256": source,
                "candidate_bundle_sha256": bundle,
                "backbone": "test-backbone",
                "backbone_sha256": "2" * 64,
                "embedding_dimension": 4,
                "clip_frames": 16,
                "expected_event_ids": sorted(row["event_id"] for row in source_rows),
                "complete": True,
                "examples": examples,
            }
        )
        path = tmp_path / f"embedding-{source_index}.json"
        path.write_text(json.dumps(artifact))
        paths.append(path)

    result = run_screen(
        embedding_paths=paths,
        base_plan_path=base_plan_path,
        base_corrections_path=base_corrections_path,
        followup_plan_path=followup_plan_path,
        followup_corrections_path=followup_corrections_path,
        output_path=tmp_path / "screen.json",
    )

    assert result["target_example_count"] == 108
    assert result["accepted"] is True
