#!/usr/bin/env python3
"""Update only the hash-bound v23/cache pointers in existing gate audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/public_research"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(payload: dict) -> str:
    value = dict(payload)
    value.pop("audit_sha256", None)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def update_gate() -> tuple[Path, str]:
    path = OUT / "continuous_causal_source_gate_2026-08-10.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    batch = OUT / "wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v23.json"
    root = OUT / "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23"
    sealed = root / "review_sealed.json"
    pilot = root / "pilot_manifest.json"
    plan = root / "review_plan.json"
    decisions = root / "review_decisions.json"
    spec = root / "review_spec.json"
    retention = root / "retention_manifest.json"
    frames = root / "raw_frames_manifest.json"
    sheet_cleanup = OUT / "agu_v23_contact_sheet_cleanup_2026-08-12.json"
    cache_cleanup = OUT / "agu_v23_regression_cache_cleanup_2026-08-12.json"
    p = payload
    p.update(
        {
            "latest_pilot_annotation_v23_batch": str(batch.relative_to(ROOT)),
            "latest_pilot_annotation_v23_batch_sha256": json.loads(batch.read_text())[
                "artifact_sha256"
            ],
            "latest_pilot_annotation_v23_batch_file_sha256": file_sha(batch),
            "latest_pilot_annotation_v23_batch_windows": 15,
            "latest_pilot_annotation_v23_batch_full_materialization": True,
            "latest_pilot_annotation_v23": {
                "artifact": str(sealed.relative_to(ROOT)),
                "artifact_sha256": json.loads(sealed.read_text())["artifact_sha256"],
                "artifact_file_sha256": file_sha(sealed),
                "pilot_manifest": str(pilot.relative_to(ROOT)),
                "pilot_manifest_sha256": json.loads(pilot.read_text())["manifest_sha256"],
                "pilot_manifest_file_sha256": file_sha(pilot),
                "plan": str(plan.relative_to(ROOT)),
                "plan_sha256": json.loads(plan.read_text())["artifact_sha256"],
                "plan_file_sha256": file_sha(plan),
                "review_decisions": str(decisions.relative_to(ROOT)),
                "review_decisions_file_sha256": file_sha(decisions),
                "review_spec": str(spec.relative_to(ROOT)),
                "review_spec_file_sha256": file_sha(spec),
                "retention_manifest": str(retention.relative_to(ROOT)),
                "retention_manifest_sha256": json.loads(retention.read_text())["artifact_sha256"],
                "retention_manifest_file_sha256": file_sha(retention),
                "frame_manifest": str(frames.relative_to(ROOT)),
                "frame_manifest_sha256": json.loads(frames.read_text())["artifact_sha256"],
                "frame_manifest_file_sha256": file_sha(frames),
                "batch": str(batch.relative_to(ROOT)),
                "batch_sha256": json.loads(batch.read_text())["artifact_sha256"],
                "batch_file_sha256": file_sha(batch),
                "windows": 15,
                "retained_frames": 315,
                "label_counts": {"not_a_shot": 14, "shot": 1},
                "outcome_counts": {"not_applicable": 14, "missed": 1},
                "exclusion_radius_seconds": 15.0,
                "pilot_only": True,
                "training_consumable": False,
                "runtime_consumable": False,
                "promotion_eligible": False,
                "exhaustive_non_shot_hard_negatives": False,
                "full_materialization": True,
                "materialization_method": "single_seek_sequential_decode_per_review_window",
                "contact_sheets_retained": False,
                "raw_frames_retained": True,
            },
            "latest_v23_contact_sheet_cleanup": str(sheet_cleanup.relative_to(ROOT)),
            "latest_v23_contact_sheet_cleanup_sha256": json.loads(sheet_cleanup.read_text())["audit_sha256"],
            "latest_v23_contact_sheet_cleanup_file_sha256": file_sha(sheet_cleanup),
            "latest_v23_contact_sheet_cleanup_files_deleted": 15,
            "latest_v23_contact_sheet_cleanup_bytes_deleted": 5389158,
            "latest_v23_regression_cache_cleanup": str(cache_cleanup.relative_to(ROOT)),
            "latest_v23_regression_cache_cleanup_sha256": json.loads(cache_cleanup.read_text())["audit_sha256"],
            "latest_v23_regression_cache_cleanup_file_sha256": file_sha(cache_cleanup),
            "latest_v23_regression_cache_cleanup_files_deleted": 133,
            "latest_v23_regression_cache_cleanup_bytes_deleted": 3002360,
            "latest_pilot_annotation_latest": "v23",
            "latest_pilot_annotation_latest_summary": "v23 additive pilot; aggregate v8-v23 totals recorded in docs/current-solution.md",
            "latest_pilot_annotation_latest_windows": 525,
            "latest_pilot_annotation_latest_retained_frames": 11025,
            "latest_pilot_annotation_latest_label_counts": {
                "not_a_shot": 414,
                "uncertain": 83,
                "shot": 28,
            },
        }
    )
    p["generated_on"] = "2026-08-12"
    p["audit_sha256"] = canonical_sha(p)
    path.write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, p["audit_sha256"]


def update_readiness(gate_sha: str) -> tuple[Path, str]:
    path = OUT / "agu_readiness_audit_2026-08-09.json"
    p = json.loads(path.read_text(encoding="utf-8"))
    p["generated_on"] = "2026-08-12"
    p["blocker"]["pilot_annotation_status"] = (
        "v23 latest source-disjoint held-out extension: 315 retained frames / 15 bounded windows across "
        "HCTV (Hazen), HCTV (Randolph) and VTV; one explicit missed shot and fourteen explicit not_a_shot pilots; "
        "v8–v23 combined pilot extensions remain non-exhaustive, independently sealed, and offline-only"
    )
    p["blocker"]["pilot_annotation_latest_artifact"] = "analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23/review_sealed.json"
    p["blocker"]["pilot_annotation_latest_artifact_sha256"] = json.loads((OUT / "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23/review_sealed.json").read_text())["artifact_sha256"]
    p["blocker"]["pilot_annotation_latest_manifest"] = "analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23/pilot_manifest.json"
    p["blocker"]["pilot_annotation_latest_manifest_sha256"] = json.loads((OUT / "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23/pilot_manifest.json").read_text())["manifest_sha256"]
    p["blocker"]["pilot_annotation_latest_windows"] = 525
    p["blocker"]["pilot_annotation_latest_retained_frames"] = 11025
    p["blocker"]["pilot_annotation_latest_batch"] = "analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v23.json"
    p["blocker"]["pilot_annotation_latest_batch_sha256"] = json.loads((OUT / "wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v23.json").read_text())["artifact_sha256"]
    p["source_gate"]["audit_sha256"] = gate_sha
    p["source_gate"]["artifact"] = "analysis_outputs/public_research/continuous_causal_source_gate_2026-08-10.json"
    p["latest_pilot_annotation_v23"] = {
        "artifact": "analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23/review_sealed.json",
        "artifact_sha256": p["blocker"]["pilot_annotation_latest_artifact_sha256"],
        "manifest": "analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23/pilot_manifest.json",
        "manifest_sha256": p["blocker"]["pilot_annotation_latest_manifest_sha256"],
        "windows": 15,
        "retained_frames": 315,
        "label_counts": {"not_a_shot": 14, "shot": 1},
        "pilot_only": True,
        "training_consumable": False,
        "runtime_consumable": False,
    }
    p["latest_pilot_annotation_v23_file_sha256"] = file_sha(OUT / "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23/review_sealed.json")
    p["latest_pilot_annotation_v23_manifest_file_sha256"] = file_sha(OUT / "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v23/pilot_manifest.json")
    p.pop("audit_sha256", None)
    p["audit_sha256"] = canonical_sha(p)
    path.write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, p["audit_sha256"]


if __name__ == "__main__":
    gate_path, gate_sha = update_gate()
    readiness_path, readiness_sha = update_readiness(gate_sha)
    print(json.dumps({"gate": str(gate_path), "gate_sha": gate_sha, "readiness": str(readiness_path), "readiness_sha": readiness_sha}, sort_keys=True))
