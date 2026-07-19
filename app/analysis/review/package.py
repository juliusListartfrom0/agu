from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

import cv2

from app.analysis.box_score import EventLedger
from app.analysis.official_evaluation import seal_raw_only_predictions, verify_raw_only_bundle
from app.analysis.schemas import (
    BoxScoreReconciliationIssueResponse,
    GameEventResponse,
    OfficialBoxScoreResponse,
    RawOnlyPredictionBundleResponse,
    ReviewDecisionResponse,
)


class ReviewPackageError(ValueError):
    pass


def build_review_package(
    bundle: RawOnlyPredictionBundleResponse | Mapping[str, object],
    *,
    raw_video_paths: Iterable[str | Path],
    output_dir: str | Path,
    fps_by_video_id: Mapping[str, float] | None = None,
    materialize_media: bool = True,
    review_sample_fps: float = 6.0,
    review_sheet_columns: int = 6,
    review_sheet_max_frames: int = 72,
) -> dict[str, object]:
    """Create a bounded candidate review package from raw videos only."""

    if review_sample_fps <= 0 or review_sheet_columns <= 0 or review_sheet_max_frames <= 0:
        raise ReviewPackageError("review sheet sampling settings must be positive")

    sealed = verify_raw_only_bundle(bundle)
    raw_by_name = {Path(path).name: Path(path) for path in raw_video_paths}
    raw_by_id: dict[str, Path] = {}
    for asset in sealed.raw_videos:
        path = raw_by_name.get(asset.filename)
        if path is None or not path.is_file():
            raise ReviewPackageError(f"missing declared raw video: {asset.filename}")
        if path.stat().st_size != asset.size_bytes or _file_sha256(path) != asset.sha256:
            raise ReviewPackageError(f"raw video hash mismatch: {asset.filename}")
        raw_by_id[asset.video_id] = path

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    review_events = [event for event in sealed.events if event.status in {"candidate", "needs_review"}]
    candidate_lines: list[str] = []
    for event in review_events:
        candidate_lines.append(json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
        if materialize_media:
            _materialize_event_media(
                event,
                raw_by_id[event.source_video_id],
                destination / "events" / event.event_id,
                float((fps_by_video_id or {}).get(event.source_video_id, 30.0)),
                review_sample_fps=review_sample_fps,
                sheet_columns=review_sheet_columns,
                sheet_max_frames=review_sheet_max_frames,
            )
    (destination / "candidates.jsonl").write_text(
        "\n".join(candidate_lines) + ("\n" if candidate_lines else ""), encoding="utf-8"
    )
    (destination / "codex_decisions.jsonl").touch(exist_ok=True)
    manifest = {
        "schema_version": "agu.review-package.v1",
        "game_id": sealed.game_id,
        "prediction_bundle_sha256": sealed.bundle_sha256,
        "raw_video_assets": [asset.model_dump(mode="json") for asset in sealed.raw_videos],
        "candidate_count": len(review_events),
        "instructions": [
            "Review only the supplied raw-video evidence window.",
            "Use unknown/needs_review when actor, outcome, shot value or causal relation is not visible.",
            "Use an add decision when fine evidence contains an extra event absent from the coarse candidates.",
            "Append ReviewDecisionResponse JSON rows; never edit candidates.jsonl.",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def apply_review_decisions(
    events: Iterable[GameEventResponse],
    decisions: Iterable[ReviewDecisionResponse],
    *,
    expected_input_sha256: str | None = None,
) -> EventLedger:
    ledger = EventLedger(events)
    applied_decision_ids: set[str] = set()
    for decision in decisions:
        if expected_input_sha256 is not None and decision.input_sha256 != expected_input_sha256:
            raise ReviewPackageError(f"review decision input hash mismatch: {decision.decision_id}")
        if decision.decision_id in applied_decision_ids:
            continue
        ledger.apply_decision(decision)
        applied_decision_ids.add(decision.decision_id)
    return ledger


def finalize_reviewed_bundle(
    bundle: RawOnlyPredictionBundleResponse | Mapping[str, object],
    *,
    decisions: Iterable[ReviewDecisionResponse],
    raw_video_paths: Iterable[str | Path],
    config: Mapping[str, object],
    expected_team_points: Mapping[str, int] | None = None,
) -> tuple[RawOnlyPredictionBundleResponse, EventLedger, OfficialBoxScoreResponse]:
    """Apply hash-bound review revisions and reseal latest accepted state."""

    sealed = verify_raw_only_bundle(bundle)
    raw_by_name = {Path(path).name: Path(path) for path in raw_video_paths}
    ordered_paths: list[Path] = []
    for asset in sealed.raw_videos:
        path = raw_by_name.get(asset.filename)
        if path is None or not path.is_file():
            raise ReviewPackageError(f"missing declared raw video: {asset.filename}")
        if path.stat().st_size != asset.size_bytes or _file_sha256(path) != asset.sha256:
            raise ReviewPackageError(f"raw video hash mismatch: {asset.filename}")
        ordered_paths.append(path)

    ledger = apply_review_decisions(
        sealed.events,
        decisions,
        expected_input_sha256=sealed.bundle_sha256,
    )
    reviewed = seal_raw_only_predictions(
        game_id=sealed.game_id,
        raw_video_paths=ordered_paths,
        events=ledger.latest_events(),
        config=config,
        model_provenance={
            **sealed.model_provenance,
            "reviewed_from_bundle_sha256": sealed.bundle_sha256,
        },
    )
    score = ledger.aggregate(expected_team_points=expected_team_points)
    if sealed.model_provenance.get("review_complete_video_coverage") == "false":
        issue = BoxScoreReconciliationIssueResponse(
            code="incomplete_video_coverage",
            severity="error",
            message="Reviewed bundle derives from a dense review that did not cover every raw-video window",
        )
        score = score.model_copy(
            update={
                "status": "needs_review",
                "reconciliation": score.reconciliation.model_copy(
                    update={"valid": False, "issues": [*score.reconciliation.issues, issue]}
                ),
            }
        )
    return reviewed, ledger, score


def _materialize_event_media(
    event: GameEventResponse,
    video_path: Path,
    output_dir: Path,
    fps: float,
    *,
    review_sample_fps: float,
    sheet_columns: int,
    sheet_max_frames: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_sec = max(0.0, event.start_frame / fps - 2.0)
    end_sec = max(start_sec + 0.5, event.end_frame / fps + 2.0)
    duration = end_sec - start_sec
    clip_path = output_dir / "clip.mp4"
    sheet_path = output_dir / "contact_sheet.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-an",
            "-vf",
            "scale=960:-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-y",
            str(clip_path),
        ],
        check=True,
    )
    _write_timestamped_contact_sheet(
        video_path=video_path,
        path=sheet_path,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_fps=review_sample_fps,
        columns=sheet_columns,
        max_frames=sheet_max_frames,
    )


def _write_timestamped_contact_sheet(
    *,
    video_path: Path,
    path: Path,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    columns: int,
    max_frames: int,
) -> None:
    duration = end_sec - start_sec
    sample_count = min(max_frames, max(2, int(math.ceil(duration * sample_fps))))
    sample_step = duration / sample_count
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ReviewPackageError(f"unable to open raw video for review sheet: {video_path}")
    frames = []
    try:
        for index in range(sample_count):
            sample_time = min(end_sec, start_sec + (index + 0.5) * sample_step)
            capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            panel = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (142, 26), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"{sample_time:.2f}s",
                (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(panel)
    finally:
        capture.release()
    if not frames:
        raise ReviewPackageError(f"no readable frames for event contact sheet: {path.parent.name}")
    rows = math.ceil(len(frames) / columns)
    blank = frames[0] * 0
    frames.extend([blank] * (rows * columns - len(frames)))
    sheet_rows = [cv2.hconcat(frames[index : index + columns]) for index in range(0, len(frames), columns)]
    sheet = cv2.vconcat(sheet_rows)
    if not cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise ReviewPackageError(f"failed to write event contact sheet: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
