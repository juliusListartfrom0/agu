"""Offline integrity audit for the NBA Games metadata/PBP mirror."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

NBA_GAMES_LOCAL_MIRROR_SCHEMA = "agu.nba-games-local-mirror-audit.v1"


def build_nba_games_local_mirror_audit(
    index_path: str | Path,
    mirror_root: str | Path,
    *,
    generated_on: str,
) -> dict[str, Any]:
    """Verify that the local metadata mirror is internally self-consistent.

    This intentionally audits metadata and official PBP files only.  It never
    treats linked YouTube media or an action-feed timestamp as frame truth.
    """

    index = Path(index_path)
    root = Path(mirror_root)
    if not index.is_file():
        raise ValueError("NBA Games index is missing")
    if not root.is_dir():
        raise ValueError("NBA Games mirror root is missing")
    index_rows = _read_jsonl(index, "NBA Games index")
    if not index_rows:
        raise ValueError("NBA Games index is empty")
    index_ids = [str(row.get("id") or "").strip() for row in index_rows]
    if any(not value for value in index_ids) or len(set(index_ids)) != len(index_ids):
        raise ValueError("NBA Games index contains invalid or duplicate source ids")

    game_dirs = sorted(path for path in (root / "games").glob("*") if path.is_dir())
    if not game_dirs:
        raise ValueError("NBA Games mirror has no game directories")
    metadata_ids: list[str] = []
    pbp_rows = 0
    games_with_pbp = 0
    games_without_pbp = 0
    box_score_rows = 0
    official_game_ids: set[str] = set()
    for game_dir in game_dirs:
        metadata_path = game_dir / "metadata.json"
        box_path = game_dir / "box-score.jsonl"
        pbp_path = game_dir / "play-by-play.jsonl"
        if not metadata_path.is_file() or not box_path.is_file() or not pbp_path.is_file():
            raise ValueError(f"NBA Games game directory is incomplete: {game_dir.name}")
        metadata = _read_json(metadata_path, "NBA Games metadata")
        source_game = _mapping(metadata.get("source_game"))
        official_game = _mapping(metadata.get("official_game"))
        source_id = str(source_game.get("id") or "").strip()
        official_id = str(official_game.get("game_id") or "").strip()
        if not source_id or not official_id:
            raise ValueError(f"NBA Games metadata linkage is incomplete: {game_dir.name}")
        metadata_ids.append(source_id)
        official_game_ids.add(official_id)
        box_score_rows += len(_read_jsonl(box_path, "NBA Games box score"))
        rows = _read_jsonl(pbp_path, "NBA Games play by play")
        pbp_rows += len(rows)
        if rows:
            games_with_pbp += 1
        else:
            games_without_pbp += 1
    if set(index_ids) != set(metadata_ids):
        raise ValueError("NBA Games index and metadata source ids do not match")
    video_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov", ".ogv"}
    ]
    artifact: dict[str, Any] = {
        "schema_version": NBA_GAMES_LOCAL_MIRROR_SCHEMA,
        "generated_on": str(generated_on),
        "purpose": "offline_metadata_and_official_pbp_mirror_integrity_audit",
        "runtime_consumable": False,
        "training_media_eligible": False,
        "codex_runtime_answer_used": False,
        "index_path": str(index),
        "mirror_root": str(root),
        "index_sha256": _file_sha256(index),
        "index_game_count": len(index_rows),
        "metadata_game_count": len(game_dirs),
        "unique_source_ids": len(set(metadata_ids)),
        "unique_official_game_ids": len(official_game_ids),
        "box_score_rows": box_score_rows,
        "play_by_play_rows": pbp_rows,
        "games_with_nonempty_play_by_play": games_with_pbp,
        "games_with_empty_play_by_play": games_without_pbp,
        "video_file_count": len(video_files),
        "video_payload_bytes": sum(path.stat().st_size for path in video_files),
        "causal_label_contract": {
            "official_pbp_is_frame_truth": False,
            "subsecond_ball_hand_rim_outcome_labels_available": False,
            "exhaustive_non_shot_hard_negatives_available": False,
        },
        "decision": "retain_metadata_pbp_only; no_training_or_runtime_promotion",
    }
    artifact["audit_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_nba_games_local_mirror_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("audit_sha256", ""))
    if artifact.get("schema_version") != NBA_GAMES_LOCAL_MIRROR_SCHEMA:
        raise ValueError("unsupported NBA Games local mirror audit schema")
    if artifact.get("runtime_consumable") is not False or artifact.get("training_media_eligible") is not False:
        raise ValueError("NBA Games local mirror audit must remain offline")
    if artifact.get("video_file_count") != 0 or artifact.get("video_payload_bytes") != 0:
        raise ValueError("local mirror audit may not include video payload")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("NBA Games local mirror audit hash mismatch")
    artifact["audit_sha256"] = claimed
    return artifact


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("NBA Games metadata linkage is not an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
