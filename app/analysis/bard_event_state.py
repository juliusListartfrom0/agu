"""Training-only BARD event-state subset contracts.

Official source:
https://github.com/GabrieleGiudic/BARD

The upstream repository declares CC BY 4.0 for its code and annotations. NBA
event clips are externally hosted media and are therefore retained only for
local research; this module never makes them runtime-consumable.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

BARD_EVENT_STATE_PLAN_SCHEMA = "agu.bard-event-state-plan.v1"
BARD_VIDEO_RESOLUTION_SCHEMA = "agu.bard-video-resolution.v1"
BARD_EVENT_STATE_SUBSET_SCHEMA = "agu.bard-event-state-subset.v1"
BARD_EMBEDDED_VIDEO_PLAN_SCHEMA = "agu.bard-embedded-video-plan.v1"
BARD_EMBEDDED_SUBSET_SCHEMA = "agu.bard-embedded-video-subset.v1"
BARD_SOURCE_URL = "https://github.com/GabrieleGiudic/BARD"
BARD_DECLARED_LICENSE = "CC-BY-4.0"
BARD_TARGET_STATES = ("live_field_goal", "free_throw", "foul_only")
SEALED_BLIND_GAME_IDS = frozenset({"0049400070", "0049400071"})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_GAME_ID = re.compile(r"\d{10}")
_EVENT_ID = re.compile(r"\d+")
_EMBEDDED_NAME = re.compile(
    r".*-(?P<game_id>\d{10})_(?P<event_id>\d+)\.mp4"
)
_ALLOWED_ACTIONS = frozenset(
    {
        "2pt shot",
        "3pt shot",
        "free throw",
        "foul",
        "rebound",
        "turnover",
        "steal",
        "block",
        "violation",
    }
)


@dataclass(frozen=True, order=True)
class BardEventStateRow:
    clip_id: str
    page_url: str
    game_id: str
    event_id: str
    season: str
    event_state: str
    shot_outcome: str | None
    actions: tuple[tuple[str, str | bool | None], ...]


@dataclass(frozen=True, order=True)
class BardEmbeddedStateRow:
    clip_id: str
    repository_path: str
    game_id: str
    event_id: str
    year: int
    event_state: str
    shot_outcome: str | None
    actions: tuple[tuple[str, str | bool | None], ...]


def parse_bard_dataset_csv(payload: str) -> list[BardEventStateRow]:
    """Parse the official semicolon CSV without executing annotation text."""

    reader = csv.DictReader(io.StringIO(payload), delimiter=";")
    if reader.fieldnames != ["urls", "actions", "numerosity"]:
        raise ValueError("BARD CSV columns are invalid")
    rows: list[BardEventStateRow] = []
    identities: set[str] = set()
    for source in reader:
        page_url = str(source.get("urls") or "").strip()
        game_id, event_id, season = _parse_event_page_url(page_url)
        if game_id in SEALED_BLIND_GAME_IDS:
            raise ValueError("BARD CSV contains a sealed blind game")
        actions = _parse_actions(str(source.get("actions") or ""))
        state, shot_outcome = _target_state(actions)
        if state is None:
            continue
        clip_id = f"{game_id}-{event_id}"
        if clip_id in identities:
            raise ValueError("BARD CSV contains duplicate event identities")
        identities.add(clip_id)
        rows.append(
            BardEventStateRow(
                clip_id=clip_id,
                page_url=page_url,
                game_id=game_id,
                event_id=event_id,
                season=season,
                event_state=state,
                shot_outcome=shot_outcome,
                actions=tuple(
                    tuple(sorted(action.items()))
                    for action in actions
                ),
            )
        )
    if not rows:
        raise ValueError("BARD CSV contains no target event-state rows")
    return rows


def parse_bard_benchmark_csv(
    payload: str,
    *,
    year: int,
) -> list[BardEmbeddedStateRow]:
    """Parse one official embedded-validation benchmark."""

    if year not in {2024, 2025}:
        raise ValueError("BARD embedded benchmark year is unsupported")
    reader = csv.DictReader(io.StringIO(payload))
    if set(reader.fieldnames or ()) != {
        "files",
        "actions_name",
        "number_actions",
    }:
        raise ValueError("BARD embedded benchmark columns are invalid")
    rows = []
    identities: set[str] = set()
    expected_prefix = f"validation/{year}/multi/"
    for source in reader:
        raw_path = str(source.get("files") or "").strip()
        while raw_path.startswith("../"):
            raw_path = raw_path[3:]
        path = Path(raw_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not raw_path.startswith(expected_prefix)
        ):
            raise ValueError("BARD embedded benchmark path is unsafe")
        match = _EMBEDDED_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError("BARD embedded clip identity is invalid")
        game_id = match.group("game_id")
        event_id = match.group("event_id")
        if game_id in SEALED_BLIND_GAME_IDS:
            raise ValueError("BARD embedded benchmark contains a blind game")
        actions = _parse_actions(str(source.get("actions_name") or ""))
        state, shot_outcome = _target_state(actions)
        if state not in {"live_field_goal", "free_throw"}:
            continue
        clip_id = path.stem
        if clip_id in identities:
            raise ValueError("BARD embedded benchmark contains duplicate clips")
        identities.add(clip_id)
        rows.append(
            BardEmbeddedStateRow(
                clip_id=clip_id,
                repository_path=raw_path,
                game_id=game_id,
                event_id=event_id,
                year=year,
                event_state=state,
                shot_outcome=shot_outcome,
                actions=tuple(
                    tuple(sorted(action.items()))
                    for action in actions
                ),
            )
        )
    if not rows:
        raise ValueError("BARD embedded benchmark has no target clips")
    return rows


def select_bard_embedded_state_rows(
    rows: Iterable[BardEmbeddedStateRow],
    *,
    per_class: int,
    seed: int,
) -> list[BardEmbeddedStateRow]:
    """Select field-goal/free-throw clips round-robin across games."""

    if per_class <= 0:
        raise ValueError("BARD embedded per_class must be positive")
    by_state: dict[str, dict[str, list[BardEmbeddedStateRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if (
            row.game_id in SEALED_BLIND_GAME_IDS
            or row.event_state not in {"live_field_goal", "free_throw"}
        ):
            raise ValueError("BARD embedded selection is invalid or blind")
        by_state[row.event_state][row.game_id].append(row)
    selected: list[BardEmbeddedStateRow] = []
    for state in ("live_field_goal", "free_throw"):
        games = by_state.get(state) or {}
        ordered_games = sorted(
            games,
            key=lambda game_id: _stable_key(seed, state, game_id),
        )
        for game_id in ordered_games:
            games[game_id].sort(
                key=lambda row: _stable_key(seed, row.clip_id)
            )
        state_rows: list[BardEmbeddedStateRow] = []
        cursor = 0
        while len(state_rows) < per_class:
            added = False
            for game_id in ordered_games:
                if cursor < len(games[game_id]):
                    state_rows.append(games[game_id][cursor])
                    added = True
                    if len(state_rows) == per_class:
                        break
            if not added:
                raise ValueError(
                    f"BARD embedded benchmark has fewer than {per_class} "
                    f"examples for {state}"
                )
            cursor += 1
        selected.extend(state_rows)
    return sorted(selected)


def build_bard_embedded_video_plan(
    rows: Sequence[BardEmbeddedStateRow],
    *,
    tree_entries: Mapping[str, Mapping[str, Any]],
    source_revision: str,
    benchmark_sha256s: Mapping[str, str],
    per_class: int,
    seed: int,
) -> dict[str, Any]:
    """Bind a balanced benchmark slice to exact Git blobs."""

    if not _GIT_REVISION.fullmatch(source_revision):
        raise ValueError("BARD embedded source revision is invalid")
    selected = select_bard_embedded_state_rows(
        rows,
        per_class=per_class,
        seed=seed,
    )
    years = sorted({row.year for row in selected})
    for year in years:
        _require_sha256(
            benchmark_sha256s.get(str(year)),
            f"BARD {year} benchmark",
        )
    examples = []
    for row in selected:
        tree = tree_entries.get(row.repository_path)
        if not isinstance(tree, Mapping):
            raise ValueError("BARD embedded clip is missing from the Git tree")
        blob_sha = str(tree.get("sha") or "")
        size = int(tree.get("size") or 0)
        if not _GIT_REVISION.fullmatch(blob_sha) or size <= 0:
            raise ValueError("BARD embedded Git tree identity is invalid")
        examples.append(
            {
                "clip_id": row.clip_id,
                "repository_path": row.repository_path,
                "download_url": (
                    "https://raw.githubusercontent.com/GabrieleGiudic/BARD/"
                    f"{source_revision}/{row.repository_path}"
                ),
                "game_id": row.game_id,
                "event_id": row.event_id,
                "year": row.year,
                "event_state": row.event_state,
                "shot_outcome": row.shot_outcome,
                "actions": [dict(action) for action in row.actions],
                "git_blob_sha1": blob_sha,
                "size_bytes": size,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": BARD_EMBEDDED_VIDEO_PLAN_SCHEMA,
        "purpose": "event_state_pretraining_only",
        "runtime_consumable": False,
        "labels_exposed_to_runtime": False,
        "codex_runtime_answer_used": False,
        "source": {
            "dataset_id": "bard-2026-embedded-validation",
            "url": BARD_SOURCE_URL,
            "revision": source_revision,
            "declared_license": BARD_DECLARED_LICENSE,
            "benchmark_sha256s": {
                str(year): benchmark_sha256s[str(year)]
                for year in years
            },
            "underlying_media_rights": (
                "repository_embedded_nba_media_local_research_only"
            ),
        },
        "blind_game_ids_excluded": sorted(SEALED_BLIND_GAME_IDS),
        "selection": {
            "strategy": "binary_state_balanced_game_round_robin_v1",
            "per_class": per_class,
            "seed": seed,
        },
        "examples": sorted(examples, key=lambda row: row["clip_id"]),
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    return _verify_bard_embedded_plan(payload)


def materialize_bard_embedded_subset(
    *,
    plan: Mapping[str, Any],
    output_dir: Path,
    fetch_video: Callable[[str], bytes],
) -> dict[str, Any]:
    """Download exact repository blobs and seal local file hashes."""

    verified = _verify_bard_embedded_plan(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    hashes: set[str] = set()
    for example in verified["examples"]:
        payload = bytes(fetch_video(str(example["download_url"])))
        _verify_mp4(payload)
        if len(payload) != int(example["size_bytes"]):
            raise ValueError("BARD embedded clip size differs from the Git tree")
        blob_sha = _git_blob_sha1(payload)
        if blob_sha != example["git_blob_sha1"]:
            raise ValueError("BARD embedded Git blob SHA-1 mismatch")
        sha256 = hashlib.sha256(payload).hexdigest()
        if sha256 in hashes:
            raise ValueError("BARD embedded subset contains duplicate media")
        hashes.add(sha256)
        relative = (
            Path(str(example["event_state"]))
            / f"{example['clip_id']}.mp4"
        )
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".mp4.part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        clips.append(
            {
                "clip_id": example["clip_id"],
                "repository_path": example["repository_path"],
                "game_id": example["game_id"],
                "event_id": example["event_id"],
                "year": example["year"],
                "event_state": example["event_state"],
                "shot_outcome": example["shot_outcome"],
                "relative_path": relative.as_posix(),
                "size_bytes": len(payload),
                "git_blob_sha1": blob_sha,
                "sha256": sha256,
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": BARD_EMBEDDED_SUBSET_SCHEMA,
        "purpose": "event_state_pretraining_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "plan_sha256": verified["plan_sha256"],
        "clips": sorted(clips, key=lambda row: row["clip_id"]),
    }
    artifact["subset_sha256"] = _canonical_sha256(artifact)
    return verify_bard_embedded_subset(artifact, root=output_dir)


def verify_bard_embedded_subset(
    value: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    artifact = json.loads(json.dumps(value))
    claimed = str(artifact.pop("subset_sha256", ""))
    if not _SHA256.fullmatch(claimed):
        raise ValueError("BARD embedded subset hash is invalid")
    if _canonical_sha256(artifact) != claimed:
        raise ValueError("BARD embedded subset hash mismatch")
    if (
        artifact.get("schema_version") != BARD_EMBEDDED_SUBSET_SCHEMA
        or artifact.get("purpose") != "event_state_pretraining_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("BARD embedded subset boundary is invalid")
    _require_sha256(artifact.get("plan_sha256"), "BARD embedded plan")
    clips = artifact.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("BARD embedded subset clips are missing")
    identities: set[str] = set()
    hashes: set[str] = set()
    for row in clips:
        clip_id = str(row.get("clip_id") or "")
        if (
            clip_id in identities
            or row.get("event_state")
            not in {"live_field_goal", "free_throw"}
            or row.get("game_id") in SEALED_BLIND_GAME_IDS
        ):
            raise ValueError("BARD embedded subset identity is invalid")
        identities.add(clip_id)
        relative = Path(str(row.get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("BARD embedded subset path is unsafe")
        payload = (root / relative).read_bytes()
        _verify_mp4(payload)
        sha256 = hashlib.sha256(payload).hexdigest()
        if (
            len(payload) != int(row.get("size_bytes") or -1)
            or sha256 != row.get("sha256")
            or _git_blob_sha1(payload) != row.get("git_blob_sha1")
        ):
            raise ValueError("BARD embedded subset file hash mismatch")
        if sha256 in hashes:
            raise ValueError("BARD embedded subset contains duplicate media")
        hashes.add(sha256)
    artifact["subset_sha256"] = claimed
    return artifact


def select_bard_event_state_rows(
    rows: Iterable[BardEventStateRow],
    *,
    per_class: int,
    seed: int,
) -> list[BardEventStateRow]:
    """Select each state round-robin across source games."""

    if per_class <= 0:
        raise ValueError("BARD per_class must be positive")
    by_state: dict[str, dict[str, list[BardEventStateRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.game_id in SEALED_BLIND_GAME_IDS:
            raise ValueError("BARD selection contains a sealed blind game")
        if row.event_state not in BARD_TARGET_STATES:
            raise ValueError("BARD selection contains an unsupported state")
        by_state[row.event_state][row.game_id].append(row)

    selected: list[BardEventStateRow] = []
    for state in BARD_TARGET_STATES:
        games = by_state.get(state) or {}
        if not games:
            raise ValueError(f"BARD has no examples for {state}")
        ordered_games = sorted(
            games,
            key=lambda game_id: _stable_key(seed, state, game_id),
        )
        for game_id in ordered_games:
            games[game_id].sort(
                key=lambda row: _stable_key(seed, row.clip_id)
            )
        state_rows: list[BardEventStateRow] = []
        cursor = 0
        while len(state_rows) < per_class:
            added = False
            for game_id in ordered_games:
                if cursor < len(games[game_id]):
                    state_rows.append(games[game_id][cursor])
                    added = True
                    if len(state_rows) == per_class:
                        break
            if not added:
                raise ValueError(
                    f"BARD has fewer than {per_class} examples for {state}"
                )
            cursor += 1
        selected.extend(state_rows)
    return sorted(selected)


def build_bard_event_state_plan(
    rows: Sequence[BardEventStateRow],
    *,
    source_revision: str,
    source_csv_sha256: str,
    per_class: int,
    seed: int,
) -> dict[str, Any]:
    """Seal a label-bearing training plan before resolving external media."""

    if not _GIT_REVISION.fullmatch(source_revision):
        raise ValueError("BARD source revision must be a full Git SHA")
    _require_sha256(source_csv_sha256, "BARD source CSV")
    selected = select_bard_event_state_rows(
        rows,
        per_class=per_class,
        seed=seed,
    )
    payload: dict[str, Any] = {
        "schema_version": BARD_EVENT_STATE_PLAN_SCHEMA,
        "purpose": "event_state_pretraining_only",
        "runtime_consumable": False,
        "labels_exposed_to_runtime": False,
        "codex_runtime_answer_used": False,
        "source": {
            "dataset_id": "bard-2026",
            "url": BARD_SOURCE_URL,
            "revision": source_revision,
            "declared_license": BARD_DECLARED_LICENSE,
            "source_csv_sha256": source_csv_sha256,
            "underlying_media_rights": (
                "external_nba_media_local_research_only_no_redistribution"
            ),
        },
        "blind_game_ids_excluded": sorted(SEALED_BLIND_GAME_IDS),
        "selection": {
            "strategy": "state_balanced_game_round_robin_v1",
            "per_class": per_class,
            "seed": seed,
        },
        "examples": [_row_payload(row) for row in selected],
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    return verify_bard_event_state_plan(payload)


def verify_bard_event_state_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = json.loads(json.dumps(value))
    claimed = str(plan.pop("plan_sha256", ""))
    if not _SHA256.fullmatch(claimed):
        raise ValueError("BARD plan hash is invalid")
    if _canonical_sha256(plan) != claimed:
        raise ValueError("BARD plan hash mismatch")
    if (
        plan.get("schema_version") != BARD_EVENT_STATE_PLAN_SCHEMA
        or plan.get("purpose") != "event_state_pretraining_only"
        or plan.get("runtime_consumable") is not False
        or plan.get("labels_exposed_to_runtime") is not False
        or plan.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("BARD plan boundary is invalid")
    source = plan.get("source") or {}
    if (
        source.get("dataset_id") != "bard-2026"
        or source.get("url") != BARD_SOURCE_URL
        or source.get("declared_license") != BARD_DECLARED_LICENSE
        or not _GIT_REVISION.fullmatch(str(source.get("revision") or ""))
    ):
        raise ValueError("BARD plan provenance is invalid")
    _require_sha256(source.get("source_csv_sha256"), "BARD source CSV")
    if plan.get("blind_game_ids_excluded") != sorted(SEALED_BLIND_GAME_IDS):
        raise ValueError("BARD plan blind exclusion is invalid")
    examples = plan.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("BARD plan examples are missing")
    clip_ids: set[str] = set()
    for row in examples:
        _verify_example(row)
        clip_id = str(row["clip_id"])
        if clip_id in clip_ids:
            raise ValueError("BARD plan has duplicate examples")
        clip_ids.add(clip_id)
    plan["plan_sha256"] = claimed
    return plan


def seal_bard_video_resolution_manifest(
    *,
    plan: Mapping[str, Any],
    video_urls: Mapping[str, str],
) -> dict[str, Any]:
    """Bind every planned NBA event page to one exact public MP4 URL."""

    verified_plan = verify_bard_event_state_plan(plan)
    examples = verified_plan["examples"]
    expected = {str(row["clip_id"]) for row in examples}
    if set(video_urls) != expected:
        raise ValueError("BARD video URLs must exactly match the plan")
    rows = []
    for example in examples:
        clip_id = str(example["clip_id"])
        video_url = str(video_urls[clip_id])
        _verify_video_url(
            video_url,
            game_id=str(example["game_id"]),
            event_id=str(example["event_id"]),
        )
        rows.append(
            {
                "clip_id": clip_id,
                "game_id": example["game_id"],
                "event_id": example["event_id"],
                "page_url": example["page_url"],
                "video_url": video_url,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": BARD_VIDEO_RESOLUTION_SCHEMA,
        "purpose": "local_research_media_resolution_only",
        "runtime_consumable": False,
        "plan_sha256": verified_plan["plan_sha256"],
        "clips": sorted(rows, key=lambda row: row["clip_id"]),
    }
    payload["resolution_sha256"] = _canonical_sha256(payload)
    return _verify_resolution_manifest(
        payload,
        plan=verified_plan,
    )


def materialize_bard_event_state_subset(
    *,
    plan: Mapping[str, Any],
    resolution_manifest: Mapping[str, Any],
    output_dir: Path,
    fetch_video: Callable[[str], bytes],
) -> dict[str, Any]:
    """Download, verify and seal one bounded local-research subset."""

    verified_plan = verify_bard_event_state_plan(plan)
    resolved = _verify_resolution_manifest(
        resolution_manifest,
        plan=verified_plan,
    )
    by_id = {row["clip_id"]: row for row in resolved["clips"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    media_hashes: set[str] = set()
    for example in verified_plan["examples"]:
        clip_id = str(example["clip_id"])
        source = by_id[clip_id]
        relative = Path(str(example["event_state"])) / f"{clip_id}.mp4"
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = bytes(fetch_video(str(source["video_url"])))
        _verify_mp4(payload)
        media_sha256 = hashlib.sha256(payload).hexdigest()
        if media_sha256 in media_hashes:
            raise ValueError(
                "BARD download contains duplicate media across distinct events"
            )
        media_hashes.add(media_sha256)
        temporary = target.with_suffix(".mp4.part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        clips.append(
            {
                "clip_id": clip_id,
                "game_id": example["game_id"],
                "event_id": example["event_id"],
                "event_state": example["event_state"],
                "shot_outcome": example["shot_outcome"],
                "relative_path": relative.as_posix(),
                "video_url": source["video_url"],
                "size_bytes": len(payload),
                "sha256": media_sha256,
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": BARD_EVENT_STATE_SUBSET_SCHEMA,
        "purpose": "event_state_pretraining_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "plan_sha256": verified_plan["plan_sha256"],
        "resolution_sha256": resolved["resolution_sha256"],
        "clips": sorted(clips, key=lambda row: row["clip_id"]),
    }
    artifact["subset_sha256"] = _canonical_sha256(artifact)
    return verify_bard_event_state_subset(artifact, root=output_dir)


def verify_bard_event_state_subset(
    value: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    artifact = json.loads(json.dumps(value))
    claimed = str(artifact.pop("subset_sha256", ""))
    if not _SHA256.fullmatch(claimed):
        raise ValueError("BARD subset hash is invalid")
    if _canonical_sha256(artifact) != claimed:
        raise ValueError("BARD subset hash mismatch")
    if (
        artifact.get("schema_version") != BARD_EVENT_STATE_SUBSET_SCHEMA
        or artifact.get("purpose") != "event_state_pretraining_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("BARD subset boundary is invalid")
    _require_sha256(artifact.get("plan_sha256"), "BARD plan")
    _require_sha256(artifact.get("resolution_sha256"), "BARD resolution")
    clips = artifact.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("BARD subset clips are missing")
    identities: set[str] = set()
    media_hashes: set[str] = set()
    for row in clips:
        clip_id = str(row.get("clip_id") or "")
        game_id = str(row.get("game_id") or "")
        if (
            clip_id in identities
            or game_id in SEALED_BLIND_GAME_IDS
            or row.get("event_state") not in BARD_TARGET_STATES
        ):
            raise ValueError("BARD subset clip identity is invalid")
        identities.add(clip_id)
        relative = Path(str(row.get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("BARD subset path is unsafe")
        target = root / relative
        payload = target.read_bytes()
        _verify_mp4(payload)
        media_sha256 = hashlib.sha256(payload).hexdigest()
        if (
            len(payload) != int(row.get("size_bytes") or -1)
            or media_sha256 != row.get("sha256")
        ):
            raise ValueError("BARD subset file hash mismatch")
        if media_sha256 in media_hashes:
            raise ValueError(
                "BARD subset contains duplicate media across distinct events"
            )
        media_hashes.add(media_sha256)
    artifact["subset_sha256"] = claimed
    return artifact


def _parse_event_page_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"www.nba.com", "nba.com"}
        or parsed.path.rstrip("/") != "/stats/events"
    ):
        raise ValueError("BARD event page URL is invalid")
    query = parse_qs(parsed.query)
    game_id = str((query.get("GameID") or [""])[0])
    event_id = str((query.get("GameEventID") or [""])[0])
    season = str((query.get("Season") or [""])[0])
    if (
        not _GAME_ID.fullmatch(game_id)
        or not _EVENT_ID.fullmatch(event_id)
        or not season
    ):
        raise ValueError("BARD event page identity is invalid")
    return game_id, event_id, season


def _parse_actions(raw: str) -> list[dict[str, str | bool | None]]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("BARD actions are not a safe literal") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("BARD actions must be a non-empty list")
    actions = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("BARD actions contain an invalid item")
        action = str(row.get("action") or "").strip().lower()
        if action not in _ALLOWED_ACTIONS:
            raise ValueError("BARD actions contain an unsupported label")
        result = row.get("result")
        if result not in {True, False, None}:
            raise ValueError("BARD actions contain an invalid result")
        actions.append(
            {
                "player": str(row.get("player") or ""),
                "action": action,
                "result": result,
                "assisted": row.get("assisted"),
                "other_player": (
                    None
                    if row.get("other_player") is None
                    else str(row.get("other_player"))
                ),
                "color": str(row.get("color") or ""),
            }
        )
    return actions


def _target_state(
    actions: Sequence[Mapping[str, str | bool | None]],
) -> tuple[str | None, str | None]:
    names = {str(row["action"]) for row in actions}
    has_field_goal = bool(names & {"2pt shot", "3pt shot"})
    has_free_throw = "free throw" in names
    if has_field_goal and has_free_throw:
        return None, None
    if has_field_goal:
        target = next(
            row
            for row in actions
            if row["action"] in {"2pt shot", "3pt shot"}
        )
        return "live_field_goal", _outcome(target.get("result"))
    if has_free_throw:
        target = next(row for row in actions if row["action"] == "free throw")
        return "free_throw", _outcome(target.get("result"))
    if "foul" in names:
        return "foul_only", None
    return None, None


def _outcome(value: str | bool | None) -> str:
    if value is True:
        return "made"
    if value is False:
        return "missed"
    raise ValueError("BARD shot action has no binary outcome")


def _row_payload(row: BardEventStateRow) -> dict[str, Any]:
    return {
        "clip_id": row.clip_id,
        "page_url": row.page_url,
        "game_id": row.game_id,
        "event_id": row.event_id,
        "season": row.season,
        "event_state": row.event_state,
        "shot_outcome": row.shot_outcome,
        "actions": [dict(action) for action in row.actions],
    }


def _verify_example(row: Mapping[str, Any]) -> None:
    game_id = str(row.get("game_id") or "")
    event_id = str(row.get("event_id") or "")
    if (
        not _GAME_ID.fullmatch(game_id)
        or game_id in SEALED_BLIND_GAME_IDS
        or not _EVENT_ID.fullmatch(event_id)
        or row.get("clip_id") != f"{game_id}-{event_id}"
        or row.get("event_state") not in BARD_TARGET_STATES
    ):
        raise ValueError("BARD plan example identity is invalid or blind")
    parsed_game, parsed_event, season = _parse_event_page_url(
        str(row.get("page_url") or "")
    )
    if (
        parsed_game != game_id
        or parsed_event != event_id
        or season != row.get("season")
    ):
        raise ValueError("BARD plan example page binding is invalid")
    outcome = row.get("shot_outcome")
    if row.get("event_state") == "foul_only":
        if outcome is not None:
            raise ValueError("BARD foul-only example has a shot outcome")
    elif outcome not in {"made", "missed"}:
        raise ValueError("BARD shot example outcome is invalid")


def _verify_video_url(url: str, *, game_id: str, event_id: str) -> None:
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "videos.nba.com"
        or not parsed.path.endswith(".mp4")
        or game_id not in parts
        or event_id not in parts
    ):
        raise ValueError("BARD video URL is not an exact NBA event MP4")


def _verify_resolution_manifest(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(json.dumps(value))
    claimed = str(manifest.pop("resolution_sha256", ""))
    if not _SHA256.fullmatch(claimed):
        raise ValueError("BARD resolution hash is invalid")
    if _canonical_sha256(manifest) != claimed:
        raise ValueError("BARD resolution hash mismatch")
    if (
        manifest.get("schema_version") != BARD_VIDEO_RESOLUTION_SCHEMA
        or manifest.get("purpose") != "local_research_media_resolution_only"
        or manifest.get("runtime_consumable") is not False
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
    ):
        raise ValueError("BARD resolution boundary is invalid")
    expected = {
        str(row["clip_id"]): row
        for row in plan["examples"]
    }
    clips = manifest.get("clips")
    if not isinstance(clips, list) or {
        str(row.get("clip_id") or "") for row in clips
    } != set(expected):
        raise ValueError("BARD resolution clips do not exactly match the plan")
    for row in clips:
        example = expected[str(row["clip_id"])]
        if (
            row.get("game_id") != example["game_id"]
            or row.get("event_id") != example["event_id"]
            or row.get("page_url") != example["page_url"]
        ):
            raise ValueError("BARD resolution clip binding is invalid")
        _verify_video_url(
            str(row.get("video_url") or ""),
            game_id=str(example["game_id"]),
            event_id=str(example["event_id"]),
        )
    manifest["resolution_sha256"] = claimed
    return manifest


def _verify_bard_embedded_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = json.loads(json.dumps(value))
    claimed = str(plan.pop("plan_sha256", ""))
    if not _SHA256.fullmatch(claimed):
        raise ValueError("BARD embedded plan hash is invalid")
    if _canonical_sha256(plan) != claimed:
        raise ValueError("BARD embedded plan hash mismatch")
    if (
        plan.get("schema_version") != BARD_EMBEDDED_VIDEO_PLAN_SCHEMA
        or plan.get("purpose") != "event_state_pretraining_only"
        or plan.get("runtime_consumable") is not False
        or plan.get("labels_exposed_to_runtime") is not False
        or plan.get("codex_runtime_answer_used") is not False
        or plan.get("blind_game_ids_excluded")
        != sorted(SEALED_BLIND_GAME_IDS)
    ):
        raise ValueError("BARD embedded plan boundary is invalid")
    source = plan.get("source") or {}
    if (
        source.get("dataset_id") != "bard-2026-embedded-validation"
        or source.get("url") != BARD_SOURCE_URL
        or source.get("declared_license") != BARD_DECLARED_LICENSE
        or not _GIT_REVISION.fullmatch(str(source.get("revision") or ""))
    ):
        raise ValueError("BARD embedded plan provenance is invalid")
    benchmark_hashes = source.get("benchmark_sha256s")
    if not isinstance(benchmark_hashes, dict) or not benchmark_hashes:
        raise ValueError("BARD embedded benchmark hashes are missing")
    for value in benchmark_hashes.values():
        _require_sha256(value, "BARD embedded benchmark")
    examples = plan.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("BARD embedded plan examples are missing")
    identities: set[str] = set()
    for row in examples:
        clip_id = str(row.get("clip_id") or "")
        game_id = str(row.get("game_id") or "")
        event_id = str(row.get("event_id") or "")
        repository_path = str(row.get("repository_path") or "")
        if (
            not clip_id
            or clip_id in identities
            or not _GAME_ID.fullmatch(game_id)
            or game_id in SEALED_BLIND_GAME_IDS
            or not _EVENT_ID.fullmatch(event_id)
            or row.get("event_state")
            not in {"live_field_goal", "free_throw"}
            or int(row.get("year") or 0) not in {2024, 2025}
            or not repository_path.startswith(
                f"validation/{row.get('year')}/multi/"
            )
            or Path(repository_path).name != f"{clip_id}.mp4"
            or not _GIT_REVISION.fullmatch(
                str(row.get("git_blob_sha1") or "")
            )
            or int(row.get("size_bytes") or 0) <= 0
        ):
            raise ValueError("BARD embedded plan example is invalid or blind")
        expected_url = (
            "https://raw.githubusercontent.com/GabrieleGiudic/BARD/"
            f"{source['revision']}/{repository_path}"
        )
        if row.get("download_url") != expected_url:
            raise ValueError("BARD embedded download URL is invalid")
        identities.add(clip_id)
    plan["plan_sha256"] = claimed
    return plan


def _verify_mp4(payload: bytes) -> None:
    if len(payload) < 12 or payload[4:8] != b"ftyp":
        raise ValueError("BARD download is not a valid MP4 payload")


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def _stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256(
        "\x1f".join((str(seed), *values)).encode()
    ).hexdigest()


def _require_sha256(value: object, label: str) -> None:
    if not _SHA256.fullmatch(str(value or "")):
        raise ValueError(f"{label} SHA-256 is invalid")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
