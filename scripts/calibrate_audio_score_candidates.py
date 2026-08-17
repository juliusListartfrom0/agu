#!/usr/bin/env python3
"""Calibrate speech-candidate ranking on benchmark-disjoint training games only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.audio_evidence import (  # noqa: E402
    AudioEvidenceArtifact,
    AudioRosterArtifact,
    canonical_sha256,
    ranked_score_window_player_candidates,
    verify_audio_evidence,
    verify_audio_roster,
)


@dataclass(frozen=True)
class TrainingCase:
    name: str
    scoreboard: Mapping[str, Any]
    audio: AudioEvidenceArtifact
    roster: AudioRosterArtifact
    play_by_play: Sequence[Mapping[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoreboard-evidence", type=Path, action="append", required=True)
    parser.add_argument("--audio-evidence", type=Path, action="append", required=True)
    parser.add_argument("--roster", type=Path, action="append", required=True)
    parser.add_argument("--play-by-play", type=Path, action="append", required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.85)
    parser.add_argument("--minimum-predictions-per-game", type=int, default=5)
    parser.add_argument("--minimum-coverage-per-game", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def truth_player_for_delta(
    delta: Mapping[str, Any],
    play_by_play: Sequence[Mapping[str, Any]],
) -> str | None:
    scoring = _scoring_records(play_by_play)
    before_key = _score_key(delta.get("before_scores") or {})
    after_key = _score_key(delta.get("after_scores") or {})
    for index, record in enumerate(scoring):
        if _score_key(record["scores"]) != after_key:
            continue
        previous_scores = scoring[index - 1]["scores"] if index else record["zero_scores"]
        if _score_key(previous_scores) != before_key:
            return None
        if str(record["team_id"]) != str(delta.get("team_id") or ""):
            return None
        if int(record["points"]) != int(delta.get("points") or 0):
            return None
        return str(record["person_id"])
    return None


def evaluate_rule(
    cases: Sequence[TrainingCase],
    *,
    expected_lag_sec: float,
    minimum_confidence: float,
    max_candidates: int,
    require_shot_action_context: bool,
    require_made_shot_context: bool,
) -> dict[str, Any]:
    per_game: list[dict[str, Any]] = []
    pooled = {"truth": 0, "predicted": 0, "correct": 0}
    for case in cases:
        counts = {"truth": 0, "predicted": 0, "correct": 0}
        for delta in case.scoreboard.get("deltas") or []:
            truth = truth_player_for_delta(delta, case.play_by_play)
            if truth is None:
                continue
            counts["truth"] += 1
            ranked = ranked_score_window_player_candidates(
                case.audio,
                case.roster,
                team_id=str(delta["team_id"]),
                before_frame=int(delta["before_frame"]),
                after_frame=int(delta["after_frame"]),
                expected_scoreboard_lag_sec=expected_lag_sec,
                minimum_confidence=minimum_confidence,
                require_shot_action_context=require_shot_action_context,
                require_made_shot_context=require_made_shot_context,
            )
            if not ranked or len(ranked) > max_candidates:
                continue
            counts["predicted"] += 1
            counts["correct"] += int(ranked[0] == truth)
        per_game.append({"name": case.name, **counts, **_metrics(counts)})
        for key in pooled:
            pooled[key] += counts[key]
    return {
        "rule": {
            "expected_scoreboard_lag_sec": expected_lag_sec,
            "minimum_confidence": minimum_confidence,
            "max_candidates": max_candidates,
            "require_shot_action_context": require_shot_action_context,
            "require_made_shot_context": require_made_shot_context,
        },
        "per_game": per_game,
        "pooled": {**pooled, **_metrics(pooled)},
    }


def calibrate(
    cases: Sequence[TrainingCase],
    *,
    minimum_precision: float,
    minimum_predictions_per_game: int = 5,
    minimum_coverage_per_game: float = 0.05,
) -> dict[str, Any]:
    candidates = [
        evaluate_rule(
            cases,
            expected_lag_sec=lag,
            minimum_confidence=confidence,
            max_candidates=max_candidates,
            require_shot_action_context=require_shot_action_context,
            require_made_shot_context=require_made_shot_context,
        )
        for lag, confidence, max_candidates, context in product(
            (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
            (0.0, 0.50, 0.65, 0.75),
            (1, 2, 3),
            ("none", "shot", "made"),
        )
        for require_shot_action_context, require_made_shot_context in (
            (False, False) if context == "none" else (True, context == "made"),
        )
    ]
    eligible = [
        item
        for item in candidates
        if item["pooled"]["predicted"] > 0
        and item["pooled"]["precision"] >= minimum_precision
        and all(
            game["predicted"] >= minimum_predictions_per_game
            and game["coverage"] >= minimum_coverage_per_game
            and game["precision"] >= minimum_precision
            for game in item["per_game"]
        )
    ]
    eligible.sort(
        key=lambda item: (
            item["pooled"]["correct"],
            item["pooled"]["coverage"],
            item["pooled"]["precision"],
        ),
        reverse=True,
    )
    best_observed = max(
        candidates,
        key=lambda item: (
            item["pooled"]["precision"],
            item["pooled"]["correct"],
            item["pooled"]["coverage"],
        ),
    )
    return {
        "promoted": bool(eligible),
        "minimum_precision": minimum_precision,
        "minimum_predictions_per_game": minimum_predictions_per_game,
        "minimum_coverage_per_game": minimum_coverage_per_game,
        "selected": eligible[0] if eligible else None,
        "best_observed": best_observed,
        "rules_evaluated": len(candidates),
        "note": "Speech output remains candidate ranking only; promotion never confirms a player actor.",
    }


def _scoring_records(play_by_play: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    team_locations = {
        str(row.get("location")): str(row.get("teamTricode"))
        for row in play_by_play
        if row.get("location") in {"h", "v"} and row.get("teamTricode")
    }
    home = team_locations.get("h")
    away = team_locations.get("v")
    if not home or not away:
        return []
    zero_scores = {home: 0, away: 0}
    records: list[dict[str, Any]] = []
    previous = dict(zero_scores)
    for row in play_by_play:
        if not _is_scoring_action(row):
            continue
        try:
            scores = {home: int(row["scoreHome"]), away: int(row["scoreAway"])}
        except (KeyError, TypeError, ValueError):
            continue
        team_id = str(row.get("teamTricode") or "")
        if team_id not in scores:
            continue
        changes = {team: scores[team] - previous[team] for team in scores}
        positive = [(team, value) for team, value in changes.items() if value > 0]
        if len(positive) != 1 or any(value < 0 for value in changes.values()):
            continue
        scoring_team, points = positive[0]
        records.append(
            {
                "scores": scores,
                "zero_scores": zero_scores,
                "team_id": scoring_team,
                "points": points,
                "person_id": str(row.get("personId") or ""),
            }
        )
        previous = scores
    return records


def _is_scoring_action(row: Mapping[str, Any]) -> bool:
    if row.get("actionType") == "Made Shot":
        return True
    if row.get("actionType") == "Free Throw":
        return "MISS" not in str(row.get("description") or "").upper()
    return False


def _score_key(scores: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    try:
        return tuple(sorted((str(team), int(score)) for team, score in scores.items()))
    except (TypeError, ValueError):
        return ()


def _metrics(counts: Mapping[str, int]) -> dict[str, float]:
    predicted = int(counts["predicted"])
    truth = int(counts["truth"])
    correct = int(counts["correct"])
    return {
        "precision": correct / predicted if predicted else 0.0,
        "recall": correct / truth if truth else 0.0,
        "coverage": predicted / truth if truth else 0.0,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    lengths = {
        len(args.scoreboard_evidence),
        len(args.audio_evidence),
        len(args.roster),
        len(args.play_by_play),
    }
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("provide matching inputs for at least two benchmark-disjoint training games")
    cases: list[TrainingCase] = []
    inputs: list[dict[str, str]] = []
    for scoreboard_path, audio_path, roster_path, pbp_path in zip(
        args.scoreboard_evidence,
        args.audio_evidence,
        args.roster,
        args.play_by_play,
        strict=True,
    ):
        scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
        audio = verify_audio_evidence(
            AudioEvidenceArtifact.model_validate_json(audio_path.read_text(encoding="utf-8"))
        )
        roster = verify_audio_roster(
            AudioRosterArtifact.model_validate_json(roster_path.read_text(encoding="utf-8"))
        )
        if scoreboard.get("raw_video_sha256") != audio.raw_video_sha256:
            raise ValueError("scoreboard/audio raw-video hash mismatch")
        cases.append(
            TrainingCase(
                name=scoreboard_path.parent.name,
                scoreboard=scoreboard,
                audio=audio,
                roster=roster,
                play_by_play=_load_jsonl(pbp_path),
            )
        )
        inputs.append(
            {
                "scoreboard_sha256": _sha256_file(scoreboard_path),
                "audio_sha256": _sha256_file(audio_path),
                "roster_sha256": _sha256_file(roster_path),
                "play_by_play_sha256": _sha256_file(pbp_path),
            }
        )
    result = {
        "schema_version": "agu.audio-score-candidate-calibration.v1",
        "role": "benchmark_disjoint_training",
        "inputs": inputs,
        **calibrate(
            cases,
            minimum_precision=args.minimum_precision,
            minimum_predictions_per_game=args.minimum_predictions_per_game,
            minimum_coverage_per_game=args.minimum_coverage_per_game,
        ),
    }
    result["artifact_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
