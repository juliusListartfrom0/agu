#!/usr/bin/env python3
"""Resumably download and SHA-seal media declared by a public research plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yt-dlp-executable", default="yt-dlp")
    parser.add_argument("--maximum-height", type=int, default=720)
    parser.add_argument("--retry-base-seconds", type=float, default=10.0)
    parser.add_argument("--retry-maximum-seconds", type=float, default=300.0)
    parser.add_argument(
        "--yt-dlp-retries",
        type=int,
        default=10,
        help="Retries inside one yt-dlp attempt; outer attempts remain resumable.",
    )
    parser.add_argument(
        "--force-ipv4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force IPv4 for networks with unstable IPv6/CDN TLS paths.",
    )
    parser.add_argument(
        "--external-downloader",
        choices=("native", "curl"),
        default="native",
        help="Use yt-dlp's native downloader or external curl transport.",
    )
    parser.add_argument(
        "--maximum-attempts",
        type=int,
        default=0,
        help="Attempts per video; 0 retries until success or process termination.",
    )
    parser.add_argument(
        "--only-provider",
        choices=("youtube", "internet_archive"),
        help="Download only one provider while retaining the full sealed plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.maximum_height <= 0 or args.yt_dlp_retries < 0:
        raise ValueError("height must be positive and yt-dlp retries non-negative")
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    jobs = filter_video_jobs(
        collect_video_jobs(payload),
        only_provider=args.only_provider,
    )
    youtube_executable = shutil.which(args.yt_dlp_executable)
    if any(job.get("media_provider", "youtube") == "youtube" for job in jobs):
        if youtube_executable is None:
            raise ValueError(f"yt-dlp executable not found: {args.yt_dlp_executable}")
    curl_executable = shutil.which("curl")
    if any(job.get("media_provider") == "internet_archive" for job in jobs):
        if curl_executable is None:
            raise ValueError("curl executable not found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "download-state.json"
    state = _load_state(state_path, plan_sha256=str(payload.get("plan_sha256") or ""))

    for job in jobs:
        video_id = job["youtube_id"]
        complete = find_complete_media(args.output_dir, video_id)
        if complete is not None:
            _record_complete(state, state_path, job, complete)
            print(f"already complete: {video_id} -> {complete.name}", flush=True)
            continue

        attempt = int((state["videos"].get(video_id) or {}).get("attempts", 0))
        while args.maximum_attempts == 0 or attempt < args.maximum_attempts:
            attempt += 1
            state["videos"][video_id] = {
                **job,
                "status": "downloading",
                "attempts": attempt,
            }
            _write_state(state_path, state)
            print(f"download attempt {attempt}: {video_id}", flush=True)
            provider = job.get("media_provider", "youtube")
            direct_output: Path | None = None
            if provider == "internet_archive":
                suffix = Path(job["media_filename"]).suffix.lower()
                if suffix not in {".mp4", ".mkv", ".webm", ".ogv", ".avi"}:
                    raise ValueError(f"unsupported direct media suffix: {suffix}")
                complete_output = args.output_dir / f"{video_id}{suffix}"
                direct_output = complete_output.with_suffix(f"{suffix}.part")
                command = _direct_download_command(
                    str(curl_executable),
                    output_path=direct_output,
                    media_url=job["media_url"],
                    retries=args.yt_dlp_retries,
                )
            else:
                command = _download_command(
                    str(youtube_executable),
                    output_dir=args.output_dir,
                    video_url=job["youtube_url"],
                    maximum_height=args.maximum_height,
                    retries=args.yt_dlp_retries,
                    force_ipv4=args.force_ipv4,
                    external_downloader=args.external_downloader,
                )
            result = subprocess.run(
                command,
                check=False,
            )
            if result.returncode == 0 and direct_output is not None:
                direct_output.replace(complete_output)
            complete = find_complete_media(args.output_dir, video_id)
            if result.returncode == 0 and complete is not None:
                _record_complete(state, state_path, job, complete)
                print(f"download complete: {video_id} -> {complete.name}", flush=True)
                break

            delay = retry_delay(
                attempt,
                base_seconds=args.retry_base_seconds,
                maximum_seconds=args.retry_maximum_seconds,
            )
            state["videos"][video_id] = {
                **job,
                "status": "retry_wait",
                "attempts": attempt,
                "last_returncode": result.returncode,
                "next_retry_seconds": delay,
            }
            _write_state(state_path, state)
            print(
                f"download failed: {video_id}; retrying in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        else:
            state["videos"][video_id] = {
                **job,
                "status": "failed",
                "attempts": attempt,
            }
            state["status"] = "failed"
            _write_state(state_path, state)
            return 1

    state["status"] = "partial_completed" if args.only_provider else "completed"
    _write_state(state_path, state)
    return 0


def collect_video_jobs(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Return unique benchmark/enrollment videos without truth values."""

    jobs: dict[str, dict[str, str]] = {}
    for benchmark in payload.get("benchmarks") or []:
        games = [benchmark.get("benchmark"), *(benchmark.get("enrollment_games") or [])]
        for game in games:
            if not isinstance(game, dict):
                raise ValueError("research plan contains an invalid game")
            video_id = str(game.get("youtube_id") or "")
            video_url = str(game.get("youtube_url") or "")
            slug = str(game.get("slug") or "")
            if not video_id or not video_url or not slug:
                raise ValueError("research plan game lacks media identity")
            existing = jobs.get(video_id)
            job = {"youtube_id": video_id, "youtube_url": video_url, "slug": slug}
            media_source = game.get("media_source")
            if media_source is not None:
                if not isinstance(media_source, dict):
                    raise ValueError("research plan contains an invalid media source")
                provider = str(media_source.get("provider") or "")
                media_url = str(media_source.get("url") or "")
                media_filename = str(media_source.get("filename") or "")
                if provider != "internet_archive" or not media_url or not media_filename:
                    raise ValueError("research plan contains an invalid media override")
                job.update(
                    {
                        "media_provider": provider,
                        "media_url": media_url,
                        "media_filename": media_filename,
                    }
                )
            if existing is not None and existing != job:
                raise ValueError(f"conflicting media entries for {video_id}")
            jobs[video_id] = job
    if not jobs:
        raise ValueError("research plan contains no media jobs")
    return [jobs[video_id] for video_id in sorted(jobs)]


def filter_video_jobs(
    jobs: list[dict[str, str]],
    *,
    only_provider: str | None,
) -> list[dict[str, str]]:
    if only_provider is None:
        return jobs
    selected = [
        job
        for job in jobs
        if job.get("media_provider", "youtube") == only_provider
    ]
    if not selected:
        raise ValueError(f"research plan contains no media jobs for {only_provider}")
    return selected


def find_complete_media(output_dir: Path, video_id: str) -> Path | None:
    matches = [
        path
        for path in output_dir.glob(f"{video_id}.*")
        if path.is_file()
        and path.suffix not in {".part", ".json", ".ytdl"}
        and not path.name.endswith(".temp.mp4")
        and re.search(r"\.f\d+\.[^.]+$", path.name) is None
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple complete media files found for {video_id}")
    return matches[0] if matches else None


def retry_delay(
    attempt: int,
    *,
    base_seconds: float,
    maximum_seconds: float,
) -> float:
    if attempt <= 0 or base_seconds < 0 or maximum_seconds < 0:
        raise ValueError("retry values must be non-negative and attempt must be positive")
    return min(maximum_seconds, base_seconds * (2 ** min(attempt - 1, 10)))


def _download_command(
    executable: str,
    *,
    output_dir: Path,
    video_url: str,
    maximum_height: int,
    retries: int = 10,
    force_ipv4: bool = False,
    external_downloader: str = "native",
) -> list[str]:
    if (
        maximum_height <= 0
        or retries < 0
        or external_downloader not in {"native", "curl"}
    ):
        raise ValueError("download command configuration is invalid")
    command = [
        executable,
        "--no-playlist",
        "--continue",
        "--newline",
        "--retries",
        str(retries),
        "--fragment-retries",
        str(retries),
        "--file-access-retries",
        str(retries),
        "--retry-sleep",
        "http:exp=1:30",
        "--socket-timeout",
        "30",
        "--throttled-rate",
        "100K",
        "--concurrent-fragments",
        "4",
        "-f",
        f"bv*[height<={maximum_height}]+ba/b[height<={maximum_height}]",
        "--merge-output-format",
        "mp4",
        "-P",
        str(output_dir),
        "-o",
        "%(id)s.%(ext)s",
        video_url,
    ]
    if force_ipv4:
        command.insert(1, "--force-ipv4")
    if external_downloader == "curl":
        command[1:1] = [
            "--downloader",
            "curl",
            "--downloader-args",
            f"curl:--retry {retries} --retry-all-errors --retry-delay 2",
        ]
    return command


def _direct_download_command(
    executable: str,
    *,
    output_path: Path,
    media_url: str,
    retries: int,
) -> list[str]:
    parsed = urlparse(media_url)
    if (
        retries < 0
        or parsed.scheme != "https"
        or parsed.hostname != "archive.org"
        or not parsed.path.startswith("/download/")
    ):
        raise ValueError("direct download configuration is invalid")
    return [
        executable,
        "--location",
        "--fail",
        "--continue-at",
        "-",
        "--retry",
        str(retries),
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--output",
        str(output_path),
        media_url,
    ]


def _load_state(path: Path, *, plan_sha256: str) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("plan_sha256") != plan_sha256:
            raise ValueError("download state belongs to a different research plan")
        return state
    return {
        "schema_version": "agu.public-research-download-state.v1",
        "plan_sha256": plan_sha256,
        "status": "running",
        "videos": {},
    }


def _record_complete(
    state: dict[str, Any],
    state_path: Path,
    job: dict[str, str],
    media_path: Path,
) -> None:
    previous = state["videos"].get(job["youtube_id"]) or {}
    state["videos"][job["youtube_id"]] = {
        **job,
        "status": "completed",
        "attempts": int(previous.get("attempts", 0)),
        "filename": media_path.name,
        "size_bytes": media_path.stat().st_size,
        "sha256": _file_sha256(media_path),
    }
    _write_state(state_path, state)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
