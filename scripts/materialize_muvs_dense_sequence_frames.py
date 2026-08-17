#!/usr/bin/env python3
"""Materialize the fixed multi-frame source sequence for a sealed MUVS plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvs_dense_sequence import (  # noqa: E402
    seal_muvs_dense_sequence_frames,
    verify_muvs_dense_sequence_frames,
    verify_muvs_dense_sequence_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_png(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise ValueError(f"could not decode MUVS dense frame: {path.name}")
    height, width = image.shape[:2]
    if width != 1280 or height < 1:
        raise ValueError(f"unexpected MUVS dense frame shape: {path.name}")
    return width, height


def _extract_sequence(
    *,
    ffmpeg_bin: str,
    url: str,
    time_secs: list[float],
    targets: list[Path],
    timeout_sec: float,
) -> list[dict[str, Any]]:
    if timeout_sec <= 0:
        raise ValueError("MUVS dense ffmpeg timeout must be positive")
    if len(time_secs) != len(targets) or len(time_secs) < 3:
        raise ValueError("MUVS dense sequence extraction requires three frames")
    if any(right <= left for left, right in zip(time_secs, time_secs[1:])):
        raise ValueError("MUVS dense sequence times must be increasing")
    target_root = targets[0].parent
    target_root.mkdir(parents=True, exist_ok=True)
    if all(target.exists() for target in targets):
        status = "verified_existing"
    else:
        temporary_pattern = target_root / f".{targets[0].stem}.part_%02d.png"
        for path in target_root.glob(f".{targets[0].stem}.part_*.png"):
            path.unlink()
        step = time_secs[1] - time_secs[0]
        if any(
            abs((right - left) - step) > 1e-6
            for left, right in zip(time_secs, time_secs[1:])
        ):
            raise ValueError("MUVS dense materializer requires equally spaced times")
        command = [
            ffmpeg_bin,
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{time_secs[0]:.6f}",
            "-i",
            url,
            "-frames:v",
            str(len(targets)),
            "-vf",
            f"fps={1.0 / step:.12g},scale=1280:-2",
            "-threads",
            "1",
            "-an",
            "-f",
            "image2",
            "-y",
            str(temporary_pattern),
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                subprocess.run(command, check=True, timeout=timeout_sec)
                temporary_paths = [
                    target_root / f".{targets[0].stem}.part_{index:02d}.png"
                    for index in range(1, len(targets) + 1)
                ]
                if not all(path.is_file() for path in temporary_paths):
                    raise ValueError("ffmpeg returned too few MUVS dense frames")
                for temporary_path, target in zip(
                    temporary_paths,
                    targets,
                    strict=True,
                ):
                    _inspect_png(temporary_path)
                    temporary_path.replace(target)
                last_error = None
                break
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                ValueError,
            ) as exc:
                last_error = exc
                for path in target_root.glob(f".{targets[0].stem}.part_*.png"):
                    path.unlink()
                if attempt < 2:
                    time.sleep(attempt + 1)
        if last_error is not None:
            raise last_error
        status = "downloaded"
    output = []
    for target in targets:
        width, height = _inspect_png(target)
        output.append(
            {
                "status": status,
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
                "width": width,
                "height": height,
            }
        )
    return output


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("MUVS dense frame workers must be in [1,4]")
    plan = verify_muvs_dense_sequence_plan(
        json.loads(args.plan.read_text(encoding="utf-8"))
    )
    if args.output.exists():
        existing = verify_muvs_dense_sequence_frames(
            json.loads(args.output.read_text(encoding="utf-8")),
            plan=plan,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": "verified_existing",
                    "frame_count": existing["frame_count"],
                    "artifact_sha256": existing["artifact_sha256"],
                }
            )
        )
        return 0

    frame_root = args.output.parent / "frames_dense_sequence_v1"
    tasks: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(plan["samples"], start=1):
        targets = [
            frame_root / f"{sample['sample_id']}_f{position:02d}.png"
            for position in range(plan["sequence_length"])
        ]
        tasks.append(
            {
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "url": sample["video_url"],
                "time_secs": [float(value) for value in sample["frame_times_sec"]],
                "targets": targets,
            }
        )

    completed: dict[str, list[dict[str, Any]]] = {}
    failures: list[tuple[dict[str, Any], Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _extract_sequence,
                ffmpeg_bin=args.ffmpeg_bin,
                url=str(task["url"]),
                time_secs=list(task["time_secs"]),
                targets=[Path(path) for path in task["targets"]],
                timeout_sec=args.timeout_sec,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                info = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((task, exc))
                print(
                    json.dumps(
                        {
                            "stage": "muvs_dense_sequence_frames",
                            "sample": task["sample_index"],
                            "sample_count": len(plan["samples"]),
                            "sample_id": task["sample_id"],
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )
                continue
            completed[str(task["sample_id"])] = [
                {
                    "path": Path(target).relative_to(args.output.parent).as_posix(),
                    "sha256": frame_info["sha256"],
                    "size_bytes": frame_info["size_bytes"],
                    "width": frame_info["width"],
                    "height": frame_info["height"],
                }
                for target, frame_info in zip(task["targets"], info, strict=True)
            ]
            print(
                json.dumps(
                    {
                        "stage": "muvs_dense_sequence_frames",
                        "sample": task["sample_index"],
                        "sample_count": len(plan["samples"]),
                        "sample_id": task["sample_id"],
                        "frames": len(info),
                        "status": info[0]["status"],
                    }
                ),
                flush=True,
            )

    if failures:
        failed_keys = ", ".join(task["sample_id"] for task, _exc in failures[:8])
        suffix = "" if len(failures) <= 8 else ", ..."
        raise RuntimeError(
            f"{len(failures)} MUVS dense frame extraction task(s) failed after retries: "
            f"{failed_keys}{suffix}"
        )

    files = []
    for sample in plan["samples"]:
        sample_id = str(sample["sample_id"])
        files.append(
            {
                "sample_id": sample_id,
                "frames": completed[sample_id],
            }
        )

    artifact = seal_muvs_dense_sequence_frames({"files": files}, plan=plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(".json.part")
    temporary_output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": "completed",
                "frame_count": artifact["frame_count"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
