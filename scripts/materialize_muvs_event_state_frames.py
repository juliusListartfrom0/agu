#!/usr/bin/env python3
"""Materialize two bounded source frames for every sealed MUVS sample."""

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

from app.analysis.muvs_event_state import (  # noqa: E402
    seal_muvs_event_state_frames,
    verify_muvs_event_state_frames,
    verify_muvs_event_state_plan,
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
        raise ValueError(f"could not decode MUVS frame: {path.name}")
    height, width = image.shape[:2]
    if width != 1280 or height < 1:
        raise ValueError(f"unexpected MUVS frame shape: {path.name}")
    return width, height


def _extract_frame(
    *,
    ffmpeg_bin: str,
    url: str,
    time_sec: float,
    target: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    if timeout_sec <= 0:
        raise ValueError("MUVS ffmpeg timeout must be positive")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_name(f"{target.stem}.part.png")
        command = [
            ffmpeg_bin,
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{time_sec:.6f}",
            "-i",
            url,
            "-frames:v",
            "1",
            "-vf",
            "scale=1280:-2",
            "-threads",
            "1",
            "-an",
            "-f",
            "image2",
            "-y",
            str(temporary),
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                subprocess.run(
                    command,
                    check=True,
                    timeout=timeout_sec,
                )
                _inspect_png(temporary)
                temporary.replace(target)
                last_error = None
                break
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                ValueError,
            ) as exc:
                last_error = exc
                if temporary.exists():
                    temporary.unlink()
                if attempt < 2:
                    time.sleep(attempt + 1)
        if last_error is not None:
            raise last_error
        status = "downloaded"
    else:
        status = "verified_existing"
    width, height = _inspect_png(target)
    return {
        "status": status,
        "sha256": _sha256(target),
        "size_bytes": target.stat().st_size,
        "width": width,
        "height": height,
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("MUVS frame workers must be in [1,8]")
    plan = verify_muvs_event_state_plan(
        json.loads(args.plan.read_text(encoding="utf-8"))
    )
    if args.output.exists():
        existing = verify_muvs_event_state_frames(
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

    frame_root = args.output.parent / "frames"
    tasks = []
    for index, sample in enumerate(plan["samples"], start=1):
        for position, time_sec in zip(
            ("frame_before", "frame_after"),
            sample["frame_times_sec"],
            strict=True,
        ):
            target = (
                frame_root
                / f"{sample['sample_id']}_{position.removeprefix('frame_')}.png"
            )
            tasks.append(
                {
                    "sample_index": index,
                    "sample_id": sample["sample_id"],
                    "position": position,
                    "url": sample["video_url"],
                    "time_sec": float(time_sec),
                    "target": target,
                }
            )

    completed: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[tuple[dict[str, Any], Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _extract_frame,
                ffmpeg_bin=args.ffmpeg_bin,
                url=str(task["url"]),
                time_sec=float(task["time_sec"]),
                target=Path(task["target"]),
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
                            "stage": "muvs_event_state_frames",
                            "sample": task["sample_index"],
                            "sample_count": len(plan["samples"]),
                            "sample_id": task["sample_id"],
                            "position": task["position"],
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )
                continue
            target = Path(task["target"])
            completed[(str(task["sample_id"]), str(task["position"]))] = {
                "path": target.relative_to(args.output.parent).as_posix(),
                "sha256": info["sha256"],
                "size_bytes": info["size_bytes"],
                "width": info["width"],
                "height": info["height"],
            }
            print(
                json.dumps(
                    {
                        "stage": "muvs_event_state_frames",
                        "sample": task["sample_index"],
                        "sample_count": len(plan["samples"]),
                        "sample_id": task["sample_id"],
                        "position": task["position"],
                        "status": info["status"],
                    }
                ),
                flush=True,
            )

    if failures:
        failed_keys = ", ".join(
            f"{task['sample_id']}:{task['position']}"
            for task, _exc in failures[:8]
        )
        suffix = "" if len(failures) <= 8 else ", ..."
        raise RuntimeError(
            f"{len(failures)} MUVS frame extraction task(s) failed after retries: "
            f"{failed_keys}{suffix}"
        )

    files = []
    for sample in plan["samples"]:
        sample_id = str(sample["sample_id"])
        files.append(
            {
                "sample_id": sample_id,
                "frame_before": completed[(sample_id, "frame_before")],
                "frame_after": completed[(sample_id, "frame_after")],
            }
        )

    artifact = seal_muvs_event_state_frames(
        {"files": files},
        plan=plan,
    )
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
