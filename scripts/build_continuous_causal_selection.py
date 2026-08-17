#!/usr/bin/env python3
"""Freeze deterministic label-hidden review windows from one continuous game."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.continuous_causal_selection import (  # noqa: E402
    build_continuous_causal_selection,
    resolve_continuous_source_video_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _freeze_paths(
    source_manifest: Path,
    source_video: Path,
    output: Path,
) -> tuple[Path, Path, Path]:
    source = source_manifest.resolve(strict=True)
    video = source_video.resolve(strict=True)
    target = output.resolve(strict=False)
    protected = (source, video)
    if any(str(path).casefold() == str(target).casefold() for path in protected):
        raise ValueError("selection output must not alias a protected input")
    if target.exists() and any(os.path.samefile(path, target) for path in protected):
        raise ValueError("selection output must not alias a protected input")
    return source, video, target


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    source_manifest = args.source_manifest.resolve(strict=True)
    source_video = resolve_continuous_source_video_path(
        source_manifest,
        source_id=args.source_id,
    )
    source_manifest, _, output = _freeze_paths(
        source_manifest,
        source_video,
        args.output,
    )
    artifact = build_continuous_causal_selection(
        source_manifest_path=source_manifest,
        source_id=args.source_id,
    )
    _atomic_write_json(output, artifact)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "output": output.as_posix(),
                "windows": len(artifact["windows"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
