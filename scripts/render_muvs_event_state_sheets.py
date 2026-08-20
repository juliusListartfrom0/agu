#!/usr/bin/env python3
"""Render label-hidden two-frame MUVS source review sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvs_event_state import (  # noqa: E402
    verify_muvs_event_state_frames,
    verify_muvs_event_state_plan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_image(
    root: Path,
    record: dict[str, object],
) -> Image.Image:
    path = root / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["size_bytes"])
        or _sha256(path) != record["sha256"]
    ):
        raise ValueError(f"MUVS source frame hash mismatch: {path.name}")
    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = verify_muvs_event_state_plan(
        json.loads(args.plan.read_text(encoding="utf-8"))
    )
    frames = verify_muvs_event_state_frames(
        json.loads(args.frames.read_text(encoding="utf-8")),
        plan=plan,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in frames["files"]:
        before = _verified_image(args.frames.parent, row["frame_before"])
        after = _verified_image(args.frames.parent, row["frame_after"])
        before.thumbnail((640, 360), Image.Resampling.LANCZOS)
        after.thumbnail((640, 360), Image.Resampling.LANCZOS)
        sheet = Image.new("RGB", (1280, 400), "white")
        sheet.paste(before, (0, 40))
        sheet.paste(after, (640, 40))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), f"{row['sample_id']}  BEFORE", fill="black")
        draw.text((650, 10), "AFTER", fill="black")
        output = args.output_dir / f"{row['sample_id']}.jpg"
        sheet.save(output, quality=94)
        records.append(
            {
                "sample_id": row["sample_id"],
                "sheet": output.name,
                "size_bytes": output.stat().st_size,
                "sha256": _sha256(output),
            }
        )
    manifest = {
        "schema_version": "agu.muvs-event-state-sheets.v1",
        "purpose": "codex_offline_muvs_source_event_state_annotation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "plan_sha256": plan["artifact_sha256"],
        "frames_sha256": frames["artifact_sha256"],
        "reviewer_visible_fields": [
            "sample_id",
            "frame_before",
            "frame_after",
        ],
        "sheet_count": len(records),
        "records": records,
    }
    manifest["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "sheet_count": len(records),
                "artifact_sha256": manifest["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
