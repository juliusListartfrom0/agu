#!/usr/bin/env python3
"""Render raw-only contact sheets for manual VRU causal review."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=320)
    return parser.parse_args()


def render_sheets(
    *, frame_manifest_path: Path, output_dir: Path, columns: int = 5, tile_width: int = 320
) -> dict[str, int]:
    if columns < 1 or tile_width < 1:
        raise ValueError("columns and tile_width must be positive")
    artifact = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "agu.vru-causal-review-frame-manifest.v1":
        raise ValueError("unsupported VRU frame manifest")
    rows = artifact.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError("VRU frame manifest requires frames")
    frame_root = frame_manifest_path.parent / str(artifact.get("frame_root") or "")
    if not frame_root.is_dir():
        raise ValueError(f"VRU frame root is missing: {frame_root}")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("VRU frame rows must be objects")
        grouped[str(row.get("review_id") or "")].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for review_id, review_rows in sorted(grouped.items()):
        review_rows.sort(key=lambda row: int(row["position"]))
        first = frame_root / str(review_rows[0]["relative_path"])
        with Image.open(first) as sample:
            tile_height = round(tile_width * sample.height / sample.width)
        banner_height = 28
        rows_per_page = (len(review_rows) + columns - 1) // columns
        sheet = Image.new(
            "RGB", (columns * tile_width, rows_per_page * (tile_height + banner_height)), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(review_rows):
            source = frame_root / str(row["relative_path"])
            if not source.is_file():
                raise ValueError(f"VRU review frame is missing: {source}")
            grid_row, grid_column = divmod(offset, columns)
            x = grid_column * tile_width
            y = grid_row * (tile_height + banner_height)
            with Image.open(source) as image:
                tile = image.convert("RGB").resize((tile_width, tile_height))
            sheet.paste(tile, (x, y + banner_height))
            draw.text(
                (x + 4, y + 5),
                f"p{int(row['position']):02d} f{int(row['frame_index'])}",
                fill="black",
            )
        sheet.save(output_dir / f"{review_id}.jpg", quality=92)
        rendered += 1
    return {"reviews": rendered, "frames": len(rows)}


def main() -> int:
    args = parse_args()
    result = render_sheets(
        frame_manifest_path=args.frame_manifest,
        output_dir=args.output_dir,
        columns=args.columns,
        tile_width=args.tile_width,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
