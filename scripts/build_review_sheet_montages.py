#!/usr/bin/env python3
"""Tile raw-only review sheets into labeled pages for efficient offline audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--tile-width", type=int, default=1280)
    args = parser.parse_args()
    if min(args.columns, args.rows, args.tile_width) < 1:
        raise ValueError("montage dimensions must be positive")
    labeled_ids = set()
    for path in args.labels:
        payload = json.loads(path.read_text(encoding="utf-8"))
        labeled_ids.update(str(row["event_id"]) for row in payload.get("examples", []))
    sheets = sorted(
        path
        for path in args.sheet_dir.glob("*.jpg")
        if path.stem not in labeled_ids
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_page = args.columns * args.rows
    records = []
    for page_index in range(0, len(sheets), per_page):
        page_sheets = sheets[page_index : page_index + per_page]
        with Image.open(page_sheets[0]) as sample:
            tile_height = round(args.tile_width * sample.height / sample.width)
        banner_height = 36
        page = Image.new(
            "RGB",
            (args.columns * args.tile_width, args.rows * (tile_height + banner_height)),
            "white",
        )
        draw = ImageDraw.Draw(page)
        for offset, path in enumerate(page_sheets):
            row, column = divmod(offset, args.columns)
            x = column * args.tile_width
            y = row * (tile_height + banner_height)
            with Image.open(path) as source:
                tile = source.convert("RGB").resize((args.tile_width, tile_height))
            page.paste(tile, (x, y + banner_height))
            draw.text((x + 10, y + 8), path.stem, fill="black")
        output = args.output_dir / f"page-{page_index // per_page + 1:03d}.jpg"
        page.save(output, quality=92)
        records.append({"page": output.name, "event_ids": [path.stem for path in page_sheets]})
    manifest = {
        "schema_version": "agu.review-sheet-montages.v1",
        "runtime_consumable": False,
        "source_sheet_dir": args.sheet_dir.name,
        "excluded_labeled_event_ids": sorted(labeled_ids),
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sheets": len(sheets), "pages": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
