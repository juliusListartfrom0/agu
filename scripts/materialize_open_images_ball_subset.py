#!/usr/bin/env python3
"""Materialize a small, deterministic Open Images ball-box subset.

The source annotations are downloaded from the official Open Images V7
metadata endpoints.  This script only materializes image pixels whose image
IDs are selected from the validation split; it never reads AGU game truth or
blind files.  The resulting manifest is training-only and is not consumed by
the AGU runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

LABELS = {
    "/m/018xm": "Ball (Object)",
    "/m/02ctlc": "Cricket ball",
    "/m/01226z": "Football",
    "/m/044r5d": "Golf ball",
    "/m/0wdt60w": "Rugby ball",
    "/m/05ctyq": "Tennis ball",
    "/m/02rgn06": "Volleyball (Ball)",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fetch_one(image_id: str, destination: Path) -> tuple[str, int, str]:
    if destination.is_file() and destination.stat().st_size > 0:
        payload = destination.read_bytes()
        return image_id, len(payload), hashlib.sha256(payload).hexdigest()
    url = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(
                url, headers={"User-Agent": "AGU-open-images-research/1"}
            )
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            destination.write_bytes(payload)
            return image_id, len(payload), hashlib.sha256(payload).hexdigest()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {last_error}")


def _select_ids(
    annotations: list[dict[str, str]],
    *,
    max_images: int,
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in annotations:
        if row.get("LabelName") in LABELS:
            grouped.setdefault(row["ImageID"], []).append(row)
    selected = sorted(grouped)[:max_images]
    return selected, {image_id: grouped[image_id] for image_id in selected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/public_sources/open_images_sports_ball_v7"),
    )
    parser.add_argument("--max-images", type=int, default=240)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.max_images <= 0 or args.workers <= 0:
        raise ValueError("max-images and workers must be positive")

    raw = args.root / "raw"
    image_dir = args.root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = raw / "validation-annotations-bbox.csv"
    urls_path = raw / "validation-images-with-rotation.csv"
    for path in (annotations_path, urls_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    annotations = _read_rows(annotations_path)
    selected_ids, grouped = _select_ids(annotations, max_images=args.max_images)
    url_rows = {
        row["ImageID"]: row
        for row in _read_rows(urls_path)
        if row["ImageID"] in selected_ids
    }
    missing = sorted(set(selected_ids) - set(url_rows))
    if missing:
        raise ValueError(f"missing validation URL rows: {missing[:3]}")

    downloads: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_fetch_one, image_id, image_dir / f"{image_id}.jpg"): image_id
            for image_id in selected_ids
        }
        for future in as_completed(futures):
            image_id = futures[future]
            try:
                fetched_id, size, digest = future.result()
            except Exception as exc:  # pragma: no cover - network dependent
                failures[image_id] = f"{type(exc).__name__}: {exc}"
                continue
            downloads[fetched_id] = {"bytes": size, "sha256": digest}

    if failures:
        raise RuntimeError(f"download failures ({len(failures)}): {failures}")
    if len(downloads) != len(selected_ids):
        raise RuntimeError("download count does not match selected image count")

    manifest: dict[str, Any] = {
        "schema_version": "agu.open-images-ball-subset.v1",
        "purpose": "offline_training_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source": {
            "dataset": "Open Images V7 validation split",
            "source_url": "https://storage.googleapis.com/openimages/web/download_v7.html",
            "annotation_url": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
            "image_index_url": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
            "image_bucket_url": "https://open-images-dataset.s3.amazonaws.com/validation/",
            "annotation_license": "CC-BY-4.0",
            "image_license": "per-image license in source index; verify before redistribution",
            "split": "validation",
            "label_map": LABELS,
            "raw_files": {
                path.name: _sha256(path)
                for path in (annotations_path, urls_path, raw / "class-descriptions-boxable.csv")
                if path.is_file()
            },
        },
        "selection": {
            "max_images": args.max_images,
            "selected_images": len(selected_ids),
            "boxes": sum(len(grouped[image_id]) for image_id in selected_ids),
        },
        "images": [],
    }
    for image_id in selected_ids:
        metadata = url_rows[image_id]
        manifest["images"].append(
            {
                "image_id": image_id,
                "path": f"images/{image_id}.jpg",
                "original_url": metadata["OriginalURL"],
                "landing_url": metadata["OriginalLandingURL"],
                "license": metadata["License"],
                "author": metadata["Author"],
                "title": metadata["Title"],
                "download": downloads[image_id],
                "boxes": [
                    {
                        "label_name": row["LabelName"],
                        "label": LABELS[row["LabelName"]],
                        "xmin": float(row["XMin"]),
                        "xmax": float(row["XMax"]),
                        "ymin": float(row["YMin"]),
                        "ymax": float(row["YMax"]),
                        "is_occluded": int(row["IsOccluded"]),
                        "is_truncated": int(row["IsTruncated"]),
                        "is_group_of": int(row["IsGroupOf"]),
                        "is_depiction": int(row["IsDepiction"]),
                        "is_inside": int(row["IsInside"]),
                    }
                    for row in grouped[image_id]
                ],
            }
        )
    manifest_path = args.root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "images": len(selected_ids),
                "boxes": manifest["selection"]["boxes"],
                "bytes": sum(item["download"]["bytes"] for item in manifest["images"]),
                "sha256": _sha256(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
