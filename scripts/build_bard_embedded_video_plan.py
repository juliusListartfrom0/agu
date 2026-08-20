#!/usr/bin/env python3
"""Build an exact Git-blob BARD embedded-validation video plan."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

from app.analysis.bard_event_state import (
    build_bard_embedded_video_plan,
    parse_bard_benchmark_csv,
    select_bard_embedded_state_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        required=True,
        help="YEAR=PATH",
    )
    tree_source = parser.add_mutually_exclusive_group(required=True)
    tree_source.add_argument("--git-tree", type=Path)
    tree_source.add_argument("--git-repository", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    benchmark_sha256s = {}
    for item in args.benchmark:
        raw = str(item)
        if "=" not in raw:
            raise ValueError("BARD benchmark must use YEAR=PATH")
        year_text, path_text = raw.split("=", 1)
        year = int(year_text)
        path = Path(path_text)
        payload = path.read_bytes()
        benchmark_sha256s[str(year)] = hashlib.sha256(payload).hexdigest()
        rows.extend(
            parse_bard_benchmark_csv(
                payload.decode("utf-8"),
                year=year,
            )
        )
    if args.git_tree is not None:
        tree_payload = json.loads(args.git_tree.read_text(encoding="utf-8"))
        if tree_payload.get("truncated") is not False:
            raise ValueError("BARD Git tree response is incomplete")
        tree_entries = {
            str(row["path"]): row
            for row in tree_payload.get("tree", [])
            if row.get("type") == "blob"
        }
    else:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(args.git_repository),
                "ls-tree",
                "-r",
                args.source_revision,
                "validation/2024/multi",
                "validation/2025/multi",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        tree_entries = {}
        for line in completed.stdout.splitlines():
            metadata, path = line.split("\t", 1)
            mode, kind, sha = metadata.split()
            if mode != "100644" or kind != "blob":
                continue
            tree_entries[path] = {
                "path": path,
                "type": kind,
                "sha": sha,
                "size": 0,
            }
        selected = select_bard_embedded_state_rows(
            rows,
            per_class=args.per_class,
            seed=args.seed,
        )

        def read_size(path: str) -> tuple[str, int]:
            url = (
                "https://raw.githubusercontent.com/GabrieleGiudic/BARD/"
                f"{args.source_revision}/{path}"
            )
            request = Request(
                url,
                method="HEAD",
                headers={"User-Agent": "AGU-offline-research/1.0"},
            )
            with urlopen(request, timeout=60.0) as response:
                size = int(response.headers.get("Content-Length") or 0)
            if size <= 0:
                raise ValueError("BARD raw HEAD omitted the clip size")
            return path, size

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for path, size in pool.map(
                read_size,
                [row.repository_path for row in selected],
            ):
                tree_entries[path]["size"] = size
    plan = build_bard_embedded_video_plan(
        rows,
        tree_entries=tree_entries,
        source_revision=args.source_revision,
        benchmark_sha256s=benchmark_sha256s,
        per_class=args.per_class,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(plan["examples"]),
                "games": len(
                    {row["game_id"] for row in plan["examples"]}
                ),
                "bytes": sum(row["size_bytes"] for row in plan["examples"]),
                "plan_sha256": plan["plan_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
