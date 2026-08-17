#!/usr/bin/env python3
"""Write the sealed offline public-research source catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.public_research_datasets import build_public_research_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = build_public_research_catalog()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_count": len(catalog["sources"]),
                "catalog_sha256": catalog["catalog_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
