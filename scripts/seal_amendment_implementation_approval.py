#!/usr/bin/env python3
"""Assemble and seal the TASK-0258 Amendment-001 implementation-approval record.

Reads the sealed implementation-scope baseline (must exist first) and the
approval-statement text from APPROVAL-HOWTO.md, then writes the 16-field
`agu.task0258-module-a-amendment-implementation-approval.v1` record.
"""

import datetime
import hashlib
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "analysis_outputs/public_research/task0258_module_a_amendment_approval")
BASELINE = os.path.join(DIR, "implementation_scope_baseline.json")
HOWTO = os.path.join(DIR, "APPROVAL-HOWTO.md")
OUT = os.path.join(DIR, "amendment_implementation_approval.json")


def canon(obj):
    if isinstance(obj, dict):
        return (
            "{"
            + ",".join(
                f"{json.dumps(k, separators=(',', ':'))}:{canon(v)}"
                for k, v in sorted(obj.items(), key=lambda kv: json.dumps(kv[0], separators=(",", ":")))
            )
            + "}"
        )
    if isinstance(obj, list):
        return "[" + ",".join(canon(v) for v in obj) + "]"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if obj is None:
        return "null"
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        return json.dumps(obj, separators=(",", ":"))
    raise TypeError(type(obj))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    # 1. baseline receipts
    base_raw = open(BASELINE, "rb").read()
    base = json.loads(base_raw)
    baseline_receipt = {
        "artifact_sha256": base["artifact_sha256"],
        "file_sha256": sha256_bytes(base_raw),
    }

    # 2. approval statement from HOWTO
    how = open(HOWTO).read()
    m = re.search(r"```text\n(.*?)\n```", how, re.S)
    assert m, "statement fence not found"
    statement = m.group(1)
    statement_hash = sha256_bytes((statement + "\n").encode("utf-8"))

    # 3. fresh review receipt (markdown review: internal == file hash)
    review_raw = open(os.path.join(DIR, "amendment_fresh_review.md"), "rb").read()
    review_hash = sha256_bytes(review_raw)

    obj = {
        "schema_version": "agu.task0258-module-a-amendment-implementation-approval.v1",
        "module_id": "existing-45-temporal-retrospective",
        "repository_root_absolute_path": ROOT,
        "repository_root_device": 16777230,
        "repository_root_inode": 3906523,
        "parent_spec_approval_receipt": {
            "artifact_sha256": "42273c9db3e5c77da8f577ed0e5142e6210faf3caee53f572402229eac71944f",
            "file_sha256": "0eb1759d992b9587ad47bf9ae08793a69bac266ada81d39258919ac61e06e9fb",
        },
        "approved_parent_file_receipts": [
            {
                "filename": "requirement.md",
                "size_bytes": 43568,
                "file_sha256": "5c455408bd9ba730470b66189316578dc935b39a5cdc550c67e2a9c08d3f80d3",
            },
            {
                "filename": "solution.md",
                "size_bytes": 59007,
                "file_sha256": "87f9a447112de8d294ffedbb1adfadb478713d94e00cdcf2e4f98999ea8a857d",
            },
            {
                "filename": "gate-review.md",
                "size_bytes": 19971,
                "file_sha256": "cc52d809f42abba25ccdfb6667325612ce2f8961842f0e375db060ea2b70fa8f",
            },
        ],
        "approved_amendment_file_receipt": {
            "path": "docs/specs/TASK-0258-temporal-canary/amendment-001-postpublication-proof.md",
            "size_bytes": 285628,
            "file_sha256": "d9127300d1f53b4b54b4856358d7e825605b9cceac5315863f31890aaa9a9ed7",
        },
        "amendment_fresh_review_receipt": {
            "artifact_sha256": review_hash,
            "file_sha256": review_hash,
        },
        "implementation_scope_baseline_receipt": baseline_receipt,
        "approval_scope": "amendment_implementation_only",
        "model_execution_authorized": False,
        "module_b_authorized": False,
        "approval_statement_sha256": statement_hash,
        "approved_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    obj["artifact_sha256"] = sha256_bytes(canon(obj).encode("utf-8"))

    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with open(OUT, "wb") as fh:
        fh.write(raw)

    print("approval_statement_sha256 =", statement_hash)
    print("fresh_review_sha256         =", review_hash)
    print("baseline_artifact_sha256    =", baseline_receipt["artifact_sha256"])
    print("baseline_file_sha256        =", baseline_receipt["file_sha256"])
    print("artifact_sha256             =", obj["artifact_sha256"])
    print("file_sha256                 =", sha256_bytes(raw))
    print("approved_at_utc             =", obj["approved_at_utc"])
    print("wrote                       =", OUT)


if __name__ == "__main__":
    main()
