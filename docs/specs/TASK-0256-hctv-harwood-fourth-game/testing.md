# TASK-0256 Testing

## Artifact verification

| Artifact | Internal or file SHA-256 |
| --- | --- |
| Source manifest file | `8786b64fc02f4d6105acbe097617a733f7276469a60a721b883ac64062d51ded` |
| Selection artifact | `5a7eaa1ecefa295c5d97632e05e7aea2187eafac0082ba6ea6341fc6b372a80f` |
| Review-plan artifact | `87c9b92c548000527f253082c6190ec82022934d9c3db244be32e50505244911` |
| Raw-frame manifest artifact | `0757e339ffd5acb97d5b8f42790cfced2de825e44deea20095abd7bedfeae776` |
| Sealed-review artifact | `5f2219a4cdcc7e8d4e40dde060c8ba82b5070eb6db471c1ce2855cea9d3e5cb6` |
| Cleanup audit file | `bdbdbda2ab81eaf1ae2cbbcb414366d7fb50822ee8c7316fbd1ee03868b4f7ba` |

The strict verifier replayed all 24 plan rows and all 1,536 JPEG size/hash
bindings. v1 and v2 frame indexes, decoded-frame hashes, and JPEG payloads were
identical before the superseded copy was removed; their ordered payload digest
was `7d771e5e569b12762902cada4ab13225b3e778361373c5c7bab6561f98c99d6c`.

## Regression

- Full suite before the final formatting-only pass: `1471 passed, 5 skipped`
  with 15 existing warnings.
- Post-format focused suite: `110 passed, 5 skipped`.
- Scoped Ruff check and Ruff format check: passed.
- Fresh-context archive review: PASS with no Critical or Required finding.
- No AGU training/extraction process remained after verification.

`ffprobe -count_packets` reported 247,504 packets while the manifest/OpenCV
nominal frame count is 247,505. The verifier intentionally accepts a difference
of at most one, and the latest selected frame is 237,832.
