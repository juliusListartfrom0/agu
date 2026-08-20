# Third-Party Notices and Distribution Boundaries

AGU source code is MIT licensed. That does not automatically relicense optional
dependencies, model weights, datasets, or generated artifacts. Distributors and
deployers must review the exact versions and assets they use.

| Component | AGU use | Upstream license/boundary | Distribution policy |
| --- | --- | --- | --- |
| PyTorch / torchvision | Action and identity inference | BSD-style upstream licenses; pretrained weights can have separate terms | Optional `inference` extra; record weight origin and checksum |
| OpenCV | Video IO, tracking helpers, face adapters | Apache-2.0 upstream; model files need separate provenance | Optional `inference` extra |
| TransNetV2 / transnetv2-pytorch | Offline scene-transition research | MIT source and checkpoint port; verify the exact downloaded artifact | Research-only generated asset; not an AGU runtime dependency |
| Ultralytics | YOLO detection, ByteTrack, BoT-SORT | AGPL-3.0 or Ultralytics Enterprise License | Isolated in `tracking-ultralytics`/`service`; never imply MIT relicensing |
| RF-DETR | Optional independent object-detection evidence | Apache-2.0 package; fine-tuned checkpoint terms remain separate | Isolated in `detection-rfdetr`; record checkpoint source, license, and checksum |
| ortizeg basketball DEIM-M | Offline basketball ball-detector transfer diagnostic | Apache-2.0 model/card and evaluation code; upstream basketball-player-detection-3 card declares CC BY 4.0 | Weight was SHA-verified, screened, then deleted; retain only provenance and non-promotable results |
| RapidOCR ONNX Runtime | Optional scoreboard OCR | Apache-2.0 package; bundled/downloaded models require provenance review | Optional `ocr` extra |
| Ollama and configured VLM | Optional local audit | Runtime/model-specific licenses | AGU does not distribute the service or model |
| MLX-VLM | Apple Silicon offline VLM conversion/inference | MIT | Optional `vlm-mlx` extra; not required by the AGU service runtime |
| BARD / E-BARD | Basketball action/detection research and training assets | CC BY 4.0 repositories and annotations; referenced broadcast media rights are separate | Record exact revision and media authorization; do not redistribute NBA footage |
| EBQwen2.5-VL-3B | Offline independent basketball-VLM screening | CC BY 4.0 model card/weights; base-model and training-media provenance must also be reviewed | Not distributed or runtime-enabled by AGU; both multi-image GGUF and native-video MLX screens failed promotion |
| Basketball-51 | Offline basketball clip pretraining screen | Kaggle page declares Apache-2.0 for the dataset; included NBA broadcast footage has separate rights that are not granted by that declaration | Local training-only subset; do not redistribute clips or treat them as runtime/benchmark evidence |
| SpaceJam or user video | Training/evaluation input | Dataset/user-specific terms | Not distributed by AGU |
| YuNet / SFace model files | Optional face evidence | Model-file terms must be checked at download time | Configure local paths; do not bundle without a provenance record |

Before a release:

1. Generate an environment inventory with
   `python scripts/generate_sbom.py --output build/sbom.json`.
2. Run `pip-audit` for known Python dependency vulnerabilities.
3. Review every shipped checkpoint/model file independently of code licenses.
4. Record source URL, version, license, checksum, and redistribution decision.
5. Do not publish private video, generated identity crops, or personal data.

This document is an engineering inventory, not legal advice.

## 2026-07-24 basketball VLM research assets

- BARD source: `GabrieleGiudic/BARD`, revision
  `add4109bf8b2034a32f3b6a83fa0d8c0ae638473`, CC BY 4.0.
- E-BARD source: `GabrieleGiudic/E-BARD`, revision
  `f8314e03179b6558fb6ef0d26a9135fdc3572b78`, CC BY 4.0.
- Official EBQwen2.5-VL-3B source: `GabrieleGiudici/EBQwen2.5-VL-3B`,
  revision `c6a93cbb325f9d20236f85bcaba7827a2808443e`, CC BY 4.0.
- Prior multi-image screening quantization source:
  `mradermacher/EBQwen2.5-VL-3B-GGUF`, revision
  `3942c6f942da55aedfd701bb29f1a37578aee942`. The Q4_K_S GGUF is
  1,834,385,792 bytes with SHA-256
  `ad046a7ba24bdc6023d79ecc9d8da9b7aee95251b9105cf93c1472136c734651`;
  the Q8 vision projector is 847,769,984 bytes with SHA-256
  `fee50a9553570d8522a0947b98af15af1c17f586b4e5def8c8a6f2978eb72858`.
  The rejected GGUF/projector copies were removed from local storage; these
  hashes remain only as screening provenance.
- The official BF16 checkpoint was also pinned at the same official revision
  and locally converted with `mlx-vlm==0.6.7` / `mlx==0.32.0` to affine
  4-bit, group-size 64. The derived `model.safetensors` is 3,073,721,056
  bytes with SHA-256
  `f2b3cf74e088a06b7bf1c6462664ca13f95e1c1c8be4c6c8a0e469f68dd1d72c`.
  MLX-VLM is MIT licensed; the converted model remains governed by the
  upstream model terms and is not redistributed by AGU.
- Scope: local, offline compatibility screening only. Ollama presents sampled
  frames as multiple images; that is not equivalent to the official native
  Qwen video-temporal input. A separate MLX native-video screen also failed its
  precision gate. Neither path is an AGU runtime dependency or answer source.

## 2026-07-27 Basketball-51 research subset

- Source: Kaggle `sarbagyashakya/basketball-51-dataset`; the dataset page
  declares Apache-2.0. The archive contains NBA broadcast clips, so that
  metadata does not itself grant rights to redistribute the underlying video.
- AGU downloaded a bounded, deterministic local subset by ZIP byte range:
  256 six-second clips, 32 from each of the eight labels and all 51 source
  groups. The sealed subset-manifest SHA-256 is
  `7f4fd24ab7a603c38a7622330694d42bf66305337522e75ff6c8c678ecce827e`.
- The clips and derived embeddings are training-only,
  `runtime_consumable=false`, excluded from blind benchmarks, and not shipped
  by AGU.
