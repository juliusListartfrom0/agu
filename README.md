# AGU

- [中文版本](#中文版本)
- [English Summary](#english-summary)

## 中文版本

### 项目简介

AGU 是一个可扩展的开源篮球视频分析引擎与 reference pipeline，组合了传统视觉、动作模型、名单人脸识别、本地 VLM、事件账本和独立验收。当前 0.x 定位是垂直领域 engine/toolkit，不宣称为通用视频 AI 框架。

AGU 的目标不是成为小程序后端或完整篮球 SaaS，而是作为可被其他系统调用的分析引擎：

```text
raw game video -> perception and identity -> event graph -> official box score -> sealed evaluation
```

当前仓库提供：

- 基于 FastAPI 的异步视频分析服务。
- 可安装的 `agu-basketball` Python 包和 `agu` CLI；`python -m app.cli` 保持兼容。
- Docker 自托管部署示例。
- 面向 Mac/CPU/MPS 的训练入口 `train_mac.py`。
- 原片独立推理、Codex 辅助训练标注、名单人脸注册去重和严格验收脚本。

当前正式方案、角色边界、入口清单和已知能力状态见 [`docs/current-solution.md`](docs/current-solution.md)。旧的一体化分析入口、重复训练入口和一次性手工测试脚本已经删除；现有 FastAPI、`agu` CLI、v3 动作模型兼容链和 `action_proxy_v1` 只作为稳定兼容层保留，不能冒充正式技术统计。

当前部署目标是 `model_checkpoints/r2plus1d_v3/best.pt`。推理预处理固定匹配 v3 训练分布：OpenCV BGR、resize 到 `112x112`、值域保持 `[0,255]`，不做 RGB 转换、`/255` 或 Kinetics normalize。

### 分析流程

1. 默认把视频切成带 overlap 的 segment，避免动作在切片边界被截断。
2. 在每个 segment 内读取视频并跟踪球员，默认使用 `YOLO` + ByteTrack，也支持 OpenCV legacy trackers。
3. YOLO 默认走 CPU、低帧率和较低 `imgsz` 跟踪；这些加速参数均从 `BASKETBALL_` 配置或请求体读取。
4. 按 `seq_length` 和 `vid_stride` 将球员轨迹切成重叠窗口；分段分析默认可使用更大的 `action_vid_stride` 减少重复推理。
5. 对每个球员窗口运行 R(2+1)D 动作分类，默认优先使用 MPS（可用时）并回退 CPU/CUDA。
6. 对低置信度片段可选调用本地 Ollama VLM 复核，也可对 segment contact sheet 做 VLM audit。
7. 融合模型、VLM 和时序证据，并做 temporal smoothing。
8. 输出 JSON 结果，可选生成标注视频。

### 动作标签

```text
0 block
1 pass
2 run
3 dribble
4 shoot
5 ball in hand
6 defense
7 pick
8 no_action
9 walk
```

### 主要功能

- 异步分析 API：
  - `POST /api/v1/analysis/run` 启动分析任务。
  - `GET /api/v1/analysis/status/{task_id}` 查询任务状态和结果。
- 多种跟踪方式：
  - 默认 `YOLO`，使用 `bytetrack.yaml`。
  - 可配置开源 tracker adapter：`BASKETBALL_TRACKER_BACKEND=bytetrack|botsort|custom`。
  - BoT-SORT/ReID 可通过 `BASKETBALL_YOLO_REID_ENABLED`、`BASKETBALL_YOLO_REID_MODEL` 和自定义 tracker YAML 接入。
  - OpenCV legacy trackers：`CSRT`、`MOSSE`、`KCF` 等。
- 球员身份 embedding：
  - 默认使用 `torchvision_mobilenet_v3_small` 生成 576 维 appearance embedding。
  - 可选 `BASKETBALL_IDENTITY_EMBEDDING_BACKEND=torchreid_osnet_x0_25` 接入本地 `torchreid` OSNet ReID 后端。
  - `sidecar_hsv_hist` 仅作为轻量 fallback/测试后端保留。
- 模型推理：
  - 从 checkpoint 加载 `R(2+1)D-18`。
- 可选 VLM 复核：
  - `off` / `low-confidence` / `always`。
- 默认长视频分段：
  - `segmented_analysis=true`，通过 `segment_duration_sec` 和 `segment_overlap_sec` 控制 segment。
  - `result.long_video.segments[]` 提供分段统计与 VLM audit 状态。
  - `result.player_identity_features[]` 提供局部轨迹模型 appearance embedding 和 continuity 特征。
  - 可选 `BASKETBALL_JERSEY_NUMBER_VLM_ENABLED=true` 后，`player_identity_features[].jersey_number_candidates[]` 会输出 VLM 读取到的球衣号码候选。
  - `result.long_video.players[]` 提供 segment-local 球员动作汇总和 `appearance_continuity_stitch_v2` 的轻量 `global_player_id` 身份候选；身份证据包含整身 embedding、躯干球衣明暗，以及可见正脸的本地 OpenCV 检测/embedding 侧路。
  - `result.long_video.identity_duplicate_candidates[]` 提供疑似重复 `global_player_id` 的合并审核候选，不自动改写统计。
  - duplicate candidate 会使用采样 frame-level bbox 判断同屏硬冲突和重复框重叠。
  - 请求体可传 `confirmed_identity_merges[]`，确认后的聚合统计会输出到 `result.long_video.merged_players[]`，原始 `players[]` 不会被覆盖。
  - 可选 `vlm_identity_merge_enabled=true` 后，VLM 会审核 duplicate candidates，并把高置信 same-player 决策转为 `confirmed_identity_merges[]`。
  - `result.long_video.event_candidates[]` 提供 `block_candidate`、`rebound_candidate`、`steal_candidate` 事件线索，并通过 `owner_candidates[]` 输出候选事件归属球员排名。
  - `result.long_video.identity_graph_summary` 汇总身份图节点、重复候选、确认合并和 VLM merge 决策数量。
- 推理加速：
  - `BASKETBALL_TRACKING_FPS=8.0` 对 YOLO 跟踪做低帧率采样。
  - `BASKETBALL_YOLO_IMGSZ=320` 降低 YOLO 输入尺寸。
  - `BASKETBALL_YOLO_DEVICE=cpu` 避免当前 Mac MPS 跑 YOLO 变慢。
  - `BASKETBALL_R2PLUS1D_DEVICE=mps_if_available` 让 R(2+1)D 在支持时走 MPS。
  - `BASKETBALL_ACTION_VID_STRIDE=24` 用于默认分段分析，减少重复动作窗口。
  - `BASKETBALL_MAX_PLAYERS_PER_SEGMENT=12` 保留出现最稳定的球员轨迹，避免噪声 track 放大推理量。
- 球员技术统计估算：
  - `statistics.points`、`assists`、`rebounds`、`blocks`、`steals`。
  - 当前为 `action_proxy_v1`，不是正式技术统计；`shoot` 会进入 `shot_attempts` / `point_candidate_count`，不会在未确认命中前直接写入正式 `points`。
  - `statistics.status`、`estimated_fields`、`candidate_fields` 会标出哪些字段只是估算、哪些字段仍需事件确认；`points` 需要命中、罚球或比分牌/事件链路确认。
  - `block` 不再直接计入正式 `statistics.blocks`；会先输出 `block_candidate`，等待球/篮筐/投篮或 VLM 确认。
  - `rebound_candidate` 和 `steal_candidate` 由简化球权状态线索生成，仍需球检测、篮筐检测或 VLM/人工确认。
  - `accurate` / `vlm-full` CLI preset 会默认开启 `scoreboard_audit`。审计会先用 OpenCV 搜索比分牌候选帧，再对候选时间进行 burst 采样，分别读取清晰帧和 LED 时序融合帧；只有同一时间至少两次读数一致且跨时间比分不下降时，才在 `long_video.scoreboard_summary` 输出最终比分。
- 输出文件：
  - JSON：`analysis_outputs/*.json`
  - 视频：`output_videos/*.mp4`
- 静态文件路径：
- `/static/outputs`
- `/static/videos`

### 对外调用边界

AGU 仓库只负责篮球视频分析服务，不承载统一 BFF、鉴权、限流、分组聚合或运营后台 API。统一 BFF/API 网关归属 `visual_coach` 的 Rust 重构实现，由 `visual_coach` 调用 AGU 获取分析任务状态与结果。

AGU 当前提供以下分析接口供外部 BFF 调用：

- `POST /api/v1/analysis/tasks`
- `GET /api/v1/analysis/tasks/{id}`
- `GET /api/v1/analysis/tasks/{id}/result`

这些接口复用 `POST /api/v1/analysis/run` 与 `GET /api/v1/analysis/status/{task_id}`，用于兼容外部统一 API 契约。

### 技术栈

- Python / FastAPI / Pydantic-Settings
- PyTorch / TorchVision
- OpenCV / YOLOv8
- NumPy / scikit-learn
- urllib，用于 Ollama HTTP client
- pytest

### 快速开始

安装服务依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-service.txt
cp .env.example .env
```

框架开发或按能力安装：

```bash
pip install -e .
pip install -e ".[api,inference]"
pip install -e ".[tracking-ultralytics]"  # 单独复核 AGPL/商业许可边界
pip install -e ".[ocr]"
agu --version
agu plugins doctor
```

启动 API：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

提交示例请求：

```bash
curl -sS -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "off",
    "generate_video": false,
    "segmented_analysis": false,
    "max_frames": 60
  }'
```

使用 CLI（旧的 `python -m app.cli` 形式继续支持）：

```bash
agu analyze \
  --video examples/lebron_shoots.mp4 \
  --preset accurate \
  --poll \
  --summary \
  --save-result analysis_outputs/example-analysis.json
```

查询任务：

```bash
agu status <task_id>
```

取消或重试任务：

```bash
agu cancel <task_id>
agu retry <failed-or-cancelled-task-id>
```

取消、`max_runtime_sec` 截止时间均在 pipeline 进度边界协作生效；当前任务状态和原始请求保存在内存，服务重启后不提供恢复。

CLI 常用 preset：

| Preset | 用途 | 说明 |
| --- | --- | --- |
| `fast` | 快速冒烟 | 关闭 VLM audit，降低跟踪成本 |
| `accurate` | 常规准确率优先 | 开启低置信 VLM、VLM audit、BoT-SORT/ReID、更高跟踪召回和 scoreboard audit；默认使用 30s segment / 3s overlap / 4 张 segment audit 帧 / 6 个质量与时序稳定性排序后的 scoreboard 候选 |
| `vlm-full` | 全程 VLM 复核 | 在 `accurate` 基础上使用 `vlm_mode=always` 并启用 VLM identity merge |

生成球员截图、证据视频和 roster 汇总：

```bash
python -m app.cli report \
  --analysis-json analysis_outputs/example-analysis.json \
  --video examples/lebron_shoots.mp4 \
  --output-dir analysis_outputs/example-player-reports \
  --dedupe-players \
  --vlm-player-filter \
  --min-roster-score 18
```

用人工事件 CSV 做可重复准确率评测：

```bash
python -m app.cli evaluate \
  --analysis-json analysis_outputs/example-analysis.json \
  --events-csv labels/events.csv \
  --require-player \
  --output-json analysis_outputs/example-eval.json \
  --output-md analysis_outputs/example-eval.md
```

CLI 准确率路线和每阶段架构 review 见 `docs/cli-accuracy-roadmap.md`。

CLI 也可从 TOML/JSON/YAML profile 读取默认值，显式参数优先：

```bash
agu analyze --video examples/lebron_shoots.mp4 --profile profiles/accurate.toml
```

插件通过 `agu.plugins` Python entry point 注册，并声明类型、能力、依赖、版本和可用性。最小示例见 `examples/plugins/minimal_plugin.py`，完整约定见 `docs/extensions.md`。

### 目录结构

```text
.
├── app/
│   ├── main.py               FastAPI app 与生命周期
│   ├── config.py             从 .env 读取配置
│   ├── dependencies.py       全局 model/service 依赖
│   ├── analysis/
│   │   ├── router.py         /api/v1/analysis 路由
│   │   ├── schemas.py        请求/返回 schema
│   │   ├── service.py        分析流程编排
│   │   ├── tracking.py       跟踪与窗口裁剪
│   │   ├── inference.py      R(2+1)D 推理，v3 预处理入口
│   │   ├── motion.py         运动特征
│   │   ├── vlm.py            Ollama VLM 验证
│   │   ├── fusion.py         模型 + VLM 融合 + 平滑
│   │   └── task_manager.py   内存任务状态
│   ├── video/
│   │   └── writer.py         标注视频输出
│   └── models/
│       ├── r2plus1d.py       best.pt 加载器
│       └── preprocessing.py  v3 对齐预处理
├── dataset.py                SpaceJam 数据集包装
├── train_mac.py              推荐训练脚本
├── scripts/                  正式推理、训练标注、评测与维护入口
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

以下目录是运行时数据，默认不入库：

```text
dataset/
model_checkpoints/
analysis_outputs/
output_videos/
```

### 环境与配置

推荐使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements-service.txt
```

依赖文件分工：

| 文件 | 用途 |
| --- | --- |
| `requirements-service.txt` | FastAPI 服务、推理和 Docker 部署 |
| `requirements-training.txt` | 训练、数据处理和实验 |
| `requirements-dev.txt` | 测试、类型检查和开发工具 |
| `requirements.txt` | 历史快照，保留用于兼容旧环境 |

### 配置项

配置通过前缀 `BASKETBALL_` 从环境变量或 `.env` 读取。示例：

```text
BASKETBALL_MODEL_PATH=model_checkpoints/r2plus1d_v3/
BASKETBALL_BASE_MODEL_NAME=best
BASKETBALL_START_EPOCH=0
BASKETBALL_LR=0.0001
BASKETBALL_NUM_CLASSES=10
BASKETBALL_SEQ_LENGTH=16
BASKETBALL_VID_STRIDE=8
BASKETBALL_ACTION_VID_STRIDE=24
BASKETBALL_BATCH_SIZE=8
BASKETBALL_R2PLUS1D_DEVICE=mps_if_available
BASKETBALL_YOLO_DEVICE=cpu
BASKETBALL_TRACKING_FPS=8.0
BASKETBALL_YOLO_IMGSZ=320
BASKETBALL_MAX_PLAYERS_PER_SEGMENT=12
BASKETBALL_TORCH_NUM_THREADS=10
BASKETBALL_PROGRESS_LOG=true
BASKETBALL_ANALYSIS_TIMEOUT_SEC=0.0
BASKETBALL_TRACKER_TYPE=YOLO
BASKETBALL_TRACKER_BACKEND=bytetrack
BASKETBALL_YOLO_TRACKER_CONFIG=
BASKETBALL_YOLO_REID_ENABLED=false
BASKETBALL_YOLO_REID_MODEL=auto
BASKETBALL_IDENTITY_EMBEDDING_BACKEND=torchvision_mobilenet_v3_small
BASKETBALL_IDENTITY_EMBEDDING_WEIGHTS=default
BASKETBALL_IDENTITY_EMBEDDING_DEVICE=mps_if_available
BASKETBALL_IDENTITY_EMBEDDING_BATCH_SIZE=16
BASKETBALL_IDENTITY_EMBEDDING_ALLOW_FALLBACK=true
BASKETBALL_FACE_IDENTITY_BACKEND=opencv_sface_if_available
BASKETBALL_FACE_DETECTION_MODEL_PATH=model_checkpoints/opencv_face/face_detection_yunet_2023mar.onnx
BASKETBALL_FACE_RECOGNITION_MODEL_PATH=model_checkpoints/opencv_face/face_recognition_sface_2021dec.onnx
BASKETBALL_FACE_DETECTION_SCORE_THRESHOLD=0.60
BASKETBALL_FACE_IDENTITY_ALLOW_FALLBACK=true
BASKETBALL_FACE_GALLERY_PATH=
BASKETBALL_FACE_GALLERY_SIMILARITY_THRESHOLD=0.45
BASKETBALL_FACE_GALLERY_MINIMUM_MARGIN=0.08
BASKETBALL_FACE_ENROLLED_MINIMUM_QUALITY=0.65
BASKETBALL_VLM_MODE=low-confidence
BASKETBALL_OLLAMA_MODEL=qwen3-vl:4b
BASKETBALL_OLLAMA_HOST=http://127.0.0.1:11434
BASKETBALL_OLLAMA_TIMEOUT=45.0
BASKETBALL_SCOREBOARD_OCR_BACKEND=rapidocr_if_available
BASKETBALL_SCOREBOARD_OCR_CONFIDENCE=0.75
BASKETBALL_JERSEY_NUMBER_VLM_ENABLED=false
BASKETBALL_JERSEY_NUMBER_VLM_FRAMES=2
BASKETBALL_LOW_CONFIDENCE=0.45
BASKETBALL_HIGH_CONFIDENCE=0.70
BASKETBALL_SMOOTHING_CONFIDENCE=0.60
BASKETBALL_OUTPUT_DIR=analysis_outputs
BASKETBALL_VIDEO_OUTPUT_DIR=output_videos
BASKETBALL_HOST=127.0.0.1
BASKETBALL_PORT=8765
```

球员面部身份默认使用 OpenCV YuNet + SFace 本地适配器。模型权重不进入仓库，分别放到上述两个配置路径；模型缺失时 `opencv_sface_if_available` 会回退到 Haar 检脸与通用外观特征，不影响服务启动。YuNet 与 SFace 可从 OpenCV Zoo 获取，运行时不需要联网。SFace 仅在同一轨迹至少两张脸形成占多数的一致聚类时输出，避免轨迹 ID 切换、背身或背景人物污染面部身份。

官方统计身份去重可使用哈希封存的标注人脸图库。先准备
`agu.annotated-face-list.v1` 清单，再运行 `scripts/build_face_gallery.py` 生成
`agu.face-gallery.v1`；`scripts/build_official_identity_graph.py` 用
`--face-gallery` 及两个人脸模型参数加载图库。AGU 运行时只接收人脸模板，由
YuNet + SFace 自行识别原片轨迹：同一图库身份可合并，不同图库身份禁止合并，
低质量或匹配间隔不足的脸保持匿名。清单必须声明 `benchmark_disjoint: true`；
同一验收比赛的高光、失误、投丢片段和技术统计不得作为该比赛的运行时图库来源。
没有现成人脸清单时，可先用
`scripts/build_face_enrollment_candidates.py` 从独立的注册视频直接检出人脸并生成逐簇审核图，
再用 `scripts/approve_face_enrollment_candidates.py` 将穷尽式的批准、合并和拒绝决定转换为
`agu.annotated-face-list.v1`。这一步只标注“是否为同一个人”并剔除裁判/无效脸，不标注比赛动作；
候选清单和审核决定都由 SHA-256 绑定，未处理完任一候选簇会失败。
来自多个独立注册来源的已审核列表可用 `scripts/merge_annotated_face_lists.py` 合并；
合并决定同样必须绑定所有源清单并穷尽处理每个源身份，避免同一球员在图库中重复注册。
正式个人技术统计应在候选生成时启用
`scripts/build_official_event_candidates.py --require-face-gallery-identity`。该门禁只保留已由封存人脸注册库锚定的 canonical player；注册身份只能通过直接人脸证据或两侧都可信且号码相同的球衣证据扩展，身体 ReID 只能聚合匿名轨迹，不能单独继承注册姓名。匿名轨迹、裁判和图库匹配不确定的人员不能直接进入个人统计。人脸库为空或未提供身份图时命令会失败关闭。
身份图输入还必须来自启用 `player_tracking=true` 的感知产物；无跟踪的通用检测结果会被拒绝，
避免把统一临时 ID 当成球员身份。
注册身份还受 `BASKETBALL_FACE_ENROLLED_MINIMUM_QUALITY` 门禁控制（默认 `0.65`）。
低于该质量的远景或模糊人脸即使余弦分数过线也保持匿名，且不能作为向其他轨迹传播注册身份的直接证据。

```json
{
  "schema_version": "agu.annotated-face-list.v1",
  "benchmark_disjoint": true,
  "annotation_producer": "codex-assisted",
  "persons": [{
    "person_id": "black-player-01",
    "team_id": "raw-dark",
    "samples": [
      {"path": "faces/black-player-01-a.jpg", "bbox": [20, 10, 140, 150]},
      {"path": "faces/black-player-01-b.jpg", "bbox": [18, 12, 138, 152]}
    ]
  }]
}
```

### 启动 API

```bash
cp .env.example .env
./venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

健康检查：

```bash
curl http://127.0.0.1:8765/health
```

本地与线上部署验收方案见 `docs/deploy-and-verify.md`。

API 契约见 `docs/api.md`，模型说明见 `docs/model-card.md`。
权重说明见 `docs/checkpoints.md`，扩展接口见 `docs/extensions.md`。
公开发布来源、许可证、数据集与权重策略见 `docs/release-notes.md` 和 `docs/datasets.md`。

### 提交分析任务

```bash
curl -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "low-confidence",
    "max_frames": 180,
    "generate_video": true,
    "tracker_conf_thres": 0.3,
    "tracker_iou_thres": 0.6,
    "tracker_min_appear_ratio": 0.02,
    "tracker_min_appear_abs": 5,
    "segmented_analysis": true,
    "segment_duration_sec": 15.0,
    "segment_overlap_sec": 2.0,
    "vlm_audit": true
  }'
```

返回示例：

```text
{"task_id":"...","status":"pending","message":"Analysis started asynchronously..."}
```

查询任务状态：

```bash
curl http://127.0.0.1:8765/api/v1/analysis/status/<task_id>
```

### 请求参数

```text
video_path                必填，视频路径
vlm_mode                  low-confidence | off | always
boxes_file                可选，初始 boxes JSON
max_frames                可选，限制读取帧数
generate_video            是否生成标注视频
tracker_conf_thres        默认 0.3
tracker_iou_thres         默认 0.6
tracker_min_appear_ratio  默认 0.02
tracker_min_appear_abs    默认 5
segmented_analysis        默认 true，统一走带 overlap 的分段分析
long_video_mode           兼容字段，true 时也启用分段分析
segment_duration_sec      默认 15.0
segment_overlap_sec       默认 2.0
segment_start_sec         默认 0.0，可用于局部 smoke
segment_end_sec           可选，局部分析结束时间
max_segments              可选，限制分段数量
vlm_audit                 默认 true，对 segment contact sheet 做 VLM audit
vlm_audit_frames          默认 6
confirmed_identity_merges 可选，确认后的 global_player_id 合并列表，用于输出 merged_players
vlm_identity_merge_enabled 可选，使用 VLM 审核 duplicate candidates 并自动生成 confirmed merge
vlm_identity_merge_max_candidates 可选，限制发送给 VLM 的候选数量
vlm_identity_merge_confidence 可选，VLM 合并确认阈值
vid_stride                可选，覆盖默认窗口步长
low_confidence            可选，覆盖低置信度阈值
high_confidence           可选，覆盖高置信度阈值
```

### 训练

推荐训练命令：

```bash
./venv/bin/python train_mac.py \
  --device mps \
  --batch-size 2 \
  --epochs 20 \
  --lr 1e-4 \
  --annotation-path dataset/annotation_dict.json \
  --augmented-path dataset/augmented_annotation_dict.json \
  --video-dir dataset/examples/ \
  --augmented-dir dataset/augmented-examples/ \
  --model-dir model_checkpoints/r2plus1d_v3/ \
  --history-path histories/history_r2plus1d_v3.txt
```

续训：

```bash
./venv/bin/python train_mac.py \
  --resume model_checkpoints/r2plus1d_v3/best.pt \
  --epochs 30
```

常用训练参数：

```text
--accum-steps
--best-metric
--early-stop-patience
--weight-decay
--label-smoothing
--fc-dropout
--no-freeze-bn
--no-class-weights
--no-sampler
--use-augmentation
--force-resplit
```

### 辅助脚本

```bash
./venv/bin/python scripts/check_training.py --history-path histories/history_r2plus1d_v3.txt
./venv/bin/python scripts/gen_augmented.py --minority-only --multiplier 3
./venv/bin/python scripts/gen_splits.py --annotation-path dataset/annotation_dict.json
./venv/bin/python scripts/build_identity_duplicate_report.py \
  --analysis-json analysis_outputs/<analysis-id>.json \
  --output-json analysis_outputs/perf_runs/identity-duplicate-report.json \
  --screenshot-dir analysis_outputs/player_stat_screenshots_20260622 \
  --contact-sheet analysis_outputs/perf_runs/identity-duplicate-review.jpg
./venv/bin/python scripts/build_player_markdown_reports.py \
  --analysis-json analysis_outputs/perf_runs/mov-full-current-20260622-224247.json \
  --video-path path/to/video.mov \
  --output-dir analysis_outputs/player_markdown_reports
```

该脚本除 `index.md`、每个球员的 Markdown 与证据素材外，还会额外输出：

- `roster-summary.json`：面向程序消费的最终 roster 摘要，包含 `global_player_id`、阵营候选、号码候选、得分/篮板/助攻/抢断/盖帽、置信度、support score 与备注。
- `roster-summary.md`：面向人工复核的最终 roster 表格与备注摘要。

如需让 VLM 对绿色框选目标做二次复核，并过滤明确不是球员的结果：

```bash
./venv/bin/python scripts/build_player_markdown_reports.py \
  --analysis-json analysis_outputs/perf_runs/mov-full-current-20260622-224247.json \
  --video-path path/to/video.mov \
  --output-dir analysis_outputs/player_markdown_reports_vlm_filtered \
  --max-players 18 \
  --dedupe-players \
  --vlm-player-filter \
  --require-vlm-player \
  --vlm-model qwen3-vl:4b \
  --vlm-concurrency 2 \
  --vlm-timeout-sec 45 \
  --vlm-cache-path analysis_outputs/player_markdown_reports_vlm_cache.json \
  --vlm-progress
```

全量常态使用建议复用同一个 `--vlm-cache-path`。脚本会在每个 player 的 VLM 判断完成后立即写入缓存；如果中途停止，下一次重跑会跳过已验证过的框选截图。报告截图面向人工审阅时，建议至少使用 `--dedupe-players --max-players 18`；当存在大量噪声短轨迹时，可再叠加 `--min-roster-score 18` 和 `--require-vlm-player`，避免把重复身份或明确非球员框写成独立球员报告。

### 入口状态

| 文件 | 状态 | 建议 |
| --- | --- | --- |
| `app/main.py` | 主服务入口 | 推荐 |
| `python -m app.cli` | API CLI 客户端 | 推荐 |
| `train_mac.py` | 当前训练入口 | 推荐 |
| `scripts/run_official_*.py` | 原片正式候选发现与 AGU 自主推理 | 当前实验入口 |
| `scripts/build_face_*.py` | Codex 辅助名单标注与 AGU 人脸注册 | 离线训练/注册入口 |
| `scripts/evaluate_official_bundle.py` | 冻结预测后的独立验收 | 正式评测入口 |

### 测试

```bash
./venv/bin/python scripts/smoke_open_source.py
./venv/bin/python -m pytest tests -q
./venv/bin/python -m compileall app train_mac.py scripts
./venv/bin/python scripts/validate_open_source_baseline.py
./venv/bin/python scripts/evaluate_public_benchmark.py --strict
./venv/bin/python -m build
```

### 贡献与许可证

- 贡献指南：`CONTRIBUTING.md`
- 许可证：`LICENSE`
- 公开发布说明：`docs/release-notes.md`
- 数据集获取说明：`docs/datasets.md`
- 第三方许可证边界：`THIRD_PARTY_NOTICES.md`
- 文档导航：`docs/README.md`
- 安全策略：`SECURITY.md`
- 版本历史：`CHANGELOG.md`

当前验证状态以 `docs/harness/TASK-BOARD.md` 的最新任务记录为准。

### 安全与运行时约束

- `POST /api/v1/analysis/run` 会校验 `video_path`，防止简单路径穿越。
- 默认仅允许分析仓库目录内的视频；可通过 `BASKETBALL_ALLOWED_VIDEO_ROOTS` 增加逗号分隔的本地目录或容器挂载目录，例如 `/Users/name/Movies,/mnt/videos`。
- 任务状态和结果保存在内存 `TaskManager` 中，进程重启后会丢失。
- 数据集、模型权重和输出目录默认不入库。
- `app/models/preprocessing.py` 已按 v3 部署口径实现：BGR、`112x112`、`[0,255]`，不 `/255`，无 Kinetics normalize。

### 开源发布状态

- 仓库包含正式包元数据、CI、插件诊断、公开 contract benchmark、SBOM 工具和社区治理文件。
- 每次发布仍必须复核第三方依赖、数据集和模型权重的许可证边界。
- 数据集与权重不随仓库分发，需要自行准备 SpaceJam 标注和视频。
- `requirements.txt` 不是严格 lockfile。
- `examples/benchmark/` 只证明公开契约与评测器可复现，不代表生产模型准确率；真实模型发布必须另附带许可数据集上的 IDF1/HOTA、event F1、比分准确率与运行时间。

## English Summary

AGU is an open-source basketball video understanding engine. It focuses on video analysis only: player tracking, action classification, segmented long-video analysis, optional local VLM review, identity evidence, duplicate-identity review candidates, event candidates, and optional annotated outputs. It is not the BFF, authentication layer, rate limiter, product backend, or operations console.

Current API entry points:

- `POST /api/v1/analysis/run` starts an async analysis task.
- `GET /api/v1/analysis/status/{task_id}` returns task progress and result.
- `/api/v1/analysis/tasks*` aliases are kept for external gateway compatibility.

Current identity and statistics behavior:

- Long videos are analyzed through overlapped segments by default.
- `player_identity_features[]` exposes model/fallback body embeddings, torso jersey luminance/dark-ratio features, optional frontal-face embeddings, and continuity evidence. Face detection uses the bundled OpenCV Haar cascade and falls back cleanly when no usable frontal face is visible.
- `long_video.players[]` exposes lightweight `global_player_id` candidates.
- `long_video.identity_duplicate_candidates[]` exposes review-only duplicate-ID merge candidates and does not rewrite statistics automatically.
- `long_video.identity_merge_decisions[]` exposes optional VLM post-processing decisions when `vlm_identity_merge_enabled=true`.
- `long_video.merged_players[]` exposes confirmed-merge statistics when `confirmed_identity_merges[]` is supplied in the request.
- `statistics.points`, `assists`, `rebounds`, `blocks`, and `steals` are action-proxy estimates, not official box-score truth. `shoot` clips are exposed as `shot_attempts` and `point_candidate_count`; `points` remains 0 until made-shot, free-throw, or scoreboard-linked scoring confirmation exists. The `statistics.status`, `estimated_fields`, and `candidate_fields` fields make that contract explicit. Block, rebound, steal, and point evidence should be confirmed through event candidates, owner candidates, ball/rim/possession evidence, VLM, scoreboard audit, or human review.

The experimental official-stat foundation is opt-in with
`BASKETBALL_OFFICIAL_STATS_ENABLED=true`. It provides normalized ball/rim/player
detections, ball trajectories, possession/shot state machines, dependent event
relations, immutable review revisions, deterministic official aggregation, and
raw-only sealed evaluation. It does **not** promote the current action candidates
to official statistics. A compatible basketball detector must be configured via
`BASKETBALL_OFFICIAL_DETECTOR_BACKEND` and
`BASKETBALL_OFFICIAL_DETECTOR_MODEL_PATH`; missing perception evidence yields
`unknown`/review work, never a fabricated count. Ultralytics is an optional
AGPL/commercial-license adapter. Prefer a license-compatible ONNX or Apache-2.0
deployment backend for an open-source release.

Codex is not part of AGU runtime recognition. It may create isolated annotation
truth and acceptance reports only. `scripts/run_official_perception.py` produces
traditional detector/tracker/team candidates;
`scripts/run_official_autonomous_inference.py` asks the configured local Ollama
VLM to adjudicate bounded raw-video windows. The acceptance evaluator must use
`scripts/evaluate_official_bundle.py --require-agu-autonomous`; this rejects any
Codex/human-confirmed event or reference-derived provenance. Configure the path
with `BASKETBALL_OFFICIAL_PLAYER_TRACKING_ENABLED`,
`BASKETBALL_OFFICIAL_PLAYER_TRACKER_CONFIG`, `BASKETBALL_OFFICIAL_VLM_ENABLED`,
`BASKETBALL_OFFICIAL_VLM_CONFIDENCE`, `BASKETBALL_OFFICIAL_VLM_FRAMES`, and
`BASKETBALL_OFFICIAL_VLM_IMAGE_WIDTH`, and
`BASKETBALL_OFFICIAL_VLM_CONTEXT_LENGTH`. Long dense windows use the separate
`BASKETBALL_OFFICIAL_VLM_TIMEOUT` rather than the short clip default.
`BASKETBALL_OFFICIAL_VLM_CONTACT_SHEET` is an experimental layout switch and is
off by default because current qwen3-vl:4b validation increased false shots.
For two-pass VLM actor review, `BASKETBALL_OFFICIAL_ACTION_OWNER_MODEL_PATH`
may point to an experimental hash-sealed AGU action-owner model. The traditional model ranks
AGU-generated player candidates before overlays are rendered; the VLM must
still validate the visible actor, so the model does not prefill an event answer.
Codex-authored training labels are accepted only through a SHA-bound,
acceptance-disjoint training manifest and are recorded in output provenance.
The setting is empty by default: a model must pass both cross-video training
validation and a separate raw-video acceptance gate before production use.
Traditional ball/rim trajectory outcomes take precedence over conflicting VLM
guesses, made shots deterministically reject linked rebound candidates, and a
3-point label requires grounded line/feet/release evidence. The offline
candidate builder also supports `--shooter-lookback-sec` to keep identity
candidates near the resolved shot outcome instead of across a long noisy
cluster. Rebound candidates carry a traditional multi-frame stable-control
score: repeated early ball proximity ranks ahead of a one-frame tip, while the
VLM still decides the visible actor. After the clean semantic pass establishes
an offensive or defensive rebound, the actor pass is restricted to the only
team consistent with the already confirmed missed shot. The optional
`--max-cluster-span-sec` experiment caps sparse ball/rim chains; it is disabled
by default because the 0--60 second acceptance probe improved recall but added
false candidates.

For reusable raw-only identity, build a sealed identity graph after perception
and pass it into candidate construction:

```bash
python scripts/build_official_identity_graph.py \
  --perception perception.json --video game.mov \
  --output identity.json \
  --embedding-backend torchvision_mobilenet_v3_small \
  --embedding-threshold 0.92

python scripts/build_official_event_candidates.py \
  --perception perception.json --video game.mov --game-id game-001 \
  --identity-graph identity.json --output candidates.json
```

The identity artifact is bound to the raw video hash and records embedding and
graph provenance. The optional `torchreid_osnet_x0_25` backend requires a local
`torchreid` installation and weights; keep it optional rather than adding its
heavy runtime dependencies to the base service. Thresholds must be calibrated
on same-time different-player negative pairs. Lowering the cosine threshold to
force a roster-sized identity count can silently merge different players.

- `long_video.scoreboard_summary` is emitted when `scoreboard_audit=true`; CLI `accurate` and `vlm-full` enable it by default. The v3 audit detects complete dark physical panels, tracks camera motion across 13 consecutive frames, and reads three raw phases, two sharpened boundary frames, and one temporal fusion. Install `requirements-ocr.txt` to let the optional offline RapidOCR adapter read large side-score digits first; unavailable or low-confidence OCR falls back to the configured VLM. Results are published only after burst/cross-anchor consensus and cross-time score/clock checks. `inconsistent_scoreboard` means evidence disagreed and no final score was published.

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-service.txt
cp .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Package and framework diagnostics:

```bash
pip install -e ".[api,inference]"
agu --version
agu plugins list
python scripts/evaluate_public_benchmark.py --strict
```

Useful docs:

- API contract: `docs/api.md`
- Model card: `docs/model-card.md`
- Checkpoints: `docs/checkpoints.md`
- Extensions: `docs/extensions.md`
- Open-source release notes: `docs/release-notes.md`
- Harness workflow and task board: `docs/harness/WORKFLOW.md`, `docs/harness/TASK-BOARD.md`
