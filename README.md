# AGU — Basketball Video Analysis Engine

[中文](#中文) · [English](#english)

AGU 是一个以 Python、FastAPI、传统视觉模型与本地 VLM 为核心的开源篮球视频分析组件。项目同时保留稳定的通用分析 API，并持续建设面向完整比赛技术统计的原片自主识别链路。

AGU is an open-source basketball video analysis component built with Python, FastAPI, conventional vision models, and local VLMs. It keeps a stable general-purpose analysis API while developing an autonomous raw-game pipeline for full box-score statistics.

---

<a id="中文"></a>

## 中文

### 1. 项目定位

AGU 的目标是从比赛原片中自主识别球员、动作和技术统计，并输出可审计的结构化结果。运行时答案必须来自 AGU 自身的传统模型与本地 VLM，不能由 Codex 或人工技术统计直接注入。

Codex 仅承担两个隔离角色：

- 训练前：辅助框选、动作标注和人脸候选审核，并生成可追溯、与验收比赛隔离的标注清单。
- 模型冻结后：对独立原片结果做验收、误差归因和报告，不向推理过程提供答案。

AGU 只提供分析引擎和稳定返回。统一鉴权、限流、跨服务聚合及 BFF/API 网关属于外部 `visual_coach` 项目，不在本仓库实现。

### 2. 当前能力与真实性边界

| 条目 | 当前状态 | 说明 |
| --- | --- | --- |
| FastAPI 异步分析 API | 可用 | 支持创建、查询、取消、重试和结果读取；任务状态保存在进程内存中。 |
| v3 动作分类链路 | 可用，兼容用途 | R(2+1)D 输出动作代理标签；`action_proxy` 不能作为官方技术统计。 |
| 跟踪、姿态、身份与 VLM 审核 | 可用/可配置 | 可按后端和本地模型路径启用；模型权重不随仓库分发。 |
| 原片官方统计离线链路 | 已实现，实验阶段 | 感知、姿态、身份图、事件候选、AGU 自主推理、对账均有独立脚本；尚未接入默认 API 返回。 |
| 人脸名单识别与身份去重 | 已实现，数据待补齐 | 支持 YuNet + SFace 人脸库、审核清单、身份图和严格名单约束；当前比赛名单覆盖不足。 |
| Codex 标注防泄漏 | 已实现 | 训练标注清单必须记录源视频、任务类型和验收包，并进行 benchmark-disjoint 校验。 |
| 单场比赛独立完成技术统计 | 尚未达到 | 当前不能宣称可无人干预、稳定完成整场比赛官方技术统计。 |
| 最终正确率 ≥ 95% | 尚未证明 | 当前一场定位 F1 为 `0.8889`，严格身份正确率为 `0`，跨比赛 owner 覆盖为 `1/11`；还缺第二场完整六项独立真值。 |

这里的最终正确率指冻结后的独立比赛验收结果，而不是训练集精度、单一动作分类准确率或人工/Codex 修正后的结果。内部较低门槛只能用于集成调试，不能替代 95% 产品验收。

### 3. 两条分析链路

#### 3.1 稳定兼容链路

```text
视频
  -> 球员检测与跟踪
  -> v3 R(2+1)D 动作分类
  -> 可选身份合并 / 姿态 / 记分牌审核 / VLM 抽检
  -> AnalysisResponse + JSON + 可选标注视频
```

这条链路服务于现有 API 和集成兼容。其动作结果是 `action_proxy`，不等价于投篮命中、篮板、助攻、抢断、盖帽等官方统计。

#### 3.2 原片官方统计链路

```text
独立比赛原片
  -> 感知包（球员、球、篮筐、轨迹）
  -> 姿态包
  -> 人脸名单 + 身份图
  -> 原片事件候选
  -> AGU 两阶段自主推理（传统模型 + 可选本地 VLM）
  -> 官方统计包
  -> 冻结后独立真值对账
```

当前支持的统计维度为：

- 两分命中/出手、三分命中/出手；
- 进攻篮板、防守篮板；
- 助攻、抢断、盖帽。

高光、投丢、失误视频和已有 CSV 只能用于训练参考、误差分析或冻结后验收；官方推理入口只接收原片产生的候选包，不读取这些答案资产。

### 4. 开源能力与 AGU 边界

AGU 优先以 adapter、配置和 feature flag 封装开源组件：

- 视频 IO：OpenCV、FFmpeg；
- 检测与跟踪：Ultralytics YOLO、ByteTrack、BoT-SORT；
- 姿态：可配置本地姿态模型；
- 人脸识别：OpenCV YuNet + SFace；
- 动作识别：PyTorch R(2+1)D；
- VLM：Ollama 兼容的本地模型；
- API：FastAPI + Pydantic。

Ultralytics 的 AGPL/商业许可需要部署方自行确认。仓库不绑定线上服务、密钥、固定模型路径或单一 GPU 环境；CPU、MPS、CUDA 和离线模型路径均通过配置层选择。RTMDet、RTMPose 等是可评估的替换方向，并不代表当前已内置对应 adapter。

### 5. 快速开始

要求 Python 3.10+。模型权重、数据集和比赛视频不包含在仓库中。

```bash
git clone <repository-url>
cd agu
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

启动服务：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

检查服务：

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/ready
```

提交兼容分析任务：

```bash
curl -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "/absolute/path/to/game.mp4",
    "generate_video": false,
    "vlm_mode": "off"
  }'
```

查询状态：

```bash
curl http://127.0.0.1:8765/api/v1/analysis/status/<task_id>
```

使用 CLI：

```bash
python -m app.cli analyze /absolute/path/to/game.mp4 --preset fast --no-wait
python -m app.cli status <task_id>
python -m app.cli report \
  --analysis-json analysis_outputs/<task_id>.json \
  --video /absolute/path/to/game.mp4 \
  --output-dir analysis_outputs/<task_id>-report
```

CLI 预设：

- `fast`：关闭 VLM，降低跟踪分辨率和采样频率，适合冒烟。
- `accurate`：提高跟踪频率和分辨率，启用 BoT-SORT/ReID、分段审核和记分牌抽检。
- `vlm-full`：在 `accurate` 基础上对所有候选启用 VLM 审核和身份合并。

### 6. API 契约

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 存活检查。 |
| `GET` | `/ready` | 模型和服务就绪检查。 |
| `POST` | `/api/v1/analysis/run` | 创建分析任务。 |
| `GET` | `/api/v1/analysis/status/{task_id}` | 查询任务状态和结果。 |
| `POST` | `/api/v1/analysis/tasks` | 创建任务的兼容别名。 |
| `GET` | `/api/v1/analysis/tasks/{task_id}` | 查询任务的兼容别名。 |
| `GET` | `/api/v1/analysis/tasks/{task_id}/result` | 读取完成结果。 |
| `POST` | `/api/v1/analysis/tasks/{task_id}/cancel` | 协作式取消任务。 |
| `POST` | `/api/v1/analysis/tasks/{task_id}/retry` | 重试失败或已取消任务。 |

`video_path` 必填且必须位于配置允许的根目录中。常用默认值：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `generate_video` | `true` | 生成标注视频。 |
| `vlm_mode` | `low-confidence` | 仅审核低置信度候选；也支持 `off`、`always`。 |
| `segmented_analysis` | `true` | 启用分段处理。 |
| `segment_duration_sec` | `15` | 分段长度。 |
| `segment_overlap_sec` | `2` | 相邻分段重叠。 |
| `vlm_audit` | `true` | 启用任务级 VLM 抽检。 |
| `scoreboard_audit` | `false` | 默认不启用记分牌抽检。 |

完整请求、响应和任务状态说明见 [`docs/api.md`](docs/api.md)。任务注册表当前位于单个服务进程内；重启会丢失任务状态，多实例部署需要外部持久化和调度层。

### 7. 原片官方统计工作流

以下命令生成的包建议写入未纳入 Git 的输出目录。示例路径需要替换为本机实际文件。

```bash
# 1. 原片感知
python -m scripts.run_official_perception \
  --video /absolute/path/to/game.mp4 \
  --model /absolute/path/to/detector.pt \
  --track-players \
  --output analysis_outputs/game/perception.json

# 2. 姿态
python -m scripts.run_official_pose \
  --video /absolute/path/to/game.mp4 \
  --model /absolute/path/to/pose.pt \
  --output analysis_outputs/game/pose.json

# 3. 身份图
python -m scripts.build_official_identity_graph \
  --perception analysis_outputs/game/perception.json \
  --video /absolute/path/to/game.mp4 \
  --face-gallery analysis_outputs/game/face_gallery.json \
  --output analysis_outputs/game/identity_graph.json

# 4. 只从原片产物生成候选
python -m scripts.build_official_event_candidates \
  --perception analysis_outputs/game/perception.json \
  --pose analysis_outputs/game/pose.json \
  --identity-graph analysis_outputs/game/identity_graph.json \
  --video /absolute/path/to/game.mp4 \
  --game-id game_a \
  --require-face-gallery-identity \
  --output analysis_outputs/game/candidates.json

# 5. AGU 自主统计；可选本地 VLM 两阶段审核
python -m scripts.run_official_autonomous_inference \
  --candidate-bundle analysis_outputs/game/candidates.json \
  --video /absolute/path/to/game.mp4 \
  --two-pass-vlm \
  --output analysis_outputs/game/official_bundle.json

# 6. 冻结后独立验收
python -m scripts.evaluate_official_bundle \
  --bundle analysis_outputs/game/official_bundle.json \
  --truth /absolute/path/to/sealed_truth.csv \
  --fps 30 \
  --require-agu-autonomous \
  --output analysis_outputs/game/evaluation.json
```

`BASKETBALL_OFFICIAL_STATS_ENABLED` 等官方统计配置目前用于实验能力和脚本默认值，不会自动把官方统计包接入 `/analysis/run`。在 API 正式切换前，兼容链路和官方链路必须保持明确区分。

### 8. 人脸名单、识别与去重

身份流程以审核过的人脸名单为准：先从非验收片段生成候选，再审核、建库，最后由身份图把轨迹绑定到稳定球员 ID。

```bash
# 1. 从与验收比赛隔离的视频生成候选
python -m scripts.build_face_enrollment_candidates \
  --video /absolute/path/to/enrollment.mp4 \
  --output-dir analysis_outputs/faces/crops \
  --manifest analysis_outputs/faces/candidates.json \
  --detector-model /absolute/path/to/face_detection_yunet.onnx \
  --recognizer-model /absolute/path/to/face_recognition_sface.onnx \
  --benchmark-disjoint

# 2. 应用人工或 Codex 审核决定
python -m scripts.approve_face_enrollment_candidates \
  --candidates analysis_outputs/faces/candidates.json \
  --decisions /absolute/path/to/decisions.json \
  --output analysis_outputs/faces/approved.json

# 3. 构建可复用人脸库
python -m scripts.build_face_gallery \
  --manifest analysis_outputs/faces/approved.json \
  --detector-model /absolute/path/to/face_detection_yunet.onnx \
  --recognizer-model /absolute/path/to/face_recognition_sface.onnx \
  --output analysis_outputs/game/face_gallery.json
```

验收比赛本身不得用于补录该比赛缺失的球员人脸；否则会破坏独立验收。无法严格匹配时应保留 `unknown`，不能用球衣颜色、轨迹顺序或 Codex 猜测代替身份真值。

### 9. Codex 辅助标注防泄漏

Codex 可以替代人工完成框选、动作和人脸候选标注，但只能产出训练资产。所有训练资产在使用前必须封存为带来源和哈希的清单，并声明隔离的验收包：

```bash
python -m scripts.seal_training_annotation_manifest \
  --producer codex \
  --source-video /absolute/path/to/training_game.mp4 \
  --annotation /absolute/path/to/annotations.json \
  --task-type action_owner \
  --benchmark-bundle /absolute/path/to/sealed_benchmark.json \
  --output analysis_outputs/training/manifest.json
```

训练示例：

```bash
python -m scripts.train_action_owner_model \
  --manifest analysis_outputs/training/manifest.json \
  --annotations /absolute/path/to/action_owner_annotations.json \
  --model-type extra_trees \
  --output model_checkpoints/action_owner.json
```

禁止把高光、投丢、失误剪辑、CSV 技术统计、Codex 判断或验收真值作为运行时特征、提示词答案或回填规则。

### 10. 配置

配置统一由 `app/config.py` 的 `Settings` 读取，环境变量前缀为 `BASKETBALL_`。新增或修改变量时必须同步 `.env.example` 和相关文档。

主要配置组：

- v3 动作模型、类别映射、序列长度、步长和批量大小；
- 检测器、跟踪器、ReID、姿态和球员身份模型；
- YuNet/SFace 模型路径、人脸库和身份阈值；
- Ollama 地址、VLM 模型、超时、审核和缓存；
- 官方感知、候选、所有权模型、融合和质量阈值；
- 输入允许根目录、上传目录、JSON/视频输出目录；
- 服务主机、端口、日志级别和并发数。

完整变量、默认值和注释以 [`.env.example`](.env.example) 为准。不要提交 `.env`、密钥、个人绝对路径、比赛数据、模型权重或生成结果。

### 11. 模型训练与 v3 契约

通用动作模型可使用：

```bash
python train_mac.py
```

兼容训练入口仍保留 `train.py`、`dataset.py` 和 `scripts/` 下的数据工具。除非训练、推理和回归测试同时更新，否则必须保持 v3 预处理契约：帧序列、空间变换、归一化、类别映射与 checkpoint 加载方式一致。

训练数据、预训练权重和第三方数据集的许可证必须单独核验；代码开源不代表所有模型权重或数据自动获得相同许可。

### 12. 目录结构

```text
app/
  main.py                  FastAPI 入口与生命周期
  config.py                BASKETBALL_ 配置
  cli.py                   analyze/status/cancel/retry/report/evaluate/plugins
  analysis/                跟踪、身份、推理、融合、VLM、任务编排
  models/                  R(2+1)D 与 v3 预处理
  video/                   标注视频输出
scripts/                   官方统计、标注、训练和评估工具
tests/                     pytest 回归测试
docs/
  api.md                   API 契约
  harness/                 工作流、验收门禁和任务记录
examples/                  轻量示例资源
dataset/                   本地数据（不提交大文件）
model_checkpoints/         本地权重（不提交）
analysis_outputs/          结构化分析产物（不提交）
output_videos/             生成视频（不提交）
```

### 13. 测试与验收

```bash
# 全量回归
pytest

# 按变更范围选择
pytest tests/test_inference.py
pytest tests/test_official_event_candidates.py tests/test_official_autonomous_inference.py

# 仓库门禁
bash scripts/codex_harness.sh verify --scope working-tree
```

服务代码或 API 文档变更还应执行 [`docs/harness/LOCAL-SERVICE-CURL-HOOK.md`](docs/harness/LOCAL-SERVICE-CURL-HOOK.md)：启动服务、检查 `/health` 和 `/ready`、提交轻量任务并查询状态。

官方技术统计的最终验收要求：

- 训练/调参与验收比赛严格隔离；
- 模型、阈值、名单和配置在验收前冻结；
- 至少两场完整、逐球员、六项统计的独立真值；
- 事件定位、身份归属和逐球员统计均有独立指标；
- 最终正确率不低于 95%，且结果来自 AGU 自主推理。

### 14. 文档、贡献与安全

- API：[`docs/api.md`](docs/api.md)
- 本地验收：[`docs/harness/LOCAL-SERVICE-CURL-HOOK.md`](docs/harness/LOCAL-SERVICE-CURL-HOOK.md)
- 开发工作流：[`docs/harness/WORKFLOW.md`](docs/harness/WORKFLOW.md)
- 当前任务状态：[`docs/harness/TASK-BOARD.md`](docs/harness/TASK-BOARD.md)
- 发布记录：[`CHANGELOG.md`](CHANGELOG.md)
- 安全策略：[`SECURITY.md`](SECURITY.md)

提交前请使用小而清晰的变更、补充对应测试并说明模型、checkpoint、预处理和 API 影响。发现安全问题时请按 `SECURITY.md` 私下报告，不要在公开 issue 中提交密钥、私人视频或身份数据。

---

<a id="english"></a>

## English

### 1. Project purpose

AGU aims to identify players, actions, and box-score events autonomously from raw game footage and produce auditable structured results. Runtime answers must come from AGU's own conventional models and local VLMs. Codex or human-authored statistics must never be injected into inference.

Codex has only two isolated roles:

- Before training: assist with boxes, action labels, and face-candidate review, producing traceable annotations kept disjoint from benchmark games.
- After model freeze: audit independent raw-footage results, attribute errors, and produce acceptance reports without supplying answers to inference.

AGU provides only the analysis engine and stable responses. Unified authentication, rate limiting, cross-service aggregation, and the BFF/API gateway belong to the external `visual_coach` project.

### 2. Current capabilities and truth boundary

| Item | Status | Notes |
| --- | --- | --- |
| Asynchronous FastAPI analysis | Available | Create, query, cancel, retry, and read results; task state is process-local memory. |
| v3 action-classification path | Available for compatibility | R(2+1)D emits proxy actions; `action_proxy` is not an official box score. |
| Tracking, pose, identity, and VLM review | Available/configurable | Backends and local model paths are configurable; weights are not distributed. |
| Raw-only official-statistics pipeline | Implemented, experimental | Perception, pose, identity graph, candidates, autonomous inference, and evaluation are separate scripts; not in the default API response. |
| Face-list recognition and deduplication | Implemented, data incomplete | Supports YuNet + SFace galleries, reviewed manifests, identity graphs, and strict roster constraints; current roster coverage is insufficient. |
| Codex annotation firewall | Implemented | Training manifests record sources, task types, benchmark bundles, and benchmark-disjoint checks. |
| Independent full-game statistics | Not achieved | The project cannot yet claim reliable unattended official statistics for a complete game. |
| Final accuracy ≥ 95% | Not demonstrated | Current localization F1 is `0.8889`, strict identity accuracy is `0`, and cross-game owner coverage is `1/11`; a second complete six-stat independent truth set is also missing. |

Final accuracy means evaluation on independent games after freeze. It is not training accuracy, isolated action-classification accuracy, or a result corrected by humans or Codex. Lower internal gates are integration aids only and do not replace the 95% product acceptance gate.

### 3. Two analysis paths

#### 3.1 Stable compatibility path

```text
video
  -> player detection and tracking
  -> v3 R(2+1)D action classification
  -> optional identity merge / pose / scoreboard review / VLM audit
  -> AnalysisResponse + JSON + optional annotated video
```

This path preserves the existing API and integrations. Its `action_proxy` output is not equivalent to official makes, rebounds, assists, steals, or blocks.

#### 3.2 Raw-only official-statistics path

```text
independent raw game
  -> perception bundle (players, ball, rim, tracks)
  -> pose bundle
  -> reviewed face roster + identity graph
  -> raw-only event candidates
  -> AGU two-stage autonomous inference (conventional models + optional local VLM)
  -> official-statistics bundle
  -> post-freeze comparison with independent truth
```

The current statistics schema covers:

- two-point and three-point makes/attempts;
- offensive and defensive rebounds;
- assists, steals, and blocks.

Highlight, miss, and turnover clips plus existing CSV statistics may be used only for training reference, error analysis, or post-freeze evaluation. The official inference entry point consumes raw-footage candidates and does not read those answer assets.

### 4. Open-source capabilities and AGU boundaries

AGU prefers adapters, configuration, and feature flags around open-source components:

- video IO: OpenCV and FFmpeg;
- detection and tracking: Ultralytics YOLO, ByteTrack, and BoT-SORT;
- pose: configurable local pose models;
- face recognition: OpenCV YuNet + SFace;
- action recognition: PyTorch R(2+1)D;
- VLM: local Ollama-compatible models;
- API: FastAPI and Pydantic.

Deployers must review Ultralytics AGPL/commercial licensing for their use. The repository does not bind the workflow to hosted services, secrets, fixed model paths, or one GPU environment; CPU, MPS, CUDA, and offline paths are configuration choices. RTMDet and RTMPose are candidate alternatives, not currently bundled adapters.

### 5. Quick start

Python 3.10+ is required. Model weights, datasets, and game footage are not included.

```bash
git clone <repository-url>
cd agu
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Start the service:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Check service health:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/ready
```

Submit a compatibility analysis task:

```bash
curl -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "/absolute/path/to/game.mp4",
    "generate_video": false,
    "vlm_mode": "off"
  }'
```

Query its status:

```bash
curl http://127.0.0.1:8765/api/v1/analysis/status/<task_id>
```

Use the CLI:

```bash
python -m app.cli analyze /absolute/path/to/game.mp4 --preset fast --no-wait
python -m app.cli status <task_id>
python -m app.cli report \
  --analysis-json analysis_outputs/<task_id>.json \
  --video /absolute/path/to/game.mp4 \
  --output-dir analysis_outputs/<task_id>-report
```

CLI presets:

- `fast`: disables VLM and lowers tracking resolution and sampling for smoke tests.
- `accurate`: increases tracking frequency and resolution and enables BoT-SORT/ReID, segmented audit, and scoreboard sampling.
- `vlm-full`: adds all-candidate VLM review and identity merging to `accurate`.

### 6. API contract

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check. |
| `GET` | `/ready` | Model and service readiness. |
| `POST` | `/api/v1/analysis/run` | Create an analysis task. |
| `GET` | `/api/v1/analysis/status/{task_id}` | Read task status and result. |
| `POST` | `/api/v1/analysis/tasks` | Compatibility alias for task creation. |
| `GET` | `/api/v1/analysis/tasks/{task_id}` | Compatibility alias for task status. |
| `GET` | `/api/v1/analysis/tasks/{task_id}/result` | Read a completed result. |
| `POST` | `/api/v1/analysis/tasks/{task_id}/cancel` | Cooperatively cancel a task. |
| `POST` | `/api/v1/analysis/tasks/{task_id}/retry` | Retry a failed or cancelled task. |

`video_path` is required and must be under a configured allowed root. Common defaults:

| Field | Default | Meaning |
| --- | --- | --- |
| `generate_video` | `true` | Generate an annotated video. |
| `vlm_mode` | `low-confidence` | Review low-confidence candidates; `off` and `always` are also supported. |
| `segmented_analysis` | `true` | Enable segmented processing. |
| `segment_duration_sec` | `15` | Segment duration. |
| `segment_overlap_sec` | `2` | Adjacent-segment overlap. |
| `vlm_audit` | `true` | Enable task-level VLM sampling. |
| `scoreboard_audit` | `false` | Scoreboard sampling is disabled by default. |

See [`docs/api.md`](docs/api.md) for full requests, responses, and task states. The task registry is currently local to one service process; restarts lose task state, and multi-instance deployments require an external persistence and scheduling layer.

### 7. Raw-only official-statistics workflow

Write generated bundles to an ignored output directory. Replace all example paths with local files.

```bash
# 1. Raw-footage perception
python -m scripts.run_official_perception \
  --video /absolute/path/to/game.mp4 \
  --model /absolute/path/to/detector.pt \
  --track-players \
  --output analysis_outputs/game/perception.json

# 2. Pose
python -m scripts.run_official_pose \
  --video /absolute/path/to/game.mp4 \
  --model /absolute/path/to/pose.pt \
  --output analysis_outputs/game/pose.json

# 3. Identity graph
python -m scripts.build_official_identity_graph \
  --perception analysis_outputs/game/perception.json \
  --video /absolute/path/to/game.mp4 \
  --face-gallery analysis_outputs/game/face_gallery.json \
  --output analysis_outputs/game/identity_graph.json

# 4. Candidates derived only from raw-footage artifacts
python -m scripts.build_official_event_candidates \
  --perception analysis_outputs/game/perception.json \
  --pose analysis_outputs/game/pose.json \
  --identity-graph analysis_outputs/game/identity_graph.json \
  --video /absolute/path/to/game.mp4 \
  --game-id game_a \
  --require-face-gallery-identity \
  --output analysis_outputs/game/candidates.json

# 5. AGU autonomous statistics with optional two-pass local VLM review
python -m scripts.run_official_autonomous_inference \
  --candidate-bundle analysis_outputs/game/candidates.json \
  --video /absolute/path/to/game.mp4 \
  --two-pass-vlm \
  --output analysis_outputs/game/official_bundle.json

# 6. Independent post-freeze acceptance
python -m scripts.evaluate_official_bundle \
  --bundle analysis_outputs/game/official_bundle.json \
  --truth /absolute/path/to/sealed_truth.csv \
  --fps 30 \
  --require-agu-autonomous \
  --output analysis_outputs/game/evaluation.json
```

Official-statistics settings such as `BASKETBALL_OFFICIAL_STATS_ENABLED` currently support experimental capabilities and script defaults. They do not automatically attach the official bundle to `/analysis/run`; keep compatibility and official outputs distinct until the API is deliberately integrated.

### 8. Face roster, recognition, and deduplication

Identity is based on an approved face roster: generate candidates from non-benchmark footage, review them, build a gallery, and let the identity graph bind tracks to stable player IDs.

```bash
# 1. Generate candidates from benchmark-disjoint footage
python -m scripts.build_face_enrollment_candidates \
  --video /absolute/path/to/enrollment.mp4 \
  --output-dir analysis_outputs/faces/crops \
  --manifest analysis_outputs/faces/candidates.json \
  --detector-model /absolute/path/to/face_detection_yunet.onnx \
  --recognizer-model /absolute/path/to/face_recognition_sface.onnx \
  --benchmark-disjoint

# 2. Apply human or Codex review decisions
python -m scripts.approve_face_enrollment_candidates \
  --candidates analysis_outputs/faces/candidates.json \
  --decisions /absolute/path/to/decisions.json \
  --output analysis_outputs/faces/approved.json

# 3. Build a reusable face gallery
python -m scripts.build_face_gallery \
  --manifest analysis_outputs/faces/approved.json \
  --detector-model /absolute/path/to/face_detection_yunet.onnx \
  --recognizer-model /absolute/path/to/face_recognition_sface.onnx \
  --output analysis_outputs/game/face_gallery.json
```

Do not enroll missing players from the benchmark game itself; doing so breaks independent acceptance. A failed strict match must remain `unknown`. Jersey color, track order, or a Codex guess cannot replace identity truth.

### 9. Codex-assisted annotation firewall

Codex may replace manual labor for boxes, actions, and face-candidate annotations, but may produce training assets only. Before use, seal every asset into a source- and hash-aware manifest that names its disjoint benchmark bundle:

```bash
python -m scripts.seal_training_annotation_manifest \
  --producer codex \
  --source-video /absolute/path/to/training_game.mp4 \
  --annotation /absolute/path/to/annotations.json \
  --task-type action_owner \
  --benchmark-bundle /absolute/path/to/sealed_benchmark.json \
  --output analysis_outputs/training/manifest.json
```

Training example:

```bash
python -m scripts.train_action_owner_model \
  --manifest analysis_outputs/training/manifest.json \
  --annotations /absolute/path/to/action_owner_annotations.json \
  --model-type extra_trees \
  --output model_checkpoints/action_owner.json
```

Highlight, miss, or turnover clips, CSV box scores, Codex judgments, and benchmark truth must never become runtime features, prompt answers, or correction rules.

### 10. Configuration

`app/config.py` owns configuration through `Settings`, using the `BASKETBALL_` environment prefix. New or changed variables must also update `.env.example` and relevant documentation.

Main groups include:

- v3 action model, class mapping, sequence length, stride, and batch size;
- detector, tracker, ReID, pose, and player-identity models;
- YuNet/SFace paths, face gallery, and identity thresholds;
- Ollama URL, VLM model, timeout, audit, and cache;
- official perception, candidates, ownership models, fusion, and quality gates;
- allowed input roots, upload directory, JSON output, and video output;
- server host, port, logging, and concurrency.

Use [`.env.example`](.env.example) as the complete annotated source of variables and defaults. Never commit `.env`, secrets, personal absolute paths, game data, model weights, or generated outputs.

### 11. Model training and the v3 contract

Train the general action model with:

```bash
python train_mac.py
```

Compatibility entry points remain in `train.py`, `dataset.py`, and data tools under `scripts/`. Preserve the v3 preprocessing contract—frame sequence, spatial transforms, normalization, class mapping, and checkpoint loading—unless training, inference, and regression tests change together.

Licenses for training data, pretrained weights, and third-party datasets require separate review. Open-source code does not automatically grant identical rights to every weight or dataset.

### 12. Repository layout

```text
app/
  main.py                  FastAPI entry point and lifecycle
  config.py                BASKETBALL_ configuration
  cli.py                   analyze/status/cancel/retry/report/evaluate/plugins
  analysis/                tracking, identity, inference, fusion, VLM, tasks
  models/                  R(2+1)D and v3 preprocessing
  video/                   annotated-video output
scripts/                   official statistics, annotation, training, evaluation
tests/                     pytest regression tests
docs/
  api.md                   API contract
  harness/                 workflows, verification gates, task records
examples/                  lightweight example assets
dataset/                   local data; do not commit large files
model_checkpoints/         local weights; do not commit
analysis_outputs/          structured generated artifacts; do not commit
output_videos/             generated videos; do not commit
```

### 13. Tests and acceptance

```bash
# Full regression suite
pytest

# Examples selected by change scope
pytest tests/test_inference.py
pytest tests/test_official_event_candidates.py tests/test_official_autonomous_inference.py

# Repository gate
bash scripts/codex_harness.sh verify --scope working-tree
```

Service-code or API-documentation changes should also run [`docs/harness/LOCAL-SERVICE-CURL-HOOK.md`](docs/harness/LOCAL-SERVICE-CURL-HOOK.md): start the service, check `/health` and `/ready`, submit a lightweight task, and query its status.

Final official-statistics acceptance requires:

- strict separation of training/tuning games and benchmark games;
- frozen models, thresholds, rosters, and configuration before evaluation;
- at least two complete independent truth sets with all six per-player statistic groups;
- separate metrics for localization, identity ownership, and per-player totals;
- final accuracy of at least 95%, produced by AGU-autonomous inference.

### 14. Documentation, contribution, and security

- API: [`docs/api.md`](docs/api.md)
- Local verification: [`docs/harness/LOCAL-SERVICE-CURL-HOOK.md`](docs/harness/LOCAL-SERVICE-CURL-HOOK.md)
- Development workflow: [`docs/harness/WORKFLOW.md`](docs/harness/WORKFLOW.md)
- Current task state: [`docs/harness/TASK-BOARD.md`](docs/harness/TASK-BOARD.md)
- Releases: [`CHANGELOG.md`](CHANGELOG.md)
- Security policy: [`SECURITY.md`](SECURITY.md)

Keep contributions small and explicit, add tests, and document model, checkpoint, preprocessing, and API effects. Report vulnerabilities privately through `SECURITY.md`; do not place secrets, private footage, or identity data in public issues.
