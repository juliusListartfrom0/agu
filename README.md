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
| 人脸名单识别与身份去重 | 已实现，跨比赛验证中 | 支持 YuNet + SFace 多模板人脸库、审核清单、身份图和严格名单约束；LAL–BOS 已冻结为 19/19，ATL–CHI 有 21/23 个条目，其中 19/23 通过严格质量门槛。 |
| Codex 标注防泄漏 | 已实现 | 训练标注清单必须记录源视频、任务类型和验收包，并进行 benchmark-disjoint 校验。 |
| 真实投篮时序门禁 | 已实现，未晋级 | 四场 118 窗口的端到端留一微调已运行；Kinetics 与 SpaceJam 初始化均未达到逐场 P≥0.95/R≥0.85，权重保持非运行时。 |
| 单场比赛独立完成技术统计 | 尚未达到 | 当前不能宣称可无人干预、稳定完成整场比赛官方技术统计。 |
| 最终正确率 ≥ 95% | 尚未证明 | 两场球队命中分值组件 F1 为 `0.8600`/`0.9703`，但这不是逐球员六项结果；身份映射后的得分事件中，LAL 仅 1 个含唯一具名候选，ATL 为 0。 |

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
  -> 可选稳定广播比分增量 / 原片解说 ASR 候选
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
- 检测与跟踪：Ultralytics YOLO、可选的离线 Transformers RF-DETR、ByteTrack、BoT-SORT；
- 姿态：可配置本地姿态模型；
- 人脸识别：OpenCV YuNet + SFace；
- 动作识别：PyTorch R(2+1)D；
- VLM：Ollama 兼容的本地模型；
- 原片音频：可选 MLX Whisper adapter（仅产生候选，不确认事件或球员）；
- API：FastAPI + Pydantic。

Ultralytics 的 AGPL/商业许可需要部署方自行确认。仓库不绑定线上服务、密钥、固定模型路径或单一 GPU 环境；CPU、MPS、CUDA 和离线模型路径均通过配置层选择。RTMDet、RTMPose 等是可评估的替换方向，并不代表当前已内置对应 adapter。

实验性 RF-DETR adapter 默认关闭。仅当
`BASKETBALL_OFFICIAL_DETECTOR_BACKEND=transformers_rfdetr` 且
`BASKETBALL_OFFICIAL_DETECTOR_MODEL_PATH` 指向已下载的本地 Transformers
模型目录时启用；加载强制离线，不隐式联网。当前该路线只用于候选提议，尚未通过
0.85 精度门禁。

[MLX Examples/`mlx-whisper`](https://github.com/ml-explore/mlx-examples) 代码为 MIT 许可。AGU 不默认指定或再分发 ASR 权重；启用时必须显式配置本地路径或模型仓库并复核该权重页面的许可证。该后端仅为 Apple Silicon 的可选本地 adapter，不进入基础依赖；其他平台可用保持相同 artifact/schema 的 ASR adapter 替换。

### 5. 快速开始

AGU 本地开发与服务固定使用 Python 3.11 的 `.venv`。模型权重、数据集和比赛视频不包含在仓库中。

```bash
git clone <repository-url>
cd agu
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

# 4b. 可选：只在候选窗口运行 YOLO11 姿态，避免整场重复计算
python -m scripts.run_official_pose_windows \
  --candidate-bundle analysis_outputs/game/candidates.json \
  --video /absolute/path/to/game.mp4 \
  --model model_checkpoints/yolo11n-pose.pt \
  --output analysis_outputs/game/pose_windows.json

# 4c. 可选：用冻结的传统真实投篮模型过滤误报，再重建候选
python -m scripts.build_official_event_candidates \
  --perception analysis_outputs/game/perception.json \
  --pose analysis_outputs/game/pose_windows.json \
  --shot-validity-model model_checkpoints/shot_validity.json \
  --video /absolute/path/to/game.mp4 --game-id game_a \
  --output analysis_outputs/game/candidates_gated.json

# 4d. 可选：用冻结的全事件窗口 R(2+1)D 时序头做第二级高精度过滤
python -m scripts.apply_shot_validity_temporal_model \
  --candidate-bundle analysis_outputs/game/candidates_gated.json \
  --video /absolute/path/to/game.mp4 \
  --model model_checkpoints/shot_validity_temporal.json \
  --backbone-checkpoint model_checkpoints/r2plus1d_18-kinetics400.pth \
  --output analysis_outputs/game/candidates_temporal_gated.json

# 5. 可选：从原片广播记分牌提取稳定增量并绑定唯一候选
python -m scripts.run_official_scoreboard_evidence \
  --candidate-bundle analysis_outputs/game/candidates.json \
  --video /absolute/path/to/game.mp4 \
  --team-id HOME --team-id AWAY \
  --sample-interval-sec 2 \
  --cache analysis_outputs/game/scoreboard.cache.json \
  --evidence-output analysis_outputs/game/scoreboard.json \
  --output analysis_outputs/game/candidates_with_scoreboard.json

# 5b. 可选：从同一原片音轨提取动作/名单姓名候选
# audio_roster.json 必须是预先封存的 agu.audio-roster.v1 注册资产；
# 可先用 scripts.build_audio_roster 从 benchmark_answers_included=false 的注册清单确定性生成。
# 重复传入 --candidate-action 可限制“未匹配到视觉候选”的语音补充类型。
# 每条语音提及只绑定到时间上最匹配的一个同类视觉候选，避免重复放大事件。
# 投篮命中/未中措辞仅保存为 speech_shot_outcome_candidate 证据；
# ASR 不会改写 outcome/status/primary_player_id，仍需视觉、因果和身份确认。
# Apple Silicon adapter: .venv/bin/python -m pip install -e '.[audio-mlx]'
python -m scripts.run_official_audio_evidence \
  --enable \
  --model /absolute/path/or/reviewed-model-repository \
  --candidate-bundle analysis_outputs/game/candidates_with_scoreboard.json \
  --video /absolute/path/to/game.mp4 \
  --roster analysis_outputs/game/audio_roster.json \
  --transcript-cache analysis_outputs/game/audio_transcript.json \
  --candidate-action assist --candidate-action block \
  --candidate-action foul --candidate-action steal \
  --candidate-action turnover \
  --evidence-output analysis_outputs/game/audio_evidence.json \
  --output analysis_outputs/game/candidates_with_audio.json

# Codex 离线复核只能封存为 runtime_consumable=false 的开发评估资产；
# scripts.seal_audio_action_review 会校验 bundle 哈希及逐候选完整覆盖，不能作为运行时答案。

# 6. AGU 自主统计；可选本地 VLM 两阶段审核
python -m scripts.run_official_autonomous_inference \
  --candidate-bundle analysis_outputs/game/candidates_with_audio.json \
  --video /absolute/path/to/game.mp4 \
  --two-pass-vlm \
  --output analysis_outputs/game/official_bundle.json

# 语义审核同时要求当前连续直播回合。回放、集锦、中场/演播室、广告或比赛暂停画面
# 即使包含清晰投篮和已注册人脸，也会被拒绝，不能进入正式技术统计。

# 只校准干净原片的事件语义，不执行第二次球员身份 VLM 调用
# 可选 --rim-detail-inset 使用传统篮筐检测生成同帧放大窗；结果仍需冻结后验收
python -m scripts.run_official_autonomous_inference \
  --candidate-bundle analysis_outputs/game/candidates.json \
  --video /absolute/path/to/game.mp4 \
  --semantic-only-vlm \
  --output analysis_outputs/game/semantic_bundle.json

# 7. 冻结后独立验收
python -m scripts.evaluate_official_bundle \
  --bundle analysis_outputs/game/official_bundle.json \
  --truth /absolute/path/to/sealed_truth.csv \
  --fps 30 \
  --require-agu-autonomous \
  --output analysis_outputs/game/evaluation.json
```

`BASKETBALL_OFFICIAL_STATS_ENABLED` 等官方统计配置目前用于实验能力和脚本默认值，不会自动把官方统计包接入 `/analysis/run`。在 API 正式切换前，兼容链路和官方链路必须保持明确区分。

RF-DETR 离线候选筛查也不会自动接入 API；显式选择
`transformers_rfdetr` backend 时，模型路径必须是本地目录。

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

# 可选：从大清单生成哈希绑定的审核子集，避免把未查看的聚类声明为已审核
python -m scripts.select_face_enrollment_candidates \
  --candidates analysis_outputs/faces/candidates.json \
  --cluster-id face-cluster-0001 \
  --cluster-id face-cluster-0002 \
  --output analysis_outputs/faces/review_subset.json

# 2. 应用人工或 Codex 审核决定
python -m scripts.approve_face_enrollment_candidates \
  --candidates analysis_outputs/faces/review_subset.json \
  --decisions /absolute/path/to/decisions.json \
  --output analysis_outputs/faces/approved.json

# 3. 构建可复用多模板人脸库
python -m scripts.build_face_gallery \
  --manifest analysis_outputs/faces/approved.json \
  --detector-model /absolute/path/to/face_detection_yunet.onnx \
  --recognizer-model /absolute/path/to/face_recognition_sface.onnx \
  --output analysis_outputs/game/face_gallery.json

# 4. 可选：把独立注册期的“人-队-号码”标注附加到人脸库
# 标注必须使用 agu.face-jersey-annotation.v1、哈希封存且不含本场事件/统计答案
python -m scripts.attach_face_jersey_annotations \
  --face-gallery analysis_outputs/game/face_gallery.json \
  --jersey-annotations /absolute/path/to/face_jersey_annotations.json \
  --output analysis_outputs/game/face_gallery_with_jersey.json

# 5. 必须通过名单覆盖门禁后才能读取验收原片
python -m scripts.check_face_gallery_coverage \
  --gallery analysis_outputs/game/face_gallery_with_jersey.json \
  --roster /absolute/path/to/active_roster.json \
  --minimum-coverage 0.95 \
  --output analysis_outputs/game/face_gallery_coverage.json
```

验收比赛本身不得用于补录该比赛缺失的球员人脸；否则会破坏独立验收。球衣号码只能作为人脸名单内的注册属性：运行时仍由 AGU OCR/VLM 从原片读取，并要求同队、分区一致和置信度门禁后才能传播该人脸身份。无法严格匹配时应保留 `unknown`，不能用球衣颜色、轨迹顺序或 Codex 猜测代替身份真值。

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

所有本地训练都应通过资源守卫启动。默认每 5 秒记录整机 CPU、内存、可用内存和
训练进程树 RSS；CPU 高于 95%、内存高于 90% 或可用内存低于 2 GiB 连续 3 次时，
守卫只终止自己启动的训练进程组并返回退出码 75。日志为可追溯 JSONL：

```bash
python scripts/run_guarded_training.py \
  --log analysis_outputs/training/resource-monitor.jsonl \
  -- python -m scripts.train_action_owner_model \
    --manifest analysis_outputs/training/manifest.json \
    --annotations /absolute/path/to/action_owner_annotations.json \
    --model-type extra_trees \
    --output model_checkpoints/action_owner.json
```

阈值可按机器能力下调，但不得以直接运行训练脚本的方式绕过监控。一次瞬时峰值会自动
复位，不会触发误杀；关键采样失败则 fail-closed，中止训练而不是无保护继续。

其他训练入口参数如下（实际执行时同样放在守卫的 `--` 之后）：

```bash
python -m scripts.train_action_owner_model \
  --manifest analysis_outputs/training/manifest.json \
  --annotations /absolute/path/to/action_owner_annotations.json \
  --model-type extra_trees \
  --output model_checkpoints/action_owner.json

# 真实投篮标签模板先导出 event_present=null，由 Codex/人工只看训练原片填写
python -m scripts.export_shot_validity_annotations \
  --candidate-bundle analysis_outputs/training/game_a_candidates.json \
  --output analysis_outputs/training/game_a_shot_labels.json

# manifest 的 task-type 必须为 shot_validity；至少两场训练比赛并逐比赛留一校准
python -m scripts.train_shot_validity_model \
  --manifest analysis_outputs/training/shot_manifest.json \
  --candidate-bundle analysis_outputs/training/game_a_candidates.json \
                     analysis_outputs/training/game_b_candidates.json \
  --annotations analysis_outputs/training/game_a_shot_labels.json \
                analysis_outputs/training/game_b_shot_labels.json \
  --minimum-precision 0.95 \
  --output model_checkpoints/shot_validity.json

# 全事件窗口时序方案：先从训练原片提取冻结骨干特征，再训练轻量传统分类头
python -m scripts.extract_shot_validity_temporal_embeddings \
  --manifest analysis_outputs/training/shot_manifest.json \
  --candidate-bundle analysis_outputs/training/game_a_candidates.json \
  --candidate-bundle analysis_outputs/training/game_b_candidates.json \
  --annotation analysis_outputs/training/game_a_shot_labels.json \
  --annotation analysis_outputs/training/game_b_shot_labels.json \
  --video /absolute/path/to/training_game_a.mp4 \
  --video /absolute/path/to/training_game_b.mp4 \
  --backbone-checkpoint model_checkpoints/r2plus1d_18-kinetics400.pth \
  --output analysis_outputs/training/shot_temporal_embeddings.json

python -m scripts.train_shot_validity_temporal_model \
  --embeddings analysis_outputs/training/shot_temporal_embeddings.json \
  --minimum-precision 0.95 \
  --minimum-per-video-recall 0.85 \
  --output model_checkpoints/shot_validity_temporal.json

# 冻结特征不足时，可端到端微调末层；每个 bundle/annotation/video 参数需成对重复
python -m scripts.train_shot_validity_r2plus1d \
  --manifest analysis_outputs/training/shot_manifest.json \
  --candidate-bundle analysis_outputs/training/game_a_candidates.json \
  --candidate-bundle analysis_outputs/training/game_b_candidates.json \
  --annotation analysis_outputs/training/game_a_shot_labels.json \
  --annotation analysis_outputs/training/game_b_shot_labels.json \
  --video /absolute/path/to/training_game_a.mp4 \
  --video /absolute/path/to/training_game_b.mp4 \
  --backbone-checkpoint /absolute/path/to/r2plus1d_checkpoint.pt \
  --checkpoint-output model_checkpoints/shot_validity_r2plus1d.pt \
  --metadata-output analysis_outputs/training/shot_validity_r2plus1d.json
```

禁止把高光、投丢、失误剪辑、CSV 技术统计、Codex 判断或验收真值作为运行时特征、提示词答案或回填规则。

### 10. 配置

配置统一由 `app/config.py` 的 `Settings` 读取，环境变量前缀为 `BASKETBALL_`。新增或修改变量时必须同步 `.env.example` 和相关文档。

主要配置组：

- v3 动作模型、类别映射、序列长度、步长和批量大小；
- 检测器、跟踪器、ReID、姿态和球员身份模型；
- YuNet/SFace 模型路径、人脸库和身份阈值；
- Ollama 地址、VLM 模型、超时、审核和缓存；
- 可选原片 ASR 开关、模型和语言；
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
| Face-list recognition and deduplication | Implemented, cross-game validation in progress | Supports YuNet + SFace multi-prototype galleries, reviewed manifests, identity graphs, and strict roster constraints; LAL–BOS is frozen at 19/19, while ATL–CHI has 21/23 entries and 19/23 pass the strict quality gate. |
| Codex annotation firewall | Implemented | Training manifests record sources, task types, benchmark bundles, and benchmark-disjoint checks. |
| Temporal real-shot gate | Implemented, not promoted | End-to-end game-held-out fine-tuning ran on 118 windows from four games; neither Kinetics nor SpaceJam initialization met per-game P≥0.95/R≥0.85, so the weights remain non-runtime. |
| Independent full-game statistics | Not achieved | The project cannot yet claim reliable unattended official statistics for a complete game. |
| Final accuracy ≥ 95% | Not demonstrated | Team made-value component F1 is `0.8600`/`0.9703` on two games, but this is not a player six-stat result; only one LAL scoring event and no ATL scoring event contains a unique named candidate after identity mapping. |

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
  -> optional stable broadcast-score deltas / raw-commentary ASR candidates
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
- detection and tracking: Ultralytics YOLO, optional offline Transformers
  RF-DETR, ByteTrack, and BoT-SORT;
- pose: configurable local pose models;
- face recognition: OpenCV YuNet + SFace;
- action recognition: PyTorch R(2+1)D;
- VLM: local Ollama-compatible models;
- raw audio: optional MLX Whisper adapter (candidate evidence only);
- API: FastAPI and Pydantic.

Deployers must review Ultralytics AGPL/commercial licensing for their use. The repository does not bind the workflow to hosted services, secrets, fixed model paths, or one GPU environment; CPU, MPS, CUDA, and offline paths are configuration choices. RTMDet and RTMPose are candidate alternatives, not currently bundled adapters.

The experimental RF-DETR adapter is disabled by default. Enable it only with
`BASKETBALL_OFFICIAL_DETECTOR_BACKEND=transformers_rfdetr` and a previously
downloaded local Transformers directory in
`BASKETBALL_OFFICIAL_DETECTOR_MODEL_PATH`; loading is forced offline. It is
currently a candidate proposer only and has not passed the 0.85 precision gate.

[MLX Examples/`mlx-whisper`](https://github.com/ml-explore/mlx-examples) code uses the MIT license. AGU neither selects default ASR weights nor redistributes them; enabling the adapter requires an explicit local path or model repository plus a license review for those weights. This Apple-Silicon backend remains optional and outside base dependencies. Other platforms can provide a replaceable ASR adapter with the same artifact/schema contract.

### 5. Quick start

AGU local development and services use the Python 3.11 `.venv` exclusively. Model weights, datasets, and game footage are not included.

```bash
git clone <repository-url>
cd agu
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

# 4b. Optional: run YOLO11 pose only inside candidate windows
python -m scripts.run_official_pose_windows \
  --candidate-bundle analysis_outputs/game/candidates.json \
  --video /absolute/path/to/game.mp4 \
  --model model_checkpoints/yolo11n-pose.pt \
  --output analysis_outputs/game/pose_windows.json

# 4c. Optional: rebuild candidates through a frozen traditional real-shot gate
python -m scripts.build_official_event_candidates \
  --perception analysis_outputs/game/perception.json \
  --pose analysis_outputs/game/pose_windows.json \
  --shot-validity-model model_checkpoints/shot_validity.json \
  --video /absolute/path/to/game.mp4 --game-id game_a \
  --output analysis_outputs/game/candidates_gated.json

# 4d. Optional: apply the frozen full-window R(2+1)D temporal head
python -m scripts.apply_shot_validity_temporal_model \
  --candidate-bundle analysis_outputs/game/candidates_gated.json \
  --video /absolute/path/to/game.mp4 \
  --model model_checkpoints/shot_validity_temporal.json \
  --backbone-checkpoint model_checkpoints/r2plus1d_18-kinetics400.pth \
  --output analysis_outputs/game/candidates_temporal_gated.json

# 5. Optional: extract stable raw-broadcast score deltas and bind unique candidates
python -m scripts.run_official_scoreboard_evidence \
  --candidate-bundle analysis_outputs/game/candidates.json \
  --video /absolute/path/to/game.mp4 \
  --team-id HOME --team-id AWAY \
  --sample-interval-sec 2 \
  --cache analysis_outputs/game/scoreboard.cache.json \
  --evidence-output analysis_outputs/game/scoreboard.json \
  --output analysis_outputs/game/candidates_with_scoreboard.json

# 5b. Optional: extract action/name candidates from the same raw audio track
# audio_roster.json must be a pre-sealed agu.audio-roster.v1 registration asset.
# It can be deterministically built with scripts.build_audio_roster from a
# registration manifest whose benchmark_answers_included field is false.
# Repeat --candidate-action to limit speech-only additions that have no matching
# visual candidate. Each speech mention attaches only to the best temporal
# overlap of the same action, preventing one mention from amplifying duplicates.
# Shot-result wording is stored only as speech_shot_outcome_candidate evidence;
# ASR never changes outcome, status, or primary_player_id.
# Apple Silicon adapter: .venv/bin/python -m pip install -e '.[audio-mlx]'
python -m scripts.run_official_audio_evidence \
  --enable \
  --model /absolute/path/or/reviewed-model-repository \
  --candidate-bundle analysis_outputs/game/candidates_with_scoreboard.json \
  --video /absolute/path/to/game.mp4 \
  --roster analysis_outputs/game/audio_roster.json \
  --transcript-cache analysis_outputs/game/audio_transcript.json \
  --candidate-action assist --candidate-action block \
  --candidate-action foul --candidate-action steal \
  --candidate-action turnover \
  --evidence-output analysis_outputs/game/audio_evidence.json \
  --output analysis_outputs/game/candidates_with_audio.json

# Offline Codex review may only be sealed as a runtime_consumable=false
# development-evaluation artifact. scripts.seal_audio_action_review verifies
# the bundle hash and exact candidate coverage; it is never a runtime answer.

# 6. AGU autonomous statistics with optional two-pass local VLM review
python -m scripts.run_official_autonomous_inference \
  --candidate-bundle analysis_outputs/game/candidates_with_audio.json \
  --video /absolute/path/to/game.mp4 \
  --two-pass-vlm \
  --output analysis_outputs/game/official_bundle.json

# Semantic review also requires the current continuous live possession. Replays,
# highlight packages, halftime/studio footage, commercials, and game breaks are
# rejected even when a shot and an enrolled face are clearly visible.

# Calibrate clean-frame event semantics without a second actor-identity VLM call
# Optional --rim-detail-inset adds a same-frame crop from traditional rim detections; post-freeze acceptance remains required
python -m scripts.run_official_autonomous_inference \
  --candidate-bundle analysis_outputs/game/candidates.json \
  --video /absolute/path/to/game.mp4 \
  --semantic-only-vlm \
  --output analysis_outputs/game/semantic_bundle.json

# 7. Independent post-freeze acceptance
python -m scripts.evaluate_official_bundle \
  --bundle analysis_outputs/game/official_bundle.json \
  --truth /absolute/path/to/sealed_truth.csv \
  --fps 30 \
  --require-agu-autonomous \
  --output analysis_outputs/game/evaluation.json
```

Official-statistics settings such as `BASKETBALL_OFFICIAL_STATS_ENABLED` currently support experimental capabilities and script defaults. They do not automatically attach the official bundle to `/analysis/run`; keep compatibility and official outputs distinct until the API is deliberately integrated.

The offline RF-DETR candidate screen is likewise not attached to the API.
Selecting the `transformers_rfdetr` backend requires an explicit local model
directory.

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

# Optional: create a hash-bound review subset without claiming unseen clusters were reviewed
python -m scripts.select_face_enrollment_candidates \
  --candidates analysis_outputs/faces/candidates.json \
  --cluster-id face-cluster-0001 \
  --cluster-id face-cluster-0002 \
  --output analysis_outputs/faces/review_subset.json

# 2. Apply human or Codex review decisions
python -m scripts.approve_face_enrollment_candidates \
  --candidates analysis_outputs/faces/review_subset.json \
  --decisions /absolute/path/to/decisions.json \
  --output analysis_outputs/faces/approved.json

# 3. Build a reusable multi-prototype face gallery
python -m scripts.build_face_gallery \
  --manifest analysis_outputs/faces/approved.json \
  --detector-model /absolute/path/to/face_detection_yunet.onnx \
  --recognizer-model /absolute/path/to/face_recognition_sface.onnx \
  --output analysis_outputs/game/face_gallery.json

# 4. Optional: attach benchmark-disjoint person/team/number enrollment labels
# The agu.face-jersey-annotation.v1 input must be hash sealed and contain no event/stat answers
python -m scripts.attach_face_jersey_annotations \
  --face-gallery analysis_outputs/game/face_gallery.json \
  --jersey-annotations /absolute/path/to/face_jersey_annotations.json \
  --output analysis_outputs/game/face_gallery_with_jersey.json

# 5. Require roster coverage before opening benchmark footage
python -m scripts.check_face_gallery_coverage \
  --gallery analysis_outputs/game/face_gallery_with_jersey.json \
  --roster /absolute/path/to/active_roster.json \
  --minimum-coverage 0.95 \
  --output analysis_outputs/game/face_gallery_coverage.json
```

Do not enroll missing players from the benchmark game itself; doing so breaks independent acceptance. A jersey number is only a registered attribute of a face-roster entry: at runtime AGU must read it from the raw video and pass same-team, partition-consensus, and confidence gates before propagating that face identity. A failed strict match must remain `unknown`. Jersey color, track order, or a Codex guess cannot replace identity truth.

Benchmark-disjoint MOT data can also train the anonymous body-ReID backbone. Use
`dataset_track` only when the upstream dataset explicitly keeps track IDs stable
across the selected sequences; otherwise keep the safer `sequence_track` default.

```bash
python scripts/prepare_mot_reid_dataset.py \
  --catalog analysis_outputs/training/teamtrack_catalog.json \
  --dataset-root dataset/public_sources/teamtrack \
  --output-root dataset/public_sources/teamtrack/reid_crops \
  --manifest analysis_outputs/training/teamtrack_reid.json \
  --identity-scope dataset_track

python scripts/train_reid_model.py \
  --manifest analysis_outputs/training/teamtrack_reid.json \
  --crop-root dataset/public_sources/teamtrack/reid_crops \
  --validation-sequence Q4_side_60-90 \
  --minimum-validation-top1 0.85 \
  --output model_checkpoints/reid/teamtrack_mobilenet_v3_small.pt
```

The trainer rejects a checkpoint that does not beat the ImageNet retrieval
baseline or the configured held-out Top-1 gate. Pass an accepted checkpoint path
through `identity_embedding_weights`; the runtime loads only its visual feature
state and hash-bound provenance, never its anonymous training class labels.

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

Run every local training command through the resource guard. By default it
records host CPU, memory, available memory, and supervised process-tree RSS
every five seconds. Three consecutive samples above 95% CPU, above 90% memory,
or below 2 GiB available memory terminate only the supervised training process
group and return exit code 75:

```bash
python scripts/run_guarded_training.py \
  --log analysis_outputs/training/resource-monitor.jsonl \
  -- python -m scripts.train_action_owner_model \
    --manifest analysis_outputs/training/manifest.json \
    --annotations /absolute/path/to/action_owner_annotations.json \
    --model-type extra_trees \
    --output model_checkpoints/action_owner.json
```

The JSONL log is part of the training evidence. Limits may be lowered for a
smaller machine, but training must not bypass the monitor. A recovered transient
resets the breach counter; an unavailable critical sampler fails closed instead
of continuing unprotected.

The remaining commands below show trainer-specific arguments; place the actual
command after the guard's `--` separator.

```bash
python -m scripts.train_action_owner_model \
  --manifest analysis_outputs/training/manifest.json \
  --annotations /absolute/path/to/action_owner_annotations.json \
  --model-type extra_trees \
  --output model_checkpoints/action_owner.json

# Export event_present=null first; Codex/humans label only the training footage
python -m scripts.export_shot_validity_annotations \
  --candidate-bundle analysis_outputs/training/game_a_candidates.json \
  --output analysis_outputs/training/game_a_shot_labels.json

# The manifest task type must be shot_validity; calibration leaves out one game
python -m scripts.train_shot_validity_model \
  --manifest analysis_outputs/training/shot_manifest.json \
  --candidate-bundle analysis_outputs/training/game_a_candidates.json \
                     analysis_outputs/training/game_b_candidates.json \
  --annotations analysis_outputs/training/game_a_shot_labels.json \
                analysis_outputs/training/game_b_shot_labels.json \
  --minimum-precision 0.95 \
  --output model_checkpoints/shot_validity.json

# Full-window temporal path: extract frozen-backbone features from training raw
# footage, then fit a lightweight traditional classifier head.
python -m scripts.extract_shot_validity_temporal_embeddings \
  --manifest analysis_outputs/training/shot_manifest.json \
  --candidate-bundle analysis_outputs/training/game_a_candidates.json \
  --candidate-bundle analysis_outputs/training/game_b_candidates.json \
  --annotation analysis_outputs/training/game_a_shot_labels.json \
  --annotation analysis_outputs/training/game_b_shot_labels.json \
  --video /absolute/path/to/training_game_a.mp4 \
  --video /absolute/path/to/training_game_b.mp4 \
  --backbone-checkpoint model_checkpoints/r2plus1d_18-kinetics400.pth \
  --output analysis_outputs/training/shot_temporal_embeddings.json

python -m scripts.train_shot_validity_temporal_model \
  --embeddings analysis_outputs/training/shot_temporal_embeddings.json \
  --minimum-precision 0.95 \
  --minimum-per-video-recall 0.85 \
  --output model_checkpoints/shot_validity_temporal.json

# When frozen features are insufficient, fine-tune late layers end to end.
# Repeat bundle/annotation/video arguments as aligned triples.
python -m scripts.train_shot_validity_r2plus1d \
  --manifest analysis_outputs/training/shot_manifest.json \
  --candidate-bundle analysis_outputs/training/game_a_candidates.json \
  --candidate-bundle analysis_outputs/training/game_b_candidates.json \
  --annotation analysis_outputs/training/game_a_shot_labels.json \
  --annotation analysis_outputs/training/game_b_shot_labels.json \
  --video /absolute/path/to/training_game_a.mp4 \
  --video /absolute/path/to/training_game_b.mp4 \
  --backbone-checkpoint /absolute/path/to/r2plus1d_checkpoint.pt \
  --checkpoint-output model_checkpoints/shot_validity_r2plus1d.pt \
  --metadata-output analysis_outputs/training/shot_validity_r2plus1d.json
```

Highlight, miss, or turnover clips, CSV box scores, Codex judgments, and benchmark truth must never become runtime features, prompt answers, or correction rules.

### 10. Configuration

`app/config.py` owns configuration through `Settings`, using the `BASKETBALL_` environment prefix. New or changed variables must also update `.env.example` and relevant documentation.

Main groups include:

- v3 action model, class mapping, sequence length, stride, and batch size;
- detector, tracker, ReID, pose, and player-identity models;
- YuNet/SFace paths, face gallery, and identity thresholds;
- Ollama URL, VLM model, timeout, audit, and cache;
- optional raw-audio ASR enablement, model, and language;
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
