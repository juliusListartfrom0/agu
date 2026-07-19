# AGU 完整技术统计实施方案

日期：2026-07-13
状态：规划完成，待分阶段实施

## 1. 目标与完成定义

AGU 需要从单机位篮球视频中输出逐球员、可审计的：

- 两分球：命中、出手。
- 三分球：命中、出手。
- 篮板：进攻、防守和总数。
- 盖帽。
- 助攻。
- 抢断。

“完整”不表示模型必须猜出每一个事件，而是每个最终统计数字必须能回溯到事件、球员身份、视频帧、传统视觉结果、端侧 VLM 判断和人工复核记录。证据不足时必须输出 `unknown` 或进入审核队列，禁止用动作 clip 数量填充正式统计。

九条 `2026-07-04` MOV 作为第一批开发/校准集；它们不能单独证明跨比赛泛化。正式发布门禁还需要按比赛隔离的验证集和至少一个不同场地、机位、队服的未见测试集。

## 2. 当前基线与核心缺口

AGU 已有：

- 球员检测/跟踪、局部动作分类和长视频分段。
- SFace、衣服明暗、外观和轨迹连续性的身份候选。
- 比分牌多帧 OCR/VLM 对账，九条 MOV 最终比分 9/9。
- `block_candidate`、`rebound_candidate`、`steal_candidate` 和候选 owner 排名。
- 端侧 Ollama VLM、CLI、API、插件和人工证据报告基础。

缺少：

- 球、篮筐/篮网、篮板、球场线和三分线检测。
- 跨遮挡的篮球轨迹以及球与手、篮筐的关系。
- 摄像机运动下的球场单应性和投篮落点。
- 明确的球权状态机、传球链、投篮结果和死球状态。
- 正式事件账本、修订历史和比分不变量。
- 面向 Codex/人工操作员的标准审核包和标注格式。

所以目标架构必须是“事件图驱动的技术统计”，而不是继续扩展 `action_proxy_v1`。

## 3. 总体架构

```text
视频
  -> 场景/机位切分 + 比分牌锚点
  -> 球场标定 + 球员/球/篮筐/姿态检测与跟踪
  -> 球员身份图 + 队伍映射
  -> 球权状态机
  -> 投篮/篮板/盖帽/抢断/助攻候选事件图
  -> 传统规则证据融合
  -> 端侧 VLM 候选审核
  -> Codex 人工复核/手动标注队列
  -> 不可变事件账本
  -> 逐球员技术统计 + 比分/球权一致性报告
```

建议新增模块：

```text
app/analysis/
  perception/
    detector_adapter.py
    ball_tracking.py
    hoop_tracking.py
    court_calibration.py
    pose_adapter.py
  game_state/
    possession.py
    event_graph.py
    shot.py
    rebound.py
    defensive_events.py
    assist.py
  review/
    evidence.py
    edge_vlm.py
    decisions.py
  box_score/
    ledger.py
    reconciliation.py
    aggregation.py
```

公共 Pydantic schema 仍集中在 `app/analysis/schemas.py`，配置仍进入 `app/config.py` 和 `.env.example`，不改变 v3 动作模型预处理契约。

## 4. 感知层设计

### 4.1 目标与跟踪

最小类别：

- `player`
- `basketball`
- `rim`
- `backboard`
- 可选 `referee`

球员沿用当前跟踪/身份链。篮球使用“小目标专用 detector + 卡尔曼/光流 + 前后向轨迹补全”，不能直接复用人体 tracker 参数。篮筐和篮板按镜头段持久跟踪，结合模板匹配和光流抵抗抖动。

运行时通过 adapter 选择：

- `onnxruntime_detector`：推荐的端侧发布口，加载 AGU 自有或许可清晰的 ONNX 权重。
- `mmdet_rtmdet` / `rtmdet_pose`：Apache-2.0 研究与训练选项。
- `ultralytics_yolo`：保留现有快速验证能力，但必须标注 AGPL-3.0/商业许可证边界。
- `opencv_motion_fallback`：仅用于球候选召回和轨迹补点，不能单独确认事件。

### 4.2 姿态与持球关系

姿态不是每帧强制运行。只在候选事件窗口对 shooter、passer、defender、rebounder 做局部推理，重点使用肩、肘、腕、髋、膝、踝关键点：

- 球与手腕距离支持持球/传球/出手释放。
- 双脚和落地点支持三分线内外判定。
- 防守者手腕接近球轨迹支持盖帽候选。

推荐将 [MMPose/RTMPose](https://github.com/open-mmlab/mmpose) 作为 Apache-2.0 可选 adapter；不要在 AGU 内重写完整姿态栈。

### 4.3 球场标定

每个稳定机位段保存 `CourtCalibration`：球场关键点、篮筐、边线、罚球线、三分线、多边形和单应矩阵。

- 固定机位：Codex/人工首帧标 4-8 个关键点，OpenCV `findHomography` 求解。
- 平移/缩放镜头：按 scene 切分，使用 ORB/光流跟踪关键点；重投影误差超阈值时重新进入人工标定。
- 三分判定使用投篮释放时 shooter 脚部/身体投影点；踩线或遮挡则 `shot_value=unknown` 并强制人工复核。

## 5. 球权状态机

全场状态按时间维护：

```text
UNKNOWN
  -> CONTROLLED(team, player)
  -> PASS_FLIGHT
  -> CONTROLLED(team, player)
  -> SHOT_FLIGHT
  -> RIM_CONTACT | MADE_CROSSING | MISS
  -> LOOSE_BALL
  -> CONTROLLED(team, player) | DEAD_BALL
```

每次状态变化保存支持证据和反证。球短暂不可见时允许轨迹预测，但超过时间/空间阈值后回到 `UNKNOWN`，禁止凭最近球员直接延续球权。

内部还需要 `TURNOVER`、`OUT_OF_BOUNDS`、`FOUL/UNKNOWN_STOPPAGE`、`FREE_THROW` 等辅助事件，即使第一版公共统计不输出它们；否则无法可靠区分抢断、普通失误和比分增加 1 分的情况。

## 6. 各统计项目的事件规则

### 6.1 两分/三分投篮

一次正式 FGA 需要：

1. 球从明确持球人的手部区域释放。
2. 球形成朝向篮筐的上升/下降轨迹，或端侧 VLM/Codex 明确确认投篮动作。
3. 球进入篮筐邻域；被盖导致提前终止仍算 FGA。

命中证据优先级：

1. 球心从篮圈平面上方到下方穿越，且有连续帧或篮网运动支持。
2. 对应队伍比分在合理时间窗内增加 2/3，且不存在其他竞争投篮。
3. 端侧 VLM 对短视频判断 `made`，随后通过 Codex/人工审核。

两分/三分判定：

- 使用标定球场中的释放点与三分线关系。
- 踩线、脚被遮挡、机位标定失效时不得猜测。
- 比分增量 2/3 可作为结果约束，但不能在多候选投篮时直接决定 shooter。

输出至少包括 `two_pt_made/attempted`、`three_pt_made/attempted`。罚球作为内部账本项单独处理，避免强行把 1 分差归入两分球。

### 6.2 篮板

正式篮板需要：

1. 前序存在确认的未中 FGA。
2. 球经历篮筐/篮板邻域或明确投篮未中轨迹。
3. 之后第一位建立稳定控制的球员被记为 rebounder。

若 rebounder 与 shooter 同队，为进攻篮板；否则为防守篮板。球直接出界、节末结束或无法确认控制时记录 `team_rebound/unknown`，不硬记到个人。

### 6.3 盖帽

正式盖帽需要：

1. 对手已经形成 FGA。
2. 防守者手/臂在球上升或接近释放后与球轨迹发生空间接触/极近交会。
3. 球轨迹出现可解释的突然速度/方向变化，或 VLM/Codex 明确看到触球。

投篮前把球断下属于抢断/失误，不算盖帽；投篮动作分类本身不能确认盖帽。

### 6.4 抢断

正式抢断需要：

1. 对方有明确控制球权。
2. 防守球员的动作导致球权丢失。
3. 防守方随后取得活球控制。

无防守者贡献的出界、走步、进攻犯规、传球自然失误不记抢断。为此必须先生成内部 `turnover` 候选，再判断是否附带 steal actor。

### 6.5 助攻

正式助攻绑定一个命中 FGM：

1. 找到同队 shooter 得分前最后一次有意传球。
2. 传球后没有另一位队友控制球权。
3. shooter 的持球/运球/单打时间没有超过所选规则集阈值。

助攻具有主观性，必须配置 `stats_ruleset`（例如 conservative-amateur-v1）。第一阶段所有助攻候选都经 Codex/人工复核；指标稳定后，仅高置信简单场景自动确认。

## 7. 事件账本与稳定输出

新增不可变 `GameEventResponse`，建议字段：

```json
{
  "event_id": "evt_000123",
  "revision": 2,
  "event_type": "field_goal_attempt",
  "start_frame": 1230,
  "release_frame": 1244,
  "outcome_frame": 1272,
  "team_id": "dark",
  "primary_player_id": "player_007",
  "secondary_player_id": "player_002",
  "shot_value": 3,
  "outcome": "made",
  "status": "codex_confirmed",
  "confidence": 0.96,
  "evidence_refs": ["ball_track:bt_88", "score_delta:sd_19"],
  "review_refs": ["review:rv_55"]
}
```

允许状态：

- `candidate`
- `vision_confirmed`
- `edge_vlm_confirmed`
- `codex_confirmed`
- `human_confirmed`
- `rejected`
- `needs_review`

最终 `OfficialBoxScoreResponse` 只聚合达到策略要求的事件；原有 `PlayerBoxScoreEstimateResponse/action_proxy_v1` 保留兼容但明确放在 `estimates`，不能混入 official 统计。

逐球员输出：

- `two_pt_made`, `two_pt_attempted`
- `three_pt_made`, `three_pt_attempted`
- `field_goals_made`, `field_goals_attempted`
- `field_goal_points`
- `offensive_rebounds`, `defensive_rebounds`, `rebounds`
- `assists`, `blocks`, `steals`
- `unresolved_event_count`, `review_status`

## 8. 比分和统计不变量

比分 OCR 是约束，不是球员归属来源。必须检查：

- 同队比分不能下降，除非 checkpoint 被判无效。
- 每个确认命中 FGM 对应 2/3 分增量或处于比分不可见区间。
- 可见锚点之间：确认的 FGM 分值 + 罚球/人工未分类分值 = 比分增量。
- 一次 FGM 只能有一个 shooter；一次 miss 最多一个个人 rebounder。
- 盖帽必须关联对手 FGA；助攻必须关联同队 made FGM；抢断必须关联对手 turnover。
- 统计差异只能生成 reconciliation issue，禁止自动篡改事件去“凑比分”。

输出 `BoxScoreReconciliationReport`：每个比分锚点、事件分值、未解释分值、重复事件、身份不确定和人工待办。

## 9. 端侧 VLM 的职责

端侧 VLM 只处理由传统视觉裁出的 3-8 秒候选窗口，不扫描整场找事件。输入包括：

- 原始短视频或 8-16 帧时序图。
- 球、篮筐、球员编号和轨迹叠加层。
- 传统规则生成的候选与冲突。
- 严格 JSON schema 和 `unknown` 选项。

适合判断：命中/未中、是否触球盖帽、谁先控制篮板、传球是否直接创造得分、抢断还是自然失误。VLM 不负责稳定长时身份，也不能单独决定三分线内外。

端侧默认沿用 Ollama adapter；Qwen3-VL 官方实现支持控制视频帧数、FPS 和视觉 token 预算，可用于限定本地成本。每次调用保存模型名、量化、prompt 版本、帧列表、原始响应和解析结果。

自动确认建议门槛：传统硬证据和 VLM 结论一致、VLM 置信度 >= 0.85、无 reconciliation 冲突；否则进入 Codex 队列。

## 10. Codex 作为人工复核员和手动标注员

Codex 不作为 FastAPI 在线依赖，而是运行离线 review workflow。

### 10.1 AGU 为 Codex 生成的审核包

```text
review_package/<game_id>/
  manifest.json
  roster.csv
  court_calibration.json
  candidates.jsonl
  reconciliation.json
  events/<event_id>/clip.mp4
  events/<event_id>/contact_sheet.jpg
  events/<event_id>/frames/*.jpg
  codex_decisions.jsonl
  human_adjudications.jsonl
```

帧上必须标出球员候选 ID、队服深浅、球/篮筐轨迹、释放/碰框/控球帧和比分锚点。Codex 不直接改分析 JSON，而是追加带 reviewer、时间、输入 hash、理由和证据帧的 decision。

### 10.2 Codex 执行的人工操作

1. **机位标定**：在每个新 scene 的关键帧标球场角点、篮筐中心和三分线；输出坐标 JSON。
2. **Roster/身份标注**：从候选截图中合并同一球员，记录队服深浅、球衣号候选和 canonical player ID。
3. **事件标注**：逐候选窗口标 start/release/outcome frame、事件类型、球员、队伍、命中、2/3 分和 secondary actor。
4. **冲突复核**：传统视觉与端侧 VLM 不一致、比分无法对账、身份冲突、踩三分线、球被遮挡时逐帧裁决。
5. **负样本标注**：审核所有高风险假阳性，并从无候选区间抽样“无事件”窗口。
6. **最终对账**：检查逐队比分变化与 accepted events，列出仍无法解释的分值和事件。

Codex decision 示例：

```json
{
  "event_id": "evt_000123",
  "decision": "confirm",
  "labels": {
    "event_type": "field_goal_attempt",
    "outcome": "made",
    "shot_value": 3,
    "primary_player_id": "player_007"
  },
  "evidence_frames": [1244, 1268, 1272],
  "confidence": 0.94,
  "reason": "release point is outside the calibrated arc and the ball crosses the rim plane downward",
  "reviewer_type": "codex",
  "input_sha256": "..."
}
```

### 10.3 人工治理

- Codex 首轮标注后，由用户/第二审核员处理所有分歧和至少 10% 的一致样本抽检。
- 助攻、踩线三分、严重遮挡和无法对账分值在第一阶段必须二审。
- `codex_confirmed` 与 `human_confirmed` 分开保存，不能把 Codex 标签静默当作绝对真值。
- 训练/验证/测试按 game 隔离；Codex 看过的测试标签不能用于调阈值后继续宣称盲测。

## 11. 标注规范与数据闭环

使用 [CVAT](https://github.com/cvat-ai/cvat) 的离线视频标注/API 适配做密集 bbox、轨迹和关键点标注；CVAT Community 核心为 MIT，并支持视频、AI assisted labeling、QA 和 API。使用 [FiftyOne](https://github.com/voxel51/fiftyone) 作为可选的错误样本检索和评测可视化层，不进入 AGU 服务运行时。

必须标注：

- 每个 scene 的 court/rim 关键点。
- 球的可见 bbox/center 和遮挡状态；投篮窗口应尽量逐帧。
- 球员 track/canonical ID、team、可见球衣号。
- 每个事件的关键帧、actors、outcome、shot value 和审核状态。
- 随机负样本、模型/VLM 分歧样本和比分未解释区间。

九条 MOV 的第一轮顺序：

1. 利用已校准 scoreboard checkpoint 找全部分数变化窗口。
2. Codex 标注所有得分窗口和相邻未中/篮板链。
3. 运行传统 detector/state machine 找 block/steal/assist 候选。
4. Codex 审核全部稀有事件和所有冲突。
5. 形成 `ground_truth_events.jsonl` 与逐球员 box score。
6. 锁定阈值后，在未参与调参的新比赛上盲测。

## 12. 可复用开源能力评估

| 能力 | 推荐组件/参考 | AGU 接入方式 | 边界与 fallback |
| --- | --- | --- | --- |
| 球员/球/篮筐检测 | MMDetection/RTMDet、可选 Ultralytics YOLO | detector adapter，发布优先 ONNX Runtime | Ultralytics 为 AGPL/商业双许可；无 detector 时不产 official stats |
| 跟踪/ReID/姿态组合 | [TrackLab](https://github.com/TrackingLaboratory/tracklab) 作为架构参考 | 复用模块边界，不直接绑定其 Hydra runtime | 当前 AGU tracker/identity 保留 fallback |
| 篮球身份数据与流程 | [TrackID3x3](https://github.com/open-starlab/TrackID3x3) | 参考 court、bbox、pose、jersey/color、TI-HOTA 评测 | 其 jersey-number 子组件含非商用许可证，不能直接成为默认依赖 |
| 姿态 | MMPose/RTMPose | 可选 pose adapter | Apache-2.0；不可用时依赖球轨迹并增加人工审核 |
| 比分/球衣 OCR | 当前 RapidOCR；可选 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | OCR adapter、ONNX/Paddle 本地部署 | OCR 失败时 VLM/Codex；模型权重单独记录许可证 |
| 球场/轨迹几何 | OpenCV | AGU 自研薄层：homography、光流、轨迹和规则 | 这是领域融合逻辑，不自研通用 CV 库 |
| 端侧视频审核 | 当前 Ollama + Qwen3-VL | VLM adapter、受限帧/token 和 JSON schema | 失败或冲突进入 Codex；不允许静默猜测 |
| 标注/QA | CVAT Community、FiftyOne | 离线导入导出脚本 | 不进入 FastAPI 服务依赖 |

### AGU 自研/封装边界

AGU 自研：稳定 schema、球权状态机、篮球事件图、比分约束、证据融合、review package、事件账本和 box-score 聚合。

AGU 封装：detector、tracker、ReID、pose、OCR、VLM、CVAT/FiftyOne 导入导出。所有外部能力必须有 feature flag、配置项、availability check、输出归一和测试替身。

## 13. 分阶段实施路线

### P0：真值和契约（先做）

- 增加 GameEvent、ReviewDecision、OfficialBoxScore、Reconciliation schema。
- 定义 JSONL/CSV 标注规范、事件关系和 ruleset。
- 实现 review package、Codex decision 导入、不可变 revision 和 box-score evaluator。
- 对九条 MOV 标全部 scoreboard delta 窗口和一批随机负样本。

退出门禁：同一标签重复导入幂等；每个 box-score 数字可追到 event；人工真值可由 evaluator 重建。

### P1：球/篮筐/球场与球权

- detector adapter、球专用轨迹、篮筐 scene track、手工/自动 court calibration。
- possession state machine 和 evidence schema。
- CVAT 导入导出与 Codex 标定工具。

退出门禁：球候选召回 >= 95%；篮筐 track 覆盖 >= 98%；球场重投影误差中位数 <= 8 px；关键球权转换召回 >= 90%。

### P2：两分/三分和命中/未中

- release、shot flight、rim crossing/contact 和 scoreboard-delta 联合判定。
- 2/3 分 homography 判定与踩线审核。
- made/miss/unknown、free-throw reconciliation-only 支持。

退出门禁：FGA 候选召回 >= 97%；made/miss >= 95%；2/3 分类 >= 95%；全部可见比分锚点无未解释 2/3 分增量。

### P3：篮板、盖帽、抢断

- miss -> loose ball -> control 的 rebound 链。
- defender hand/ball trajectory 的 block 链。
- possession loss -> defender control 的 turnover/steal 链。

退出门禁：候选召回优先达到 95%；Codex/人工确认后的 precision >= 90%；所有正式事件满足关系不变量。

### P4：助攻和正式账本

- pass graph、直接得分关系和可配置 assist ruleset。
- OfficialBoxScore 聚合、事件修订和 reconciliation report。
- API/CLI 增加 `events`, `official_box_score`, `review_queue`，保持旧 estimates 字段兼容。

退出门禁：简单助攻场景 >= 90% precision；主观/复杂场景全部进入审核；逐球员统计能由事件账本确定性重算。

### P5：端侧 VLM、Codex 主动学习和跨比赛验收

- 候选级端侧 VLM 批处理、缓存、prompt/version 审计。
- Codex review workflow、双审、冲突和随机抽检。
- 按不确定性、比分冲突、身份冲突选择下一批标注样本。
- 在未见比赛、场地和队服上盲测。

退出门禁：自动接受事件 precision >= 95%；球员归属 >= 95%；所有目标统计类别宏平均 F1 >= 0.90；人工审核量降到候选事件的 20% 以下，且比分差异为 0 或有明确 unresolved 原因。

## 14. 测试和验收矩阵

- 单元测试：状态转换、shot value、事件关系、账本幂等、revision、比分约束。
- 合成轨迹测试：命中、未中、airball、盖帽、抢断、进攻/防守篮板、助攻/非助攻。
- Adapter contract：detector/pose/OCR/VLM 缺失时明确 fallback，不崩溃、不产伪 official stats。
- Golden clip：每个事件至少 20 个正样本和困难负样本，逐帧标签固定。
- Game-level：按比赛统计 event precision/recall/F1、actor accuracy、IDF1/HOTA、2/3 accuracy、score reconciliation 和 review workload。
- 回归：现有 v3 inference、九 MOV scoreboard 9/9、身份与 API/CLI 测试必须持续通过。

关键指标必须分三层报告：candidate recall、自动确认 precision、人工复核后最终准确率。禁止只报告人工修正后的结果来掩盖自动链路质量。

## 15. 风险与应对

- **手机单机位看不到球**：提高候选召回、前后向轨迹、VLM/Codex；仍不可见则 unresolved。
- **同色队服和严重遮挡**：持续身份图、球衣号和人工 roster；统计暂挂 local track，不错误合并。
- **镜头移动导致三分误判**：scene 级标定和重投影门禁；踩线/失效强制审核。
- **比分牌只显示偶发帧**：沿用 burst OCR/VLM，扩大 score-delta 候选窗口；不把缺失锚点当作 miss。
- **助攻规则主观**：固定 ruleset、保存理由和二审。
- **稀有 block/steal 数据不足**：以高 recall 候选 + Codex 全审启动，主动学习补困难样本。
- **许可证污染**：所有模型/权重和工具经 adapter/optional extra；Ultralytics 和非商用 jersey pipeline 不进入默认 MIT 分发。

## 16. 第一批可执行任务

1. TASK-A：冻结 GameEvent/ReviewDecision/OfficialBoxScore/Reconciliation schema。
2. TASK-B：实现 `build_event_review_package.py`、`apply_review_decisions.py` 和 `evaluate_box_score.py`。
3. TASK-C：Codex 对九 MOV 做 scene/court/roster 初始标定并标全部 scoreboard delta 窗口。
4. TASK-D：训练/接入 ball+rim detector adapter，输出统一 detection/track schema。
5. TASK-E：实现 possession/shot 状态机，先只发布 shot candidate，不改正式统计。
6. TASK-F：达到 P2 门禁后再依次开放 rebound、block、steal、assist。

顺序不能倒置：没有真值、事件账本和 review workflow 时，继续增加启发式计数只会产生不可验证的“完整技术统计”。
