# 参考辅助的整场技术统计审核

## 目标与适用范围

该流程面向“有原片，也有人工剪辑、事件明细和球队汇总”的比赛。AGU
不把汇总 CSV 伪装成模型预测，而是把三种证据分开：

1. 将事件明细展开成可追溯的原子事件账本并重算逐球员统计；
2. 用参考剪辑的硬切边界和感知哈希把事件段定位回原片；
3. 由 Codex 对分层样本逐页审核，写入追加式 decision 记录。

该流程适合历史比赛回填、第三方统计对账、训练数据清洗和人工审核提效。
它不代表 AGU 已在没有任何参考资料的未见比赛上达到同等准确率。

## 输入契约

参考目录包含：

- `明细数据表.csv`：`Team, Player, OriginTime, Event, Info, Object`；
- `<队伍>数据.csv`：逐球员和球队汇总；
- `<队伍>/highlight_<球员>.mov`、`missed_<球员>.mov`；
- 队伍高光、防守、违例犯规合集；
- 统计图片仅用于视觉交叉检查，不作为解析主输入。

原片目录按文件名排序后应与明细中的节次一一对应。当前数据是 6 个节次、
6 段原片。所有路径都由 CLI 参数传入；AGU 不保存用户目录、视频或统计原件。

## 关系事件展开

JUSHOOP 明细使用压缩表达，不能逐行直接计数：

- `助攻(A, B, 2分/3分)` 同时产生 A 的助攻和 B 的命中投篮；
- `盖帽(A, B, 2分/3分)` 同时产生 A 的盖帽和 B 的投丢/被盖；
- `抢断(A, B)` 同时产生 A 的抢断和 B 的失误；
- `进攻犯规(A)` 同时产生 A 的犯规和失误；
- 普通投篮、罚球、篮板、失误和犯规按直接事件保留。

每个推导事件保留 `source_row`、`derivation` 和关联球员，避免为了对上汇总而
静默修改数字。当前比赛 630 行明细展开为 605 条可聚合原子事件。

## 视频定位

1. FFmpeg 将原片按可配置 FPS 缩放为 17×16 灰度帧；
2. OpenCV difference hash 形成 256-bit 描述符，BFMatcher 找最近与次近原片帧；
3. FFmpeg scene score 决定参考剪辑的硬切边界；0.5 秒内的连续转场切点合并；
4. 同一原片、相同时间偏移的匹配样本聚类，输出参考段对应的原片起止时间；
5. 无稳定证据的段保持 `needs_review`，禁止强行匹配。

默认参数来自本场敏感性检查，但均可替换：

```text
raw_sample_fps=4.0
reference_sample_fps=1.0
scene_threshold=0.15
minimum_scene_seconds=0.5
max_hamming_distance=56
minimum_hamming_margin=3
```

## 运行

```bash
.venv/bin/python scripts/audit_reference_game.py \
  --reference-dir /path/to/reference \
  --raw-dir /path/to/raw-periods \
  --output-dir analysis_outputs/game-audit \
  --fingerprint

.venv/bin/python scripts/build_reference_codex_review.py \
  --matches-json analysis_outputs/game-audit/reference_raw_matches.json \
  --output-dir analysis_outputs/game-audit/codex_review
```

核心输出：

- `event_ledger.jsonl`：不可变原子事件；
- `player_box_score.json`：逐球员统计与原始参考行；
- `reconciliation.json`：字段级对账和准确率分母；
- `reference_raw_matches.json`：每个参考事件段的原片位置和证据质量；
- `codex_review/review_queue.json`、`codex_decisions.jsonl`：审核包和追加决定；
- Excel 工作簿：摘要、逐球员统计、事件账本、字段对账和待审核段。

## 当前比赛验证结果

| 层级 | 分子/分母 | 结果 | 含义 |
| --- | ---: | ---: | --- |
| 账本字段重算 | 304/304 | 100.00% | 19 名球员、16 个核心计数字段与汇总一致 |
| 参考段定位到原片 | 477/481 | 99.17% | 4 个长段保留人工定位 |
| Codex 分层审核 | 60/60 | 100.00% | 参考帧与原片帧为同一比赛片段 |
| 单侧 95% 精确下界 | — | 95.13% | 对“原片定位正确率”通过 95% 门禁 |

抽样固定种子并分层覆盖：球员高光 15、球员投丢 15、球队高光 10、违例犯规
10、防守 10。零错误时单侧 95% Clopper–Pearson 下界为
`0.05^(1/60)=95.13%`。这比仅报告 60/60 更保守，也明确了准确率的统计范围。

## 可复用开源能力评估

- **FFmpeg/ffprobe**：成熟的视频解码、缩放和场景分数能力；通过可配置可执行文件
  adapter 调用。FFmpeg 最终许可证取决于构建选项，分发镜像时必须记录构建配置和
  LGPL/GPL 边界；AGU 不捆绑权重。
- **OpenCV**：Apache-2.0，复用 `BFMatcher` 和本地数组处理；CPU 离线可运行，
  无需新增在线服务。
- **PySceneDetect**：BSD-3-Clause，可作为更复杂转场的候选 adapter；当前 FFmpeg
  已满足硬切合集，不引入额外长期依赖。
- **CVAT/FiftyOne**：可在大规模人工复核时接入，但本任务的 JSONL decision 和审核页
  已够用，因此保持可选而不绑定业务流程。

## AGU 自研/封装边界

AGU 自研并稳定维护：事件关系展开、账本 schema、逐球员聚合、字段对账、准确率
定义、证据 manifest、审核队列和 decision provenance。视频解码、场景检测和哈希
匹配均为可替换 adapter；失败时 fallback 为“输出待审核段”，而不是猜测。

本方案不修改 FastAPI 路由、服务任务编排、v3 动作模型预处理或现有
`action_proxy_v1` 输出。若未来进入在线 API，路径、阈值和 adapter 选择必须先进入
`app/config.py`、`.env.example`、README 和 API 文档。

## 测试策略与剩余风险

- 单元测试覆盖时间解析、关系事件展开、字段分母、稀疏匹配聚类、短场景强证据
  fallback 和转场切点抑制；
- 整场门禁检查 19 名球员 × 16 字段、481 个参考段和 60 个 Codex 样本；
- 4 个未定位段不会影响已统计数字，但在证据完整性上仍是显式待办；
- 同场固定机位和人工参考资料会显著降低难度。要证明“纯自动、跨比赛 95%”，仍需
  按比赛隔离的盲测、球员身份真值和完整事件级 precision/recall。
