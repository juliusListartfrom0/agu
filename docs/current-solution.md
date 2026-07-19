# AGU 当前独立技术统计方案

日期：2026-07-19
状态：核心代码已具备，准确率目标尚未达成

## 目标与边界

AGU 的正式目标是仅从原始比赛视频识别逐球员两分、三分、篮板、盖帽、助攻和抢断，并让每个统计数字可追溯到视频帧、感知结果、事件关系和模型决策。最终产品目标是独立比赛技术统计正确率不低于 95%；当前 `strict event F1 >= 0.85` 只作为模型集成的中间工程门禁，不代表最终 95% 目标已经完成。

正式推理只能读取原片、共享模型、配置和独立注册的人脸名单。高光、失误/投丢合集、原始技术统计 CSV/图片以及 Codex 的事件答案都不得进入推理进程。预测先冻结并计算 SHA-256，参考真值随后由独立评测步骤读取。

## 角色分工

- AGU 传统模型负责人物、球、篮筐、姿态、轨迹、球权、动作候选、身份图和确定性事件关系。
- AGU 本地 VLM 只审核传统模型裁出的有限事件窗口，判断动作语义、结果和候选 actor；它只能从 AGU 给出的身份候选中选择。
- Codex 只代替人工完成训练/注册标注和离线验收，不作为在线依赖，也不能向 AGU 运行时提供事件、球员或统计答案。
- 参考高光、失误/投丢视频和统计表只用于构建冻结真值、误差分析及验收报告。

## 正式数据流

```text
原片
  -> 检测/跟踪/姿态/球轨迹
  -> 标注人脸名单识别 + 身份去重
  -> 球权与投篮/篮板/防守事件候选图
  -> AGU 本地 VLM 两阶段审核（语义 -> actor）
  -> 因果与完整性门禁
  -> 不可变事件账本
  -> 逐球员六项技术统计
  -> 冻结预测 + 独立原片/真值验收
```

身份图只允许直接、质量达标的人脸匹配，或独立可信的同号码证据传播已注册身份。身体 ReID 可以合并匿名轨迹，但不能把注册姓名传播给没有直接身份依据的轨迹；证据不足必须保持匿名或拒绝出数。

## Codex 标注防火墙

两类 Codex 产物可以进入训练或注册资产，但不能进入正式推理答案：

1. `agu.training-annotation.v1`：候选框、轨迹、持球、出手者、篮板首次控制等训练标签；必须与验收视频哈希不重叠并声明 `runtime_consumable=false`。
2. `agu.annotated-face-list.v1` / `agu.face-gallery.v1`：名单人物的人脸框、同人合并和无效人物拒绝；运行时仍由 AGU 的 YuNet + SFace 从原片独立识别。

## 代码入口与保留范围

- `app/main.py`、`app/analysis/router.py`、`app/analysis/task_manager.py`：稳定 FastAPI 异步任务接口。
- `app/analysis/perception/`、`game_state/`、`box_score/`、`official_*.py`：原片正式感知、候选、推理、账本和评测。
- `app/analysis/face_enrollment.py`、`face_gallery.py`、`official_identity.py`、`identity_graph.py`：名单注册、识别和身份去重。
- `app/analysis/training_annotation.py`、`review/` 与相应脚本：Codex 辅助训练标注和独立验收边界。
- `app/analysis/service.py`、`tracking.py`、`inference.py`、`fusion.py`、`vlm.py`：现有 API 和 v3 动作模型兼容链，也是正式链路复用的传统模型/VLM 基础。
- `train_mac.py`、`dataset.py`、`scripts/gen_*.py`：可复现的传统动作模型训练与数据准备。
- `tests/`、`scripts/verify_harness.py`：契约、泄漏、因果关系、API 和回归门禁。

已删除重复或无正式调用者的 `hybrid_analysis.py`、`train.py`、`app/models/r2plus1d_v2.py`、三个 `manual_test_*` 脚本和根目录 `make_smoke_data.py`。模型权重、视频、数据集和生成结果继续留在忽略目录或本地环境，不进入仓库。

## 开源组件与 AGU 自研边界

- 检测/姿态优先通过 adapter 接入许可清晰的 RTMDet/RTMPose/ONNX Runtime；Ultralytics/YOLO-World 仅作可选研究后端并明确 AGPL/商业许可风险。
- 跟踪、几何和视频 IO 复用 OpenCV；OCR 复用 RapidOCR/PaddleOCR；本地 VLM 通过 Ollama/Qwen adapter 接入。
- AGU 自研并稳定输出 schema、候选融合、身份约束、事件因果图、不可变账本、泄漏防火墙和评测协议，不绑定某一外部模型实现。
- 外部后端不可用或证据不足时回退为候选、`unknown` 或拒绝出数，禁止编造正式统计。

## 当前能力结论

仓库已经具备完整的可复用实现骨架，但尚不能声明可独立完成一场比赛且达到 95%。当前 Game A 早期片段事件定位 F1 为 0.8889，严格球员身份 F1 为 0；独立注册素材对现有动作 owner 的覆盖仅 1/11。第二场素材只有部分真值，不能形成第二个完整六项盲测门禁。

下一步不是让 Codex补运行时答案，而是补齐与验收比赛隔离的全 roster 人脸名单和域内动作训练数据，训练 AGU 自身模型，然后在两场完整、独立比赛上分别报告 strict event F1、actor accuracy、逐字段准确率、比分对账和 unresolved 工作量。只有每场最终正确率达到 95% 才能宣告目标完成。
