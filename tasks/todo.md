# 提升 SearchSession 处理帧率

计划：`~/.claude/plans/rtsp-admin-kaiya-4012-192-168-30-26-8554-reflective-snail.md`

## 背景

现网 RTSP 源稳定 25 FPS，处理循环却被压到 ~1.05 FPS，低于确认窗口所需的最低采样密度
（≥80px 需 ≈1.33Hz，64–79px 需 ≈1.5Hz），远脸确认退化成偶发事件。

## 任务

- [x] 1. `config.py`：新增预算/ROI/预览调优项，`roi_max_tracks_per_pass` 8→3
- [x] 2. `domain.py`：`FaceObservation.embedding` 可空；`SearchMetrics` 新增指标字段
- [x] 3. `backends.py`：`FaceBackend` 拆成 `detect_faces` / `embed_faces` / `analyze`
- [x] 4. `tests/conftest.py`：`FakeFaceBackend` 补齐新协议 + 调用计数
- [x] 5. `service.py`：主循环重排为「检测 → 去重 → 筛选 → 统一嵌入」
- [x] 6. `service.py`：帧预算调度，ROI 与预览降级为机会性阶段
- [x] 7. `service.py`：per-track ROI 冷却退避；已确认 track 跳过 ROI
- [x] 8. `service.py`：预览按订阅者 + 降采样 + 独立节流
- [x] 9. `video.py`：`grab`/`retrieve` 分离，跳帧不付解码成本
- [x] 10. `api.py` + `domain.py`：`effective_config` 直出当前生效判定条件
- [x] 11. `static/monitor.html`：运行条件面板
- [x] 12. 新增单测：等价性、嵌入次数、ROI 冷却、预算跳过
- [x] 13. `uv run pytest` + `uv run ruff check .` 全绿

## 复盘

### 改了什么

1. **`FaceBackend` 拆成 detect / embed 两段**（`backends.py`）。原先 `analyze()` 走 `app.get()`，
   检测和 ArcFace 绑死。现在 `detect_faces()` 只跑 SCRFD，`embed_faces()` 单独跑 ArcFace 且
   **一次批量推理**（`get_feat` 接受图片列表）。`analyze()` 保留为两者组合，enrollment 路径不变。
2. **主循环改为「检测 → 去重 → 质量筛选 → 统一嵌入一次」**（`service.py:_run`）。被
   `_merge_faces` 去重掉的、被质量门拒掉的脸，都不再产生 ArcFace 调用。
3. **帧预算调度**（`_roi_fits_budget`）。旧的 Hz 判据是间隔*下限*，单轮 ≥100ms 后恒为真，
   限流自我失效。新增基于 `face_roi` 历史 p95 的成本预估 + `min_processed_fps` 硬地板。
4. **per-track ROI 指数退避**（`_note_roi_outcome`），已确认 track 跳过 ROI，
   `roi_max_tracks_per_pass` 8→3，ROI 用固定 320 单尺度（全帧仍是 Auto 双尺度）。
5. **预览按订阅者门控 + 降采样到 960 宽 + 独立 5Hz 节流**。`PreviewHub` 新增显式
   `subscribe()`/`unsubscribe()`，由 MJPEG 端点包住自己的生命周期。
6. **`video.py` grab/retrieve 分离**：队列满时只 `grab()` 跳过解码，同时驱逐最旧帧，
   保持"最新帧优先"语义不变。
7. **可观测性**：新增 `source_fps` / `drop_rate` / `roi_calls_per_frame` /
   `end_to_end_p95_latency_ms` / `frame_width×height` / `budget_skips`，以及
   `effective_config`（直出 CPU/CUDA 分支解析后的真实速率与门槛，含确认窗口
   所要求的最低采样率）。`/monitor` 加了 4 个诊断格。

### 验证结果（真实模型，非 fake）

- **等价性**：同一张 6 人图，新 `analyze()` 与旧 `app._app.get()` 逐脸比对，
  `cos(new, old) = 1.000000`，`max|1-cos| = 5.96e-08`。bbox 完全一致。
- **成本实测**（CPU，1280×886，6 张脸）：
  | | 耗时 |
  |---|---|
  | `detect_faces` Auto(128+640) | 96 ms |
  | `detect_faces` 320 单尺度 | 26 ms（3.7× 更省） |
  | `analyze`（检测+6张嵌入） | 497 ms |
  | **旧 ROI 通道**（8× analyze） | **1487 ms** |
  | **新 ROI 通道**（3× detect-only@320） | **88 ms → 16.9× 更省** |
- **确认能力不退化**（最关键）：`person-search-eval` 同一素材、同一阈值，
  改动前后 stash 对比：

  | | main | 改动后 |
  |---|---|---|
  | face_observations | 240 | 240 |
  | accepted_faces | 240 | 240 |
  | evidence_collected | 40 | 40 |
  | **confirmed_events** | **1** | **1** |
  | elapsed | 27.28s | 25.02s |

  检测数、证据数、确认数**完全一致**，说明纯性能改动没有动到判定语义。
- `uv run pytest` 130 passed（原 120 + 新增 10）；`uv run ruff check .` 全绿。

### 教训

- 本次根因里最隐蔽的一条：**基于「距上次多久」的限流，在循环变慢后会自动失效**。
  它只能约束下限，不能约束占空比。已记入 `tasks/lessons.md`。
- 顺手删掉了 `_needs_roi_face_pass()`：ROI 选择加入冷却副作用后，一个「看起来是纯谓词」
  的包装函数会静默消耗退避计数。无人调用，直接移除而不是留着当陷阱。
- `cli.py` 的离线评估复用了 `SearchSession` 的 ROI 私有方法，改协议时必须同步，
  否则校准跑的是一条生产里不存在的流水线。

### 仍未做（留作下一轮）

- ROI 多裁剪**批量** ONNX 推理（需绕开 `app.get`，自己接 SCRFD 前后处理）。
  CPU 上批量对 ArcFace 只带来 1.1× 收益，但 T4 上应显著更高。
- 真机 RTSP 复测：需要在 192.168.17.60 上跑，确认 `processed_fps` ≥ 8 且
  `source_fps` 仍为 25。本地只有 CPU，无法代表 T4 的绝对帧率。

---

# 修复远距离零匹配：ROI 预算死锁 + 开启 48-63px 远脸档

计划：`~/.claude/plans/elegant-spinning-whale.md`

## 背景

实拍约 4 分钟的搜索，`远距离诊断` 面板报 1960 次人脸检测、**0 次**关联、0 次证据、
0 次质量通过。操作者认为该拍摄距离属于合理工况。

两个独立事实：

1. `face_size_counts = {48_63: 284, lt48: 1676}` —— 全部检测短边 <64px，而
   `effective_search_min_face_px` 在 `tiny_face_enabled=False` 下为 64。
   `_is_face_matchable` 全 False → 不跑 `embed_faces` → 关联入参为空。
   `associated 0` / `evidence_collected 0` / `quality_accepted 0` 是**同一个门控的三种显示**。
2. **真 bug**：`face_source_counts = {full_frame: 1960, roi: 1}`、
   `budget_skips = {face_roi: 1498}` —— ROI 全程只跑过 1 次。`_roi_fits_budget` 拿
   `1/target_loop_hz`(100ms) 的单帧余额，去卡一个**只在刚跑完全帧人脸检测那一帧**才被
   求值的阶段，而那一帧必做成本已是 `13+112=125ms`，余额结构性为负。第一帧能过只因
   `face_roi` 尚无 p95 样本（`estimated=0.0` 落进 falsy 分支），此后再无一次通过。

两者必须一起修：只修 ROI 不降门，64px 仍吃掉全部检测；只开远脸档不修 ROI，
tiny 策略要求 `det_score ≥ 0.65`，而全帧 640 下 18px 的脸达不到。

## 任务

- [x] 1. `config.py`：新增 `budget_credit_max_frames`；改写 `target_loop_hz` 注释
- [x] 2. `service.py`：`_roi_fits_budget(started)` 改为信用桶判据，跳过计数拆
      `face_roi_floor` / `face_roi_credit`
- [x] 3. `service.py`：新增 `_settle_budget_credit`，在迭代末尾双向截断补充/扣减
- [x] 4. `service.py`：`_effective_config` 增报 `tiny_face_shadow_mode`
- [x] 5. `.env.example` + `scripts/deploy_t4.sh`：开启 tiny 档、关闭影子模式
- [x] 6. `monitor.html`：新增 `REJECTION REASONS` 格；远脸格区分「正式 / 影子」
- [x] 7. `cli.py`：校准 ROI 注释，说明离线与线上仅差信用门
- [x] 8. `tests/test_service.py`：重写 4 个预算测试（含本 bug 的直接回归锁定）

## 复盘

### 改了什么

把可选阶段的准入判据从「单帧余额」换成「按 `target_loop_hz` 补充的信用桶」：
便宜帧攒额度、贵帧花额度，信用与欠账双向截断在 `refill * budget_credit_max_frames`。
`target_loop_hz` 由此重新变成真正起作用的参数（它现在就是补充速率，名副其实），
没有引入占空比之类的第二个旋钮。`min_processed_fps` 硬地板原样保留并优先于信用。

### 验证结果

- 用面板实测 p95（person 13ms / face_full 112ms / roi 82ms，face_full 隔帧触发）
  回放新旧判据：

  | | ROI 运行 | ROI 跳过 | 循环 | ROI 实际速率 |
  |---|---|---|---|---|
  | 旧 | **1** | 299 | 14.5fps | **0.0Hz** |
  | 新 | 226 | 74 | 10.0fps | 3.8Hz |

  旧判据复现了现网的 `roi 1` / `face_roi 1498` 形态；新判据把 ROI 拉到配置的 4Hz 附近，
  循环稳定在 10.0fps —— 恰好是 `target_loop_hz`，信用桶按定义收敛。
  10.0fps 对 `required_sampling_hz 2.0` 仍有 5× 余量。
- `uv run pytest` 131 passed；`uv run ruff check .` 全绿。
- API 实际启动，`/healthz` 200，`/monitor` 已含 `REJECTION REASONS` 格。
- 环境变量实测：`effective_search_min_face_px` 64→48，55px 脸落进 tiny 策略
  （thr 0.64 / 6 帧），`_is_shadow_policy` 为 False → 发正式 `confirmed`。

### 仍未做

- **真机同距离复测**（唯一能证明召回真的回来了的一步）：需在同一拍摄距离复现，
  验收 `roi` 持续增长、`48_63` 占比上升、`associated > 0`。
- **阈值标定**：`tiny_face_shadow_mode=false` 意味着一次远脸确认直接置 `found`。
  在让机器人基于确认行动前，必须用 `person-search-eval` 标定
  `tiny_face_similarity_threshold` / `tiny_face_aggregate_similarity_threshold`，
  不要留用默认的 0.64 / 0.68。
- `.gitignore` 未忽略 `.env`。本次把 `.env` 记为开启远脸档的途径，而 RTSP URI 常带凭据，
  与代码里到处在做的 `_sanitize_source` 相抵。建议补一行，但不在本次改动范围内。

## 远脸 `6/6帧` 恒满却永不确认：修复诊断盲区

### 背景

现场：目标登记 quality 0.96，画面人脸 ~50px，UI 徽标恒显 `6/6帧`，从不确认，
`BEST SIMILARITY` 永远是 `—`。用户合理推断"6 帧太多"。

**根因不是帧数。** 诊断面板 `evidence_eligible 428` 对 `above_threshold 5` ——
只有约 1% 的样本越过 tiny 档的 0.64，而 `_is_confirmed` 要求**中位数** ≥ 阈值，
确认在数学上不可能。而 `6/6` 是假信号：tiny 策略 `collect_all_observations=True`
使 `accepts_observation` 恒真，**不论相似度高低所有样本都入队**，deque 被裁到 6，
几秒填满并永久停在那里。真实相似度服务端一直在记
（`_target_status["best_observed_similarity"]`），只是 `renderTargets` 从没渲染。

用户在盲飞：唯一可见的进度条恒满，唯一能解释失败的数字没被显示。

### 已完成（阶段一：只改可观测性，判定逻辑零变更）

- [x] `confirmation.py`：`track_progress()` 由 `dict[int, tuple[int, int]]` 改为返回
      `TrackProgress`（`observed` / `required` / `qualifying` / `threshold` /
      `median_similarity` / `best_similarity`）。`observed` 的 docstring 明写它在
      `collect_all_observations` 下会饱和、不代表进度。
- [x] `confirmation.py`：抽出 `_policy_threshold` / `_policy_required` / `_policy_window`，
      消除 `_is_confirmed` / `_expire_evidence` / `track_progress` 三处重复的判空分支，
      保证进度上报与确认判定同源。
- [x] `confirmation.py`：给 `min_top1_margin` 加注释说明它由调用方按帧过滤、
      不参与窗口判定（此前读者会以为 `_is_confirmed` 漏了它）。
- [x] `service.py`：progress 汇总改为按 **qualifying 票数**（并列时比中位数）挑 track，
      而非按采集数——否则会持续上报一个满仓但全是低分的 track。
- [x] `domain.py` / `service.py`：`TargetSearchView` 增 `qualifying_evidence` /
      `median_similarity` / `required_similarity`。`required_similarity` 是关键，
      用户必须看到"需 0.64"才知道差多远。
- [x] `monitor.html`：徽标改为 `证据 6/6` + 次行 `最佳 0.41 / 需 0.64 · 中位 0.41 ·
      达标 0/6 · 相似度不足`，并加 `REJECTION_TEXT` 中文映射。仍走 `textContent`，
      目标名不进 innerHTML。
- [x] `monitor.html`：`BEST SIMILARITY` 回退读 `best_observed_similarity`，
      使其在抑制候选事件的远脸档也有值。`metrics.best_similarity` 语义
      （已确认决策的最佳值）保持不被污染。
- [x] 回归测试：`test_saturated_tiny_evidence_reports_qualifying_shortfall`（确认器层）
      与 `test_saturated_tiny_evidence_surfaces_similarity_shortfall`（服务层，
      用 0.41 复现现场数值）。此前**没有任何测试覆盖这个故障形态**。
- [x] `uv run pytest` 134 passed；`uv run ruff check .` 全绿。

### 仍未做（阶段二：据实测校准，不拍脑袋改阈值）

- [ ] **真机复测**（唯一能证明修复有效的一步）：接现场 RTSP 重跑同一场景，
      确认徽标显示实际相似度、不再出现恒满的 `6/6`。
- [ ] **用分布定阈值**：`person-search-eval` 取两个直方图——同一人 50-63px 的相似度，
      与**不同人**同尺寸的相似度。tiny 阈值应落在两者的分离点，而不是落在"能确认"
      的位置。若两分布重叠严重，结论就是 50px 在本场景不可用；那是必须报告的真实
      结论，不能靠降阈值掩盖。
- [ ] **时间窗**：`tiny_face_evidence_window_seconds=3.0` 要求 6 个样本跨 3 秒，
      移动机器人在 3 秒内姿态差异很大，低分姿态会把中位数拉下来。可考察缩短到
      1.5-2.0s，或把中位数改为上四分位 / 最佳-K 均值（更贴合移动采样，但会提高
      误报，必须用上面的负样本分布验证）。按实测决定，不预先承诺。
