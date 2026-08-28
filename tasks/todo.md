# 提升 SearchSession 处理帧率

## 命中证据内存交接

- [x] `service.py`：正式 `confirmed` 时编码原始帧和人脸裁剪，仅短时保存在进程内存，并在停止/终态立即清除。
- [x] `domain.py`：`confirmed` / `target_found` payload 支持不透明 `evidence_id`，不嵌入图片。
- [x] `api.py`：新增 `GET /v1/searches/{search_id}/evidence/{evidence_id}`，要求 `PERSON_SEARCH_EVIDENCE_API_KEY` 的 `X-API-Key`；支持 `face_crop`（默认）和 `frame`。
- [x] 覆盖确认、读取、清除和 API 鉴权测试；图片不落日志、磁盘、fixture 或仓库。

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

## 修复策略单向棘轮：由远走近的轨迹终身按远脸档苛判

### 背景

现场症状是"证据够了，但相似度总差点意思"。上一项（`6ceb350`）已经证明其中一半是
**显示问题**：远脸档 `collect_all_observations=True` 让计数器恒显 `6/6`。这一项处理
另一半，是**真正的判定 bug**。

`confirmation.py` 的策略选择只有"收紧"一个方向：`_is_stricter_policy` 返回 False 时，
`else` 分支把传入的宽策略丢掉、继续用锁定的严策略。`state.policy` 只在轨迹被删除时清空。
后果是：旅客在远处被首次看到（55px → 48-63 档）后，即使走到 3 米、人脸已 120px，
整条轨迹仍按 `0.64 / 6 帧 / 聚合 0.68 / 仅 person_strict / 不发 candidate` 判定，
而该尺寸本应只需 `0.55 / 3 帧 / 1.5s`。监控视频下 ArcFace 相似度中位数常落在
0.55-0.62——正好过得了 0.55、过不了 0.64，于是表现为"帧数攒满、就是不确认"。

第二个问题：远脸档的**第二道门**（质量加权聚合 embedding `>=0.68`）此前是
`_is_confirmed` 里的局部变量，算完即弃。面板可以显示"中位 0.66 ✓"却永远不确认，
且没有任何字段解释原因。

### 已做

- [x] `confirmation.py`：在 `_is_stricter_policy` 分支后补 `elif policy != state.policy`
      降档分支，按当前帧尺寸重新解析策略；升档与降档**都**清空证据窗口，
      避免一个窗口混入不同阈值下采集的样本。已确认轨迹（`state.confirmed`）不清证据，
      不会被打回重新取证。`state.shadow_confirmed` 那条过渡分支未动。
- [x] `confirmation.py`：把聚合相似度抽成 `_aggregate_similarity(state, target)`，
      `_is_confirmed` 与 `track_progress` 共用，进度与判定不会各自复算而漂移。
- [x] `confirmation.py`：`TrackProgress` 增 `aggregate_similarity` / `aggregate_threshold`；
      `track_progress(target=None)` 保持向后兼容，不传 target 时聚合值为 None。
- [x] `domain.py` / `service.py`：`TargetSearchView` 增 `aggregate_similarity` /
      `required_aggregate_similarity`，`_target_status` 同步加默认值。
- [x] `monitor.html`：次行加 `聚合 0.66 / 需 0.68 ✗`，仍走 `textContent`。
- [x] 回归测试三条：`test_track_confirms_on_the_normal_tier_after_its_face_grows_past_the_far_bar`、
      `test_relaxing_the_tier_clears_the_evidence_window`、
      `test_track_progress_reports_the_far_face_aggregate_gate`。
      **已验证前两条在移除降档分支后确实失败**（`assert 0 == 1` 无确认事件；
      `observed=3, required=6, threshold=0.64` 即 120px 脸仍被锁在远脸档），
      不是恒真断言。
- [x] `uv run pytest` 137 passed；`uv run ruff check .` 全绿。

### 仍未做

- [ ] **真机复测**：起流后走一遍"远处进入画面 → 走近到 2-3 米"，确认同一条轨迹
      在脸变大后按 `0.55 / 3 帧` 确认，而不是继续卡在 `0.64 / 6 帧`；
      并读新暴露的聚合值，判断"差点意思"里有多少是显示问题、多少是真实短板。
- [ ] **误报回归**：本改动是**放宽**确认条件，方向上会增加误确认。而 `0.55` 按
      README 自己的说法"只用于跑通流程"，从未标定过。必须用 `person-search-eval`
      在同一素材上对比：`face_observations` 必须完全不变（未碰检测），
      `confirmed_events` 只允许增加且每条新增都要人工确认是真阳性。
- [ ] **`video.py:41-43` 的 join 竞态**（本次验证时偶发命中，与本改动无关）：
      `self._thread = Thread(...)` 与 `self._thread.start()` 之间有窗口，
      此时 `stop()` 从另一线程进来会 `RuntimeError: cannot join thread before it is
      started`。全套测试里约 1/8 概率让
      `test_effective_config_reports_the_resolved_rates_and_gates` 失败
      （该测试用真实 `camera/device_index=0`）。改法是先赋局部变量、`start()` 之后
      再发布 `self._thread`，或两者置于同一把锁下。

## 移动机器人场景：远脸召回 + 确认速度

计划：`~/.claude/plans/plan-cozy-flute.md`

### 背景

现场读数 `1 · 寻找中 · 49px · 证据 0/6 · 最佳 0.70 / 需 0.64 · 达标 0/6 · 人脸过小 / 检测置信度不足`。
`最佳 0.70 ≥ 需 0.64` 说明**识别能力不是瓶颈**——有脸走完了嵌入并越过了远脸档的阈值。
而 tiny 档 `collect_all_observations=True` 会把通过质量门 + 关联的观测**全部**入队，
`证据 0/6` 因此意味着几乎没有观测走到确认器。故障在漏斗上游。

机器人自己在动、人可能走可能坐；要求「准一点、快一点」；算力 T4；登记照只有一张；
确认条件分两档用开关切。

### 任务

- [x] 0. 提交基线（单向棘轮修复）；修 `video.py` join 竞态；`.gitignore` 加 `.env`
- [x] 1. `config.py`：多尺度 / 深扫 / 迟滞 / 运动补偿 / 翻转 TTA / `match_profile` 全部可调项
      + `full_frame_detection_scales()` / `roi_detection_scale()`
- [x] 2. `backends.py`：`detect_faces` 接受一组尺度（SCRFD 自己跨尺度 NMS）；
      `embed_faces` 同批次翻转 TTA，登记侧走同一实现
- [x] 3. `confirmation.py`：档位命名化 + `resolve_face_tier` 迟滞；`_window_statistic`
      （median / top_k_mean）由 `_is_confirmed` 与 `track_progress` 共用；
      `requires_strict_association` 拆出 `allows_relaxed_association`
- [x] 4. `tracker.py`：`update(motion=...)` 在 IoU 前抵消全局位移；`velocity` 改为
      观测间测量并按 `missed` 归一
- [x] 5. `service.py`：`_estimate_camera_motion`（phaseCorrelate + Hanning 窗）、
      深扫节奏、ROI 尺度自适应、`_track_tiers` 会话级档位、relaxed 关联按
      `is_stricter_policy` 决定是否降档、拒绝原因与其人脸尺寸同源
- [x] 6. `domain.py` / `monitor.html`：`window_similarity` + `window_statistic` + `tier`
      + `last_rejection_face_px`；新增 `MOTION / SHARPNESS`、`MATCH PROFILE` 两格
- [x] 7. `cli.py`：离线链路同步全部改动（多尺度、深扫、运动补偿、档位、关联语义）
- [x] 8. `evaluation.py` + `cli.py`：`--dump-similarities` 与 `summarize_similarity_samples`
- [x] 9. 回归测试 22 条；`uv run pytest` 159 passed；`uv run ruff check .` 全绿
- [x] 10. `README.md` / `CLAUDE.md` / `.env.example` / `deploy_t4.sh` 同步

### 复盘

#### 改了什么（按根因）

| 根因 | 处置 |
|---|---|
| 640 档把 49px 脸压到 ~16px（stride-8 下限），`det_score` 结构性达不到 0.65 | CUDA 叠加 1280 尺度，`face_deep_scan_every_n` 可降为隔轮深扫 |
| ROI 定值 320 对大裁剪是**降采样** | 按裁剪自适应，320 为下限、640 为上限 |
| 档位抖动每帧清空证据窗口 | `face_tier_hysteresis_px=6`，档位由 session 按轨迹解析一次 |
| 相机运动 → IoU 关联失败 → 新 track id → 证据归零 | phaseCorrelate 估全局平移，关联前施加；顺手修 `velocity` 语义 |
| 远脸档拒绝 `person_relaxed`，坐姿在结构上不可确认 | 拆出 `allows_relaxed_association`，默认允许；仍禁止 face-only |
| 中位数对移动采样不友好、6 帧/3s 太慢 | `match_profile=responsive` → 4 帧/2s/top-K 均值/检测分 0.55，聚合门不动 |
| 只有一张登记照，域差无解 | 翻转 TTA（同批次，登记与搜索同源），抬分布而非降门槛 |
| 无法用分布定阈值 | `--dump-similarities` + 同人/异人分布摘要 |

#### 两条被实测否掉的假设（写进计划、差点改代码）

1. **清晰度阈值随尺寸缩放**：用 skimage 真实人脸裁剪实测，Laplacian 方差随人脸变小
   **上升**（40px 3689 / 200px 821），固定 45 对小脸其实更宽松；归一化到 112 反而
   把小脸压到门槛附近，是收紧。**放弃该改动**，只保留可观测性（blur p50/p95 上面板）。
2. **`assess_face` 整数截断把 49px 量成 47px**：`int(x2)-int(x1)` 恒等于 `int(x2-x1)`
   或再大 1，永远不会更小。真正的原因是**面板把两个不同观测拼成一行**：
   `last_face_px`（见过的最大脸，49px、已接受、相似度 0.70）和 `last_rejection_reason`
   （另一张 40px 脸的原因）。改为原因与其尺寸同源上报。

#### 验证结果

- `uv run pytest` 159 passed（原 137 + 新增 22）；连跑 5 次无 flaky；`ruff check .` 全绿。
- 关键回归**已验证非恒真**：档位迟滞用 `face_tier_hysteresis_px=0` 对照，
  抖动侧 `observed` 恒为 1 且从不确认，迟滞侧攒满 4 帧并确认；相机运动用同一段
  90px/帧位移对照，补偿侧单一 track id，未补偿侧第二帧即换 id。
- `_estimate_camera_motion` 对 `np.roll` 60px 的合成位移回报 60±2px（全帧像素，
  估计跑在降采样图上）。
- 翻转 TTA 用桩识别器验证：**一次**推理拿到裁剪与镜像两行，特征相加归一化。
- API 实启动，`/healthz` 200，`/monitor` 含两个新诊断格；monitor.html 脚本经 node 解析通过。

### 已部署到 60 服务器（ky-sv / 192.168.17.60）

按仓库既有做法走 bundle：`~/deploy/insightface-release-<sha>` 逐版目录，从上一版
clone 后 `git fetch <bundle>` + `merge --ff-only`，`models/yolox_tiny.onnx` 从上一版
复制（SHA-256 校验通过）。GitHub 上的 `origin/main` 仍停在 `6ceb350`，未推送。

两次部署：`0bd9e98`（计划原样）→ `ecf02f8`（按 T4 实测修正检测配置）。当前容器
`person-search:t4` @ `ecf02f8`，`healthy`，`PERSON_SEARCH_HOST=0.0.0.0`，
局域网 `http://192.168.17.60:8000/monitor` 可访问。旁边的
`airport-robot-dispatch-{frontend,backend}` 未受影响，无 dangling 镜像残留。

五层 GPU 核验全部通过：宿主 `Tesla T4` → 容器 `nvidia-smi` → ORT
`CUDAExecutionProvider` → YOLOX `CUDAExecutionProvider` → InsightFace
`CUDAExecutionProvider`。真实 SCRFD 多尺度调用返回正常（这是本地无法验证的那一段）。

#### T4 实测：检测尺度（这条改变了计划的结论）

| 配置 | p50 | 21px 脸 | 52px 脸 | 311px 脸 |
|---|---|---|---|---|
| `[128,640]`（原生产） | 108 ms | **漏检** | 0.75 | 0.90 |
| `[128,640,1280]`（计划原案） | 181 ms | 0.69 | 0.82 | 0.91 |
| `[1280]`（**实际部署**） | **57 ms** | 0.69 | 0.82 | 0.91 |

单档 1280 比原生产**便宜一半**且每个尺寸都不更差。根因是 CUDA 上每换一次 ONNX
输入形状约付 30 ms 重规划（单档实测 320=5ms / 640=18ms / 1280=53ms），「叠加一档」
的钱几乎全花在形状切换上。同一原因下把 ROI 尺度从「逐裁剪取整」改成两档量化
（233 ms/轮 → 136 ms/轮）——那是我当天刚写的代码，靠实测才发现。

部署后 `face_full` p50 62 ms / p95 72 ms，低于 `face_detection_hz_cuda=10` 的 100 ms
周期，远脸档要求的 `2.0 Hz` 有充足余量，**因此不需要动
`PERSON_SEARCH_FACE_DEEP_SCAN_EVERY_N`**。

另一处修正：`[128,640]` 对**检测框短边 <45px** 的脸是完全漏检，对 52px 则是
检得到但分低。现场 1960 次检测说明用户那个距离属于后者，所以 1280 带来的是
`0.75 → 0.82` 的分数抬升与更稳的关键点，不是「从无到有」——计划里把两者混为
一谈，高估了这一刀的收益。

### 回归修复：`FACE_DETECTION_SIZE=1280` 打死了人脸登记

用户反馈「人脸登记无法使用，检测不到人脸」。是上一条部署引入的回归，已定位、实测
复现并修好。

根因是那个变量有**两个消费者，只测了一个**。搜索路径显式传尺度（`[1280]`，测过）；
登记路径 `analyze(enrollment=True)` 传 `detection_size=None`，SCRFD 回落到
`prepare()` 设的 `input_sizes`，于是登记也跑在从未测过的单档 1280 上。而 SCRFD 会把
图**放大**到输入尺寸且无上限，脸在网络输入上超过约 500px 就越过 stride-32 锚点上限，
**整脸漏检**。登记照按定义是近距离大脸，所以必然失败，不是偶发。

用真人脸贴入画布实测（`buffalo_l`，CPU）：

| 画布 / 占高 | 脸 px | `auto[128,640]` | `[640]` | `[1280]` |
|---|---|---|---|---|
| 960×1280 / 0.75 | 960 | 0.905 | 0.721 | **MISS** |
| 960×1280 / 0.60 | 768 | 0.873 | 0.844 | **MISS** |
| 960×1280 / 0.45 | 576 | 0.865 | 0.865 | **MISS** |
| 1920×1080 / 0.90 | 972 | 0.885 | 0.885 | **MISS** |

端到端复现与修复后对比（同一张人像，`face_detection_size=1280`）：

```
修复前: EnrollmentError: no face detected  code=no_face
修复后: OK 482x661px det=0.867 q=0.927   （搜索尺度仍为 (1280,)）
```

修法：登记不再继承搜索尺度。新增 `enrollment_detection_size`（0 = Auto 128+640），
`detect_faces` 在 `enrollment=True` 且未显式传尺度时自行解析，不再依赖 `prepare()`
回落；显式传参仍然优先，ROI 通道不受影响。1280 对搜索的实测收益全部保留。

- [x] `config.py`：`enrollment_detection_size` + `enrollment_detection_scales()`
- [x] `backends.py`：`detect_faces` 登记路径显式传尺度
- [x] 回归测试按「生产配置下断言行为」写，不再只钉默认值（163 passed）
- [x] `CLAUDE.md` 登记/搜索差异表加尺度行；`.env.example`；`lessons.md` 三条

**已知边界，未动**：搜索侧单档 1280 对占 1080p 画面高度 ≥65% 的脸同样漏检。机器人
工作距离的脸是 100–300px，不在该区间，近距离行人另有 ROI（320/640）兜底。按本文件
第 44 条，加尺度前必须先量形状切换成本，所以这里只记录不顺手改。

- [ ] **真机复测登记**：在 .60 用真实登记照走一遍 `POST /v1/targets`，确认返回
      `face_width/face_height` 与 `detection_score`，而不是 `no_face`。

### 仍未做（需要真机 RTSP + 画面里有人）
- [ ] **同距离复测**：`face_size_counts` 中 `48_63` 占比、
      `rejection_counts.detection_score_low` 下降幅度、`associated > 0`、
      `证据` 是否单调上升、`相机位移 p95`。
- [ ] **走近 + 坐下两段**：确认档位徽标随尺寸变化但不抖动，
      `association_counts.person_relaxed` 在坐姿段出现。
- [ ] **用分布定阈值**（lessons 第四次要求）：`--dump-similarities` 跑正负两份素材，
      按同人/异人分离点设 `tiny_face_similarity_threshold` /
      `tiny_face_aggregate_similarity_threshold`。重叠严重就如实报告该距离不可用。
- [ ] **误报回归**：本轮是放宽方向，且翻转 TTA 改变了嵌入数值，
      `cos(new, old) = 1.0` 那条等价性判据**不再适用**。改为：正样本素材
      `confirmed_events` 只允许增加且逐条人工确认；负样本素材两档均必须为 0。
- [ ] **首次确认耗时**：用户说的「快」目前没有任何指标表达，需要手工记录中位耗时。
- [ ] ROI 多裁剪批量 ONNX 推理（上一轮遗留，T4 上收益要先看实测）。

## 命中证据交接：裁错框与失败语义（2026-08-28）

接手 codex 未完成的工作（会话中途 402 中断，改动未自查未提交）。codex 的主体改动是
完整的，复查时找到两个真实缺陷并补了回归测试。

### codex 已完成、我复核通过

- [x] `MatchDecision.face_bbox` 携带人脸检测框，裁剪不再用人体轨迹框
- [x] TTL 120s → 600s；自动完成不再清空证据（显式停止/超时/失败仍立即清空）
- [x] `DELETE .../evidence/{id}` 释放接口，对刚释放的 ID 幂等
- [x] `SearchView.confirmed_results` 作为事件漏收时的对账源
- [x] `privacy.py`：uvicorn 访问日志脱敏证据路径中的两个 ID
- [x] `/healthz` 增加 `confirmed_evidence_v1` 能力位

### 我修的两个缺陷（均已验证测试非恒真）

- [x] 终态补发的 `target_found` 沿用命中时的 payload，而字节已被 `_transition`
      清空，事件里 `evidence_available` 仍是 true → 调度端只能拉到 404 并白等重试
      窗口。改为发布前按当下可用性重算（`_refresh_evidence_availability`）。
- [x] 「随搜索销毁」与「你已释放」共用 `evidence_released` 404，而 404 在调度端
      可重试 → 不可恢复的情况被重试到 600 秒耗尽。拆出 `evidence_discarded`（410），
      调度端映射为永久失败，一次判定。

### 验证

- insightface：pytest **177 passed**（连跑 3 次无 flaky）、ruff 通过
- dispatch 后端：pytest **79 passed**、ruff 通过；前端 vitest **39 passed**、
  typecheck / lint / build 通过
- 两个缺陷分别用「移除修复」对照验证过会失败，不是恒真断言

### 部署（192.168.17.60）

- [x] insightface `0dffc78` 推送并部署，容器 healthy，`CUDAExecutionProvider` 已验证
- [x] `/healthz` 返回新能力位 `confirmed_evidence_v1`；`openapi.json` 含 `delete`
- [x] 无密钥 DELETE → 403（接口存在且强制鉴权）
- [x] dispatch `c210d48` 推送并部署，alembic 升到 `0007`

**部署时踩到的坑**：本机 `.env.production` 缺 `DISPATCH_EVIDENCE_ENCRYPTION_KEY` 与
`DISPATCH_INSIGHTFACE_EVIDENCE_API_KEY`（上一轮直接在服务器上配的，没回写本机），
而 `deploy.sh` 会用本机文件**整体覆盖**服务器的 `shared/.env`。若直接部署，compose
的 `:?` 会当场失败，且服务器上唯一的密钥副本被覆盖丢失。已先从服务器读回这两个值
补进本机文件，再逐键 diff 确认无遗漏才部署。

### 仍未做（需要真机 + 授权照片）

- [ ] **端到端验收命中裁剪**：真实寻人任务命中后，确认管理台从「同步中」翻到
      「已保存」，且弹窗能同时显示登记照与**人脸特写**（不是整个人的大图）——
      这正是本轮修的那个 bug 的现场判据。
- [ ] **`evidence_discarded` 现场验证**：命中后立刻手工结束任务，确认调度端一次
      判定为失败并显示「已随寻人会话销毁」，而不是转圈到 600 秒。
- [ ] 上一轮遗留的远脸召回真机复测项（见上文各节）仍然全部未做。

## 候机厅场景特调：先仪表，后调优（2026-08-28）

现场反馈：快速走过的旅客完全不确认；低头玩手机的坐姿检不到脸；相似度上不去；远脸认不出。
相机随机器人持续移动，取舍是**宁可误报也不能漏**。

先把问题定住：「凑够 N 帧」是硬条件且分三档（80px 以上 0.55/3 帧/1.5s；64-79px 0.60/4 帧/2s；
48-63px 0.64/6 帧/3s + 5 票 + 聚合 0.68），每档隐含 **2.0 Hz 最低采样率**。但候机厅一帧 6-10
张脸时嵌入成本（lessons 20：6 张脸 400ms，再叠 flip TTA 翻倍）很可能把循环压到 1-2 Hz，正卡在
需求线下面——此时调阈值毫无意义。而「帧不够」和「分不够」当时**没有任何指标能区分**。

### Phase 1：仪表（不改变任何判定）

- [x] 门集中到 `_evaluate`，返回第一个失败的门；`_is_confirmed` = `gate is None`
- [x] `TrackOutcome` 轨迹验尸：确认时发（带首次确认耗时），未确认轨迹删除时发（带卡点门）
- [x] 保留「离确认最近的一次尝试」快照——删除时窗口必然已空，读实时值会恒报「帧数不足」
- [x] `SearchMetrics`：`time_to_confirm_seconds` / `track_dwell_seconds` /
      `track_sampling_hz` / `unconfirmed_gate_counts`，snapshot 出 p50/p95
- [x] `monitor.html` 两个新格子（TIME TO CONFIRM / CONFIRMATION BLOCKERS），
      SAMPLING HEADROOM 改用实测 `achieved_sampling_hz`，无实测时标注「估:循环率」
- [x] `person-search-eval` 每个阈值输出 `track_outcomes`

### Phase 2：机制到位，默认全关

- [x] `MATCH_PROFILE_OVERRIDES` 注册表 + `transit` 场景档（缩窗口/缩采样间隔/降票数，
      **不动任何相似度阈值**）；`evidence_min_interval_seconds` 变成可配
- [x] 离场结算 `departure_adjudication_enabled`（默认 false）：只救 `insufficient_samples`，
      要求 `窗口统计量 >= 阈值 + 0.05`，走 shadow 通道成对发 confirmed/lost
- [x] `deploy_t4.sh` 暴露 `T4_MATCH_PROFILE=transit` 与 `T4_DEPARTURE_ADJUDICATION`

### 验证

- `uv run pytest` **195 passed**（原 177 + 18），`uv run ruff check .` 通过
- 等价性：新增 2000 次随机扫描，逐例对比 `_evaluate(...).gate is None` 与**改动前原样抄下来的**
  `_is_confirmed`，并断言扫描里既有确认也有不确认（否则恒真）
- 非恒真反向验证：把验尸改成读实时窗口 → 6 个测试失败；把聚合门与票数门顺序对调 → 2 个测试
  失败，而等价性扫描仍然通过（顺序只影响上报、不影响判定，正是想要的分工）

### 仍未做（需要真机 / 素材）

- [ ] **现场读数**：跑一次真实寻人，读 `实际采样率 vs 需求采样率` 与 `未确认卡点` 分桶。
      实际 < 需求 → 减负载（关 flip TTA、降 face_detection_hz、缩 ROI 路数）；
      实际 ≥ 需求但卡在统计量/聚合 → 才轮到标定阈值。**这一步之前不要调任何阈值。**
- [ ] **坐姿低头**：先看 `budget_skips` 的 `face_roi_floor`/`face_roi_credit` 是否又把 ROI
      补检饿死（lessons 18/24 已犯过两次）。静止的人时间管够，恰恰是 ROI 最该救的对象；
      `roi_max_tracks_per_pass=3` 对候机厅偏紧，lessons 10 的基线是 8。
      但要诚实：机器人站立高度看低头的人，SCRFD 可能根本给不出框，这是几何问题不是阈值问题。
- [ ] **用分布定阈值**（lessons 第五次要求）：`--dump-similarities` 跑正负两份素材再定
      `tiny_face_similarity_threshold` / `tiny_face_aggregate_similarity_threshold`。
- [ ] **离场结算下游**：shadow 事件是 `tiny_shadow_confirmed`，调度端目前只消费 `target_found`。
      要真正用上这条线索通道，dispatch 侧需要订阅并以「线索」而非「命中」呈现。
