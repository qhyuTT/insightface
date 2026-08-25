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
