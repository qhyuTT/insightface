# InsightFace 项目工作约定

## 开始工作

- 使用中文沟通；代码、API、日志字段保留原文。
- 先读 [项目记忆](docs/project-memory.md)，涉及流畅度、硬件或部署时再读
  [当前技术方案](docs/technical-solution-20261015.md) 以及 `tasks/lessons.md` 的相关条目。
- 多步骤工作先维护 `tasks/todo.md`，完成后补验证证据；用户纠正形成的通用规则记入
  `tasks/lessons.md`。新现场结论更新项目记忆，保留日期、条件和未决事项。
- 保留无关工作区改动。不要把不同日期、不同任务、不同相机或单/双路测试拼成同一基准。

## 项目边界与代码地图

下列 Python 模块均位于 `src/person_search/`，除非另有路径说明。

- 本仓库是 Python 寻人识别服务，机器人 Android 推流在相邻 `swcaresRobot`，调度与代理
  在独立调度项目。修复前先确定责任边界；相邻仓库不会随本仓库发布自动生效。
- `api.py`：REST、WebSocket、MJPEG 与证据接口。
- `service.py`：会话生命周期、分阶段推理、预览与多目标/多源协调。
- `video.py`：最新帧、有界 AI 队列、重连和采集指标。
- `gstreamer_capture.py` + `scripts/rk3588_gstreamer_capture.py`：系统 Python 硬解桥接。
- `hybrid_backends.py`、`rpp_backends.py`：SCRFD RPP 路径及原生内存生命周期。
- `rknn_backends.py`、`rknn_runtime.py`：YOLOX/ArcFace RKNN 路径与 ABI。
- `tracker.py`、`face_tracking.py`、`confirmation.py`、`quality.py`：跟踪、关联及身份确认。
- `config.py`、`domain.py`：配置、契约及指标；`static/monitor.html`：监控页。

## 当前现场基线

- 用户通常只开一路海康；优先验证这一路。机器人自带 720p 为可选来源，不能用双路合计
  FPS 宣称单路流畅。项目支持双源，但不等于最新单源结果已证明双源性能。
- 2026-10-15 实测海康主码流为 H.265、1920×1080、15 FPS、约 2 Mbps；不是当时在跑 4K。
  `101` 是主码流编号，不代表分辨率；`102` 是子码流。再次排查必须重新核对实际参数。
- 当前硬件分工：MPP 解码，RPP BF16 做 SCRFD 人脸检测，RKNN NPU 做 YOLOX 行人检测和
  ArcFace 特征。不得称“NPU 解码”，不得由 TOPS 相加推断实际速度。
- 2026-10-15 已部署 `rpp-native-input-20261015-v1`，具体镜像与回滚见项目记忆；版本、
  IP 和服务状态均是时间快照，实施操作前重新核实。

## 性能与正确性

- 保持采集、AI 和预览独立；AI 队列满不能阻止更新预览 latest。预览只取新帧、不补历史帧，
  超预算不额外睡眠，无订阅者不编码，停止/断流须唤醒等待者。
- NV12 优先使用逐帧 `GstVideoMeta` 的 stride/offset；caps 不足以证明紧凑布局。
  必须检查两平面边界和重叠，不能简单截取前 `width*height*3/2` 字节。
- `pyrt.DeviceAllocation.copy_from_numpy` 已实测持 GIL 阻塞；仅改异步 enqueue 不足以修复。
  原生输入路径使用复用 pinned buffer、CDLL 拷贝及 stream 同步，当前由
  `PERSON_SEARCH_RPP_RELEASE_GIL=true` 显式启用；代码和通用 Compose 默认关闭。
- RPP 操作保留全局执行锁及 owner 引用。原生 copy/enqueue/sync 失败后隔离整个 engine，
  不复用或释放生命周期未知的 DMA 缓冲。同步成功后擦除输入 staging；正常关闭先 context、
  再 stream、再 staging/设备/engine/runtime。不要用移除锁换 FPS。
- 保持 RPP plan cache 默认关闭；此 SDK 曾在 BF16 plan 反序列化时原生崩溃。
- 性能修改不得顺带降低识别门槛、关闭 flip TTA、变更模型/输入尺度或把 shadow 当确认。
  `<48 px` 人脸不能用于身份确认；以原始视频帧的人脸框尺寸判断，不用网页缩放尺寸。
- RGA 是否实际采用必须看运行转换状态与颜色校验；设备存在不代表已启用，不删除校验。

## 测量与验证

- 分别测 ZLM 压缩帧、MPP 解码帧、主服务预览发布、HTTPS 接收及浏览器呈现。
  JPEG 发布 FPS 不等于浏览器 FPS，网络接收测试不等于光学端到端延迟测试。
- 双源优先看 `sources[]` 逐源指标；聚合值、滑窗速率、累计计数及分位数样本口径要分清。
- 用户已确认 `.184:18443` 仍卡，不能继续把 `.17.215` 经 TUN 作为已证实根因。
  额外预览/测速客户端会改变负载，记录测试条件，避免并发下载制造拥塞。
- 开发使用 Python 3.11 与 `uv.lock`；常规验证：`uv run pytest`、`uv run ruff check .`、
  `git diff --check`。现有依赖不全时按 `pyproject.toml` 的实际 extras 安装。
- 新硬件路径需做合成输入的完整输出一致性、A/B/A 延迟及线程心跳测试，再测真实会话。
  单测通过、provider ready、单图基准都不能替代真实视频性能或业务识别精度验收。

## 部署、安全与回滚

- 只有一个 Uvicorn worker；会话和目标特征在进程内，不能直接增加 worker 数扩容。
- 使用实际已部署 Compose 做基线；核对 host network/IPC、cpuset、安全设置、设备与挂载。
  不用仓库模板直接覆盖现场配置。MPP 依赖设备树和 dma_heap，RPP 依赖 host IPC。
- 在空闲或用户已授权中断时切换。额外 RPP 诊断进程曾无法分配上下文而主服务正常，
  遇到这种情况在空闲独占环境验证；不要擅自重启驱动、卸载模块或重启盒子。
- overlay 同步核对 `service.py`、Settings 和 backend 接口。必须在实际候选镜像验证活动
  SearchSession 的 view/JSON 和路由；曾因缺配置字段导致“ready 正常、任务启动后 500”。
- 发布后核对镜像/源码哈希、healthy/restart、三路 provider、能力集合、真实 active 和错误日志，
  留可执行回滚。不要回滚到缺 NV12 布局修复的旧镜像。
- 不提交或输出相机密码、API key、Basic Auth 密码、签名材料或完整 runtime.env。
  保持 TLS/SSH 主机校验，不用宽松信任修连通性；保护文件不得靠放宽权限读取。
- 真人照片、视频、embedding 和证据只按业务内存/授权接口生命周期处理；排查不落盘真实媒体，
  不加入测试素材。合成张量可用于回归；日志与文档只保留必要的数值和脱敏来源。
- `/mediaserver/` 的 Nginx Basic Auth 与 ZLM API secret 是不同认证；凭据不写进记忆文档。
