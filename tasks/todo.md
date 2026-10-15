# `.215` 双路优化实施与验收（2026-10-14 续，计划 v2）

## 已生效（本轮逐项核对）
- [x] 生产切到 `perfpreview-20261014-v1`；`video/domain/service/rga/monitor` 五文件与工作树 SHA256 一致；容器 healthy、restart 0、`readyz` 就绪；provider 保持 RPP_BF16 / RKNN_NPU / RKNN_NPU，embedding contract 未变；回滚镜像 `streamcontract-20261014` 保留
- [x] 断流退避生效：40 分钟仅 7 次 `capture frame timed out`，时间戳间隔 4/7/9s（对照：同口径此前 60 分钟 401 次、helper 每 5 秒重启一轮）
- [x] 逐源指标已接线：`sources[]` 含 `capture_p50/p95/max_latency_ms`、`last_decoded_frame_age_ms`、`queue_length/capacity`、`stage_p50/p95/max`、`stage_sample_counts`、`effective_hz`；顶层按设计只聚合计数/速率（`latency_scope=per_source`）
- [x] 丢帧计数自洽：双路采样 `drop_rate≈0.39`，与 decode/processed 一致；AI 队列始终有界（≤2）
- [x] 本地基线：`uv run pytest` rc=0（进度 100%，无失败）、`uv run ruff check .` 全绿；新增回归覆盖 RGA 低亮度 clamp、auto 转换器择优、逐源指标与速率求和、断流退避/epoch、阶段窗口过期
- [x] RGA 低亮度修复已部署（Y<16 clamp 到 16）
- [x] RPP 开关从裸环境变量搬进 `Settings`（`rpp_release_gil` / `rpp_engine_cache_enabled`），后端显式传入引擎工厂，面板经 `effective_config` 自证 `rpp_release_gil` / `rpp_engine_cache_enabled` / `rpp_detection_execution_wait_mode`；新增回归覆盖「显式参数优先于环境变量」与「hybrid 工厂收到 Settings 标志」
- [x] 核对盒上 SDK 前置：`liburpp.so.2` 导出 `T rtStreamCreate/Destroy/Synchronize`（共 11 个 `rtStream*`）；容器入口脚本已 export `PYTHONPATH/LD_LIBRARY_PATH=/usr/local/rpp/lib`（`docker exec` 里 import 不到 pyrt 是探测方式问题，非缺陷）
- [x] 本轮本地基线：`uv run pytest` rc=0、`ruff` 全绿（含新增三项回归）；实测 4K 亮场 RGA 16.21ms < OpenCV 18.48ms、暗场 25.38ms > 18.50ms → 暗场自动回落 CPU 合理

## 盒上实测（RPP 原生流与 NPU `load` 口径，2026-10-14 17:xx）
- [x] `liburpp.so.2` 导出 `rtStreamSynchronize` 等 11 个 `rtStream*`；容器入口脚本用自算 `LIBDIR` 注入 `LD_LIBRARY_PATH`（只给 `PYTHONPATH` 时 pyrt 因缺 `libpython3.11.so.1.0` 导入失败，属探测方式问题非缺陷）
- [x] 用**本地未部署版** `rpp_backends.py` 临时 `docker cp` 进容器冒烟，跑完立即还原：还原哈希 `e8c73231a092…` 与改动前一致（共两次）
- [x] `sync_execute` 与 `native_stream_sync` 输出 **max_abs_diff = 0.0**；单线程 60 次 p50/p95：sync **91.64/117.19ms**、stream **95.03/124.53ms**（单线程释放 GIL 无收益，+3~6% 开销）
- [x] **双线程对照（并发一个 CPU 密集线程，模拟双路）**：release_gil 关 40 次 4488.2ms（112.2ms/次）+ 并发 `cv2.resize` 25.1Hz；开 4091.3ms（102.28ms/次）+ 42.5Hz → RPP 吞吐 **+9%**、并发 CPU 工作 **+1.7×**。`run()` 仍全程持 `_RPP_EXECUTION_LOCK`，两次 RPP 调用依旧串行，收益来自设备等待期间让出 GIL（修正上一版"两路仍串行故收益待定"的措辞）
- [x] 同输入对照：RKNN SCRFD(1280×736 int8) p50/p95 = **54.71/57.40ms**，比 RPP(BF16) 91.64/117.19ms 快约 1.7×（精度不同，改路由前必须同输入端到端 A/B）
- [x] NPU `load` 口径判**不可用**：空闲 + RPP 负载 + RKNN 负载约 4 分钟、30+ 次采样恒为 `100@950000000Hz`，而 GPU 同名属性空闲报 `0@300MHz`

## 调度端断流看护（2026-10-14 18:xx，`airport-robot-dispatch`）
- [x] 定位根因：机器人 `maxDurationSeconds` 与 publish token 寿命都写死 **1800s**，而寻人任务可跑 3 小时 → "任务启动时一次性下发"在 30 分钟后必然静默断流，且**没有任何组件会重发**
- [x] 核对机器人侧语义：`RobotVideoStreamController.start()` 的 fingerprint 含 `publishUrl`(token) 与时长；不满足 allAlreadyRunning 时先 `stopAllLocked()` 再只重建命令列出的源 → **看护必须重发任务的全部源**，只补缺失源会把健康那一路一起停掉
- [x] 实现（`app/services.py`）：抽出 `_video_stream_command()` 与首次启动共用（避免 token/载荷口径漂移）；新增 `start_video_stream_watchdog()` + 周期 `_video_stream_watchdog_tick()`（缺失源超 grace 后重发全部源；全部在线但在上限前 `lead` 秒内则预防性刷新）+ `_reissue_video_streams()`（持 `_video_stream_lock`、逐源换新 token、不阻塞等待、失败绝不触发首次启动的清理路径）+ `_online_source_ids()`（状态展示复用）；`app/main.py` lifespan 启动、`close()` 取消
- [x] 新增 4 项设置：`video_stream_watchdog_interval_seconds=10`、`video_stream_watchdog_offline_grace_seconds=30`、`video_stream_watchdog_retry_backoff_seconds=20`、`video_stream_reissue_lead_seconds=300`
- [x] 回归：新增 `tests/test_video_watchdog.py` 7 项（掉一路后重发全部源且 token 刷新 / grace 内不动作 / 媒体列表不可读不动作 / 到期前刷新 / 无活跃任务不动作 / 不抢在显式启动前 / 只启动一次且 close 取消）全绿；ruff 通过
- [ ] 待现场：装新 APK 后做真实断流 + 跨 30 分钟阈值演练（并入终验 I）

## 调度端看护已上线（2026-10-14 12:37Z 发布 / 12:44Z 补日志）
- [x] 发布目录：`/home/EIS100RK3588/apps/airport-robot-dispatch/releases/20261014T1237Z-video-watchdog`，`current` 已切到它；`.env.production` 是指向 `shared/.env` 的 symlink，重建前后逐项核对**有效环境变量完全一致**（只有代码变化）
- [x] 回滚点：镜像 `airport-robot-dispatch-box-backend:before-video-watchdog`（=旧 `latest` 87aeaf801592）保留；回滚 = `current` 切回旧 release + `docker tag before-video-watchdog latest` + `up -d --no-build`
- [x] 部署验证：容器 healthy、`/health` = `{"status":"ok"}`、`alembic` 无操作、后台任务查询正常；容器内 `app/{services,config,main}.py` 哈希与本地逐字节一致
- [x] 发布前用 diff 证明"盒上版本 == 我改动前的基线"：services.py 的 7 处、main.py 的 1 处全部是我的改动；config.py 除我的插入外仅 `video_publish_base_url` 默认值不同（被 `.env.production` 覆盖且机器人会重写主机名，属惰性差异）
- [x] 新增启动日志并把级别定为 WARNING（本项目生产日志默认只出 WARNING+，INFO 不可见）：`Video stream watchdog started interval=... grace=... backoff=... lead=...`，用于证明看护真的在跑
- [ ] 待你：起一次真实任务做断流演练（拔一路 / 等 30 分钟跨阈值），观察是否自动重发并恢复
- [x] 最终部署验证（12:44Z 第二次重建）：容器内 `app/services.py` = 本地 `74a481180e04700a`；启动日志实测出现 `Video stream watchdog started interval=10.0s grace=30.0s backoff=20.0s lead=300.0s` → 看护任务确在运行；容器 healthy、`/health` ok、`watchdog tick failed` 计数 0
- [x] Review 中发现自己改动的一个真实缺陷并修复（第三次重建）：原实现把 `TaskStatus.FINISHING` 也当作"活跃任务"，任务结束清理推流时，30s 宽限后看护可能把已结束任务的推流重新拉起；现限定为仅 `ACTIVE`（`_video_stream_watchdog_tick` 与 `_reissue_video_streams` 双 guard），并补回归 `test_watchdog_does_not_restart_streams_for_a_finishing_task`；dispatch 全量 **171 项通过**、ruff 通过
- [x] 最终部署核验：容器内 `app/services.py` = 本地 `089e67381986c3f0`；启动标记实测再次出现；容器 healthy、restart 0；回滚镜像 `before-video-watchdog` 仍在
- 说明：启动标记用 WARNING 级别是刻意的——应用无 logging 配置，root 停在 WARNING，INFO 在生产不可见；任何 tick 异常都会以 WARNING `Video stream watchdog tick failed` 每 10s 出现一次，可直接 grep 判断存活

## 实测修正（收敛或推翻上一版结论）
- [x] 「预览已达 15fps」是顶层求和口径：`PREVIEW_HZ=15` 只是上限，逐源实测中位数 4K **4.2fps**、720p **7.8–9.1fps**（顶层 10–15fps = 两路之和）
- [x] 4K 采集 p95 ≈ **420ms**（385–435ms），720p ≈ 185ms；而 NV12→BGR 转换本身仅 11.3ms（`converter=opencv_selected`）→ 420ms 来自管道读/调度/占用，不是颜色转换
- [x] 全帧人脸检测逐源实测 **1.0–3.9Hz（中位 2.0–2.7Hz）**，配置 4Hz + 错峰 → 距 8–12Hz 目标很远
- [x] 帧龄尾部大：4K 中位 33–399ms、最大 7.2s；720p 最大 5.6s（断流窗口）
- [x] 顶层 `queue_capacity` 是逐源求和（2+2=4），非配置冲突；逐源 `queue_capacity=2` 与 `frame_queue_size` 默认一致
- [x] NPU `load` 口径未确认：空闲（无搜索、`rpp_server` 1.3% CPU）相隔 20s 两次均报 `100@950000000Hz` → 撤回「NPU 满载即瓶颈」
- [x] ZLM 帧计数不能当解码率：计数差分 4K 15.5/s（声明 20fps）、720p 18.3/s（声明 15fps）

## 待办（重排：双路长跑放到最后终验）
- [ ] A RPP 双路吞吐验证（离线）：本地部分已完成（见下）；**仅剩盒上原生冒烟**（建流/同步/销毁 + `PERSON_SEARCH_RPP_RELEASE_GIL=true` 下一次真实 SCRFD run，不部署镜像）
- [ ] A0 部署第 5 步前，先把 `deploy/Dockerfile.rk3588.performance` 的 `BASE_IMAGE` 改为 `perfpreview-20261014-v1`（沿用旧基镜像会静默回退预览提频/逐源指标/RGA 修复）
- [ ] B 带宽决策（单路短测即可推进）：同一路分别测 4K 主码流 vs 子码流（`Channels/102`）的单源入站/处理/采集 p95，决定是否切子码流
- [ ] C 指标口径收口：NPU/RGA `load` 受控基线（同一输入、有/无推理对照），确认口径后再引用
- [ ] D 上游自愈（跨仓库，需授权）：机器人上报 phase/失败分类 + FAILED 按实例清理（需装新 APK）；调度按 `source_ids_from_media` 有界退避重发并重签 token
- [ ] E 检测频率 A/B（单路短测可得比例）：`FACE_DETECTION_HZ_CUDA` 4 vs 6 vs 8 → 逐源 `effective_hz.face_full` 与 `achieved_sampling_hz`；精度回归
- [ ] F 延迟预算 A/B（单路短测）：jitter 500 vs 300/150/100ms；相机 GOP 4s→1s；回归绿帧与远脸
- [ ] G ZLM 冗余：确认无 HLS/音频消费者后关闭 `enable_hls`/`enable_audio`
- [ ] H 真端到端实测（需现场配合，可延后）：画面内毫秒时间码比对预览画面
- [ ] I **终验（最后做）**：真实双路连续 ≥10 分钟（争取 ≥2h），逐源全字段 + ZLM `bytesSpeed` + eth0 RX/TX 同步采；断流/恢复演练；H 的 E2E 结论
- [ ] J 仅在 A–H 不达标时：WebRTC 预览、DMABUF/共享内存减少 4K 复制

口径说明：单路 1–3 分钟短测只给比例与相对结论（记录窗口与配置），不能替代 I 的稳定性结论；I 需要用户开真实双路，已确认放到最后。

现场结论：本轮把「已写未部署」这条前提消掉了——生产已在 `perfpreview-20261014-v1` 上运行审查过的五个文件，断流退避与逐源指标均已生效并有生产证据。真正的剩余瓶颈是三项：4K 采集 p95 ≈420ms（占住采集线程）、每路全帧检测仅 ≈2Hz（4Hz 配置加错峰）、以及网络预算（两路入站 ≈9–11 Mbps 对盒子公网 ≈9.8/10.2 Mbps，两路高帧率预览上行不可行）。NPU 与 ZLM 帧计数的口径都不可直接采信，相关结论已撤回或降级。机器人侧仍无自愈，需装新 APK 并由调度补看护。
# `.215` 修复识别服务能力契约不兼容（2026-10-14）

- [x] 复用调度容器已有的 `DISPATCH_INSIGHTFACE_EVIDENCE_API_KEY`，安全写入识别服务 `runtime.env` 的 `PERSON_SEARCH_EVIDENCE_API_KEY`
- [x] 增加容器健康检查：`/readyz` 就绪且 `/healthz` 声明调度端要求的全部 6 项能力；Compose 强制要求 runtime env 文件存在
- [x] 完成能力测试、镜像更新和盒端重建；只检查密钥匹配结果，不输出密钥或散列
- [x] 验证全部能力、识别 provider、健康状态和证据接口鉴权，记录部署前后状态与重启次数

## 根因（现场只读核对）

调度端 `REQUIRED_INSIGHTFACE_CAPABILITIES` 要求 `replace_active`、`active_search`、
`search_timeout`、`request_lookup`、`event_replay`、`confirmed_evidence_v1`。盒端识别服务
当前 `/healthz` 返回前五项；`confirmed_evidence_v1` 只有在设置
`PERSON_SEARCH_EVIDENCE_API_KEY` 后才发布。识别容器没有该变量，调度容器已有
`DISPATCH_INSIGHTFACE_EVIDENCE_API_KEY`，且已配置证据加密密钥。不是缺少 API 实现：当前
服务 OpenAPI 已有证据 GET/DELETE 路由，镜像代码会按配置发布能力。

## 执行与验收

- 调度服务的证据 API 密钥已配置；识别服务环境里原先没有对应变量。Compose 指定的
  /ssd/person-search-rpp/runtime.env 当时也不存在，且 required: false 允许服务静默启动。
- sync_rk3588_evidence_key.py 从调度容器读取现有密钥，不打印密钥，创建 root-owned、0600 的运行时文件。
  盒端核对两个容器的密钥完全相同。
- 新增 check_rk3588_contract.py，Compose 和 Dockerfile 健康检查同时核对 /readyz 与全部 6 项能力；
  Compose 现在强制要求运行时环境文件存在。能力测试精确断言六项集合。
- 盒端镜像 person-search-rk3588-rpp:streamcontract-20261014，revision 同名；容器 running/healthy，
  restart count 为 0。健康检查通过；provider 为 RKNN_NPU、RPP_BF16、RKNN_NPU；/healthz 返回六项能力全量集合。
- 证据 GET 和 DELETE 均验证：无密钥为 403；调度共享密钥通过鉴权，到未知会话资源后为 404。
  过程只记录状态码和密钥匹配布尔值，不输出密钥或散列。
- tests/test_api.py：14 passed。检查中发现监控页测试还锁定旧 Mac RTSP 地址，已将断言和 README
  监控默认值更新为盒端 ZLMediaKit 两路地址。Ruff、Python 编译、git diff --check 和盒端 Compose 配置校验通过。
- 本轮没有运行中的搜索会话；ZLMediaKit 此前核对为 0 路上游流。能力兼容已修复，真实流的识别与证据图片
  端到端验收仍需上游推流和新寻人任务。

# `.215` 双路流与内网监控（2026-10-14）

- [x] 复现第一路 H.265 容器内 `failed to init mpp ctx`，确认同段视频宿主 MPP 可解
- [x] 在隔离容器中验证设备树只读挂载 + `systempaths=unconfined` 可恢复 H.265 MPP 解码
- [x] 定位断流后 GStreamer stdout 读取可能无限阻塞，给整帧读取加超时以触发原有重连循环
- [ ] 现场验证断流后进入 `source_lost`，恢复推流后重新出帧
- [x] 正式容器切到 `streamcontract-20261014`（包含前述 H.265 修复），`readyz` 就绪、设备树可读、第一路录制 H.265 样本可硬解
- [ ] 上游重新推流并启动寻人后，验证第一路实时画面和处理 FPS、第二路识别、真实断流恢复
- [x] 核对盒端 ZLMediaKit、识别容器、两路流及当前任务状态
- [x] 定位第一路无帧原因，区分上游推流和识别拉流故障
- [x] 监控页自动接回当前任务，并显示每路流的状态、FPS、分辨率和识别结果
- [x] 部署页面并从内网 HTTPS 入口核对页面、API、WebSocket 和第二路 MJPEG 预览

现场结论：先前两路均由 ZLMediaKit 接收；`ip-camera` 为 3840×2160 H.265，识别源 1 保持 0 FPS；`local-camera` 为 1280×720 H.264，识别源 2 约 10 FPS，曾出现目标确认（最佳相似度约 0.654）。第一路故障定位到旧容器缺少可读的 `/proc/device-tree/compatible`，新容器已在盒端用第一路录制数据解出 H.265 帧。10 月 14 日 11:03 上游两路推流同时断开，ZLMediaKit 的 `getMediaList` 当前返回空列表；新容器重建后旧内存会话失效，故实时出帧、识别和推流恢复验收等待新任务与上游推流。断流读取超时已用盒端沉默子进程核对：设 1 秒截止时间后按时返回并触发重连路径。

# `.184` RPP 检测 + RKNN NPU 识别容器化

本轮继续：完成 `ipc: host` 与 SDK 对象生命周期修复；恢复故障后的 RPP 驱动，再完成硬件 readiness 和同输入性能对比。

## `.184` 切换到容器基线（2026-10-11）

- [x] 确认 `18080` 由旧原生单视频流 `person-search-rk3588.service` 占用；其代码目录为 `/ssd/person-search`
- [x] Compose 与镜像默认端口改为 `18080`，保留 `18081` 作为显式隔离测试端口
- [x] 停止并禁用旧原生 unit，保留代码目录和 unit 文件
- [x] 让 `mppdma-20261011` 容器接管 `18080`；健康检查通过、重启 0 次、monitor HTTP 200、仅 18080 监听
- [ ] 重新登记目标，用真实 RTSP 完成单路/双路端到端 FPS、识别和断流验收

> 下面记录的是切换前的 staging/性能测试时间线；其“18080 保持原生服务”状态已被本节的后续切换取代。端到端性能验收仍未通过。

## Mac 摄像头 RTSP 盒端验收（2026-10-11）

- [x] 确认 Mac RTSP 发布端输出 H.264 1280×720@30，并成功解出实际视频帧
- [x] 记录 staging 搜索 85 个样本：持续 `INITIALIZING`、0 FPS、0×0；不把它误报为推理性能
- [x] 定位 staging 容器缺少全部 `/dev/dma_heap/*`，而 184 主机 MPP 解码可工作
- [x] Compose 映射 6 个 DMA heap 设备；停止搜索时主动中断 GStreamer helper
- [x] 全量 pytest、Ruff、锁文件和 Compose 配置校验通过
- [x] 仅重建 `.184:18081` staging；容器内显式 H.264 MPP helper 输出有效 `1280×720 NV12` 帧；服务 healthy、无重启、helper 已清零
- [ ] 重新登记目标后采集真实视频端到端 FPS；生产 `.184:18080` 保持不动

## 双路低分辨率评估（2026-10-11）

- [x] 同盒同图测 1920×1080、1280×720 的 RKNN 与 RPP+RKNN 双检测延迟
- [x] 按用户澄清改为一个寻人会话同时接两路视频：检查 API、采集、跟踪、确认和事件来源的现有边界
- [x] 两个隔离执行单元组成一个对外会话：各自采集/跟踪/确认、共享目标/模型，任一路命中后收束；兼容单路 API
- [x] 监控页支持第二个 RTSP 输入和双路预览；状态与命中事件标记 `source_id`
- [x] 本地双路模拟采集、任一路命中、停止清理、来源脱敏、参数校验回归
- [x] 双路镜像 `dual-20261011` 部署至 `.184:18081` staging；healthy、零重启，RPP_BF16/RKNN_NPU/RKNN_NPU readiness，通过双路/单路 API 结构烟测；`18080` 生产未动
- [ ] 用两条真实 RTSP 验证断流/重连与端到端 FPS；对比 720p/1080p，确定生产配置

- [x] 确认 `.184` RPP SDK/设备可用；同样本 RPP/ONNX SCRFD 检测框差 <1 px
- [x] 对比 ArcFace：现有 RKNN（含 TTA）比 RPP 端到端路径更快；识别保留 RKNN
- [x] 停止本轮旧版 `18081` staging 容器；确认原生 `18080` 健康且未切换
- [x] 实现 `hybrid` 路由：SCRFD 搜索/登记走 RPP；ArcFace 和 YOLOX 走 RKNN NPU
- [x] 强制 provider：face detection=`RPP_BF16`，face recognition/person detection=`RKNN_NPU`；禁止 CPU 推理回退
- [x] 预生成并校验 SCRFD 1280×736 / 640×640 静态图，镜像只读挂载；容器不做 CPU 图优化
- [x] 保持 `arcface-v1-rknn` embedding contract 与现有 `w600k_r50.rknn`，不改变登记特征语义
- [x] ARM64 镜像排除 ONNX Runtime 推理依赖；挂载 RPP/RKNN 设备与主机采集运行库
- [x] 全量测试、ruff、锁文件、Shell 语法、Compose 与 diff 校验通过；静态图已生成，镜像已构建
- [x] 定位并修复容器与宿主 `rpp_server` 的 SysV 共享内存隔离：Compose 使用 `ipc: host`
- [x] 保留 RPP Logger、模型/plan 数据至引擎释放；增加缓存加载生命周期回归测试
- [x] 获授权后受控重启 `.184` 恢复 Oops 的 RPP 驱动；原有调度、媒体和 `18080` 服务均恢复健康
- [x] 定位 BF16 plan 反序列化原生崩溃，默认禁用 plan 缓存；旧 plan 移至 staging 隔离目录保留
- [x] 定位首次执行缺少 RPP `include/kernel` 配置，Compose 增加只读挂载
- [x] 定位 RPP SDK 首次执行需 `gcc`/`ld` 编译设备 kernel，镜像补 ARM64 编译工具与 C headers
- [x] 修复 fresh-build 路径释放无用 `pyrt.Runtime` 时的 SIGSEGV；最终镜像独立基准进程正常退出
- [x] `.184` 隔离 `18081` 的 RPP 搜索/登记两图真实设备执行与 readiness；provider 为 RPP_BF16/RKNN_NPU/RKNN_NPU
- [x] 同图人脸框约 1px 内一致；单图测速 RPP 搜索 95.36ms > RKNN 40.93ms，登记 100.09ms > 85.69ms
- [x] 4K 升采样图双模型串/并行对比：混合 241.95/183.27ms，纯 RKNN 169.80/131.45ms；性能验收未过，不切换生产
- [ ] 用真实 RTSP 视频源验证采集与整链吞吐；目前 staging 未提供该源，静态图不能代替现场验收
- [x] 性能测试未支持自主切换；之后按用户明确要求将容器设为优化基线，性能验收仍未通过

## 切换前盒端状态（历史）

- 当时 `18080` `/readyz` 为 `ready`，三个原 provider 均为 `RKNN_NPU`，embedding contract 未变。
- 当时 `18081` staging 镜像 `perf-20261011-rppfix4` healthy、重启次数为 0，但性能较纯 RKNN 路径慢；此状态已由下方明确授权的端口切换取代。
- 宿主 `rpp_server` 已随重启恢复；重启后 SDK plan 加载和容器资源挂载分别暴露新问题，代码和 Compose 已修复，待两图真实执行确认。

## 切换后盒端状态（2026-10-11）

- 旧 `person-search-rk3588.service` 为 `inactive/disabled`；`/ssd/person-search` 和原 unit 文件均保留。
- `person-search-rk3588-rpp:mppdma-20261011` 现在监听 `18080`，healthy、重启次数 0；provider 为 RPP_BF16/RKNN_NPU/RKNN_NPU。
- `http://192.168.30.184:18080/monitor` 从盒端和 Mac 均返回 HTTP 200；`18081` 无监听，`/v1/searches/active` 为 `null`。
- 容器重建会清空进程内登记；真实视频、双路吞吐和识别准确率尚未验收，不能声称性能已经提升。

# RK3588 识别 10–15 FPS 实测与 RGA 加速（2026-09-15）

- [x] 只读核对现网：三模型均为 `RKNN_NPU`、A76 affinity 生效、NV12 桥接生效，`target_loop_hz=4`
- [x] 盒端静态 4K 基准：YOLOX 108.2ms、SCRFD 65.0ms、串行 163.0ms（完整检测上限 6.13fps）
- [x] 为 `/dev/rga` 配置持久 `root:render 0660` 权限并验证非 root 可访问
- [x] 实现 RGA NV12→BGR 可选加速、首帧实测择优、CPU 自动回退和运行时可观测性
- [x] 补齐单元测试、全量 pytest、ruff 与 diff 静态检查（338 passed）
- [x] 部署到 `.203`，验证三模型 readiness、设备权限和 RGA 失败自动回退
- [x] 用真实 RTSP A/B 全帧检测同帧/错峰；`target_loop_hz` 只控制 ROI 信用，不再误作循环限速旋钮
- [x] 最终配置保留错峰和 3 路 ROI：处理循环 4.93→10.25fps；全帧人脸约3.52Hz，未冒充10Hz完整识别
- [x] 撤回 `roi_max_tracks_per_pass=1`：机场多人补检容量保持 3，不以降低多人召回换帧率

# RK3588 登记检测模型隔离（2026-09-14）

- [ ] 在盒端用同一近距离人脸 A/B `SCRFD 1280x736 INT8` 与 `SCRFD 640x640 FP16`，记录检测分和质量门结果
- [ ] 核对登记调用链只选择独立 FP16 640 图，搜索/ROI 继续使用 1280x736 INT8 图
- [ ] 补齐生产配置、可观测性与回归测试，不调整 `min_enrollment_detection_score`
- [ ] 部署并通过 `/readyz`、真实登记请求及搜索静态输入检查

# RK3588 性能与识别率优化（2026-09-14，实测驱动）

实测见 `docs/rk3588-deployment.md` 的「性能基线」节。推理全部保留在 NPU；CPU 侧只做解码后转换、缩放、NMS、预览。

- [x] P0 配置：`CPUAffinity=4-7`、关 CMC、`target_loop_hz=4`、`match_profile=transit`、`face_track_buffer_seconds=2.5`
- [x] P1 `GStreamerCapture.grab()` 不做 NV12→BGR（丢弃帧不再付 19ms 转换）
- [x] P1 `rknn_core_mask`（`rknn_set_core_mask`，默认 0 不改行为；缺符号/被拒只告警）
- [x] P1 RKNN 后端上报真实静态输入尺寸与掩码是否生效到 `effective_config` / 面板
- [x] P1 采集 helper 的 stderr 尾部进日志（脱敏 RTSP 凭据）
- [x] 文档：CPU 是 4×A76+4×A55；`/dev/rga` 权限；基线数字
- [x] 测试 + ruff 全绿（新增 7 个测试）
- [x] 部署到 .203、重启：三个 provider 均 `RKNN_NPU`，进程 affinity `4-7`，环境变量全部生效
- [x] 步骤 0：RKNN vs ONNX 一致性 —— ArcFace cos **1.0000**（3 个裁剪），SCRFD 分数相关 0.9999、
      max|diff| 0.0011。旧快照的 `best_observed_similarity=0.0246` 是目标不在画面，不是链路问题。
- [x] ArcFace 对齐改为 Umeyama 最小二乘（与 InsightFace/skimage 参考逐元素一致；LMEDS 在远脸噪声下 10% 裁剪偏 ≥6px）；已部署
- [x] **现场读数**（2026-09-14 20:27 任务，用户坐姿约 8 分钟后走近）：处理 6.0–6.6 fps / 源 13 fps，无重连；
      坐姿人脸 56–165px（4K）落���脸档，最高相似度 0.537 < 0.64，卡点 `similarity_low`（关联与 ROI 均正常：
      `person_strict` 677、`person_relaxed` 34、`face_fallback` 110、`roi/frame` 0.55–0.96）；
      走近后普通档 3 帧、相似度 0.653 确认，调度侧收到并完成返程。
- [ ] **现场读数（对齐修正后）**：跑一次真实寻人，读 `processed_fps`、`stage_p95_latency_ms`、
      `unconfirmed_gate_counts`、`association_counts`（期望 `person_relaxed`>0、`unassociated` 占比下降）、
      `roi_calls_per_frame`>0、`rknn_core_mask_applied`。
- [x] P2/P3：盒上转换 + 基准 + 留出帧 ONNX 一致性（表见 docs）。定案 SCRFD 1280×736 INT8（48ms，
      53/53 匹配）+ YOLOX 640×384 FP16（78ms，91/91）；YOLOX INT8 丢 8/91 人弃用；ArcFace 保持 FP16。
      已部署（21:35），`static inputs: person (384, 640) face (736, 1280)`。
- [x] 登记照被 INT8 远景模型拒绝（`detection_score_low`）→ 新增 `rknn_enrollment_face_detection_model`，
      登记指回 FP16 640²，登记门槛不变；已部署（`search (736,1280) / enrollment (640,640)`）。
- [ ] **现场读数（新模型）**：用户重新登记后坐姿对比 `best_observed_similarity`（旧模型 0.537）。
- [ ] 运维：请管理员放开 `/dev/rga`，之后 `mppvideodec format=BGR` 可省掉全部 CPU 色彩转换。

## 复盘

- 1 FPS 的根因不在推理而在采集：`videoconvert` 在 4K 上 1.18 fps；NPU 三个模型合计每帧 ~190ms。
- 「坐着只露头 / 行人」识别差的共同约束是 640² 静态 SCRFD 对 4K 帧的 6 倍降采样（76px 脸 → 12.7px），
  不是阈值；下一步是输入形状与 INT8，不是调 similarity。
- systemd 的 `EnvironmentFile=` 无论书写顺序都覆盖 `Environment=`，unit 里放在其后的重复块不起作用；已删除。

# RK3588 RTSP 绿帧与预览延迟修复（2026-09-14）

- [x] 增加 GStreamer 稳定缓冲配置，修复 H.265 绿帧
- [x] 增加 NV12 帧桥接，绕过 4K `videoconvert` BGR 瓶颈
- [x] 将最新帧预览从推理线程解耦，并补充预览延迟指标
- [x] 补齐本地测试、部署配置和重启后的现场验证
- [x] 在 RK3588 上重启服务、重新登记目标并完成页面流验收（VLC 同屏视觉对比仍需人工完成）

## 现场结果

- 盒端生效配置：`gstreamer / nv12 / tcp / 500ms / drop=false`。
- `/readyz`：person、face detection、face recognition 均为 `RKNN_NPU`。
- 页面 MJPEG 连续 12 秒收到 70 帧，`bad_jpegs=0`、`green_frames=0`，约 5.66 FPS。
- 推理循环从原先约 1 FPS 提升到约 3.6–5 FPS；页面帧龄 P95 约 175ms。

# RK3588 寻人现场识别优化（2026-09-14）

- [x] 现场验证坐姿/近距离链路：`person_relaxed`、`face_fallback` 均实际生效，近距离普通脸成功确认。
- [x] 定位当前瓶颈：平均处理约 `2.84fps`、丢帧约 `57.8%`、`face_embed` P95 约 `173ms`，ROI 仅运行 `1` 次。
- [ ] 保持全帧人脸检测为寻人广撒网主通道；ROI 只能作为独立补充通道，不能替代全帧检测或因 ROI 饿死全帧。
- [ ] RK3588 单目标寻人先以 `roi_max_tracks_per_pass=1` 做现场 A/B，不提高全局循环频率、不降低相似度门槛。
- [ ] 将全帧检测、人体跟踪和 ArcFace 识别的预算解耦，在保住全帧召回的前提下降低识别阻塞。
- [ ] 若 ROI 仍不足，再评估 CMC/ArcFace RKNN 路径的耗时，以及静态 SCRFD 1280 模型的召回/延迟回归。
- [ ] 只验证寻人服务；登机口变更逻辑不在本轮范围。

# RK3588 原生部署适配（eis100，2026-09-10）

## 本轮清单

- [x] 增加 RKNN Runtime（ctypes）及 YOLOX / SCRFD / ArcFace 后端
- [x] 增加 RK3588 GStreamer 硬解输入，保留 OpenCV/ONNX 默认路径
- [x] 增加 RKNN 模型转换脚本、原生部署脚本和盒子配置说明
- [x] 补齐配置、后端、视频输入测试，并通过本地质量检查
- [x] 在盒子上做无破坏性 smoke test，确认端口、NPU 和服务 ready
- [ ] 等用户提供真实 RTSP 与登记照片后，完成现场识别及阈值标定

---

# 线上 embedding、版本与生命周期收尾（2026-08-31）

## 基线

- 工作树干净，`main` 与 `origin/main` 同步；待收尾改动位于 `b97eb9b` / `ca1c6b7`。
- `uv run pytest`：14 failed, 264 passed；合法 embedding 路径因 helper 缺失进入 `FAILED`。
- `uv run ruff check .`：6 个 F821、1 个 SIM102。

## 任务

- [x] 1. 补齐 embedding 数值校验、provider 输出对账与失败指标。
- [x] 2. 固定 ArcFace embedding 契约并贯通登记、搜索、事件和 readiness。
- [x] 3. 修复同目标 `replace_active` 的 embedding 生命周期竞态。
- [x] 4. 修复 reader 假停止、模型半初始化与敏感清理重试边界。
- [x] 5. 对齐 package/API/build revision，并将强 readiness 接入 T4 canary。
- [x] 6. 补齐单测、全量测试、ruff、bash 语法和 diff 检查。

## 验证

- `uv run pytest`：314 passed；原始 14 个失败全部消除。
- reader / detector / service 并发与生命周期测试并行重复 5 次，全绿。
- `uv run ruff check .`、`bash -n scripts/deploy_t4.sh`、`git diff --check` 全绿。
- 本机真实 YOLOX + `buffalo_l` readiness 通过：CPU provider、固定 SHA-256、112×112、
  512 维和 flip-TTA contract 均与 manifest 一致。

## 后续边界

- 未执行真实 T4 容器切换或现场 RTSP 验收。
- 后端显式 `close()` / in-flight lease、离线 CLI 全路径资源托管、VLC `kill` 兜底留后续。

---

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

### 部署（192.168.17.60，2026-08-28）

- [x] `cb1c82c` 推送 GitHub，服务器 `insightface-release-cb1c82c` 构建部署，容器 healthy，
      部署日志验证 `CUDAExecutionProvider`
- [x] 回滚点 `person-search:t4.rollback-0dffc78-20260828T081235Z`（上一次 0dffc78 上线
      漏了打标签，这次补上）
- [x] `current` 符号链接更新到 cb1c82c（此前停在 6ceb350，已停留三代）
- [x] 生产配置核对：`match_profile=conservative`、`departure_adjudication=False`、
      远脸仍 `6/3.0s/5 票`、`evidence_statistic=median`、`face_detection_size=1280`
      —— **本次上线行为零变化**，只多了可观测性
- [x] `/monitor` 已带 `confirmLatencyDiag` / `blockerDiag` 两个新格子

**踩到的坑**：release 目录逐代 `cp -a`，`origin` 指向上一个 release 目录而非 GitHub，
`git fetch origin main` 静默 no-op、HEAD 停在 0dffc78 而 `git status` 干净。已 `remote
set-url` 修正并把这条写进服务器 `~/deploy/README.txt` 与 lessons。

- [ ] 下一步仍是**现场读数**：跑一次真实寻人，看 `实际采样率 vs 需求采样率` 和
      `未确认卡点` 分桶，再决定是减负载还是标定阈值。切 transit 档只需
      `T4_MATCH_PROFILE=transit ./scripts/deploy_t4.sh` 重跑一次。

### 部署（192.168.17.60，2026-09-01）

上线 `6c7ae3c`，容器 `person-search:t4` healthy，`current` 已指向
`insightface-release-6c7ae3c`。本次是**行为变更上线**，不是纯加固：

- `PERSON_SEARCH_MATCH_PROFILE=transit`（原 conservative）：远脸档
  `6 帧/3.0s/5 票` → `4 帧/2.0s/3 票`，检测分 `0.65 → 0.55`，
  统计量 `median → top_k_mean`，采样间隔全档位 `0.2s → 0.1s`。
  **没有动任何相似度阈值**——放宽的是帧预算和检测分。
- `PERSON_SEARCH_TINY_FACE_SHADOW_MODE=false` + `T4_ALLOW_PHYSICAL_ACTIONS=true`：
  维持线上原有行为（远脸命中直接发 `target_found`）。新脚本默认是 shadow，
  保留现状需要显式开这道闸。
- 其余不变：`face_detection_size=1280`、`departure_adjudication=false`、
  `bind 0.0.0.0`。Evidence API key 从在跑的容器里读出直接传入，未落地未打印。

回滚点：`person-search:t4.rollback-cb1c82c-20260831T141230Z`（上线前那版）、
`person-search:t4.rollback-67f93ce-20260831T235506Z`，以及脚本自动保留的
`person-search.previous.20260901075509.3968025` 容器。

#### 这次上线卡了两轮，都在 Dockerfile 的 HEALTHCHECK 一行上

1. `0564295` **构建即失败**：`CMD-SHELL` 是 compose 的写法，Dockerfile 解析器直接拒绝。
   金丝雀流程让失败停在构建阶段，生产容器全程未被触碰。
2. `67f93ce` 构建通过、服务正常，但容器被标 `unhealthy`：修构建时顺手加的 `\$`
   是错的。Docker 的变量替换不作用于 HEALTHCHECK，`${}` 本就原样存入、由运行时 sh
   展开，加转义反而让 sh 读成字面 `$`，curl 退出 3。
   T4 实测 plain `rc=0` / escaped `rc=3`，`6c7ae3c` 修掉后容器 5s 内 healthy。

根因不是手误，是**一条把坏字符串原样钉住的测试**：
`assert 'CMD-SHELL curl ... $${...}' in source` 对着一个永远构建不出来的 Dockerfile
长期显绿。已改成断言正确形式 + 三条反向守卫，并逐个把坏形式代回确认会红。
详见 `tasks/lessons.md`。

另：服务器 `git fetch origin main` 期间 GitHub 连接被重置，改用 README.txt 里的
bundle 通道完成。`insightface-release-0564295` 目录留在服务器上（对应镜像从未构建成功）。

#### 仍未做

- [ ] **现场读数**（第三次列入）：跑一次真实寻人，读 `实际采样率 vs 需求采样率` 与
      `未确认卡点` 分桶。transit 放宽的是帧预算，**它是否解决问题取决于卡点在哪**——
      卡在 `insufficient_samples` 才有用，卡在 `window_statistic_low` 则无效且徒增误报。
      这次上线把判据摆上了监控页，但没有人读过。
- [ ] **transit 的误报代价没有测量**：检测分 0.65→0.55 和 median→top_k_mean 都是放宽，
      在候机厅多人场景下的假阳率未知。按 lessons 第五次要求，仍需 `--dump-similarities`
      跑正负素材定阈值。

# 双路视频性能第一阶段审查（2026-10-14）

- [x] 扫描盒端与机器人仓库，阅读既有部署记录和相关 lessons
- [x] 还原拉流、缓存、AI、预览与指标调用链（并行只读审查）
- [x] 只读核对 .215 运行版本、硬件路径和当前流状态
- [x] 形成带代码证据的问题清单、检查命令、测试矩阵和最小修改计划
- [x] 校验报告引用、命令与现有针对性测试；保留现有未提交修改

本轮仅新增排查文档/检查工具和任务记录，不修改业务代码，不重启或切换设备服务。

交付：`docs/dual-stream-performance-audit-20261014.md`（完整9节）、
`scripts/collect_dual_stream_audit.py`（只读脱敏采集）、
`docs/dual-stream-audit-20261014.samples.jsonl`（后续第一路断流的5个真实样本）。
146项针对测试通过；采集脚本Ruff、JSON/隐私白名单及实际盒端执行通过。
14:19正常双路约46秒采证，14:22第一路发布消失而第二路继续；未修改生产配置/重启服务。

## 双路性能只读续查

- [x] 复核当前流状态，定位14:21:49发布连接EOF的接入链证据
- [x] 核查机器人keepalive/单路失败上报和盒端转发超时
- [x] 补充报告：区分已确认中断与候选原因，提出最小验证项

续查结论：14:33第一路仍失联，第二路持续；Nginx RTSPS SNI透传至7554，stream timeout为1h。
机器人运行中FAILED没有上报/移除，相同fingerprint重试可误判已运行；已补报告R06–R08和P0-d。
普通用户与sudo -n均无法读取Nginx error.log，未修改权限。
续查验证：使用现有JDK17/Android SDK离线执行机器人四个推流相关测试类，17项全通过。
现有测试不覆盖控制器运行后FAILED与同参数重试，后续最小修复必须补行为回归。

# 双路性能最小优化实施（用户已授权）

- [x] 采集持续更新latest，AI覆盖旧帧/取最新，准确计数和重连退避
- [x] 每源完整滑窗指标/分阶段分位数，修前端标签与延迟范围
- [x] 机器人FAILED状态可见性及同参数真实重试、行为回归
- [x] 本地全量测试/静态检查，盒端同样本采集/预览梯度基准
- [x] 构建可回退镜像，检查活动任务并受控部署/就绪验收
- [ ] 记录优化前后实际吞吐/资源/延迟、尚未达到目标的限制

优化验证：识别380项全量测试、机器人189项全量JVM测试+lint+debug assemble通过。
盒端合成双路200ms慢AI对照：JPEG 4.91/4.91→14.48/14.75FPS，CPU127.51%→183.70%。
用户已明确批准切换；perfpreview-20261014-v1已执行受控Compose重建，正在等硬件readiness。

上线：perfpreview-20261014-v1 running/healthy、restarts0、readyz与六项能力通过；预览15Hz。
旧镜像及原Compose目录保留，runtime.env/模型/SDK未变。等待重新登记和双路上游实测。

## 2026-10-15 海康 1080p NV12 padding 导致无预览

- [x] 核对 ZLM 接收与 InsightFace 状态，定位 `source_lost` 的直接错误
- [x] 从实际硬件解码 sample 读取 VideoMeta，确认 stride/offset 与 caps 默认布局不同
- [x] 按真实布局打包 NV12，补充合成 padding、越界与 metadata 优先级回归
- [x] 在盒端独立验证修复后帧，再备份并更新 helper，验证活动搜索和 MJPEG 恢复

ZLM 已收到海康 H.265 / 1920×1080 / 15 fps，约 2 Mbps；解码器每约 10 秒以 rc=4 退出，
错误 `unexpected NV12 payload size: 3732480 != 3110400`。实际 VideoMeta 为 stride=[2304,2304]，
Y offset=0、UV offset=2488320；caps VideoInfo 却给出紧凑 stride1920、UV offset2073600。
本轮不修改机器人、不降画质、不重启识别/媒体容器，禁止保存或输出真实视频帧/人脸照片。

修复优先读取 per-buffer GstVideoMeta，缺失时才回退 caps VideoInfo，按两平面 stride/offset
拷贝有效行，保持紧凑 NV12 快路径与 PSN1 协议；不通过总长度猜布局、不直接截断 UV。
34 项新合成布局回归与 38 项 RK3588 原测试合计 72 项通过，ruff 与 diff 检查通过。
最终 helper SHA-256 `cee8cbc3c171dd02e54d251800e272163373dca4307ecc7b50c998e1628fd06a`。

盒端候选文件独立读取 10 帧真实硬件解码 sample，全部打包为 1920×1080 / 3110400 字节，
打包平均 16.172 ms、最大 30.856 ms（仅此次 10 帧样本，不作为长期性能承诺）。确认原 helper
SHA-256 `897e5052b256fce7a0934b781d2bcd8f5771a8c34bd84f0c2fe06bb08c5e292b` 与本地 HEAD 一致后，
备份至盒端 `/home/EIS100RK3588/apps/person-search-hotfixes/20261015-nv12-layout`，
原子替换运行容器的 `/app/scripts/rk3588_gstreamer_capture.py`，校验值与最终源码一致。

未重启容器，原活动搜索自行重试后状态恢复 `running`：1920×1080、decode_fps=13.7、
decoded_frames=603、processed_frames=416，最新帧年龄约 23 ms。MJPEG HTTP=200，
5.01 秒接收 3620864 字节、观察到 55 个 JPEG 分段头；全部数据在内存统计后丢弃，未保存图片。
浏览器页面已请用户刷新确认；本轮未改 Nginx/Android/摄像头配置，未重建容器镜像，未提交或推送此仓。
运行容器重启会保留文件，但从旧镜像重新创建会丢失修复，后续镜像构建必须包含本次源码改动。

## 2026-10-15 单路海康预览卡顿分段定位

- [x] 核对 ZLM 资源、封装配置与识别端实时吞吐/帧龄
- [x] 对比 ZLM 本地压缩帧输出、硬解输出及 MJPEG 实际到达间隔
- [x] 复核预览调度与 CPU 开销，按证据确定瓶颈和最小优化方案
- [x] 记录验证结果及仍未覆盖的公网/浏览器端限制

初始快照：ZLM CPU 6.52%，识别 CPU 162.79%；decode13.8fps、preview10fps，
preview encode p95 54.64ms。HLS 开启但未据此认定背压或擅自关闭；不保存真实媒体帧。

同一流并行只读探测（额外两个本地 RTSP 读者，持续约 32 秒，前 2 秒预热后统计）：

| 环节 | 实测 FPS | 帧到达间隔 P95 / 最大 | >200ms 间隔 |
| --- | --- | --- | --- |
| ZLM 本地 RTSP 压缩访问单元，尚未解码 | 15.007 | 121.595 / 185.079 ms | 0/450 |
| 同源独立硬解 NV12，不映射/复制图像 | 14.974 | 122.247 / 183.520 ms | 0/449 |
| 识别服务本地 MJPEG HTTP | 9.854 | 203.964 / 289.590 ms | 18/316 |

MJPEG 32.072 秒内 317 个分段头、19530941 字节，约 4.87 Mbps；对比媒体接收
233336～264001 bytes/s（约 1.87～2.11 Mbps）。真实压缩/解码图像及 JPEG 只在内存使用后
丢弃，未保存；仅输出数值/格式元数据。额外读者测试结束后 readerCount 回到 1。
18 次状态采样均 running，ZLM 媒体帧差分约 14.62～16.01fps，应用 decode13.4～14.1fps，
preview9.2～10.4fps、编码/resize/叠框/指标合计 P95 约 69～74ms。额外硬解探测会增加负载，
但探测前预览已约 10fps，故不能把本次数字当无探测长期性能承诺。

本窗口不支持 ZLM 收流/硬解吞吐不足为主要原因；卡顿在应用采集后处理/预览生成阶段
更明显。`service.py` 的独立预览线程读取 latest，不消费 AI 队列；15Hz 是轮询上限，
无新帧也空等一周期，超预算仍至少 sleep10ms。当前默认 960×540、JPEG quality82，
预览帧龄从 BGR 读取后计时，不含 jitterbuffer500ms，不能称真正端到端延迟。
最小后续方案是为 latest 增加有界 Condition 等待，按开始时刻限频、超预算不额外睡眠，
并测真实编码/拷贝阶段成本；本轮只定位，未改调度业务代码/画质/HLS/音频或重启容器。
尚未测浏览器真实收包与公网瞬时拥塞，也未证明偶发上游中断彻底消失。

ZLM 自带管理页 `/webassist/` 没有独立账号密码，使用 URL secret 参数；只向用户说明
盒端配置文件位置，未输出/保存 API 密钥，未降低外层认证。

## 2026-10-15 预览调度修复与媒体管理入口认证

- [x] 查实用户实际 `/mediaserver/` 入口的两层认证，定位现有媒体账号，不修改认证
- [x] 修复 latest 轮询空等及超预算额外等待，补充生命周期/限频/新帧回归
- [x] 本地测试与静态检查，盒端独立实际流对照后选择最小部署方式
- [x] 验证上线预览吞吐/间隔、保留回滚并记录剩余限制

媒体入口实际是 Nginx Basic Auth，realm="ZLMediaKit administration"，代理到盒内3000；
与自带6680/webassist不可混淆。只读检查现有保护文件后，用media账户经私有CA严格校验
请求用户给出的完整HTTPS入口，返回200；账号密码仅直接告知用户，不记录到仓库、不重置认证。

调度新增Condition有界等待latest，不消费AI队列；按编码开始时刻限15Hz，超预算不额外sleep，
取最新帧且不补历史帧。stop/EOF/clear唤醒，无订阅不编码，空返回不忙循环。
15项调度回归与全量436项测试通过（7条既有依赖弃用告警），限定Ruff和diff检查通过。
同一路真实海康相邻25秒窗口、独立进程复用真实reader/preview方法、无新推理context对照：
旧调度12.012fps、间隔P95 145.975ms/最大198.883ms；新调度13.766fps、P95 107.196ms/
最大152.605ms。解码分别14.9/15.0fps，进程CPU48.72%/47.81%，发布帧龄P95
76.857ms→71.218ms。场景和JPEG大小随时间变化，该对照不是完全相同图像回放，也不是主服务已上线。
固定video SHA256 b707fc2d0742bcc9a40f7ebe21b9071793994d844cefcb4a99292c6cbbf86060，
service SHA256 6673d418f15c1a0e0852156fe17865a4ce3c5794c865382cc7b702d9bd2d52cc。
镜像overlay补充COPY硬解helper，以持久包含已验证的NV12布局修复。

电脑严格TLS+现有Basic Auth经18443入口32.413秒只统计MJPEG分段：8.494fps、3.35Mbps，
帧到达间隔P95 477.183ms、最大1595.460ms，7次>500ms。尚需同时取盒内对照，不能将异时段
观测直接断言为网络根因；不保存真实帧/JPEG，凭据仅在受控进程内存中使用。

用户明确允许中断识别并切换。通过调度端一次POST正常结束任务，返回completed且cleanup_pending=false，
确认调度与识别均空闲后，仅重建识别容器。实际发布使用当前已部署Compose副本，逐项比较
12个bind mount、13个设备映射、host网络/IPC、4-7 CPU及安全设置；运行环境只改变构建版本标记。
保护的runtime.env仅在内存读取并由匿名memfd交给Compose，不改变权限，不写明文副本/日志。
新镜像 `person-search-rk3588-rpp:preview-scheduling-20261015-v1`，
ID `sha256:551be906fa7cad108635024c5410e1aa16fab6e0fec05a1dc84d980139fd511b`；
容器healthy、restart0，readyz=ready，三个provider保持RKNN_NPU/RPP_BF16/RKNN_NPU，六项能力存在。
发布目录 `/home/EIS100RK3588/apps/person-search-hotfixes/20261015-preview-scheduling/build/deploy`。
回滚镜像 `person-search-rk3588-rpp:preview-before-20261015-nv12` 保留旧调度且包含NV12修复，
ID前缀c3b61f1cb220；避免回滚到未修stride的旧镜像再次无画面。原镜像与发布目录保留。

同窗口旧服务盒内MJPEG9.872fps、P95间隔193.394ms/最大327.359ms；电脑经18443只5.458fps，
P95 508.024ms/最大952.392ms，13次>500ms。电脑192.168.31.206/23访问17.215走utun1024/
198.18.0.1，访问盒子Wi-Fi30.184走en0。Nginx /v1 已关闭proxy buffering/request buffering且
tcp_nodelay开启，未据此乱改。证书SAN包含30.184、17.215及公网地址，已提供可严格TLS访问的
Wi-Fi入口供用户复测。独立静态资源两路径并发下载分别3.215/4.377Mbps（合计共享网络），
不得将其当各自独占带宽上限；额外客户端也会争用带宽，后续真实浏览器验收应限制订阅者数量。
主服务全量识别与浏览器新任务尚待用户重新发起后采证；不把独立预览对照误称上线真实会话改善。

上线后用户启动会话暴露500：本地HEAD的service诊断读取rpp_release_gil/
rpp_engine_cache_enabled，但旧基础镜像Settings未包含字段。此前只做import/ready验收未执行活动
view，是本次打包兼容验证遗漏；不得将该500归因摄像头或网络。逐模块AST检查列出仅这两个
缺字段，改为保守False回退，不升级SDK/后端或启用GIL优化。增加旧Settings缺两字段回归，
全量437项通过、Ruff/diff通过。实际旧镜像Settings执行_effective_config成功，真实运行镜像内
合成初始化/运行/断流/停止四态SearchSession.view和JSON序列化全通过（不创建模型/摄像头上下文）。

用户500报障在已授权修复范围内执行v2受控重建；当时调度已无活动任务，识别active视图500，
仅本次已诊断错误使用显式recover-500标记，Uvicorn正常lifespan清理内部会话，未强杀设备。
当前镜像 `person-search-rk3588-rpp:preview-scheduling-20261015-v2`，
ID `sha256:08cc98ed8e3ff5427f129bf5f39e11fc502aaa818c91c229cf2f7dc95f00aeb4`，
service SHA256 `bfd450efb2426ae1df17a56f720b7b601a742d234e4511597398c54b6456cf13`。
healthz/active已200（active=null），readyz=ready、三个provider保持原硬件。
原v1不可再用于发布，performance.env与发布脚本已指向v2，回滚镜像仍为含NV12修复的旧调度。
补充镜像内ASGI验收：不接入生产manager、不创建摄像头/模型，用合成活动会话请求healthz、
active、按ID查询和按request_id查询四条真实路由均HTTP200；线上容器healthy/restart0。
当前等待用户重新登记/发起真实任务，500恢复已验证，主服务新预览的实测及浏览器卡顿验收未完成。

## 2026-10-15 盒端卡顿续查（用户184入口仍卡）

- [x] 同窗口采样ZLM压缩帧、MPP独立硬解与主服务采集/预览的停顿分布
- [x] 核对硬解实际插件、帧PTS/缓存、NV12桥接与主进程GIL/推理等待
- [x] 用无额外网络预览负载的受控实验确定阻塞位置，修复后再验收

用户确认经Wi-Fi184:18443入口同样PPT，不能再将TUN/17入口作为主因。机器人已完成
转发解耦和持续推流窗口验证，但这不是所有间歇故障已排除；本轮重点盒端分段实测。

同窗口ZLM压缩帧15.027fps、独立NV12解码14.929fps、主服务MJPEG10.947fps；前两者最大
间隔267/276ms，未见持续超载/断流。受控合成SCRFD完整推理A/B/A表明SDK input copy
占用GIL约67–115ms；只替换stream等待没有消除停顿。独立11,304,960B合成输入传输对照：
Python绑定P50 90/107ms，native pinned P50 17.765ms，回读字节完全一致；5ms心跳最长
停顿从153/135ms降到21.783ms。未使用或保存真人帧/身份数据，不变模型参数。

- [x] 分离输入拷贝、执行与输出拷贝，确认主要GIL阻塞来自SDK输入绑定
- [x] 为显式release_gil模式接入复用pinned buffer及CDLL输入拷贝，测试ABI/失败隔离/生命周期
- [x] 实机完整模型输出一致性及心跳A/B/A验证
- [x] 构建配套Settings/backend的镜像，空闲切换并验证活动路由与真实预览


native pinned-input实现保留默认关闭，仅盒端显式启用；输入CDLL memcpy+rtMemcpy(kind=1)，
与enqueue/native stream sync同受全局RLock保护，成功同步后擦除staging，正常close按context→
stream→staging擦除/free→设备/engine/runtime释放；未知拷贝/执行/同步故障强引用隔离至进程退出。
补充64位ABI/size_t上溢/数据一致性/非法输入/关闭幂等/失败隔离/stream销毁失败顺序测试。
全量456项通过（7条既有告警），限定Ruff/diff通过。

额外诊断进程曾被驱动拒绝创建设备；主服务始终healthy/ready。空闲时停止主服务，独占验证
成功；不据此断言硬件损坏。相同SCRFD图/权重、zeros/random/ones三组纯合成输入，九输出
逐值一致(max_abs_error=0)。A/B/A每轮20次@4Hz：旧输入拷贝P50 91.244/91.000ms，
新18.773ms；完整推理P50 123.289/123.568→61.886ms；5ms心跳最大136.998/131.688→
17.163ms，>50ms停顿20/20→0。该实验尚不能代替真实活动视频的帧率/流畅度验收。

受控发布镜像person-search-rk3588-rpp:rpp-native-input-20261015-v1，ID
sha256:bb03d35f7314f95c060bbb7b375b87b8538653c5e55b927cee49508dcee65d97。
基于已上线v2只覆盖config/rpp_backends/hybrid_backends，12 bind mounts/13设备/host network和
IPC/4-7 cpuset/安全参数均逐项保持；环境仅改build revision和显式RPP_RELEASE_GIL=true。
发布配置在原hotfix目录build/deploy-rpp-input，rollback.env指向preview-scheduling-20261015-v2
且显式关闭GIL模式；不修改保护runtime.env、不保存明文凭据。原文件/镜像仍保留。
readyz确认三provider不变。用户已重启活动任务，正在完成真实预览和活动接口验收。


上线真实活动会话同窗口32秒：ZLM压缩输出14.981fps、独立NV12解码14.972fps，间隔最大
146.921/146.862ms，均0次>200ms。主服务盒内MJPEG13.937fps（448帧），间隔P95
106.091ms/最大154.023ms、0次>200ms/500ms；相对前一真实任务窗口10.947fps/最大
264.997ms有改善，但不是相同画面回放。主服务解码14.9–15.2fps，预览13.6–14.2fps，
JPEG编码P95约26ms（此前约78ms），预览帧龄P95约77ms。第二窗口实际插件枚举确认
mppvideodec，采样成功；不把decodebin工厂名误认为软件解码。

电脑经192.168.30.184:18443严格TLS（保留17.215证书主机名验证，未关闭证书验证）
32.148秒，HTTP200，441帧，13.778fps/6.823Mbps；到达间隔P50 63.727/P95
201.539ms/最大303.715ms，24次>200ms、0次>500ms。该测量增加一个HTTPS订阅者，
不能当作网络独占容量测试，也不代表浏览器渲染实测；端到端仍有短时抖动，未宣称绝不卡顿。
同窗口服务解码约15fps、预览13.4–14.4fps，实际rpp_mode=native_stream_sync。

实际新镜像内隔离ASGI检查（独立假backend/合成零值输入、不触碰生产manager/摄像头/模型）：
initializing/running/source_lost/stopping四态，healthz、active、按ID和request_id查询共16次
HTTP200，effective_config明确RPP_RELEASE_GIL=true。无额外httpx依赖安装，纯ASGI调用。
真实线上healthz/readyz/active持续正常。模型输出一致性和性能均有实证，主观浏览器观感尚待
用户查看；不把当前15fps摄像头说成可以达到30/60fps。此次改动在InsightFace，未再改Android。

## 2026-10-15 项目记忆与技术方案交付

- [x] 将本轮根因、发布/回滚与指标口径整理为可持续读取的项目记忆
- [x] 新建项目AGENTS.md并关联记忆、经验和当前技术方案
- [x] 整理当前RK3588单路海康技术方案，区分实现、现场证据与后续规划
- [x] 校对源码/配置/文档链接及敏感信息，修正README入口和过期现场描述


交付AGENTS.md、docs/project-memory.md、docs/technical-solution-20261015.md，并更新README
当前基线入口和1080p单路实测描述。项目记忆保存用户单路优先偏好、边界、四类修复、严格测试
口径、镜像与回滚索引；AGENTS要求相关后续任务先读取记忆。技术方案包含架构图、硬件分工、
数据与会话流程、确认策略、原生输入优化、预览/API/安全、发布与验收和后续规划。
文档链接/围栏/凭据模式检查及git diff --check通过；独立只读复核确认模型/API/默认与现场覆盖/
RPP生命周期描述匹配源码和Compose。此次文档验收不重复运行未变动的业务测试。


## 2026-10-15 两仓库提交与远程同步

- [x] 核对当前分支、远程与全部改动，明确机器人main/识别eis100的提交边界
- [x] 复核测试、静态检查和敏感信息，准备已授权的源码、测试与文档提交
- [x] 分别推送并核对远程提交号、上游状态与工作区

用户明确要求将swcaresRobot和insightface当前代码push到各自远程。机器人实现已经在main，
仅本轮跨仓库排查记录待提交；识别提交NV12布局、预览调度、旧Settings兼容、RPP原生输入
优化及对应测试和项目文档。不包含私有配置、模型媒体素材或签名材料。

推送前复核：456项pytest通过（7条既有依赖告警），全仓Ruff通过，diff检查通过；暂存17个文本文件，未包含私有配置、模型、媒体、包体或签名材料，新增行敏感模式检查无命中。

远程同步结果：swcaresRobot main已推送6b69e7c，insightface eis100已推送b73678a；git ls-remote核对远程分支与本地提交一致，两仓库工作区均干净、领先/落后均0。当前记录以独立文档提交同步。

## 2026-10-15 框时序与确认诊断

- [x] 补齐活动 SearchView 缺失的采样率、未确认门和确认耗时字段，并核对单/双源口径。
- [x] 将预览叠框绑定来源帧和 epoch，限制过期框，保持预览最新帧与 AI 队列独立。
- [x] 增加针对性回归，运行 pytest、Ruff、diff 检查；核对现场配置和活动状态后验证候选镜像。
- [x] 取得一次同人走过的脱敏终态：正式确认，耗时1.451s，轨迹采样4.82Hz。
- [ ] 取得异人走过的负样本及用户对实时框的主观观察；单次正样本不代表总体召回/误报验收。

验证：本地 `uv run pytest -q` 457项通过（7条既有告警），`uv run ruff check .`、
`git diff --check`、监控页 `node --check` 通过。现场旧镜像与本地基线三文件SHA256一致；
候选仅覆盖domain/service/monitor，隔离合成活动会话四态路由均HTTP200且新诊断字段可读。
用户手动结束任务后发布 `box-confirm-20261015-v1`（镜像ID
`sha256:b9a7b79c2df47dcca8304e51988108fa2ea94795b812979b3306a9f539281360`）。
现场Compose核对保留12个bind mount、13个设备映射、host network/IPC、cpuset 4-7，运行环境
只变更build revision；上线后healthy/restart=0、healthz/readyz/active/monitor均200、
三provider仍为RKNN_NPU/RPP_BF16/RKNN_NPU，近期日志无Traceback/ERROR/500。
原镜像 `rpp-native-input-20261015-v1` 与现场 `swcares-box-confirm-release.py rollback`
保留。用户重启任务后的单次真实正样本：1920×1080、decoded 648 / processed 496，
终态completed、found_count=1、confirmed_result_count=1。人脸观测28次中20次<48px，
8次通过质量门，4次相似度越过阈值，7次严格人体关联；最终3/3证据、最佳相似度0.653
（需0.55）、确认耗时1.451s、轨迹采样4.82Hz，未确认门为空。预览叠框龄P95 307.9ms，
过期跳过17帧，预览帧龄P95 72.9ms、编码P95 25.9ms。此为一次正样本，无同画面旧版
A/B，也无负样本和人工框精度检查，不能断言总体业务效果提升。
