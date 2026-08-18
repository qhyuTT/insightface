# EIS100-RK3588 部署说明

这份说明针对 RK3588 的 RKNN 路线，不复用 NVIDIA T4 的 CUDA Dockerfile。
当前仓库提供的是运行时适配和交付约束；`.rknn` 模型必须在主机上用与板端
匹配的 RKNN Toolkit2 转换并验证，不能在设备启动时在线下载。

## 目标运行形态

第一阶段按单路 1080p 验收：

```text
RTSP
  -> MPP/GStreamer 硬件解码
  -> 最新帧队列
  -> RKNN YOLOX 行人检测
  -> ByteTrack
  -> RKNN SCRFD + ArcFace 适配器
  -> 多帧确认
  -> FastAPI/WebSocket
```

设备参数中的“三路 1080p 编解码”不等于当前应用已经具备三路推理能力：
`SearchManager` 目前有意限制为一个活动搜索任务。单路的延迟、温度和精度通过后，
再增加三路调度、每路独立队列/指标、NPU core 分配和过载降级策略；在此之前不能把
硬件解码路数直接当作三路人脸识别吞吐结论。

`PERSON_SEARCH_INFERENCE_BACKEND=rknn` 时，缺少 RKNN Lite、模型文件或配置的
校验和不匹配会直接报告 `model_unavailable`，禁止静默回退到 CPU。

人脸适配器可以通过 `PERSON_SEARCH_RKNN_FACE_ADAPTER=package.module:create_adapter`
注入。工厂函数接收 `Settings` 并返回实现 `ensure_ready()`/`analyze()` 的 SCRFD+
ArcFace 适配器；这样厂商输出布局变化不会污染 API 和搜索状态机。

## 迁移前置条件

在 EIS100 上确认以下版本，并将结果写入模型 manifest：

1. Linux 发行版、Python 版本和 glibc 版本。
2. `rknn-toolkit2` 转换版本与板端 `rknn-toolkit-lite2`/`librknnrt.so` 版本。
3. GStreamer 是否提供 `rtspsrc`、`h264parse`、`mppvideodec`（或厂商等价插件）。
4. 19.2TOPS 算力卡的型号、驱动和 SDK；在拿到资料前不把它计入吞吐预算。

当前项目锁定 Python 3.11；很多厂商 RKNN Lite wheel 只提供特定的 CPython ABI。
如果 EIS100 只提供 Python 3.10，需正式增加 3.10 兼容层并重新生成/验证锁文件，或由
厂商提供匹配 Python 3.11 的 ARM64 runtime，不能仅跳过 `requires-python` 检查后上线。

## 模型转换与验证

在 x86 构建机上分别转换 YOLOX、SCRFD 和 ArcFace：

- 固定输入尺寸，避免动态 shape。
- 使用真实摄像头截图作为 INT8 校准集。
- 对 YOLOX 先验证 INT8；ArcFace 先以 FP16/高精度为基线。
- 比较 ONNX 与 RKNN 的检测框 IoU、检测召回和 embedding cosine 分布。
- 转换后重新执行 `person-search-eval`，不能直接沿用 CPU 模型的 0.55 阈值。

当前 `RknnPersonDetector` 复用本项目 ONNX YOLOX-Tiny 的前后处理，因此模型契约
必须逐项一致：输入为 BGR、letterbox 填充值 114、数值范围 0..255；layout/dtype 由
`PERSON_SEARCH_RKNN_PERSON_INPUT_LAYOUT` 和 `PERSON_SEARCH_RKNN_PERSON_INPUT_DTYPE`
声明。NCHW 输入会显式传给 RKNN Lite 的 `data_format`，不能只改 tensor shape 而遗漏
运行时 layout。

输出必须是已反量化的浮点 fused raw tensor：416×416 模型为 `[1,3549,C]` 且
`C >= 6`。Rockchip model zoo 常见的三路 `[1,C,H,W]` YOLOX head 不满足这个契约，
不能直接放进当前 backend；需要在转换/导出流程中保留 fused 输出，或先实现并验证对应
的三 head 后处理。若板端返回 int8/uint8 原始输出，还必须依据转换报告中的 scale 和
zero-point 反量化，当前代码会明确拒绝，避免把量化整数当作坐标和置信度静默解码。

运行时模型目录至少应包含：

```text
/opt/person-search/models/
  yolox_tiny.rknn
  scrfd.rknn
  arcface.rknn
  manifest.json
```

每个文件都必须有 SHA-256。运行时会在初始化 person 模型以及创建 face adapter 前校验
配置的 SHA-256；空 checksum 只用于本地集成阶段，不符合生产交付要求。可从
[models/rk3588-manifest.example.json](../models/rk3588-manifest.example.json) 复制 manifest 模板。

## 视频采集

优先使用类似下面的 GStreamer 管线（插件名称以设备镜像实际情况为准）：

```text
rtspsrc location=... protocols=tcp latency=100 !
rtph264depay ! h264parse ! mppvideodec !
videoconvert ! video/x-raw,format=BGR !
appsink drop=true max-buffers=1 sync=false
```

默认按 H.264 + `mppvideodec` 生成管线。若 EIS100 镜像使用 H.265 或不同的厂商
插件，分别设置 `PERSON_SEARCH_GSTREAMER_RTSP_CODEC=h265` 和
`PERSON_SEARCH_GSTREAMER_DECODER=<实际插件名>`；延迟可通过
`PERSON_SEARCH_GSTREAMER_LATENCY_MS` 调整。部署前先确认 OpenCV 本身启用了
GStreamer（PyPI 通用 wheel 通常不适合承担这项硬件集成）：

```bash
gst-inspect-1.0 rtspsrc
gst-inspect-1.0 h264parse
gst-inspect-1.0 mppvideodec
python -c 'import cv2; print(cv2.getBuildInformation())' | grep -i gstreamer
```

若设备没有 MPP 插件，先使用 OpenCV fallback，但应记录 CPU 占用和帧年龄；
不要把软件解码结果直接当成三路能力的证明。

## 离线交付

- 预置 ARM64 wheelhouse、RKNN runtime、模型和 manifest。
- 禁止首次启动访问 GitHub 或 InsightFace 上游。
- 使用 systemd/容器健康检查监控 `/healthz` 与 `/readyz`。
- 日志轮转放在 SSD，避免持续写入板载存储。
- 商业交付前确认 `buffalo_l` 的授权；必要时替换为权利清晰的识别模型。

## systemd 与资源边界

仓库提供了 [`deploy/systemd/person-search.service`](../deploy/systemd/person-search.service)
基线单元。它按 `/opt/person-search/app`、`/opt/person-search/venv` 和
`/etc/person-search/person-search.env` 布局运行，默认设置：

- `Restart=on-failure`，停止超时 20 秒；
- `MemoryHigh=12G`、`MemoryMax=14G`，为 16G 设备上的系统、硬解和厂商服务留余量；
- 最大 256 个 task 和 4096 个文件描述符；
- 只读系统目录、私有临时目录和 `NoNewPrivileges`；
- OpenCV 解码上传图片前最多接受 2000 万像素，避免小压缩包触发超大内存分配。
- multipart 临时文件写入 `/run/person-search`（tmpfs），避免高频登记磨损板载存储。

安装时不要假设 NPU 设备节点一定属于 `video` 组。先用 `ls -l /dev/rknpu* /dev/dri/*`
确认 EIS100 镜像实际权限，再把 `person-search` 用户加入 `video`、`render` 或厂商组。
若厂商 runtime 需要额外可写目录，用 systemd drop-in 增加精确的 `ReadWritePaths=`，不要
直接去掉整个文件系统保护。若系统不是 cgroup v2，先确认 `MemoryHigh`/`MemoryMax` 是否生效：

```bash
systemctl show person-search -p MemoryCurrent -p MemoryHigh -p MemoryMax -p TasksCurrent
```

启用示例：

```bash
sudo install -m 0644 deploy/systemd/person-search.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now person-search
```

## 网络与 API 安全

服务没有内建用户系统，生产配置应继续监听 `127.0.0.1`，由带 TLS、鉴权和请求速率限制
的反向代理暴露；不要直接把 8000 端口开放到办公网或公网。`/monitor`、MJPEG 和 WebSocket
都包含人脸或实时视频信息，也必须位于同一鉴权边界内。

反向代理还应限制并发登记和请求体大小。单张图片上限是 10 MiB，批量接口最多 20 张；
若确实开放满规格批量上传，请把代理上限设在约 210 MiB 并配合低并发/速率限制。实际
业务批次更小时应继续收紧，避免 multipart 在进入 FastAPI handler 前占满临时空间。

RTSP 地址由 API 客户端提供，属于服务端网络访问入口。必须设置
`PERSON_SEARCH_RTSP_ALLOWED_HOSTS`，值是逗号分隔的精确主机名、IP 或网段，例如：

```dotenv
PERSON_SEARCH_RTSP_ALLOWED_HOSTS=camera-01.local,192.168.31.0/24
```

运行时不会为了校验去解析 DNS；主机名规则按 URI 中的主机精确匹配，网段规则只匹配
IP 字面量。空值保留 PoC 的开放行为，不适合可被其他终端访问的部署。另可通过
`PERSON_SEARCH_MAX_ENROLLED_TARGETS` 限制尚未使用的内存 embedding 数量，盒子模板默认 20。

## 健康检查、日志和告警

`/healthz` 只判断 API 进程/事件循环存活，适合高频 liveness；`/readyz` 会实际初始化并
检查人脸与行人后端，适合启动验收和低频 readiness，失败返回 503。不要把一次 RTSP
断流当成进程死亡反复重启服务，断流由任务内重连状态机处理。

```bash
curl -fsS --max-time 2 http://127.0.0.1:8000/healthz
curl -fsS --max-time 60 http://127.0.0.1:8000/readyz
journalctl -u person-search --since '10 min ago' --no-pager
```

日志保留在 journald/SSD，建议限制 `SystemMaxUse` 和 `MaxRetentionSec`，不要持续写板载
eMMC。至少对下列信号告警：服务连续重启、`/readyz` 失败、内存接近 `MemoryHigh`、
`p95_frame_age_ms` 持续超过 500 ms、`source_reconnects` 增长以及设备温度/降频。温度和
CPU 频率可从 `/sys/class/thermal` 与 `/sys/devices/system/cpu` 采集；NPU 指标以厂商驱动
实际暴露的接口为准。

## 验收建议

- 单路 1080p：处理帧年龄 P95 小于 500 ms，丢帧受控，连续运行 24 小时内内存不持续增长。
- 识别：召回率至少 90%，误确认不超过 0.1 次/小时，负样本至少 10 小时。
- 断流后在配置的重连窗口内恢复，且旧 tracker 不得跨源复用。
- 启动时明确显示实际 provider、模型版本、量化方式和 runtime 版本。
