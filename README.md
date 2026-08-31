# Robot Person Search PoC

用一张目标照片，在 RTSP 或 USB 摄像头视频中实时寻找目标人物的 PoC。系统使用行人检测与跟踪维持轨迹，只用多帧人脸证据确认身份，避免拥挤场景中单帧误认直接触发机器人动作。

## 架构

```text
RTSP / USB
  -> 最新帧队列（容量 2）
  -> YOLOX-Tiny ONNX person detector
  -> ByteTrack 高/低置信度两阶段关联
  -> InsightFace SCRFD + ArcFace
  -> 人脸与 person track 空间关联
  -> 轨迹级 candidate / confirmed / lost 状态机
  -> REST + WebSocket API
```

服务一次只运行一个搜索任务，每个任务可包含最多 20 个目标。目标照片和 embedding 只保存在进程内存，搜索停止后自动清除。RTSP URI 在 API 响应中会脱敏。

InsightFace 1.x 默认使用 128 与 640 双尺度人脸检测：128 负责近距离大脸，640 负责视频中的较小人脸。可用 `PERSON_SEARCH_FACE_DETECTION_SIZE` 强制单一尺寸，但通常不建议覆盖默认值。

1080P 下 640 那一档会把 1920 压到 640（0.333×），远处人脸到网络输入只剩十几像素，接近 SCRFD stride-8 的锚点下限。T4 实测（真实人脸贴入 1080P 画布，det_thresh 0.45）：

| 画面人脸短边 | `[128, 640]`（旧默认） | `[1280]` |
|---|---|---|
| 21 px | 未检出 | 0.69 |
| 25 px | 未检出 | 0.77 |
| 28 px | 未检出 | 0.78 |
| 35 px | 未检出 | 0.84 |
| 52 px | 0.75 | 0.82 |
| 110 px | 0.86 | 0.90 |
| 219 px | 0.91 | 0.81 |
| 311 px | 0.90 | 0.91 |
| 407 px | 0.80 | 0.80 |

单独一档 `1280` 在每个远距离尺寸上都更好，近距离也没有丢失，**而且更便宜**：T4 上 `[1280]` 一次 57 ms，`[128, 640]` 108 ms，`[128, 640, 1280]` 181 ms。原因是 CUDA 上**每切换一次 ONNX 输入形状要付约 30 ms 的重规划**，远大于单尺度自身的推理时间（1280 单档 53 ms、640 单档 18 ms、320 单档 5 ms），所以「多加一档」的代价几乎全在形状切换上，「换掉那一档」才是正确形态。生产部署因此使用 `PERSON_SEARCH_FACE_DETECTION_SIZE=1280`，`PERSON_SEARCH_FACE_DETECTION_EXTRA_SCALE_CUDA` 保留为通用机制（默认 `1280`，`0` 关闭）。同理，ROI 补检尺度量化成两档而不是逐裁剪取整——逐裁剪取整实测 233 ms/轮，量化后 136 ms/轮。若 `face_full` P95 仍超出循环能吸收的范围，用 `PERSON_SEARCH_FACE_DEEP_SCAN_EVERY_N` 降为隔轮深扫。

登记照和视频帧使用不同门控：登记照至少需要 `0.60` 检测分，并保持严格的正脸姿态要求；视频检测从 `0.45` 开始，不因 roll/yaw 直接拒绝，而是将姿态作为质量软分。视频帧仍使用更严格的运动模糊过滤，避免低质量帧成为确认依据。`PERSON_SEARCH_MAX_ABS_ROLL_DEGREES` 和 `PERSON_SEARCH_MAX_YAW_PROXY` 只约束登记照。

### 机场场景识别规则

人脸尺寸使用推理输入原始帧中的人脸框短边，不是浏览器缩放后的预览尺寸。默认按以下规则累计同一轨迹的多帧证据：

- 普通人脸（短边 `>= 80 px`）：相似度至少 `0.55`，在 `1.5 s` 内取得 `3` 帧证据后确认；确认前可以发布 `candidate` 事件。
- 小人脸（短边 `64-79 px`）：相似度至少 `0.60`，在 `2 s` 内取得 `4` 帧证据后确认；为控制误报，不发布 `candidate` 事件。
- 超小人脸（短边 `48-63 px`）：默认关闭；开启 `PERSON_SEARCH_TINY_FACE_ENABLED=true` 后，仅接受检测分 `>=0.65` 的脸，在 `3 s` 内取得 `6` 帧，至少 `5` 帧相似度 `>=0.64`，且质量加权聚合相似度 `>=0.68` 才命中。多目标时 Top1/Top2 差值还必须 `>=0.08`，不发布 `candidate`。默认 `PERSON_SEARCH_TINY_FACE_SHADOW_MODE=true`，命中只发布 `tiny_shadow_confirmed` 诊断事件，不将目标标记为已找到。`PERSON_SEARCH_TINY_FACE_MIN_PX` 只允许向上收紧，不能配置到 `48` 以下。
- 短边 `<48 px`：始终标记为 `face_too_small`，不进入身份确认。超小脸开关关闭时，`<64 px` 仍按原有逻辑拒绝。

档位边界带迟滞（`PERSON_SEARCH_FACE_TIER_HYSTERESIS_PX`，默认 `6 px`）：离开当前档必须越过边界这么多像素。换档会清空证据窗口，而移动中的机器人会让同一张脸在 `48/64/80` 三条边界上反复穿越，没有迟滞时窗口永远填不满——面板上看起来是「证据恒为 0」，实际是档位在抖。当前生效的档位随每个目标一起上报，监控页直接显示。

窗口统计量默认是**中位数**。中位数只对「不收集低分样本」的档位是恒真的（那些档位窗口里全是达标样本），真正起作用的地方是超小脸档——它按设计收集全部观测。`PERSON_SEARCH_MATCH_PROFILE=responsive` 会把超小脸档换成 `4` 帧 / `2 s` / 最佳 K 均值（K=3）/ 检测分 `0.55`，聚合门 `0.68` 两档都保留（它是最便宜的误报闸）。profile 只填写环境变量没有显式设置的字段，因此 `.env` 里的设定永远优先。**responsive 是拿误报换速度，标定完成前不要用它驱动机器人动作。**

人脸与人体轨迹按以下顺序关联，避免把拥挤画面中的脸分配给错误旅客：

1. 主规则检查人脸中心是否位于人体框的横向范围和上方 `60%` 区域；多个人体框同时满足时选择面积最小的包含框，事件中的 `association` 为 `person_strict`。
2. 坐姿、遮挡或人体框截断导致主规则失败时，只有在人脸中心被唯一一个完整人体框包含的情况下才放宽关联，记为 `person_relaxed`；若同时落入多个人体框则拒绝猜测。
3. 仍无法关联人体时，可用人脸框 IoU 建立短时兜底轨迹，轨迹 ID 为负数，记为 `face_fallback`。放宽关联和人脸兜底都使用更严格的 `0.60 / 4 帧 / 2 s` 策略——但只在该策略确实更严时才替换；`48-63 px` 超小脸的档位比它更严，不会被它放宽。超小脸**不允许**走 `face_fallback`（没有任何人体证据），但**允许** `person_relaxed`（`PERSON_SEARCH_TINY_FACE_ALLOW_RELAXED_ASSOCIATION`，默认开启）：坐姿和人体框截断正是这条路径，而它已经拒绝在多个人体框重叠时猜测归属。关掉它，「远处坐着」在结构上就不可能被确认。

移动机器人会让整个画面一起位移，纯 IoU 关联把这读成「所有轨迹都丢了」，而新的 `track_id` 会让证据窗口从零开始——这是移动中召回上不去的一个独立原因。`PERSON_SEARCH_CAMERA_MOTION_COMPENSATION`（默认开启）在降采样灰度图上用相位相关估计全局平移，在 IoU 关联前施加到轨迹框上；面板的 `相机位移 p95` 用于判断是否需要升级到仿射估计。

只有一张登记照时，`PERSON_SEARCH_EMBEDDING_FLIP_TTA`（默认开启）把每个裁剪与其水平镜像放进同一个批次、特征相加后归一化，登记侧和搜索侧走同一条实现。它抬高的是相似度分布本身，而不是降低门槛。

批量搜索会区分“完整身份竞争集”和“仍待确认目标集”：目标找到后不再累计证据，但仍保留在 Top1/Top2 身份竞争中。已找到目标继续出现在画面时，其人脸不会降级分配给剩余目标。

CUDA 环境下全帧人脸检测默认按 `10 Hz` 运行。每个缺少 `>=80 px` 合格人脸的人体轨迹都会独立进入头肩 ROI 候选，不会被画面中的无关近脸关闭；每轮最多调度 `3` 条人体轨迹的人体上方 `50%` 区域（`roi_max_tracks_per_pass=3`），T4 默认 `4 Hz`，CPU 默认关闭。`roi_batch_size` 默认是 `8`，它只表示同一检测尺度下每个批次的裁剪数，不是每轮的轨迹上限。ROI 检测尺度按裁剪大小量化成两档：不超过 `PERSON_SEARCH_ROI_FACE_DETECTION_SIZE`（默认 `320`）的裁剪仍上采样到该值，更大的裁剪用 `PERSON_SEARCH_ROI_FACE_DETECTION_MAX_SIZE`（默认 `640`）而不再被压回 320。只用两档是因为每个新的 ONNX 输入形状在 CUDA 上要付约 30 ms 重规划。ROI 只能改善检测和关键点稳定性，不会增加原始身份纹理。

### 运行预算与诊断

以下容量保护参数都可用同名 `PERSON_SEARCH_...` 环境变量覆盖，并会在
`GET /v1/searches/{search_id}` 的 `effective_config` 中回显。它们限制单帧工作的峰值，
不会改变相似度或证据门槛：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `roi_max_tracks_per_pass` | `3` | 一次 ROI pass 最多尝试的人体轨迹数 |
| `roi_batch_size` | `8` | 同一输入尺度下每个检测批次的 ROI 裁剪数；不是轨迹数上限 |
| `arcface_micro_batch_size` | `16` | 每次 ArcFace embedding 调用最多处理的人脸数；启用 flip TTA 时镜像行在批次内部加倍，但仍按人脸数切块 |
| `max_faces_per_frame` | `64` | 一帧进入 ArcFace 前保留的人脸上限；超出后按人体关联、尺寸和质量优先保留 |

建议持续采集搜索响应中的 `source_fps`、`processed_fps`、`dropped_frames`、`drop_rate`、
`roi_batch_count`、`embedding_batch_count`、`faces_dropped_by_budget` 和
`embedding_failures`。`stage_p95_latency_ms` 显示 `person`、`face_full`、`face_roi`、
`face_embed` 等阶段的 P95，`effective_hz` 显示各阶段实际调用频率；`budget_skips` 会区分
`face_roi_floor`（处理帧率已触及下限）和 `face_roi_credit`（信用桶余额不足）。
`end_to_end_p95_latency_ms` 是从帧采集到本轮处理完成的 P95，可作为 frame age（帧龄）
的聚合代理，不能用浏览器预览延迟代替。`faces_dropped_by_budget` 既包含每帧上限淘汰，
也包含嵌入容量回退时本帧放弃的脸；`embedding_failures` 则表示可恢复的嵌入调用错误。

YOLOX 的 ONNX Runtime 可选调优变量为 `PERSON_SEARCH_ORT_INTRA_OP_NUM_THREADS`、
`PERSON_SEARCH_ORT_INTER_OP_NUM_THREADS` 和 `PERSON_SEARCH_ORT_CUDA_DEVICE_ID`。留空时
保持 ORT 原生线程池和设备选择；设置时使用非负整数，设备编号对应 `nvidia-smi -L`。
部署脚本中的对应前缀是 `T4_ORT_*`，只在显式设置时传入。当前这些变量配置的是
YOLOX detector session，InsightFace `FaceAnalysis` 的 provider/session 仍需单独核验。

## 使用 uv 安装

项目固定使用 Python 3.11，并由 `uv.lock` 锁定依赖。

```bash
uv sync --extra test --extra inference-cpu
```

只运行不依赖真实模型的测试：

```bash
uv sync --extra test
uv run pytest
uv run ruff check .
```

`insightface==1.0.1` 会安装 CPU 版 ONNX Runtime，并在第一次使用 `buffalo_l` 时下载模型。离线机器人应提前将模型放到 `~/.insightface/models/buffalo_l/`，或通过 `PERSON_SEARCH_INSIGHTFACE_ROOT` 指定目录。

### NVIDIA GPU

InsightFace 自身会传递依赖 CPU 版 `onnxruntime`，而 `onnxruntime-gpu` 使用相同 Python namespace。GPU 环境不能让两者共存；先完成 CPU PoC，再在目标 CUDA/cuDNN 环境中用 uv 替换为匹配版本的 `onnxruntime-gpu`，并通过 `/v1/searches/{id}` 的 `provider` 确认实际为 `CUDAExecutionProvider`。GPU 环境应单独生成和保存经过验证的锁文件。

#### 在 NVIDIA T4 上快速测试

T4 具备 16 GB 显存，运行当前的 YOLOX-Tiny + `buffalo_l` 模型没有显存压力。机场场景建议输入 `1920x1080`，保持 CUDA 全帧人脸检测 `10 Hz`，并启用默认的 `4 Hz` 人体 ROI 补充检测（每轮最多调度 `3` 条人体轨迹；`roi_batch_size=8` 只是同一尺度的批处理分组大小）；不要在未回归误报率和 P95 延迟前继续下调最小人脸尺寸。服务器需要已安装 NVIDIA 驱动、CUDA/cuDNN 和 Python 3.11；先确认驱动可以看到 GPU：

```bash
nvidia-smi
```

在服务器上执行以下命令。这里先按项目锁文件安装基础依赖，再将 CPU runtime 替换为 GPU runtime；这是测试用流程，生产环境应为 GPU 依赖单独生成并保存锁文件。

```bash
# 在项目目录执行
uv sync --extra test --extra inference-cpu

# CPU/GPU runtime 不能共存
uv pip uninstall --python .venv/bin/python onnxruntime
# 1.23.2 是 CUDA 12.x 的示例版本；CUDA 11.x 请换成匹配服务器的版本
uv pip install --python .venv/bin/python "onnxruntime-gpu==1.23.2"

# 必须看到 CUDAExecutionProvider
.venv/bin/python -c \
  'import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'

# 下载并校验 YOLOX-Tiny
.venv/bin/person-search-download-models

# 首次运行会下载 buffalo_l，并同时预热 InsightFace
.venv/bin/python -c \
  'from person_search.backends import InsightFaceBackend; from person_search.config import Settings; b=InsightFaceBackend(Settings()); b.ensure_ready(); print("provider:", b.provider_name)'
```

启动 API：

```bash
export PERSON_SEARCH_HOST=0.0.0.0
export PERSON_SEARCH_PORT=8000
export PERSON_SEARCH_PREFER_CUDA=true
export PERSON_SEARCH_INSIGHTFACE_ROOT=/opt/models/.insightface
export PERSON_SEARCH_YOLOX_MODEL=/opt/insightface/models/yolox_tiny.onnx

# 使用 .venv 入口，避免 uv run 按 CPU 锁文件重新同步环境
.venv/bin/person-search-api
```

服务启动后可检查：

```bash
curl http://127.0.0.1:8000/healthz
```

创建搜索任务后，`GET /v1/searches/{search_id}` 返回的 `provider` 应包含 `CUDAExecutionProvider`。摄像头的 RTSP 地址还必须能从 T4 服务器直接访问。测试时不建议直接暴露 8000 端口；若只需本地浏览器访问，可以保持服务监听 `127.0.0.1`，再使用 SSH 端口转发。

#### 使用 Docker 部署 T4

仓库根目录提供了 [Dockerfile](Dockerfile)，默认通过 DaoCloud 国内代理拉取 `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`，Ubuntu APT 源和 Python 包源都切换为阿里云。镜像使用 Python 3.11（`enum.StrEnum` 需要 3.11，Ubuntu 22.04 默认只有 3.10，因此从 deadsnakes PPA 安装正式版）。服务器需要安装 NVIDIA 驱动和 NVIDIA Container Toolkit，且 `nvidia-smi` 正常工作。

部署脚本在构建前会依次探测阿里云、中科大、PyPI 官方，选第一个可用的作为 Python 包源。这一步是必要的：镜像站可能对某个 IP 限流并返回 403，而失败会发生在构建的中段，白费上面所有层。用 `T4_PIP_INDEX_URL` 可以指定固定源，跳过探测；用 `T4_PIP_INDEX_CANDIDATES` 可以自定义候选列表。

推荐在 T4 服务器直接使用部署脚本。脚本会预取 YOLOX 权重、构建带 commit 标识的镜像、验证 NVIDIA Container Toolkit，先在 loopback canary 容器中完成健康/provider 检查，再安全替换同名旧容器；旧容器会保留为 rollback point。模型目录持久化，且会使用真实 YOLOX session 验证 `CUDAExecutionProvider`：

完整的源码发布、服务器更新、容器替换、GitHub 不可达备用方案和 GPU 核验步骤见 [T4 部署与 GPU 验证](docs/t4-deployment.md)。

```bash
git clone https://github.com/qhyuTT/insightface.git
cd insightface
chmod +x scripts/deploy_t4.sh
./scripts/deploy_t4.sh
```

默认只监听服务器的 `127.0.0.1:8000`，通过本地 SSH 隧道访问监控页：

```bash
ssh -L 8000:127.0.0.1:8000 user@t4-server
```

然后打开 `http://127.0.0.1:8000/monitor`。脚本使用 host 网络，因此从 Mac 建立到 T4 的 RTSP 反向隧道后，容器可以直接读取服务器上的 `rtsp://127.0.0.1:18554/camera`。

常用覆盖参数：

```bash
# 对局域网开放监控页；同时需要配置服务器防火墙/安全组
T4_BIND_HOST=0.0.0.0 ./scripts/deploy_t4.sh

# DaoCloud 不可用时切回官方 CUDA 镜像
T4_CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  ./scripts/deploy_t4.sh

# 从国内对象存储下载 YOLOX，并在部署时预下载 buffalo_l
T4_YOLOX_MODEL_URL=https://your-cn-oss.example/yolox_tiny.onnx \
T4_PRELOAD_INSIGHTFACE=1 \
  ./scripts/deploy_t4.sh
```

`T4_PRELOAD_INSIGHTFACE=1` 会在部署阶段下载 `buffalo_l`；若服务器无法访问 InsightFace 上游，可保持默认值 `0`，并将准备好的模型目录放入 Docker volume `person-search-models`。

部署脚本默认以 shadow 模式运行超小脸确认。只有同时设置
`T4_TINY_FACE_SHADOW_MODE=false` 和 `T4_ALLOW_PHYSICAL_ACTIONS=true` 才允许该档触发正式动作。
生产部署还应设置 `T4_EXPECTED_COMMIT`（完整或短 SHA）和 `T4_EXPECTED_REMOTE_URL`；脚本默认拒绝 dirty checkout。
候选容器默认使用 `T4_CANARY_PORT=18000`，正式端口为 `T4_PORT=8000`，优雅停止超时由
`T4_STOP_TIMEOUT_SECONDS=30` 控制。

##### YOLOX 权重的获取方式

构建期下载权重是整条链路里最容易失败的一步，一旦失败还会丢掉上面所有缓存层。因此脚本默认先在宿主机上把 `models/yolox_tiny.onnx` 下好，`Dockerfile` 再从构建上下文 `COPY` 进镜像；只有上下文里没有这个文件时才回退到构建期下载。宿主机预取的好处是断点可续、可以换源重试，而且失败了不影响已经构建好的层。

预取依次尝试 `gh-proxy.com`、`ghfast.top`、`ghproxy.net` 三个 GitHub 代理，最后才是 GitHub 官方地址，共两轮。请求固定使用 HTTP/1.1，因为部分代理在 HTTP/2 下会在传输中途返回 `INTERNAL_ERROR`，导致 `curl` 以 92 退出。下载完成后校验 SHA-256，校验不通过的文件会被丢弃（通常是代理返回的错误页）。

所有源都失败时脚本只告警、不中断，继续走构建期下载。也可以手动放好权重后重跑：

```bash
# 已有权重时直接跳过下载（校验通过即复用）
cp /path/to/yolox_tiny.onnx models/yolox_tiny.onnx
./scripts/deploy_t4.sh

# 自定义代理列表；留空表示只用官方地址
T4_YOLOX_MIRRORS="https://your-proxy.example/" ./scripts/deploy_t4.sh

# 完全关闭宿主机预取，改回构建期下载
T4_PREFETCH_YOLOX=0 ./scripts/deploy_t4.sh
```

`models/yolox_tiny.onnx` 仍然不进 Git（见 [.gitignore](.gitignore)），但 [.dockerignore](.dockerignore) 为它开了 `models/*` 的例外，好让它能进入构建上下文。

构建镜像（`onnxruntime-gpu==1.23.2` 是 CUDA 12.x 示例版本）：

```bash
docker build -t person-search:t4 .
```

若 DaoCloud 代理不可用，可以切回 Docker Hub 官方镜像，或把参数替换为服务器已有的私有镜像地址：

```bash
docker build \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  -t person-search:t4 .
```

Docker 构建时会下载并校验 YOLOX-Tiny。默认地址是 GitHub release；如果服务器访问 GitHub 较慢，可先把相同模型上传到国内对象存储，然后覆盖下载地址，SHA-256 校验仍会执行：

```bash
docker build \
  --build-arg YOLOX_MODEL_URL=https://your-cn-oss.example/yolox_tiny.onnx \
  -t person-search:t4 .
```

运行容器并持久化 InsightFace 模型缓存：

```bash
docker run --rm --gpus all \
  --name person-search \
  -p 8000:8000 \
  -v person-search-models:/models \
  person-search:t4
```

启动日志中没有 provider 信息时，可在另一个终端检查容器内的 GPU runtime：

```bash
docker exec person-search python -c \
  'import onnxruntime as ort; print(ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'
```

若服务器 CUDA/cuDNN 版本与示例不匹配，应通过 `--build-arg CUDA_IMAGE=...` 选择对应的 NVIDIA CUDA 基础镜像，并通过 `--build-arg ONNXRUNTIME_GPU_VERSION=...` 选择匹配的 GPU runtime。首次登记 `buffalo_l` 仍可能需要从 InsightFace 上游下载模型；也可以把已下载的模型目录挂载到 `/models/.insightface`。

## 准备 YOLOX 模型

行人检测模型不提交到代码仓库。使用项目命令从官方 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) release 下载并校验：

```bash
uv run person-search-download-models
```

默认放置到：

```text
models/yolox_tiny.onnx
```

也可通过环境变量覆盖：

```bash
export PERSON_SEARCH_YOLOX_MODEL=/absolute/path/to/yolox_tiny.onnx
```

下载器固定校验 SHA-256 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`。如需使用自行训练的权重，导出模型的 YOLOX 临时环境同样建议使用 uv 管理：

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git /tmp/yolox-export
cd /tmp/yolox-export
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install -e .
uv run python tools/export_onnx.py -n yolox-tiny \
  -c /path/to/yolox_tiny.pth --output-name /absolute/path/to/yolox_tiny.onnx
```

## 启动 API

```bash
uv run person-search-api
```

默认只监听 `127.0.0.1:8000`，Swagger UI 位于 `http://127.0.0.1:8000/docs`。机器人需要远程访问时设置 `PERSON_SEARCH_HOST=0.0.0.0`，并在受控网络或反向代理鉴权之后使用。

### 安全边界（重要）

默认监听 loopback 是有意的安全边界。当前没有全局 API 认证：除证据下载与释放接口（配置
`PERSON_SEARCH_EVIDENCE_API_KEY` 后才受 `X-API-Key` 保护）外，目标登记、搜索创建/停止、
MJPEG 预览、WebSocket 事件、监控页和文档都可以匿名访问。不要把服务端口直接暴露到公网或
不受控局域网；需要绑定 `0.0.0.0` 时，必须放在反向代理/API gateway 或 mTLS 后，并用
防火墙/网络 ACL 限制调用方。证据 API key 只保护证据接口，不是全局凭据。

RTSP source 为支持本地隧道和摄像头网段，允许 loopback、私网及任意可解析的主机名。这也
意味着未受信调用方可能诱导服务访问内网地址（SSRF/端口探测）。只接受受信控制面提交的
`source`，并在反代鉴权之外于防火墙/网络层配置 RTSP 主机 allowlist 或出站 ACL；不要把
RTSP 用户名、密码写入仓库、命令历史或日志。API 响应中的 RTSP URI 会脱敏，但这不替代
访问控制。

可视化监控页位于 `http://127.0.0.1:8000/monitor`（根路径也会打开该页面）。页面可以完成目标照片登记、启动/停止搜索，并显示带人物框和人脸框的实时视频：普通轨迹为蓝框，候选目标为黄框，连续多帧确认后为绿框。状态、FPS、延迟、相似度和识别事件会同步更新。视频预览使用只保留最新帧的 MJPEG 流，不会因浏览器读取慢而阻塞推理。

当前摄像头已按正方向输出，服务直接使用原始视频帧推理，不额外做旋转处理。若后续更换摄像头或转码链路，先在 RTSP 客户端确认画面方向和分辨率一致，再接入识别服务。

勾选监控页的调试标注，或在接口中设置 `source.debug_preview=true`，预览会在人脸框旁显示短边像素、质量拒绝原因或 `ok`、关联方式以及相似度。白色人脸框表示通过质量门控，红色框表示被拒绝。`GET /v1/searches/{search_id}` 还返回尺寸分桶、全帧/ROI 来源、匹配阶段计数、阶段 P95/实际频率以及每个目标各自的最佳观测相似度、最近脸宽和证据进度。`evidence_eligible` 表示通过静态门控并送入确认器，`evidence_collected` 表示经过帧去重和最小时间间隔后真正写入证据窗口；监控页的“远距离诊断”区会直接展示这些数据。

RTSP 默认通过 FFmpeg 的 TCP transport 拉流，避免局域网丢包导致 H.264 花屏。只在网络可靠且更看重最低延迟时，才设置 `PERSON_SEARCH_RTSP_TRANSPORT=udp` 改回 UDP。

### 发布 Mac 摄像头 RTSP

本机摄像头通过 MediaMTX + FFmpeg 发布。先安装依赖：

```bash
brew install mediamtx ffmpeg
```

使用项目脚本在后台一键启停：

```bash
./scripts/local_rtsp.sh start
./scripts/local_rtsp.sh status
./scripts/local_rtsp.sh stop
```

脚本使用 `launchctl` 管理后台任务，重复执行 `start` 或 `stop` 是安全的。`start` 会检查依赖、端口占用、后台任务和实际视频帧；失败时自动清理本次启动的任务。`stop` 只停止脚本管理的摄像头和 MediaMTX，不会终止占用相同端口的其他程序。

其他管理命令：

```bash
./scripts/local_rtsp.sh restart
./scripts/local_rtsp.sh logs
./scripts/local_rtsp.sh logs -f
```

首次运行需要在 macOS“系统设置 → 隐私与安全性 → 摄像头”中允许 FFmpeg。脚本默认采集 `1920x1080` 并发布到 `rtsp://127.0.0.1:8554/camera`；同一局域网内的 T4 使用 Mac 的局域网 IP，例如：

服务器中的 RTSP source：

```json
{"type": "rtsp", "uri": "rtsp://192.168.31.241:8554/camera"}
```

常用参数可通过环境变量覆盖：

```bash
LOCAL_RTSP_CAMERA_DEVICE=1 \
LOCAL_RTSP_FRAME_RATE=25 \
./scripts/local_rtsp.sh restart
```

`LOCAL_RTSP_VIDEO_SIZE` 默认已经是 `1920x1080`；摄像头不支持该模式时可以显式覆盖，但应重新评估实际人脸短边是否仍能达到 `64 px`。

完整参数列表使用 `./scripts/local_rtsp.sh help` 查看。无论哪种方式，都应先从 T4 服务器测试 RTSP 能否读取到一帧。

验证 OpenCV 可以拉取一帧：

```bash
uv run python - <<'PY'
import cv2

cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/camera")
ok, frame = cap.read()
print("opened:", cap.isOpened())
print("read:", ok)
print("shape:", None if frame is None else frame.shape)
cap.release()
PY
```

FFmpeg 占用摄像头期间，搜索请求应使用 RTSP source，不能同时使用 `camera/device_index=0`。

监控页面只提供 RTSP 视频源，默认地址为 `rtsp://192.168.31.241:8554/camera`。

主要接口：

- `POST /v1/targets`：multipart 字段 `name` 和 `image`；姓名必填，照片必须恰好包含一张质量合格的人脸。姓名会出现在监控页的找到横幅和事件列表中。
- `POST /v1/searches`：启动 RTSP 或 USB 搜索。
- `POST /v1/batch-searches`：一次上传最多 20 组姓名与照片并启动异步搜索。
- `GET /v1/searches/{search_id}`：查询状态、provider、FPS、P95 延迟和累计人脸诊断计数。
- `GET /v1/searches/{search_id}/preview.mjpg`：获取后端画框后的实时 MJPEG 预览。
- `WS /v1/searches/{search_id}/events?after_seq=0`：订阅 `candidate`、`confirmed`、`lost` 和 `search_status`。
- `GET /v1/searches/{search_id}/evidence/{evidence_id}?variant=face_crop`：在命中后拉取短时有效的命中人脸裁剪；`variant=frame` 可拉取同一时刻原始帧。请求必须带 `X-API-Key`，并配置 `PERSON_SEARCH_EVIDENCE_API_KEY`。
- `DELETE /v1/searches/{search_id}`：停止任务并清除登记数据。

搜索请求示例：

```json
{
  "target_id": "returned-target-uuid",
  "source": {
    "type": "rtsp",
    "uri": "rtsp://user:password@camera/stream",
    "debug_preview": true
  }
}
```

`debug_preview` 只增加预览标注和诊断可见性，不改变匹配阈值。

目标登记示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/targets \
  -F 'name=张三' \
  -F 'image=@samples/target.jpg'
```

USB 摄像头使用：

```json
{
  "target_id": "returned-target-uuid",
  "source": {"type": "camera", "device_index": 0}
}
```

批量接口使用 `multipart/form-data`。`targets` 与 `source` 是 JSON 字符串，`images`
为重复文件字段；`image_filename` 必须与上传文件名一致：

```bash
curl -X POST http://127.0.0.1:8000/v1/batch-searches \
  -F 'targets=[{"name":"张三","image_filename":"zhangsan.jpg"},{"name":"李四","image_filename":"lisi.jpg"}]' \
  -F 'source={"type":"rtsp","uri":"rtsp://192.168.31.241:8554/camera"}' \
  -F 'images=@zhangsan.jpg' \
  -F 'images=@lisi.jpg'
```

接口立即返回 `search_id`。每个目标首次确认后发布 `target_found`，并从后续匹配名单中移除；
其他未找到目标继续搜索。全部找到后发布 `all_found` 并进入 `completed`。可选的
`timeout_seconds` 超时后会进入 `timed_out`，查询响应中的 `unfound_target_ids` 给出未找到名单。

### 命中证据交接

正式 `confirmed`（及其后续的 `target_found`）事件可能带有不透明的 `data.evidence_id`。执行器仅把
对应的 JPEG 原始帧和人脸裁剪保存在进程内存，默认最长 600 秒；证据绝不写入日志、事件 payload、磁盘或数据库。
控制面可在收到 `confirmed` 后使用其共享密钥拉取裁剪，随后自行按隐私策略加密保存，并调用同一路径的 `DELETE` 确认释放。自动完成（`completed`）时证据会保留到 TTL；显式停止、超时或失败时立即清空。`GET /v1/searches/{search_id}` 的 `confirmed_results` 可作为事件漏收时的可靠查询源，其中的 `evidence_available` 表示该条裁剪此刻是否仍可下载。

`confirmed` 事件同时带 `face_bbox`（人脸检测框）与 `bbox`（人体跟踪框），两者都是归一化坐标；裁剪取自 `face_bbox`。未配置 `PERSON_SEARCH_EVIDENCE_API_KEY` 时不会编码任何图片，事件里也不会出现 `evidence_id`。

取图与释放的失败语义各不相同，调用方必须区别对待：

| 状态 | `code` | 含义 | 是否应重试 |
| --- | --- | --- | --- |
| 404 | `evidence_not_found` | ID 未知 | 否 |
| 404 | `evidence_released` | 已被自己的 `DELETE` 释放 | 否（幂等成功） |
| 410 | `evidence_expired` | 超过 TTL | 否 |
| 410 | `evidence_discarded` | 随搜索显式停止/超时/失败被清空 | 否 |
| 403 | `invalid_evidence_api_key` | 密钥不匹配 | 否 |
| 503 | `evidence_access_not_configured` | 执行器未配置密钥 | 否 |

```bash
curl -H "X-API-Key: $PERSON_SEARCH_EVIDENCE_API_KEY" \
  'http://127.0.0.1:8000/v1/searches/SEARCH_ID/evidence/EVIDENCE_ID?variant=face_crop' \
  --output matched-face.jpg
```

## 离线视频验证

```bash
uv run person-search-eval \
  --name "张三" \
  --photo samples/target.jpg \
  --video samples/crowd.mp4 \
  --output-dir artifacts/run-001
```

输出 `annotated.mp4` 和 `report.json`。默认相似度阈值 `0.55` 只用于跑通流程，不能作为机器人动作的生产阈值。应使用真实摄像头采集目标与非目标轨迹，按“可接受的每小时误确认数”重新标定 `PERSON_SEARCH_SIMILARITY_THRESHOLD`。

批量标定兼容版本 1 manifest。远距离回归建议使用版本 2，将每个目标区间标记为 `<48`、`48-55`、`56-63`、`64-79` 或 `>=80` 像素档；空列表仍表示纯负样本：

```json
{
  "version": 2,
  "cases": [
    {
      "id": "gate-a-positive",
      "photo": "media/target.jpg",
      "video": "media/gate-a.mp4",
      "target_name": "张三",
      "expected_intervals_seconds": [
        {"start": 12.4, "end": 28.7, "face_px_bucket": "48-55"},
        {"start": 51.0, "end": 63.2, "face_px_bucket": "56-63"}
      ]
    },
    {
      "id": "gate-b-negative",
      "photo": "media/target.jpg",
      "video": "media/gate-b.mp4",
      "target_name": "张三",
      "expected_intervals_seconds": []
    }
  ]
}
```

```bash
uv run person-search-eval \
  --manifest evaluation/manifest.json \
  --thresholds 0.50 0.55 0.60 0.65 \
  --output-dir artifacts/airport-eval
```

每个 case 会生成独立的 `annotated.mp4` 和 schema v2 `report.json`，根目录报告按尺寸档汇总召回率和平均/P95 确认延迟。`aggregate` 和 `recommended_similarity_threshold` 只统计真实生产确认；Shadow 命中使用 `shadow_confirmed` 状态并单独进入 `shadow_metrics` / `shadow_aggregate`，绝不会据此生成生产阈值推荐。只有生产整体召回率至少 90%、误确认不超过每小时 0.01 次，并累计至少 100 小时非目标视频时才推荐阈值；数据不足或没有阈值通过时不会给出生产推荐。评测会复用线上的多尺度检测、深扫节奏、相机运动补偿、ROI、详细关联、尺寸分层和人脸兜底策略。

### 用分布定阈值

只数确认次数无法回答「这个距离到底能不能用」。加上 `--dump-similarities` 会额外写出 `similarities.json`（每个通过质量门的观测一行：尺寸、尺寸档、检测分、清晰度、档位、关联方式、相似度）以及报告中的 `similarity_distribution`：

```bash
uv run person-search-eval \
  --manifest evaluation/manifest.json \
  --dump-similarities \
  --output-dir artifacts/calibration
```

离线 `report.json` 的 `quality_diagnostics` 还会区分两类嵌入异常：
`embedding_failures` 是没有得到有效向量的**人脸输入数**，
`embedding_provider_failures` 是可恢复的**推理调用失败数**（一次调用可能在
OOM 拆分或重试后对应多张脸，因此两者不应相加）；`embedding_batch_count` 和
`faces_dropped_by_budget` 用来判断是否触发了微批次或单帧人脸预算。坏响应会按脸
跳过，评测会继续生成报告而不会把整段视频标成成功匹配。

区间标注不含逐脸真值，所以标签是**推导**出来的，报告里会写明这一点：落在任何 `expected_intervals_seconds` 之外的帧目标不在场，其中每张脸都算异人样本；落在区间内的帧只取该帧相似度最高的那张脸作为同人样本，其余不猜。

`tiny_face_similarity_threshold` / `tiny_face_aggregate_similarity_threshold` 应落在同人与异人两个分布的分离点，而不是落在「刚好能确认」的位置。**若两个分布在 `48-63 px` 上重叠严重，正确结论是该距离在本场景不可用、需要机器人先靠近，而不是继续降阈值。**

## 当前边界

- ArcFace 无法识别背身或不可见的人脸；当前要求上游提供方向正确的视频帧。ByteTrack 只在短时间内延续人体轨迹，人脸 IoU 兜底也只用于短时连续画面，不能替代 Person ReID。
- `48-63 px` 超小脸功能默认关闭，必须先完成尺寸分桶正负样本标定，再保持 `PERSON_SEARCH_TINY_FACE_SHADOW_MODE=true` 验证；只有验收通过后才可设为 `false` 并驱动机器人身份动作。
- `PERSON_SEARCH_MATCH_PROFILE=responsive` 是拿误报换确认速度。标定完成前只能用于观察「能不能识别到」，不能驱动机器人动作。
- 未实现 Person ReID 和活体检测，照片或屏幕重放可能命中。
- `buffalo_l` 预训练模型仅限非商业研究用途。代码可以用于内部 PoC，但产品化或客户交付前必须取得模型授权或替换成权利清晰的模型。
- 当前为单进程内存状态，只能使用一个 Uvicorn worker；多机器人扩展需要外部任务队列和状态存储。
- 当前 CPU 实测处理速度较低，监控页能实时显示最新结果，但检测框更新频率取决于硬件；机器人侧建议使用 CUDA/专用 NPU 或更轻模型。
