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

首版一次只允许一个搜索任务。目标照片和 embedding 只保存在进程内存，搜索停止后自动清除。RTSP URI 在 API 响应中会脱敏。

InsightFace 1.x 默认使用 128 与 640 双尺度人脸检测：128 负责近距离大脸，640 负责视频中的较小人脸。可用 `PERSON_SEARCH_FACE_DETECTION_SIZE` 强制单一尺寸，但通常不建议覆盖默认值。

登记照和视频帧使用不同的清晰度门控：登记照允许轻微柔焦，只要正脸、尺寸和姿态合格；视频识别帧保持更严格的运动模糊过滤，避免低质量帧成为确认依据。

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

T4 具备 16 GB 显存，运行当前的 YOLOX-Tiny + `buffalo_l` 模型没有显存压力。服务器需要已安装 NVIDIA 驱动、CUDA/cuDNN 和 Python 3.11；先确认驱动可以看到 GPU：

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

仓库根目录提供了 [Dockerfile](Dockerfile)，默认通过 DaoCloud 国内代理拉取 `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`，并将 Ubuntu APT 源切换为阿里云、Python 包源切换为清华源。服务器需要安装 NVIDIA 驱动和 NVIDIA Container Toolkit，且 `nvidia-smi` 正常工作。

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

可视化监控页位于 `http://127.0.0.1:8000/monitor`（根路径也会打开该页面）。页面可以完成目标照片登记、启动/停止搜索，并显示带人物框和人脸框的实时视频：普通轨迹为蓝框，候选目标为黄框，连续多帧确认后为绿框。状态、FPS、延迟、相似度和识别事件会同步更新。视频预览使用只保留最新帧的 MJPEG 流，不会因浏览器读取慢而阻塞推理。

### 使用 VLC 发布 Mac 摄像头 RTSP

本机内建 FaceTime 摄像头可通过项目命令发布为 `rtsp://127.0.0.1:8554/camera`：

```bash
uv run person-search-vlc-camera
```

第一次运行需要在 macOS“系统设置 → 隐私与安全性 → 摄像头”中允许 VLC。该命令以前台方式运行，按 `Ctrl+C` 停止。只查看底层 VLC 命令而不打开摄像头：

```bash
uv run person-search-vlc-camera --print-only
```

默认的 `127.0.0.1` 只对本机可见。若 T4 与 Mac 在同一局域网，使用 Mac 的局域网 IP 发布，并在服务器搜索请求中填写同一个地址：

```bash
# 将 192.168.1.20 换成 Mac 的局域网 IP
uv run person-search-vlc-camera --host 192.168.1.20
```

服务器中的 RTSP source：

```json
{"type": "rtsp", "uri": "rtsp://192.168.1.20:8554/camera"}
```

若 T4 在云服务器或无法主动访问 Mac，可在 Mac 上建立 SSH 反向端口转发。此时 VLC 仍使用默认的 `127.0.0.1`，服务器搜索请求改为转发后的端口：

```bash
# 在 Mac 上执行；18554 可换成服务器上的空闲端口
ssh -N -T -o ExitOnForwardFailure=yes \
  -R 18554:127.0.0.1:8554 user@t4-server
```

```json
{"type": "rtsp", "uri": "rtsp://127.0.0.1:18554/camera"}
```

无论哪种方式，先从 T4 服务器测试 RTSP 能否读取到一帧；直接跨公网暴露 RTSP 端口不建议用于测试之外的场景。

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

VLC 占用摄像头期间，搜索请求应使用 RTSP source，不能同时使用 `camera/device_index=0`。

主要接口：

- `POST /v1/targets`：multipart 字段 `name` 和 `image`；姓名必填，照片必须恰好包含一张质量合格的人脸。姓名会出现在监控页的找到横幅和事件列表中。
- `POST /v1/searches`：启动 RTSP 或 USB 搜索。
- `GET /v1/searches/{search_id}`：查询状态、provider、FPS 和 P95 延迟。
- `GET /v1/searches/{search_id}/preview.mjpg`：获取后端画框后的实时 MJPEG 预览。
- `WS /v1/searches/{search_id}/events?after_seq=0`：订阅 `candidate`、`confirmed`、`lost` 和 `search_status`。
- `DELETE /v1/searches/{search_id}`：停止任务并清除登记数据。

搜索请求示例：

```json
{
  "target_id": "returned-target-uuid",
  "source": {"type": "rtsp", "uri": "rtsp://user:password@camera/stream"}
}
```

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

## 离线视频验证

```bash
uv run person-search-eval \
  --name "张三" \
  --photo samples/target.jpg \
  --video samples/crowd.mp4 \
  --output-dir artifacts/run-001
```

输出 `annotated.mp4` 和 `report.json`。默认相似度阈值 `0.55` 只用于跑通流程，不能作为机器人动作的生产阈值。应使用真实摄像头采集目标与非目标轨迹，按“可接受的每小时误确认数”重新标定 `PERSON_SEARCH_SIMILARITY_THRESHOLD`。

## 当前边界

- ArcFace 无法识别背身或不可见的人脸；ByteTrack 只在短时间内延续已经确认的轨迹。
- 未实现 Person ReID 和活体检测，照片或屏幕重放可能命中。
- `buffalo_l` 预训练模型仅限非商业研究用途。代码可以用于内部 PoC，但产品化或客户交付前必须取得模型授权或替换成权利清晰的模型。
- 当前为单进程内存状态，只能使用一个 Uvicorn worker；多机器人扩展需要外部任务队列和状态存储。
- 当前 CPU 实测处理速度较低，监控页能实时显示最新结果，但检测框更新频率取决于硬件；机器人侧建议使用 CUDA/专用 NPU 或更轻模型。
