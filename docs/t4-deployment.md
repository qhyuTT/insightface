# T4 部署与 GPU 验证

本文说明如何将已验证的 `main` 分支更新到 NVIDIA T4 服务器，并区分“服务器有 GPU”“容器能访问 GPU”和“推理模型实际使用 CUDA”三个层次。

## 安全边界（上线前必读）

服务默认只监听 `127.0.0.1`。当前没有全局 API 认证：目标登记、搜索创建/停止、MJPEG
预览、WebSocket 事件、`/monitor` 与文档接口均可匿名访问；只有配置
`PERSON_SEARCH_EVIDENCE_API_KEY` 后，证据下载/释放接口才要求 `X-API-Key`。因此设置
`PERSON_SEARCH_HOST=0.0.0.0` 时，必须把服务放在反向代理/API gateway 或 mTLS 后，并用
防火墙/网络 ACL 限制来源；证据 key 不能当作全局认证。

RTSP 输入允许 loopback、私网和任意 DNS 主机名，以兼容本地隧道与摄像头网段。未受信请求
可借此诱导服务访问内网地址（SSRF/端口探测），所以除 API 鉴权外还应在防火墙/网络层配置
RTSP 主机 allowlist 或出站 ACL，只让受信控制面提交 source。URI 凭据不会出现在 API
响应，但仍不要写入仓库、命令历史或日志。

## 机场场景运行基线

机场中的旅客常见坐姿、行走和局部遮挡，部署时建议给识别服务提供未经降采样的 `1920x1080` 输入。人脸框短边按推理原始帧计量：`>=80 px` 使用 `0.55 / 3 帧 / 1.5 s`，`64-79 px` 使用更严格的 `0.60 / 4 帧 / 2 s` 且不发布 `candidate`，`<64 px` 直接拒绝。将 1080P 转成 720P 会把同一张脸线性缩小约三分之一，可能跨过这两个边界。

T4/CUDA 默认全帧人脸检测频率为 `10 Hz`。当全帧没有合格人脸或只有小人脸时，额外以 `4 Hz` 检查最多 `3` 条高置信人体轨迹的 ROI（`roi_max_tracks_per_pass=3`）。`roi_batch_size` 默认 `8` 只是同一检测尺度下的批处理分组大小，不是每轮的轨迹上限。上线前应在实际 RTSP 链路确认响应中的 provider、P95 延迟、`dropped_frames` 及人脸诊断计数；不要只根据浏览器预览是否流畅判断推理负载。

运行时容量保护的默认值为：`roi_max_tracks_per_pass=3`、`roi_batch_size=8`、
`arcface_micro_batch_size=16`、`max_faces_per_frame=64`。前者限制一次 ROI pass
尝试的轨迹数，第二项限制同尺度检测批次的裁剪数，第三项限制一次 ArcFace 调用的
人脸数，最后一项限制一帧进入 ArcFace 的人脸数。它们都可以用同名
`PERSON_SEARCH_...` 环境变量覆盖，并在搜索响应的 `effective_config` 中回显；超出
预算时会优先保留有明确人体关联、尺寸较大且质量较高的脸，并记录降级计数。

上线后建议持续记录以下字段，而不是只看预览：`source_fps`、`processed_fps`、
`dropped_frames`、`drop_rate`、`roi_batch_count`、`embedding_batch_count`、
`faces_dropped_by_budget`、`embedding_failures`、`stage_p95_latency_ms`、`effective_hz`
和 `budget_skips`。其中 `budget_skips` 的 `face_roi_floor` 表示处理帧率触及
`min_processed_fps`，`face_roi_credit` 表示 ROI 信用桶余额不足；两者处置方式不同。
`end_to_end_p95_latency_ms` 统计从帧采集到本轮处理完成的 P95，可作为 frame age
（帧龄）的聚合代理。将它与 `source_fps`、`processed_fps` 一起看，才能区分网络抖动、
队列积压和模型推理过慢。

离线 schema v2 报告的 `quality_diagnostics` 会进一步拆分嵌入异常：
`embedding_failures` 统计没有拿到有效向量的人脸输入数，
`embedding_provider_failures` 统计可恢复的 provider 调用失败数；OOM 拆分或重试会
让一次调用对应多张脸，因此两者不能直接相加。`embedding_batch_count` 与
`faces_dropped_by_budget` 用于确认微批次和单帧预算是否实际生效。遇到坏的 provider
响应时，离线回放和在线帧都会按脸丢弃并继续运行，相关计数应纳入验收记录。

当前摄像头已放正，服务直接使用 RTSP 原始帧进行检测和特征提取。部署前应从 T4 服务器确认 RTSP 画面方向、`1920x1080` 分辨率和实际帧率；排障时可临时设置 `debug_preview=true`，预览会显示人脸短边、拒绝原因、关联路径和相似度；搜索状态还会返回累计的接受、小脸、未关联、拒绝原因和关联方式计数。

不要把服务器地址、用户名、密码、SSH 私钥或 RTSP 凭据写入仓库。生产环境优先使用 SSH key，不在命令历史中传递密码。

## 发布源码

先在开发机验证并推送源码：

```bash
uv run pytest -q
uv run ruff check .
uv build
git push origin main
```

记录将要部署的提交：

```bash
git rev-parse HEAD
git status --short
```

只有测试通过且工作区状态符合预期时才继续部署。

## 更新服务器仓库

服务器仓库应保持在 `main` 分支且没有本地改动：

```bash
cd /path/to/insightface
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

部署前确认服务器的提交与开发机、GitHub 上的提交一致。不要在服务器使用 `git reset --hard` 覆盖未确认的本地改动。

### GitHub 不可达时

如果服务器无法访问 GitHub，可在开发机生成增量 bundle。`OLD_COMMIT` 必须是服务器当前提交：

```bash
git bundle create update.bundle main ^OLD_COMMIT
git bundle verify update.bundle
scp update.bundle user@server:/tmp/person-search-update.bundle
```

在服务器执行 fast-forward：

```bash
cd /path/to/insightface
git fetch /tmp/person-search-update.bundle main
git merge --ff-only FETCH_HEAD
rm /tmp/person-search-update.bundle
git rev-parse HEAD
git status --short
```

bundle 方式仍保留完整 Git 提交历史，不能用未经 `git bundle verify` 校验的压缩包直接覆盖仓库。

## 构建并替换容器

只允许本机通过 SSH 隧道访问时使用默认绑定：

```bash
./scripts/deploy_t4.sh
```

需要在受控局域网内直接访问时：

```bash
T4_BIND_HOST=0.0.0.0 ./scripts/deploy_t4.sh
```

脚本按照以下顺序执行：

1. 校验或下载 `models/yolox_tiny.onnx`。
2. 构建新的 `person-search:t4` 镜像，此时旧容器继续运行。
3. 用新镜像执行 `nvidia-smi`，验证 NVIDIA Container Toolkit。
4. 在独立的 loopback canary 端口（默认 `18000`）启动候选容器；旧容器仍继续提供服务。
5. 等待候选容器 `/healthz` 成功，并创建真实 YOLOX ONNX session 验证 CUDA provider（可选预加载 InsightFace）。
6. 候选通过后先停止并删除候选以释放 GPU，再以 `--stop-timeout` 停止旧容器，并将旧容器改名为带时间戳的 rollback point。
7. 使用 host 网络、`--gpus all`、`restart=unless-stopped` 和 `person-search-models` 模型卷启动新容器；再次执行 `/healthz` 和 YOLOX provider 检查。
8. 新容器任一检查失败时，脚本删除失败容器、恢复 rollback point，并重新执行健康和 provider 检查。

替换容器会清空进程内的登记目标和搜索任务，但不会删除 `person-search-models` 模型卷。通过 `T4_STOP_TIMEOUT_SECONDS`（默认 30 秒）可调整优雅停止时间；`T4_CANARY_PORT` 可调整候选端口。成功部署后旧容器会以 `person-search.previous.<timestamp>` 名称保留，确认新版本稳定后可手动删除。

部署前脚本会打印并校验 Git commit、分支、origin 和工作树状态。生产建议设置 `T4_EXPECTED_COMMIT=<完整或短 SHA>` 与 `T4_EXPECTED_REMOTE_URL=<可信 origin>`；默认拒绝 dirty checkout，临时实验才使用 `T4_ALLOW_DIRTY=1`。

## 部署后检查

确认只存在预期的应用容器，并检查健康状态：

```bash
docker ps -a --filter name=^/person-search$
docker inspect person-search --format 'Health={{.State.Health.Status}} Image={{.Image}} Started={{.State.StartedAt}}'
curl --fail http://127.0.0.1:8000/healthz
docker logs --tail 100 person-search
```

局域网部署还应从另一台机器访问：

```bash
curl --fail http://SERVER_IP:8000/healthz
curl --fail --output /dev/null --write-out '%{http_code}\n' http://SERVER_IP:8000/monitor
```

预期分别返回 `{"status":"ok"}` 和 HTTP `200`。

检查是否有替换遗留的旧镜像：

```bash
docker images
docker image ls --filter dangling=true
docker system df
```

不要直接执行 `docker system prune -a`；它可能删除其他项目的镜像和构建缓存。

## 确认 GPU 是否实际使用

以下检查必须逐层通过。只有 `nvidia-smi` 成功，并且 YOLOX 与 InsightFace 都输出 `CUDAExecutionProvider`，才能确认本项目实际使用 GPU 推理。

### 1. 宿主机识别 GPU

```bash
nvidia-smi -L
```

T4 服务器应显示类似：

```text
GPU 0: Tesla T4 (...)
```

### 2. 容器获得 GPU 设备

```bash
docker inspect person-search --format '{{json .HostConfig.DeviceRequests}}'
docker exec person-search nvidia-smi -L
```

`DeviceRequests` 应包含 GPU 请求，容器内也应看到 Tesla T4。

### 3. ONNX Runtime 提供 CUDA

```bash
docker exec person-search python -c \
  'import onnxruntime as ort; print(ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'
```

这一步只证明 CUDA provider 可用，还不能证明具体模型选择了它。

### 4. YOLOX 实际选择 CUDA

```bash
docker exec person-search python -c \
  'from person_search.config import Settings; from person_search.detector import YoloXOnnxDetector; detector=YoloXOnnxDetector(Settings()); detector.ensure_ready(); print(detector.provider_name); assert detector.provider_name == "CUDAExecutionProvider"'
```

### 5. InsightFace 实际选择 CUDA

```bash
docker exec person-search python -c \
  'from person_search.backends import InsightFaceBackend; from person_search.config import Settings; backend=InsightFaceBackend(Settings()); backend.ensure_ready(); print(backend.provider_name); assert backend.provider_name == "CUDAExecutionProvider"'
```

首次执行可能下载 `buffalo_l`。离线环境应提前把模型放入 `person-search-models` 卷对应的 `/models/.insightface/models/buffalo_l/`。

ONNX Runtime 在部分无显示设备的服务器上可能警告无法读取 `/sys/class/drm/card0/device/vendor`。只要模型 session 成功建立、最终 provider 是 `CUDAExecutionProvider`，并且命令以状态码 0 结束，该警告不表示回退到了 CPU。

运行搜索后还可以查询搜索状态：

```bash
curl http://127.0.0.1:8000/v1/searches/SEARCH_ID
```

响应中的 `provider` 应同时表明 face 和 person detector 使用 `CUDAExecutionProvider`。

YOLOX 的 ONNX Runtime 线程与设备可选通过 `PERSON_SEARCH_ORT_INTRA_OP_NUM_THREADS`、
`PERSON_SEARCH_ORT_INTER_OP_NUM_THREADS` 和 `PERSON_SEARCH_ORT_CUDA_DEVICE_ID` 设置；
留空表示采用 ORT 原生默认，值必须是非负整数，CUDA 编号以 `nvidia-smi -L` 为准。
部署脚本提供对应的 `T4_ORT_*` 变量，并且只在显式设置时传入。它们目前只作用于
YOLOX detector session；InsightFace 的 provider 仍要按上面的独立命令核验。

### 6. 超小脸验证

`48-63px` 超小脸策略在库默认值里仍为关闭，由部署侧开启：`scripts/deploy_t4.sh` 默认下发
`PERSON_SEARCH_TINY_FACE_ENABLED=true` 与 `PERSON_SEARCH_TINY_FACE_SHADOW_MODE=true`。
这是一个**先观察再标定的运营默认**——实拍显示该距离下全部人脸短边 <64px，不开该档则零召回。
Shadow 命中只发布诊断事件，不会把目标标记为已找到。只有同时显式设置
`T4_TINY_FACE_SHADOW_MODE=false` 与 `T4_ALLOW_PHYSICAL_ACTIONS=true`，部署脚本才允许
超小脸正式确认触发 `target_found`；未完成按尺寸召回率和负样本验收前不要打开该开关。

需要先只观察、不置 `found` 时，用 shadow 模式单独起容器：

```bash
docker run --rm --gpus all \
  --name person-search \
  -e PERSON_SEARCH_TINY_FACE_ENABLED=true \
  -e PERSON_SEARCH_TINY_FACE_SHADOW_MODE=true \
  -p 8000:8000 \
  -v person-search-models:/models \
  person-search:t4
```

`GET /v1/searches/{search_id}` 的 `face_size_counts`、`rejection_counts`、`face_source_counts`、`match_stage_counts`、`stage_p95_latency_ms`、`effective_hz` 和 `budget_skips` 必须持续采集（监控页 `远距离诊断` 已全部呈现）。其中 `evidence_eligible` 是送入确认器的观测数，`evidence_collected` 才是去重和时间间隔后实际入窗数；`budget_skips` 按 `face_roi_floor` / `face_roi_credit` 区分 ROI 被跳过的原因。离线 schema v2 报告将生产 `aggregate` 与 `shadow_aggregate` 分开，Shadow 通过不能生成生产阈值推荐。

**仍然欠着的验收**：按尺寸分桶召回率、至少 100 小时负样本、误确认率不高于 `0.01/h`。在补齐之前，不要让机器人基于一次 48-63px 确认采取物理动作；`tiny_face_similarity_threshold` / `tiny_face_aggregate_similarity_threshold` 也应先用 `person-search-eval` 对实拍素材标定，而不是留用默认的 `0.64` / `0.68`。低于 `48px` 的人脸仍会被硬拒绝，`PERSON_SEARCH_TINY_FACE_MIN_PX` 不能下调突破该底线。
