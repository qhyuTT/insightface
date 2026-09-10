# EIS100 RK3588 原生部署

这条部署路径不依赖 Docker。盒子是 ARM64 Ubuntu 20.04，系统已经提供
Rockchip MPP/GStreamer 和 `mppvideodec`，因此应用使用项目自己的 Python 3.11 环境，
视频采集则委托给 `/usr/bin/python3` + 系统 OpenCV/GStreamer helper；YOLOX、SCRFD、
ArcFace 通过 `librknnrt.so` 调 RK3588 NPU。

## 盒端已经确认的条件

- 架构：`aarch64`，8 个 Cortex-A55，约 15 GiB 内存。
- NPU：`rockchip,rk3588-rknpu`，实际可用 DRM 节点为 `/dev/dri/renderD129`；不应假设一定存在 `/dev/rknpu0`。
- 视频：系统 OpenCV 带 GStreamer，已安装 `mppvideodec`、FFmpeg 和 Rockchip MPP。
- 现有 `0.0.0.0:8000` 已被其他服务监听，所以本服务默认使用 `18080`，不停止或覆盖旧服务。
- 盒端现有的 `algorithmShow` / `MediaServer` 不在本项目切换范围内。

盒端现有 `librknnrt.so` 的字符串版本为 `1.5.2`，现有 fire RKNN 模型记录的编译器版本为
`2.3.2` 且可以被该运行库初始化。新模型仍必须在这台盒子上执行实际 `rknn_init`/推理
smoke test；如果出现 `model version not match`，应换用与盒端运行库匹配的 Toolkit2，或由
厂商提供兼容运行库，不能直接覆盖系统库。

## 第一步：准备转换环境

RKNN Toolkit2 只用于把 ONNX 转成 RKNN，不需要安装到盒端运行环境。先在有对应
Toolkit2 wheel 的 Linux/aarch64 或厂商支持的转换环境执行：

```bash
export RKNN_TOOLKIT_WHEEL=/path/to/rknn_toolkit2-*.whl
./scripts/prepare_rknn_toolkit.sh
source .rknn-venv/bin/activate
```

转换前准备三份源模型：

```text
models/yolox_tiny.onnx
~/.insightface/models/buffalo_l/det_10g.onnx
~/.insightface/models/buffalo_l/w600k_r50.onnx
```

默认生成 FP RKNN 模型，便于先做 ONNX/RKNN 结果对齐：

```bash
./.rknn-venv/bin/python scripts/convert_rk3588_models.py \
  --yolox models/yolox_tiny.onnx \
  --insightface-root ~/.insightface \
  --output-dir models/rk3588
```

只有准备了代表性图片目录/列表并完成精度回归后，才使用 `--quantize --dataset`。
转换结果应包含：

```text
models/rk3588/yolox_tiny.rknn
models/rk3588/scrfd_10g_640.rknn
models/rk3588/w600k_r50.rknn
models/rk3588/manifest.json
```

`manifest.json` 记录源模型和输出模型 SHA-256，部署时应保留。RKNN 人脸检测模型是
静态 `640x640`，ArcFace 是 `112x112`；静态 NPU 模型不能像 ONNX SCRFD 那样由请求
临时切换多尺度，因此 RK3588 配置把全帧和登记都固定在 640。

## 第二步：上传到盒端

先把 `librknnrt.so` 放入部署目录。盒端现有运行库可作为临时验证来源：

```text
/home/EIS100RK3588/project/EdgeAlgoFrame/lib/RK/librknn_api/RK3588/librknnrt.so
```

不把这个二进制提交到 Git；通过部署脚本的 `RK3588_RUNTIME_LIBRARY` 上传。
模型目录通过 `RK3588_MODEL_DIR` 上传。脚本默认只同步源码，不使用 `rsync --delete`，
不会删除 `/ssd/person-search` 中已有内容：

```bash
RK3588_MODEL_DIR=models/rk3588 \
RK3588_RUNTIME_LIBRARY=/path/to/librknnrt.so \
./scripts/deploy_rk3588.sh
```

如果转换机上没有该 ARM64 运行库，也可以让部署脚本从盒端现有项目复制一份到本项目目录：

```bash
RK3588_MODEL_DIR=models/rk3588 \
RK3588_REMOTE_RUNTIME_SOURCE=/home/EIS100RK3588/project/EdgeAlgoFrame/lib/RK/librknn_api/RK3588/librknnrt.so \
./scripts/deploy_rk3588.sh
```

脚本要求当前分支是 `eis100` 且工作区干净；它会在 `/ssd/person-search/.venv` 用
Python 3.11 创建环境并执行 `uv sync --no-dev`。盒端目前没有 `uv` 或 Python 3.11，
可以让脚本仅在当前用户目录引导安装 uv 管理的 CPython 3.11（不改系统 Python）：

```bash
RK3588_BOOTSTRAP_HOST=1 ./scripts/deploy_rk3588.sh
```

也可以预先执行 `scripts/prepare_rk3588_host.sh`，或者通过
`RK3588_PYTHON=/absolute/path/to/python3.11` 指定现有 Python。引导步骤需要盒端能访问
`astral.sh`；若网络受限，应在另一台机器准备好 Python/uv 后再指定绝对路径。

然后将 `deploy/rk3588.env.example` 复制为盒端的
`/ssd/person-search/.env`，填入实际模型/运行库路径和 RTSP 参数。不要把摄像头账号密码
写入 Git、命令历史或服务日志。

## 第三步：启动和端口

部署脚本安装一个用户级 systemd unit，不需要 Docker 或 sudo：

```bash
RK3588_START_SERVICE=1 ./scripts/deploy_rk3588.sh
```

它只会启动 `person-search-rk3588.service`，不会触碰盒端已有的 8000、554、1935 等服务。
远端检查：

```bash
ssh EIS100RK3588@192.168.11.203 \
  'systemctl --user --no-pager status person-search-rk3588.service'
ssh EIS100RK3588@192.168.11.203 \
  'curl --fail http://127.0.0.1:18080/healthz'
```

服务绑定 `0.0.0.0:18080` 后，局域网客户端可访问：

```bash
curl http://192.168.11.203:18080/healthz
curl http://192.168.11.203:18080/readyz
```

若盒端启用了 UFW 或上游交换机 ACL，需要由有权限的管理员只放行可信测试机到
`18080/tcp`；本项目不会自动修改防火墙，也不会为了腾出 8000 而停掉现有服务。API 当前
不是全局鉴权服务，直接绑定局域网前应使用网络 ACL、反向代理或临时 SSH 隧道。

## 第四步：逐层验证

先验证运行库和模型 ready，再接 RTSP：

```bash
ssh EIS100RK3588@192.168.11.203 \
  'cd /ssd/person-search && .venv/bin/python -c \
  "from person_search.config import Settings; from person_search.service import SearchManager; \
  s=Settings(); print(s.inference_backend, s.video_backend); print(SearchManager(s).ensure_ready())"'
```

预期 provider 为 `RKNN_NPU`，`/readyz` 应返回成功。若报权限问题，检查当前用户是否能
访问 `/dev/dri/renderD129`；不要创建假的 `/dev/rknpu0`。必要时由管理员把运行用户加入
设备对应的 `render`/`video` 组并重新登录。

然后用真实 RTSP 启动搜索，观察：

- `provider`、`frame_width`/`frame_height`、`source_fps`、`processed_fps`、`drop_rate`；
- `stage_p95_latency_ms` 中 `person`、`face_full`、`face_embed`；
- `face_size_counts`、`rejection_counts`、`association_counts`、`budget_skips`；
- 首次登记是否有有效人脸和 512 维 embedding，而不是 `no_face`。

当前没有真实 RTSP 地址和登记照片，因此部署完成前只能验证模型加载、输入链路和健康检查，
不能提前声称现场识别准确率。收到素材后先做同距离正/负样本回放，再决定是否关闭
`PERSON_SEARCH_TINY_FACE_SHADOW_MODE` 或调整相似度阈值。
