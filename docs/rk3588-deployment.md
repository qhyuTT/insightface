# EIS100 RK3588 原生部署

本文是 RKNN-only 原生部署路径，不是 2026-10-15 的线上部署。当前线上使用
RPP BF16 + RKNN 的 `box-confirm-20261015-v1` 容器；现状、单路实测和回滚见
[当前技术方案](technical-solution-20261015.md) 与 [项目记忆](project-memory.md)。

这条部署路径不依赖 Docker。盒子是 ARM64 Ubuntu 20.04，系统已经提供
Rockchip MPP/GStreamer 和 `mppvideodec`，因此应用使用项目自己的 Python 3.11 环境，
视频采集则委托给 `/usr/bin/python3` + 系统 GStreamer helper；4K H.265 保留 MPP 的
NV12 输出，由应用进程转换为 BGR，避免系统 OpenCV 的 4K `videoconvert` 瓶颈。YOLOX、
SCRFD、ArcFace 通过 `librknnrt.so` 调 RK3588 NPU。

## 盒端已经确认的条件

- 架构：`aarch64`，4×Cortex-A76（cpu4-7，2.0GHz）+ 4×Cortex-A55（cpu0-3，1.7GHz），约 15 GiB
  内存。CPU 侧前后处理在小核上慢 3–7 倍，systemd unit 用 `CPUAffinity=4-7` 钉在大核。
- RGA：用 udev 将 `/dev/rga` 持久设为 `root:render 0660`，并把运行账号加入 `render` 组；
  不使用 `0666`。服务的 `gstreamer_nv12_converter=auto` 会在第一帧校验颜色并实测 RGA/OpenCV，
  只保留更快路径，RGA 加载、权限或转换失败均自动回退 OpenCV。
- NPU：`rockchip,rk3588-rknpu`，实际可用 DRM 节点为 `/dev/dri/renderD129`；不应假设一定存在 `/dev/rknpu0`。
- 视频：系统 OpenCV/GStreamer，已安装 `mppvideodec`、FFmpeg、Rockchip MPP 和 Python GI。
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

RTSP 主码流建议使用以下采集参数：`TCP`、`500ms` jitterbuffer、
`PERSON_SEARCH_GSTREAMER_DROP_ON_LATENCY=false`、
`PERSON_SEARCH_GSTREAMER_OUTPUT_FORMAT=nv12`。`drop-on-latency=true` 在当前盒端的
H.265 MPP 解码链上可能产生绿帧；NV12 桥接则绕开 4K `videoconvert`，BGR 仍作为应用内部
推理接口，不影响 YOLOX/SCRFD/ArcFace。

## 性能基线（2026-09-14，盒端实测）

相机为 H.265 3840×2160 @ 20fps。NPU 侧 FP16 模型单次推理（单核 / 三核掩码）：

| 模型 | 单核 | `RKNN_CORE_MASK=7` |
|---|---|---|
| YOLOX-Tiny 416² | 52 ms | 49 ms |
| SCRFD-10G 640² | 87 ms | 74 ms |
| ArcFace r50 112²（每张脸 ×2，flip TTA） | 24 ms | 15.7 ms |

CPU 侧每帧（4K）：大核合计约 25–30 ms，小核 80–125 ms；NV12→BGR 4K 在大核 19 ms。
采集链 `mppvideodec ! videoconvert ! BGR` 只有 1.18 fps（4K 软件色彩转换），
`mppvideodec ! NV12 ! appsink` 为 19.5 fps，因此默认 NV12 桥接。

由此得出的 unit 默认值（见 `deploy/person-search-rk3588.service` 内注释）：
`CPUAffinity=4-7`、`RKNN_CORE_MASK=7`、`CAMERA_MOTION_COMPENSATION=false`（固定相机，
`camera_motion_px_p95` 4.7px）、`TARGET_LOOP_HZ=4`（这是 ROI 信用预算分母，不是循环限速）、
`STAGGER_FULL_FRAME_DETECTION=true`、人体/人脸全帧目标频率均为 4Hz、`MATCH_PROFILE=transit`、
`ROI_MAX_TRACKS_PER_PASS=3`、`FACE_TRACK_BUFFER_SECONDS=2.5`。静态 RKNN 图忽略 `FACE_DETECTION_SIZE`，
面板 `effective_config.face_detection_static_input` 显示的才是真实输入尺寸。

### 10 FPS 调度与 RGA 复测（2026-09-15）

4K 端到端独立基准为 YOLOX 108.2ms、SCRFD 65.0ms；两者同帧串行平均 163.0ms，
仅完整检测的上限就是 6.13fps，尚未包含 ArcFace。因此提高 `TARGET_LOOP_HZ` 不会提速；旧调度
在低帧率时让两个阶段每帧同时逾期，形成永久锁相。RK3588 部署现在将两个全帧阶段错峰，使用
绝对 deadline 保持各自约 4Hz，同一帧最多运行其中一个；`processed_fps` 包含轻量跟踪帧，验收仍
必须同时看 `effective_hz.person`、`effective_hz.face_full`、`effective_hz.face_embed` 和证据采样率。

当前 `librga 2.2` 的虚拟地址 4K NV12→BGR 实测约 24–27ms，OpenCV NEON 路径约 18–26ms，
结果最大像素误差为 1，二者没有稳定胜负。因此 `auto` 首帧各跑一次，并要求 RGA 至少快 5%
才选用；这条用户态复制链上不能假定“用了 RGA 就一定更快”。真正的零拷贝需要传递 MPP DMA
buffer，不属于当前 stdout 帧协议的范围。

### 16:9 静态输入与 INT8（2026-09-14 晚，盒上转换与实测）

640² 对 16:9 帧有 44% 黑边，4K→640 是 6 倍降采样，76px 的脸进网络只有 12.7px；坐姿远距离
实测最高相似度 0.537，卡在远脸档 0.64。在盒子 `.rknn-venv`（toolkit2 2.3.2，源 ONNX 在
`models/input/`）上转了以下候选，校准集为该相机 120 帧（`models/calib/`），留出 40 帧与
ONNX 对比（三核掩码，NPU 单次推理）：

| 模型 | 延迟 | 留出帧检出 | 与 ONNX 一致性 |
|---|---|---|---|
| SCRFD 640² FP16（旧） | 71 ms | 38 脸 | 逐元素一致 |
| SCRFD 960×544 INT8 | 25.5 ms | 51 脸 | 51/51 匹配，分数差 0.007，关键点 0.42px |
| **SCRFD 1280×736 INT8（现用）** | **48 ms** | **53 脸** | 53/53 匹配，分数差 0.009，关键点 0.38px |
| SCRFD 1280×736 FP16 | 168 ms | 53 脸 | 逐元素一致 |
| YOLOX 416² FP16（旧） | 48 ms | 82 人 | 逐元素一致 |
| **YOLOX 640×384 FP16（现用）** | **78 ms** | **91 人** | 91/91 匹配 |
| YOLOX 640×384 INT8 | 28 ms | 87 人 | 83/91 匹配，分数差 0.03 —— 弃用 |

ArcFace 保持 FP16：INT8 嵌入会整体平移相似度并使阈值失效。复现命令（在盒上）：

```bash
.rknn-venv/bin/python scripts/convert_rk3588_models.py --only face --scrfd-size 1280x736 \
  --quantize --scrfd-dataset models/calib/lists/scrfd_1280x736.txt --suffix _int8 \
  --yolox models/input/models/yolox_tiny.onnx --insightface-root models/input/insightface \
  --output-dir models/rk3588/candidates/scrfd_1280x736_int8
.rknn-venv/bin/python scripts/convert_rk3588_models.py --only yolox --yolox-size 640x384 \
  --yolox models/input/models/yolox_tiny.onnx --insightface-root models/input/insightface \
  --output-dir models/rk3588/candidates/yolox_640x384_fp16
```

登记照走 `PERSON_SEARCH_RKNN_ENROLLMENT_FACE_DETECTION_MODEL`（现指回 FP16 640²）：INT8 大图模型是按
远景校准的，对特写登记照给出的检测分低于 0.6 的登记门槛（`detection_score_low`），而登记门槛本身
不应因此放宽——这是 CLAUDE.md 里"登记与搜索尺度分离"在 RKNN 侧的落地。

校准图必须先按运行时的 letterbox 规则生成（`_preprocess_scrfd_rgb` / `_preprocess_yolox`），
YOLOX 校准需 `quant_img_RGB2BGR=True`（脚本已处理）。换 YOLOX 尺寸时 unit 里的
`PERSON_INPUT_WIDTH/HEIGHT` 必须同步，`ensure_ready` 会校验并拒绝不匹配的模型。

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
- `preview_fps`、`preview_frame_age_p95_ms` 和 `preview_encode_p95_latency_ms`；
- `stage_p95_latency_ms` 中 `person`、`face_full`、`face_embed`；
- `effective_config.face_detection_static_input` / `rknn_core_mask_applied`：静态图真实尺寸与三核掩码是否生效；
- `effective_config.gstreamer_nv12_converter_requested/active`：RGA 请求值与首帧实测选中的转换器；
- `unconfirmed_gate_counts`、`achieved_sampling_hz` 对 `required_sampling_hz`：卡在
  `insufficient_samples` 要减负载，卡在 `window_statistic_low` 说明输入分辨率不够；
- 采集 helper 退出时服务日志会带最后几行 stderr（已脱敏），`source_epoch` 增长时先看它；
- `face_size_counts`、`rejection_counts`、`association_counts`、`budget_skips`；
- 首次登记是否有有效人脸和 512 维 embedding，而不是 `no_face`。

当前没有真实 RTSP 地址和登记照片，因此部署完成前只能验证模型加载、输入链路和健康检查，
不能提前声称现场识别准确率。收到素材后先做同距离正/负样本回放，再决定是否关闭
`PERSON_SEARCH_TINY_FACE_SHADOW_MODE` 或调整相似度阈值。
