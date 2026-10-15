# 项目记忆

最后整理：2026-10-15。此文件保存可交接的现场事实；不是实时监控。操作前重新检查镜像、
地址、任务状态与运行配置。工作规则见 [AGENTS.md](../AGENTS.md)，完整方案见
[RK3588 技术方案](technical-solution-20261015.md)。

## 用户使用方式与项目边界

- 常态只开一路海康，优先解决这一路画质、推流与网页预览；机器人自带 720p 不作为主验收流。
- Android 推流属于 `~/WorkSpace/swcaresRobot`，盒端识别属于本仓库；调度/反向代理是另一项目。
- 机器人此前已完成读取/发送解耦、队列与写入优化，并安装 release、推送对应代码。
  一次约 187 秒窗口源约 1.997 Mbps、发布约 2.041 Mbps，队列未持续积压；这不是所有间歇
  故障均已排除的保证。用户主动结束任务后的 EOF 不能当成推流异常。
- 本轮后续修复在本仓库，未再次修改 Android。本仓库的提交/推送状态应通过
  `git status`、`git log` 与远程分支核对，不能由镜像已部署推断代码已推送。

## 当前视频与硬件

- 2026-10-15 海康当前主码流 H.265 / 1920×1080 / 15 FPS / CBR 约 2048 kbps。
  `Streaming/Channels/101` 表示主码流，不能据编号断言 4K；`102` 为子码流，当时为低清画面。
- 盒端 EIS100 RK3588；实测 GStreamer 工厂包含 `mppvideodec`，输出 NV12。
- 人脸检测：RPP BF16，SCRFD 搜索 1280×736、登记 640×640。
- 行人检测：YOLOX-Tiny RKNN，640×384；人脸识别：ArcFace RKNN，112×112、512 维，flip TTA。
- 主服务 Python 3.11；GStreamer helper 使用系统 Python，不能混用依赖/ABI。
- RPP 和 RKNN 是不同加速资源；MPP 解码也独立于它们。RGA 颜色转换允许经校验回退 OpenCV，
  不把安装状态当实际启用状态。历史 4K/双路报告只代表当时任务。

## 2026-10-15 已定位并处理的问题

### NV12 布局导致无画面

1080p MPP 输出实际 stride=2304，UV offset=2488320，总缓冲 3732480 bytes，与紧凑 NV12 的
3110400 bytes 不同。旧 helper 假定紧凑布局，导致 source_lost/无画面。现按逐帧
GstVideoMeta 的两平面 stride/offset 提取有效行，并检查边界/重叠；无 meta 时才用 caps 信息。

### 预览线程额外等待

latest 使用有界 Condition 等待新 frame_id，不消费 AI 队列；按编码开始时刻限频，超预算不额外
sleep、不补历史帧，无订阅不编码，stop/EOF/clear 唤醒。该修复已进入当前镜像。

### 覆盖旧镜像后活动接口 500

最新 service 的诊断读取旧 Settings 不存在的 `rpp_release_gil` / `rpp_engine_cache_enabled`。
import 和 readyz 未暴露问题，真实任务 view 才报错。已做保守字段回退并补旧 Settings 回归；
当前 RPP 发布同时包含 config/rpp_backends/hybrid_backends，已验证实际镜像活动路由。

### RPP 输入拷贝持 GIL，阻塞采集和预览

相同图、相同权重的受控 A/B/A：原 Python 输入拷贝中位约 91 ms；仅替换执行等待不能修复。
复用 pinned staging，经 ctypes.CDLL 调用 memcpy、rtMemcpy(H2D=1)，配合 enqueue 和原生
stream sync 后，输入拷贝约 18.8 ms；完整推理中位约 123 → 61.9 ms；5ms 心跳最大停顿
约 137 → 17.2 ms，20 次推理中的 >50ms 停顿从 20 次降到 0。

zeros/random/ones 三组纯合成输入，九个输出均逐值一致，max_abs_error=0。原生操作仍受全局锁
保护，成功后擦除 staging，生命周期未知的失败保留整个 owner，不释放或复用 DMA 缓冲。
开关 `PERSON_SEARCH_RPP_RELEASE_GIL` 在代码/模板默认 false，本次现场显式 true。

## 真实任务验证与结论边界

| 项目 | 修复前窗口 | 修复后窗口 |
|---|---:|---:|
| ZLM 压缩输出 | 15.027 FPS | 14.981 FPS |
| 独立 NV12 解码 | 14.929 FPS | 14.972 FPS |
| 盒内 MJPEG 发布 | 10.947 FPS | 13.937 FPS |
| 盒内 MJPEG 最长间隔 | 264.997 ms | 154.023 ms |

前后是不同真实任务窗口，不是相同画面回放。修复后盒内 32 秒无 >200ms 间隔；JPEG 编码 P95
约 78 → 26ms，预览帧龄 P95 约 77ms。采集/预览在 AI 工作量变化时仍维持约 15/14 FPS。

电脑直连 `.30.184:18443` 严格 TLS 测量 32.148 秒：441 帧、13.778 FPS、6.823 Mbps，
间隔 P95 201.539ms、最大 303.715ms，24 次 >200ms，0 次 >500ms。这是增加一个 HTTPS
订阅者的网络接收测试，不是浏览器渲染或真实摄像头到屏幕的光学延迟测量。

已通过全量 456 项测试、限定 Ruff 与 diff 检查；实际镜像隔离合成会话四态 × 四路由共 16 次
HTTP200。最终线上 healthy/restart=0，healthz/readyz/active 均 200，近期日志无 traceback/500。
不能据此声称双路/4K/长时稳定性或业务误报漏报验收通过。

## 2026-10-15 框时序与确认诊断候选

活动单路快照为1920×1080/约15 FPS，处理约12.8 FPS，人体/全帧人脸检测各约3.8–3.9Hz。
当时累计1416次人脸观测中1087次短边小于48px；这是该会话累计分布，不能归因到某一次
目标走过。预览把最新视频帧与上一轮AI框异步叠加，可能造成可见框滞后；原始检测框坐标
是否偏移仍待同帧验证。内部 `SearchMetrics` 已算轨迹采样率、未确认门和确认耗时，但
`SearchView` 缺字段会丢弃这些数值，监控页因此无法有效区分卡点。

用户结束活动任务后已发布 `box-confirm-20261015-v1`，镜像ID为
`sha256:b9a7b79c2df47dcca8304e51988108fa2ea94795b812979b3306a9f539281360`。
仅覆盖 `domain.py`、`service.py`、`monitor.html`：补齐诊断字段，预览框绑定
`source_epoch/frame_id/captured_at`，超400ms或跨epoch不继续画旧框；采集最新帧和AI队列
仍独立。单/双源诊断口径已回归。现场使用原Compose验证挂载、设备、host network/IPC、
cpuset和运行环境，只有build revision变化。上线后healthy/restart=0、healthz/readyz/
active/monitor均200，provider仍为RKNN_NPU/RPP_BF16/RKNN_NPU，近期无Traceback/ERROR/500。
旧镜像 `rpp-native-input-20261015-v1` 与现场
`swcares-box-confirm-release.py rollback` 保留。真实走过结果与未决验证见下；
仍需人工观察实时框，并补异人负样本。

上线后第一例用户重启的单路真实任务已正式确认：1920×1080，decoded 648 / processed 496，
终态 completed、found_count=1；该轨迹3/3证据、最佳相似度0.653（普通档需0.55）、
实际采样4.82Hz，首次证据至确认1.451s。全任务28次人脸观测中20次<48px、8次质量通过，
不能把这些累计值当成目标走过期间的逐帧分布。预览叠框龄P95 307.9ms，因过期跳过17帧；
预览帧龄P95 72.9ms、编码P95 25.9ms。单例达到1.5s目标，但未与旧版同画面对照，
缺异人负样本和人工框精度观察，不能宣称总体召回、误报或原始框精度已改善。

## 运行位置、发布与回滚快照

- 盒子地址：`192.168.30.184`（Wi-Fi）与 `192.168.17.215`；HTTPS 外部入口端口 18443。
- 主识别容器：`person-search-rk3588-rpp`，服务 18080；媒体容器：`zlmediakit`。
- 当前镜像：`person-search-rk3588-rpp:rpp-native-input-20261015-v1`。
- 镜像 ID：`sha256:bb03d35f7314f95c060bbb7b375b87b8538653c5e55b927cee49508dcee65d97`。
- 原生输入修复的直接回滚镜像：`person-search-rk3588-rpp:preview-scheduling-20261015-v2`；
  回滚配置显式关闭 GIL 模式。更早含 NV12 修复的 `preview-before-20261015-nv12` 也保留；
  不使用缺 NV12 修复的旧镜像，也不使用活动 view 曾报 500 的 preview-scheduling v1。
- 发布目录：`/home/EIS100RK3588/apps/person-search-hotfixes/20261015-preview-scheduling/`。
  当前 Compose/环境在 `build/deploy-rpp-input/`，含 performance.env 与 rollback.env。
- 目录中的 `swcares-rpp-input-release.py verify|up|rollback` 为本次现场脚本，不是仓库通用工具；
  会保留挂载/设备/CPU/安全设置并拒绝活动会话切换。再次执行前检查脚本与目标镜像。
- 12 个 bind mount、13 个设备映射，host network/IPC，cpuset 4-7；生产保护配置为
  `/ssd/person-search-rpp/runtime.env`。受控发布仅通过匿名 memfd 向 Compose 提供其内容，
  没有放宽文件权限或记录明文。
- RPP native 对象借用 logger/图权重，必须保留 owner。BF16 plan 反序列化曾原生崩溃，
  `PERSON_SEARCH_RPP_ENGINE_CACHE_ENABLED=false` 保持不变。
- 额外诊断进程曾报无可用 RPP 设备，而主服务正常；空闲停主服务后的独占验证通过。
  不能把该现象直接判为硬件损坏，更不能擅自重启驱动/盒子。

## 后续排查入口

- 先核实用户当前相机参数与单/双源，再按同窗口分段测量；`.184` 已复现过卡顿，TUN 不能继续
  当作已证实主因。不要开启多路测速/预览人为争用带宽。
- JPEG 带宽本次约 6.8 Mbps，高于原 H.265 约 2 Mbps；需要时再评估压缩视频预览，尚未实现。
- `/mediaserver/` 是 Nginx Basic Auth 的管理入口，和 ZLM API secret 是两层不同认证；
  账号密码不记录到记忆、文档和日志。
- 安全验收保留：真人图像/视频仅内存处理，不保存成排查素材；TLS/SSH 主机校验严格；
  不降低识别门槛冒充性能改善。项目记忆不包含任何密码或密钥。
