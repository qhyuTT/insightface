# 实时寻人 PoC 实施清单

- [x] 使用 uv 建立 Python 工程、锁文件、配置、领域模型和依赖边界
- [x] 实现 InsightFace 登记/识别、YOLOX 行人检测和 ByteTrack 适配
- [x] 实现视频采集、轨迹级多帧确认和搜索任务生命周期
- [x] 实现 FastAPI REST/WebSocket 接口与离线评测 CLI
- [x] 添加单元/集成测试、示例配置和使用文档
- [x] 运行测试与静态检查，记录实际模型验证限制
- [x] 增加 uv 管理的 VLC 摄像头 RTSP 启动命令并完成本机拉流验证

## 可视化监控页

- [x] 为搜索会话增加带人物框/匹配状态的实时预览帧
- [x] 增加 MJPEG 预览接口和监控页 REST/WS 集成
- [x] 补充接口测试、运行完整测试并在本机启动验证

## T4 Docker 部署

- [x] 添加基于 CUDA/cuDNN runtime 的 GPU Dockerfile
- [x] 配置国内 CUDA、APT 和 PyPI 镜像源及构建参数
- [x] 添加镜像构建、GPU 启动、provider 验证和模型缓存文档
- [x] 添加可重复执行的 T4 构建、启动和 CUDA provider 验证脚本

## 目标姓名

- [x] 登记接口同时接收姓名和照片，并在目标/事件数据中保留姓名
- [x] 监控页显示姓名并补充姓名字段校验测试

## 1080p 远距离人脸优化

- [x] 实现 48–63px 超小脸安全分层和轨迹级多帧聚合
- [x] 将 ROI 补检改为逐人体轨迹触发，并修正真实 provider/推理频率诊断
- [x] 在查询接口和监控页暴露尺寸分桶、阶段漏斗和证据进度
- [x] 让离线评测复用线上小脸/ROI/关联策略，支持按人脸像素分桶
- [x] 补充边界、多帧、ROI、provider 和评测回归测试
- [x] 更新配置/部署文档，运行完整测试与 Ruff

验证记录：两张现场图在 CPU Provider 下回归，近图主脸为 145px；远图主脸为 62px、检测分 0.8393、与近图登记脸相似度 0.7628，并严格关联到人体轨迹。26–31px 背景脸保持硬拒绝。尚未具备 T4 和 100 小时真实负样本，因此生产开关保持默认关闭，启用后仍默认 shadow。

## 超小脸改动的 review 修复

- [x] 只有达标观测才刷新 `last_similarity` / `last_face_seen`，修复预览显示低分相似度与 shadow 轨迹永不发出 `lost`
- [x] 用 `requires_strict_association` / `shadow_eligible` 替代 `collect_all_observations` 的身份代理语义（service 与 cli 同步）
- [x] Top1/Top2 差值不足时把 `identity_margin_low` 同时写回次优目标
- [x] 补充 3 项回归测试并验证回滚修复后必定失败

复盘：前两项同源——`service.py` 的 `similarity >= threshold` 入口门移进 `TrackConfirmation.process` 时，赋值语句留在了 `continue` 之前。低于阈值的观测仍按设计计入超小脸证据窗口，但不再被当作一次“看见”，因此普通档与小脸档行为与改动前完全等价，只有超小脸档的上报和 grace 计时被修正。验收标准 `MAX_FALSE_CONFIRMATIONS_PER_HOUR=0.01` 与 `MIN_NEGATIVE_EXPOSURE_HOURS=100` 经确认全局保留，它只影响“是否给出生产阈值推荐”，不影响评测跑数。测试 98 项全过、Ruff 通过；三项新测试均已确认在回滚对应修复后失败。

## 第二轮远距安全 review 修复

- [x] 将全批次身份竞争集与仍待确认目标集分离，阻止已找到目标的人脸误确认剩余目标
- [x] 固化不可下调的 48px 搜索安全底线，并在配置、质量门和消费端防御
- [x] 按目标分别维护最佳观测，并校准 `evidence_eligible` / `evidence_collected` 语义
- [x] 处理 `tiny_shadow_lost` 前端状态，离线报告隔离生产与 Shadow 指标
- [x] 补充回归测试、更新文档并完成全量验证

验证记录：118 项测试、Ruff、差异完整性和监控页脚本语法检查全部通过。现场两图 CPU 回归保持 145px/62px 主脸、相似度 0.7628、严格人体关联；26/30/31px 背景脸继续硬拒绝。T4 与 100 小时负样本验收仍需在部署环境完成，生产 tiny 开关保持默认关闭。
