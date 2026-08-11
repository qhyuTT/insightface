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
