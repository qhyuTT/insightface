# 双路视频最小优化实施与验证

本轮根据排查报告实施 P0 与可验证的 RGA 兼容修复。保留模型文件、输入尺寸、识别阈值、TTA、全画面召回与多人 ROI 容量。没有安装驱动、升级 SDK、切换推理设备或调整机器人原始编码参数。

## 实施内容

- `video.py` 持续读取/转换新帧，独立更新预览 latest；AI队列满时覆盖旧任务，不再用 grab 抑制预览更新。AI取帧有界排空至最新包，每个淘汰包单独计数；保持frame_id/epoch与重连后清理。读失败采用1/2/4…秒退避，成功读帧才重置，停止信号可中断等待。
- 采集计数来自成功BGR读取；AI循环、JPEG发布和阶段完成率用10秒窗口。每源完整暴露采集耗时、最后帧年龄、队列、阶段P50/P95/max/样本数；双路只聚合有意义计数/速率，延迟在每源查看。断流实时速率衰减归零，终态归零但累计采集数保留。旧字段仍可读取，`source_fps`现在明确是BGR采集率。
- 页面标注JPEG发布FPS与“解码后→发布/AI完成”，不冒充传感器或浏览器FPS、完整端到端延迟。
- RK3588 Compose新增可调`PERSON_SEARCH_PREVIEW_HZ`，默认15；普通平台Settings默认5不变。JPEG仍最大宽960、quality82、INTER_AREA，推理使用原分辨率。盒端纯微基准未证明INTER_LINEAR在4K整体更快，因此不改预览插值。
- RGA对Y<16的输入复制并将亮度截到16，UV与原NV12不变，匹配现有OpenCV颜色语义；保留误差阈值3、启动测速、自动CPU回退。不是强制采用RGA。
- 机器人控制器根据真实FORWARDING/PUBLISHING状态判断幂等；FAILED按当前实例identity清理，不会误删新会话。新增脱敏首故障side/type/elapsed日志与主动停止标记。没有新增WSS事件或自动重连；机器人修改需安装新包后才生效，不直接操作实体机器人。

## 盒端受控前后对照

使用当前识别容器同一CPU affinity与OpenCV，在内存生成4K NV12@20Hz、720P NV12@15Hz，两路AI工作线程各模拟200ms耗时（sleep），运行相同JPEG预览路径。每项12秒，同一reader策略仅切换新旧源码、预览5/15Hz。没有拉机器人视频或执行神经模型，也没有新增推理context。数据摘要与源文件SHA256在 `dual-stream-optimization-20261014.evidence.json`。

| 方案 | 4K JPEG FPS | 720P JPEG FPS | 4K发布帧龄P95 | 720P发布帧龄P95 | 进程CPU |
|---|---:|---:|---:|---:|---:|
| 原reader / 5Hz | 4.91 | 4.91 | 125.40ms | 89.22ms | 127.51% |
| 新reader / 5Hz | 4.92 | 4.92 | 82.58ms | 67.90ms | 161.11% |
| 新reader / 15Hz | 14.48 | 14.75 | 69.00ms | 78.36ms | 183.70% |

新reader的4K/720P成功读取分别240/180帧，无因AI队列满而跳过的BGR读取；AI完成率仍约4.9FPS，预览约提升3倍。AI队列有界，最终清理为0。**CPU增加是持续提供新预览帧的成本**，该数据仅证明盒端采集策略/慢AI/JPEG独立性，不证明实际MPP吞吐、真实模型竞争、浏览器呈现或机器人端到端延迟。原drop计数与新计数定义不同，不能用“新drop更多”判断退化。

可复跑：

```bash
# 使用不同reader文件分别跑相同命令；只产生合成输入，无模型和网络
python scripts/benchmark_preview_pipeline.py --seconds 12 --preview-hz 15
python scripts/benchmark_preview_pipeline.py --reader-file /path/to/old/video.py --seconds 12 --preview-hz 5
```

## RGA正确性与性能

盒端头文件核对ABI/flags/stride无误。原实现遇到Y<16时与OpenCV最大差20；修复后遍历Y=0..255、5组UV及合法/全范围随机样本，最大差1、原buffer没有修改。6次热调用包含检测低亮度、复制、截断与转换：

| 分辨率/场景 | OpenCV P50 | RGA P50 |
|---|---:|---:|
| 720P正常 | 2.29ms | 2.80ms |
| 720P暗场 | 2.64ms | 4.32ms |
| 4K正常 | 18.48ms | 16.21ms |
| 4K暗场 | 18.50ms | 25.38ms |

因此暗场自动选择CPU合理，颜色修复不等于所有场景加速。实际摄像头仍需读取每源converter_active/reason。

## 验证与发布状态

- 本地识别仓库全量380项测试通过；新回归覆盖慢AI不停止latest、不同帧计数守恒、取最新、断流退避/epoch、实时速率归零、阶段窗口/合法零时间原点、每源完整指标与终态累计保留。Ruff、JavaScript语法、diff检查通过。
- 机器人最终固定源码，单Gradle进程运行全量189项JVM测试、lintRobotDebug、assembleRobotDebug全部通过；视频修改无lint告警，5条既有告警在构建/副屏布局。新增controller测试包含运行后失败/同参数重试、观察器延迟、替换/stop/close竞态与单源失败隔离。实体断流/安装/物理行为未测试。
- 候选ARM64镜像 `person-search-rk3588-rpp:perfpreview-20261014-v1` 已在盒子构建，ID前缀`97741258b7e8`；仅覆盖video/domain/service/rga/monitor五个已审查文件，使用现有镜像/依赖，构建无网络访问。新容器无网络、无设备运行的import/编译检查通过，五文件SHA256与本地一致。
- 候选目录 `/home/EIS100RK3588/insightface-performance-20261014-v1`，Compose project=`deploy`，`deploy/performance.env`只含非敏感发布参数；继续读取现有root保护的`/ssd/person-search-rpp/runtime.env`，不复制或公开密钥、不改变权限。Compose配置已用sudo只读校验通过。
- 当前旧镜像 `person-search-rk3588-rpp:streamcontract-20261014` 保留作回滚。**用户明确批准后已切换生产**：新容器running/healthy、restart count=0，readyz就绪，六项能力契约通过，provider保持RPP_BF16/RKNN_NPU/RKNN_NPU；旧失联任务随受控重建结束，当前需重新登记/启动真实双路验收。

受控切换命令（会重建识别容器，其他容器不重建）：

```bash
sudo docker compose --project-name deploy \
  --env-file /home/EIS100RK3588/insightface-performance-20261014-v1/deploy/performance.env \
  -f /home/EIS100RK3588/insightface-performance-20261014-v1/deploy/person-search-rk3588-rpp.compose.yaml \
  up -d --no-build person-search
```

回滚沿用原目录Compose/原.env，先核对其image tag指向streamcontract-20261014再执行up --no-build；保留原镜像、代码、runtime.env和model挂载。不要用新目录的15Hz变量回滚后误称原配置已恢复。

发布后验证：readyz/healthz六项能力、三个provider及embedding contract一致；真实两路重新登记/寻人，在5/10/15Hz梯度读每源BGR采集、JPEG、face_full Hz、frame_age、queue、converter、CPU/RSS/温度和异常。人工延长AI验证预览持续；断开并恢复单源验证epoch与确认可靠性。沿用报告光学法测完整E2E，不能拿盒内帧龄代替。

目前**没有保证每路8～12Hz全图检测、浏览器15～25FPS或E2E<500ms**的现场证据。检测配置仍4Hz，H.265安全jitter仍500ms；后续调频/降低jitter必须用真实双路/弱网/小脸精度验证，不能盲改制造FPS。机器人253秒发布EOF的首因仍待新故障日志/厂家与网络采证。
