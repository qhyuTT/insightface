# 一个寻人会话接入两路视频

`POST /v1/searches` 兼容原来的单路 `source`，双路时改用 `sources`（恰好两个不同的视频源）：

```json
{
  "target_id": "已登记的目标 ID",
  "sources": [
    {"type": "rtsp", "uri": "rtsp://camera-one/live"},
    {"type": "rtsp", "uri": "rtsp://camera-two/live"}
  ],
  "timeout_seconds": 300,
  "request_id": "本次任务的幂等键"
}
```

这仍只占用一个全局寻人会话。每路有独立采集、断线重连、人体/人脸跟踪与确认窗口，共用目标特征及加载的模型。任一路确认目标后，停止另一条流水线并结束整个会话；一条源断开时另一条继续工作。当前双路模式只支持一个目标，单路批量寻人不受影响。

`GET /v1/searches/{search_id}` 返回原有 `source`（第一路，向后兼容），新增 `sources` 数组，包含每路的 `source_id`（`source-1`/`source-2`）、脱敏来源、状态、处理 FPS、源 FPS 与画面尺寸。`processed_fps`、`source_fps`、`dropped_frames` 为两路汇总；其他旧诊断字段仍以第一路为主，分析第二路时看 `sources`。`target_found` 事件和 `confirmed_results` 带 `source_id`，用于区分命中摄像机；预览地址分别为：

```text
/v1/searches/{search_id}/preview.mjpg?source_id=source-1
/v1/searches/{search_id}/preview.mjpg?source_id=source-2
```

网页 `/monitor` 的第二个 RTSP 输入框留空时保持单路行为，填入后展示两路预览。RTSP 凭据不会出现在搜索状态响应中；不要把含凭据的完整 URL 写进日志、文档或工单。

建议先用两路摄像机子码流（例如 1280×720）验收实时吞吐，再检查远处小脸的召回。降低输入视频分辨率能降低解码和预处理开销，但模型静态输入尺寸并未改变，检测推理耗时不会等比例下降。未经两路真实 RTSP 的现场测试，不应把单图基准外推为两路端到端 FPS，也不应据此切换生产服务。
