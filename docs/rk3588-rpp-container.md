# RK3588 RPP + RKNN hybrid container

Current deployed snapshot (2026-10-15):
`person-search-rk3588-rpp:box-confirm-20261015-v1`. The production runtime
source matches `eis100/fba1b6f` across 26 tracked application files and two
runtime scripts (capture helper and health contract). The
previous `rpp-native-input-20261015-v1` image is retained for rollback. This
document's 2026-10-11 staging results are historical; use the
[current technical solution](technical-solution-20261015.md) and
[deployment snapshot](project-memory.md) for current behavior and verification.
The live bridge still copies packed NV12 over stdout and converts it to BGR in
the service. It does not use a dma-buf-to-memfd model-service input path.

The performance-oriented route assigns SCRFD face detection (search
`1280x736`, enrollment `640x640`) to the RPP accelerator, and assigns both
ArcFace face recognition and YOLOX person detection to the RK3588 RKNN NPU.
This is workload routing, not a fused accelerator or a promise that the two
TOPS ratings add linearly. Neural-network inference has no CPU fallback.

CPU still performs ordinary image preparation/postprocessing (resize, crop,
alignment, and NMS). Dynamic SCRFD graphs are specialized once before
deployment with ONNX Runtime's CPU graph optimizer; the container does not
run that optimizer or execute neural-network inference on CPU.

This split follows the measured path: ArcFace on RPP was slower end-to-end
than the existing RKNN path once host/device transfers and flip-TTA were
included. RPP remains responsible for SCRFD, where its execution path is useful;
ArcFace stays on the NPU for throughput and embedding compatibility.

The container targets the Ubuntu 20.04 ARM64 runtime on `.184`. It carries the
API's CPython 3.11 environment, but delegates video capture to the host's
Python 3.8/GStreamer bridge and mounts the host's MPP/GStreamer, RGA, RPP SDK,
and RKNN runtime. The RPP SDK's `include/kernel` resources are mounted
read-only because engine execution copies its kernel configuration from there.
The image includes ARM64 `gcc`/`ld` and C headers because this SDK compiles
device-kernel objects on first execution; this is not CPU model inference.
The host's GStreamer plugin directory is used as the only
plugin system path, avoiding a second, version-mismatched plugin stack in the
container. The RPP SDK's host-linked `libjsoncpp` is also mounted read-only.
The RKNN NPU uses `/dev/dri/card1` on this box, so that DRM node is mapped along
with its render nodes. Keep those host libraries, plugins, and device nodes
intact. Hardware MPP decoding also needs the host DMA-heap character devices
under `/dev/dma_heap` (system, system-dma32, uncached, and CMA variants); the
Compose file maps each node explicitly so Docker's device cgroup permits the
decoder to allocate frames. Mapping `/dev/mpp_service` alone is not sufficient.
The MPP library also reads `/proc/device-tree/compatible` when opening the
RK3588 H.265 decoder. Docker's masked `/sys/firmware` makes that symlink point
to a missing directory in an ordinary container. Compose binds the host device
tree read-only and sets `systempaths=unconfined`; both settings are required.
Without them H.264 can still decode while H.265 fails with `failed to init mpp
ctx`, leaving that source at 0 FPS. Validate by decoding a real H.265 frame
inside the container, not by checking ZLMediaKit's publisher status alone.
The capture bridge applies `rtsp_read_timeout_seconds` (5 seconds by default)
to the complete frame read from the GStreamer helper. A stalled helper is
terminated so `LatestFrameReader` can report `source_lost` and reopen the RTSP
source with exponential backoff (capped by `rtsp_reconnect_max_seconds`). A
healthy service or publisher listing alone does not prove that decoded frames
and recognition are progressing; check per-source processed FPS in the monitor.
The dispatch backend requires six `/healthz` capabilities: `replace_active`,
`active_search`, `search_timeout`, `request_lookup`, `event_replay`, and
`confirmed_evidence_v1`. The last one is advertised only when
`PERSON_SEARCH_EVIDENCE_API_KEY` is set. Put that variable in the protected
runtime env file with the same value as dispatch's
`DISPATCH_INSIGHTFACE_EVIDENCE_API_KEY`; do not put either value in source
control or logs. Run `scripts/sync_rk3588_evidence_key.py --apply` as root on
the box to copy the already configured dispatch key without printing it; the
script backs up an existing runtime file, creates it if absent, preserves its
owner, and sets mode `0600`. The Compose health check verifies `/readyz` and the complete
capability set, so a missing conditional capability leaves the container
unhealthy with the missing capability named in its health log.
The host `rpp_server` uses SysV shared-memory IDs from the client
process, so the container must use `ipc: host` as configured in Compose.

## Model and embedding contract

- Search SCRFD: `det_10g.onnx`, pre-specialized to `1280x736` and executed by
  RPP BF16.
- Enrollment SCRFD: the same source model pre-specialized to `640x640` and
  executed by RPP BF16.
- Face recognition: existing `w600k_r50.rknn` on RKNN NPU, using the
  existing three-core mask configuration.
- Person detection: existing `yolox_tiny_640x384.rknn` on RKNN.
- Face embeddings keep the existing `arcface-v1-rknn` contract and model file;
  no re-enrollment is required when its checksum and TTA setting match the
  current service.

Prepare static SCRFD graphs on the development/build machine, then copy the
resulting files to the staging model directory mounted read-only into the
container:

```bash
uv run --extra inference-rpp --extra rpp-model-tools -- \
  python scripts/prepare_rpp_scrfd.py \
  --source-model ~/.insightface/models/buffalo_l/det_10g.onnx \
  --output-dir models/rpp-static
```

This is graph specialization only, not model inference. The service verifies
the source-weight hash, graph shapes and outputs, then passes those static
graphs to RPP. Copy the generated files (without deleting any existing
staging data) to `$PERSON_SEARCH_RPP_ROOT/models/rpp-static` on `.184`. If the
SCRFD weights or input dimensions change, regenerate the artifacts before
starting the container.

The current RPP Python SDK's `FP16` and `BF16` builder flags alias the same
enum value. The backend therefore labels and reports the actual plan as BF16;
do not infer precision from the enum name alone.

## Stage and deployment ports

The Compose file defaults to port `18080`. For an isolated side-by-side stage,
set `PERSON_SEARCH_PORT=18081` in the staging `.env`; keep the legacy native
code and systemd unit as a rollback reference. As of 2026-10-11, the native
`person-search-rk3588.service` was stopped and disabled by explicit operator
request, and this container became the service on `18080`. Nothing under
`/ssd/person-search` or the original systemd unit was deleted.

```bash
cp deploy/rk3588-rpp.env.example /ssd/person-search-rpp-staging/.env
# Edit only the staging path/port/tag values; keep camera/API secrets out of Git.
set -a
. /ssd/person-search-rpp-staging/.env
set +a
docker compose -f deploy/person-search-rk3588-rpp.compose.yaml config --quiet
docker compose -f deploy/person-search-rk3588-rpp.compose.yaml up -d --build
```

The `.env` file is only Compose configuration. The camera's existing runtime
environment may be supplied through `PERSON_SEARCH_RUNTIME_ENV_FILE`; do not
put secret values in command arguments or logs. Compose's explicit deployment
settings take precedence over values in that runtime file, including the
backend and port. If Docker Hub is unreachable from the box, set
`PERSON_SEARCH_BASE_IMAGE` to an accessible ARM64 Ubuntu 20.04 mirror; it is a
build argument and does not alter the runtime image contents.

## Performance acceptance

For an isolated stage, check `http://127.0.0.1:18081/readyz`; for the active
deployment, use `http://127.0.0.1:18080/readyz`. Require all of the following:

- person detection reports `RKNN_NPU`;
- face detection reports `RPP_BF16` and face recognition reports `RKNN_NPU`;
- search/enrollment static shapes are `1280x736` and `640x640`;
- the recognition contract remains `arcface-v1-rknn` and matches the active
  RKNN model checksum and TTA setting;
- the container health check is healthy and the RKNN/RPP device nodes are
  visible;
- the GStreamer helper can open the same RTSP source and deliver valid frames;
- same-input SCRFD output parity and end-to-end latency meet the application's
  existing acceptance thresholds, including RKNN embedding and flip-TTA costs.

RPP engines are built on startup, so first readiness can take several minutes.
This EIS100 SDK produces BF16 plan files that abort the process with a native
`Invalid DataType` on deserialization. Plan caching is therefore disabled by
default (`PERSON_SEARCH_RPP_ENGINE_CACHE_ENABLED=false`) until a specific SDK
version passes a build/serialize/deserialize/execute round trip. Existing plan
files are ignored, not deleted. Building a fresh engine does not create a
`pyrt.Runtime`; this SDK crashed while destroying an otherwise-unused Runtime.
The operator-requested port cutover does not mean performance acceptance has
passed. Until API, two-camera, accuracy, and latency checks pass, report this
deployment as the selected optimization baseline, not as a validated
performance improvement over the legacy RKNN-only service.

## Staging status (2026-10-11)

The first staging attempt lacked shared host IPC. The host `rpp_server` log
reported that it could not map the client's shared-memory IDs; the RPP kernel
driver then faulted while pinning user pages. Subsequent client processes
blocked while creating an RPP execution context, including a native-host
process. The driver retained references after the server was stopped and a
normal `rmmod rpp` was rejected. The isolated diagnostic program also needed
to retain its RPP Logger while using the Runtime; the serving backend now
retains the logger and plan bytes for the engine's lifetime.

The host was rebooted with authorization, restoring its RPP driver and server;
the original `18080` service and the dispatch/media containers recovered. On
the first post-reboot staging run, cached plans aborted at deserialization.
Fresh engine construction then exposed a missing container mount for
`/usr/local/rpp/include/kernel` and missing `gcc`/`ld`; the SDK failed when it
could not compile its device kernels. These findings are addressed in the current code and
Compose configuration. Both RPP input sizes execute and `18081` has reported
`ready` with detection=`RPP_BF16`, recognition/person=`RKNN_NPU`. Its embedding
contract matches `18080`.

Performance acceptance **has not passed**. On the same 512x512 astronaut test
image, 12 hot `detect_faces` calls per backend (including image preparation and
postprocessing) measured:

| SCRFD graph | RKNN P50 | RPP P50 | RPP result |
| --- | ---: | ---: | --- |
| Search 1280x736 | 40.93 ms | 95.36 ms | Same face, box within ~1 px |
| Enrollment 640x640 | 85.69 ms | 100.09 ms | Same face, box within ~1 px |

On that image upscaled to 3840x2160, a 12-repeat pair of YOLOX person detection
plus SCRFD face detection took 169.80 ms P50 serial / 131.45 ms P50 parallel
with both on RKNN, versus 241.95 ms serial / 183.27 ms parallel with face
detection on RPP. Parallelism helps each route but does not make this BF16 RPP
path faster than RKNN. The upscaled single image is a latency probe, not an
accuracy or live-camera throughput guarantee. At the time of these measurements,
they did not support a performance-driven cutover or a claim that the card's
theoretical TOPS improved the application. The operator later explicitly
selected this container as the optimization baseline; that operational
cutover is not a performance acceptance result.

The final staging image is `person-search-rk3588-rpp:perf-20261011-rppfix4`.
It remained healthy with zero automatic restarts after a manual stop/start and
another two-graph readiness check. A separate RPP benchmark process exited
normally after resource cleanup. The native `18080` service and the dispatch,
database, and media containers remained healthy. At that stage, no live RTSP
source had yet been validated, so camera and end-to-end search acceptance
remained open.

The subsequent Mac-camera RTSP check found that the container lacked the host
`/dev/dma_heap/*` nodes even though `/dev/mpp_service` was mapped. The host MPP
decoder could read the stream, but the container helper remained at zero frames.
Compose now maps all six host DMA heaps. Image
`person-search-rk3588-rpp:mppdma-20261011` is healthy, reports
RPP_BF16/RKNN_NPU/RKNN_NPU, and the in-container explicit H.264 MPP helper
returned a valid 1280×720 NV12 frame from the Mac RTSP source. Search-stop now
interrupts the helper subprocess as well. This proves hardware decode reaches
the container; it is not yet an end-to-end recognition/FPS result. Targets are
held in service memory, so recreating staging cleared the previous enrollment;
re-enroll a target before the live search acceptance run. Production `18080`
remained ready and unchanged at that point, before the operator-requested
cutover below.

## Port cutover (2026-10-11)

At the operator's request, the native single-stream systemd service was
disabled and stopped without removing its source directory or unit file. The
`person-search-rk3588-rpp:mppdma-20261011` container now uses port `18080`; its
hybrid route is RPP_BF16 for face detection and RKNN NPU for person detection
and face recognition. Readiness and the ArcFace embedding contract were
verified after cutover. Live end-to-end FPS and two-source camera acceptance
remain pending; service recreation also clears in-memory target enrollment.

## Local checks

```bash
uv run --no-sync ruff check .
uv run --no-sync pytest -q
uv lock --check
```

The ARM64 image and hardware readiness checks must additionally run on an
RK3588 host; local x86 tests cannot validate the RPP SDK or device access.
