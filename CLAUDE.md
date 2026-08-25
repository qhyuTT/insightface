# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Despite the directory name, this is **not** the upstream InsightFace repo. It is `robot-person-search-poc`: a PoC that takes one target photo and finds that person in live RTSP/USB video, exposed as a FastAPI service. It *consumes* `insightface` as a dependency.

Docs (`README.md`, `docs/`, `tasks/`) are written in Chinese; code comments are English. Follow that split.

## Commands

Everything goes through `uv` (Python pinned to 3.11 — `enum.StrEnum` requires it). Never use bare `pip` as the main flow.

```bash
uv sync --extra test                          # test deps only — enough for the whole test suite
uv sync --extra test --extra inference-cpu    # adds insightface + onnxruntime for real inference

uv run pytest                                 # 159 tests, ~5s, no models required
uv run pytest tests/test_tracker.py           # single file
uv run pytest tests/test_service.py::test_enroll_requires_exactly_one_face   # single test
uv run ruff check .                           # line-length 100, target py311

uv run person-search-download-models          # YOLOX-Tiny -> models/yolox_tiny.onnx, SHA-256 verified
uv run person-search-api                      # serves 127.0.0.1:8000; /docs, /monitor
uv run person-search-eval --photo X --video Y --output-dir Z   # offline eval / threshold calibration
```

The test suite injects fakes (`tests/conftest.py`) for both model backends, so it never downloads or loads a model. Keep it that way — a test that needs `buffalo_l` or `yolox_tiny.onnx` doesn't belong in `tests/`.

Auxiliary scripts:

```bash
./scripts/local_rtsp.sh start|status|stop|restart|logs   # publish the Mac camera via MediaMTX+FFmpeg (launchctl)
./scripts/deploy_t4.sh                                   # build + run the CUDA container on a T4 server
```

## Architecture

Per-frame pipeline, one module per stage:

```
video.LatestFrameReader     capture thread, queue(maxsize=2), drops old frames
  -> detector.YoloXOnnxDetector    person boxes (ONNX, COCO class 0)
  -> tracker.ByteTracker           high/low-score two-stage IoU association
  -> backends.InsightFaceBackend   SCRFD detect + ArcFace embed + quality.assess_face
  -> confirmation.TrackConfirmation  face->track association, multi-frame evidence
  -> service.SearchSession         events, metrics, MJPEG preview
  -> api                           REST + WebSocket
```

### The core invariant: identity is confirmed per *track*, never per frame

This is the reason `confirmation.py` exists and the thing most likely to be broken by a well-meaning change. A single high-similarity frame must never confirm a match. `TrackConfirmation` accumulates `_Evidence` per track and requires all of:

- `evidence_required` (3) samples inside a sliding `evidence_window_seconds` (1.5s) window,
- samples ≥0.2s apart and from distinct `frame_id`s (no double-counting one frame),
- the window's **statistic** (`evidence_statistic`: `median`, or `top_k_mean` under
  `match_profile=responsive`) ≥ `similarity_threshold`. Note the median gate is
  structurally satisfied for tiers that refuse sub-threshold observations — the
  statistic only bites on the far-face tier, which banks everything by design.

Only then does state flip `candidate` → `confirmed`. `confirmed_track_grace_seconds` keeps a confirmed track alive briefly when the track or face drops out, then emits `lost`.

Face→track association (`associate_faces_to_tracks`) requires the face center to fall in the **upper 60%** of a person box, and picks the *smallest* containing box as least ambiguous. At most one face — the highest quality one — feeds a track per frame.

Two things exist purely to stop earned evidence from being thrown away, and both are easy to undo by accident:

- **Tier hysteresis** (`face_tier_hysteresis_px`). Changing tier clears the evidence window; a moving camera sweeps one face across 48/64/80 px repeatedly. Leaving a tier costs a margin, and the tier is resolved once per track by the *session* (`_track_tiers`), not per target, because it is a property of the observation. `resolve_face_tier` is the only place that decides it.
- **Camera motion compensation** (`camera_motion_compensation`). `ByteTracker._states`-keyed evidence dies with a track id, and pure IoU reads a pan as "every track lost". `service._estimate_camera_motion` estimates global translation with `cv2.phaseCorrelate` on a downscaled gray frame and `ByteTracker.update(detections, motion=...)` applies it before association. Velocity is measured between *observations* (`observed_bbox`), never against the already-advanced prediction.

### Enrollment and search gates are deliberately different

`quality.assess_face(..., enrollment=bool)` branches on this flag, and the asymmetry is intentional (recorded in `tasks/lessons.md`):

| | enrollment | search frames |
|---|---|---|
| min detection score | 0.60 | 0.45 |
| min face px | 100 | 80 |
| min blur variance | 5.0 (lenient) | 45.0 (**stricter**) |
| roll / yaw gates | enforced | not enforced (pose only feeds the soft score) |

Video frames get the *stricter* blur gate because motion blur must not become confirmation evidence; enrollment photos tolerate mild soft focus. Pose limits (`max_abs_roll_degrees`, `max_yaw_proxy`) constrain enrollment only. Do not unify these thresholds.

Similarly, `face_detection_size: int = 0` means InsightFace 1.x Auto mode (128 **and** 640 dual-scale). 128 catches close-up faces that a fixed 640 pass misses. `tests/test_backend_config.py` exists solely to pin this default — don't hardcode 640.

`face_detection_extra_scale_cuda` (default 1280) appends a larger scale on CUDA; SCRFD merges scales itself through NMS, so `detect_faces` passes a list and pays one call. But measured on a T4, **every distinct ONNX input shape costs ~30ms of re-planning**, which dwarfs the scale's own inference (1280 alone 53ms, 640 alone 18ms, 320 alone 5ms). So `[128,640]` costs 108ms and `[128,640,1280]` costs 181ms, while `[1280]` alone costs 57ms *and* scores higher at every face size (0.69 at a 21px face where `[128,640]` misses entirely; 0.82 against 0.75 at 52px). Production therefore *replaces* the scale — `PERSON_SEARCH_FACE_DETECTION_SIZE=1280` in `deploy_t4.sh` — rather than adding one. Same reason `roi_detection_scale` quantizes onto two rungs instead of rounding per crop: adaptive per-crop scales measured 233ms/pass against 136ms quantized. If you are tempted to add a scale, measure the shape-switch cost first; `face_deep_scan_every_n` is the escape hatch when a large scale must be amortised.

### Threading and lifecycle

Three layers, all in one process:

1. `LatestFrameReader` capture thread — bounded queue, evicts the oldest frame and calls `on_drop` rather than blocking. Handles RTSP reconnect with exponential backoff.
2. `SearchSession._run` worker thread — the pipeline loop. Person and face detection run at independent rates (`*_hz_cpu` vs `*_hz_cuda`, selected by sniffing `"CUDA"` in the provider names), so they are throttled separately from frame arrival.
3. FastAPI async handlers — every call into blocking manager/session code goes through `asyncio.to_thread`.

`EventHub` and `PreviewHub` both fan out via a `threading.Condition` plus a monotonic `seq`. `PreviewHub` keeps only the newest JPEG, so a slow browser can never stall inference. WebSocket clients resume with `?after_seq=`.

State is **in-process memory only** and this is load-bearing:

- `SearchManager` allows exactly one active search (`search_capacity_exceeded`, 409).
- `main.py` pins `workers=1`. More than one uvicorn worker silently breaks the model.
- `_on_finished` deletes the targets and their embeddings when a search ends; photos and embeddings never touch disk.

Scaling past one robot needs an external queue and state store — that's a rewrite, not a config change.

### Conventions worth matching

- Backends are `Protocol`s (`FaceBackend`, `PersonDetector`). Depend on the protocol, not the concrete class, so tests can inject fakes.
- Heavy models load lazily via `ensure_ready()`, raising `ModelUnavailableError` with an actionable message — never at import time.
- Errors: raise `PersonSearchError(message, code=..., status_code=...)`; `api._problem` renders `{"detail": {"code", "message"}}`. Worker exceptions pass through `_safe_error` and RTSP URIs through `_sanitize_source` — neither internals nor credentials reach a response.
- Config is `Settings` (pydantic-settings, `PERSON_SEARCH_` env prefix, `.env`). Add tunables there, not as literals.
- Every module starts with `from __future__ import annotations`.
- Target names come from users: pass them through events, but never interpolate into HTML in `static/monitor.html`.

## Operational constraints

- `onnxruntime` and `onnxruntime-gpu` share one Python namespace and **cannot coexist**. GPU setup means uninstalling the CPU wheel first, and GPU environments need their own lock file. Verify with the `provider` field on `GET /v1/searches/{id}`, not by assuming.
- `models/` is gitignored (except its README). `.dockerignore` re-includes `models/*` so a host-prefetched `yolox_tiny.onnx` can enter the build context.
- The default `similarity_threshold` (0.55) is a get-it-running value, **not** a production threshold. Recalibrate with `person-search-eval` against real footage before any robot acts on a confirmation.
- `buffalo_l` is licensed for non-commercial research only.

## Workflow files

`tasks/todo.md` tracks plans and retrospectives; `tasks/lessons.md` records corrections as durable rules. Read `tasks/lessons.md` before changing detection, quality, or threshold logic — several non-obvious defaults above are justified there.
