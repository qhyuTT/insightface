"""Optional RKNN Lite backends for Rockchip edge devices.

The development and CI environments do not ship Rockchip's proprietary
``rknnlite`` package.  Imports are therefore deliberately lazy: the rest of
the application remains usable on CPU/ONNX Runtime, while an RK3588 device
gets a clear, actionable error when the runtime or model artifact is missing.

Model conversion (YOLOX/SCRFD/ArcFace -> ``.rknn``) is intentionally kept out
of the runtime package.  Conversion must happen on a host with a matching
RKNN Toolkit2 version and a checked-in calibration/manifest workflow.
"""

from __future__ import annotations

import hashlib
import importlib
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np

from .config import Settings
from .domain import FaceObservation
from .errors import ModelUnavailableError


class RknnRuntimeProtocol(Protocol):
    """Small subset of :class:`rknnlite.api.RKNNLite` used by this project."""

    def load_rknn(self, path: str) -> int | None: ...

    def init_runtime(self, **kwargs: Any) -> int | None: ...

    def inference(self, *args: Any, **kwargs: Any) -> Sequence[np.ndarray]: ...

    def release(self) -> Any: ...


RuntimeFactory = Callable[[], RknnRuntimeProtocol]


def _default_runtime_factory() -> RknnRuntimeProtocol:
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:  # pragma: no cover - depends on the target board
        raise ModelUnavailableError(
            "RKNN Lite runtime is unavailable; install the board/vendor "
            "rknnlite package and matching librknnrt.so"
        ) from exc
    return RKNNLite()


def verify_sha256(path: Path, expected: str | None) -> None:
    """Verify an optional model checksum before loading it."""

    if not expected:
        return
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelUnavailableError(f"cannot read RKNN model {path}: {exc}") from exc
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ModelUnavailableError(
            f"RKNN model checksum mismatch for {path}: expected {expected}, got {actual}"
        )


class RknnSession:
    """Lazy, checked RKNN Lite session wrapper.

    ``runtime_factory`` is injectable for unit tests and for vendor-specific
    runtime shims.  The wrapper accepts both older RKNN Lite APIs (which do
    not accept ``core_mask``) and newer ones.
    """

    provider_name = "RKNNLite"

    def __init__(
        self,
        model_path: Path,
        *,
        core_mask: int | None = None,
        expected_sha256: str | None = None,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.core_mask = core_mask
        self.expected_sha256 = expected_sha256
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._runtime: RknnRuntimeProtocol | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._runtime is not None

    def ensure_ready(self) -> None:
        with self._lock:
            self._ensure_ready_locked()

    def _ensure_ready_locked(self) -> None:
        """Initialize the runtime while ``_lock`` is held.

        Keeping the check and initialization in one locked helper is important:
        ``inference`` must not observe a runtime that was released between its
        initial readiness check and the actual call.
        """

        if self._runtime is not None:
            return
        if not self.model_path.is_file():
            raise ModelUnavailableError(
                f"RKNN model not found at {self.model_path}; provide a converted .rknn artifact"
            )
        verify_sha256(self.model_path, self.expected_sha256)
        runtime = self._runtime_factory()
        try:
            result = runtime.load_rknn(str(self.model_path))
            if result not in (None, 0):
                raise RuntimeError(f"load_rknn returned {result}")
            kwargs = {} if self.core_mask is None else {"core_mask": self.core_mask}
            try:
                result = runtime.init_runtime(**kwargs)
            except TypeError:
                # Older rknnlite releases do not expose core_mask.
                if not kwargs:
                    raise
                result = runtime.init_runtime()
            if result not in (None, 0):
                raise RuntimeError(f"init_runtime returned {result}")
        except ModelUnavailableError:
            with suppress(Exception):
                runtime.release()
            raise
        except Exception as exc:
            with suppress(Exception):
                runtime.release()
            raise ModelUnavailableError(
                f"failed to initialize RKNN model {self.model_path}: {exc}"
            ) from exc
        self._runtime = runtime

    def inference(
        self,
        inputs: Sequence[np.ndarray],
        *,
        data_format: Sequence[str] | None = None,
    ) -> Sequence[np.ndarray]:
        input_list = list(inputs)
        formats = None if data_format is None else list(data_format)
        if formats is not None and len(formats) != len(input_list):
            raise ModelUnavailableError(
                "RKNN input data_format count must match the number of input tensors"
            )
        with self._lock:
            self._ensure_ready_locked()
            assert self._runtime is not None  # ensured while holding the lock
            try:
                if formats is not None:
                    # RKNNLite defaults to NHWC.  NCHW models require this
                    # keyword (as documented by the vendor examples); do not
                    # silently drop it and feed a tensor in the wrong layout.
                    return self._runtime.inference(
                        inputs=input_list,
                        data_format=formats,
                    )
                try:
                    return self._runtime.inference(inputs=input_list)
                except TypeError:
                    # A few vendor builds expose ``inference(list)`` only.
                    return self._runtime.inference(input_list)
            except Exception as exc:
                raise ModelUnavailableError(f"RKNN inference failed: {exc}") from exc

    def warmup(
        self,
        inputs: Sequence[np.ndarray],
        *,
        data_format: Sequence[str] | None = None,
    ) -> None:
        self.inference(inputs, data_format=data_format)

    def release(self) -> None:
        with self._lock:
            runtime, self._runtime = self._runtime, None
            # Keep the lock while releasing the native handle.  Otherwise a
            # concurrent ensure_ready() could create a second runtime before
            # the first handle has finished tearing down its NPU resources.
            if runtime is not None:
                runtime.release()

    def __enter__(self) -> Self:
        self.ensure_ready()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class RknnPersonDetector:
    """YOLOX-compatible detector backed by RKNN Lite.

    The preprocessing and postprocessing intentionally reuse the project's
    tested YOLOX helpers.  The converted RKNN model must preserve the raw
    ``[1, 3549, 85]`` output contract (or an equivalent first output tensor).
    """

    provider_name = RknnSession.provider_name

    def __init__(
        self,
        settings: Settings,
        confidence: float = 0.25,
        nms: float = 0.45,
        *,
        session: RknnSession | None = None,
    ) -> None:
        self.settings = settings
        self.confidence = confidence
        self.nms = nms
        model_path = settings.rknn_person_model
        if model_path is None:
            model_path = settings.yolox_model.with_suffix(".rknn")
        self._session = session or RknnSession(
            model_path,
            core_mask=settings.rknn_core_mask,
            expected_sha256=settings.rknn_person_sha256,
        )

    def ensure_ready(self) -> None:
        self._session.ensure_ready()

    def detect(self, frame: np.ndarray):
        # Imports stay local so CPU-only environments do not pay an RKNN import
        # cost and so this module remains independently testable.
        from .detector import _decode_yolox, _nms, _preprocess
        from .domain import Detection

        image, ratio = _preprocess(
            frame, (self.settings.person_input_height, self.settings.person_input_width)
        )
        if self.settings.rknn_person_input_layout == "nhwc":
            image = np.ascontiguousarray(image.transpose(1, 2, 0))
        if self.settings.rknn_person_input_dtype == "uint8":
            image = np.clip(image, 0, 255).astype(np.uint8)
        data_format = ["nchw"] if self.settings.rknn_person_input_layout == "nchw" else None
        outputs = self._session.inference([image[None]], data_format=data_format)
        if not outputs:
            raise ModelUnavailableError("RKNN YOLOX model returned no output tensors")
        raw = np.asarray(outputs[0])
        if raw.ndim == 3:
            if raw.shape[0] != 1:
                raise ModelUnavailableError(
                    f"unexpected RKNN YOLOX batch shape {raw.shape}; expected batch size 1"
                )
            raw = raw[0]
        if raw.ndim != 2 or raw.shape[1] < 6:
            raise ModelUnavailableError(
                "unexpected RKNN YOLOX output shape "
                f"{raw.shape}; expected one fused [1, N, C] tensor with C >= 6"
            )
        if not np.issubdtype(raw.dtype, np.floating):
            raise ModelUnavailableError(
                "RKNN YOLOX output is quantized integer data; export a model with "
                "dequantized floating outputs or provide an output dequantization adapter"
            )
        predictions = _decode_yolox(
            raw, (self.settings.person_input_height, self.settings.person_input_width)
        )
        boxes = predictions[:, :4]
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        scores = predictions[:, 4] * predictions[:, 5]
        keep = np.where(scores >= self.confidence)[0]
        if keep.size == 0:
            return []
        selected = _nms(boxes_xyxy[keep], scores[keep], self.nms)
        return [
            Detection(
                bbox=boxes_xyxy[keep[index]].astype(np.float32),
                score=float(scores[keep[index]]),
            )
            for index in selected
        ]

    def release(self) -> None:
        self._session.release()


class RknnFaceAdapter(Protocol):
    """Vendor/model-specific SCRFD + ArcFace adapter contract."""

    model_name: str

    def ensure_ready(self) -> None: ...

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]: ...


class RknnFaceBackend:
    """Delegating face backend for converted SCRFD/ArcFace models.

    SCRFD output ordering and anchor configuration differ between RKNN model
    exports.  Rather than silently guessing, the runtime accepts an explicit
    adapter supplied by the board integration.  This keeps the API stable and
    makes unsupported model conversions fail loudly.
    """

    provider_name = RknnSession.provider_name

    def __init__(
        self,
        settings: Settings,
        *,
        adapter: RknnFaceAdapter | None = None,
    ) -> None:
        self.settings = settings
        self._adapter = adapter
        self._adapter_injected = adapter is not None
        self._adapter_spec = settings.rknn_face_adapter
        self._ready = False
        self._lock = threading.RLock()
        self.model_name = settings.insightface_model

    def ensure_ready(self) -> None:
        with self._lock:
            self._ensure_ready_locked()

    def _ensure_ready_locked(self) -> None:
        if self._ready:
            return

        artifacts = (
            (
                "face detection",
                self.settings.rknn_face_detection_model,
                self.settings.rknn_face_detection_sha256,
            ),
            (
                "face recognition",
                self.settings.rknn_face_recognition_model,
                self.settings.rknn_face_recognition_sha256,
            ),
        )
        missing: list[str] = []
        for label, path, _ in artifacts:
            if path is None:
                missing.append(f"{label} model path is not configured")
            elif not Path(path).is_file():
                missing.append(f"{label} model not found at {path}")
        if missing:
            raise ModelUnavailableError(
                "RKNN face backend needs converted SCRFD and ArcFace artifacts; "
                + "; ".join(missing)
            )

        # Verify configured artifacts before importing or initializing a
        # third-party adapter.  This prevents an adapter from loading an
        # unverified model and also makes the checksum settings effective.
        for _, path, expected_sha256 in artifacts:
            assert path is not None
            verify_sha256(Path(path), expected_sha256)

        adapter = self._adapter
        loaded_here = False
        if adapter is None and self._adapter_spec:
            try:
                module_name, factory_name = self._adapter_spec.split(":", 1)
                factory = getattr(importlib.import_module(module_name), factory_name)
                adapter = factory(self.settings)
                loaded_here = True
            except Exception as exc:
                raise ModelUnavailableError(
                    f"failed to load RKNN face adapter {self._adapter_spec}: {exc}"
                ) from exc
        if adapter is None:
            raise ModelUnavailableError(
                "RKNN face backend needs a validated SCRFD/ArcFace adapter; "
                "set PERSON_SEARCH_RKNN_FACE_ADAPTER=package.module:create_adapter"
            )
        try:
            adapter.ensure_ready()
            model_name = adapter.model_name
            if not isinstance(model_name, str) or not model_name.strip():
                raise TypeError("adapter.model_name must be a non-empty string")
        except ModelUnavailableError:
            if loaded_here:
                with suppress(Exception):
                    release = getattr(adapter, "release", None)
                    if release is not None:
                        release()
            raise
        except Exception as exc:
            if loaded_here:
                with suppress(Exception):
                    release = getattr(adapter, "release", None)
                    if release is not None:
                        release()
            raise ModelUnavailableError(
                f"failed to initialize RKNN face adapter: {exc}"
            ) from exc
        self._adapter = adapter
        self.model_name = model_name
        self._ready = True

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        # RKNN native contexts are not generally safe for concurrent calls.
        # Enrollment requests can overlap the active search loop, so serialize
        # adapter initialization and inference just like InsightFaceBackend.
        with self._lock:
            self._ensure_ready_locked()
            assert self._adapter is not None
            try:
                return self._adapter.analyze(frame, enrollment=enrollment)
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(f"RKNN face inference failed: {exc}") from exc

    def release(self) -> None:
        with self._lock:
            adapter = self._adapter
            release = getattr(adapter, "release", None)
            try:
                if release is not None:
                    release()
            finally:
                self._ready = False
                if not self._adapter_injected:
                    self._adapter = None
