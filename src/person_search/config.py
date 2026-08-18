from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PERSON_SEARCH_", env_file=".env", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"

    # ``auto`` preserves the current CPU/ONNX behaviour.  RK3588 deployments
    # should set this explicitly to ``rknn`` so a missing vendor runtime cannot
    # silently fall back to a slow CPU path.
    inference_backend: Literal["auto", "onnxruntime", "rknn"] = "auto"
    insightface_model: str = "buffalo_l"
    insightface_root: Path = Path("~/.insightface").expanduser()
    insightface_allow_download: bool = True
    yolox_model: Path = Path("models/yolox_tiny.onnx")
    prefer_cuda: bool = True
    onnx_intra_op_threads: int = Field(default=0, ge=0)
    onnx_inter_op_threads: int = Field(default=0, ge=0)
    rknn_person_model: Path | None = None
    rknn_face_detection_model: Path | None = None
    rknn_face_recognition_model: Path | None = None
    rknn_face_adapter: str | None = None
    # RK3588 exposes three NPU cores.  Zero is RKNNLite's auto mode; 1..7 are
    # the valid bit-mask combinations for cores 0..2.
    rknn_core_mask: int | None = Field(default=None, ge=0, le=7)
    rknn_person_input_layout: Literal["nchw", "nhwc"] = "nchw"
    rknn_person_input_dtype: Literal["float32", "uint8"] = "float32"
    rknn_person_sha256: str | None = None
    rknn_face_detection_sha256: str | None = None
    rknn_face_recognition_sha256: str | None = None
    # 0 uses InsightFace 1.x Auto mode: 128x128 + 640x640. The small pass is
    # important for close-up faces that can be missed by a fixed 640x640 pass.
    face_detection_size: int = 0
    # YOLOX uses stride 8/16/32 heads, so converted fixed input dimensions must
    # be positive multiples of 32.
    person_input_width: int = Field(default=416, ge=32, le=4096, multiple_of=32)
    person_input_height: int = Field(default=416, ge=32, le=4096, multiple_of=32)

    input_fps: float = Field(default=15.0, ge=0.1)
    person_detection_hz_cpu: float = Field(default=5.0, ge=0.0)
    person_detection_hz_cuda: float = Field(default=12.0, ge=0.0)
    person_detection_hz_rknn: float = Field(default=10.0, ge=0.0)
    face_detection_hz_cpu: float = Field(default=5.0, ge=0.0)
    face_detection_hz_cuda: float = Field(default=8.0, ge=0.0)
    face_detection_hz_rknn: float = Field(default=6.0, ge=0.0)
    frame_queue_size: int = Field(default=2, ge=1, le=32)

    # Edge devices should not spend more CPU time encoding a UI preview than
    # running inference.  The preview remains enabled by default for PoC
    # compatibility, but is bounded and can be disabled in production.
    preview_enabled: bool = True
    preview_fps: float = Field(default=5.0, ge=0.0)
    preview_max_width: int = Field(default=960, ge=0)
    preview_jpeg_quality: int = Field(default=78, ge=1, le=100)
    capture_backend: Literal["auto", "opencv", "gstreamer"] = "auto"
    gstreamer_rtsp_codec: Literal["h264", "h265"] = "h264"
    gstreamer_decoder: str = Field(
        default="mppvideodec", pattern=r"^[A-Za-z0-9_-]+$"
    )
    gstreamer_latency_ms: int = Field(default=100, ge=0, le=5000)
    max_capture_width: int = Field(default=1920, ge=0)
    max_capture_height: int = Field(default=1080, ge=0)

    face_detection_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_enrollment_detection_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_enrollment_face_px: int = 100
    min_search_face_px: int = 80
    min_enrollment_blur_variance: float = 5.0
    min_search_blur_variance: float = 45.0
    min_brightness: float = 35.0
    max_brightness: float = 225.0
    max_abs_roll_degrees: float = 25.0
    max_yaw_proxy: float = 0.45

    similarity_threshold: float = Field(default=0.55, ge=-1.0, le=1.0)
    evidence_required: int = Field(default=3, ge=1)
    evidence_window_seconds: float = Field(default=1.5, gt=0.0)
    confirmed_track_grace_seconds: float = Field(default=2.0, gt=0.0)
    candidate_emit_interval_seconds: float = Field(default=0.5, ge=0.0)
    # Keep the in-memory biometric store bounded. Targets are normally removed
    # after a search or by DELETE /v1/targets/{id}, but clients can enroll and
    # abandon targets indefinitely unless the service enforces a ceiling.
    max_enrolled_targets: int = Field(default=32, ge=1, le=1000)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    rtsp_reconnect_max_seconds: float = Field(default=10.0, gt=0.0)
    # Comma-separated exact host names, IP addresses, or IP networks. An empty
    # value keeps the PoC's historical behavior; production edge deployments
    # should restrict this to the camera VLAN/hosts to mitigate RTSP SSRF.
    rtsp_allowed_hosts: str = Field(default="", max_length=4096)

    @field_validator(
        "rknn_person_sha256",
        "rknn_face_detection_sha256",
        "rknn_face_recognition_sha256",
        mode="before",
    )
    @classmethod
    def normalize_rknn_sha256(cls, value: object) -> str | None:
        """Accept blank optional env values but reject malformed checksums."""

        if value is None:
            return None
        checksum = str(value).strip().lower()
        if not checksum:
            return None
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise ValueError("RKNN SHA-256 must contain exactly 64 hexadecimal characters")
        return checksum


@lru_cache
def get_settings() -> Settings:
    return Settings()
