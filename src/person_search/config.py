from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PERSON_SEARCH_", env_file=".env", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    insightface_model: str = "buffalo_l"
    insightface_root: Path = Path("~/.insightface").expanduser()
    yolox_model: Path = Path("models/yolox_tiny.onnx")
    prefer_cuda: bool = True
    # 0 uses InsightFace 1.x Auto mode: 128x128 + 640x640. The small pass is
    # important for close-up faces that can be missed by a fixed 640x640 pass.
    face_detection_size: int = 0
    person_input_width: int = 416
    person_input_height: int = 416

    input_fps: float = 15.0
    person_detection_hz_cpu: float = 5.0
    person_detection_hz_cuda: float = 12.0
    face_detection_hz_cpu: float = 5.0
    face_detection_hz_cuda: float = 10.0
    frame_queue_size: int = 2

    face_detection_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_enrollment_detection_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_enrollment_face_px: int = 100
    min_search_face_px: int = 64
    preferred_search_face_px: int = 80
    min_enrollment_blur_variance: float = 5.0
    min_search_blur_variance: float = 45.0
    min_brightness: float = 35.0
    max_brightness: float = 225.0
    max_abs_roll_degrees: float = 25.0
    max_yaw_proxy: float = 0.45

    similarity_threshold: float = Field(default=0.55, ge=-1.0, le=1.0)
    small_face_similarity_threshold: float = Field(default=0.60, ge=-1.0, le=1.0)
    fallback_face_detection_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    evidence_required: int = Field(default=3, ge=1)
    evidence_window_seconds: float = 1.5
    small_face_evidence_required: int = Field(default=4, ge=1)
    small_face_evidence_window_seconds: float = Field(default=2.0, gt=0)
    confirmed_track_grace_seconds: float = 2.0
    candidate_emit_interval_seconds: float = 0.5
    face_track_iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    face_track_buffer_seconds: float = Field(default=1.0, gt=0)
    face_fallback_enabled: bool = True
    roi_face_detection_hz_cuda: float = Field(default=4.0, ge=0.0)
    roi_face_detection_hz_cpu: float = Field(default=0.0, ge=0.0)
    roi_max_tracks_per_pass: int = Field(default=8, ge=1)
    roi_min_person_height_px: int = Field(default=120, ge=1)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    rtsp_reconnect_max_seconds: float = 10.0
    rtsp_open_timeout_seconds: float = Field(default=5.0, gt=0)
    rtsp_read_timeout_seconds: float = Field(default=5.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
