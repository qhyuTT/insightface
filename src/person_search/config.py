from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HARD_MIN_SEARCH_FACE_PX = 48


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PERSON_SEARCH_", env_file=".env", extra="ignore")

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
    tiny_face_enabled: bool = False
    tiny_face_shadow_mode: bool = True
    tiny_face_min_px: int = Field(default=HARD_MIN_SEARCH_FACE_PX, ge=HARD_MIN_SEARCH_FACE_PX)
    tiny_face_detection_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
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
    tiny_face_similarity_threshold: float = Field(default=0.64, ge=-1.0, le=1.0)
    tiny_face_aggregate_similarity_threshold: float = Field(default=0.68, ge=-1.0, le=1.0)
    tiny_face_evidence_required: int = Field(default=6, ge=1)
    tiny_face_consistent_votes_required: int = Field(default=5, ge=1)
    tiny_face_evidence_window_seconds: float = Field(default=3.0, gt=0)
    tiny_face_evidence_min_interval_seconds: float = Field(default=0.2, ge=0)
    tiny_face_min_top1_margin: float = Field(default=0.08, ge=0.0, le=2.0)
    confirmed_track_grace_seconds: float = 2.0
    candidate_emit_interval_seconds: float = 0.5
    face_track_iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    face_track_buffer_seconds: float = Field(default=1.0, gt=0)
    face_fallback_enabled: bool = True
    roi_face_detection_hz_cuda: float = Field(default=4.0, ge=0.0)
    roi_face_detection_hz_cpu: float = Field(default=0.0, ge=0.0)
    roi_max_tracks_per_pass: int = Field(default=3, ge=1)
    roi_min_person_height_px: int = Field(default=120, ge=1)
    roi_person_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    # A track whose ROI crop keeps yielding nothing backs off exponentially, so a
    # permanently face-less person (turned away, occluded) stops burning the budget.
    roi_backoff_max_skips: int = Field(default=16, ge=0)
    # ROI crops are already tight, so the full-frame Auto dual-scale pass is pure
    # waste here. Only the full-frame default (face_detection_size) must stay Auto.
    roi_face_detection_size: int = Field(default=320, ge=0)

    # The loop rate the budget tries to hold. It refills a credit bucket that the
    # opportunistic ROI pass spends, so a slow stage throttles itself down instead
    # of collapsing the loop below the sampling density the confirmation window
    # needs — and instead of being starved to zero by one frame's remainder.
    target_loop_hz: float = Field(default=10.0, gt=0)
    min_processed_fps: float = Field(default=2.0, gt=0)
    # How many target periods' worth of credit an idle stretch may bank, and
    # symmetrically how much debt one slow pass may leave behind.
    budget_credit_max_frames: float = Field(default=2.0, ge=1.0)
    preview_hz: float = Field(default=5.0, ge=0.0)
    preview_max_width: int = Field(default=960, ge=64)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    rtsp_reconnect_max_seconds: float = 10.0
    rtsp_open_timeout_seconds: float = Field(default=5.0, gt=0)
    rtsp_read_timeout_seconds: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def validate_face_tiers(self) -> Self:
        if not self.tiny_face_min_px < self.min_search_face_px <= self.preferred_search_face_px:
            raise ValueError(
                "face size tiers must satisfy tiny_face_min_px < "
                "min_search_face_px <= preferred_search_face_px"
            )
        if self.tiny_face_consistent_votes_required > self.tiny_face_evidence_required:
            raise ValueError(
                "tiny_face_consistent_votes_required cannot exceed tiny_face_evidence_required"
            )
        return self

    @property
    def effective_search_min_face_px(self) -> int:
        """Return the configured search limit without crossing the safety floor."""
        configured = self.tiny_face_min_px if self.tiny_face_enabled else self.min_search_face_px
        return max(HARD_MIN_SEARCH_FACE_PX, configured)


@lru_cache
def get_settings() -> Settings:
    return Settings()
