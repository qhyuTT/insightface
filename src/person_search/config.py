from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HARD_MIN_SEARCH_FACE_PX = 48

# InsightFace 1.x Auto mode resolves det_size=0 to these two scales; mirrored here
# so the pipeline can extend the list instead of re-deriving it. See
# insightface.app.face_analysis.DEFAULT_DET_SIZES.
AUTO_DETECTION_SCALES = (128, 640)

# The "responsive" profile trades false-accept headroom for time-to-confirmation.
# It only fills in fields the operator did not set explicitly, so an env override
# always wins over the profile.
RESPONSIVE_PROFILE_OVERRIDES: dict[str, object] = {
    "tiny_face_evidence_required": 4,
    "tiny_face_evidence_window_seconds": 2.0,
    "tiny_face_consistent_votes_required": 3,
    "tiny_face_detection_threshold": 0.55,
    "evidence_statistic": "top_k_mean",
}



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
    # An extra, larger full-frame scale. On 1080p the 640 pass shrinks a 49px face
    # to ~16px at the network input -- the stride-8 anchor floor -- so its
    # det_score cannot reach the far-face tier's bar however clean the face is.
    # A larger scale is the only way to hand the detector real pixels; SCRFD
    # already merges multiple scales through NMS. 0 disables it.
    face_detection_extra_scale_cuda: int = Field(default=1280, ge=0)
    face_detection_extra_scale_cpu: int = Field(default=0, ge=0)
    # 1 = every full-frame pass carries the extra scale. Raise it when the deep
    # scale costs more than the loop can absorb: only every Nth pass goes deep.
    face_deep_scan_every_n: int = Field(default=1, ge=1)
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
    # A far face that only reaches the relaxed association path (exactly one person
    # box contains its center) is the seated or truncated case, not an ambiguous
    # one. Dropping it outright made "sitting at a distance" unconfirmable by
    # construction, which is the opposite of what the strict flag was for.
    tiny_face_allow_relaxed_association: bool = True
    match_profile: Literal["conservative", "responsive"] = "conservative"
    # How a window of similarities is reduced to the one number the verdict reads.
    # A moving robot samples plenty of bad poses, and those drag a median down even
    # when the good frames sit well clear of the threshold.
    evidence_statistic: Literal["median", "top_k_mean"] = "median"
    evidence_top_k: int = Field(default=3, ge=1)
    # Leaving a size tier requires crossing its boundary by this margin. Without it
    # a robot in motion sweeps one face back and forth across 48/64/80 px, and
    # every flip clears the evidence window.
    face_tier_hysteresis_px: int = Field(default=6, ge=0)
    # Average each crop's embedding with its mirror. One extra ArcFace row per face
    # lifts the whole similarity distribution, which is the only lever left when
    # the operator has a single enrollment photo.
    embedding_flip_tta: bool = True
    # Estimate the frame-to-frame global shift and apply it to track boxes before
    # IoU association. A panning robot breaks pure-IoU association, and every new
    # track id restarts the evidence window from zero.
    camera_motion_compensation: bool = True
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
    # A fixed ROI scale downsamples any crop bigger than itself, throwing away the
    # pixels the crop existed to preserve. This is the ceiling instead: small crops
    # are still upsampled to roi_face_detection_size, large ones are not shrunk.
    roi_face_detection_max_size: int = Field(default=640, ge=0)

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
    def apply_match_profile(self) -> Self:
        """Fill in the responsive profile for fields the operator left alone.

        Runs before ``validate_face_tiers`` so the profile's values are the ones
        checked for consistency. Anything set through the environment stays put:
        the profile is a default, never an override.
        """
        if self.match_profile != "responsive":
            return self
        for name, value in RESPONSIVE_PROFILE_OVERRIDES.items():
            if name not in self.model_fields_set:
                setattr(self, name, value)
        return self

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

    def full_frame_detection_scales(self, *, is_cuda: bool, deep: bool = True) -> tuple[int, ...]:
        """Return the detector input scales for one full-frame pass, ascending.

        ``deep=False`` drops the extra large scale, which is how
        ``face_deep_scan_every_n`` keeps the average cost down without giving the
        pass a second throttling knob of its own.
        """
        base = (
            AUTO_DETECTION_SCALES
            if self.face_detection_size <= 0
            else (self.face_detection_size,)
        )
        extra = (
            self.face_detection_extra_scale_cuda
            if is_cuda
            else self.face_detection_extra_scale_cpu
        )
        if not deep or extra <= 0:
            return tuple(sorted(set(base)))
        return tuple(sorted(set(base) | {extra}))

    def roi_detection_scale(self, crop_width: int, crop_height: int) -> int:
        """Return an ROI detector scale that never downsamples a small crop.

        Quantized onto a two-rung ladder rather than rounded per crop. Distinct
        ONNX input shapes are what cost money here: on a T4, switching the SCRFD
        input shape between two calls costs ~30ms of re-planning, against ~5ms for
        a 320px crop's own inference. Rounding each crop to its own multiple of 32
        measured 233ms per pass against 136ms for the same crops quantized -- the
        adaptive scale would have spent its entire budget on shape changes.
        """
        longest = max(int(crop_width), int(crop_height), 1)
        floor = self.roi_face_detection_size
        ceiling = max(floor, self.roi_face_detection_max_size)
        return int(floor if longest <= floor else ceiling)


@lru_cache
def get_settings() -> Settings:
    return Settings()
