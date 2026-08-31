from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HARD_MIN_SEARCH_FACE_PX = 48

# InsightFace 1.x Auto mode resolves det_size=0 to these two scales; mirrored here
# so the pipeline can extend the list instead of re-deriving it. See
# insightface.app.face_analysis.DEFAULT_DET_SIZES.
AUTO_DETECTION_SCALES = (128, 640)

# Named bundles of evidence settings, one per deployment scene. A profile only
# fills in fields the operator did not set explicitly, so an env override always
# wins over the profile -- it is a set of defaults, never an override.
#
# "responsive" trades false-accept headroom for time-to-confirmation.
#
# "transit" is for a hall where the subject walks past rather than lingers: it
# shortens the far-face window and drops the minimum gap between samples, so a
# short dwell can still fill a window. Its numbers are UNCALIBRATED -- they are
# deliberately no looser than "responsive" on any similarity threshold, because
# choosing a threshold without a same-person/different-person distribution is
# just writing false accepts into a default. Measure with
# `person-search-eval --dump-similarities` and override per field from the
# environment.
MATCH_PROFILE_OVERRIDES: dict[str, dict[str, object]] = {
    "conservative": {},
    "responsive": {
        "tiny_face_evidence_required": 4,
        "tiny_face_evidence_window_seconds": 2.0,
        "tiny_face_consistent_votes_required": 3,
        "tiny_face_detection_threshold": 0.55,
        "evidence_statistic": "top_k_mean",
    },
    "transit": {
        "tiny_face_evidence_required": 4,
        "tiny_face_evidence_window_seconds": 2.0,
        "tiny_face_consistent_votes_required": 3,
        "tiny_face_detection_threshold": 0.55,
        "evidence_statistic": "top_k_mean",
        # A walker is frame-starved, not sample-redundant: consecutive frames are
        # genuinely different looks at a moving face, so the 0.2s spacing that
        # keeps a *stationary* subject from banking the same pose six times is
        # pure loss here. Duplicate frames are already refused by frame_id.
        "evidence_min_interval_seconds": 0.1,
        "tiny_face_evidence_min_interval_seconds": 0.1,
    },
}



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PERSON_SEARCH_",
        env_file=".env",
        extra="ignore",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    host: str = Field(default="127.0.0.1", min_length=1, max_length=255, pattern=r"^\S+$")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    # Evidence is intentionally an opt-in, short-lived in-memory hand-off. The
    # executor does not expose a face image unless its caller configured a
    # separate shared credential.
    evidence_api_key: str | None = Field(default=None, max_length=256, repr=False)
    evidence_ttl_seconds: float = Field(default=600.0, gt=0, le=3600)

    insightface_model: str = "buffalo_l"
    insightface_root: Path = Path("~/.insightface").expanduser()
    yolox_model: Path = Path("models/yolox_tiny.onnx")
    prefer_cuda: bool = True
    # 0 uses InsightFace 1.x Auto mode: 128x128 + 640x640. The small pass is
    # important for close-up faces that can be missed by a fixed 640x640 pass.
    # This is the *search* scale. Enrollment does not read it -- see
    # enrollment_detection_size, which stays on Auto whatever this is set to.
    # ``0`` means InsightFace Auto mode; explicit detector sizes are kept on a
    # 32-pixel grid because the exported models and their post-processing assume
    # that stride.  A malformed value here otherwise fails much later, inside an
    # inference worker, which is particularly painful to diagnose in deployment.
    face_detection_size: int = Field(default=0, ge=0, multiple_of=32)
    # Enrollment photos are close-ups, so they need their own scale rather than the
    # search scale: 0 keeps Auto's 128+640. Only override with measurements taken on
    # real enrollment photos, not on the far faces face_detection_size is tuned for.
    enrollment_detection_size: int = Field(default=0, ge=0)
    # An extra, larger full-frame scale. On 1080p the 640 pass shrinks a 49px face
    # to ~16px at the network input -- the stride-8 anchor floor -- so its
    # det_score cannot reach the far-face tier's bar however clean the face is.
    # A larger scale is the only way to hand the detector real pixels; SCRFD
    # already merges multiple scales through NMS. 0 disables it.
    face_detection_extra_scale_cuda: int = Field(default=1280, ge=0, multiple_of=32)
    face_detection_extra_scale_cpu: int = Field(default=0, ge=0, multiple_of=32)
    # 1 = every full-frame pass carries the extra scale. Raise it when the deep
    # scale costs more than the loop can absorb: only every Nth pass goes deep.
    face_deep_scan_every_n: int = Field(default=1, ge=1)
    person_input_width: int = Field(default=416, ge=32, le=4096, multiple_of=32)
    person_input_height: int = Field(default=416, ge=32, le=4096, multiple_of=32)

    input_fps: float = Field(default=15.0, gt=0, le=240)
    # Person and full-frame face detection are mandatory stages.  Zero therefore
    # is not a hidden "0.1 Hz" fallback (the old loop used max(hz, 0.1)); use a
    # positive rate here.  Optional stages below (ROI and preview) explicitly use
    # zero to mean disabled, and retain that documented sentinel consistently.
    person_detection_hz_cpu: float = Field(default=5.0, gt=0, le=240)
    person_detection_hz_cuda: float = Field(default=12.0, gt=0, le=240)
    face_detection_hz_cpu: float = Field(default=5.0, gt=0, le=240)
    face_detection_hz_cuda: float = Field(default=10.0, gt=0, le=240)
    frame_queue_size: int = Field(default=2, ge=1, le=256)

    face_detection_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_enrollment_detection_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_enrollment_face_px: int = Field(default=100, ge=1, le=4096)
    min_search_face_px: int = Field(default=64, ge=1, le=4096)
    preferred_search_face_px: int = Field(default=80, ge=1, le=4096)
    tiny_face_enabled: bool = False
    tiny_face_shadow_mode: bool = True
    tiny_face_min_px: int = Field(default=HARD_MIN_SEARCH_FACE_PX, ge=HARD_MIN_SEARCH_FACE_PX)
    tiny_face_detection_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    min_enrollment_blur_variance: float = Field(default=5.0, ge=0)
    min_search_blur_variance: float = Field(default=45.0, ge=0)
    min_brightness: float = Field(default=35.0, ge=0, le=255)
    max_brightness: float = Field(default=225.0, ge=0, le=255)
    max_abs_roll_degrees: float = Field(default=25.0, ge=0, le=180)
    max_yaw_proxy: float = Field(default=0.45, ge=0, le=1)

    similarity_threshold: float = Field(default=0.55, ge=-1.0, le=1.0)
    small_face_similarity_threshold: float = Field(default=0.60, ge=-1.0, le=1.0)
    fallback_face_detection_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    evidence_required: int = Field(default=3, ge=1)
    evidence_window_seconds: float = Field(default=1.5, gt=0, le=3600)
    small_face_evidence_required: int = Field(default=4, ge=1)
    small_face_evidence_window_seconds: float = Field(default=2.0, gt=0)
    tiny_face_similarity_threshold: float = Field(default=0.64, ge=-1.0, le=1.0)
    tiny_face_aggregate_similarity_threshold: float = Field(default=0.68, ge=-1.0, le=1.0)
    tiny_face_evidence_required: int = Field(default=6, ge=1)
    tiny_face_consistent_votes_required: int = Field(default=5, ge=1)
    tiny_face_evidence_window_seconds: float = Field(default=3.0, gt=0)
    tiny_face_evidence_min_interval_seconds: float = Field(default=0.2, ge=0)
    # Minimum gap between two banked samples on one track, for the normal and small
    # tiers. It buys temporal diversity, not de-duplication -- one frame can never
    # be banked twice, that is enforced by frame_id. Lower it when the subject
    # moves through frame too fast to supply spaced-out samples.
    evidence_min_interval_seconds: float = Field(default=0.2, ge=0)
    tiny_face_min_top1_margin: float = Field(default=0.08, ge=0.0, le=2.0)
    # A far face that only reaches the relaxed association path (exactly one person
    # box contains its center) is the seated or truncated case, not an ambiguous
    # one. Dropping it outright made "sitting at a distance" unconfirmable by
    # construction, which is the opposite of what the strict flag was for.
    tiny_face_allow_relaxed_association: bool = True
    match_profile: Literal["conservative", "responsive", "transit"] = "conservative"
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
    # A track that leaves before it can fill an evidence window is the one failure
    # the tier system cannot help with: the frames simply were not there. When
    # enabled, such a track is judged once on its way out, trading the frames it
    # never got for a higher bar on the frames it did, and reports through the
    # existing shadow channel (`tiny_shadow_confirmed`) rather than as a production
    # confirmation -- it is a lead for a human, not grounds for a robot to act.
    # Off by default: it is a deliberate move toward false accepts.
    departure_adjudication_enabled: bool = False
    departure_min_samples: int = Field(default=2, ge=1)
    departure_similarity_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    confirmed_track_grace_seconds: float = Field(default=2.0, ge=0, le=3600)
    candidate_emit_interval_seconds: float = Field(default=0.5, ge=0, le=3600)
    face_track_iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    face_track_buffer_seconds: float = Field(default=1.0, gt=0)
    face_fallback_enabled: bool = True
    # ROI and preview are optional work.  Their explicit zero sentinel means
    # disabled; all other rates are bounded to avoid accidental busy loops.
    roi_face_detection_hz_cuda: float = Field(default=4.0, ge=0.0, le=240)
    roi_face_detection_hz_cpu: float = Field(default=0.0, ge=0.0, le=240)
    roi_max_tracks_per_pass: int = Field(default=3, ge=1)
    roi_min_person_height_px: int = Field(default=120, ge=1)
    roi_person_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    # A track whose ROI crop keeps yielding nothing backs off exponentially, so a
    # permanently face-less person (turned away, occluded) stops burning the budget.
    roi_backoff_max_skips: int = Field(default=16, ge=0)
    # ROI crops are already tight, so the full-frame Auto dual-scale pass is pure
    # waste here. Only the full-frame default (face_detection_size) must stay Auto.
    roi_face_detection_size: int = Field(default=320, ge=0, multiple_of=32)
    # A fixed ROI scale downsamples any crop bigger than itself, throwing away the
    # pixels the crop existed to preserve. This is the ceiling instead: small crops
    # are still upsampled to roi_face_detection_size, large ones are not shrunk.
    roi_face_detection_max_size: int = Field(default=640, ge=0, multiple_of=32)

    # Optional performance controls.  These are limits, rather than correctness
    # knobs: exceeding them causes a recorded/degraded frame, never a process-wide
    # failure.  Keeping them in Settings makes the effective runtime contract
    # visible in SearchView and allows a deployment to tune memory independently.
    roi_batch_enabled: bool = True
    roi_batch_size: int = Field(default=8, ge=1, le=256)
    arcface_micro_batch_size: int = Field(default=16, ge=1, le=256)
    max_faces_per_frame: int = Field(default=64, ge=1, le=4096)

    # The loop rate the budget tries to hold. It refills a credit bucket that the
    # opportunistic ROI pass spends, so a slow stage throttles itself down instead
    # of collapsing the loop below the sampling density the confirmation window
    # needs — and instead of being starved to zero by one frame's remainder.
    target_loop_hz: float = Field(default=10.0, gt=0)
    min_processed_fps: float = Field(default=2.0, gt=0)
    # How many target periods' worth of credit an idle stretch may bank, and
    # symmetrically how much debt one slow pass may leave behind.
    budget_credit_max_frames: float = Field(default=2.0, ge=1.0)
    preview_hz: float = Field(default=5.0, ge=0.0, le=240)
    preview_max_width: int = Field(default=960, ge=64)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    rtsp_reconnect_max_seconds: float = Field(default=10.0, gt=0, le=3600)
    rtsp_open_timeout_seconds: float = Field(default=5.0, gt=0, le=3600)
    rtsp_read_timeout_seconds: float = Field(default=5.0, gt=0, le=3600)

    # Terminal sessions retain only a metadata snapshot after the worker is done;
    # these bounds are used by SearchManager's TTL/LRU cleanup.  A positive TTL is
    # intentional: request-id reconciliation needs a finite, non-zero window.
    terminal_session_ttl_seconds: float = Field(default=600.0, gt=0, le=86400)
    max_retained_sessions: int = Field(default=100, ge=1, le=10000)
    max_enrolled_targets: int = Field(default=100, ge=1, le=10000)

    @field_validator("evidence_api_key")
    @classmethod
    def validate_evidence_api_key(cls, value: str | None) -> str | None:
        """Keep the optional header credential safe to compare and transport."""
        if value is None or value == "":
            # An empty environment override has historically meant "disabled".
            return value
        if any(ord(char) < 0x21 or ord(char) == 0x7F for char in value):
            raise ValueError("evidence_api_key must not contain whitespace or control characters")
        return value

    @model_validator(mode="after")
    def apply_match_profile(self) -> Self:
        """Fill in the selected scene profile for fields the operator left alone.

        Runs before ``validate_face_tiers`` so the profile's values are the ones
        checked for consistency. Anything set through the environment stays put:
        the profile is a default, never an override.
        """
        for name, value in MATCH_PROFILE_OVERRIDES.get(self.match_profile, {}).items():
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

    @model_validator(mode="after")
    def validate_runtime_bounds(self) -> Self:
        """Validate relationships that individual field constraints cannot see.

        Keeping these checks at settings construction time turns configuration
        mistakes into a deterministic startup error instead of a divide-by-zero,
        busy reconnect loop, or an evidence window that can never fill.
        """
        if self.min_brightness >= self.max_brightness:
            raise ValueError("min_brightness must be less than max_brightness")

        # ``roi_face_detection_max_size=0`` is retained as the documented
        # "unbounded/disabled ceiling" sentinel.  When both rungs are explicit,
        # however, the upper rung must not be below the floor.
        if (
            self.roi_face_detection_size > 0
            and self.roi_face_detection_max_size > 0
            and self.roi_face_detection_max_size < self.roi_face_detection_size
        ):
            raise ValueError(
                "roi_face_detection_max_size cannot be less than roi_face_detection_size"
            )

        # A spacing larger than the evidence window makes the requested quorum
        # mathematically impossible.  The first sample is free, hence N-1 gaps.
        if (
            self.evidence_required - 1
        ) * self.evidence_min_interval_seconds > self.evidence_window_seconds + 1e-9:
            raise ValueError(
                "evidence_required and evidence_min_interval_seconds cannot fill "
                "evidence_window_seconds"
            )
        if (
            self.small_face_evidence_required - 1
        ) * self.evidence_min_interval_seconds > self.small_face_evidence_window_seconds + 1e-9:
            raise ValueError(
                "small_face_evidence_required and evidence_min_interval_seconds cannot "
                "fill small_face_evidence_window_seconds"
            )
        if (
            self.tiny_face_evidence_required - 1
        ) * self.tiny_face_evidence_min_interval_seconds \
            > self.tiny_face_evidence_window_seconds + 1e-9:
            raise ValueError(
                "tiny_face_evidence_required and tiny_face_evidence_min_interval_seconds "
                "cannot fill tiny_face_evidence_window_seconds"
            )
        return self

    @property
    def effective_search_min_face_px(self) -> int:
        """Return the configured search limit without crossing the safety floor."""
        configured = self.tiny_face_min_px if self.tiny_face_enabled else self.min_search_face_px
        return max(HARD_MIN_SEARCH_FACE_PX, configured)

    def enrollment_detection_scales(self) -> tuple[int, ...]:
        """Return the detector input scales for an enrollment photo, ascending.

        Deliberately independent of ``face_detection_size``. That setting is tuned
        against *search* frames, where the face is far away and a single large scale
        wins; an enrollment photo is a close-up by definition, and SCRFD upscales
        whatever it is given to fill the input, with no ceiling. Past roughly 500px
        at the network input a face overshoots the stride-32 anchors and is missed
        outright -- so ``PERSON_SEARCH_FACE_DETECTION_SIZE=1280`` (what production
        pins for search) makes an ordinary portrait undetectable: measured MISS at
        every frame fraction from 0.45 up, against 0.865-0.905 on Auto. The 128 pass
        is what shrinks a frame-filling face back under the anchor ceiling, which is
        why enrollment must keep the dual-scale default whatever search uses.
        """
        if self.enrollment_detection_size > 0:
            return (self.enrollment_detection_size,)
        return AUTO_DETECTION_SCALES

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
