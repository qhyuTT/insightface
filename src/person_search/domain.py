from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator


class SearchStatus(StrEnum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    SOURCE_LOST = "source_lost"
    STOPPING = "stopping"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    STOPPED = "stopped"
    FAILED = "failed"


class MatchState(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    LOST = "lost"


class TargetSearchStatus(StrEnum):
    SEARCHING = "searching"
    FOUND = "found"


class SourceType(StrEnum):
    RTSP = "rtsp"
    CAMERA = "camera"
    FILE = "file"


class SourceConfig(BaseModel):
    type: SourceType
    uri: str | None = None
    device_index: int | None = Field(default=None, ge=0)
    debug_preview: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> SourceConfig:
        if self.type in (SourceType.RTSP, SourceType.FILE) and not self.uri:
            raise ValueError("uri is required for rtsp and file sources")
        if self.type == SourceType.CAMERA and self.device_index is None:
            raise ValueError("device_index is required for camera sources")
        return self


class SearchCreate(BaseModel):
    target_id: str
    source: SourceConfig
    timeout_seconds: float | None = Field(default=None, gt=0)
    replace_active: bool = False
    request_id: str | None = Field(default=None, max_length=128)


class TargetSearchView(BaseModel):
    target_id: str
    name: str
    status: TargetSearchStatus = TargetSearchStatus.SEARCHING
    found_at: int | None = None
    best_similarity: float | None = None
    best_observed_similarity: float | None = None
    last_face_px: int | None = None
    evidence_count: int = 0
    required_evidence: int = 0
    qualifying_evidence: int = 0
    # The number the verdict actually reads, plus the name of the reduction that
    # produced it. Reporting a value without its statistic invites reading a
    # top-K mean as a median.
    window_similarity: float | None = None
    window_statistic: str | None = None
    required_similarity: float | None = None
    aggregate_similarity: float | None = None
    required_aggregate_similarity: float | None = None
    # Which size tier is judging this track. A moving robot crosses tiers, and the
    # thresholds above are meaningless without knowing which one is in force.
    tier: str | None = None
    last_rejection_reason: str | None = None
    # The size of the face the rejection reason belongs to. It is not always
    # last_face_px: the largest face seen and the largest rejected face are
    # different observations, and pairing one's size with the other's reason
    # produced "49px / face_too_small" against a 48px floor.
    last_rejection_face_px: int | None = None


class SearchView(BaseModel):
    search_id: str
    target_id: str | None = None
    target_name: str = "目标"
    status: SearchStatus
    source: SourceConfig
    provider: str | None = None
    processed_fps: float = 0.0
    p95_latency_ms: float = 0.0
    dropped_frames: int = 0
    face_observations: int = 0
    accepted_faces: int = 0
    small_faces: int = 0
    unassociated_faces: int = 0
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    association_counts: dict[str, int] = Field(default_factory=dict)
    face_size_counts: dict[str, int] = Field(default_factory=dict)
    face_source_counts: dict[str, int] = Field(default_factory=dict)
    match_stage_counts: dict[str, int] = Field(default_factory=dict)
    stage_p95_latency_ms: dict[str, float] = Field(default_factory=dict)
    effective_hz: dict[str, float] = Field(default_factory=dict)
    source_fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    roi_calls_per_frame: float = 0.0
    drop_rate: float = 0.0
    end_to_end_p95_latency_ms: float = 0.0
    camera_motion_px_p95: float = 0.0
    blur_variance_p50: float = 0.0
    blur_variance_p95: float = 0.0
    budget_skips: dict[str, int] = Field(default_factory=dict)
    effective_config: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    targets: list[TargetSearchView] = Field(default_factory=list)
    found_count: int = 0
    total_count: int = 0
    unfound_target_ids: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    request_id: str | None = None


class TargetView(BaseModel):
    target_id: str
    name: str = "目标"
    face_width: int
    face_height: int
    detection_score: float
    quality_score: float
    model: str


class SearchEvent(BaseModel):
    search_id: str
    target_id: str
    target_name: str = "目标"
    state: MatchState
    timestamp_ms: int
    track_id: int
    bbox: tuple[float, float, float, float]
    similarity: float
    quality: float
    evidence_count: int
    model: str
    association: str = "person_strict"


@dataclass(slots=True)
class Detection:
    bbox: np.ndarray
    score: float


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: np.ndarray
    score: float


@dataclass(slots=True)
class FaceObservation:
    bbox: np.ndarray
    detection_score: float
    # None until embed_faces() runs. Detection is cheap and most detections are
    # discarded (dedup, quality, association), so ArcFace is deferred until a face
    # has actually earned it.
    embedding: np.ndarray | None
    quality: float
    landmarks: np.ndarray | None = None
    accepted: bool = True
    rejection_reasons: tuple[str, ...] = ()
    # Carried so the sharpness gate's own number can reach the panel. A gate whose
    # value is never reported cannot be calibrated against real footage.
    blur_variance: float = 0.0

    @property
    def short_side(self) -> int:
        return max(0, int(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])))


@dataclass(slots=True)
class Target:
    target_id: str
    embedding: np.ndarray
    view: TargetView
    name: str = "目标"


@dataclass(slots=True)
class SearchMetrics:
    frame_count: int = 0
    dropped_frames: int = 0
    started_at: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    face_observations: int = 0
    accepted_faces: int = 0
    small_faces: int = 0
    unassociated_faces: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    association_counts: dict[str, int] = field(default_factory=dict)
    face_size_counts: dict[str, int] = field(default_factory=dict)
    face_source_counts: dict[str, int] = field(default_factory=dict)
    match_stage_counts: dict[str, int] = field(default_factory=dict)
    stage_latencies_ms: dict[str, list[float]] = field(default_factory=dict)
    stage_call_counts: dict[str, int] = field(default_factory=dict)
    frame_width: int = 0
    frame_height: int = 0
    roi_calls: int = 0
    budget_skips: dict[str, int] = field(default_factory=dict)
    end_to_end_latencies_ms: list[float] = field(default_factory=list)
    camera_motion_px: list[float] = field(default_factory=list)
    blur_variances: list[float] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        elapsed = 0.0
        if not self.started_at:
            fps = 0.0
        else:
            import time

            elapsed = max(time.monotonic() - self.started_at, 1e-6)
            fps = self.frame_count / elapsed
        p95 = float(np.percentile(self.latencies_ms[-1000:], 95)) if self.latencies_ms else 0.0
        stage_p95_latency_ms = {
            stage: float(np.percentile(latencies[-1000:], 95))
            for stage, latencies in sorted(self.stage_latencies_ms.items())
            if latencies
        }
        end_to_end_p95 = (
            float(np.percentile(self.end_to_end_latencies_ms[-1000:], 95))
            if self.end_to_end_latencies_ms
            else 0.0
        )
        arrived = self.dropped_frames + self.frame_count
        drop_rate = self.dropped_frames / arrived if arrived else 0.0
        return {
            "processed_fps": fps,
            "p95_latency_ms": p95,
            "dropped_frames": self.dropped_frames,
            "face_observations": self.face_observations,
            "accepted_faces": self.accepted_faces,
            "small_faces": self.small_faces,
            "unassociated_faces": self.unassociated_faces,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "association_counts": dict(sorted(self.association_counts.items())),
            "face_size_counts": dict(sorted(self.face_size_counts.items())),
            "face_source_counts": dict(sorted(self.face_source_counts.items())),
            "match_stage_counts": dict(sorted(self.match_stage_counts.items())),
            "stage_p95_latency_ms": stage_p95_latency_ms,
            "effective_hz": {
                stage: count / elapsed if elapsed else 0.0
                for stage, count in sorted(self.stage_call_counts.items())
            },
            "source_fps": arrived / elapsed if elapsed else 0.0,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "roi_calls_per_frame": (
                self.roi_calls / self.frame_count if self.frame_count else 0.0
            ),
            "drop_rate": drop_rate,
            "end_to_end_p95_latency_ms": end_to_end_p95,
            "budget_skips": dict(sorted(self.budget_skips.items())),
            "camera_motion_px_p95": _percentile(self.camera_motion_px, 95),
            "blur_variance_p50": _percentile(self.blur_variances, 50),
            "blur_variance_p95": _percentile(self.blur_variances, 95),
        }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values[-1000:], percentile)) if values else 0.0
