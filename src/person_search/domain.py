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
    last_rejection_reason: str | None = None


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
    embedding: np.ndarray
    quality: float
    landmarks: np.ndarray | None = None
    accepted: bool = True
    rejection_reasons: tuple[str, ...] = ()

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
        }
