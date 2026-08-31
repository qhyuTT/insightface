from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self
from urllib.parse import urlsplit

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

METRICS_SAMPLE_LIMIT = 1000


class BoundedFloatSeries(list[float]):
    """A list-compatible bounded ring buffer for high-frequency diagnostics.

    The project historically exposed these samples as lists (and a few callers
    compare them directly with list literals), so a small list subclass preserves
    that API while preventing a long-running search from retaining every frame.
    ``maxlen`` is intentionally fixed for now; changing it would alter percentile
    semantics between workers and should be a versioned configuration decision.
    """

    maxlen = METRICS_SAMPLE_LIMIT

    def __init__(self, values: Any = ()) -> None:
        super().__init__(values)
        self._trim()

    def _trim(self) -> None:
        if len(self) > self.maxlen:
            del self[: len(self) - self.maxlen]

    def append(self, value: float) -> None:
        super().append(value)
        self._trim()

    def extend(self, values: Any) -> None:
        super().extend(values)
        self._trim()

    def insert(self, index: int, value: float) -> None:
        super().insert(index, value)
        self._trim()

    def __setitem__(self, key: int | slice, value: Any) -> None:
        super().__setitem__(key, value)
        self._trim()

    def __imul__(self, count: int) -> Self:
        super().__imul__(count)
        self._trim()
        return self

    def __iadd__(self, values: Any) -> Self:
        self.extend(values)
        return self


class BoundedStageSeries(dict[str, BoundedFloatSeries]):
    """Coerce stage samples assigned by legacy callers into bounded series."""

    def __init__(self, values: Any = ()) -> None:
        super().__init__()
        if hasattr(values, "items"):
            for key, samples in values.items():
                self[key] = samples
        else:
            for key, samples in values:
                self[key] = samples

    @staticmethod
    def _coerce(values: Any) -> BoundedFloatSeries:
        return values if isinstance(values, BoundedFloatSeries) else BoundedFloatSeries(values)

    def __setitem__(self, key: str, values: Any) -> None:
        super().__setitem__(key, self._coerce(values))

    def setdefault(self, key: str, default: Any = ()) -> BoundedFloatSeries:
        if key not in self:
            self[key] = default
        return super().__getitem__(key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        incoming = dict(*args, **kwargs)
        for key, values in incoming.items():
            self[key] = values

    def __ior__(self, other: Any) -> Self:
        self.update(other)
        return self


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
    # Source values cross a process boundary and are later handed to OpenCV/FFmpeg.
    # Reject unknown keys and normalize only surrounding whitespace so a typo or an
    # ambiguous source cannot silently select a different capture path.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: SourceType
    # Keep credentials out of accidental object repr/logging; SearchView applies
    # its own explicit sanitizer before exposing this field over the API.
    uri: str | None = Field(default=None, max_length=4096, repr=False)
    device_index: int | None = Field(default=None, ge=0)
    debug_preview: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> SourceConfig:
        # Treat an explicitly empty URI as absent.  This gives callers one stable
        # error for ``null`` and ``""`` while retaining the public field shape.
        if self.uri is not None and not self.uri:
            self.uri = None

        if self.type in (SourceType.RTSP, SourceType.FILE):
            if not self.uri:
                raise ValueError(f"uri is required for {self.type.value} sources")
            if self.device_index is not None:
                raise ValueError("device_index is only valid for camera sources")
            if self.type == SourceType.RTSP:
                self.uri = _validate_rtsp_uri(self.uri)
            else:
                self.uri = _validate_file_uri(self.uri)
        elif self.type == SourceType.CAMERA:
            if self.device_index is None:
                raise ValueError("device_index is required for camera sources")
            if self.uri is not None:
                raise ValueError("uri is only valid for rtsp and file sources")
        return self


_RTSP_SCHEMES = frozenset({"rtsp", "rtsps"})
_URI_MAX_LENGTH = 4096
_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f]")
_ENCODED_CONTROL = re.compile(r"%(?:0[0-9a-f]|1[0-9a-f]|7f)", re.IGNORECASE)
_HOST_CHARS = re.compile(r"^[A-Za-z0-9._:%-]+$")


def _validate_rtsp_uri(uri: str) -> str:
    """Validate and return an RTSP URI without exposing credentials in errors.

    ``urllib.parse.urlsplit`` is deliberately used only for syntax; it accepts
    malformed ports and host names lazily, so both ``hostname`` and ``port`` are
    touched inside guarded blocks below.  Authentication is allowed (cameras
    commonly require it), but whitespace, fragments and invalid authorities are
    rejected before FFmpeg sees them.
    """
    value = uri.strip()
    if (
        not value
        or len(value) > _URI_MAX_LENGTH
        or _CONTROL_OR_SPACE.search(value)
        or _ENCODED_CONTROL.search(value)
    ):
        raise ValueError("RTSP URI must be a non-empty URI without whitespace")
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        raise ValueError("RTSP URI has an invalid authority or port") from exc
    if parts.scheme.lower() not in _RTSP_SCHEMES:
        raise ValueError("RTSP URI must use rtsp:// or rtsps://")
    if not parts.netloc:
        raise ValueError("RTSP URI must include a hostname")
    try:
        hostname = parts.hostname
    except ValueError as exc:
        raise ValueError("RTSP URI has an invalid hostname") from exc
    if not hostname or "%" in hostname or not _HOST_CHARS.fullmatch(hostname):
        raise ValueError("RTSP URI has an invalid hostname")
    try:
        port = parts.port
    except ValueError as exc:
        # This covers non-numeric and out-of-range ports.  Do not include the
        # original URI in the public validation message (it may contain a secret).
        raise ValueError("RTSP URI has an invalid port") from exc
    # A raw ``@`` is only a userinfo delimiter.  More than one delimiter makes
    # the authority ambiguous (different URI consumers disagree on which part is
    # credentials versus host), and can turn a typo into a connection to a
    # different host.  Credentials containing ``@`` must use ``%40`` instead.
    if "@" in parts.netloc:
        userinfo, authority = parts.netloc.rsplit("@", 1)
        if "@" in userinfo:
            raise ValueError("RTSP URI has invalid credentials")
    else:
        authority = parts.netloc
    if authority.endswith(":"):
        raise ValueError("RTSP URI has an invalid port")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("RTSP URI has an invalid port")
    if parts.fragment:
        raise ValueError("RTSP URI must not contain a fragment")
    # ``urlsplit`` lower-cases neither the scheme nor the hostname in the input;
    # preserving the caller's spelling keeps source display backwards compatible.
    return value


def _validate_file_uri(uri: str) -> str:
    """Validate a local file path or ``file://`` URI.

    The evaluator accepts ordinary relative/absolute paths.  A URI with another
    scheme is almost certainly a mistaken network source and is rejected here;
    existence is intentionally not checked until the video reader opens it.
    """
    value = uri.strip()
    if _ENCODED_CONTROL.search(value):
        raise ValueError("file URI must not contain percent-encoded control characters")
    if not value or len(value) > _URI_MAX_LENGTH or _CONTROL_OR_SPACE.search(value):
        raise ValueError("file URI must be a non-empty path without whitespace")
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        raise ValueError("file URI is malformed") from exc
    if parts.scheme:
        if parts.scheme.lower() != "file":
            raise ValueError("file source URI must use a local path or file://")
        if parts.fragment:
            raise ValueError("file URI must not contain a fragment")
        if parts.query:
            raise ValueError("file URI must not contain a query")
        # RFC 8089's local form allows an empty authority or ``localhost`` only;
        # credentials and ports are not meaningful for a local file and retaining
        # them can leak secrets into diagnostics or make different readers open
        # different paths.  Touch ``port`` explicitly because ``urlsplit`` defers
        # malformed-port errors until that property is read.
        try:
            parsed_port = parts.port
        except ValueError as exc:
            raise ValueError("file URI has an invalid authority or port") from exc
        if parts.username is not None or parts.password is not None:
            raise ValueError("file URI must not contain credentials")
        authority = parts.netloc
        if parsed_port is not None or authority.endswith(":"):
            raise ValueError("file URI must not contain a port")
        # Do not permit a remote file host: this source is opened by the local
        # process and accepting one would make the meaning platform-dependent.
        try:
            hostname = parts.hostname
        except ValueError as exc:
            raise ValueError("file URI has an invalid hostname") from exc
        if hostname not in (None, "", "localhost"):
            raise ValueError("file URI must refer to a local host")
        if not parts.path:
            raise ValueError("file URI must include a path")
        return value
    # A plain path may contain percent signs, but a URI-looking ``//host/path``
    # would be a network share and is rejected for the same reason as remote file
    # URIs.  Absolute and relative local paths remain supported.
    if value.startswith("//"):
        raise ValueError("file path must be local")
    return value


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


class ConfirmedSearchResult(BaseModel):
    """A confirmed event retained as the HTTP reconciliation source of truth."""

    search_id: str
    target_id: str
    target_name: str = "目标"
    state: MatchState = MatchState.CONFIRMED
    timestamp_ms: int
    track_id: int
    bbox: tuple[float, float, float, float]
    face_bbox: tuple[float, float, float, float] | None = None
    similarity: float
    quality: float
    evidence_count: int
    model: str
    embedding_contract_id: str | None = None
    association: str = "person_strict"
    evidence_id: str | None = None
    evidence_expires_at_ms: int | None = None
    evidence_available: bool = False


class SearchView(BaseModel):
    search_id: str
    target_id: str | None = None
    target_name: str = "目标"
    status: SearchStatus
    source: SourceConfig
    provider: str | None = None
    embedding_contract_id: str | None = None
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
    # Performance-budget diagnostics.  These are additive response fields so old
    # clients remain valid while the optimizer can distinguish batching from
    # deliberate frame-level degradation.
    roi_batch_count: int = Field(default=0, ge=0)
    embedding_batch_count: int = Field(default=0, ge=0)
    faces_dropped_by_budget: int = Field(default=0, ge=0)
    embedding_failures: int = Field(default=0, ge=0)
    embedding_output_failures: int = Field(default=0, ge=0)
    budget_skips: dict[str, int] = Field(default_factory=dict)
    effective_config: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    targets: list[TargetSearchView] = Field(default_factory=list)
    found_count: int = 0
    total_count: int = 0
    unfound_target_ids: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    request_id: str | None = None
    # Confirmed payloads are retained with the session so HTTP reconciliation can
    # recover a hit even when the websocket event was missed.
    confirmed_results: list[ConfirmedSearchResult] = Field(default_factory=list)


class TargetView(BaseModel):
    target_id: str
    name: str = "目标"
    face_width: int
    face_height: int
    detection_score: float
    quality_score: float
    model: str
    embedding_contract_id: str | None = None


class SearchEvent(BaseModel):
    search_id: str
    target_id: str
    target_name: str = "目标"
    state: MatchState
    timestamp_ms: int
    track_id: int
    bbox: tuple[float, float, float, float]
    # ``bbox`` remains the tracked person box for backwards compatibility;
    # ``face_bbox`` is the detector box used for the evidence crop.
    face_bbox: tuple[float, float, float, float] | None = None
    similarity: float
    quality: float
    evidence_count: int
    model: str
    embedding_contract_id: str | None = None
    association: str = "person_strict"
    # Opaque, short-lived reference to evidence held only by this executor. It
    # is intentionally not an image URL and has no meaning after the task ends.
    evidence_id: str | None = None
    evidence_expires_at_ms: int | None = None
    evidence_available: bool | None = None


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
    # ``None`` is used only by a terminal SearchSession metadata snapshot after
    # its biometric buffers have been released.  Manager-owned, live targets are
    # always populated; keeping the optional type makes that lifecycle boundary
    # explicit instead of retaining a zeroed ndarray that still pins memory.
    embedding: np.ndarray | None
    view: TargetView
    name: str = "目标"
    embedding_contract: EmbeddingContract | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingContract:
    schema_version: str
    model_name: str
    model_sha256: str
    embedding_dimension: int
    input_size: tuple[int, int]
    flip_tta: bool

    @property
    def contract_id(self) -> str:
        flip = 1 if self.flip_tta else 0
        width, height = self.input_size
        return (
            f"{self.schema_version}:{self.model_name}:{self.model_sha256[:16]}:"
            f"d{self.embedding_dimension}:{width}x{height}:flip{flip}"
        )


class EmbeddingContractView(BaseModel):
    schema_version: str
    model_name: str
    model_sha256: str
    embedding_dimension: int
    input_size: tuple[int, int]
    flip_tta: bool
    contract_id: str

    @classmethod
    def from_contract(cls, contract: EmbeddingContract) -> EmbeddingContractView:
        return cls(
            schema_version=contract.schema_version,
            model_name=contract.model_name,
            model_sha256=contract.model_sha256,
            embedding_dimension=contract.embedding_dimension,
            input_size=contract.input_size,
            flip_tta=contract.flip_tta,
            contract_id=contract.contract_id,
        )


@dataclass(slots=True)
class SearchMetrics:
    frame_count: int = 0
    dropped_frames: int = 0
    started_at: float = 0.0
    latencies_ms: BoundedFloatSeries = field(default_factory=BoundedFloatSeries)
    face_observations: int = 0
    accepted_faces: int = 0
    small_faces: int = 0
    unassociated_faces: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    association_counts: dict[str, int] = field(default_factory=dict)
    face_size_counts: dict[str, int] = field(default_factory=dict)
    face_source_counts: dict[str, int] = field(default_factory=dict)
    match_stage_counts: dict[str, int] = field(default_factory=dict)
    stage_latencies_ms: BoundedStageSeries = field(default_factory=BoundedStageSeries)
    stage_call_counts: dict[str, int] = field(default_factory=dict)
    frame_width: int = 0
    frame_height: int = 0
    roi_calls: int = 0
    budget_skips: dict[str, int] = field(default_factory=dict)
    end_to_end_latencies_ms: BoundedFloatSeries = field(default_factory=BoundedFloatSeries)
    camera_motion_px: BoundedFloatSeries = field(default_factory=BoundedFloatSeries)
    blur_variances: BoundedFloatSeries = field(default_factory=BoundedFloatSeries)
    # Per-track outcomes, recorded once per track. These are the only numbers that
    # can separate "the person was never sampled often enough" from "the samples
    # never scored high enough" --- the two failure modes look identical on every
    # other counter, because a track's state is dropped the moment it goes away.
    time_to_confirm_seconds: BoundedFloatSeries = field(default_factory=BoundedFloatSeries)
    track_dwell_seconds: BoundedFloatSeries = field(default_factory=BoundedFloatSeries)
    track_sampling_hz: BoundedFloatSeries = field(default_factory=BoundedFloatSeries)
    roi_batch_count: int = 0
    embedding_batch_count: int = 0
    faces_dropped_by_budget: int = 0
    embedding_failures: int = 0
    embedding_output_failures: int = 0

    def __post_init__(self) -> None:
        """Wrap constructor-provided lists while preserving the old dataclass API."""
        for name in (
            "latencies_ms",
            "end_to_end_latencies_ms",
            "camera_motion_px",
            "blur_variances",
            "time_to_confirm_seconds",
            "track_dwell_seconds",
            "track_sampling_hz",
        ):
            values = getattr(self, name)
            if not isinstance(values, BoundedFloatSeries):
                setattr(self, name, BoundedFloatSeries(values))
        if not isinstance(self.stage_latencies_ms, BoundedStageSeries):
            self.stage_latencies_ms = BoundedStageSeries(self.stage_latencies_ms)
    unconfirmed_gate_counts: dict[str, int] = field(default_factory=dict)

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
            "roi_batch_count": self.roi_batch_count,
            "embedding_batch_count": self.embedding_batch_count,
            "faces_dropped_by_budget": self.faces_dropped_by_budget,
            "embedding_failures": self.embedding_failures,
            "embedding_output_failures": self.embedding_output_failures,
            "time_to_confirm_p50_seconds": _percentile(self.time_to_confirm_seconds, 50),
            "time_to_confirm_p95_seconds": _percentile(self.time_to_confirm_seconds, 95),
            "track_dwell_p50_seconds": _percentile(self.track_dwell_seconds, 50),
            # The sampling rate tracks actually achieved. Compare against
            # effective_config.required_sampling_hz: below it, the evidence quorum
            # cannot be met inside the window however good the faces are.
            "achieved_sampling_hz": _percentile(self.track_sampling_hz, 50),
            "confirmed_tracks": len(self.time_to_confirm_seconds),
            "unconfirmed_gate_counts": dict(sorted(self.unconfirmed_gate_counts.items())),
        }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values[-1000:], percentile)) if values else 0.0
