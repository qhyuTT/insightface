from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from .backends import FaceBackend, InsightFaceBackend
from .config import Settings
from .confirmation import (
    FaceMatchPolicy,
    TrackConfirmation,
    TrackOutcome,
    associate_faces_to_tracks_detailed,
    default_face_match_policy,
    fallback_face_match_policy,
    is_stricter_policy,
    normalize_bbox,
)
from .detector import PersonDetector, YoloXOnnxDetector
from .domain import (
    FaceObservation,
    MatchState,
    SearchEvent,
    SearchMetrics,
    SearchStatus,
    SearchView,
    SourceConfig,
    SourceType,
    Target,
    TargetSearchView,
    TargetView,
    Track,
)
from .errors import (
    EnrollmentError,
    ModelUnavailableError,
    PersonSearchError,
    SearchStopTimeoutError,
)
from .face_tracking import FaceTracker
from .quality import normalize_embedding
from .tracker import ByteTracker
from .video import LatestFrameReader

MAX_TARGET_NAME_LENGTH = 80
STOP_WAIT_SECONDS = 15.0
# Global motion is estimated on a downscaled grayscale frame: a couple of
# milliseconds, and translation of a whole scene survives the downsample intact.
MOTION_ESTIMATE_WIDTH = 320
# phaseCorrelate reports its peak strength. A weak peak means the two frames were
# not a translation of each other (a cut, a reconnect, a fast rotation), and a
# fabricated shift is worse for association than no shift at all.
MOTION_MIN_RESPONSE = 0.05


class EventHub:
    def __init__(self, capacity: int = 256):
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._condition = threading.Condition()
        self._seq = 0

    def publish(self, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            self._seq += 1
            envelope = {
                "schema_version": "1",
                "seq": self._seq,
                "event_id": str(uuid.uuid4()),
                "type": event_type,
                "occurred_at": int(time.time() * 1000),
                "data": data,
            }
            self._events.append(envelope)
            self._condition.notify_all()
            return envelope

    def after(self, seq: int, timeout: float = 1.0) -> list[dict[str, Any]]:
        with self._condition:
            if not any(item["seq"] > seq for item in self._events):
                self._condition.wait(timeout=timeout)
            return [item.copy() for item in self._events if item["seq"] > seq]


class PreviewHub:
    """Keeps only the newest annotated JPEG so preview clients never delay inference."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._seq = 0
        self._jpeg: bytes | None = None
        self._subscribers = 0

    def publish(self, jpeg: bytes) -> None:
        with self._condition:
            self._seq += 1
            self._jpeg = jpeg
            self._condition.notify_all()

    def after(self, seq: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        with self._condition:
            if self._seq <= seq:
                self._condition.wait(timeout=timeout)
            return self._seq, self._jpeg if self._seq > seq else None

    def subscribe(self) -> None:
        """Register a viewer. A subscriber is a stream lifetime, not a single poll."""
        with self._condition:
            self._subscribers += 1

    def unsubscribe(self) -> None:
        with self._condition:
            self._subscribers = max(0, self._subscribers - 1)

    @property
    def has_subscribers(self) -> bool:
        """Whether anyone is watching. Nobody watching means no encode cost."""
        with self._condition:
            return self._subscribers > 0


@dataclass(frozen=True, slots=True)
class _EvidenceItem:
    """Encoded evidence that is never persisted or included in event payloads."""

    frame_jpeg: bytes
    face_crop_jpeg: bytes
    expires_at: float
    expires_at_ms: int = 0


@dataclass
class SearchSession:
    search_id: str
    target: Target | None
    source: SourceConfig
    settings: Settings
    face_backend: FaceBackend
    person_detector: PersonDetector
    on_finished: Callable[[str, list[str]], None]
    targets: list[Target] | None = None
    timeout_seconds: float | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        if self.targets is None:
            self.targets = [self.target] if self.target is not None else []
        if not self.targets:
            raise ValueError("at least one target is required")
        if self.target is None:
            self.target = self.targets[0]
        self.status = SearchStatus.INITIALIZING
        self.error: str | None = None
        self.metrics = SearchMetrics()
        self.events = EventHub()
        self.preview = PreviewHub()
        self._tracker = ByteTracker()
        self._face_tracker = FaceTracker(
            iou_threshold=self.settings.face_track_iou_threshold,
            buffer_seconds=self.settings.face_track_buffer_seconds,
        )
        self._confirmations = {
            target.target_id: TrackConfirmation(self.settings) for target in self.targets
        }
        # The identity gallery is immutable for the lifetime of the session. Found
        # targets leave only the active set so they remain negative competitors for
        # every target that is still searching.
        self._identity_targets = {target.target_id: target for target in self.targets}
        self._active_targets = dict(self._identity_targets)
        self._target_status = {
            target.target_id: {
                "status": "searching",
                "found_at": None,
                "best_similarity": None,
                "best_observed_similarity": None,
                "last_face_px": None,
                "evidence_count": 0,
                "required_evidence": 0,
                "qualifying_evidence": 0,
                "window_similarity": None,
                "window_statistic": None,
                "required_similarity": None,
                "aggregate_similarity": None,
                "required_aggregate_similarity": None,
                "tier": None,
                "last_rejection_reason": None,
                "last_rejection_face_px": None,
            }
            for target in self.targets
        }
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._reader: LatestFrameReader | None = None
        self._lock = threading.RLock()
        self._track_states: dict[int, tuple[str, float]] = {}
        self._shadow_tracks: set[int] = set()
        # Consecutive empty ROI passes per track, and how many passes that track is
        # currently backed off for.
        self._roi_misses: dict[int, int] = {}
        self._roi_skips: dict[int, int] = {}
        # The size tier each live track is judged by. Held here rather than per
        # target because the tier follows the observation, and every target has to
        # resolve the hysteresis margin against the same answer.
        self._track_tiers: dict[int, str] = {}
        # Loop credit for opportunistic stages. Cheap frames bank headroom, an
        # expensive optional pass spends it. See _roi_fits_budget.
        self._budget_credit = 0.0
        self._motion_hanning: np.ndarray | None = None
        self._debug_faces: list[tuple[FaceObservation, str, float | None]] = []
        self._finished = threading.Event()
        self._stop_requested = False
        self._deferred_events: list[tuple[str, dict[str, Any]]] = []
        self._evidence: dict[str, _EvidenceItem] = {}
        # Keep the metadata after the bytes are claimed/released so HTTP
        # reconciliation can still explain why a crop is unavailable.
        self._confirmed_results: list[dict[str, Any]] = []
        # Explicitly acknowledged ids are retained for one TTL window so DELETE
        # can be retried safely without allowing tombstones to grow forever.
        self._released_evidence: dict[str, float] = {}
        # Ids whose bytes a lifecycle wipe destroyed before any consumer claimed
        # them.  Kept apart from ``_released_evidence`` so a stop/timeout/failure
        # is never reported as a successful downstream hand-off.
        self._discarded_evidence: set[str] = set()
        self._evidence_cleanup_timer: threading.Timer | None = None
        self._evidence_cleanup_generation = 0

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._run, name=f"search-{self.search_id[:8]}", daemon=True
        )
        self._worker.start()

    @property
    def finished(self) -> threading.Event:
        return self._finished

    def stop(self, timeout: float = STOP_WAIT_SECONDS) -> None:
        self._stop.set()
        self._stop_requested = True
        # An explicit stop is a caller-requested abort, so its evidence must not
        # outlive the search.  Automatic completion follows a different path and
        # intentionally keeps the bytes until their TTL or an acknowledgement.
        self.clear_evidence()
        self._transition(SearchStatus.STOPPING, None)
        if self._reader:
            self._reader.stop()
        if (
            self._worker
            and self._worker is not threading.current_thread()
            and not self._finished.wait(timeout=timeout)
        ):
            raise SearchStopTimeoutError("搜索线程未能在停止时限内退出；请稍后重试或重启识别服务")

    def get_evidence(self, evidence_id: str, variant: str) -> tuple[bytes, str]:
        """Return a single short-lived image held exclusively in process memory."""
        with self._lock:
            self._cleanup_expired_evidence_locked()
            item = self._evidence.get(evidence_id)
            if item is None or item.expires_at <= time.monotonic() or self._stop.is_set():
                self._evidence.pop(evidence_id, None)
                self._raise_missing_evidence_locked(evidence_id)
            if variant == "frame":
                return item.frame_jpeg, "image/jpeg"
            if variant == "face_crop":
                return item.face_crop_jpeg, "image/jpeg"
        raise PersonSearchError(
            "evidence variant must be frame or face_crop",
            code="invalid_evidence_variant",
            status_code=422,
        )

    def release_evidence(self, evidence_id: str) -> None:
        """Release one evidence item after a downstream consumer has persisted it.

        The operation is idempotent for a recently released id, which lets a
        caller safely retry its acknowledgement after a lost HTTP response.
        Expired and unknown ids remain a 404 so callers can distinguish a late
        acknowledgement from a successful claim.
        """
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired_evidence_locked(now)
            item = self._evidence.pop(evidence_id, None)
            if item is not None:
                self._released_evidence[evidence_id] = item.expires_at
                self._set_evidence_available_locked(evidence_id, False)
                self._schedule_evidence_cleanup_locked()
                return
            released_until = self._released_evidence.get(evidence_id)
            if released_until is not None and released_until > now:
                return
            self._released_evidence.pop(evidence_id, None)
            self._raise_missing_evidence_locked(evidence_id)

    def _raise_missing_evidence_locked(self, evidence_id: str) -> None:
        if evidence_id in self._discarded_evidence:
            # 410: the bytes are gone for good, so a caller must stop retrying.
            # A distinct code keeps "the search ended" apart from "you released
            # it" and from "you were too slow".
            raise PersonSearchError(
                "evidence discarded with the search",
                code="evidence_discarded",
                status_code=410,
            )
        result = next(
            (
                item
                for item in self._confirmed_results
                if item.get("evidence_id") == evidence_id
            ),
            None,
        )
        if result is not None:
            expires_at_ms = result.get("evidence_expires_at_ms")
            if isinstance(expires_at_ms, int) and expires_at_ms <= int(time.time() * 1000):
                raise PersonSearchError(
                    "evidence expired", code="evidence_expired", status_code=410
                )
            raise PersonSearchError(
                "evidence not found (released)", code="evidence_released", status_code=404
            )
        raise PersonSearchError("evidence not found", code="evidence_not_found", status_code=404)

    def clear_evidence(self) -> None:
        with self._lock:
            # Anything still held was destroyed without being claimed; remember
            # which ids so a late GET/DELETE can say so instead of implying the
            # consumer already took delivery.
            self._discarded_evidence.update(self._evidence)
            self._evidence.clear()
            self._released_evidence.clear()
            self._evidence_cleanup_generation += 1
            timer = self._evidence_cleanup_timer
            self._evidence_cleanup_timer = None
            if timer is not None:
                timer.cancel()
            for result in self._confirmed_results:
                if result.get("evidence_id"):
                    result["evidence_available"] = False

    def cleanup_expired_evidence(self) -> int:
        """Drop expired JPEGs and return the number removed.

        This is safe to call from the deadline timer as well as from request
        paths. Metadata is intentionally retained in ``confirmed_results``.
        """
        with self._lock:
            removed = self._cleanup_expired_evidence_locked()
            if self._evidence_cleanup_timer is None and (
                self._evidence or self._released_evidence
            ):
                self._schedule_evidence_cleanup_locked()
            return removed

    def _cleanup_expired_evidence_locked(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        expired = [eid for eid, item in self._evidence.items() if item.expires_at <= now]
        for evidence_id in expired:
            self._evidence.pop(evidence_id, None)
            self._set_evidence_available_locked(evidence_id, False)
        stale_released = [
            evidence_id
            for evidence_id, released_until in self._released_evidence.items()
            if released_until <= now
        ]
        for evidence_id in stale_released:
            self._released_evidence.pop(evidence_id, None)
        return len(expired)

    def _set_evidence_available_locked(self, evidence_id: str, available: bool) -> None:
        for result in self._confirmed_results:
            if result.get("evidence_id") == evidence_id:
                result["evidence_available"] = available

    def _refresh_evidence_availability(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Re-check evidence liveness at publish time.

        ``target_found`` is built when the track confirms but published at the
        terminal transition, and a stop, timeout or failure wipes the bytes in
        between.  Publishing the confirmation-time flag would send the control
        plane after a crop that no longer exists, costing it a retry budget and
        a misleading error.
        """
        evidence_id = payload.get("evidence_id")
        if not evidence_id:
            return payload
        with self._lock:
            available = self._evidence_is_available_locked(str(evidence_id))
        if available == payload.get("evidence_available"):
            return payload
        return {**payload, "evidence_available": available}

    def _evidence_is_available_locked(self, evidence_id: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        item = self._evidence.get(evidence_id)
        return item is not None and item.expires_at > now and not self._stop.is_set()

    def _store_evidence(self, frame: np.ndarray, bbox: np.ndarray) -> str | None:
        """JPEG-encode the confirmation frame and face crop without touching disk."""
        x1, y1, x2, y2 = (round(value) for value in bbox)
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        frame_ok, frame_jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        crop_ok, crop_jpeg = cv2.imencode(
            ".jpg", frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not frame_ok or not crop_ok:
            return None
        evidence_id = str(uuid.uuid4())
        now_mono = time.monotonic()
        expires_at = now_mono + self.settings.evidence_ttl_seconds
        expires_at_ms = int((time.time() + self.settings.evidence_ttl_seconds) * 1000)
        with self._lock:
            self._evidence[evidence_id] = _EvidenceItem(
                frame_jpeg=frame_jpeg.tobytes(),
                face_crop_jpeg=crop_jpeg.tobytes(),
                expires_at=expires_at,
                expires_at_ms=expires_at_ms,
            )
            self._schedule_evidence_cleanup_locked()
        return evidence_id

    def _schedule_evidence_cleanup_locked(self) -> None:
        self._evidence_cleanup_generation += 1
        generation = self._evidence_cleanup_generation
        timer = self._evidence_cleanup_timer
        if timer is not None:
            timer.cancel()
        deadlines = [item.expires_at for item in self._evidence.values()]
        deadlines.extend(self._released_evidence.values())
        if not deadlines:
            self._evidence_cleanup_timer = None
            return
        delay = max(0.001, min(deadlines) - time.monotonic())
        timer = threading.Timer(delay, self._run_evidence_cleanup, args=(generation,))
        timer.daemon = True
        self._evidence_cleanup_timer = timer
        timer.start()

    def _run_evidence_cleanup(self, generation: int) -> None:
        with self._lock:
            if generation != self._evidence_cleanup_generation:
                return
            self._evidence_cleanup_timer = None
            self._cleanup_expired_evidence_locked()
            self._schedule_evidence_cleanup_locked()

    def view(self) -> SearchView:
        with self._lock:
            self._cleanup_expired_evidence_locked()
            metrics = self.metrics.snapshot()
            target_views = [
                TargetSearchView(
                    target_id=target.target_id,
                    name=target.name,
                    **self._target_status[target.target_id],
                )
                for target in self.targets
            ]
            found_count = sum(item.status == "found" for item in target_views)
            return SearchView(
                search_id=self.search_id,
                target_id=self.target.target_id,
                target_name=self.target.name,
                status=self.status,
                source=_sanitize_source(self.source),
                provider=f"face={self.face_backend.provider_name},person={self.person_detector.provider_name}",
                error=self.error,
                targets=target_views,
                found_count=found_count,
                total_count=len(target_views),
                unfound_target_ids=[
                    item.target_id for item in target_views if item.status != "found"
                ],
                timeout_seconds=self.timeout_seconds,
                request_id=self.request_id,
                effective_config=self._effective_config(),
                **metrics,
                confirmed_results=[
                    {
                        **result,
                        "evidence_available": (
                            self._evidence_is_available_locked(str(result["evidence_id"]))
                            if result.get("evidence_id")
                            else False
                        ),
                    }
                    for result in self._confirmed_results
                ],
            )

    def _effective_config(self) -> dict[str, object]:
        """Report the gates that are *actually* in force, not the raw settings.

        The CPU/CUDA rate split is resolved at runtime, so reading `Settings` alone
        cannot tell you what the pipeline is doing. This surfaces the resolved
        values so "what are my recognition conditions?" has a direct answer.
        """
        settings = self.settings
        face_provider = getattr(
            self.face_backend, "detection_provider_name", self.face_backend.provider_name
        )
        face_is_cuda = "CUDA" in face_provider
        person_is_cuda = "CUDA" in self.person_detector.provider_name
        return {
            "person_detection_hz": (
                settings.person_detection_hz_cuda
                if person_is_cuda
                else settings.person_detection_hz_cpu
            ),
            "face_detection_hz": (
                settings.face_detection_hz_cuda if face_is_cuda else settings.face_detection_hz_cpu
            ),
            "roi_face_detection_hz": (
                settings.roi_face_detection_hz_cuda
                if face_is_cuda
                else settings.roi_face_detection_hz_cpu
            ),
            "target_loop_hz": settings.target_loop_hz,
            "min_processed_fps": settings.min_processed_fps,
            "preview_hz": settings.preview_hz,
            "face_detection_size": settings.face_detection_size or "auto(128+640)",
            "full_frame_detection_scales": list(
                settings.full_frame_detection_scales(is_cuda=face_is_cuda)
            ),
            "face_deep_scan_every_n": settings.face_deep_scan_every_n,
            "roi_face_detection_size": settings.roi_face_detection_size,
            "roi_face_detection_max_size": settings.roi_face_detection_max_size,
            "roi_max_tracks_per_pass": settings.roi_max_tracks_per_pass,
            "match_profile": settings.match_profile,
            "evidence_statistic": settings.evidence_statistic,
            "evidence_top_k": settings.evidence_top_k,
            "face_tier_hysteresis_px": settings.face_tier_hysteresis_px,
            "camera_motion_compensation": settings.camera_motion_compensation,
            "embedding_flip_tta": settings.embedding_flip_tta,
            "similarity_threshold": settings.similarity_threshold,
            "small_face_similarity_threshold": settings.small_face_similarity_threshold,
            "tiny_face_enabled": settings.tiny_face_enabled,
            "tiny_face_shadow_mode": settings.tiny_face_shadow_mode,
            "tiny_face_similarity_threshold": settings.tiny_face_similarity_threshold,
            "tiny_face_aggregate_similarity_threshold": (
                settings.tiny_face_aggregate_similarity_threshold
            ),
            "tiny_face_detection_threshold": settings.tiny_face_detection_threshold,
            "tiny_face_evidence_required": settings.tiny_face_evidence_required,
            "tiny_face_evidence_window_seconds": settings.tiny_face_evidence_window_seconds,
            "tiny_face_consistent_votes_required": settings.tiny_face_consistent_votes_required,
            "tiny_face_allow_relaxed_association": settings.tiny_face_allow_relaxed_association,
            "effective_search_min_face_px": settings.effective_search_min_face_px,
            "min_search_face_px": settings.min_search_face_px,
            "preferred_search_face_px": settings.preferred_search_face_px,
            "min_search_blur_variance": settings.min_search_blur_variance,
            "face_detection_threshold": settings.face_detection_threshold,
            "evidence_required": settings.evidence_required,
            "evidence_window_seconds": settings.evidence_window_seconds,
            "evidence_min_interval_seconds": settings.evidence_min_interval_seconds,
            "tiny_face_evidence_min_interval_seconds": (
                settings.tiny_face_evidence_min_interval_seconds
            ),
            "departure_adjudication_enabled": settings.departure_adjudication_enabled,
            "departure_min_samples": settings.departure_min_samples,
            "departure_similarity_margin": settings.departure_similarity_margin,
            "small_face_evidence_required": settings.small_face_evidence_required,
            "small_face_evidence_window_seconds": settings.small_face_evidence_window_seconds,
            # The sampling rate the confirmation window implies. Below this, the
            # evidence quorum cannot be met inside the window.
            "required_sampling_hz": (
                settings.evidence_required / settings.evidence_window_seconds
            ),
            "small_face_required_sampling_hz": (
                settings.small_face_evidence_required
                / settings.small_face_evidence_window_seconds
            ),
            "tiny_face_required_sampling_hz": (
                settings.tiny_face_evidence_required
                / settings.tiny_face_evidence_window_seconds
            ),
        }

    def _run(self) -> None:
        self.metrics.started_at = time.monotonic()
        reader = LatestFrameReader(
            self.source,
            self.settings,
            on_status=self._transition,
            on_drop=self._on_drop,
        )
        self._reader = reader
        tracks: list[Track] = []
        last_person_at = -1e9
        last_face_at = -1e9
        last_roi_face_at = -1e9
        last_preview_at = -1e9
        face_detection_provider = getattr(
            self.face_backend, "detection_provider_name", self.face_backend.provider_name
        )
        face_is_cuda = "CUDA" in face_detection_provider
        person_is_cuda = "CUDA" in self.person_detector.provider_name
        person_hz = (
            self.settings.person_detection_hz_cuda
            if person_is_cuda
            else self.settings.person_detection_hz_cpu
        )
        face_hz = (
            self.settings.face_detection_hz_cuda
            if face_is_cuda
            else self.settings.face_detection_hz_cpu
        )
        roi_face_hz = (
            self.settings.roi_face_detection_hz_cuda
            if face_is_cuda
            else self.settings.roi_face_detection_hz_cpu
        )
        shallow_scales = self.settings.full_frame_detection_scales(
            is_cuda=face_is_cuda, deep=False
        )
        deep_scales = self.settings.full_frame_detection_scales(is_cuda=face_is_cuda)
        deep_scan_every_n = self.settings.face_deep_scan_every_n
        face_pass_index = 0
        previous_motion_gray: np.ndarray | None = None
        try:
            reader.start()
            while not self._stop.is_set():
                if (
                    self.timeout_seconds is not None
                    and time.monotonic() - self.metrics.started_at >= self.timeout_seconds
                ):
                    self._transition(SearchStatus.TIMED_OUT, None, publish=False)
                    break
                packet = reader.get(timeout=0.5)
                if packet is None:
                    if reader.ended.is_set():
                        break
                    continue
                started = time.monotonic()
                now = packet.captured_at
                with self._lock:
                    self.metrics.frame_height, self.metrics.frame_width = packet.frame.shape[:2]
                if now - last_person_at >= 1.0 / max(person_hz, 0.1):
                    stage_started = time.monotonic()
                    detections = self.person_detector.detect(packet.frame)
                    motion, previous_motion_gray = self._estimate_camera_motion(
                        packet.frame, previous_motion_gray
                    )
                    tracks = self._tracker.update(detections, motion=motion)
                    self._record_stage("person", stage_started)
                    last_person_at = now
                faces = []
                if now - last_face_at >= 1.0 / max(face_hz, 0.1):
                    deep = face_pass_index % deep_scan_every_n == 0
                    face_pass_index += 1
                    stage_started = time.monotonic()
                    faces = self.face_backend.detect_faces(
                        packet.frame,
                        enrollment=False,
                        detection_size=deep_scales if deep else shallow_scales,
                    )
                    # face_full always covers every pass, which is the cost the
                    # budget must reason about; face_full_deep isolates the large
                    # scale so its price is visible when tuning deep_scan_every_n.
                    self._record_stage("face_full", stage_started)
                    if deep:
                        self._record_stage("face_full_deep", stage_started)
                    self._record_face_source("full_frame", len(faces))
                    last_face_at = now
                    roi_clock = time.monotonic()
                    roi_tracks = self._tracks_needing_roi_face_pass(faces, tracks)
                    if (
                        roi_face_hz > 0
                        and roi_clock - last_roi_face_at >= 1.0 / roi_face_hz
                        and roi_tracks
                        and self._roi_fits_budget(started)
                    ):
                        stage_started = time.monotonic()
                        roi_faces = self._analyze_person_rois(packet.frame, roi_tracks)
                        self._record_stage("face_roi", stage_started)
                        self._record_face_source("roi", len(roi_faces))
                        faces = _merge_faces(faces, roi_faces)
                        last_roi_face_at = roi_clock
                    self._record_face_metrics(faces)

                # ArcFace runs once, here, and only on faces that survived dedup and
                # the quality gate. Detections thrown away by _merge_faces or
                # _is_face_matchable never cost an embedding.
                matchable = [
                    face for face in faces if SearchSession._is_face_matchable(self, face)
                ]
                if matchable:
                    stage_started = time.monotonic()
                    accepted_faces = self.face_backend.embed_faces(packet.frame, matchable)
                    self._record_stage("face_embed", stage_started)
                else:
                    accepted_faces = []
                self._record_target_observations(accepted_faces)
                self._record_rejected_observations(
                    [face for face in faces if not SearchSession._is_face_matchable(self, face)]
                )
                detailed = associate_faces_to_tracks_detailed(accepted_faces, tracks)
                association_by_face = {
                    face_index: track_id for face_index, (track_id, _) in detailed.items()
                }
                modes_by_face = {face_index: mode for face_index, (_, mode) in detailed.items()}
                all_tracks = list(tracks)
                fallback_policy = fallback_face_match_policy(self.settings)
                # The tier is a property of the observation, not of one target, so
                # the session owns it: every target must judge a given track by the
                # same size tier, and the hysteresis margin has to be resolved once.
                policies_by_face: dict[int, FaceMatchPolicy] = {
                    face_index: default_face_match_policy(
                        face,
                        self.settings,
                        self._tier_of_associated_track(face_index, association_by_face),
                    )
                    for face_index, face in enumerate(accepted_faces)
                }
                for face_index, mode in list(modes_by_face.items()):
                    policy = policies_by_face[face_index]
                    if mode == "person_relaxed" and not policy.allows_relaxed_association:
                        association_by_face.pop(face_index, None)
                        modes_by_face.pop(face_index, None)
                        continue
                    if mode == "person_relaxed" and is_stricter_policy(fallback_policy, policy):
                        # A relaxed association is weaker body evidence, so it pulls
                        # the face up to the small-face bar -- but only when that bar
                        # is actually higher. Applying it unconditionally would have
                        # relaxed the far tier instead of tightening it.
                        policies_by_face[face_index] = fallback_policy
                unassociated_indices = [
                    face_index
                    for face_index, face in enumerate(accepted_faces)
                    if face_index not in association_by_face
                    and not policies_by_face[face_index].requires_strict_association
                    and face.detection_score >= self.settings.fallback_face_detection_threshold
                ]
                if self.settings.face_fallback_enabled and unassociated_indices:
                    fallback_faces = [accepted_faces[index] for index in unassociated_indices]
                    fallback_tracks = self._face_tracker.update(fallback_faces, now)
                    all_tracks.extend(track for track in fallback_tracks if track is not None)
                    for fallback_index, fallback_track in enumerate(fallback_tracks):
                        if fallback_track is None:
                            continue
                        original_index = unassociated_indices[fallback_index]
                        association_by_face[original_index] = fallback_track.track_id
                        modes_by_face[original_index] = "face_fallback"
                        policies_by_face[original_index] = fallback_policy
                with self._lock:
                    self.metrics.unassociated_faces += sum(
                        face_index not in association_by_face
                        for face_index in range(len(accepted_faces))
                    )
                    for mode in modes_by_face.values():
                        self.metrics.association_counts[mode] = (
                            self.metrics.association_counts.get(mode, 0) + 1
                        )
                    self.metrics.match_stage_counts["associated"] = (
                        self.metrics.match_stage_counts.get("associated", 0)
                        + len(association_by_face)
                    )
                for face_index, track_id in association_by_face.items():
                    self._track_tiers[track_id] = policies_by_face[face_index].tier
                live_track_ids = {track.track_id for track in all_tracks}
                self._track_tiers = {
                    key: value
                    for key, value in self._track_tiers.items()
                    if key in live_track_ids
                }

                self._debug_faces = []
                faces_by_target: dict[str, list[tuple[int, FaceObservation]]] = {
                    target_id: [] for target_id in self._active_targets
                }
                for face_index, face in enumerate(accepted_faces):
                    if not self._active_targets:
                        break
                    ranked_matches = self._rank_identity_matches(face)
                    if not ranked_matches:
                        continue
                    target_id, similarity = ranked_matches[0]
                    top1_margin = (
                        similarity - ranked_matches[1][1]
                        if len(ranked_matches) > 1
                        else float("inf")
                    )
                    mode = modes_by_face.get(face_index, "unassociated")
                    self._debug_faces.append((face, mode, similarity))
                    policy = policies_by_face[face_index]
                    if similarity >= policy.threshold:
                        with self._lock:
                            self.metrics.match_stage_counts["above_threshold"] = (
                                self.metrics.match_stage_counts.get("above_threshold", 0) + 1
                            )
                    if top1_margin < policy.min_top1_margin:
                        with self._lock:
                            self.metrics.match_stage_counts["ambiguous_identity"] = (
                                self.metrics.match_stage_counts.get("ambiguous_identity", 0) + 1
                            )
                            # The runner-up lost this face to the ambiguity too, so it
                            # must see the reason instead of a silent diagnostic gap.
                            for ambiguous_id, _ in ranked_matches[:2]:
                                if ambiguous_id in self._active_targets:
                                    self._target_status[ambiguous_id]["last_rejection_reason"] = (
                                        "identity_margin_low"
                                    )
                        continue
                    if face_index not in association_by_face:
                        with self._lock:
                            if target_id in self._active_targets:
                                self._target_status[target_id]["last_rejection_reason"] = (
                                    "unassociated"
                                )
                        continue
                    if target_id not in self._active_targets:
                        with self._lock:
                            self.metrics.match_stage_counts["inactive_identity_top1"] = (
                                self.metrics.match_stage_counts.get("inactive_identity_top1", 0) + 1
                            )
                        continue
                    if not policy.accepts_observation(face.detection_score, similarity):
                        with self._lock:
                            self.metrics.match_stage_counts["evidence_policy_rejected"] = (
                                self.metrics.match_stage_counts.get("evidence_policy_rejected", 0)
                                + 1
                            )
                            reason = (
                                "detection_score_low"
                                if face.detection_score < policy.min_detection_score
                                else "similarity_low"
                            )
                            self._target_status[target_id]["last_rejection_reason"] = reason
                        continue
                    faces_by_target[target_id].append((face_index, face))
                    with self._lock:
                        self.metrics.match_stage_counts["evidence_eligible"] = (
                            self.metrics.match_stage_counts.get("evidence_eligible", 0) + 1
                        )
                        self._target_status[target_id]["last_rejection_reason"] = (
                            None if similarity >= policy.threshold else "similarity_low"
                        )

                track_states: dict[int, tuple[str, float]] = {}
                for target_id, target in list(self._active_targets.items()):
                    target_faces = faces_by_target.get(target_id, [])
                    local_faces = [face for _, face in target_faces]
                    local_associations = {
                        local_index: association_by_face[global_index]
                        for local_index, (global_index, _) in enumerate(target_faces)
                    }
                    local_modes = {
                        local_index: modes_by_face[global_index]
                        for local_index, (global_index, _) in enumerate(target_faces)
                    }
                    local_policies = {
                        local_index: policies_by_face[global_index]
                        for local_index, (global_index, _) in enumerate(target_faces)
                    }
                    confirmation = self._confirmations[target_id]
                    confirmation_result = confirmation.process_with_stats(
                        frame_id=packet.frame_id,
                        timestamp=now,
                        frame_shape=packet.frame.shape,
                        tracks=all_tracks,
                        faces=local_faces,
                        target=target,
                        associations=local_associations,
                        association_modes=local_modes,
                        face_policies=local_policies,
                    )
                    decisions = confirmation_result.decisions
                    with self._lock:
                        self.metrics.match_stage_counts["evidence_collected"] = (
                            self.metrics.match_stage_counts.get("evidence_collected", 0)
                            + confirmation_result.evidence_collected
                        )
                    self._record_track_outcomes(confirmation_result.outcomes)
                    self._handle_decisions(target_id, target, decisions, packet.frame)
                    progress = confirmation.track_progress(target)
                    if progress:
                        # Rank by qualifying samples, not banked ones: under
                        # collect_all_observations a track sits permanently at
                        # observed == required while contributing nothing.
                        best = max(
                            progress.values(),
                            key=lambda item: (item.qualifying, item.window_similarity or -1.0),
                        )
                        with self._lock:
                            current = self._target_status[target_id]
                            current["evidence_count"] = best.observed
                            current["required_evidence"] = best.required
                            current["qualifying_evidence"] = best.qualifying
                            current["window_similarity"] = best.window_similarity
                            current["window_statistic"] = best.window_statistic
                            current["required_similarity"] = best.threshold
                            current["aggregate_similarity"] = best.aggregate_similarity
                            current["required_aggregate_similarity"] = best.aggregate_threshold
                            current["tier"] = best.tier
                    else:
                        with self._lock:
                            current = self._target_status[target_id]
                            current["evidence_count"] = 0
                            current["qualifying_evidence"] = 0
                            current["window_similarity"] = None
                            current["aggregate_similarity"] = None
                    for track_id, (state, similarity) in confirmation.active_track_states().items():
                        state_value = (
                            "shadow"
                            if track_id in self._shadow_tracks and state.value == "confirmed"
                            else state.value
                        )
                        previous = track_states.get(track_id)
                        if (
                            previous is None
                            or state_value in {"confirmed", "shadow"}
                            or previous[0] != "confirmed"
                        ):
                            track_states[track_id] = (state_value, similarity)
                self._track_states = track_states
                if not self._active_targets:
                    self._transition(SearchStatus.COMPLETED, None, publish=False)
                    break
                preview_hz = self.settings.preview_hz
                preview_clock = time.monotonic()
                if (
                    preview_hz > 0
                    and preview_clock - last_preview_at >= 1.0 / preview_hz
                    and self._publish_preview(packet.frame, all_tracks, faces)
                ):
                    last_preview_at = preview_clock
                frame_cost = time.monotonic() - started
                self._settle_budget_credit(frame_cost)
                with self._lock:
                    self.metrics.frame_count += 1
                    self.metrics.latencies_ms.append(frame_cost * 1000.0)
                    self.metrics.end_to_end_latencies_ms.append(
                        (time.monotonic() - packet.captured_at) * 1000.0
                    )
        except Exception as exc:  # noqa: BLE001 - the worker must fail closed and release resources
            self._transition(SearchStatus.FAILED, _safe_error(exc), publish=False)
        finally:
            reader.stop()
            if self._stop_requested or self.status not in (
                SearchStatus.FAILED,
                SearchStatus.STOPPED,
                SearchStatus.COMPLETED,
                SearchStatus.TIMED_OUT,
            ):
                self._transition(SearchStatus.STOPPED, None, publish=False)
            try:
                self.on_finished(self.search_id, [target.target_id for target in self.targets])
            finally:
                self._publish_terminal_event()
                self._finished.set()

    def _publish_preview(
        self, frame: np.ndarray, tracks: list[Track], faces: list[FaceObservation]
    ) -> bool:
        """Encode an annotated preview. Returns False when the work was skipped.

        Nobody watching means no copy and no JPEG encode — this runs on the worker
        thread, so an unwatched preview was pure inference tax.
        """
        if not self.preview.has_subscribers:
            return False
        # Downscale before annotating: a 1440p copy + q82 encode per frame costs
        # more than some inference stages.
        height, width = frame.shape[:2]
        scale = min(1.0, self.settings.preview_max_width / max(width, 1))
        if scale < 1.0:
            canvas = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            canvas = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = (int(value * scale) for value in track.bbox)
            state, similarity = self._track_states.get(track.track_id, ("tracking", 0.0))
            if state == "confirmed":
                color, label = (60, 220, 95), f"FOUND  {similarity:.2f}"
            elif state == "shadow":
                color, label = (210, 110, 245), f"SHADOW  {similarity:.2f}"
            elif state == "candidate":
                color, label = (0, 184, 255), f"CANDIDATE  {similarity:.2f}"
            else:
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            top = max(0, y1 - text_height - 12)
            cv2.rectangle(canvas, (x1, top), (x1 + text_width + 14, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 7, max(text_height + 2, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (10, 18, 20),
                2,
                cv2.LINE_AA,
            )
        for face in faces:
            x1, y1, x2, y2 = (int(value * scale) for value in face.bbox)
            color = (232, 232, 232) if face.accepted else (70, 70, 230)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
            if self.source.debug_preview:
                debug = next((item for item in self._debug_faces if item[0] is face), None)
                reasons = "ok" if face.accepted else ",".join(face.rejection_reasons)
                mode = debug[1] if debug else "rejected"
                similarity = "" if debug is None or debug[2] is None else f" sim={debug[2]:.2f}"
                label = f"{face.short_side}px {reasons} {mode}{similarity}"
                cv2.putText(
                    canvas,
                    label,
                    (max(0, x1), max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            self.preview.publish(encoded.tobytes())
        return True

    def _handle_decisions(
        self,
        target_id: str,
        target: Target | None,
        decisions: list,
        frame: np.ndarray | tuple[int, ...],
    ) -> None:
        if target is None:
            return
        frame_shape = frame.shape if isinstance(frame, np.ndarray) else frame
        for decision in decisions:
            event = SearchEvent(
                search_id=self.search_id,
                target_id=target.target_id,
                target_name=target.name,
                state=decision.state,
                timestamp_ms=int(time.time() * 1000),
                track_id=decision.track_id,
                bbox=normalize_bbox(decision.bbox, frame_shape),
                face_bbox=(
                    None
                    if decision.face_bbox is None
                    else normalize_bbox(decision.face_bbox, frame_shape)
                ),
                similarity=decision.similarity,
                quality=decision.quality,
                evidence_count=decision.evidence_count,
                model=self.face_backend.model_name,
                association=decision.association,
            )
            payload = event.model_dump(mode="json", exclude_none=True)
            if (
                not decision.shadow
                and decision.state.value == "confirmed"
                and isinstance(frame, np.ndarray)
                and decision.face_bbox is not None
                and bool(self.settings.evidence_api_key)
            ):
                # The person track box is intentionally retained in the public
                # ``bbox`` field.  Crop from the detector's actual face box so a
                # full-body track cannot produce a misleading giant crop.
                try:
                    evidence_id = self._store_evidence(frame, decision.face_bbox)
                except Exception:  # noqa: BLE001 - optional evidence must not undo a hit
                    # JPEG encoding is an auxiliary hand-off. A malformed box,
                    # OpenCV failure, or memory pressure must leave the confirmed
                    # recognition intact and surface only as unavailable evidence.
                    evidence_id = None
                if evidence_id is not None:
                    payload["evidence_id"] = evidence_id
                    with self._lock:
                        item = self._evidence.get(evidence_id)
                    if item is not None:
                        payload["evidence_expires_at_ms"] = item.expires_at_ms
            if not decision.shadow and decision.state.value == "confirmed":
                payload["evidence_available"] = bool(payload.get("evidence_id"))
                # Store an immutable-ish dict snapshot.  SearchView validates it
                # through ConfirmedSearchResult, while the availability bit is
                # refreshed on every view after TTL/release cleanup.
                with self._lock:
                    self._confirmed_results.append(
                        {
                            **payload,
                            "state": MatchState.CONFIRMED.value,
                        }
                    )
            if decision.shadow:
                event_type = f"tiny_shadow_{decision.state.value}"
                payload["state"] = event_type.removeprefix("tiny_")
            else:
                event_type = decision.state.value
            self.events.publish(event_type, payload)
            with self._lock:
                current = self._target_status[target_id]
                current["best_similarity"] = max(
                    decision.similarity, current["best_similarity"] or -1.0
                )
                if decision.shadow and decision.state.value == "confirmed":
                    self._shadow_tracks.add(decision.track_id)
                    current["evidence_count"] = decision.evidence_count
                    current["required_evidence"] = decision.evidence_count
                    current["last_rejection_reason"] = "shadow_only"
                    self.metrics.match_stage_counts["shadow_confirmed"] = (
                        self.metrics.match_stage_counts.get("shadow_confirmed", 0) + 1
                    )
                elif decision.shadow and decision.state.value == "lost":
                    self._shadow_tracks.discard(decision.track_id)
                    current["evidence_count"] = 0
                    current["last_rejection_reason"] = "shadow_lost"
                elif decision.state.value == "confirmed":
                    self._shadow_tracks.discard(decision.track_id)
                    current["status"] = "found"
                    current["found_at"] = payload["timestamp_ms"]
                    current["evidence_count"] = decision.evidence_count
                    current["required_evidence"] = decision.evidence_count
                    self._deferred_events.append(("target_found", payload))
                    self._active_targets.pop(target_id, None)
                    self.metrics.match_stage_counts["confirmed"] = (
                        self.metrics.match_stage_counts.get("confirmed", 0) + 1
                    )

    def _record_face_metrics(self, faces: list[FaceObservation]) -> None:
        with self._lock:
            self.metrics.face_observations += len(faces)
            self.metrics.accepted_faces += sum(
                SearchSession._is_face_matchable(self, face) for face in faces
            )
            self.metrics.small_faces += sum(
                SearchSession._is_face_matchable(self, face)
                and face.short_side < self.settings.preferred_search_face_px
                for face in faces
            )
            for face in faces:
                bucket = _face_size_bucket(face.short_side)
                self.metrics.face_size_counts[bucket] = (
                    self.metrics.face_size_counts.get(bucket, 0) + 1
                )
                self.metrics.blur_variances.append(face.blur_variance)
                for reason in face.rejection_reasons:
                    self.metrics.rejection_counts[reason] = (
                        self.metrics.rejection_counts.get(reason, 0) + 1
                    )
            self.metrics.match_stage_counts["detected"] = self.metrics.match_stage_counts.get(
                "detected", 0
            ) + len(faces)
            self.metrics.match_stage_counts["quality_accepted"] = (
                self.metrics.match_stage_counts.get("quality_accepted", 0)
                + sum(SearchSession._is_face_matchable(self, face) for face in faces)
            )

    def _record_face_source(self, source: str, count: int) -> None:
        with self._lock:
            self.metrics.face_source_counts[source] = (
                self.metrics.face_source_counts.get(source, 0) + count
            )

    def _record_stage(self, stage: str, started: float) -> None:
        latency_ms = (time.monotonic() - started) * 1000.0
        with self._lock:
            self.metrics.stage_latencies_ms.setdefault(stage, []).append(latency_ms)
            self.metrics.stage_call_counts[stage] = self.metrics.stage_call_counts.get(stage, 0) + 1

    def _record_target_observations(self, faces: list[FaceObservation]) -> None:
        if not faces or not self._active_targets:
            return
        best_in_frame: dict[str, tuple[float, FaceObservation]] = {}
        for face in faces:
            for target_id, similarity in self._rank_identity_matches(face):
                if target_id not in self._active_targets:
                    continue
                previous = best_in_frame.get(target_id)
                if previous is None or similarity > previous[0]:
                    best_in_frame[target_id] = (similarity, face)
        with self._lock:
            for target_id, (similarity, face) in best_in_frame.items():
                current = self._target_status[target_id]
                previous_best = current["best_observed_similarity"]
                current["best_observed_similarity"] = max(
                    similarity, previous_best if previous_best is not None else -1.0
                )
                current["last_face_px"] = face.short_side
                if not face.accepted:
                    current["last_rejection_reason"] = ",".join(face.rejection_reasons)
                    current["last_rejection_face_px"] = face.short_side

    def _record_rejected_observations(self, faces: list[FaceObservation]) -> None:
        """Record why rejected faces were dropped, without paying for an embedding.

        Similarity is unknowable for these faces by design — they never reach
        ArcFace — but "a face this big was rejected for this reason" is the more
        actionable half of the diagnostic anyway.

        The reason and the size it belongs to are written together. They used to
        come from different observations: ``last_face_px`` tracked the largest face
        seen while the reason came from the largest *rejected* one, so the panel
        could read "49px / face_too_small" against a 48px floor and send the
        operator hunting for a bug in the size gate.
        """
        if not faces or not self._active_targets:
            return
        largest = max(faces, key=lambda face: face.short_side)
        with self._lock:
            for target_id in self._active_targets:
                current = self._target_status[target_id]
                if current["last_face_px"] is None or largest.short_side > current["last_face_px"]:
                    current["last_face_px"] = largest.short_side
                if largest.rejection_reasons:
                    current["last_rejection_reason"] = ",".join(largest.rejection_reasons)
                    current["last_rejection_face_px"] = largest.short_side

    def _rank_identity_matches(self, face: FaceObservation) -> list[tuple[str, float]]:
        """Rank against the immutable batch gallery, including found targets."""
        if face.embedding is None:
            return []
        return sorted(
            (
                (target_id, float(np.dot(target.embedding, face.embedding)))
                for target_id, target in self._identity_targets.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

    def _record_track_outcomes(self, outcomes: list[TrackOutcome]) -> None:
        """Fold one frame's per-track post-mortems into the metrics.

        ``track_sampling_hz`` needs a dwell to divide by, so a track confirmed on
        its very first banked sample contributes nothing to it. That is honest:
        a single sample says nothing about a rate, and inventing one would make
        the achieved-vs-required comparison read high exactly when it matters.
        """
        if not outcomes:
            return
        with self._lock:
            for outcome in outcomes:
                if outcome.time_to_confirm_seconds is not None:
                    self.metrics.time_to_confirm_seconds.append(outcome.time_to_confirm_seconds)
                self.metrics.track_dwell_seconds.append(outcome.dwell_seconds)
                if outcome.dwell_seconds > 0 and outcome.sampled > 1:
                    self.metrics.track_sampling_hz.append(
                        outcome.sampled / outcome.dwell_seconds
                    )
                if not outcome.confirmed and outcome.blocking_gate is not None:
                    self.metrics.unconfirmed_gate_counts[outcome.blocking_gate] = (
                        self.metrics.unconfirmed_gate_counts.get(outcome.blocking_gate, 0) + 1
                    )

    def _is_face_matchable(self, face: FaceObservation) -> bool:
        return bool(face.accepted and face.short_side >= self.settings.effective_search_min_face_px)

    def _tier_of_associated_track(
        self, face_index: int, association_by_face: dict[int, int]
    ) -> str | None:
        """Return the tier the face's track is currently judged by, if any."""
        track_id = association_by_face.get(face_index)
        return None if track_id is None else self._track_tiers.get(track_id)

    def _stage_p95_ms(self, stage: str) -> float:
        with self._lock:
            latencies = self.metrics.stage_latencies_ms.get(stage)
            if not latencies:
                return 0.0
            return float(np.percentile(latencies[-200:], 95))

    def _record_budget_skip(self, stage: str) -> None:
        with self._lock:
            self.metrics.budget_skips[stage] = self.metrics.budget_skips.get(stage, 0) + 1

    def _roi_fits_budget(self, started: float) -> bool:
        """Admit the ROI pass only when banked loop credit covers its measured cost.

        The previous check compared the stage p95 against what was left of a
        *single* ``target_loop_hz`` frame budget.  But ROI is only ever considered
        on the frame that just paid for full-frame face detection, whose mandatory
        cost already exceeds that budget, so the remainder was structurally
        negative: once the first pass recorded a p95 the stage was skipped forever
        and small faces lost the only stage that can recover them.

        A credit bucket refilled at ``target_loop_hz`` fixes the denominator.  A
        stage that grows slower than its refill rate drains the bucket and
        throttles itself down — that is the duty-cycle ceiling — instead of being
        starved to zero by one frame's arithmetic.
        """
        estimated = self._stage_p95_ms("face_roi") / 1000.0
        # The processed-fps floor is inviolable, whatever the credit says.
        if (time.monotonic() - started) + estimated >= 1.0 / self.settings.min_processed_fps:
            self._record_budget_skip("face_roi_floor")
            return False
        if estimated > self._budget_credit:
            self._record_budget_skip("face_roi_credit")
            return False
        return True

    def _settle_budget_credit(self, frame_cost_seconds: float) -> None:
        """Refill the loop credit for this frame and charge what the frame cost.

        Debt is capped as well as credit: one catastrophic pass must not starve
        the optional stage for an unbounded stretch afterwards.
        """
        refill = 1.0 / self.settings.target_loop_hz
        cap = refill * self.settings.budget_credit_max_frames
        self._budget_credit = float(
            np.clip(self._budget_credit + refill - frame_cost_seconds, -cap, cap)
        )

    def _estimate_camera_motion(
        self, frame: np.ndarray, previous_gray: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the global pixel shift since the last person pass, and the new state.

        A robot that pans moves every box in the frame at once, which pure-IoU
        association reads as "every track lost" -- and a fresh track id restarts the
        evidence window from zero, so a moving camera silently caps how much
        evidence a person can ever accumulate. Handing the shift to the tracker
        keeps one person on one id.

        Only translation is estimated. A fast on-the-spot rotation needs an affine
        fit; ``camera_motion_px_p95`` on the panel is what says whether that day
        has come.
        """
        if not self.settings.camera_motion_compensation:
            return None, None
        stage_started = time.monotonic()
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = MOTION_ESTIMATE_WIDTH / max(width, 1)
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (MOTION_ESTIMATE_WIDTH, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale = 1.0
        current = np.float32(gray)
        if previous_gray is None or previous_gray.shape != current.shape:
            return None, current
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            previous_gray, current, self._motion_window(current.shape)
        )
        self._record_stage("cmc", stage_started)
        if not np.isfinite(response) or response < MOTION_MIN_RESPONSE:
            return None, current
        # phaseCorrelate reports the shift that maps the previous frame onto the
        # current one, which is exactly what a track box needs added to it.
        motion = np.asarray([shift_x, shift_y], dtype=np.float32) / scale
        with self._lock:
            self.metrics.camera_motion_px.append(float(np.hypot(motion[0], motion[1])))
        return motion, current

    def _motion_window(self, shape: tuple[int, ...]) -> np.ndarray:
        """Return a cached Hanning window; without one, frame edges bias the peak."""
        if self._motion_hanning is None or self._motion_hanning.shape != shape:
            self._motion_hanning = cv2.createHanningWindow(
                (shape[1], shape[0]), cv2.CV_32F
            )
        return self._motion_hanning

    def _tracks_needing_roi_face_pass(
        self, faces: list[FaceObservation], tracks: list[Track]
    ) -> list[Track]:
        """Return tracks that do not own a preferred-size full-frame face.

        Tracks that keep yielding nothing, and tracks already confirmed, are
        excluded — neither can turn more ROI passes into new evidence.
        """
        if not tracks:
            return []
        accepted = [face for face in faces if SearchSession._is_face_matchable(self, face)]
        associations = associate_faces_to_tracks_detailed(accepted, tracks)
        satisfied_track_ids = {
            track_id
            for face_index, (track_id, _) in associations.items()
            if accepted[face_index].short_side >= self.settings.preferred_search_face_px
        }
        confirmed_track_ids = {
            track_id
            for track_id, (state, _) in self._track_states.items()
            if state in ("confirmed", "shadow")
        }
        candidates: list[Track] = []
        for track in tracks:
            if track.track_id in satisfied_track_ids or track.track_id in confirmed_track_ids:
                self._roi_misses.pop(track.track_id, None)
                self._roi_skips.pop(track.track_id, None)
                continue
            remaining_skips = self._roi_skips.get(track.track_id, 0)
            if remaining_skips > 0:
                self._roi_skips[track.track_id] = remaining_skips - 1
                continue
            candidates.append(track)
        live_ids = {track.track_id for track in tracks}
        self._roi_misses = {
            key: value for key, value in self._roi_misses.items() if key in live_ids
        }
        self._roi_skips = {key: value for key, value in self._roi_skips.items() if key in live_ids}
        return candidates

    def _analyze_person_rois(self, frame: np.ndarray, tracks: list[Track]) -> list[FaceObservation]:
        height, width = frame.shape[:2]
        ranked = sorted(tracks, key=lambda track: track.score, reverse=True)
        observations: list[FaceObservation] = []
        analyzed_tracks = 0
        for track in ranked:
            x1, y1, x2, y2 = track.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if y2 - y1 < self.settings.roi_min_person_height_px or x2 <= x1 or y2 <= y1:
                continue
            if analyzed_tracks >= self.settings.roi_max_tracks_per_pass:
                break
            analyzed_tracks += 1
            roi_bottom = min(y2, y1 + max(1, int(self.settings.roi_person_fraction * (y2 - y1))))
            roi = frame[y1:roi_bottom, x1:x2]
            # A crop is already tight, so the full-frame Auto dual-scale pass would
            # only double the cost here. Detection only — embedding happens once,
            # later, after dedup against the full-frame results. The scale keeps a
            # small crop upsampled while never shrinking a large one back below the
            # pixels the crop existed to preserve.
            found = self.face_backend.detect_faces(
                roi,
                enrollment=False,
                detection_size=self.settings.roi_detection_scale(roi.shape[1], roi.shape[0]),
            )
            self._note_roi_outcome(track.track_id, hit=bool(found))
            for face in found:
                observations.append(
                    replace(
                        face,
                        bbox=face.bbox + np.asarray([x1, y1, x1, y1], dtype=np.float32),
                        landmarks=(
                            None
                            if face.landmarks is None
                            else face.landmarks + np.asarray([x1, y1], dtype=np.float32)
                        ),
                    )
                )
        with self._lock:
            self.metrics.roi_calls += analyzed_tracks
        return observations

    def _note_roi_outcome(self, track_id: int, *, hit: bool) -> None:
        """Back a track off exponentially while its ROI crop keeps coming up empty."""
        if hit:
            self._roi_misses.pop(track_id, None)
            self._roi_skips.pop(track_id, None)
            return
        misses = self._roi_misses.get(track_id, 0) + 1
        self._roi_misses[track_id] = misses
        self._roi_skips[track_id] = min(2**misses, self.settings.roi_backoff_max_skips)

    def _on_drop(self) -> None:
        with self._lock:
            self.metrics.dropped_frames += 1

    def _transition(self, status: SearchStatus, error: str | None, *, publish: bool = True) -> None:
        with self._lock:
            if self.status in (
                SearchStatus.STOPPED,
                SearchStatus.FAILED,
                SearchStatus.COMPLETED,
                SearchStatus.TIMED_OUT,
            ):
                return
            if status == self.status and error == self.error:
                return
            self.status = status
            self.error = error
        if status in {
            SearchStatus.STOPPED,
            SearchStatus.FAILED,
            SearchStatus.TIMED_OUT,
        }:
            self.clear_evidence()
        if publish:
            self.events.publish(
                "search_status",
                {"search_id": self.search_id, "status": status.value, "error": error},
            )

    def _publish_terminal_event(self) -> None:
        status = self.status
        self.events.publish(
            "search_status",
            {"search_id": self.search_id, "status": status.value, "error": self.error},
        )
        for event_type, payload in self._deferred_events:
            self.events.publish(event_type, self._refresh_evidence_availability(payload))
        self._deferred_events.clear()
        if status == SearchStatus.COMPLETED:
            self.events.publish("all_found", {"search_id": self.search_id})
        elif status == SearchStatus.TIMED_OUT:
            self.events.publish(
                "search_timeout",
                {
                    "search_id": self.search_id,
                    "unfound_target_ids": list(self._active_targets),
                },
            )


class SearchManager:
    def __init__(
        self,
        settings: Settings,
        face_backend: FaceBackend | None = None,
        person_detector: PersonDetector | None = None,
    ):
        self.settings = settings
        self.face_backend = face_backend or InsightFaceBackend(settings)
        self.person_detector = person_detector or YoloXOnnxDetector(settings)
        self._targets: dict[str, Target] = {}
        self._sessions: dict[str, SearchSession] = {}
        self._active_search_id: str | None = None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()

    def enroll(self, image: np.ndarray, name: str = "目标") -> TargetView:
        target_name = _normalize_target_name(name)
        faces = self.face_backend.analyze(image, enrollment=True)
        if not faces:
            raise EnrollmentError("no face detected", code="no_face")
        if len(faces) > 1:
            raise EnrollmentError("exactly one face is required", code="multiple_faces")
        face = faces[0]
        if not face.accepted:
            reasons = ", ".join(face.rejection_reasons)
            raise EnrollmentError(f"face quality is too low: {reasons}", code="face_quality_low")
        try:
            embedding = normalize_embedding(face.embedding)
        except ValueError as exc:
            raise EnrollmentError(str(exc), code="invalid_embedding") from exc
        target_id = str(uuid.uuid4())
        width = int(face.bbox[2] - face.bbox[0])
        height = int(face.bbox[3] - face.bbox[1])
        view = TargetView(
            target_id=target_id,
            name=target_name,
            face_width=width,
            face_height=height,
            detection_score=face.detection_score,
            quality_score=face.quality,
            model=self.face_backend.model_name,
        )
        with self._lock:
            self._targets[target_id] = Target(
                target_id=target_id, embedding=embedding, view=view, name=target_name
            )
        return view

    def delete_target(self, target_id: str) -> bool:
        with self._lock:
            if self._active_search_id:
                session = self._sessions[self._active_search_id]
                if any(target.target_id == target_id for target in session.targets):
                    raise PersonSearchError(
                        "target is used by an active search", code="target_in_use", status_code=409
                    )
            return self._targets.pop(target_id, None) is not None

    def get_target(self, target_id: str) -> Target:
        with self._lock:
            target = self._targets.get(target_id)
        if target is None:
            raise PersonSearchError("target not found", code="target_not_found", status_code=404)
        return target

    def start_search(self, target_id: str, source: SourceConfig) -> SearchView:
        return self.start_batch_search([target_id], source)

    def start_batch_search(
        self,
        target_ids: list[str],
        source: SourceConfig,
        timeout_seconds: float | None = None,
        replace_active: bool = False,
        request_id: str | None = None,
    ) -> SearchView:
        if not target_ids:
            raise PersonSearchError(
                "at least one target is required", code="invalid_targets", status_code=422
            )
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise PersonSearchError(
                "timeout_seconds must be positive", code="invalid_timeout", status_code=422
            )
        with self._lifecycle_lock:
            with self._lock:
                active_id = self._active_search_id
                active_session = self._sessions.get(active_id) if active_id else None
                if request_id and active_session and active_session.request_id == request_id:
                    return active_session.view()
                targets: list[Target] = []
                for target_id in target_ids:
                    target = self._targets.get(target_id)
                    if target is None:
                        raise PersonSearchError(
                            "target not found", code="target_not_found", status_code=404
                        )
                    targets.append(target)
            if active_id is not None:
                if not replace_active:
                    raise PersonSearchError(
                        "only one search may run at a time",
                        code="search_capacity_exceeded",
                        status_code=409,
                    )
                self.stop_search(active_id)
            ensure_ready = getattr(self.person_detector, "ensure_ready", None)
            if ensure_ready:
                ensure_ready()
            search_id = str(uuid.uuid4())
            session = SearchSession(
                search_id=search_id,
                target=targets[0],
                source=source,
                settings=self.settings,
                face_backend=self.face_backend,
                person_detector=self.person_detector,
                on_finished=self._on_finished,
                targets=targets,
                timeout_seconds=timeout_seconds,
                request_id=request_id,
            )
            with self._lock:
                self._sessions[search_id] = session
                self._active_search_id = search_id
            try:
                session.start()
            except Exception:
                with self._lock:
                    self._sessions.pop(search_id, None)
                    if self._active_search_id == search_id:
                        self._active_search_id = None
                raise
            return session.view()

    def get_search(self, search_id: str) -> SearchView:
        return self._get_session(search_id).view()

    def get_session(self, search_id: str) -> SearchSession:
        return self._get_session(search_id)

    def search_by_request_id(self, request_id: str) -> SearchView | None:
        """Return the session associated with an idempotency key, if retained.

        Search sessions are kept in memory after they reach a terminal state so
        callers can still inspect their final view and event history.  This
        lookup lets a control-plane client reconcile a POST whose response was
        lost even when the search finished before the reconciliation request.
        No image data is persisted by this index; it only scans the existing
        in-memory session metadata.
        """
        normalized = request_id.strip()
        if not normalized:
            return None
        with self._lock:
            session = next(
                (
                    candidate
                    for candidate in self._sessions.values()
                    if candidate.request_id == normalized
                ),
                None,
            )
        return session.view() if session is not None else None

    def stop_search(self, search_id: str) -> None:
        with self._lifecycle_lock:
            session = self._get_session(search_id)
            if session.finished.is_set():
                # DELETE remains an explicit resource stop even when the worker
                # happened to auto-complete just before the request arrived.
                session.clear_evidence()
                return
            session.stop()

    def active_search(self) -> SearchView | None:
        with self._lock:
            active = self._active_search_id
            session = self._sessions.get(active) if active else None
        return session.view() if session else None

    def shutdown(self) -> None:
        with self._lock:
            active = self._active_search_id
        if active:
            try:
                self.stop_search(active)
            except SearchStopTimeoutError:
                # Keep the process alive until the worker can release its slot; the
                # caller will still see a clear timeout if this is an API stop.
                pass
        with self._lock:
            sessions = list(self._sessions.values())
            self._targets.clear()
        for session in sessions:
            session.clear_evidence()

    def _get_session(self, search_id: str) -> SearchSession:
        with self._lock:
            session = self._sessions.get(search_id)
        if session is None:
            raise PersonSearchError("search not found", code="search_not_found", status_code=404)
        return session

    def _on_finished(self, search_id: str, target_ids: list[str]) -> None:
        with self._lock:
            if self._active_search_id == search_id:
                self._active_search_id = None
            for target_id in target_ids:
                self._targets.pop(target_id, None)


def _sanitize_source(source: SourceConfig) -> SourceConfig:
    if source.type != SourceType.RTSP or not source.uri:
        return source.model_copy()
    parts = urlsplit(source.uri)
    host = parts.hostname or "source"
    port = f":{parts.port}" if parts.port else ""
    return source.model_copy(update={"uri": f"{parts.scheme}://{host}{port}/***"})


def _normalize_target_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise EnrollmentError("target name is required", code="invalid_target_name")
    if len(normalized) > MAX_TARGET_NAME_LENGTH:
        raise EnrollmentError(
            f"target name exceeds {MAX_TARGET_NAME_LENGTH} characters",
            code="invalid_target_name",
        )
    return normalized


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ModelUnavailableError):
        return exc.message
    return f"{type(exc).__name__}: processing failed"


def _merge_faces(
    primary: list[FaceObservation], secondary: list[FaceObservation], iou_threshold: float = 0.45
) -> list[FaceObservation]:
    """Merge full-frame and ROI face observations without double counting a face."""
    merged = list(primary)
    for candidate in secondary:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _bbox_iou(existing.bbox, candidate.bbox) >= iou_threshold
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
        elif _prefer_face(candidate, merged[duplicate_index]):
            merged[duplicate_index] = candidate
    return merged


def _prefer_face(candidate: FaceObservation, existing: FaceObservation) -> bool:
    if candidate.accepted != existing.accepted:
        return candidate.accepted
    return candidate.quality > existing.quality


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return float(intersection / max(area_first + area_second - intersection, 1e-6))


def _face_size_bucket(short_side: int) -> str:
    if short_side < 48:
        return "lt48"
    if short_side < 64:
        return "48_63"
    if short_side < 80:
        return "64_79"
    return "gte80"
