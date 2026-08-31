from __future__ import annotations

import logging
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
MAX_REQUEST_ID_LENGTH = 128
STOP_WAIT_SECONDS = 15.0
# Global motion is estimated on a downscaled grayscale frame: a couple of
# milliseconds, and translation of a whole scene survives the downsample intact.
MOTION_ESTIMATE_WIDTH = 320
# phaseCorrelate reports its peak strength. A weak peak means the two frames were
# not a translation of each other (a cut, a reconnect, a fast rotation), and a
# fabricated shift is worse for association than no shift at all.
MOTION_MIN_RESPONSE = 0.05
logger = logging.getLogger(__name__)


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
        events, _, _ = self.after_with_meta(seq, timeout=timeout)
        return events

    def after_with_meta(
        self, seq: int, timeout: float = 1.0
    ) -> tuple[list[dict[str, Any]], bool, int | None]:
        """Return events plus whether the requested cursor has fallen behind.

        The original ``after`` API is intentionally kept for callers that only
        need a list.  Replay-aware clients can use the metadata to reconcile a
        cursor that was evicted from the bounded history instead of silently
        assuming that the stream is complete.
        """
        with self._condition:
            if not any(item["seq"] > seq for item in self._events):
                self._condition.wait(timeout=timeout)
            oldest_seq = self._events[0]["seq"] if self._events else None
            gap = oldest_seq is not None and seq < oldest_seq - 1
            return (
                [item.copy() for item in self._events if item["seq"] > seq],
                gap,
                oldest_seq,
            )


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

    def clear(self) -> None:
        """Release the retained JPEG when a session reaches its terminal state."""
        with self._condition:
            self._jpeg = None
            self._seq += 1
            self._condition.notify_all()

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
            incoming_targets = [self.target] if self.target is not None else []
        else:
            incoming_targets = list(self.targets)
        if not incoming_targets:
            raise ValueError("at least one target is required")
        # A session owns its gallery.  SearchManager may still have the same
        # Target objects in its enrollment store while the worker is running;
        # copying here means terminal cleanup can release biometric arrays
        # without zeroing a manager/external reference that is about to be used
        # for another request.
        self.targets = [_clone_target(target) for target in incoming_targets]
        primary_id = self.target.target_id if self.target is not None else self.targets[0].target_id
        self.target = next(
            (target for target in self.targets if target.target_id == primary_id),
            self.targets[0],
        )
        self._target_metadata = tuple(
            (target.target_id, target.name) for target in self.targets
        )
        self._primary_target_id = self.target.target_id
        self._primary_target_name = self.target.name
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
        # A failed queue flush is retained for the next idempotent cleanup pass;
        # dropping the only reference after an exception would turn a transient
        # reader failure into a permanent buffer leak.
        self._reader_cleanup_pending: object | None = None
        # ``clear_sensitive_state`` may detach the reader while a worker is
        # still in its startup path (for example when shutdown races
        # ``LatestFrameReader.start``).  Keep a separate tombstone so the
        # worker cannot publish a fresh reader reference after cleanup has
        # already claimed the old one.
        self._reader_detached = False
        self._reader_stop_lock = threading.Lock()
        self._reader_stop_succeeded = False
        self._lock = threading.RLock()
        self._track_states: dict[int, tuple[str, float]] = {}
        self._shadow_tracks: set[int] = set()
        # Consecutive empty ROI passes per track, and how many passes that track is
        # currently backed off for.
        self._roi_misses: dict[int, int] = {}
        self._roi_skips: dict[int, int] = {}
        self._roi_last_pass: dict[int, int] = {}
        self._roi_schedule_counter = 0
        # The size tier each live track is judged by. Held here rather than per
        # target because the tier follows the observation, and every target has to
        # resolve the hysteresis margin against the same answer.
        self._track_tiers: dict[int, str] = {}
        # Incremented whenever the reader reports a source loss. The worker
        # consumes this generation and resets temporal state so track ids and
        # evidence never cross a reconnect boundary.
        self._source_epoch = 0
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
        self._sensitive_cleared = False
        # A stop/shutdown may time out while capture or inference is blocked.
        # Keep a single daemon watcher so the terminal cleanup is retried after
        # the worker finally exits (and so a transient cleanup failure does not
        # become a permanent biometric-buffer leak).
        self._sensitive_cleanup_watcher: threading.Thread | None = None
        self._finished_at: float | None = None

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._run, name=f"search-{self.search_id[:8]}", daemon=True
        )
        self._worker.start()

    def _stop_reader_once(self, reader: object) -> None:
        """Stop one reader at most once after a successful call.

        ``stop`` can be reached concurrently from an API request, the worker
        finalizer, and deferred sensitive cleanup. Serializing the call avoids
        duplicate joins (which can each wait several seconds) while retaining a
        failed call for a later retry.
        """
        with self._reader_stop_lock:
            with self._lock:
                if self._reader_stop_succeeded:
                    return
            stop_reader = getattr(reader, "stop", None)
            if stop_reader is None:
                with self._lock:
                    self._reader_stop_succeeded = True
                return
            stop_reader()
            with self._lock:
                self._reader_stop_succeeded = True

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
        with self._lock:
            reader = self._reader
        if reader is not None:
            try:
                self._stop_reader_once(reader)
            except Exception as exc:  # noqa: BLE001 - worker finalizer still owns terminal cleanup
                logger.warning("reader stop request failed: %s", type(exc).__name__)
        if (
            self._worker
            and self._worker is not threading.current_thread()
            and not self._finished.wait(timeout=timeout)
        ):
            self.defer_sensitive_cleanup()
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
        # A stop/timeout can leave the capture worker winding down for a few
        # seconds. Release the latest annotated frame immediately rather than
        # retaining it until that worker reaches its finalizer. Preview cleanup
        # is auxiliary; a broken viewer must never abort evidence cleanup.
        try:
            self.preview.clear()
        except Exception as exc:  # noqa: BLE001
            logger.warning("preview clear failed: %s", type(exc).__name__)

    def clear_sensitive_state(self) -> None:
        """Release biometric arrays while retaining a JSON-safe terminal view.

        Completed sessions stay queryable for request reconciliation and event
        replay.  They must not, however, keep the enrolled target embeddings or
        per-track ArcFace evidence alive for the entire process lifetime.
        """
        reader: LatestFrameReader | None = None
        worker_for_retry: object | None = None
        cleanup_complete = True

        def cleanup_failed(label: str, exc: Exception) -> None:
            nonlocal cleanup_complete
            cleanup_complete = False
            logger.warning("%s cleanup failed: %s", label, type(exc).__name__)

        with self._lock:
            if self._sensitive_cleared:
                return
            worker = getattr(self, "_worker", None)
            try:
                finished = self._finished.is_set()
            except Exception:  # noqa: BLE001 - legacy sessions may omit the event
                finished = False
            if (
                _worker_is_alive(worker)
                and worker is not threading.current_thread()
                and not finished
            ):
                # Never wipe a gallery while another thread may still be inside
                # provider inference. The caller's stop/shutdown path signals
                # the worker first; the watcher then retries after its terminal
                # event. A worker invoking this method from its own finalizer is
                # the one safe exception.
                try:
                    self.defer_sensitive_cleanup()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("sensitive cleanup deferral failed: %s", type(exc).__name__)
                return
            # Claim the reader slot before touching any other component.  The
            # worker may still be between construction and ``self._reader =``;
            # ``_reader_detached`` makes that startup path observe the claim and
            # stop its local reader instead of attaching a reference after this
            # cleanup pass.
            reader = (
                self._reader_cleanup_pending
                if self._reader_cleanup_pending is not None
                else self._reader
            )
            self._reader = None
            self._reader_cleanup_pending = None
            self._reader_detached = True
            targets_by_identity: dict[int, Target] = {}
            # Legacy/integration callers may hand a session a malformed target
            # container. Discover each source independently so one broken
            # dictionary/list cannot prevent the remaining buffers from being
            # wiped (or the reader/preview from being detached below).
            def read_attr(name: str, default: object, label: str) -> object:
                try:
                    return getattr(self, name, default)
                except Exception as exc:  # noqa: BLE001
                    cleanup_failed(label, exc)
                    return default

            def discover_targets(value: object, label: str) -> list[Target]:
                try:
                    if isinstance(value, dict):
                        return list(value.values())
                    return list(value or ())  # type: ignore[arg-type]
                except Exception as exc:  # noqa: BLE001
                    cleanup_failed(label, exc)
                    return []

            for label, value in (
                ("target discovery", read_attr("targets", [], "target discovery")),
                (
                    "identity target discovery",
                    read_attr("_identity_targets", {}, "identity target discovery"),
                ),
                (
                    "active target discovery",
                    read_attr("_active_targets", {}, "active target discovery"),
                ),
            ):
                for target in discover_targets(value, label):
                    try:
                        targets_by_identity[id(target)] = target
                    except Exception as exc:  # noqa: BLE001
                        cleanup_failed("target discovery", exc)
            try:
                primary_target = read_attr("target", None, "primary target discovery")
                if primary_target is not None:
                    targets_by_identity[id(primary_target)] = primary_target
            except Exception as exc:  # noqa: BLE001
                cleanup_failed("primary target discovery", exc)
            for target in targets_by_identity.values():
                try:
                    _wipe_array(target.embedding)
                except Exception as exc:  # noqa: BLE001 - defensive for legacy targets
                    cleanup_failed("target embedding", exc)

            try:
                confirmations = list(read_attr("_confirmations", {}, "confirmation discovery").values())  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                cleanup_failed("confirmation discovery", exc)
                confirmations = []
            for confirmation in confirmations:
                try:
                    confirmation.clear_sensitive()
                except Exception as exc:  # noqa: BLE001 - one stale track must not block the rest
                    cleanup_failed("confirmation", exc)

            # Keep scalar target/status metadata available for terminal views and
            # for callers that inspect a session after the worker exits, but
            # replace every gallery value with an embedding-free copy.  Clearing
            # the dictionaries outright made a completed batch look as if its
            # unfound targets had never existed and broke the long-standing
            # identity-competition diagnostics.
            def metadata_target(target: Target) -> Target:
                """Return an embedding-free copy even for a malformed legacy target."""
                try:
                    return _clone_target(target, include_embedding=False)
                except Exception as exc:  # noqa: BLE001 - retain scalar metadata if possible
                    cleanup_failed("target snapshot", exc)
                    try:
                        return Target(
                            target_id=target.target_id,
                            embedding=None,
                            view=target.view,
                            name=target.name,
                        )
                    except Exception as fallback_exc:  # noqa: BLE001
                        cleanup_failed("target snapshot fallback", fallback_exc)
                        # Keep a last-resort legacy object only after severing
                        # its embedding reference.  A failed snapshot must not
                        # leave a credential-bearing ndarray attached forever;
                        # the failed flag below permits a later retry.
                        try:
                            target.embedding = None
                        except Exception as detach_exc:  # noqa: BLE001
                            cleanup_failed("target embedding detach", detach_exc)
                        return target

            def metadata_gallery(value: object, label: str) -> dict[str, Target]:
                try:
                    items = list(value.items())  # type: ignore[union-attr]
                except Exception as exc:  # noqa: BLE001
                    cleanup_failed(label, exc)
                    return {}
                snapshot: dict[str, Target] = {}
                for target_id, target in items:
                    try:
                        snapshot[target_id] = metadata_target(target)
                    except Exception as exc:  # noqa: BLE001
                        cleanup_failed(label, exc)
                return snapshot

            try:
                self._identity_targets = metadata_gallery(
                    read_attr("_identity_targets", {}, "identity gallery snapshot"),
                    "identity gallery snapshot",
                )
            except Exception as exc:  # noqa: BLE001 - continue with other buffers
                cleanup_failed("identity gallery snapshot", exc)
            try:
                self._active_targets = metadata_gallery(
                    read_attr("_active_targets", {}, "active gallery snapshot"),
                    "active gallery snapshot",
                )
            except Exception as exc:  # noqa: BLE001
                cleanup_failed("active gallery snapshot", exc)
            try:
                self.targets = [
                    metadata_target(target)
                    for target in discover_targets(
                        read_attr("targets", [], "target list snapshot"),
                        "target list snapshot",
                    )
                ]
            except Exception as exc:  # noqa: BLE001
                cleanup_failed("target list snapshot", exc)
            try:
                primary_target = read_attr("target", None, "primary target snapshot")
                if primary_target is not None:
                    self.target = metadata_target(primary_target)
            except Exception as exc:  # noqa: BLE001
                cleanup_failed("primary target snapshot", exc)

            try:
                debug_faces = list(read_attr("_debug_faces", [], "debug face discovery"))
            except Exception as exc:  # noqa: BLE001 - malformed legacy debug storage
                cleanup_failed("debug face discovery", exc)
                debug_faces = []
            for item in debug_faces:
                try:
                    face, _, _ = item
                    _wipe_array(face.embedding)
                    _wipe_array(face.bbox)
                    _wipe_array(face.landmarks)
                except Exception as exc:  # noqa: BLE001
                    cleanup_failed("debug face", exc)
            try:
                self._debug_faces.clear()
            except Exception as exc:  # noqa: BLE001
                cleanup_failed("debug face storage", exc)
                try:
                    self._debug_faces = []
                except Exception as fallback_exc:  # noqa: BLE001
                    cleanup_failed("debug face storage fallback", fallback_exc)
            for label, attr in (
                ("deferred event storage", "_deferred_events"),
                ("ROI miss storage", "_roi_misses"),
                ("ROI skip storage", "_roi_skips"),
                ("track tier storage", "_track_tiers"),
            ):
                try:
                    value = read_attr(attr, {}, label)
                    value.clear()
                except Exception as exc:  # noqa: BLE001
                    cleanup_failed(label, exc)
                    try:
                        setattr(
                            self,
                            attr,
                            {} if attr.endswith(("misses", "skips", "tiers")) else [],
                        )
                    except Exception as fallback_exc:  # noqa: BLE001
                        cleanup_failed(f"{label} fallback", fallback_exc)
            self._roi_last_pass = {}
            self._roi_schedule_counter = 0
            # Tracker memories contain bbox/velocity arrays tied to decoded
            # frames. They are not biometric embeddings, but resetting both
            # trackers releases those frame references along the same lifecycle
            # boundary.
            try:
                read_attr("_tracker", None, "person tracker discovery").reset()  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                cleanup_failed("person tracker", exc)
            try:
                read_attr("_face_tracker", None, "face tracker discovery").reset()  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                cleanup_failed("face tracker", exc)
            self._budget_credit = 0.0
            self._motion_hanning = None
            try:
                self.source = _sanitize_source(read_attr("source", None, "source discovery"))  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001 - source must not retain credentials
                cleanup_failed("source sanitization", exc)
                # Source validation should make this unreachable, but lifecycle
                # cleanup must remain best-effort even for legacy sessions.
                try:
                    # ``model_copy(update=...)`` does not revalidate and a
                    # malformed legacy object could return the original URI.
                    # Use a freshly validated local-camera placeholder so no
                    # RTSP credentials (or an arbitrary file path) survive.
                    self.source = SourceConfig(type=SourceType.CAMERA, device_index=0)
                except Exception as fallback_exc:  # noqa: BLE001
                    cleanup_failed("source sanitization fallback", fallback_exc)
                    try:
                        # ``model_construct`` is deliberately the final escape
                        # hatch for a monkeypatched/legacy validator; its fields
                        # are explicit and contain no caller-controlled URI.
                        self.source = SourceConfig.model_construct(
                            type=SourceType.CAMERA,
                            uri=None,
                            device_index=0,
                            debug_preview=False,
                        )
                    except Exception as placeholder_exc:  # noqa: BLE001
                        cleanup_failed("source sanitization placeholder", placeholder_exc)
            worker_for_retry = self._worker

        if reader is not None:
            reader_cleanup_failed = False
            try:
                self._stop_reader_once(reader)
            except Exception as exc:  # noqa: BLE001
                cleanup_failed("reader stop", exc)
                reader_cleanup_failed = True
            try:
                clear_reader = getattr(reader, "clear", None)
                if clear_reader is not None:
                    clear_reader()
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                cleanup_failed("reader buffer", exc)
                reader_cleanup_failed = True
            if reader_cleanup_failed:
                with self._lock:
                    self._reader_cleanup_pending = reader
        try:
            self.preview.clear()
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            cleanup_failed("preview", exc)
        with self._lock:
            # Do not make a partial wipe look complete.  A later lifecycle pass
            # (or the deferred watcher) must be allowed to retry failed actions.
            if self._reader_cleanup_pending is not None:
                cleanup_complete = False
            self._sensitive_cleared = cleanup_complete
        if not cleanup_complete and _worker_is_alive(worker_for_retry):
            # The worker finalizer can reach this method before publishing its
            # ``finished`` event. Arrange a retry on the watcher rather than
            # recursively calling cleanup on the worker thread itself.
            self.defer_sensitive_cleanup()

    def defer_sensitive_cleanup(self) -> None:
        """Retry sensitive cleanup once a timed-out worker has exited.

        Wiping arrays while inference is still executing can race the provider,
        so a stop timeout does not forcefully mutate live state.  A daemon waiter
        instead follows the worker's terminal event and retries the idempotent
        cleanup routine. Sessions that never started (or whose thread is already
        dead) are cleaned synchronously.
        """
        with self._lock:
            if self._sensitive_cleared:
                return
            worker = self._worker
            worker_alive = _worker_is_alive(worker)
            watcher = self._sensitive_cleanup_watcher
            if worker_alive and watcher is not None:
                try:
                    watcher_alive = watcher.is_alive()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("sensitive cleanup watcher liveness failed: %s", type(exc).__name__)
                    watcher_alive = False
                # A watcher object is published before ``start`` so a second
                # concurrent caller cannot create a duplicate. ``ident`` is
                # still ``None`` during that tiny hand-off window.
                watcher_started = getattr(watcher, "ident", None) is not None
                if watcher_alive or not watcher_started:
                    return
            if worker_alive:
                watcher = threading.Thread(
                    target=self._wait_and_clear_sensitive,
                    name=f"cleanup-{self.search_id[:8]}",
                    daemon=True,
                )
                self._sensitive_cleanup_watcher = watcher
            else:
                watcher = None
        if watcher is None:
            try:
                self.clear_sensitive_state()
            except Exception as exc:  # noqa: BLE001 - caller must retain original error
                logger.warning("sensitive cleanup retry failed: %s", type(exc).__name__)
            return
        try:
            watcher.start()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                if self._sensitive_cleanup_watcher is watcher:
                    self._sensitive_cleanup_watcher = None
            logger.warning("sensitive cleanup watcher start failed: %s", type(exc).__name__)
            try:
                self.clear_sensitive_state()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("sensitive cleanup fallback failed: %s", type(cleanup_exc).__name__)

    def _wait_and_clear_sensitive(self) -> None:
        # Normally ``_finished`` is set by the worker finalizer.  If thread
        # startup failed before entering ``_run``, however, the event may never
        # be published; observing the thread liveness keeps that rare path from
        # retaining sensitive state forever.
        while not self._finished.wait(timeout=0.5):
            with self._lock:
                worker = self._worker
                if not _worker_is_alive(worker):
                    break
        try:
            self.clear_sensitive_state()
        except Exception as exc:  # noqa: BLE001 - final cleanup is best effort
            logger.warning("deferred sensitive cleanup failed: %s", type(exc).__name__)

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
                    target_id=target_id,
                    name=name,
                    **self._target_status[target_id],
                )
                for target_id, name in self._target_metadata
            ]
            found_count = sum(item.status == "found" for item in target_views)
            return SearchView(
                search_id=self.search_id,
                target_id=self._primary_target_id,
                target_name=self._primary_target_name,
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
            # Runtime cost guards.  ``getattr`` keeps views readable for sessions
            # created by older callers that supply a pre-hardening Settings object.
            "roi_batch_enabled": bool(getattr(settings, "roi_batch_enabled", True)),
            "roi_batch_size": int(getattr(settings, "roi_batch_size", 8)),
            "arcface_micro_batch_size": int(
                getattr(settings, "arcface_micro_batch_size", 16)
            ),
            "max_faces_per_frame": int(getattr(settings, "max_faces_per_frame", 64)),
            "match_profile": settings.match_profile,
            "evidence_statistic": settings.evidence_statistic,
            "evidence_top_k": settings.evidence_top_k,
            "face_tier_hysteresis_px": settings.face_tier_hysteresis_px,
            "camera_motion_compensation": settings.camera_motion_compensation,
            "source_epoch": self._source_epoch,
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
        try:
            reader = LatestFrameReader(
                self.source,
                self.settings,
                on_status=self._on_reader_status,
                on_drop=self._on_drop,
            )
        except Exception as exc:  # noqa: BLE001 - startup failures must still release the slot
            self._transition(SearchStatus.FAILED, _safe_error(exc), publish=False)
            self._finished_at = time.monotonic()
            target_ids = [target.target_id for target in self.targets]
            try:
                self.on_finished(self.search_id, target_ids)
            except Exception as callback_exc:  # noqa: BLE001
                logger.warning("search completion callback failed: %s", type(callback_exc).__name__)
            try:
                self._publish_terminal_event()
            except Exception as event_exc:  # noqa: BLE001
                logger.warning("terminal event publication failed: %s", type(event_exc).__name__)
            try:
                self.clear_sensitive_state()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("sensitive state cleanup failed: %s", type(cleanup_exc).__name__)
            self._finished.set()
            return
        # Publish the reader under the same lock used by stop/terminal cleanup.
        # A stop can arrive before the worker reaches this line; in that case
        # cleanup has already detached the slot and the newly-created reader
        # must be stopped locally instead of being attached after the fact.
        attach_reader = False
        with self._lock:
            if not self._reader_detached and not self._stop.is_set():
                self._reader = reader
                attach_reader = True
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
        seen_source_epoch = -1
        try:
            if attach_reader:
                reader.start()
            else:
                # A concurrent stop/cleanup claimed the reader slot before the
                # worker could attach it.  Do not start a detached capture
                # thread; the surrounding ``finally`` still drives the normal
                # terminal transition and cleanup path.
                self._stop.set()
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
                packet_epoch = getattr(packet, "source_epoch", None)
                with self._lock:
                    source_epoch = (
                        int(packet_epoch)
                        if isinstance(packet_epoch, (int, np.integer))
                        else self._source_epoch
                    )
                if source_epoch != seen_source_epoch:
                    self._reset_temporal_state()
                    tracks = []
                    previous_motion_gray = None
                    seen_source_epoch = source_epoch
                    # Never let a reconnect wait for the previous cadence
                    # deadline before running the mandatory stages on its first
                    # fresh frame.
                    last_person_at = -1e9
                    last_face_at = -1e9
                    last_roi_face_at = -1e9
                    last_preview_at = -1e9
                    face_pass_index = 0
                started = time.monotonic()
                now = packet.captured_at
                with self._lock:
                    self.metrics.frame_height, self.metrics.frame_width = packet.frame.shape[:2]
                # Face/track association is deterministic for a given detection
                # list and track snapshot.  Keep the full-frame result keyed by
                # observation identity so the common no-ROI path does not run the
                # same Hungarian/greedy matcher twice in one frame.
                association_cache: dict[int, tuple[int, str]] = {}
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
                    roi_tracks = self._tracks_needing_roi_face_pass(
                        faces, tracks, association_cache=association_cache
                    )
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
                        # Rate-limit from completion time: a long ROI inference
                        # must not make the next pass immediately eligible and
                        # create a burst on top of the still-running loop.
                        last_roi_face_at = time.monotonic()
                    self._record_face_metrics(faces)

                # A source loss may race an in-flight inference. Do not commit
                # observations from the old connection to the new temporal state.
                with self._lock:
                    epoch_changed = self._source_epoch != source_epoch
                if epoch_changed:
                    continue

                # ArcFace runs once, here, and only on faces that survived dedup and
                # the quality gate. Detections thrown away by _merge_faces or
                # _is_face_matchable never cost an embedding.
                matchable = [
                    face for face in faces if SearchSession._is_face_matchable(self, face)
                ]
                matchable_before_budget = matchable
                matchable = self._limit_matchable_faces(
                    matchable, tracks, association_cache=association_cache
                )
                budget_limited = len(matchable) != len(matchable_before_budget)
                if matchable:
                    stage_started = time.monotonic()
                    accepted_faces = self._embed_faces_microbatched(packet.frame, matchable)
                    self._record_stage("face_embed", stage_started)
                else:
                    accepted_faces = []
                with self._lock:
                    epoch_changed = self._source_epoch != source_epoch
                if epoch_changed:
                    continue
                # Compute identity ranking once per face and share it between the
                # diagnostics pass and the confirmation pass below.  This is
                # especially valuable for a 20-target gallery where each ranking
                # otherwise performs a complete dot-product + sort twice.
                ranked_matches_by_face = {
                    id(face): self._rank_identity_matches(face) for face in accepted_faces
                }
                self._record_target_observations(
                    accepted_faces, ranked_matches_by_face=ranked_matches_by_face
                )
                self._record_rejected_observations(
                    [face for face in faces if not SearchSession._is_face_matchable(self, face)]
                )
                detailed = self._cached_associations(
                    accepted_faces,
                    tracks,
                    association_cache,
                    allow_cache=not budget_limited,
                )
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
                    ranked_matches = ranked_matches_by_face.get(id(face), ())
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
            try:
                try:
                    self._stop_reader_once(reader)
                except Exception:  # noqa: BLE001 - cleanup must continue if capture teardown fails
                    self._transition(
                        SearchStatus.FAILED,
                        "video source cleanup failed",
                        publish=False,
                    )
                if self._stop_requested or self.status not in (
                    SearchStatus.FAILED,
                    SearchStatus.STOPPED,
                    SearchStatus.COMPLETED,
                    SearchStatus.TIMED_OUT,
                ):
                    self._transition(SearchStatus.STOPPED, None, publish=False)
            finally:
                target_ids = [target.target_id for target in self.targets]
                self._finished_at = time.monotonic()
                try:
                    self.on_finished(self.search_id, target_ids)
                except Exception as exc:  # noqa: BLE001 - callback failures are isolated
                    logger.warning("search completion callback failed: %s", type(exc).__name__)
                try:
                    self._publish_terminal_event()
                except Exception as exc:  # noqa: BLE001 - terminal cleanup must still run
                    logger.warning("terminal event publication failed: %s", type(exc).__name__)
                try:
                    self.clear_sensitive_state()
                except Exception as exc:  # noqa: BLE001 - best-effort memory cleanup
                    logger.warning("sensitive state cleanup failed: %s", type(exc).__name__)
                finally:
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

    def _record_target_observations(
        self,
        faces: list[FaceObservation],
        *,
        ranked_matches_by_face: dict[int, list[tuple[str, float]]] | None = None,
    ) -> None:
        if not faces or not self._active_targets:
            return
        best_in_frame: dict[str, tuple[float, FaceObservation]] = {}
        for face in faces:
            ranked = (
                ranked_matches_by_face.get(id(face))
                if ranked_matches_by_face is not None
                else self._rank_identity_matches(face)
            )
            for target_id, similarity in ranked or ():
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

    def _limit_matchable_faces(
        self,
        faces: list[FaceObservation],
        tracks: list[Track],
        *,
        association_cache: dict[int, tuple[int, str]] | None = None,
    ) -> list[FaceObservation]:
        """Apply a per-frame ArcFace budget while preserving useful evidence.

        Crowded frames can contain hundreds of detector boxes.  Sending all of
        them to ArcFace creates a latency/VRAM spike that then starves the
        confirmation sampler.  Faces owned by a live person track are retained
        ahead of unassociated faces; within each group larger, sharper and
        higher-confidence observations win.  The original observation order is
        restored after selection so downstream association and event payloads
        remain deterministic.
        """
        if not faces:
            return []
        configured = getattr(self.settings, "max_faces_per_frame", len(faces))
        try:
            limit = max(1, int(configured))
        except (TypeError, ValueError):
            limit = len(faces)
        if len(faces) <= limit:
            return faces

        cache = association_cache if association_cache is not None else {}
        cache_method = getattr(self, "_cached_associations", None)
        if callable(cache_method):
            detailed = cache_method(faces, tracks, cache)
        else:
            detailed = SearchSession._cached_associations(self, faces, tracks, cache)
        associated = {face_index: value for face_index, value in detailed.items()}

        def priority(item: tuple[int, FaceObservation]) -> tuple[object, ...]:
            index, face = item
            association = associated.get(index)
            track_id = association[0] if association is not None else None
            state = getattr(self, "_track_states", {}).get(track_id, ("tracking", 0.0))[0]
            # A candidate/shadow track already carrying evidence is more valuable
            # than a fresh tracking box.  Confirmed tracks are normally excluded
            # from ROI, but keeping their face here is harmless and deterministic.
            state_rank = {"candidate": 0, "shadow": 0, "confirmed": 1}.get(state, 2)
            return (
                0 if association is not None and association[1] == "person_strict" else (
                    1 if association is not None else 2
                ),
                state_rank,
                -face.short_side,
                -float(face.quality),
                -float(face.detection_score),
                index,
            )

        selected_indices = {
            index for index, _ in sorted(enumerate(faces), key=priority)[:limit]
        }
        dropped = len(faces) - len(selected_indices)
        with self._lock:
            if hasattr(self.metrics, "faces_dropped_by_budget"):
                self.metrics.faces_dropped_by_budget += dropped
        return [face for index, face in enumerate(faces) if index in selected_indices]

    def _embed_faces_microbatched(
        self, frame: np.ndarray, faces: list[FaceObservation]
    ) -> list[FaceObservation]:
        """Run ArcFace in bounded chunks, with an OOM-aware split fallback."""
        if not faces:
            return []
        configured = getattr(self.settings, "arcface_micro_batch_size", len(faces))
        try:
            batch_size = max(1, int(configured))
        except (TypeError, ValueError):
            batch_size = len(faces)
        embedded: list[FaceObservation] = []
        for offset in range(0, len(faces), batch_size):
            embedded.extend(
                self._embed_face_chunk(frame, faces[offset : offset + batch_size])
            )
        return embedded

    def _embed_face_chunk(
        self,
        frame: np.ndarray,
        faces: list[FaceObservation],
        *,
        split_depth: int = 0,
    ) -> list[FaceObservation]:
        if not faces:
            return []
        stop_event = getattr(self, "_stop", None)
        if stop_event is not None and stop_event.is_set():
            return []
        with self._lock:
            if hasattr(self.metrics, "embedding_batch_count"):
                self.metrics.embedding_batch_count += 1
        try:
            embedded = self.face_backend.embed_faces(frame, faces)
        except ModelUnavailableError:
            # A missing/unloadable model is a process-level fault. Let the worker
            # fail closed so the API exposes its actionable 503-style message.
            raise
        except Exception as exc:
            if not _is_embedding_capacity_error(exc):
                if not _is_recoverable_embedding_error(exc):
                    raise
                # CUDA/TensorRT can transiently reject one execution (for example
                # after a context reset) even though the next frame is usable. Drop
                # this chunk and keep the search alive; a counter makes the degraded
                # frame visible in the existing status payload.
                with self._lock:
                    if hasattr(self.metrics, "embedding_failures"):
                        self.metrics.embedding_failures += 1
                return []
            # Retry at most four levels deep.  This bounds the number of expensive
            # retries while still allowing a 16-row batch to fall back to singles
            # on a very small GPU.  Faces that still cannot be embedded are dropped
            # for this frame; the next frame gets a fresh chance.
            if len(faces) > 1 and split_depth < 4:
                midpoint = max(1, len(faces) // 2)
                return self._embed_face_chunk(
                    frame, faces[:midpoint], split_depth=split_depth + 1
                ) + self._embed_face_chunk(
                    frame, faces[midpoint:], split_depth=split_depth + 1
                )
            with self._lock:
                if hasattr(self.metrics, "faces_dropped_by_budget"):
                    self.metrics.faces_dropped_by_budget += len(faces)
            return []
        if embedded is None:
            with self._lock:
                if hasattr(self.metrics, "embedding_failures"):
                    self.metrics.embedding_failures += 1
            return []
        try:
            embedded_list = list(embedded)
        except Exception as exc:  # noqa: BLE001 - malformed provider output is a frame-local miss
            logger.warning(
                "embedding provider returned a non-iterable result: %s",
                type(exc).__name__,
            )
            with self._lock:
                if hasattr(self.metrics, "embedding_failures"):
                    self.metrics.embedding_failures += 1
            return []
        # Most backends return ``dataclasses.replace(face, embedding=...)``.  A
        # provider can nevertheless return a scalar, a foreign object, a NaN
        # vector, or rows in a different order.  Reconcile only validated
        # FaceObservation rows to the input detections; anything that cannot be
        # mapped unambiguously is dropped for this frame.  This keeps malformed
        # output out of association/ranking while preserving object identity for
        # the normal path (and therefore reusing the association cache).
        expected_dimensions: set[int] = set()
        try:
            gallery = getattr(self, "_identity_targets", {})
            gallery_values = gallery.values() if isinstance(gallery, dict) else ()
            for target in gallery_values:
                normalized = _safe_normalize_embedding(getattr(target, "embedding", None))
                if normalized is not None:
                    expected_dimensions.add(normalized.size)
        except Exception:  # noqa: BLE001 - a malformed legacy gallery must not break a frame
            expected_dimensions.clear()
        expected_dimension = (
            next(iter(expected_dimensions)) if len(expected_dimensions) == 1 else None
        )

        source_boxes = [_coerce_face_bbox(getattr(face, "bbox", None)) for face in faces]
        source_by_identity = {id(face): index for index, face in enumerate(faces)}
        unused_sources = set(range(len(faces)))
        assignments: dict[int, tuple[FaceObservation, np.ndarray]] = {}
        malformed = False

        for output_index, result in enumerate(embedded_list):
            if not isinstance(result, FaceObservation):
                malformed = True
                continue
            result_bbox = _coerce_face_bbox(getattr(result, "bbox", None))
            embedding = _safe_normalize_embedding(getattr(result, "embedding", None))
            if result_bbox is None or embedding is None:
                malformed = True
                continue
            if expected_dimension is not None and embedding.size != expected_dimension:
                malformed = True
                continue

            # Identity is the strongest mapping signal for backends that mutate
            # the supplied observation in place.
            source_index = source_by_identity.get(id(result))
            if source_index is not None and source_index in unused_sources:
                if source_boxes[source_index] is not None and np.allclose(
                    source_boxes[source_index], result_bbox, rtol=0.0, atol=1e-3
                ):
                    unused_sources.remove(source_index)
                    assignments[source_index] = (result, embedding)
                    continue

            candidates = [
                index
                for index in unused_sources
                if source_boxes[index] is not None
                and np.allclose(
                    source_boxes[index], result_bbox, rtol=0.0, atol=1e-3
                )
            ]
            # Duplicate boxes are inherently ambiguous.  An exact positional
            # match is still safe for the common full-cardinality response;
            # otherwise fail closed instead of assigning one person's vector to
            # another person's detection.
            if len(candidates) != 1:
                positional = (
                    output_index
                    if output_index in unused_sources
                    and output_index in candidates
                    else None
                )
                if positional is None:
                    malformed = True
                    continue
                source_index = positional
            else:
                source_index = candidates[0]
            unused_sources.remove(source_index)
            assignments[source_index] = (result, embedding)

        accepted: list[FaceObservation] = []
        for source_index in range(len(faces)):
            item = assignments.get(source_index)
            if item is None:
                continue
            _, embedding = item
            source = faces[source_index]
            try:
                source.embedding = embedding
                accepted.append(source)
            except Exception:  # noqa: BLE001 - immutable legacy observations are copied
                try:
                    accepted.append(replace(source, embedding=embedding))
                except Exception:  # noqa: BLE001 - a malformed observation is a local miss
                    malformed = True

        if malformed:
            with self._lock:
                if hasattr(self.metrics, "embedding_failures"):
                    self.metrics.embedding_failures += 1
        return accepted

    def _cached_associations(
        self,
        faces: list[FaceObservation],
        tracks: list[Track],
        cache: dict[int, tuple[int, str]],
        *,
        allow_cache: bool = True,
    ) -> dict[int, tuple[int, str]]:
        """Reuse detection-time associations when the face objects are unchanged."""
        if allow_cache and faces and cache and all(id(face) in cache for face in faces):
            return {index: cache[id(face)] for index, face in enumerate(faces)}
        detailed = associate_faces_to_tracks_detailed(faces, tracks)
        for face_index, value in detailed.items():
            if 0 <= face_index < len(faces):
                cache[id(faces[face_index])] = value
        return detailed

    def _rank_identity_matches(self, face: FaceObservation) -> list[tuple[str, float]]:
        """Rank against the immutable batch gallery, including found targets."""
        face_embedding = _safe_normalize_embedding(getattr(face, "embedding", None))
        if face_embedding is None:
            return []
        ranked: list[tuple[str, float]] = []
        try:
            target_items = self._identity_targets.items()
        except Exception:  # noqa: BLE001 - fail closed for malformed legacy galleries
            return []
        for target_id, target in target_items:
            target_embedding = _safe_normalize_embedding(
                getattr(target, "embedding", None)
            )
            if target_embedding is None or target_embedding.size != face_embedding.size:
                continue
            try:
                similarity = float(np.dot(target_embedding, face_embedding))
            except (TypeError, ValueError, OverflowError, FloatingPointError):
                continue
            if np.isfinite(similarity):
                ranked.append((target_id, similarity))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

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
        self,
        faces: list[FaceObservation],
        tracks: list[Track],
        *,
        association_cache: dict[int, tuple[int, str]] | None = None,
    ) -> list[Track]:
        """Return tracks that do not own a preferred-size full-frame face.

        Tracks that keep yielding nothing, and tracks already confirmed, are
        excluded — neither can turn more ROI passes into new evidence.
        """
        if not tracks:
            return []
        accepted = [face for face in faces if SearchSession._is_face_matchable(self, face)]
        associations = associate_faces_to_tracks_detailed(accepted, tracks)
        if association_cache is not None:
            for face_index, value in associations.items():
                if 0 <= face_index < len(accepted):
                    association_cache[id(accepted[face_index])] = value
        satisfied_track_ids = {
            track_id
            for face_index, (track_id, _) in associations.items()
            if accepted[face_index].short_side >= self.settings.preferred_search_face_px
        }
        confirmed_track_ids = {
            track_id
            for track_id, (state, _) in getattr(self, "_track_states", {}).items()
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
        last_pass = getattr(self, "_roi_last_pass", {})
        self._roi_last_pass = {
            key: value for key, value in last_pass.items() if key in live_ids
        }
        # A score-only sort lets one high-confidence track consume every pass.
        # Older attempts are promoted first, while score remains the tie-breaker
        # for tracks that have never been examined.  The scheduler state is kept
        # lazily so small test stubs and legacy callers need no new fields.
        last_pass: dict[int, int] = self._roi_last_pass
        candidates.sort(
            key=lambda track: (last_pass.get(track.track_id, -1), -track.score, track.track_id)
        )
        self._roi_last_pass = last_pass
        return candidates

    def _analyze_person_rois(self, frame: np.ndarray, tracks: list[Track]) -> list[FaceObservation]:
        height, width = frame.shape[:2]
        last_pass: dict[int, int] = getattr(self, "_roi_last_pass", {})
        ranked = sorted(
            tracks,
            key=lambda track: (last_pass.get(track.track_id, -1), -track.score, track.track_id),
        )
        prepared: list[tuple[Track, np.ndarray, int, int, int]] = []
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
            scale = self.settings.roi_detection_scale(roi.shape[1], roi.shape[0])
            prepared.append((track, roi, x1, y1, scale))
            # Mark the attempt before inference.  If a provider errors or returns
            # no face, the next pass still rotates to the other tracks.
            last_pass[track.track_id] = getattr(self, "_roi_schedule_counter", 0)
            self._roi_schedule_counter = last_pass[track.track_id] + 1

        observations_by_track: dict[int, list[FaceObservation]] = {}
        # Keep first-seen bucket order (rather than sorting scales) so debug output
        # and deterministic tests retain the scheduler's track order.
        buckets: dict[int, list[tuple[Track, np.ndarray, int, int, int]]] = {}
        for item in prepared:
            buckets.setdefault(item[4], []).append(item)
        batch_enabled = bool(getattr(self.settings, "roi_batch_enabled", True))
        configured_batch_size = getattr(self.settings, "roi_batch_size", 8)
        try:
            batch_size = max(1, int(configured_batch_size))
        except (TypeError, ValueError):
            batch_size = 1
        for scale, bucket in buckets.items():
            for offset in range(0, len(bucket), batch_size):
                chunk = bucket[offset : offset + batch_size]
                rois = [item[1] for item in chunk]
                found_lists = SearchSession._detect_roi_batch(
                    self, rois, scale, enabled=batch_enabled
                )
                for item, found in zip(chunk, found_lists, strict=True):
                    track, _, x1, y1, _ = item
                    observations_by_track[track.track_id] = list(found)
                    self._note_roi_outcome(track.track_id, hit=bool(found))

        observations: list[FaceObservation] = []
        for track, _, x1, y1, _ in prepared:
            for face in observations_by_track.get(track.track_id, []):
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

    def _detect_roi_batch(
        self,
        rois: list[np.ndarray],
        scale: int,
        *,
        enabled: bool,
    ) -> list[list[FaceObservation]]:
        """Dispatch one fixed-scale ROI group, with a legacy fallback.

        Third-party/fake backends written before ``detect_faces_batch`` remain
        valid: each crop is sent through the existing method when the optional
        capability is absent or disabled.  A malformed batch response also falls
        back rather than dropping an entire frame.
        """
        if not rois:
            return []
        batch_method = getattr(self.face_backend, "detect_faces_batch", None)
        if enabled and callable(batch_method):
            try:
                result = batch_method(
                    rois, enrollment=False, detection_size=scale
                )
                if isinstance(result, list) and len(result) == len(rois):
                    with self._lock:
                        if hasattr(self.metrics, "roi_batch_count"):
                            self.metrics.roi_batch_count += 1
                    return [list(item or []) for item in result]
            except (TypeError, AttributeError, NotImplementedError):
                # A legacy implementation may expose a similarly named helper
                # with a narrower signature.  Use the proven single-crop path.
                pass
        results: list[list[FaceObservation]] = []
        for roi in rois:
            results.append(
                list(
                    self.face_backend.detect_faces(
                        roi, enrollment=False, detection_size=scale
                    )
                    or []
                )
            )
        with self._lock:
            if hasattr(self.metrics, "roi_batch_count"):
                self.metrics.roi_batch_count += len(rois)
        return results

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

    def _on_reader_status(self, status: SearchStatus, error: str | None) -> None:
        """Bridge reader status changes and mark reconnect boundaries."""
        if self._stop.is_set() and status not in {
            SearchStatus.STOPPING,
            SearchStatus.STOPPED,
            SearchStatus.FAILED,
            SearchStatus.COMPLETED,
            SearchStatus.TIMED_OUT,
        }:
            return
        if status == SearchStatus.SOURCE_LOST:
            with self._lock:
                self._source_epoch += 1
                reader = self._reader
            # Frames decoded before the reconnect belong to the old source. Drop
            # queued packets immediately; the worker will reset trackers and
            # confirmation windows when it observes the incremented epoch. The
            # getattr keeps fake/legacy readers used by integrations compatible.
            clear_reader = getattr(reader, "clear", None)
            if clear_reader is not None:
                try:
                    clear_reader()
                except Exception as exc:  # noqa: BLE001 - a failed flush must not kill the reader thread
                    logger.warning("source epoch reader flush failed: %s", type(exc).__name__)
        self._transition(status, error)

    def _reset_temporal_state(self) -> None:
        """Reset association/evidence state after a source epoch changes."""
        try:
            self._tracker.reset()
        except Exception as exc:  # noqa: BLE001 - replace a broken tracker at the boundary
            logger.warning("person tracker epoch reset failed: %s", type(exc).__name__)
            self._tracker = ByteTracker()
        try:
            self._face_tracker.reset()
        except Exception as exc:  # noqa: BLE001 - replace a broken tracker at the boundary
            logger.warning("face tracker epoch reset failed: %s", type(exc).__name__)
            self._face_tracker = FaceTracker(
                iou_threshold=self.settings.face_track_iou_threshold,
                buffer_seconds=self.settings.face_track_buffer_seconds,
            )
        for confirmation in list(self._confirmations.values()):
            # A reconnect is a privacy and correctness boundary: drop old
            # embeddings as well as the track mapping so no evidence can cross
            # from one camera connection into the next.
            try:
                confirmation.clear_sensitive()
            except Exception as exc:  # noqa: BLE001 - one stale track must not poison a new epoch
                logger.warning("confirmation epoch reset failed: %s", type(exc).__name__)
        for face, _, _ in self._debug_faces:
            _wipe_array(face.embedding)
            _wipe_array(face.bbox)
            _wipe_array(face.landmarks)
        self._debug_faces.clear()
        self._track_states.clear()
        self._shadow_tracks.clear()
        self._roi_misses.clear()
        self._roi_skips.clear()
        self._track_tiers.clear()
        self._roi_last_pass = {}
        self._roi_schedule_counter = 0
        self._budget_credit = 0.0
        self._motion_hanning = None
        with self._lock:
            for state in self._target_status.values():
                if state.get("status") == "found":
                    continue
                state.update(
                    {
                        "evidence_count": 0,
                        "required_evidence": 0,
                        "qualifying_evidence": 0,
                        "window_similarity": None,
                        "window_statistic": None,
                        "required_similarity": None,
                        "aggregate_similarity": None,
                        "required_aggregate_similarity": None,
                        "tier": None,
                        "last_face_px": None,
                        "best_similarity": None,
                        "best_observed_similarity": None,
                        "last_rejection_reason": None,
                        "last_rejection_face_px": None,
                    }
                )

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
        self._request_index: dict[str, str] = {}
        self._active_search_id: str | None = None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._prune_timer: threading.Timer | None = None
        self._prune_timer_deadline: float | None = None
        self._prune_generation = 0
        self._shutdown = False

    def enroll(self, image: np.ndarray, name: str = "目标") -> TargetView:
        self._ensure_open()
        target_name = _normalize_target_name(name)
        # Fail before model inference when the bounded enrollment gallery is
        # already full.  The insertion-time check below remains necessary for a
        # concurrent caller racing this fast path.
        with self._lock:
            max_targets = int(getattr(self.settings, "max_enrolled_targets", 100))
            if len(self._targets) >= max_targets:
                raise PersonSearchError(
                    "too many enrolled targets",
                    code="target_capacity_exceeded",
                    status_code=429,
                )
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
        # Shutdown can race the model inference above.  Re-check while holding
        # the lifecycle gate immediately before publishing the new target so a
        # request that started before shutdown cannot resurrect manager state.
        try:
            with self._lifecycle_lock:
                self._ensure_open()
                with self._lock:
                    if len(self._targets) >= max_targets:
                        raise PersonSearchError(
                            "too many enrolled targets",
                            code="target_capacity_exceeded",
                            status_code=429,
                        )
                    self._targets[target_id] = Target(
                        target_id=target_id, embedding=embedding, view=view, name=target_name
                    )
        except Exception:
            # The embedding is a local sensitive buffer until the insertion is
            # committed.  Do not leave it alive when a lifecycle/capacity check
            # rejects the request.
            _wipe_array(embedding)
            raise
        return view

    def delete_target(self, target_id: str) -> bool:
        with self._lifecycle_lock:
            with self._lock:
                # A callback/prune failure must not turn a stale active id into a
                # KeyError that blocks all target administration.
                _, session = self._active_session_locked()
                if session is not None and any(
                    target.target_id == target_id for target in session.targets
                ):
                    raise PersonSearchError(
                        "target is used by an active search", code="target_in_use", status_code=409
                    )
                target = self._targets.pop(target_id, None)
            if target is not None:
                _wipe_array(target.embedding)
            return target is not None

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
        self._ensure_open()
        if not target_ids:
            raise PersonSearchError(
                "at least one target is required", code="invalid_targets", status_code=422
            )
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise PersonSearchError(
                "timeout_seconds must be positive", code="invalid_timeout", status_code=422
            )
        normalized_request_id = _normalize_request_id(request_id)
        if len(set(target_ids)) != len(target_ids):
            raise PersonSearchError(
                "target_ids must be unique", code="duplicate_targets", status_code=422
            )
        self._prune_sessions()
        with self._lifecycle_lock:
            # The first check happens before validation for a cheap fast path;
            # this one closes the shutdown race after the caller acquired the
            # lifecycle gate.
            self._ensure_open()
            with self._lock:
                active_id, active_session = self._active_session_locked()
                if normalized_request_id:
                    retained_id = self._request_index.get(normalized_request_id)
                    retained = self._sessions.get(retained_id) if retained_id else None
                    if retained is not None:
                        return retained.view()
                    if retained_id is not None:
                        self._request_index.pop(normalized_request_id, None)
                if normalized_request_id and active_session and (
                    active_session.request_id == normalized_request_id
                ):
                    return active_session.view()
                targets: list[Target] = []
                for target_id in target_ids:
                    target = self._targets.get(target_id)
                    if target is None:
                        raise PersonSearchError(
                            "target not found", code="target_not_found", status_code=404
                        )
                    targets.append(target)
            # The worker callback does not take ``_lifecycle_lock`` and may
            # release the active slot while target validation above is running.
            # Re-read under ``_lock`` so a session that just became terminal
            # cannot produce a stale capacity error.
            with self._lock:
                active_id, active_session = self._active_session_locked()
            if active_id is not None:
                if not replace_active:
                    raise PersonSearchError(
                        "only one search may run at a time",
                        code="search_capacity_exceeded",
                        status_code=409,
                    )
                if active_session is not None:
                    if active_session.finished.is_set():
                        active_session.clear_evidence()
                    else:
                        active_session.stop()
            ensure_ready = getattr(self.person_detector, "ensure_ready", None)
            if ensure_ready:
                ensure_ready()
            # Model warm-up may take seconds and shutdown is allowed to happen
            # while it runs.  Never publish a session after the manager closed.
            self._ensure_open()
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
                request_id=normalized_request_id,
            )
            with self._lock:
                self._sessions[search_id] = session
                if normalized_request_id:
                    self._request_index[normalized_request_id] = search_id
                self._active_search_id = search_id
            try:
                session.start()
            except Exception:
                with self._lock:
                    self._sessions.pop(search_id, None)
                    if normalized_request_id and self._request_index.get(normalized_request_id) == search_id:
                        self._request_index.pop(normalized_request_id, None)
                    if self._active_search_id == search_id:
                        self._active_search_id = None
                try:
                    session.clear_evidence()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed-start evidence cleanup failed: %s", type(exc).__name__)
                try:
                    session.defer_sensitive_cleanup()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed-start sensitive cleanup failed: %s", type(exc).__name__)
                raise
            return session.view()

    def get_search(self, search_id: str) -> SearchView:
        self._prune_sessions()
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
        normalized = _normalize_request_id(request_id)
        if not normalized:
            return None
        self._prune_sessions()
        with self._lock:
            search_id = self._request_index.get(normalized)
            session = self._sessions.get(search_id) if search_id else None
            if session is None and search_id is not None:
                self._request_index.pop(normalized, None)
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
        self._prune_sessions()
        with self._lock:
            _, session = self._active_session_locked()
        return session.view() if session else None

    def shutdown(self) -> None:
        # Serialize the state transition with the critical portions of enroll
        # and start_batch_search.  Expensive worker stopping stays outside this
        # short section, but no new target/session can be committed after the
        # shutdown bit becomes visible.
        with self._lifecycle_lock, self._lock:
            active_id = self._active_search_id
            active_session = self._sessions.get(active_id) if active_id else None
            if active_session is not None:
                try:
                    if active_session.finished.is_set():
                        active_session = None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("active session liveness check failed during shutdown: %s", type(exc).__name__)
                    active_session = None
            self._shutdown = True
            # Once shutdown is visible there is no active slot to advertise.
            # Keep the local session reference above so a still-running worker
            # can be stopped without consulting the now-cleared slot.
            self._active_search_id = None
            prune_timer = self._prune_timer
            self._prune_timer = None
            self._prune_timer_deadline = None
            self._prune_generation += 1
            self._request_index.clear()
        if prune_timer is not None:
            try:
                prune_timer.cancel()
            except Exception as exc:  # noqa: BLE001
                logger.warning("terminal timer shutdown cancellation failed: %s", type(exc).__name__)
        if active_session is not None:
            try:
                active_session.stop()
            except SearchStopTimeoutError:
                # Keep the process alive until the worker can release its slot; the
                # caller will still see a clear timeout if this is an API stop.
                logger.warning("active search did not stop before shutdown timeout")
            except Exception as exc:  # noqa: BLE001 - shutdown must clean the rest
                logger.warning("active search shutdown failed: %s", type(exc).__name__)
        with self._lock:
            sessions = list(self._sessions.values())
            targets = list(self._targets.values())
            self._targets.clear()
        for target in targets:
            _wipe_array(target.embedding)
        for session in sessions:
            try:
                session.clear_evidence()
            except Exception as exc:  # noqa: BLE001 - continue releasing other sessions
                logger.warning("session evidence shutdown cleanup failed: %s", type(exc).__name__)
            # Finished/dead sessions can be wiped immediately.  A live worker
            # gets a daemon watcher that retries after its terminal event, which
            # avoids racing an in-flight provider call.
            try:
                session.defer_sensitive_cleanup()
            except Exception as exc:  # noqa: BLE001 - shutdown remains best effort
                logger.warning("session sensitive shutdown cleanup failed: %s", type(exc).__name__)

    def _ensure_open(self) -> None:
        with self._lock:
            if self._shutdown:
                raise PersonSearchError(
                    "search manager is shut down",
                    code="manager_shutdown",
                    status_code=503,
                )

    def _active_session_locked(self) -> tuple[str | None, SearchSession | None]:
        """Return the live active session and clear stale slot pointers.

        ``_active_search_id`` is updated by a worker callback, while request
        handlers can observe it in the small interval before that callback (or
        after a failed/legacy callback).  Treat a missing or terminal session as
        stale so one abandoned id cannot permanently consume the singleton
        search slot.  The caller must hold ``self._lock``.
        """
        active_id = self._active_search_id
        if active_id is None:
            return None, None
        session = self._sessions.get(active_id)
        stale = session is None
        if session is not None:
            try:
                stale = session.finished.is_set()
            except Exception as exc:  # noqa: BLE001 - a broken legacy session is not a live slot
                logger.warning("active session liveness check failed: %s", type(exc).__name__)
                stale = True
            if not stale:
                try:
                    status = getattr(session, "status", None)
                    if status in {
                        SearchStatus.COMPLETED,
                        SearchStatus.TIMED_OUT,
                        SearchStatus.STOPPED,
                        SearchStatus.FAILED,
                    }:
                        stale = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("active session status check failed: %s", type(exc).__name__)
                    stale = True
        if stale:
            self._active_search_id = None
            if session is None:
                # A failed start/callback can leave an idempotency index entry
                # pointing at a session that was already removed.  Drop only
                # those dangling entries; retained terminal sessions keep their
                # request keys for reconciliation.
                for request_key, search_id in list(self._request_index.items()):
                    if search_id == active_id:
                        self._request_index.pop(request_key, None)
            return None, None
        return active_id, session

    def _get_session(self, search_id: str) -> SearchSession:
        self._prune_sessions()
        with self._lock:
            session = self._sessions.get(search_id)
        if session is None:
            raise PersonSearchError("search not found", code="search_not_found", status_code=404)
        return session

    def _on_finished(self, search_id: str, target_ids: list[str]) -> None:
        removed_targets: list[Target] = []
        with self._lock:
            if self._active_search_id == search_id:
                self._active_search_id = None
            for target_id in target_ids:
                try:
                    target = self._targets.pop(target_id, None)
                    if target is not None:
                        removed_targets.append(target)
                except Exception as exc:  # noqa: BLE001 - one target must not block slot release
                    logger.warning("terminal target cleanup failed: %s", type(exc).__name__)
        for target in removed_targets:
            _wipe_array(target.embedding)
        # Clearing the active slot and one-shot targets is the critical part of
        # completion.  Retention/janitor bookkeeping is best effort: an
        # unexpected metadata failure must never make the worker callback raise
        # and leave callers believing the search is still active.
        try:
            self._prune_sessions()
        except Exception as exc:  # noqa: BLE001 - callback must be fail-safe
            logger.warning("terminal session pruning failed: %s", type(exc).__name__)
        # ``SearchSession`` invokes this callback just before setting its
        # ``finished`` event. Schedule a short follow-up even when the first
        # prune pass cannot yet see the terminal bit.
        try:
            self._schedule_prune_timer()
        except Exception as exc:  # noqa: BLE001 - callback must be fail-safe
            logger.warning("terminal session timer failed: %s", type(exc).__name__)

    def _prune_sessions(self) -> None:
        """Retain only bounded, recently finished sessions for reconciliation."""
        to_release: list[SearchSession] = []
        try:
            now = time.monotonic()
            try:
                ttl = float(getattr(self.settings, "terminal_session_ttl_seconds", 3600.0))
            except (TypeError, ValueError) as exc:
                logger.warning("terminal session TTL is invalid: %s", type(exc).__name__)
                ttl = 3600.0
            try:
                max_retained = max(1, int(getattr(self.settings, "max_retained_sessions", 32)))
            except (TypeError, ValueError) as exc:
                logger.warning("max retained sessions is invalid: %s", type(exc).__name__)
                max_retained = 32
            with self._lock:
                terminal: list[SearchSession] = []
                for session in list(self._sessions.values()):
                    try:
                        if session.finished.is_set() and session._finished_at is not None:
                            terminal.append(session)
                    except Exception as exc:  # noqa: BLE001 - malformed legacy entries are skipped
                        logger.warning("terminal session inspection failed: %s", type(exc).__name__)
                expired = [
                    session
                    for session in terminal
                    if now - float(session._finished_at or now) >= ttl
                ]
                expired_ids = {session.search_id for session in expired}
                retained = [
                    session for session in terminal if session.search_id not in expired_ids
                ]
                retained.sort(key=lambda session: float(session._finished_at or 0.0))
                if len(retained) > max_retained:
                    expired.extend(retained[: len(retained) - max_retained])
                for session in expired:
                    try:
                        removed = self._sessions.pop(session.search_id, None)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("terminal session removal failed: %s", type(exc).__name__)
                        removed = None
                    if removed is not None:
                        if self._active_search_id == session.search_id:
                            self._active_search_id = None
                        try:
                            for request_id, indexed_search_id in list(self._request_index.items()):
                                if indexed_search_id == session.search_id:
                                    self._request_index.pop(request_id, None)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("terminal request index cleanup failed: %s", type(exc).__name__)
                        to_release.append(session)
            # A janitor must never stop at the first bad session.  Keep each
            # cleanup component isolated and ask the session's own watcher to
            # retry if a worker or a legacy object is still holding buffers.
            for session in to_release:
                try:
                    session.clear_evidence()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("terminal evidence cleanup failed: %s", type(exc).__name__)
                try:
                    session.clear_sensitive_state()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("terminal sensitive cleanup failed: %s", type(exc).__name__)
                try:
                    if not getattr(session, "_sensitive_cleared", False):
                        session.defer_sensitive_cleanup()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("terminal sensitive cleanup retry failed: %s", type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - janitor failures must not kill request/timer threads
            logger.warning("terminal session pruning failed: %s", type(exc).__name__)
        finally:
            # Always re-arm the timer, including when one session's cleanup or a
            # malformed legacy object raised unexpectedly.
            try:
                self._schedule_prune_timer()
            except Exception as exc:  # noqa: BLE001
                logger.warning("terminal session timer scheduling failed: %s", type(exc).__name__)

    def _schedule_prune_timer(self) -> None:
        """Arrange lazy-but-automatic TTL cleanup for idle managers.

        A dedicated janitor thread per manager would multiply background threads
        in tests and in embedding applications. A single daemon timer per manager
        wakes at the nearest terminal deadline (or shortly after a worker invokes
        ``on_finished``) and reschedules itself after pruning.
        """
        with self._lock:
            if self._shutdown:
                old_timer = self._prune_timer
                self._prune_timer = None
                self._prune_timer_deadline = None
                self._prune_generation += 1
                if old_timer is not None:
                    try:
                        old_timer.cancel()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("terminal timer cancellation failed: %s", type(exc).__name__)
                return
            now = time.monotonic()
            try:
                ttl = max(
                    0.001,
                    float(getattr(self.settings, "terminal_session_ttl_seconds", 3600.0)),
                )
            except (TypeError, ValueError) as exc:
                logger.warning("terminal session TTL is invalid: %s", type(exc).__name__)
                ttl = 3600.0
            try:
                max_retained = max(
                    1, int(getattr(self.settings, "max_retained_sessions", 32))
                )
            except (TypeError, ValueError) as exc:
                logger.warning("max retained sessions is invalid: %s", type(exc).__name__)
                max_retained = 32
            terminal: list[SearchSession] = []
            for session in list(self._sessions.values()):
                try:
                    if session._finished_at is not None:
                        terminal.append(session)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("terminal session timer inspection failed: %s", type(exc).__name__)
            if not terminal:
                timer = self._prune_timer
                self._prune_timer = None
                self._prune_timer_deadline = None
                self._prune_generation += 1
                if timer is not None:
                    try:
                        timer.cancel()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("terminal timer cancellation failed: %s", type(exc).__name__)
                return
            deadlines: list[float] = []
            finished_count = 0
            for session in terminal:
                try:
                    if session.finished.is_set():
                        finished_count += 1
                        deadlines.append(float(session._finished_at or now) + ttl)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("terminal session timer inspection failed: %s", type(exc).__name__)
            # The callback is made before ``finished.set()``; give it a brief
            # grace period rather than risking cleanup while terminal events are
            # still being published.
            delay = min(deadlines) - now if deadlines else 0.05
            if finished_count > max_retained:
                delay = 0.01
            delay = max(0.01, min(delay, 60.0))
            scheduled_deadline = now + delay
            old_timer = self._prune_timer
            old_deadline = self._prune_timer_deadline
            # Frequent status/active requests call ``_prune_sessions``. Keep an
            # existing timer when it already fires no later than the newly
            # computed deadline; cancel/recreate only when a new terminal session
            # introduces an earlier deadline or the prior timer is gone.
            if old_timer is not None:
                try:
                    timer_alive = old_timer.is_alive()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("terminal timer liveness check failed: %s", type(exc).__name__)
                    timer_alive = False
                if timer_alive:
                    if old_deadline is not None and old_deadline <= scheduled_deadline + 0.01:
                        return
                    try:
                        old_timer.cancel()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("terminal timer cancellation failed: %s", type(exc).__name__)
            self._prune_generation += 1
            generation = self._prune_generation
            timer = threading.Timer(
                delay, self._run_prune_timer, args=(generation,)
            )
            timer.daemon = True
            self._prune_timer = timer
            self._prune_timer_deadline = scheduled_deadline
            try:
                timer.start()
            except Exception:
                # Do not leave an unstarted timer object looking live to the
                # next request; a subsequent prune pass can install a fresh
                # janitor safely.
                if self._prune_timer is timer:
                    self._prune_timer = None
                    self._prune_timer_deadline = None
                    self._prune_generation += 1
                raise

    def _run_prune_timer(self, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._prune_generation:
                return
            self._prune_timer = None
            self._prune_timer_deadline = None
            if self._shutdown:
                return
        try:
            self._prune_sessions()
        except Exception as exc:  # noqa: BLE001 - timer threads must remain self-healing
            logger.warning("terminal session timer callback failed: %s", type(exc).__name__)
            try:
                self._schedule_prune_timer()
            except Exception as schedule_exc:  # noqa: BLE001
                logger.warning(
                    "terminal session timer reschedule failed: %s",
                    type(schedule_exc).__name__,
                )


def _sanitize_source(source: SourceConfig) -> SourceConfig:
    if source.type != SourceType.RTSP or not source.uri:
        return source.model_copy()
    try:
        parts = urlsplit(source.uri)
        parsed_port = parts.port
    except (TypeError, ValueError):
        # Validation normally prevents this path. Keep log/status rendering
        # safe for legacy sessions carrying a malformed URI.
        return source.model_copy(update={"uri": "rtsp://source/***"})
    host = parts.hostname or "source"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed_port}" if parsed_port else ""
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


def _normalize_request_id(request_id: str | None) -> str | None:
    """Normalize an idempotency key without allowing unbounded/control input."""
    if request_id is None:
        return None
    if not isinstance(request_id, str):
        raise PersonSearchError(
            "request_id must be a string", code="invalid_request_id", status_code=422
        )
    # Validate the submitted value before trimming it.  ``str.strip`` would
    # otherwise erase a leading/trailing newline (or another control/format
    # character), allowing that character to reach logs, metrics, or a
    # request-index key while making it invisible to the caller.  ``isprintable``
    # covers the Unicode control/format and line-separator characters that the
    # old ASCII-only ordinal check missed (for example U+0085, U+200B, U+2028).
    if any(not char.isprintable() for char in request_id):
        raise PersonSearchError(
            "request_id is invalid", code="invalid_request_id", status_code=422
        )
    normalized = request_id.strip()
    if not normalized:
        return None
    if len(normalized) > MAX_REQUEST_ID_LENGTH:
        raise PersonSearchError(
            "request_id is invalid", code="invalid_request_id", status_code=422
        )
    return normalized


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ModelUnavailableError):
        # Loader exceptions often include absolute model paths, environment
        # values, or provider internals.  Keep those details in controlled logs
        # while exposing a stable, actionable terminal message to API clients.
        logger.error("model unavailable: %s", exc.message)
        return "model unavailable; verify model files and runtime configuration"
    return f"{type(exc).__name__}: processing failed"


def _is_embedding_capacity_error(exc: Exception) -> bool:
    """Recognise provider allocation failures that are safe to retry smaller."""
    if isinstance(exc, (MemoryError, OverflowError)):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "failed to allocate",
            "alloc_failed",
            "bfc arena",
            "cuda out of memory",
            "cudnn_status_alloc",
            "resource exhausted",
        )
    )


def _is_recoverable_embedding_error(exc: Exception) -> bool:
    """Whether a recogniser/provider error can be isolated to this frame.

    Programming errors and explicit model-unavailable failures should still
    terminate the worker. Runtime/provider allocation and execution failures are
    safe to degrade for one frame, preserving the stream and confirmation window
    instead of turning a transient CUDA hiccup into a terminal search failure.
    """
    if isinstance(exc, (RuntimeError, OSError, ValueError)):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "rknn_err_malloc_fail",
            "cuda_error_out_of_memory",
            "cuda execution provider",
            "tensorrt",
            "provider execution",
            "execution provider",
            "resource exhausted",
            "allocation failed",
        )
    )


def _wipe_array(value: np.ndarray | None) -> None:
    """Best-effort zeroing for biometric buffers held by a terminal session."""
    if value is None:
        return
    try:
        value.fill(0)
    except (AttributeError, TypeError, ValueError):
        return


def _worker_is_alive(worker: object | None) -> bool:
    """Read worker liveness without allowing a broken test/integration object to leak."""
    if worker is None:
        return False
    is_alive = getattr(worker, "is_alive", None)
    if not callable(is_alive):
        return False
    try:
        return bool(is_alive())
    except Exception as exc:  # noqa: BLE001 - lifecycle checks are defensive
        logger.warning("worker liveness check failed: %s", type(exc).__name__)
        return False


def _clone_target(target: Target, *, include_embedding: bool = True) -> Target:
    """Copy a target without sharing its mutable biometric buffer.

    Terminal sessions pass ``include_embedding=False`` to retain only the
    display metadata needed for reconciliation.  Live copies always normalize to
    a standalone float32 array so cleanup cannot mutate the manager's gallery.
    """
    embedding = None
    if include_embedding and target.embedding is not None:
        embedding = np.asarray(target.embedding, dtype=np.float32).copy()
    return Target(
        target_id=target.target_id,
        embedding=embedding,
        view=target.view,
        name=target.name,
    )


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
