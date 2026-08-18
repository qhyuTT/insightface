from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from .backends import FaceBackend, InsightFaceBackend
from .config import Settings
from .confirmation import TrackConfirmation, associate_faces_to_tracks, normalize_bbox
from .detector import PersonDetector, YoloXOnnxDetector
from .domain import (
    FaceObservation,
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
from .errors import EnrollmentError, ModelUnavailableError, PersonSearchError
from .quality import normalize_embedding
from .rknn_backend import RknnFaceBackend, RknnPersonDetector
from .tracker import ByteTracker
from .video import LatestFrameReader

MAX_TARGET_NAME_LENGTH = 80
# Terminal sessions are kept as lightweight, read-only archives so clients can
# still fetch the final status/events immediately after a search finishes.  The
# active-session map itself must not retain an unbounded number of sessions.
MAX_FINISHED_SESSIONS = 32


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
        self._confirmations = {
            target.target_id: TrackConfirmation(self.settings) for target in self.targets
        }
        self._active_targets = {target.target_id: target for target in self.targets}
        self._target_status = {
            target.target_id: {"status": "searching", "found_at": None, "best_similarity": None}
            for target in self.targets
        }
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._reader: LatestFrameReader | None = None
        self._lock = threading.RLock()
        self._track_states: dict[int, tuple[str, float]] = {}
        self._sensitive_data_released = False
        self._last_preview_at = -1e9

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._run, name=f"search-{self.search_id[:8]}", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._reader:
            self._reader.stop()
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout=5.0)
        self._transition(SearchStatus.STOPPED, None)

    def view(self) -> SearchView:
        with self._lock:
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
                **metrics,
            )

    def release_sensitive_data(self) -> None:
        """Drop per-search inference state once the worker has terminated.

        A finished session can remain available briefly for the status/events
        endpoints, but it must not keep enrolled face embeddings, tracker
        arrays, or other inference-only state alive indefinitely.
        """

        with self._lock:
            if self._sensitive_data_released:
                return
            for target in self.targets:
                embedding = np.asarray(target.embedding)
                # Best-effort scrubbing also protects callers that still hold
                # a reference to the original Target object/array.
                if embedding.flags.writeable:
                    embedding.fill(0)
                target.embedding = np.empty(0, dtype=np.float32)
            self._active_targets.clear()
            self._confirmations.clear()
            self._tracker.reset()
            self._track_states.clear()
            self._reader = None
            self._sensitive_data_released = True

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
        def provider_rate(provider: str, cpu: float, cuda: float, rknn: float) -> float:
            normalized = provider.upper()
            if "RKNN" in normalized:
                return rknn
            if "CUDA" in normalized:
                return cuda
            return cpu

        person_hz = provider_rate(
            self.person_detector.provider_name,
            self.settings.person_detection_hz_cpu,
            self.settings.person_detection_hz_cuda,
            self.settings.person_detection_hz_rknn,
        )
        face_hz = provider_rate(
            self.face_backend.provider_name,
            self.settings.face_detection_hz_cpu,
            self.settings.face_detection_hz_cuda,
            self.settings.face_detection_hz_rknn,
        )
        try:
            reader.start()
            while not self._stop.is_set():
                if (
                    self.timeout_seconds is not None
                    and time.monotonic() - self.metrics.started_at >= self.timeout_seconds
                ):
                    self._transition(SearchStatus.TIMED_OUT, None)
                    self.events.publish(
                        "search_timeout",
                        {
                            "search_id": self.search_id,
                            "unfound_target_ids": list(self._active_targets),
                        },
                    )
                    break
                packet = reader.get(timeout=0.5)
                if packet is None:
                    if reader.ended.is_set():
                        break
                    continue
                started = time.monotonic()
                now = packet.captured_at
                self.metrics.frame_age_ms.append(
                    max(0.0, (time.monotonic() - packet.captured_at) * 1000.0)
                )
                if now - last_person_at >= 1.0 / max(person_hz, 0.1):
                    detections = self.person_detector.detect(packet.frame)
                    tracks = self._tracker.update(detections)
                    last_person_at = now
                faces = []
                if now - last_face_at >= 1.0 / max(face_hz, 0.1):
                    faces = self.face_backend.analyze(packet.frame, enrollment=False)
                    last_face_at = now
                faces_by_target = self._assign_faces_to_active_targets(faces)
                face_track_associations = associate_faces_to_tracks(faces, tracks)
                face_indices = {id(face): index for index, face in enumerate(faces)}

                track_states: dict[int, tuple[str, float]] = {}
                for target_id, target in list(self._active_targets.items()):
                    confirmation = self._confirmations[target_id]
                    local_associations: dict[int, int] = {}
                    for local_index, face in enumerate(faces_by_target[target_id]):
                        original_index = face_indices.get(id(face))
                        if original_index is not None:
                            track_id = face_track_associations.get(original_index)
                            if track_id is not None:
                                local_associations[local_index] = track_id
                    decisions = confirmation.process(
                        frame_id=packet.frame_id,
                        timestamp=now,
                        frame_shape=packet.frame.shape,
                        tracks=tracks,
                        faces=faces_by_target[target_id],
                        target=target,
                        face_track_associations=local_associations,
                    )
                    for decision in decisions:
                        event = SearchEvent(
                            search_id=self.search_id,
                            target_id=target.target_id,
                            target_name=target.name,
                            state=decision.state,
                            timestamp_ms=int(time.time() * 1000),
                            track_id=decision.track_id,
                            bbox=normalize_bbox(decision.bbox, packet.frame.shape),
                            similarity=decision.similarity,
                            quality=decision.quality,
                            evidence_count=decision.evidence_count,
                            model=self.face_backend.model_name,
                        )
                        payload = event.model_dump(mode="json")
                        self.events.publish(decision.state.value, payload)
                        with self._lock:
                            current = self._target_status[target_id]
                            current["best_similarity"] = max(
                                decision.similarity, current["best_similarity"] or -1.0
                            )
                            if decision.state.value == "confirmed":
                                current["status"] = "found"
                                current["found_at"] = payload["timestamp_ms"]
                                self.events.publish("target_found", payload)
                                self._active_targets.pop(target_id, None)
                    for track_id, (state, similarity) in confirmation.active_track_states().items():
                        previous = track_states.get(track_id)
                        if (
                            previous is None
                            or state.value == "confirmed"
                            or previous[0] != "confirmed"
                        ):
                            track_states[track_id] = (state.value, similarity)
                self._track_states = track_states
                all_targets_found = not self._active_targets
                if self._preview_due(now):
                    self._publish_preview(packet.frame, tracks, faces)
                    self._last_preview_at = now
                self.metrics.frame_count += 1
                self.metrics.latencies_ms.append((time.monotonic() - started) * 1000.0)
                if all_targets_found:
                    self._transition(SearchStatus.COMPLETED, None)
                    self.events.publish("all_found", {"search_id": self.search_id})
                    break
        except Exception as exc:  # noqa: BLE001 - the worker must fail closed and release resources
            self._transition(SearchStatus.FAILED, _safe_error(exc))
        finally:
            reader.stop()
            if self.status not in (
                SearchStatus.FAILED,
                SearchStatus.STOPPED,
                SearchStatus.COMPLETED,
                SearchStatus.TIMED_OUT,
            ):
                self._transition(SearchStatus.STOPPED, None)
            self.on_finished(self.search_id, [target.target_id for target in self.targets])

    def _assign_faces_to_active_targets(
        self, faces: list[FaceObservation]
    ) -> dict[str, list[FaceObservation]]:
        """Assign accepted faces to active targets with one matrix multiply.

        Batch searches may contain up to twenty targets.  The previous nested
        Python ``dot`` loop repeated work for every face; a small dense matrix
        multiply is both faster on ARM and guarantees that a completed target
        cannot win a face assignment.
        """

        active = list(self._active_targets.items())
        result: dict[str, list[FaceObservation]] = {target_id: [] for target_id, _ in active}
        accepted = [face for face in faces if face.accepted]
        if not active or not accepted:
            return result
        target_ids = [target_id for target_id, _ in active]
        target_embeddings = np.stack([target.embedding for _, target in active]).astype(
            np.float32, copy=False
        )
        face_embeddings = np.stack([face.embedding for face in accepted]).astype(
            np.float32, copy=False
        )
        similarities = face_embeddings @ target_embeddings.T
        winners = np.argmax(similarities, axis=1)
        best_scores = similarities[np.arange(len(accepted)), winners]
        threshold = self.settings.similarity_threshold
        for face, winner, similarity in zip(
            accepted, winners.tolist(), best_scores.tolist(), strict=True
        ):
            if similarity >= threshold:
                result[target_ids[winner]].append(face)
        return result

    def _publish_preview(
        self, frame: np.ndarray, tracks: list[Track], faces: list[FaceObservation]
    ) -> None:
        canvas = frame
        scale = 1.0
        settings = getattr(self, "settings", None)
        max_width = int(getattr(settings, "preview_max_width", 960))
        if max_width > 0 and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            canvas = cv2.resize(
                frame,
                (max_width, max(1, round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            # Drawing must never mutate the frame held by the capture queue.
            canvas = frame.copy()

        def scaled_box(box: np.ndarray) -> tuple[int, int, int, int]:
            return tuple(round(float(value) * scale) for value in box)  # type: ignore[return-value]

        for track in tracks:
            x1, y1, x2, y2 = scaled_box(track.bbox)
            state, similarity = self._track_states.get(track.track_id, ("tracking", 0.0))
            if state == "confirmed":
                color, label = (60, 220, 95), f"FOUND  {similarity:.2f}"
            elif state == "candidate":
                color, label = (0, 184, 255), f"CANDIDATE  {similarity:.2f}"
            else:
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
            )
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
            x1, y1, x2, y2 = scaled_box(face.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (232, 232, 232), 1)
        ok, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, int(getattr(settings, "preview_jpeg_quality", 82))],
        )
        if ok:
            self.preview.publish(encoded.tobytes())

    def _preview_due(self, timestamp: float) -> bool:
        if not self.settings.preview_enabled or self.settings.preview_fps <= 0:
            return False
        return timestamp - self._last_preview_at >= 1.0 / self.settings.preview_fps

    def _on_drop(self) -> None:
        with self._lock:
            self.metrics.dropped_frames += 1

    def _transition(self, status: SearchStatus, error: str | None) -> None:
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
            if status == SearchStatus.SOURCE_LOST:
                self.metrics.source_reconnects += 1
                # Track IDs and evidence belong to the old video timeline;
                # retaining them across a reconnect can produce a false
                # confirmation when a different person appears first.
                self._tracker.reset()
                for confirmation in self._confirmations.values():
                    confirmation.reset()
                self._track_states.clear()
        self.events.publish(
            "search_status",
            {"search_id": self.search_id, "status": status.value, "error": error},
        )


class SearchManager:
    def __init__(
        self,
        settings: Settings,
        face_backend: FaceBackend | None = None,
        person_detector: PersonDetector | None = None,
    ):
        self.settings = settings
        if settings.inference_backend == "rknn":
            self.face_backend = face_backend or RknnFaceBackend(settings)
            self.person_detector = person_detector or RknnPersonDetector(settings)
        else:
            self.face_backend = face_backend or InsightFaceBackend(settings)
            self.person_detector = person_detector or YoloXOnnxDetector(settings)
        self._targets: dict[str, Target] = {}
        self._sessions: dict[str, SearchSession] = {}
        self._finished_sessions: dict[str, SearchSession] = {}
        self._finished_session_order: deque[str] = deque()
        self._active_search_id: str | None = None
        self._lock = threading.RLock()

    def enroll(self, image: np.ndarray, name: str = "目标") -> TargetView:
        with self._lock:
            if len(self._targets) >= self.settings.max_enrolled_targets:
                raise _target_capacity_error()
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
            # The expensive analysis runs outside the manager lock. Recheck
            # here so simultaneous requests still cannot exceed the ceiling.
            if len(self._targets) >= self.settings.max_enrolled_targets:
                if embedding.flags.writeable:
                    embedding.fill(0)
                raise _target_capacity_error()
            self._targets[target_id] = Target(
                target_id=target_id, embedding=embedding, view=view, name=target_name
            )
        return view

    def delete_target(self, target_id: str) -> bool:
        with self._lock:
            if self._active_search_id:
                session = self._sessions.get(self._active_search_id)
                if session is not None and any(
                    target.target_id == target_id for target in session.targets
                ):
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

    def readiness(self) -> tuple[bool, dict[str, Any]]:
        """Load/check inference backends for an explicit readiness probe.

        ``/healthz`` remains a cheap liveness endpoint.  This method is used by
        ``/readyz`` and may perform model initialization, which is desirable
        during deployment because a missing RKNN artifact should fail before a
        search request arrives.
        """

        checks: dict[str, Any] = {
            "backend": self.settings.inference_backend,
            "face": {"provider": self.face_backend.provider_name, "ready": False},
            "person": {"provider": self.person_detector.provider_name, "ready": False},
        }
        ready = True
        for key, backend in (("face", self.face_backend), ("person", self.person_detector)):
            ensure_ready = getattr(backend, "ensure_ready", None)
            try:
                if ensure_ready is not None:
                    ensure_ready()
                checks[key] = {
                    "provider": backend.provider_name,
                    "ready": True,
                }
            except Exception as exc:  # noqa: BLE001 - readiness must explain the failure
                ready = False
                checks[key] = {
                    "provider": getattr(backend, "provider_name", "unknown"),
                    "ready": False,
                    "error": _safe_error(exc),
                }
        checks["ready"] = ready
        return ready, checks

    def start_search(self, target_id: str, source: SourceConfig) -> SearchView:
        return self.start_batch_search([target_id], source)

    def start_batch_search(
        self,
        target_ids: list[str],
        source: SourceConfig,
        timeout_seconds: float | None = None,
    ) -> SearchView:
        if not target_ids:
            raise PersonSearchError(
                "at least one target is required", code="invalid_targets", status_code=422
            )
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise PersonSearchError(
                "timeout_seconds must be positive", code="invalid_timeout", status_code=422
            )
        with self._lock:
            targets: list[Target] = []
            for target_id in target_ids:
                target = self._targets.get(target_id)
                if target is None:
                    raise PersonSearchError(
                        "target not found", code="target_not_found", status_code=404
                    )
                targets.append(target)
            if self._active_search_id is not None:
                raise PersonSearchError(
                    "only one search may run at a time",
                    code="search_capacity_exceeded",
                    status_code=409,
                )
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
            )
            self._sessions[search_id] = session
            self._active_search_id = search_id
            session.start()
            return session.view()

    def get_search(self, search_id: str) -> SearchView:
        return self._get_session(search_id).view()

    def get_session(self, search_id: str) -> SearchSession:
        return self._get_session(search_id)

    def stop_search(self, search_id: str) -> None:
        self._get_session(search_id).stop()

    def shutdown(self) -> None:
        with self._lock:
            active = self._active_search_id
        if active:
            self.stop_search(active)
        with self._lock:
            for session in (*self._sessions.values(), *self._finished_sessions.values()):
                session.release_sensitive_data()
            self._targets.clear()
            self._sessions.clear()
            self._finished_sessions.clear()
            self._finished_session_order.clear()
            self._active_search_id = None
        # RKNN sessions own native runtime handles/NPU resources. Release them
        # explicitly instead of relying on interpreter teardown, while keeping
        # shutdown best-effort so one vendor backend cannot block the other.
        released: set[int] = set()
        for backend in (self.face_backend, self.person_detector):
            if id(backend) in released:
                continue
            released.add(id(backend))
            release = getattr(backend, "release", None)
            if release is not None:
                try:
                    release()
                except Exception:  # noqa: BLE001,S110 - shutdown is best effort
                    pass

    def _get_session(self, search_id: str) -> SearchSession:
        with self._lock:
            session = self._sessions.get(search_id)
            if session is None:
                session = self._finished_sessions.get(search_id)
        if session is None:
            raise PersonSearchError("search not found", code="search_not_found", status_code=404)
        return session

    def _on_finished(self, search_id: str, target_ids: list[str]) -> None:
        with self._lock:
            if self._active_search_id == search_id:
                self._active_search_id = None
            for target_id in target_ids:
                self._targets.pop(target_id, None)
            session = self._sessions.pop(search_id, None)
            if session is not None:
                session.release_sensitive_data()
                self._finished_sessions[search_id] = session
                self._finished_session_order.append(search_id)
            while len(self._finished_session_order) > MAX_FINISHED_SESSIONS:
                expired_id = self._finished_session_order.popleft()
                self._finished_sessions.pop(expired_id, None)


def _sanitize_source(source: SourceConfig) -> SourceConfig:
    if source.type != SourceType.RTSP or not source.uri:
        return source.model_copy()
    parts = urlsplit(source.uri)
    host = parts.hostname or "source"
    port = f":{parts.port}" if parts.port else ""
    return SourceConfig(type=SourceType.RTSP, uri=f"{parts.scheme}://{host}{port}/***")


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


def _target_capacity_error() -> PersonSearchError:
    return PersonSearchError(
        "enrolled target capacity exceeded; delete an unused target and retry",
        code="target_capacity_exceeded",
        status_code=429,
    )


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ModelUnavailableError):
        return exc.message
    return f"{type(exc).__name__}: processing failed"
