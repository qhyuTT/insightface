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
from .confirmation import TrackConfirmation, normalize_bbox
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
    TargetView,
    Track,
)
from .errors import EnrollmentError, ModelUnavailableError, PersonSearchError
from .quality import normalize_embedding
from .tracker import ByteTracker
from .video import LatestFrameReader

MAX_TARGET_NAME_LENGTH = 80


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
    target: Target
    source: SourceConfig
    settings: Settings
    face_backend: FaceBackend
    person_detector: PersonDetector
    on_finished: Callable[[str, str], None]

    def __post_init__(self) -> None:
        self.status = SearchStatus.INITIALIZING
        self.error: str | None = None
        self.metrics = SearchMetrics()
        self.events = EventHub()
        self.preview = PreviewHub()
        self._tracker = ByteTracker()
        self._confirmation = TrackConfirmation(self.settings)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._reader: LatestFrameReader | None = None
        self._lock = threading.RLock()
        self._track_states: dict[int, tuple[str, float]] = {}

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
            return SearchView(
                search_id=self.search_id,
                target_id=self.target.target_id,
                target_name=self.target.name,
                status=self.status,
                source=_sanitize_source(self.source),
                provider=f"face={self.face_backend.provider_name},person={self.person_detector.provider_name}",
                error=self.error,
                **metrics,
            )

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
        is_cuda = "CUDA" in (
            self.face_backend.provider_name + self.person_detector.provider_name
        )
        person_hz = (
            self.settings.person_detection_hz_cuda
            if is_cuda
            else self.settings.person_detection_hz_cpu
        )
        face_hz = (
            self.settings.face_detection_hz_cuda
            if is_cuda
            else self.settings.face_detection_hz_cpu
        )
        try:
            reader.start()
            while not self._stop.is_set():
                packet = reader.get(timeout=0.5)
                if packet is None:
                    if reader.ended.is_set():
                        break
                    continue
                started = time.monotonic()
                now = packet.captured_at
                if now - last_person_at >= 1.0 / max(person_hz, 0.1):
                    detections = self.person_detector.detect(packet.frame)
                    tracks = self._tracker.update(detections)
                    last_person_at = now
                faces = []
                if now - last_face_at >= 1.0 / max(face_hz, 0.1):
                    faces = self.face_backend.analyze(packet.frame, enrollment=False)
                    last_face_at = now
                decisions = self._confirmation.process(
                    frame_id=packet.frame_id,
                    timestamp=now,
                    frame_shape=packet.frame.shape,
                    tracks=tracks,
                    faces=faces,
                    target=self.target,
                )
                for decision in decisions:
                    self._track_states[decision.track_id] = (
                        decision.state.value,
                        decision.similarity,
                    )
                    event = SearchEvent(
                        search_id=self.search_id,
                        target_id=self.target.target_id,
                        target_name=self.target.name,
                        state=decision.state,
                        timestamp_ms=int(time.time() * 1000),
                        track_id=decision.track_id,
                        bbox=normalize_bbox(decision.bbox, packet.frame.shape),
                        similarity=decision.similarity,
                        quality=decision.quality,
                        evidence_count=decision.evidence_count,
                        model=self.face_backend.model_name,
                    )
                    self.events.publish(decision.state.value, event.model_dump(mode="json"))
                self._publish_preview(packet.frame, tracks, faces)
                self.metrics.frame_count += 1
                self.metrics.latencies_ms.append((time.monotonic() - started) * 1000.0)
        except Exception as exc:  # noqa: BLE001 - the worker must fail closed and release resources
            self._transition(SearchStatus.FAILED, _safe_error(exc))
        finally:
            reader.stop()
            if self.status not in (SearchStatus.FAILED, SearchStatus.STOPPED):
                self._transition(SearchStatus.STOPPED, None)
            self.on_finished(self.search_id, self.target.target_id)

    def _publish_preview(
        self, frame: np.ndarray, tracks: list[Track], faces: list[FaceObservation]
    ) -> None:
        canvas = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = (int(value) for value in track.bbox)
            state, similarity = self._track_states.get(track.track_id, ("tracking", 0.0))
            if state == "confirmed":
                color, label = (60, 220, 95), f"FOUND  {similarity:.2f}"
            elif state == "candidate":
                color, label = (0, 184, 255), f"CANDIDATE  {similarity:.2f}"
            else:
                color, label = (255, 190, 55), f"PERSON  #{track.track_id}"
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
            x1, y1, x2, y2 = (int(value) for value in face.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (232, 232, 232), 1)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            self.preview.publish(encoded.tobytes())

    def _on_drop(self) -> None:
        with self._lock:
            self.metrics.dropped_frames += 1

    def _transition(self, status: SearchStatus, error: str | None) -> None:
        with self._lock:
            if self.status in (SearchStatus.STOPPED, SearchStatus.FAILED):
                return
            if status == self.status and error == self.error:
                return
            self.status = status
            self.error = error
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
        self.face_backend = face_backend or InsightFaceBackend(settings)
        self.person_detector = person_detector or YoloXOnnxDetector(settings)
        self._targets: dict[str, Target] = {}
        self._sessions: dict[str, SearchSession] = {}
        self._active_search_id: str | None = None
        self._lock = threading.RLock()

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
                if session.target.target_id == target_id:
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
        with self._lock:
            target = self._targets.get(target_id)
            if target is None:
                raise PersonSearchError("target not found", code="target_not_found", status_code=404)
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
                target=target,
                source=source,
                settings=self.settings,
                face_backend=self.face_backend,
                person_detector=self.person_detector,
                on_finished=self._on_finished,
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
            self._targets.clear()

    def _get_session(self, search_id: str) -> SearchSession:
        with self._lock:
            session = self._sessions.get(search_id)
        if session is None:
            raise PersonSearchError("search not found", code="search_not_found", status_code=404)
        return session

    def _on_finished(self, search_id: str, target_id: str) -> None:
        with self._lock:
            if self._active_search_id == search_id:
                self._active_search_id = None
            self._targets.pop(target_id, None)


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


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ModelUnavailableError):
        return exc.message
    return f"{type(exc).__name__}: processing failed"
