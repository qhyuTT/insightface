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
    associate_faces_to_tracks_detailed,
    default_face_match_policy,
    fallback_face_match_policy,
    normalize_bbox,
)
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
                "last_rejection_reason": None,
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
        self._debug_faces: list[tuple[FaceObservation, str, float | None]] = []
        self._finished = threading.Event()
        self._stop_requested = False
        self._deferred_events: list[tuple[str, dict[str, Any]]] = []

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
        self._transition(SearchStatus.STOPPING, None)
        if self._reader:
            self._reader.stop()
        if (
            self._worker
            and self._worker is not threading.current_thread()
            and not self._finished.wait(timeout=timeout)
        ):
            raise SearchStopTimeoutError("搜索线程未能在停止时限内退出；请稍后重试或重启识别服务")

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
                request_id=self.request_id,
                effective_config=self._effective_config(),
                **metrics,
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
            "roi_face_detection_size": settings.roi_face_detection_size,
            "roi_max_tracks_per_pass": settings.roi_max_tracks_per_pass,
            "similarity_threshold": settings.similarity_threshold,
            "small_face_similarity_threshold": settings.small_face_similarity_threshold,
            "tiny_face_enabled": settings.tiny_face_enabled,
            "tiny_face_similarity_threshold": settings.tiny_face_similarity_threshold,
            "effective_search_min_face_px": settings.effective_search_min_face_px,
            "min_search_face_px": settings.min_search_face_px,
            "preferred_search_face_px": settings.preferred_search_face_px,
            "min_search_blur_variance": settings.min_search_blur_variance,
            "face_detection_threshold": settings.face_detection_threshold,
            "evidence_required": settings.evidence_required,
            "evidence_window_seconds": settings.evidence_window_seconds,
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
                # The per-frame budget. Optional stages consult what is left of it,
                # so a slow stage can no longer drag the loop below the sampling
                # density the confirmation window needs.
                budget_seconds = 1.0 / self.settings.target_loop_hz
                if now - last_person_at >= 1.0 / max(person_hz, 0.1):
                    stage_started = time.monotonic()
                    detections = self.person_detector.detect(packet.frame)
                    tracks = self._tracker.update(detections)
                    self._record_stage("person", stage_started)
                    last_person_at = now
                faces = []
                if now - last_face_at >= 1.0 / max(face_hz, 0.1):
                    stage_started = time.monotonic()
                    faces = self.face_backend.detect_faces(packet.frame, enrollment=False)
                    self._record_stage("face_full", stage_started)
                    self._record_face_source("full_frame", len(faces))
                    last_face_at = now
                    roi_clock = time.monotonic()
                    roi_tracks = self._tracks_needing_roi_face_pass(faces, tracks)
                    if (
                        roi_face_hz > 0
                        and roi_clock - last_roi_face_at >= 1.0 / roi_face_hz
                        and roi_tracks
                        and self._roi_fits_budget(started, budget_seconds)
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
                policies_by_face: dict[int, FaceMatchPolicy] = {
                    face_index: default_face_match_policy(face, self.settings)
                    for face_index, face in enumerate(accepted_faces)
                }
                for face_index, mode in list(modes_by_face.items()):
                    if policies_by_face[face_index].requires_strict_association:
                        if mode != "person_strict":
                            association_by_face.pop(face_index, None)
                            modes_by_face.pop(face_index, None)
                        continue
                    if mode == "person_relaxed":
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
                    self._handle_decisions(target_id, target, decisions, packet.frame.shape)
                    progress = confirmation.track_progress()
                    if progress:
                        _, (evidence_count, required_evidence) = max(
                            progress.items(), key=lambda item: item[1][0]
                        )
                        with self._lock:
                            current = self._target_status[target_id]
                            current["evidence_count"] = evidence_count
                            current["required_evidence"] = required_evidence
                    else:
                        with self._lock:
                            self._target_status[target_id]["evidence_count"] = 0
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
                with self._lock:
                    self.metrics.frame_count += 1
                    self.metrics.latencies_ms.append((time.monotonic() - started) * 1000.0)
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
        frame_shape: tuple[int, ...],
    ) -> None:
        if target is None:
            return
        for decision in decisions:
            event = SearchEvent(
                search_id=self.search_id,
                target_id=target.target_id,
                target_name=target.name,
                state=decision.state,
                timestamp_ms=int(time.time() * 1000),
                track_id=decision.track_id,
                bbox=normalize_bbox(decision.bbox, frame_shape),
                similarity=decision.similarity,
                quality=decision.quality,
                evidence_count=decision.evidence_count,
                model=self.face_backend.model_name,
                association=decision.association,
            )
            payload = event.model_dump(mode="json")
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

    def _record_rejected_observations(self, faces: list[FaceObservation]) -> None:
        """Record why rejected faces were dropped, without paying for an embedding.

        Similarity is unknowable for these faces by design — they never reach
        ArcFace — but "a face this big was rejected for this reason" is the more
        actionable half of the diagnostic anyway.
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

    def _is_face_matchable(self, face: FaceObservation) -> bool:
        return bool(face.accepted and face.short_side >= self.settings.effective_search_min_face_px)

    def _stage_p95_ms(self, stage: str) -> float:
        with self._lock:
            latencies = self.metrics.stage_latencies_ms.get(stage)
            if not latencies:
                return 0.0
            return float(np.percentile(latencies[-200:], 95))

    def _record_budget_skip(self, stage: str) -> None:
        with self._lock:
            self.metrics.budget_skips[stage] = self.metrics.budget_skips.get(stage, 0) + 1

    def _roi_fits_budget(self, started: float, budget_seconds: float) -> bool:
        """Run the ROI pass only when its measured cost still fits this frame.

        The old Hz throttle was an interval *floor*, so once a single iteration
        exceeded 1/hz it passed unconditionally and ROI ran on every frame at 100%
        duty. This is the missing ceiling: past the budget, ROI is skipped rather
        than allowed to starve the confirmation window of samples.
        """
        elapsed = time.monotonic() - started
        remaining = budget_seconds - elapsed
        # Never let the floor be violated even if the budget is already blown.
        hard_ceiling = 1.0 / self.settings.min_processed_fps
        if elapsed >= hard_ceiling:
            self._record_budget_skip("face_roi")
            return False
        estimated = self._stage_p95_ms("face_roi") / 1000.0
        if estimated and estimated > max(remaining, 0.0):
            self._record_budget_skip("face_roi")
            return False
        return True

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
            # later, after dedup against the full-frame results.
            found = self.face_backend.detect_faces(
                roi,
                enrollment=False,
                detection_size=self.settings.roi_face_detection_size,
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
            self.events.publish(event_type, payload)
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
            self._targets.clear()

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
