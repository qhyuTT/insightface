from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import FakeFaceBackend, FakePersonDetector, make_face

from person_search.config import Settings
from person_search.domain import Detection, SearchStatus, SourceConfig, Target, TargetView, Track
from person_search.errors import EnrollmentError, PersonSearchError
from person_search.service import (
    PreviewHub,
    SearchManager,
    SearchSession,
    _merge_faces,
    _sanitize_source,
)


def test_enroll_requires_exactly_one_face() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    no_face = SearchManager(Settings(), FakeFaceBackend([]), FakePersonDetector())
    with pytest.raises(EnrollmentError, match="no face"):
        no_face.enroll(frame)

    multiple = SearchManager(
        Settings(),
        FakeFaceBackend([make_face(), make_face(bbox=(100, 20, 160, 80))]),
        FakePersonDetector(),
    )
    with pytest.raises(EnrollmentError, match="exactly one"):
        multiple.enroll(frame)


def test_enroll_rejects_quality_failure() -> None:
    manager = SearchManager(
        Settings(), FakeFaceBackend([make_face(accepted=False)]), FakePersonDetector()
    )
    with pytest.raises(EnrollmentError, match="face quality"):
        manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8))


def test_target_embedding_is_normalized_and_can_be_deleted() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target_view = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), " Alice ")
    assert target_view.name == "Alice"
    assert manager.get_target(target_view.target_id).name == "Alice"
    assert np.linalg.norm(manager.get_target(target_view.target_id).embedding) == pytest.approx(1.0)
    assert manager.delete_target(target_view.target_id)
    assert not manager.delete_target(target_view.target_id)
    with pytest.raises(PersonSearchError) as caught:
        manager.get_target(target_view.target_id)
    assert caught.value.code == "target_not_found"


def test_enroll_rejects_blank_target_name() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    with pytest.raises(EnrollmentError, match="target name"):
        manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "   ")


@pytest.mark.parametrize("state", ["candidate", "confirmed"])
def test_preview_only_draws_active_match_tracks(state: str) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    track = Track(7, np.asarray([20, 20, 80, 90], dtype=np.float32), 0.9)

    hidden = SimpleNamespace(_track_states={}, preview=PreviewHub())
    SearchSession._publish_preview(hidden, frame, [track], [])
    _, hidden_jpeg = hidden.preview.after(0, timeout=0)
    assert hidden_jpeg is not None
    hidden_frame = cv2.imdecode(np.frombuffer(hidden_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.max(hidden_frame) == 0

    visible = SimpleNamespace(_track_states={7: (state, 0.9)}, preview=PreviewHub())
    SearchSession._publish_preview(visible, frame, [track], [])
    _, visible_jpeg = visible.preview.after(0, timeout=0)
    assert visible_jpeg is not None
    visible_frame = cv2.imdecode(np.frombuffer(visible_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.max(visible_frame) > 0


def test_person_roi_faces_translate_bbox_and_landmarks_to_frame_coordinates() -> None:
    face = make_face(bbox=(5, 7, 45, 47))
    face.landmarks = np.asarray([[10, 12], [30, 12], [20, 22]], dtype=np.float32)
    backend = FakeFaceBackend([face])
    session = SimpleNamespace(
        settings=Settings(roi_max_tracks_per_pass=8, roi_min_person_height_px=120),
        face_backend=backend,
    )
    frame = np.zeros((400, 500, 3), dtype=np.uint8)
    track = Track(7, np.asarray([100, 80, 300, 320], dtype=np.float32), 0.9)

    observations = SearchSession._analyze_person_rois(session, frame, [track])

    assert backend.calls == 1
    np.testing.assert_array_equal(observations[0].bbox, [105, 87, 145, 127])
    np.testing.assert_array_equal(
        observations[0].landmarks,
        [[110, 92], [130, 92], [120, 102]],
    )
    np.testing.assert_array_equal(face.bbox, [5, 7, 45, 47])


def test_person_roi_pass_uses_top_eight_valid_tracks_by_score() -> None:
    backend = FakeFaceBackend([])
    session = SimpleNamespace(
        settings=Settings(roi_max_tracks_per_pass=8, roi_min_person_height_px=120),
        face_backend=backend,
    )
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    tracks = [
        Track(
            track_id=index,
            bbox=np.asarray([index * 10, 10, index * 10 + 50, 210], dtype=np.float32),
            score=float(index),
        )
        for index in range(10)
    ]

    SearchSession._analyze_person_rois(session, frame, tracks)

    assert backend.calls == 8


def test_merge_faces_replaces_duplicate_with_higher_quality_roi_result() -> None:
    full_frame = make_face(bbox=(10, 10, 70, 70), quality=0.6)
    better_roi = make_face(bbox=(12, 12, 72, 72), quality=0.9)
    distinct = make_face(bbox=(120, 20, 180, 80), quality=0.7)

    merged = _merge_faces([full_frame], [better_roi, distinct])

    assert len(merged) == 2
    assert merged[0] is better_roi
    assert merged[1] is distinct


def test_merge_faces_prefers_accepted_observation_over_higher_scored_rejection() -> None:
    accepted = make_face(bbox=(10, 10, 70, 70), quality=0.5)
    rejected = make_face(bbox=(11, 11, 71, 71), accepted=False, quality=0.99)

    merged = _merge_faces([accepted], [rejected])
    replaced = _merge_faces([rejected], [accepted])

    assert merged[0] is accepted
    assert replaced[0] is accepted


def test_face_metrics_and_rtsp_sanitization_keep_new_diagnostics_and_source_options() -> None:
    session = SimpleNamespace(
        _lock=threading.RLock(),
        settings=Settings(preferred_search_face_px=80),
        metrics=SimpleNamespace(
            face_observations=0,
            accepted_faces=0,
            small_faces=0,
            rejection_counts={},
        ),
    )
    small = make_face(bbox=(0, 0, 64, 70))
    rejected = make_face(accepted=False)

    SearchSession._record_face_metrics(session, [small, rejected])

    assert session.metrics.face_observations == 2
    assert session.metrics.accepted_faces == 1
    assert session.metrics.small_faces == 1
    assert session.metrics.rejection_counts == {"face_blurry": 1}

    sanitized = _sanitize_source(
        SourceConfig(
            type="rtsp",
            uri="rtsp://user:secret@camera.test:8554/live/path?token=hidden",
            debug_preview=True,
        )
    )
    assert sanitized.uri == "rtsp://camera.test:8554/***"
    assert sanitized.debug_preview is True


def test_found_target_is_removed_while_other_targets_continue(monkeypatch) -> None:
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    packets = [
        SimpleNamespace(frame_id=1, captured_at=1.0, frame=frame),
        SimpleNamespace(frame_id=2, captured_at=1.3, frame=frame),
    ]

    class FakeReader:
        def __init__(self, source, settings, on_status, on_drop):
            self.ended = threading.Event()

        def start(self) -> None:
            pass

        def get(self, timeout=0.5):
            if packets:
                return packets.pop(0)
            self.ended.set()
            return None

        def stop(self) -> None:
            pass

    monkeypatch.setattr("person_search.service.LatestFrameReader", FakeReader)
    target_view_1 = TargetView(
        target_id="target-1",
        name="张三",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target_view_2 = target_view_1.model_copy(update={"target_id": "target-2", "name": "李四"})
    targets = [
        Target("target-1", np.asarray([1.0, 0.0], dtype=np.float32), target_view_1, "张三"),
        Target("target-2", np.asarray([0.0, 1.0], dtype=np.float32), target_view_2, "李四"),
    ]
    detector = FakePersonDetector([Detection(np.asarray([0, 0, 110, 119], dtype=np.float32), 0.99)])
    session = SearchSession(
        search_id="search-1",
        target=targets[0],
        targets=targets,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(evidence_required=1),
        face_backend=FakeFaceBackend([make_face()]),
        person_detector=detector,
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    events = session.events.after(0, timeout=0)
    found = [event for event in events if event["type"] == "target_found"]
    assert [event["data"]["target_id"] for event in found] == ["target-1"]
    assert list(session._active_targets) == ["target-2"]
    view = session.view()
    assert view.found_count == 1
    assert view.unfound_target_ids == ["target-2"]


def test_unassociated_small_face_confirms_through_face_fallback(monkeypatch) -> None:
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    packets = [
        SimpleNamespace(frame_id=index, captured_at=timestamp, frame=frame)
        for index, timestamp in enumerate((0.0, 0.25, 0.5, 0.75))
    ]

    class FakeReader:
        def __init__(self, source, settings, on_status, on_drop):
            self.ended = threading.Event()

        def start(self) -> None:
            pass

        def get(self, timeout=0.5):
            if packets:
                return packets.pop(0)
            self.ended.set()
            return None

        def stop(self) -> None:
            pass

    monkeypatch.setattr("person_search.service.LatestFrameReader", FakeReader)
    target_view = TargetView(
        target_id="target-1",
        name="张三",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        "target-1",
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        "张三",
    )
    session = SearchSession(
        search_id="search-fallback",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(
            small_face_evidence_required=4,
            small_face_evidence_window_seconds=2.0,
        ),
        face_backend=FakeFaceBackend([make_face(bbox=(20, 20, 84, 84))]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    events = session.events.after(0, timeout=0)
    assert not any(event["type"] == "candidate" for event in events)
    confirmed = [event for event in events if event["type"] == "confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0]["data"]["association"] == "face_fallback"
    assert confirmed[0]["data"]["track_id"] < 0
    assert confirmed[0]["data"]["evidence_count"] == 4
    assert session._track_states[confirmed[0]["data"]["track_id"]][0] == "confirmed"
    assert session.status == SearchStatus.COMPLETED
    view = session.view()
    assert view.face_observations == 4
    assert view.accepted_faces == 4
    assert view.small_faces == 4
    assert view.unassociated_faces == 0
    assert view.association_counts == {"face_fallback": 4}


def test_start_failure_rolls_back_session_and_active_slot(monkeypatch) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")

    def fail_start(_: SearchSession) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(SearchSession, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        manager.start_batch_search(
            [target.target_id],
            SourceConfig(type="camera", device_index=0),
            request_id="request-start-failure",
        )

    assert manager.active_search() is None
    assert manager._sessions == {}
    assert manager.get_target(target.target_id).target_id == target.target_id


def test_active_search_is_idempotent_by_request_id(monkeypatch) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    starts: list[str] = []
    monkeypatch.setattr(SearchSession, "start", lambda session: starts.append(session.search_id))

    first = manager.start_batch_search(
        [target.target_id],
        SourceConfig(type="camera", device_index=0),
        request_id="request-idempotent",
    )
    repeated = manager.start_batch_search(
        [target.target_id],
        SourceConfig(type="camera", device_index=0),
        request_id="request-idempotent",
    )

    assert repeated.search_id == first.search_id
    assert repeated.request_id == "request-idempotent"
    assert starts == [first.search_id]
    assert list(manager._sessions) == [first.search_id]


def test_search_lookup_by_request_id_survives_terminal_transition(monkeypatch) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    monkeypatch.setattr(SearchSession, "start", lambda session: None)

    created = manager.start_batch_search(
        [target.target_id],
        SourceConfig(type="camera", device_index=0),
        request_id="request-lookup",
    )
    session = manager.get_session(created.search_id)
    session._transition(SearchStatus.TIMED_OUT, None, publish=False)
    manager._on_finished(created.search_id, [target.target_id])

    found = manager.search_by_request_id(" request-lookup ")
    assert found is not None
    assert found.search_id == created.search_id
    assert found.request_id == "request-lookup"
    assert found.status == SearchStatus.TIMED_OUT
    assert manager.search_by_request_id("missing") is None


def test_terminal_target_found_event_can_immediately_start_next_search(monkeypatch) -> None:
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    class ControlledReader:
        def __init__(self, source, settings, on_status, on_drop):
            self.source = source
            self.ended = threading.Event()
            self.stopped = threading.Event()
            self.returned_frame = False

        def start(self) -> None:
            pass

        def get(self, timeout=0.5):
            if self.source.device_index == 0 and not self.returned_frame:
                self.returned_frame = True
                return SimpleNamespace(frame_id=1, captured_at=1.0, frame=frame)
            self.stopped.wait(0.01)
            return None

        def stop(self) -> None:
            self.stopped.set()
            self.ended.set()

    monkeypatch.setattr("person_search.service.LatestFrameReader", ControlledReader)
    settings = Settings(evidence_required=1)
    detector = FakePersonDetector(
        [Detection(np.asarray([0, 0, 110, 119], dtype=np.float32), 0.99)]
    )
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), detector)
    enrollment_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    first_target = manager.enroll(enrollment_frame, "张三")
    next_target = manager.enroll(enrollment_frame, "李四")
    original_start = SearchSession.start
    active_at_terminal_event = []
    replacement_views = []
    replacement_errors: list[Exception] = []

    def start_with_terminal_hook(session: SearchSession) -> None:
        if session.source.device_index == 0:
            original_publish = session.events.publish

            def publish(event_type: str, data: dict):
                if event_type == "target_found" and not active_at_terminal_event:
                    active_at_terminal_event.append(manager.active_search())
                    try:
                        replacement_views.append(
                            manager.start_batch_search(
                                [next_target.target_id],
                                SourceConfig(type="camera", device_index=1),
                                replace_active=True,
                                request_id="request-next",
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - asserted below
                        replacement_errors.append(exc)
                return original_publish(event_type, data)

            session.events.publish = publish  # type: ignore[method-assign]
        original_start(session)

    monkeypatch.setattr(SearchSession, "start", start_with_terminal_hook)
    first = manager.start_batch_search(
        [first_target.target_id],
        SourceConfig(type="camera", device_index=0),
        request_id="request-first",
    )
    first_session = manager.get_session(first.search_id)

    assert first_session.finished.wait(timeout=2.0)
    assert active_at_terminal_event == [None]
    assert replacement_errors == []
    assert len(replacement_views) == 1
    replacement = replacement_views[0]
    assert manager.active_search().search_id == replacement.search_id  # type: ignore[union-attr]
    assert [
        event["type"] for event in first_session.events.after(0, timeout=0)
    ][-3:] == ["search_status", "target_found", "all_found"]

    manager.stop_search(replacement.search_id)
    assert manager.active_search() is None
