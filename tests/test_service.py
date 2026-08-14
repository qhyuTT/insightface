from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import FakeFaceBackend, FakePersonDetector, make_face

from person_search.config import Settings
from person_search.domain import Detection, SourceConfig, Target, TargetView, Track
from person_search.errors import EnrollmentError, PersonSearchError
from person_search.service import PreviewHub, SearchManager, SearchSession


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
