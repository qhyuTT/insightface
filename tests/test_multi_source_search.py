from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from conftest import FakeFaceBackend, FakePersonDetector, make_face
from fastapi.testclient import TestClient

from person_search.api import create_app
from person_search.config import Settings
from person_search.domain import SearchCreate, SearchStatus, SearchView, SourceConfig
from person_search.errors import PersonSearchError
from person_search.service import MultiSourceSearchSession, PreviewHub, SearchManager, SearchSession
from person_search.video import FramePacket


def _manager() -> tuple[SearchManager, str]:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    return manager, target.target_id


def _sources() -> list[SourceConfig]:
    return [
        SourceConfig(type="rtsp", uri="rtsp://first:secret@camera-one/live"),
        SourceConfig(type="rtsp", uri="rtsp://second:secret@camera-two/live"),
    ]


def test_two_sources_share_one_search_but_not_tracking_state(monkeypatch) -> None:
    monkeypatch.setattr(SearchSession, "start", lambda self: None)

    def finish_stop(self, timeout=15.0):
        self.status = SearchStatus.STOPPED
        self._finished.set()

    monkeypatch.setattr(SearchSession, "stop", finish_stop)
    manager, target_id = _manager()
    created = manager.start_batch_search([target_id], _sources(), request_id="two-cameras")
    session = manager.get_session(created.search_id)
    assert isinstance(session, MultiSourceSearchSession)
    assert created.search_id == manager.active_search().search_id
    assert len(created.sources) == 2
    assert created.sources[0]["source"]["uri"] == "rtsp://camera-one/***"
    assert created.sources[1]["source"]["uri"] == "rtsp://camera-two/***"
    first, second = session._children.values()
    assert first.source != second.source
    assert first._tracker is not second._tracker
    assert first._confirmations[target_id] is not second._confirmations[target_id]
    first.metrics.track_sampling_hz.append(2.0)
    second.metrics.track_sampling_hz.append(4.0)
    first.metrics.unconfirmed_gate_counts["insufficient_samples"] = 1
    second.metrics.unconfirmed_gate_counts["aggregate_low"] = 2
    diagnostics = session.view()
    assert [item["achieved_sampling_hz"] for item in diagnostics.sources] == [2.0, 4.0]
    assert diagnostics.achieved_sampling_hz == 0.0  # A per-source rate cannot be summed.
    assert diagnostics.unconfirmed_gate_counts == {
        "insufficient_samples": 1,
        "aggregate_low": 2,
    }
    assert session.preview_for("source-2") is second.preview
    with pytest.raises(PersonSearchError, match="source not found"):
        session.preview_for("source-3")

    second._target_status[target_id]["status"] = "found"
    second._active_targets.clear()
    second.events.publish("target_found", {"search_id": created.search_id, "target_id": target_id})
    second.status = SearchStatus.COMPLETED
    second._finished.set()

    assert session.finished.wait(2)
    assert first.finished.is_set()
    assert session.status == SearchStatus.COMPLETED
    assert manager.active_search() is None
    view = manager.get_search(created.search_id)
    assert view.found_count == 1
    assert [event["data"]["source_id"] for event in session.events.after(0, 0) if event["type"] == "target_found"] == ["source-2"]


def test_dual_rknn_contexts_are_source_owned_reused_and_closed(monkeypatch) -> None:
    from person_search import rknn_backends

    created = []

    class FakeRknnYoloXDetector:
        provider_name = "RKNN_NPU"
        static_input_hw = (640, 640)
        core_mask_applied = True

        def __init__(self, settings, confidence=0.25, nms=0.45, *, model_factory=None):
            self.settings = settings
            self.confidence = confidence
            self.nms = nms
            self._model_factory = model_factory
            self.closed = False
            created.append(self)

        @property
        def core_mask(self):
            return self.settings.rknn_core_mask

        def ensure_ready(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(rknn_backends, "RknnYoloXDetector", FakeRknnYoloXDetector)
    monkeypatch.setattr(SearchSession, "start", lambda self: None)

    def finish_stop(self, timeout=15.0):
        self.status = SearchStatus.STOPPED
        self._finished.set()

    monkeypatch.setattr(SearchSession, "stop", finish_stop)
    settings = Settings(rknn_core_mask=7)
    base = FakeRknnYoloXDetector(settings)
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), base)
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    search = manager.start_batch_search([target.target_id], _sources())
    session = manager.get_session(search.search_id)
    first, second = session._children.values()
    assert first.person_detector is created[1]
    assert second.person_detector is created[2]
    assert [child._effective_config()["rknn_core_mask"] for child in (first, second)] == [1, 2]
    assert base.core_mask == 7
    manager.stop_search(search.search_id)
    next_target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    manager.start_batch_search([next_target.target_id], _sources())
    assert len(created) == 3
    manager.shutdown()
    assert all(detector.closed for detector in created)


def test_dual_rknn_context_failure_falls_back_to_shared_detector(monkeypatch) -> None:
    from person_search import rknn_backends

    created = []

    class FakeRknnYoloXDetector:
        provider_name = "RKNN_NPU"
        static_input_hw = (640, 640)

        def __init__(self, settings, confidence=0.25, nms=0.45, *, model_factory=None):
            self.settings = settings
            self.confidence = confidence
            self.nms = nms
            self._model_factory = model_factory
            self.closed = False
            created.append(self)

        @property
        def core_mask_applied(self):
            return self.settings.rknn_core_mask != 2

        def ensure_ready(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(rknn_backends, "RknnYoloXDetector", FakeRknnYoloXDetector)
    settings = Settings(rknn_core_mask=7)
    base = FakeRknnYoloXDetector(settings)
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), base)
    assert manager._person_detectors_for_dual_source() is None
    assert all(detector.closed for detector in created[1:])
    assert not base.closed
    manager.shutdown()
    assert base.closed


def test_explicit_stop_cleans_both_video_sources(monkeypatch) -> None:
    monkeypatch.setattr(SearchSession, "start", lambda self: None)
    stopped = []

    def finish_stop(self, timeout=15.0):
        stopped.append(self.source.device_index)
        self.status = SearchStatus.STOPPED
        self._finished.set()

    monkeypatch.setattr(SearchSession, "stop", finish_stop)
    manager, target_id = _manager()
    sources = [SourceConfig(type="camera", device_index=index) for index in (0, 1)]
    created = manager.start_batch_search([target_id], sources)
    manager.stop_search(created.search_id)
    assert sorted(stopped) == [0, 1]
    assert manager.get_search(created.search_id).status == SearchStatus.STOPPED


def test_invalid_two_source_requests_are_rejected_without_uri_leak() -> None:
    sources = _sources()
    assert SearchCreate(target_id="target", sources=sources).source is None
    with pytest.raises(ValueError):
        SearchCreate(target_id="target", sources=[sources[0]])
    with pytest.raises(ValueError):
        SearchCreate(target_id="target", source=sources[0], sources=sources)
    with pytest.raises(ValueError):
        SearchCreate(target_id="target", sources=[sources[0], sources[0]])


def test_two_source_batch_rejects_multiple_targets() -> None:
    manager, target_id = _manager()
    with pytest.raises(PersonSearchError, match="one target"):
        manager.start_batch_search([target_id, "another"], _sources())
    assert manager.active_search() is None


def test_both_capture_workers_process_frames_in_one_session(monkeypatch) -> None:
    from person_search import service

    readers = []

    class FakeReader:
        def __init__(self, source, settings, on_status, on_drop):
            self.source = source
            self.on_status = on_status
            self.ended = threading.Event()
            self.stopped = threading.Event()
            self.frame_id = 0
            readers.append(self)

        def start(self):
            self.on_status(SearchStatus.RUNNING, None)

        def get(self, timeout=0.5):
            if self.stopped.wait(0.02):
                return None
            self.frame_id += 1
            return FramePacket(
                self.frame_id,
                time.monotonic(),
                np.zeros((128, 128, 3), dtype=np.uint8),
            )

        def stop(self):
            self.stopped.set()
            self.ended.set()

        def clear(self):
            pass

    monkeypatch.setattr(service, "LatestFrameReader", FakeReader)
    manager, target_id = _manager()
    manager.face_backend.observations = []
    created = manager.start_batch_search(
        [target_id],
        [SourceConfig(type="camera", device_index=index) for index in (0, 1)],
    )
    deadline = time.monotonic() + 3
    while (len(readers) < 2 or any(reader.frame_id == 0 for reader in readers)) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(readers) == 2
    assert all(reader.frame_id > 0 for reader in readers)
    assert {reader.source.device_index for reader in readers} == {0, 1}
    manager.stop_search(created.search_id)
    assert all(reader.stopped.is_set() for reader in readers)
    assert manager.get_search(created.search_id).status == SearchStatus.STOPPED


def test_api_accepts_two_sources_and_routes_previews_by_source_id() -> None:
    selected = []
    previews = {"source-1": PreviewHub(), "source-2": PreviewHub()}
    previews["source-1"].publish(b"first-jpeg")
    previews["source-2"].publish(b"second-jpeg")

    class StubManager:
        def start_batch_search(self, target_ids, source, *args):
            selected.append(source)
            return SearchView(search_id="search-1", status=SearchStatus.STOPPED, source=source[0])

        def get_session(self, search_id):
            assert search_id == "search-1"
            return self

        def preview_for(self, source_id):
            return previews[source_id]

        status = SearchStatus.STOPPED

        def shutdown(self):
            pass

    with TestClient(create_app(Settings(), StubManager())) as client:
        response = client.post(
            "/v1/searches",
            json={
                "target_id": "target-1",
                "sources": [source.model_dump() for source in _sources()],
            },
        )
        preview = client.get("/v1/searches/search-1/preview.mjpg?source_id=source-2")
    assert response.status_code == 201
    assert len(selected[0]) == 2
    assert preview.status_code == 200
    assert b"second-jpeg" in preview.content
    assert b"first-jpeg" not in preview.content


def test_two_source_metrics_keep_latency_per_source_and_sum_rates(monkeypatch) -> None:
    from types import SimpleNamespace
    monkeypatch.setattr(SearchSession, 'start', lambda self: None)
    manager, target_id = _manager()
    created = manager.start_batch_search([target_id], _sources())
    session = manager.get_session(created.search_id)
    first, second = session._children.values()
    for child, fps, age in ((first, 6.0, 15.0), (second, 12.0, 180.0)):
        child._reader = SimpleNamespace(metrics_snapshot=lambda fps=fps: {
            'decode_fps': fps, 'decoded_frames': int(fps * 10),
            'queue_length': 1, 'queue_capacity': 2,
            'capture_p95_latency_ms': 30.0,
        })
        child.metrics.preview_frame_ages_ms.append(age)
        child.metrics.dropped_frames = 3
    view = session.view()
    assert view.decode_fps == view.source_fps == 18.0
    assert view.decoded_frames == 180
    assert view.drop_rate == pytest.approx(6 / 180)
    assert view.sources[0]["drop_rate"] == pytest.approx(3 / 60)
    assert "target_id" not in view.sources[0]
    assert "request_id" not in view.sources[0]
    assert view.queue_length == 2
    assert view.queue_capacity == 4
    assert view.latency_scope == 'per_source'
    assert view.preview_frame_age_p95_ms == 0.0
    assert view.stage_p95_latency_ms == {}
    assert view.sources[0]['preview_frame_age_p95_ms'] == 15.0
    assert view.sources[1]['preview_frame_age_p95_ms'] == 180.0
    assert view.sources[1]['decode_fps'] == 12.0
    # Stop the coordinator without invoking fake reader lifecycle methods.
    for child in session._children.values():
        child._reader = None
        child.status = SearchStatus.STOPPED
        child._finished.set()
    assert session.finished.wait(2)


def test_capture_totals_survive_reader_cleanup_while_live_rates_stop() -> None:
    from types import SimpleNamespace
    manager, target_id = _manager()
    target = manager._targets[target_id]
    session = SearchSession(search_id="diagnostics", target=target,
                            source=SourceConfig(type="camera", device_index=0),
                            settings=manager.settings, face_backend=manager.face_backend,
                            person_detector=manager.person_detector, on_finished=lambda *_: None,
                            timeout_seconds=None, request_id=None)
    reader = SimpleNamespace(stop=lambda: None, metrics_snapshot=lambda: {
        "decode_fps": 20.0, "decoded_frames": 120, "queue_length": 1,
        "queue_capacity": 2, "last_decoded_frame_age_ms": 15.0})
    session._reader = reader
    session._stop_reader_once(reader)
    session._reader = None
    session.status = SearchStatus.STOPPED
    view = session.view()
    assert view.decoded_frames == 120
    assert view.decode_fps == view.source_fps == view.processed_fps == view.preview_fps == 0.0
    assert view.queue_length == 0
    assert view.last_decoded_frame_age_ms is None
