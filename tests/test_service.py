from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import FakeFaceBackend, FakePersonDetector, make_face

from person_search.config import Settings
from person_search.confirmation import MatchDecision
from person_search.domain import (
    Detection,
    MatchState,
    SearchMetrics,
    SearchStatus,
    SourceConfig,
    Target,
    TargetView,
    Track,
)
from person_search.errors import EnrollmentError, ModelUnavailableError, PersonSearchError
from person_search.service import (
    PreviewHub,
    SearchManager,
    SearchSession,
    _clone_target,
    _coerce_face_bbox,
    _merge_faces,
    _normalize_request_id,
    _safe_error,
    _safe_normalize_embedding,
    _sanitize_source,
)


def _preview_stub(track_states: dict, *, settings: Settings | None = None) -> SimpleNamespace:
    """A minimal stand-in for SearchSession with a subscribed preview hub."""
    hub = PreviewHub()
    hub.subscribe()
    return SimpleNamespace(
        _track_states=track_states,
        preview=hub,
        settings=settings or Settings(),
        source=SourceConfig(type="camera", device_index=0),
    )


def _roi_stub(settings: Settings, backend) -> SimpleNamespace:
    """A minimal stand-in for SearchSession's ROI bookkeeping."""
    return SimpleNamespace(
        settings=settings,
        face_backend=backend,
        _roi_misses={},
        _roi_skips={},
        _track_states={},
        _lock=threading.Lock(),
        metrics=SearchMetrics(),
        _note_roi_outcome=lambda track_id, *, hit: None,
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
    assert target_view.embedding_contract_id == FakeFaceBackend.embedding_contract.contract_id
    assert manager.get_target(target_view.target_id).name == "Alice"
    assert np.linalg.norm(manager.get_target(target_view.target_id).embedding) == pytest.approx(1.0)
    assert manager.delete_target(target_view.target_id)
    assert not manager.delete_target(target_view.target_id)
    with pytest.raises(PersonSearchError) as caught:
        manager.get_target(target_view.target_id)
    assert caught.value.code == "target_not_found"


def test_manager_readiness_exposes_complete_provider_and_embedding_contract() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([]), FakePersonDetector())

    readiness = manager.ensure_ready()

    assert readiness["providers"] == {
        "person_detection": "CPUExecutionProvider",
        "face_detection": "CPUExecutionProvider",
        "face_recognition": "CPUExecutionProvider",
    }
    assert readiness["embedding_contract"] == {
        "schema_version": "arcface-v1",
        "model_name": "fake-arcface",
        "model_sha256": "0" * 64,
        "embedding_dimension": 2,
        "input_size": [112, 112],
        "flip_tta": False,
        "contract_id": FakeFaceBackend.embedding_contract.contract_id,
    }


def test_enroll_rejects_blank_target_name() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    with pytest.raises(EnrollmentError, match="target name"):
        manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "   ")


@pytest.mark.parametrize("state", ["candidate", "confirmed"])
def test_preview_only_draws_active_match_tracks(state: str) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    track = Track(7, np.asarray([20, 20, 80, 90], dtype=np.float32), 0.9)

    hidden = _preview_stub({})
    SearchSession._publish_preview(hidden, frame, [track], [])
    _, hidden_jpeg = hidden.preview.after(0, timeout=0)
    assert hidden_jpeg is not None
    hidden_frame = cv2.imdecode(np.frombuffer(hidden_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.max(hidden_frame) == 0

    visible = _preview_stub({7: (state, 0.9)})
    SearchSession._publish_preview(visible, frame, [track], [])
    _, visible_jpeg = visible.preview.after(0, timeout=0)
    assert visible_jpeg is not None
    visible_frame = cv2.imdecode(np.frombuffer(visible_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.max(visible_frame) > 0


def test_preview_is_skipped_when_nobody_is_watching() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    track = Track(7, np.asarray([20, 20, 80, 90], dtype=np.float32), 0.9)
    stub = SimpleNamespace(
        _track_states={7: ("confirmed", 0.9)},
        preview=PreviewHub(),
        settings=Settings(),
        source=SourceConfig(type="camera", device_index=0),
    )

    assert SearchSession._publish_preview(stub, frame, [track], []) is False
    _, jpeg = stub.preview.after(0, timeout=0)
    assert jpeg is None

    stub.preview.subscribe()
    assert SearchSession._publish_preview(stub, frame, [track], []) is True


def test_preview_downscales_to_the_configured_width() -> None:
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    stub = _preview_stub({}, settings=Settings(preview_max_width=960))

    SearchSession._publish_preview(stub, frame, [], [])

    _, jpeg = stub.preview.after(0, timeout=0)
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 960
    assert decoded.shape[0] == 540


def test_person_roi_faces_translate_bbox_and_landmarks_to_frame_coordinates() -> None:
    face = make_face(bbox=(5, 7, 45, 47))
    face.landmarks = np.asarray([[10, 12], [30, 12], [20, 22]], dtype=np.float32)
    backend = FakeFaceBackend([face])
    session = _roi_stub(
        Settings(roi_max_tracks_per_pass=8, roi_min_person_height_px=120), backend
    )
    frame = np.zeros((400, 500, 3), dtype=np.uint8)
    track = Track(7, np.asarray([100, 80, 300, 320], dtype=np.float32), 0.9)

    observations = SearchSession._analyze_person_rois(session, frame, [track])

    # Detection only: the ROI pass must never pay for an embedding.
    assert backend.detect_calls == 1
    assert backend.embed_calls == 0
    np.testing.assert_array_equal(observations[0].bbox, [105, 87, 145, 127])
    np.testing.assert_array_equal(
        observations[0].landmarks,
        [[110, 92], [130, 92], [120, 102]],
    )
    np.testing.assert_array_equal(face.bbox, [5, 7, 45, 47])


def test_person_roi_pass_uses_top_n_valid_tracks_by_score() -> None:
    backend = FakeFaceBackend([])
    session = _roi_stub(
        Settings(roi_max_tracks_per_pass=8, roi_min_person_height_px=120), backend
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

    assert backend.detect_calls == 8


def test_person_roi_pass_batches_crops_by_fixed_scale() -> None:
    class BatchBackend(FakeFaceBackend):
        def __init__(self) -> None:
            super().__init__([])
            self.batch_sizes: list[int] = []
            self.batch_scales: list[int | Sequence[int] | None] = []

        def detect_faces_batch(
            self, frames, *, enrollment=False, detection_size=None
        ):
            self.batch_sizes.append(len(frames))
            self.batch_scales.append(detection_size)
            return [[] for _ in frames]

    backend = BatchBackend()
    settings = Settings(
        roi_max_tracks_per_pass=5,
        roi_min_person_height_px=120,
        roi_batch_enabled=True,
        roi_batch_size=2,
    )
    session = _roi_stub(settings, backend)
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    tracks = [
        Track(
            index,
            np.asarray([index * 20, 0, index * 20 + 100, 240], dtype=np.float32),
            0.9,
        )
        for index in range(5)
    ]

    SearchSession._analyze_person_rois(session, frame, tracks)

    assert backend.batch_sizes == [2, 2, 1]
    assert backend.batch_scales == [320, 320, 320]
    assert session.metrics.roi_batch_count == 3


def test_roi_scheduler_rotates_tracks_after_each_attempt() -> None:
    session = _roi_stub(
        Settings(roi_max_tracks_per_pass=1, roi_min_person_height_px=120),
        FakeFaceBackend([]),
    )
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    tracks = [
        Track(1, np.asarray([0, 0, 120, 240], dtype=np.float32), 0.99),
        Track(2, np.asarray([200, 0, 320, 240], dtype=np.float32), 0.50),
    ]

    first = SearchSession._tracks_needing_roi_face_pass(session, [], tracks)
    SearchSession._analyze_person_rois(session, frame, first)
    second = SearchSession._tracks_needing_roi_face_pass(session, [], tracks)

    assert [track.track_id for track in first] == [1, 2]
    assert [track.track_id for track in second] == [2, 1]


def test_matchable_face_budget_prefers_tracked_faces_and_counts_drops() -> None:
    session = _roi_stub(Settings(max_faces_per_frame=2), FakeFaceBackend([]))
    faces = [
        make_face(bbox=(0, 0, 40, 40)),
        make_face(bbox=(50, 0, 130, 80)),
        make_face(bbox=(150, 0, 220, 70)),
    ]
    tracks = [Track(7, np.asarray([40, 0, 140, 160], dtype=np.float32), 0.8)]

    selected = SearchSession._limit_matchable_faces(session, faces, tracks)

    assert [face.short_side for face in selected] == [80, 70]
    assert session.metrics.faces_dropped_by_budget == 1


@pytest.mark.parametrize(
    "provider_result",
    [
        None,
        0.0,
        [None],
        [SimpleNamespace(embedding=np.asarray([np.nan], dtype=np.float32))],
        ["not-a-face"],
    ],
)
def test_live_embedding_chunk_drops_malformed_provider_results(provider_result) -> None:
    session = _evidence_session("search-malformed-embedding")[0]
    session.face_backend.embed_faces = lambda frame, faces: provider_result  # type: ignore[method-assign]
    face = make_face()

    embedded = session._embed_face_chunk(
        np.zeros((160, 160, 3), dtype=np.uint8), [face]
    )

    assert embedded == []
    assert session.metrics.embedding_failures == 1
    assert session.metrics.embedding_output_failures == 1


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.0,
        [],
        [[1.0, 0.0]],
        [0.0, 0.0],
        [np.nan, 0.0],
        [np.inf, 0.0],
        np.asarray([1.0 + 1.0j, 0.0]),
    ],
)
def test_safe_normalize_embedding_rejects_malformed_values(value) -> None:
    assert _safe_normalize_embedding(value) is None


def test_safe_normalize_embedding_handles_float32_extremes_without_returning_zero() -> None:
    normalized = _safe_normalize_embedding(np.asarray([3e38, 3e38], dtype=np.float32))

    assert normalized is not None
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    assert np.linalg.norm(normalized) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value",
    [None, [0, 0, 1], [0, 0, np.nan, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
)
def test_coerce_face_bbox_rejects_malformed_values(value) -> None:
    assert _coerce_face_bbox(value) is None


def test_live_embedding_results_are_reordered_by_bbox_and_written_to_inputs() -> None:
    session = _evidence_session("search-reordered-embedding")[0]
    first = make_face(bbox=(0, 0, 60, 60))
    second = make_face(bbox=(80, 0, 140, 60))
    session.face_backend.embed_faces = lambda frame, faces: [  # type: ignore[method-assign]
        replace(second, embedding=np.asarray([0.0, 4.0], dtype=np.float32)),
        replace(first, embedding=np.asarray([3.0, 0.0], dtype=np.float32)),
    ]

    embedded = session._embed_face_chunk(
        np.zeros((160, 160, 3), dtype=np.uint8), [first, second]
    )

    assert len(embedded) == 2
    assert embedded[0] is first
    assert embedded[1] is second
    np.testing.assert_array_equal(first.embedding, [1.0, 0.0])
    np.testing.assert_array_equal(second.embedding, [0.0, 1.0])
    assert session.metrics.embedding_failures == 0
    assert session.metrics.embedding_output_failures == 0


def test_partial_duplicate_bbox_embedding_response_fails_closed() -> None:
    session = _evidence_session("search-duplicate-embedding")[0]
    faces = [make_face(), make_face()]
    session.face_backend.embed_faces = lambda frame, inputs: [  # type: ignore[method-assign]
        replace(inputs[0], embedding=np.asarray([1.0, 0.0], dtype=np.float32))
    ]

    embedded = session._embed_face_chunk(
        np.zeros((160, 160, 3), dtype=np.uint8), faces
    )

    assert embedded == []
    assert session.metrics.embedding_failures == 1
    assert session.metrics.embedding_output_failures == 2


def test_oom_split_counts_failed_call_without_double_counting_face_outputs() -> None:
    session = _evidence_session("search-oom-split")[0]
    faces = [
        make_face(bbox=(0, 0, 60, 60)),
        make_face(bbox=(80, 0, 140, 60)),
    ]

    def embed(frame, inputs):
        if len(inputs) > 1:
            raise MemoryError("out of memory")
        return [replace(inputs[0], embedding=np.asarray([1.0, 0.0], dtype=np.float32))]

    session.face_backend.embed_faces = embed  # type: ignore[method-assign]

    embedded = session._embed_face_chunk(
        np.zeros((160, 160, 3), dtype=np.uint8), faces
    )

    assert len(embedded) == 2
    assert session.metrics.embedding_batch_count == 3
    assert session.metrics.embedding_failures == 1
    assert session.metrics.embedding_output_failures == 0


def test_roi_selection_is_per_track_and_not_suppressed_by_an_unrelated_near_face() -> None:
    session = _roi_stub(Settings(preferred_search_face_px=80), FakeFaceBackend([]))
    tracks = [
        Track(1, np.asarray([0, 0, 120, 240], dtype=np.float32), 0.9),
        Track(2, np.asarray([200, 0, 320, 240], dtype=np.float32), 0.8),
    ]
    near_face = make_face(bbox=(10, 10, 100, 100))

    selected = SearchSession._tracks_needing_roi_face_pass(session, [near_face], tracks)

    assert [track.track_id for track in selected] == [2]


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
        # A real SearchMetrics rather than a hand-rolled namespace: the stub version
        # broke every time the pipeline started recording one more field, which says
        # nothing about the behaviour under test.
        metrics=SearchMetrics(),
    )
    small = make_face(bbox=(0, 0, 64, 70), blur_variance=120.0)
    rejected = make_face(accepted=False, blur_variance=20.0)

    SearchSession._record_face_metrics(session, [small, rejected])

    assert session.metrics.face_observations == 2
    assert session.metrics.accepted_faces == 1
    assert session.metrics.small_faces == 1
    assert session.metrics.rejection_counts == {"face_blurry": 1}
    assert session.metrics.face_size_counts == {"48_63": 1, "64_79": 1}
    assert session.metrics.match_stage_counts == {
        "detected": 2,
        "quality_accepted": 1,
    }
    # The sharpness gate's own numbers reach the panel, rejected faces included:
    # a gate nobody can read cannot be calibrated against real footage.
    assert session.metrics.blur_variances == [120.0, 20.0]
    assert session.metrics.snapshot()["blur_variance_p50"] == pytest.approx(70.0)

    sanitized = _sanitize_source(
        SourceConfig(
            type="rtsp",
            uri="rtsp://user:secret@camera.test:8554/live/path?token=hidden",
            debug_preview=True,
        )
    )
    assert sanitized.uri == "rtsp://camera.test:8554/***"
    assert sanitized.debug_preview is True


def test_matchable_face_guard_keeps_48px_hard_floor_after_unvalidated_settings() -> None:
    settings = Settings.model_construct(tiny_face_enabled=True, tiny_face_min_px=1)
    session = SimpleNamespace(settings=settings)

    assert not SearchSession._is_face_matchable(session, make_face(bbox=(20, 20, 67, 67)))
    assert SearchSession._is_face_matchable(session, make_face(bbox=(20, 20, 68, 68)))


def test_associated_low_similarity_normal_face_is_not_evidence_eligible(
    monkeypatch,
) -> None:
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    packets = [SimpleNamespace(frame_id=0, captured_at=0.0, frame=frame)]

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
        target_id="target-low",
        name="低相似度目标",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        "target-low",
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        "低相似度目标",
    )
    similarity = 0.3
    face = make_face(
        embedding=(similarity, float(np.sqrt(1.0 - similarity**2))),
        bbox=(20, 20, 100, 100),
    )
    session = SearchSession(
        search_id="search-low-similarity",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(),
        face_backend=FakeFaceBackend([face]),
        person_detector=FakePersonDetector(
            [Detection(np.asarray([0, 0, 140, 159], dtype=np.float32), 0.99)]
        ),
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    counts = session.metrics.match_stage_counts
    assert counts["associated"] == 1
    assert counts["evidence_policy_rejected"] == 1
    assert counts.get("evidence_eligible", 0) == 0
    assert counts["evidence_collected"] == 0
    assert session.view().targets[0].last_rejection_reason == "similarity_low"


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
        face_backend=FakeFaceBackend([make_face(bbox=(20, 20, 100, 100))]),
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


def test_found_target_remains_an_identity_competitor_until_real_runner_up_appears(
    monkeypatch,
) -> None:
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    packets = [
        SimpleNamespace(frame_id=index, captured_at=index * 0.25, frame=frame)
        for index in range(13)
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

    class SequenceFaceBackend:
        model_name = "fake-arcface"
        provider_name = "CPUExecutionProvider"
        detection_provider_name = "CPUExecutionProvider"
        recognition_provider_name = "CPUExecutionProvider"
        embedding_contract = FakeFaceBackend.embedding_contract

        def __init__(self, observations):
            self.observations = list(observations)
            self.calls = 0

        def detect_faces(self, frame, *, enrollment=False, detection_size=None):
            observation = self.observations[self.calls]
            self.calls += 1
            return [replace(observation, embedding=None)]

        def embed_faces(self, frame, faces):
            # This backend hands out one scripted observation per call, so the
            # embedding is recovered from the same script position.
            index = max(0, self.calls - 1)
            source = self.observations[index]
            return [replace(face, embedding=source.embedding) for face in faces]

        def analyze(self, frame, *, enrollment=False):
            return self.embed_faces(frame, self.detect_faces(frame, enrollment=enrollment))

    monkeypatch.setattr("person_search.service.LatestFrameReader", FakeReader)
    target_view_a = TargetView(
        target_id="target-a",
        name="A",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target_view_b = target_view_a.model_copy(update={"target_id": "target-b", "name": "B"})
    b_embedding = np.asarray([0.75, np.sqrt(1.0 - 0.75**2)], dtype=np.float32)
    targets = [
        Target("target-a", np.asarray([1.0, 0.0], dtype=np.float32), target_view_a, "A"),
        Target("target-b", b_embedding, target_view_b, "B"),
    ]
    near_a = make_face(bbox=(20, 20, 100, 100))
    tiny_a = make_face(bbox=(20, 20, 68, 68))
    tiny_b = make_face(embedding=tuple(b_embedding), bbox=(20, 20, 68, 68))
    backend = SequenceFaceBackend([near_a, *([tiny_a] * 6), *([tiny_b] * 6)])
    session = SearchSession(
        search_id="search-identity-gallery",
        target=targets[0],
        targets=targets,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(
            evidence_required=1,
            tiny_face_enabled=True,
            tiny_face_shadow_mode=False,
        ),
        face_backend=backend,
        person_detector=FakePersonDetector(
            [Detection(np.asarray([0, 0, 140, 159], dtype=np.float32), 0.99)]
        ),
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    found = [
        event["data"]["target_id"]
        for event in session.events.after(0, timeout=0)
        if event["type"] == "target_found"
    ]
    assert found == ["target-a", "target-b"]
    assert backend.calls == 13
    assert set(session._identity_targets) == {"target-a", "target-b"}
    assert session.metrics.match_stage_counts["inactive_identity_top1"] == 6
    assert session.status == SearchStatus.COMPLETED


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
            evidence_api_key="test-evidence-key",
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
    assert confirmed[0]["data"]["face_bbox"] == [0.125, 0.125, 0.525, 0.525]
    # The worker has already auto-transitioned to completed here; evidence must
    # remain fetchable for HTTP reconciliation instead of being cleared in the
    # terminal transition.
    completed_crop, _ = session.get_evidence(
        confirmed[0]["data"]["evidence_id"], "face_crop"
    )
    decoded_crop = cv2.imdecode(
        np.frombuffer(completed_crop, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded_crop.shape[:2] == (64, 64)
    assert session._track_states[confirmed[0]["data"]["track_id"]][0] == "confirmed"
    assert session.status == SearchStatus.COMPLETED
    view = session.view()
    assert view.face_observations == 4
    assert view.accepted_faces == 4
    assert view.small_faces == 4
    assert view.unassociated_faces == 0
    assert view.association_counts == {"face_fallback": 4}


@pytest.mark.parametrize(
    ("shadow_mode", "event_type"),
    [(True, "tiny_shadow_confirmed"), (False, "confirmed")],
)
def test_tiny_face_confirms_only_on_a_strict_person_track(
    monkeypatch, shadow_mode: bool, event_type: str
) -> None:
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    packets = [
        SimpleNamespace(frame_id=index, captured_at=index * 0.25, frame=frame) for index in range(6)
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
        target_id="target-tiny",
        name="张三",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        "target-tiny",
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        "张三",
    )
    similarity = 0.75
    tiny_face = make_face(
        embedding=(similarity, float(np.sqrt(1.0 - similarity**2))),
        bbox=(20, 20, 68, 68),
    )
    session = SearchSession(
        search_id="search-tiny",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(
            tiny_face_enabled=True,
            tiny_face_shadow_mode=shadow_mode,
        ),
        face_backend=FakeFaceBackend([tiny_face]),
        person_detector=FakePersonDetector(
            [Detection(np.asarray([0, 0, 140, 150], dtype=np.float32), 0.99)]
        ),
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    events = session.events.after(0, timeout=0)
    confirmed = [event for event in events if event["type"] == event_type]
    assert len(confirmed) == 1
    assert confirmed[0]["data"]["association"] == "person_strict"
    assert confirmed[0]["data"]["evidence_count"] == 6
    assert not any(event["type"] == "candidate" for event in events)
    if shadow_mode:
        assert confirmed[0]["data"]["state"] == "shadow_confirmed"
        assert not any(event["type"] == "confirmed" for event in events)
        assert session.status == SearchStatus.STOPPED
    else:
        assert confirmed[0]["data"]["state"] == "confirmed"
        assert session.status == SearchStatus.COMPLETED
    view = session.view()
    assert view.targets[0].last_face_px == 48
    assert view.targets[0].best_observed_similarity == pytest.approx(0.75)
    assert view.targets[0].evidence_count == 6
    assert view.targets[0].status.value == ("searching" if shadow_mode else "found")
    assert session.metrics.match_stage_counts["evidence_eligible"] == 6
    assert session.metrics.match_stage_counts["evidence_collected"] == 6


def test_ambiguous_identity_margin_records_rejection_on_both_targets(monkeypatch) -> None:
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    packets = [SimpleNamespace(frame_id=0, captured_at=0.0, frame=frame)]

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
    # Orthogonal targets with a face exactly between them: both score 0.707, so the
    # top1/top2 margin is 0.0 and the 48px face is too ambiguous to become evidence.
    targets = [
        Target("target-1", np.asarray([1.0, 0.0], dtype=np.float32), target_view_1, "张三"),
        Target("target-2", np.asarray([0.0, 1.0], dtype=np.float32), target_view_2, "李四"),
    ]
    half = float(np.sqrt(0.5))
    session = SearchSession(
        search_id="search-ambiguous",
        target=targets[0],
        targets=targets,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(tiny_face_enabled=True),
        face_backend=FakeFaceBackend([make_face(embedding=(half, half), bbox=(20, 20, 68, 68))]),
        person_detector=FakePersonDetector(
            [Detection(np.asarray([0, 0, 140, 150], dtype=np.float32), 0.99)]
        ),
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    assert session.metrics.match_stage_counts["ambiguous_identity"] == 1
    assert "evidence_eligible" not in session.metrics.match_stage_counts
    view = session.view()
    reasons = {item.target_id: item.last_rejection_reason for item in view.targets}
    assert reasons == {
        "target-1": "identity_margin_low",
        "target-2": "identity_margin_low",
    }
    assert all(item.best_observed_similarity == pytest.approx(half) for item in view.targets)
    assert all(item.last_face_px == 48 for item in view.targets)
    assert view.found_count == 0


def test_shadow_lost_event_does_not_mark_target_found() -> None:
    target_view = TargetView(
        target_id="target-shadow",
        name="Shadow 目标",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        "target-shadow",
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        "Shadow 目标",
    )
    session = SearchSession(
        search_id="search-shadow-lost",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(tiny_face_enabled=True),
        face_backend=FakeFaceBackend([]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )
    base = {
        "track_id": 7,
        "bbox": np.asarray([0, 0, 100, 150], dtype=np.float32),
        "similarity": 0.75,
        "quality": 0.8,
        "evidence_count": 6,
        "association": "person_strict",
        "shadow": True,
    }

    session._handle_decisions(
        target.target_id,
        target,
        [MatchDecision(state=MatchState.CONFIRMED, **base)],
        (160, 160, 3),
    )
    session._handle_decisions(
        target.target_id,
        target,
        [MatchDecision(state=MatchState.LOST, **base)],
        (160, 160, 3),
    )

    events = session.events.after(0, timeout=0)
    assert [event["type"] for event in events] == [
        "tiny_shadow_confirmed",
        "tiny_shadow_lost",
    ]
    assert events[-1]["data"]["state"] == "shadow_lost"
    assert session.view().targets[0].status.value == "searching"
    assert session.view().targets[0].last_rejection_reason == "shadow_lost"
    assert session.view().found_count == 0
    assert session._shadow_tracks == set()


def test_tiny_face_does_not_use_face_only_fallback(monkeypatch) -> None:
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    packets = [
        SimpleNamespace(frame_id=index, captured_at=index * 0.25, frame=frame) for index in range(6)
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
        target_id="target-tiny",
        name="张三",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        "target-tiny",
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        "张三",
    )
    session = SearchSession(
        search_id="search-tiny-unassociated",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(tiny_face_enabled=True),
        face_backend=FakeFaceBackend([make_face(embedding=(1.0, 0.0), bbox=(20, 20, 68, 68))]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    events = session.events.after(0, timeout=0)
    assert not any(event["type"] in {"candidate", "confirmed"} for event in events)
    assert session.view().unassociated_faces == 6
    assert session.view().association_counts == {}


def test_completed_search_keeps_real_face_crop_until_release_or_stop() -> None:
    target_view = TargetView(
        target_id="target-evidence",
        name="Evidence 目标",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        "target-evidence", np.asarray([1.0, 0.0], dtype=np.float32), target_view, "Evidence 目标"
    )
    session = SearchSession(
        search_id="search-evidence",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(evidence_api_key="test-evidence-key", evidence_ttl_seconds=60),
        face_backend=FakeFaceBackend([]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )
    frame = np.full((160, 160, 3), 127, dtype=np.uint8)
    session._handle_decisions(
        target.target_id,
        target,
        [
            MatchDecision(
                state=MatchState.CONFIRMED,
                track_id=7,
                bbox=np.asarray([20, 30, 120, 140], dtype=np.float32),
                face_bbox=np.asarray([40, 50, 80, 90], dtype=np.float32),
                similarity=0.75,
                quality=0.8,
                evidence_count=3,
                association="person_strict",
                shadow=False,
            )
        ],
        frame,
    )

    confirmed = session.events.after(0, timeout=0)[0]
    evidence_id = confirmed["data"]["evidence_id"]
    assert confirmed["data"]["bbox"] == [0.125, 0.1875, 0.75, 0.875]
    assert confirmed["data"]["face_bbox"] == [0.25, 0.3125, 0.5, 0.5625]
    assert confirmed["data"]["evidence_available"] is True
    assert confirmed["data"]["evidence_expires_at_ms"] > int(time.time() * 1000)

    session._transition(SearchStatus.COMPLETED, None, publish=False)
    crop, crop_type = session.get_evidence(evidence_id, "face_crop")
    full_frame, frame_type = session.get_evidence(evidence_id, "frame")
    assert crop_type == frame_type == "image/jpeg"
    decoded_crop = cv2.imdecode(np.frombuffer(crop, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded_crop.shape[:2] == (40, 40)
    assert len(full_frame) > len(crop)
    result = session.view().confirmed_results[0]
    assert result.evidence_id == evidence_id
    assert result.evidence_available is True
    assert result.face_bbox == (0.25, 0.3125, 0.5, 0.5625)

    session.stop(timeout=0)
    assert session.view().confirmed_results[0].evidence_available is False
    # A stop destroyed bytes nobody claimed. That is a different outcome from a
    # consumer acknowledging delivery, and the code has to say so: the caller
    # must stop retrying, and its operator must not read "released" and believe
    # the crop was stored somewhere downstream.
    with pytest.raises(PersonSearchError) as exc_info:
        session.get_evidence(evidence_id, "face_crop")
    assert exc_info.value.code == "evidence_discarded"
    assert exc_info.value.status_code == 410


def test_confirmed_decision_without_face_box_never_falls_back_to_body_crop() -> None:
    target_view = TargetView(
        target_id="target-no-face-crop",
        name="No Crop",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        target_view.target_id,
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        target_view.name,
    )
    session = SearchSession(
        search_id="search-no-face-crop",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(),
        face_backend=FakeFaceBackend([]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )
    session._handle_decisions(
        target.target_id,
        target,
        [
            MatchDecision(
                state=MatchState.CONFIRMED,
                track_id=10,
                bbox=np.asarray([0, 0, 120, 150], dtype=np.float32),
                similarity=0.8,
                quality=0.9,
                evidence_count=3,
            )
        ],
        np.full((160, 160, 3), 100, dtype=np.uint8),
    )

    confirmed = session.events.after(0, timeout=0)[0]["data"]
    assert confirmed["evidence_available"] is False
    assert "evidence_id" not in confirmed
    assert session._evidence == {}


def test_unconfigured_evidence_endpoint_does_not_encode_or_advertise_a_crop() -> None:
    target_view = TargetView(
        target_id="target-no-evidence-key",
        name="No Evidence Key",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        target_view.target_id,
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        target_view.name,
    )
    session = SearchSession(
        search_id="search-no-evidence-key",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(),
        face_backend=FakeFaceBackend([]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )
    session._handle_decisions(
        target.target_id,
        target,
        [
            MatchDecision(
                state=MatchState.CONFIRMED,
                track_id=12,
                bbox=np.asarray([0, 0, 120, 150], dtype=np.float32),
                face_bbox=np.asarray([20, 20, 60, 60], dtype=np.float32),
                similarity=0.8,
                quality=0.9,
                evidence_count=3,
            )
        ],
        np.full((160, 160, 3), 100, dtype=np.uint8),
    )

    confirmed = session.events.after(0, timeout=0)[0]["data"]
    assert confirmed["evidence_available"] is False
    assert "evidence_id" not in confirmed
    assert session._evidence == {}


def test_evidence_encoding_failure_does_not_undo_completed_match(monkeypatch) -> None:
    target_view = TargetView(
        target_id="target-evidence-failure",
        name="Evidence Failure",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        target_view.target_id,
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        target_view.name,
    )
    session = SearchSession(
        search_id="search-evidence-failure",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(),
        face_backend=FakeFaceBackend([]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )
    monkeypatch.setattr(
        session,
        "_store_evidence",
        lambda frame, bbox: (_ for _ in ()).throw(RuntimeError("JPEG encoding failed")),
    )

    original_handle = session._handle_decisions

    def confirm_on_first_frame(target_id, current_target, decisions, frame) -> None:
        original_handle(
            target_id,
            current_target,
            [
                MatchDecision(
                    state=MatchState.CONFIRMED,
                    track_id=11,
                    bbox=np.asarray([0, 0, 120, 150], dtype=np.float32),
                    face_bbox=np.asarray([20, 20, 60, 60], dtype=np.float32),
                    similarity=0.8,
                    quality=0.9,
                    evidence_count=3,
                )
            ],
            frame,
        )

    monkeypatch.setattr(session, "_handle_decisions", confirm_on_first_frame)
    packets = [
        SimpleNamespace(
            frame_id=1,
            captured_at=1.0,
            frame=np.full((160, 160, 3), 100, dtype=np.uint8),
        )
    ]
    monkeypatch.setattr("person_search.service.LatestFrameReader", _fake_reader(packets))

    session._run()

    events = session.events.after(0, timeout=0)
    confirmed = next(event for event in events if event["type"] == "confirmed")
    found = next(event for event in events if event["type"] == "target_found")
    assert confirmed["data"]["evidence_available"] is False
    assert found["data"]["evidence_available"] is False
    assert "evidence_id" not in confirmed["data"]
    assert [event["type"] for event in events][-3:] == [
        "search_status",
        "target_found",
        "all_found",
    ]
    assert session.status == SearchStatus.COMPLETED
    assert session.view().targets[0].status.value == "found"
    assert session.view().confirmed_results[0].evidence_available is False
    assert target.target_id not in session._active_targets


def test_evidence_ttl_actively_reclaims_completed_session_bytes() -> None:
    target_view = TargetView(
        target_id="target-expiring-evidence",
        name="TTL 目标",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        target_view.target_id,
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        target_view.name,
    )
    session = SearchSession(
        search_id="search-expiring-evidence",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(evidence_api_key="test-evidence-key", evidence_ttl_seconds=0.05),
        face_backend=FakeFaceBackend([]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )
    session._handle_decisions(
        target.target_id,
        target,
        [
            MatchDecision(
                state=MatchState.CONFIRMED,
                track_id=9,
                bbox=np.asarray([0, 0, 100, 150], dtype=np.float32),
                face_bbox=np.asarray([20, 20, 60, 60], dtype=np.float32),
                similarity=0.8,
                quality=0.9,
                evidence_count=3,
            )
        ],
        np.full((160, 160, 3), 100, dtype=np.uint8),
    )
    session._transition(SearchStatus.COMPLETED, None, publish=False)
    evidence_id = session.events.after(0, timeout=0)[0]["data"]["evidence_id"]

    deadline = time.monotonic() + 1.0
    while evidence_id in session._evidence and time.monotonic() < deadline:
        time.sleep(0.01)

    assert evidence_id not in session._evidence
    assert session.view().confirmed_results[0].evidence_available is False
    with pytest.raises(PersonSearchError) as exc_info:
        session.get_evidence(evidence_id, "face_crop")
    assert exc_info.value.code == "evidence_expired"
    assert exc_info.value.status_code == 410


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


def test_replace_active_with_same_target_stages_embedding_before_old_callback(monkeypatch) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    monkeypatch.setattr(SearchSession, "start", lambda session: None)
    first = manager.start_batch_search(
        [target.target_id], SourceConfig(type="camera", device_index=0)
    )
    first_session = manager.get_session(first.search_id)

    def finish_old_session(timeout=15.0) -> None:
        first_session._finished.set()
        manager._on_finished(first.search_id, [target.target_id])

    first_session.stop = finish_old_session  # type: ignore[method-assign]
    replacement = manager.start_batch_search(
        [target.target_id],
        SourceConfig(type="camera", device_index=1),
        replace_active=True,
    )
    replacement_session = manager.get_session(replacement.search_id)

    embedding = replacement_session.targets[0].embedding
    assert embedding is not None
    np.testing.assert_array_equal(embedding, [1.0, 0.0])
    assert target.target_id not in manager._targets


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


@pytest.mark.parametrize(
    "request_id",
    [
        "\nrequest-id",
        "request-id\r",
        "\u0085request-id",
        "request\u200b-id",
        "request\u2028id",
    ],
)
def test_request_id_rejects_control_and_format_characters_before_trimming(
    request_id: str,
) -> None:
    with pytest.raises(PersonSearchError) as exc_info:
        _normalize_request_id(request_id)

    assert exc_info.value.code == "invalid_request_id"
    assert exc_info.value.status_code == 422


def test_request_id_still_trims_printable_outer_spaces() -> None:
    assert _normalize_request_id("  request-id  ") == "request-id"


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
    detector = FakePersonDetector([Detection(np.asarray([0, 0, 110, 119], dtype=np.float32), 0.99)])
    manager = SearchManager(
        settings,
        FakeFaceBackend([make_face(bbox=(20, 20, 100, 100))]),
        detector,
    )
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
    assert [event["type"] for event in first_session.events.after(0, timeout=0)][-3:] == [
        "search_status",
        "target_found",
        "all_found",
    ]

    manager.stop_search(replacement.search_id)
    assert manager.active_search() is None


def test_roi_pass_uses_a_single_fixed_detection_size() -> None:
    """A tight crop must not pay for the full-frame Auto dual-scale pass."""
    backend = FakeFaceBackend([])
    settings = Settings(roi_face_detection_size=320, roi_min_person_height_px=120)
    session = _roi_stub(settings, backend)
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    track = Track(1, np.asarray([0, 0, 120, 240], dtype=np.float32), 0.9)

    SearchSession._analyze_person_rois(session, frame, [track])

    assert backend.detection_sizes == [320]


def test_roi_backoff_skips_a_track_that_keeps_yielding_nothing() -> None:
    settings = Settings(roi_backoff_max_skips=16)
    session = _roi_stub(settings, FakeFaceBackend([]))
    tracks = [Track(1, np.asarray([0, 0, 120, 240], dtype=np.float32), 0.9)]

    # First pass: no cooldown yet, the track is a candidate.
    assert SearchSession._tracks_needing_roi_face_pass(session, [], tracks) == tracks

    SearchSession._note_roi_outcome(session, 1, hit=False)
    assert session._roi_skips[1] == 2

    # The next two passes are skipped, then the track is eligible again.
    assert SearchSession._tracks_needing_roi_face_pass(session, [], tracks) == []
    assert SearchSession._tracks_needing_roi_face_pass(session, [], tracks) == []
    assert SearchSession._tracks_needing_roi_face_pass(session, [], tracks) == tracks

    # Backoff grows with each consecutive miss.
    SearchSession._note_roi_outcome(session, 1, hit=False)
    assert session._roi_skips[1] == 4

    # A hit clears it immediately.
    SearchSession._note_roi_outcome(session, 1, hit=True)
    assert 1 not in session._roi_skips
    assert 1 not in session._roi_misses


def test_roi_backoff_is_capped_and_forgets_dead_tracks() -> None:
    session = _roi_stub(Settings(roi_backoff_max_skips=4), FakeFaceBackend([]))
    for _ in range(6):
        SearchSession._note_roi_outcome(session, 7, hit=False)
    assert session._roi_skips[7] == 4

    # A track that no longer exists must not leak its bookkeeping forever.
    SearchSession._tracks_needing_roi_face_pass(session, [], [])
    SearchSession._tracks_needing_roi_face_pass(
        session, [], [Track(9, np.asarray([0, 0, 10, 10], dtype=np.float32), 0.5)]
    )
    assert 7 not in session._roi_skips
    assert 7 not in session._roi_misses


def test_confirmed_tracks_are_excluded_from_the_roi_pass() -> None:
    session = _roi_stub(Settings(), FakeFaceBackend([]))
    tracks = [
        Track(1, np.asarray([0, 0, 120, 240], dtype=np.float32), 0.9),
        Track(2, np.asarray([200, 0, 320, 240], dtype=np.float32), 0.8),
    ]
    session._track_states = {1: ("confirmed", 0.9)}

    selected = SearchSession._tracks_needing_roi_face_pass(session, [], tracks)

    assert [track.track_id for track in selected] == [2]


def _budget_stub(
    settings: Settings, roi_p95_ms: float, credit_seconds: float = 0.0
) -> SimpleNamespace:
    metrics = SearchMetrics()
    if roi_p95_ms:
        metrics.stage_latencies_ms["face_roi"] = [roi_p95_ms]
    stub = SimpleNamespace(
        settings=settings,
        metrics=metrics,
        _lock=threading.Lock(),
        _budget_credit=credit_seconds,
    )
    stub._stage_p95_ms = lambda stage: SearchSession._stage_p95_ms(stub, stage)
    stub._record_budget_skip = lambda stage: SearchSession._record_budget_skip(stub, stage)
    return stub


def test_roi_runs_on_a_frame_that_already_exceeded_the_target_period() -> None:
    """The regression: ROI is only ever considered on an over-budget frame.

    Full-frame face detection alone costs more than one target period, so the
    old single-frame remainder was structurally negative and the stage was
    skipped forever. Banked credit from cheap frames must still admit it.
    """
    session = _budget_stub(Settings(target_loop_hz=10.0), roi_p95_ms=82.0, credit_seconds=0.1)
    started = time.monotonic() - 0.125  # person 13ms + face_full 112ms, over the 100ms period

    assert SearchSession._roi_fits_budget(session, started) is True
    assert session.metrics.budget_skips == {}


def test_roi_backs_off_once_its_cost_has_drained_the_credit() -> None:
    """A stage slower than its refill rate throttles itself down, as before."""
    session = _budget_stub(Settings(target_loop_hz=10.0), roi_p95_ms=90.0, credit_seconds=0.02)
    started = time.monotonic()

    assert SearchSession._roi_fits_budget(session, started) is False
    assert session.metrics.budget_skips["face_roi_credit"] == 1


def test_roi_is_skipped_once_the_processed_fps_floor_is_already_breached() -> None:
    """The floor outranks any amount of banked credit."""
    session = _budget_stub(
        Settings(target_loop_hz=10.0, min_processed_fps=2.0), roi_p95_ms=0.0, credit_seconds=10.0
    )
    started = time.monotonic() - 0.6  # already past the 500ms floor

    assert SearchSession._roi_fits_budget(session, started) is False
    assert session.metrics.budget_skips["face_roi_floor"] == 1


def test_budget_credit_is_capped_in_both_directions() -> None:
    """Idle time cannot bank an unbounded burst, and one slow pass cannot starve."""
    settings = Settings(target_loop_hz=10.0, budget_credit_max_frames=2.0)
    cap = 2.0 / 10.0
    session = _budget_stub(settings, roi_p95_ms=0.0)
    session._settle_budget_credit = lambda cost: SearchSession._settle_budget_credit(session, cost)

    for _ in range(20):
        session._settle_budget_credit(0.001)  # a long stretch of near-free frames
    assert session._budget_credit == pytest.approx(cap)

    session._settle_budget_credit(5.0)  # one catastrophic pass
    assert session._budget_credit == pytest.approx(-cap)


def test_quality_rejected_faces_never_reach_arcface() -> None:
    """Detection is cheap; ArcFace is not. Rejected faces must cost zero embeddings."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    blurry = make_face(bbox=(20, 20, 120, 120), accepted=False)
    backend = FakeFaceBackend([blurry])

    detected = backend.detect_faces(frame)
    settings = Settings()
    session = SimpleNamespace(settings=settings)
    matchable = [face for face in detected if SearchSession._is_face_matchable(session, face)]

    assert matchable == []
    assert backend.embed_calls == 0


def test_detect_then_embed_matches_analyze() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    face = make_face(bbox=(20, 20, 120, 120))
    backend = FakeFaceBackend([face])

    combined = backend.embed_faces(frame, backend.detect_faces(frame))
    direct = backend.analyze(frame)

    assert len(combined) == len(direct) == 1
    np.testing.assert_array_equal(combined[0].bbox, direct[0].bbox)
    np.testing.assert_array_equal(combined[0].embedding, direct[0].embedding)


def test_effective_config_reports_the_resolved_rates_and_gates() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "Alice")
    search = manager.start_search(
        target.target_id, SourceConfig(type="camera", device_index=0)
    )
    session = manager.get_session(search.search_id)

    config = session.view().effective_config

    # The fakes report CPU providers, so the CPU branch must be the one reported.
    assert config["face_detection_hz"] == Settings().face_detection_hz_cpu
    assert config["person_detection_hz"] == Settings().person_detection_hz_cpu
    assert config["face_detection_size"] == "auto(128+640)"
    assert config["required_sampling_hz"] == pytest.approx(3 / 1.5)
    assert config["small_face_required_sampling_hz"] == pytest.approx(4 / 2.0)
    assert config["effective_search_min_face_px"] == Settings().effective_search_min_face_px

    manager.stop_search(search.search_id)


def test_saturated_tiny_evidence_surfaces_similarity_shortfall(monkeypatch) -> None:
    """Reproduce the field failure and assert the view explains it.

    A 50px face banks 6/6 samples yet never confirms because similarity sits
    far below the tiny threshold. The view must report the qualifying count and
    the required similarity, not just the saturated evidence counter.
    """
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    packets = [
        SimpleNamespace(frame_id=index, captured_at=index * 0.25, frame=frame) for index in range(8)
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
        target_id="target-far",
        name="张三",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        "target-far",
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        "张三",
    )
    similarity = 0.41
    tiny_face = make_face(
        embedding=(similarity, float(np.sqrt(1.0 - similarity**2))),
        bbox=(20, 20, 70, 70),
    )
    settings = Settings(tiny_face_enabled=True)
    session = SearchSession(
        search_id="search-far",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=settings,
        face_backend=FakeFaceBackend([tiny_face]),
        person_detector=FakePersonDetector(
            [Detection(np.asarray([0, 0, 140, 150], dtype=np.float32), 0.99)]
        ),
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    events = session.events.after(0, timeout=0)
    assert not any(event["type"] in {"confirmed", "tiny_shadow_confirmed"} for event in events)
    view_target = session.view().targets[0]
    assert view_target.status.value == "searching"
    assert view_target.evidence_count == view_target.required_evidence == 6
    assert view_target.qualifying_evidence == 0
    assert view_target.window_similarity == pytest.approx(similarity)
    assert view_target.required_similarity == pytest.approx(
        settings.tiny_face_similarity_threshold
    )
    assert view_target.last_rejection_reason == "similarity_low"


def _fake_reader(packets: list) -> type:
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

    return FakeReader


def _single_target() -> Target:
    view = TargetView(
        target_id="target-scale",
        name="目标",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    return Target("target-scale", np.asarray([1.0, 0.0], dtype=np.float32), view, "目标")


def test_full_frame_pass_uses_the_resolved_scales_on_the_deep_scan_cadence(monkeypatch) -> None:
    """On 1080p the 640 pass leaves a 49px face at ~16px, below what SCRFD can score."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    packets = [
        SimpleNamespace(frame_id=index, captured_at=float(index), frame=frame)
        for index in range(4)
    ]
    monkeypatch.setattr("person_search.service.LatestFrameReader", _fake_reader(packets))
    backend = FakeFaceBackend([])
    session = SearchSession(
        search_id="search-scales",
        target=_single_target(),
        source=SourceConfig(type="camera", device_index=0),
        settings=Settings(face_detection_extra_scale_cpu=1280, face_deep_scan_every_n=2),
        face_backend=backend,
        # No person tracks, so nothing can trigger an ROI pass and every recorded
        # detection size belongs to the full-frame pass.
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )

    session._run()

    assert backend.detection_sizes == [(128, 640, 1280), (128, 640), (128, 640, 1280), (128, 640)]


def test_roi_pass_never_downsamples_a_crop_it_exists_to_preserve() -> None:
    backend = FakeFaceBackend([])
    settings = Settings(roi_face_detection_size=320, roi_face_detection_max_size=640)
    session = _roi_stub(settings, backend)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # A close person: the upper half of the body box is far larger than 320, and
    # shrinking it back to 320 would throw away the pixels the crop was taken for.
    near = Track(1, np.asarray([0, 0, 500, 1000], dtype=np.float32), 0.9)
    far = Track(2, np.asarray([600, 0, 690, 260], dtype=np.float32), 0.9)

    SearchSession._analyze_person_rois(session, frame, [near, far])

    # 500x500 goes to the upper rung instead of being squeezed back to 320; the far
    # crop still gets upsampled to the floor. Two rungs keep the number of distinct
    # ONNX input shapes -- the real cost on CUDA -- down to two.
    assert backend.detection_sizes == [640, 320]


def test_rejection_reason_is_reported_with_the_size_of_the_face_it_belongs_to() -> None:
    """The reason and the pixel count must describe one observation, not two."""
    session = SimpleNamespace(
        _lock=threading.RLock(),
        settings=Settings(),
        _active_targets={"target-1": None},
        _target_status={
            "target-1": {"last_face_px": 49, "last_rejection_reason": None,
                         "last_rejection_face_px": None}
        },
    )
    rejected = make_face(bbox=(0, 0, 40, 40), accepted=False)

    SearchSession._record_rejected_observations(session, [rejected])

    status = session._target_status["target-1"]
    # The largest face seen stays at 49px, but the reason now carries its own 40px
    # rather than borrowing the other face's size.
    assert status["last_face_px"] == 49
    assert status["last_rejection_reason"] == "face_blurry"
    assert status["last_rejection_face_px"] == 40


def test_camera_motion_estimate_recovers_a_known_shift() -> None:
    session = SimpleNamespace(
        settings=Settings(),
        _lock=threading.RLock(),
        metrics=SearchMetrics(),
        _motion_hanning=None,
    )
    session._record_stage = lambda stage, started: SearchSession._record_stage(
        session, stage, started
    )
    session._motion_window = lambda shape: SearchSession._motion_window(session, shape)
    rng = np.random.default_rng(7)
    first = rng.integers(0, 255, size=(540, 960, 3), dtype=np.uint8)
    shift = 60
    second = np.roll(first, shift, axis=1)

    motion, state = SearchSession._estimate_camera_motion(session, first, None)
    assert motion is None and state is not None

    motion, _ = SearchSession._estimate_camera_motion(session, second, state)

    assert motion is not None
    # Reported in full-frame pixels even though the estimate runs downscaled.
    assert motion[0] == pytest.approx(shift, abs=2.0)
    assert motion[1] == pytest.approx(0.0, abs=2.0)
    assert session.metrics.snapshot()["camera_motion_px_p95"] == pytest.approx(shift, abs=2.0)


def test_camera_motion_estimate_is_skipped_when_disabled() -> None:
    session = SimpleNamespace(settings=Settings(camera_motion_compensation=False))
    frame = np.zeros((540, 960, 3), dtype=np.uint8)

    assert SearchSession._estimate_camera_motion(session, frame, None) == (None, None)


def _evidence_session(search_id: str, *, settings: Settings | None = None):
    """A session wired for evidence tests, with its confirmed decision helper."""
    target_view = TargetView(
        target_id=f"{search_id}-target",
        name="Evidence 目标",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    target = Target(
        target_view.target_id,
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
        target_view.name,
    )
    session = SearchSession(
        search_id=search_id,
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=settings
        or Settings(evidence_api_key="test-evidence-key", evidence_ttl_seconds=600),
        face_backend=FakeFaceBackend([]),
        person_detector=FakePersonDetector([]),
        on_finished=lambda search_id, target_ids: None,
    )

    def confirm() -> str:
        session._handle_decisions(
            target.target_id,
            target,
            [
                MatchDecision(
                    state=MatchState.CONFIRMED,
                    track_id=7,
                    bbox=np.asarray([20, 30, 120, 140], dtype=np.float32),
                    face_bbox=np.asarray([40, 50, 80, 90], dtype=np.float32),
                    similarity=0.75,
                    quality=0.8,
                    evidence_count=3,
                )
            ],
            np.full((160, 160, 3), 127, dtype=np.uint8),
        )
        return session.events.after(0, timeout=0)[0]["data"]["evidence_id"]

    return session, confirm


def test_acknowledged_release_stays_distinct_from_a_lifecycle_discard() -> None:
    session, confirm = _evidence_session("search-release-vs-discard")
    evidence_id = confirm()

    session.release_evidence(evidence_id)
    # Idempotent inside the TTL window so a lost 204 can be safely retried.
    session.release_evidence(evidence_id)

    with pytest.raises(PersonSearchError) as exc_info:
        session.get_evidence(evidence_id, "face_crop")
    assert exc_info.value.code == "evidence_released"
    assert exc_info.value.status_code == 404
    assert session.view().confirmed_results[0].evidence_available is False


def test_deferred_target_found_reports_evidence_wiped_by_the_terminal_transition() -> None:
    """A timed-out search must not advertise a crop it already destroyed.

    ``target_found`` is built at confirmation time but published at the terminal
    transition. Between the two, a timeout wipes the JPEGs, so the confirmation
    -time ``evidence_available`` is stale by the time anyone reads it.
    """
    session, confirm = _evidence_session("search-deferred-target-found")
    evidence_id = confirm()
    assert session._deferred_events[0][1]["evidence_available"] is True

    session._transition(SearchStatus.TIMED_OUT, None, publish=False)
    session._publish_terminal_event()

    found = [
        event for event in session.events.after(0, timeout=0) if event["type"] == "target_found"
    ]
    assert len(found) == 1
    assert found[0]["data"]["evidence_id"] == evidence_id
    assert found[0]["data"]["evidence_available"] is False


def test_deferred_target_found_keeps_evidence_live_through_normal_completion() -> None:
    """The refresh must not zero out the case the whole feature exists for."""
    session, confirm = _evidence_session("search-completed-target-found")
    evidence_id = confirm()

    session._transition(SearchStatus.COMPLETED, None, publish=False)
    session._publish_terminal_event()

    found = [
        event for event in session.events.after(0, timeout=0) if event["type"] == "target_found"
    ]
    assert found[0]["data"]["evidence_available"] is True
    crop, media_type = session.get_evidence(evidence_id, "face_crop")
    assert media_type == "image/jpeg"
    assert crop


def test_sensitive_cleanup_retries_after_one_component_fails() -> None:
    """A partial cleanup must not set the terminal idempotence flag."""
    session, _ = _evidence_session("search-retry-sensitive-cleanup")
    confirmation = next(iter(session._confirmations.values()))
    calls = 0

    def flaky_cleanup() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated cleanup failure")

    confirmation.clear_sensitive = flaky_cleanup  # type: ignore[method-assign]

    session.clear_sensitive_state()
    assert calls == 1
    assert session._sensitive_cleared is False

    session.clear_sensitive_state()
    assert calls == 2
    assert session._sensitive_cleared is True
    assert all(target.embedding is None for target in session.targets)


def test_deferred_sensitive_cleanup_retries_transient_component_failure() -> None:
    session, _ = _evidence_session("search-auto-retry-sensitive-cleanup")
    confirmation = next(iter(session._confirmations.values()))
    calls = 0

    def flaky_cleanup() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated transient cleanup failure")

    confirmation.clear_sensitive = flaky_cleanup  # type: ignore[method-assign]

    session.defer_sensitive_cleanup()

    assert calls == 2
    assert session._sensitive_cleared is True
    assert session._sensitive_cleanup_failed is False


def test_sensitive_cleanup_continues_after_debug_reader_and_tracker_failures(monkeypatch) -> None:
    session, _ = _evidence_session("search-fault-safe-cleanup")
    face = make_face()
    session._debug_faces = [(face, "person_strict", 0.9)]

    class FlakyDebugFaces(list):
        def __init__(self, values):
            super().__init__(values)
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1
            if self.clear_calls == 1:
                raise RuntimeError("debug storage failure")
            super().clear()

    session._debug_faces = FlakyDebugFaces(session._debug_faces)

    class FlakyReader:
        def __init__(self) -> None:
            self.stop_calls = 0
            self.clear_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

        def clear(self) -> None:
            self.clear_calls += 1
            if self.clear_calls == 1:
                raise RuntimeError("reader storage failure")

    reader = FlakyReader()
    session._reader = reader  # type: ignore[assignment]

    class FlakyPreview:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1
            if self.clear_calls == 1:
                raise RuntimeError("preview storage failure")

    preview = FlakyPreview()
    session.preview = preview  # type: ignore[assignment]

    class FlakyTracker:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1
            if self.reset_calls == 1:
                raise RuntimeError("tracker storage failure")

    session._tracker = FlakyTracker()  # type: ignore[assignment]
    session._face_tracker = FlakyTracker()  # type: ignore[assignment]

    original_clone = _clone_target
    clone_calls = 0

    def flaky_clone(target, *, include_embedding=True):
        nonlocal clone_calls
        clone_calls += 1
        if clone_calls == 1:
            raise RuntimeError("target snapshot failure")
        return original_clone(target, include_embedding=include_embedding)

    monkeypatch.setattr("person_search.service._clone_target", flaky_clone)

    original_sanitize = _sanitize_source
    sanitize_calls = 0

    def flaky_sanitize(source):
        nonlocal sanitize_calls
        sanitize_calls += 1
        if sanitize_calls == 1:
            raise RuntimeError("source sanitizer failure")
        return original_sanitize(source)

    monkeypatch.setattr("person_search.service._sanitize_source", flaky_sanitize)
    session.source = SourceConfig(
        type="rtsp", uri="rtsp://user:secret@example.test:8554/live"
    )

    session.clear_sensitive_state()

    # Every component was attempted even though several independent operations
    # failed. The reader remains retained for a retry, and the source fallback
    # cannot leak the original RTSP credentials.
    assert reader.stop_calls == 1
    assert reader.clear_calls == 1
    assert preview.clear_calls == 1
    assert session._sensitive_cleared is False
    assert session._reader_cleanup_pending is reader
    assert "secret" not in str(session.source.model_dump())
    assert np.all(face.embedding == 0)

    session.clear_sensitive_state()

    # The stop call succeeded on the first pass and is intentionally
    # de-duplicated across retries; only the failed queue flush is repeated.
    assert reader.stop_calls == 1
    assert reader.clear_calls == 2
    assert preview.clear_calls == 2
    assert session._sensitive_cleared is True
    assert session._reader_cleanup_pending is None


def test_worker_finished_event_survives_reader_and_preview_cleanup_errors(monkeypatch) -> None:
    session, _ = _evidence_session("search-finished-after-cleanup-errors")

    class BrokenReader:
        def __init__(self, source, settings, on_status, on_drop):
            self.ended = threading.Event()
            self.ended.set()
            self.start_calls = 0
            self.stop_calls = 0
            self.clear_calls = 0

        def start(self) -> None:
            self.start_calls += 1

        def get(self, timeout=0.5):
            return None

        def stop(self) -> None:
            self.stop_calls += 1
            raise RuntimeError("reader stop failure")

        def clear(self) -> None:
            self.clear_calls += 1
            raise RuntimeError("reader clear failure")

    monkeypatch.setattr("person_search.service.LatestFrameReader", BrokenReader)

    class BrokenPreview:
        def clear(self) -> None:
            raise RuntimeError("preview clear failure")

    session.preview = BrokenPreview()  # type: ignore[assignment]
    session._run()

    assert session.finished.is_set()
    assert session.status == SearchStatus.FAILED


def test_reader_constructor_failure_still_finishes_and_releases_slot(monkeypatch) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target_view = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")

    def fail_reader(*args, **kwargs):
        raise RuntimeError("reader constructor failure")

    monkeypatch.setattr("person_search.service.LatestFrameReader", fail_reader)
    search = manager.start_search(
        target_view.target_id, SourceConfig(type="camera", device_index=0)
    )
    session = manager.get_session(search.search_id)

    assert session.finished.wait(timeout=2.0)
    assert session.status == SearchStatus.FAILED
    assert manager.active_search() is None
    assert session._sensitive_cleared is True


def test_manager_completion_callback_is_fail_safe_and_next_search_can_start(monkeypatch) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    first_target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    second_target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "李四")
    monkeypatch.setattr(SearchSession, "start", lambda session: None)

    first = manager.start_search(
        first_target.target_id, SourceConfig(type="camera", device_index=0)
    )
    session = manager.get_session(first.search_id)
    session._transition(SearchStatus.COMPLETED, None, publish=False)

    original_prune = manager._prune_sessions
    prune_calls = 0

    def fail_once() -> None:
        nonlocal prune_calls
        prune_calls += 1
        if prune_calls == 1:
            raise RuntimeError("simulated janitor failure")
        original_prune()

    monkeypatch.setattr(manager, "_prune_sessions", fail_once)
    # The completion callback itself must not leak the janitor exception.
    manager._on_finished(first.search_id, [first_target.target_id])

    assert manager.active_search() is None
    assert first_target.target_id not in manager._targets

    # Restore the normal janitor before exercising the next request.
    monkeypatch.setattr(manager, "_prune_sessions", original_prune)
    second = manager.start_search(
        second_target.target_id, SourceConfig(type="camera", device_index=1)
    )
    assert second.search_id != first.search_id


def test_delete_target_ignores_a_stale_active_search_id() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    manager._active_search_id = "missing-session"

    assert manager.delete_target(target.target_id) is True
    assert manager._active_search_id is None


def test_active_search_clears_a_stale_slot_pointer() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    manager._active_search_id = "missing-session"

    assert manager.active_search() is None
    assert manager._active_search_id is None


def test_start_search_ignores_terminal_session_left_in_active_slot(monkeypatch) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    first_target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    second_target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "李四")
    monkeypatch.setattr(SearchSession, "start", lambda session: None)

    first = manager.start_search(
        first_target.target_id, SourceConfig(type="camera", device_index=0)
    )
    first_session = manager.get_session(first.search_id)
    first_session._transition(SearchStatus.COMPLETED, None, publish=False)
    first_session._finished_at = time.monotonic()
    first_session._finished.set()
    # Deliberately leave ``_active_search_id`` untouched as a failed callback or
    # an older integration could do. The next request must not see a capacity
    # conflict from this terminal session.
    replacement = manager.start_search(
        second_target.target_id, SourceConfig(type="camera", device_index=1)
    )

    assert replacement.search_id != first.search_id
    assert manager._active_search_id == replacement.search_id
    manager.shutdown()


def test_terminal_session_ttl_prunes_without_a_followup_request() -> None:
    settings = Settings(terminal_session_ttl_seconds=0.05)
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), FakePersonDetector())
    target_view = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    target = manager.get_target(target_view.target_id)
    terminal_session = SearchSession(
        search_id="terminal-only-timer",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=settings,
        face_backend=manager.face_backend,
        person_detector=manager.person_detector,
        on_finished=lambda search_id, target_ids: None,
        request_id="terminal-only-request",
    )
    terminal_session._finished_at = time.monotonic()
    terminal_session._finished.set()
    with manager._lock:
        manager._sessions[terminal_session.search_id] = terminal_session
        manager._request_index[terminal_session.request_id] = terminal_session.search_id  # type: ignore[index]
    manager._schedule_prune_timer()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with manager._lock:
            removed = terminal_session.search_id not in manager._sessions
        if removed:
            break
        time.sleep(0.01)

    with manager._lock:
        assert terminal_session.search_id not in manager._sessions
        assert "terminal-only-request" not in manager._request_index
    assert terminal_session._sensitive_cleared is True
    manager.shutdown()


def test_prune_sessions_is_best_effort_and_keeps_janitor_alive() -> None:
    settings = Settings(terminal_session_ttl_seconds=60.0)
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), FakePersonDetector())
    targets = [
        manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), name)
        for name in ("过期一", "过期二", "保留")
    ]
    sessions: list[SearchSession] = []
    for index, target_view in enumerate(targets):
        target = manager.get_target(target_view.target_id)
        session = SearchSession(
            search_id=f"prune-{index}",
            target=target,
            source=SourceConfig(type="camera", device_index=0),
            settings=settings,
            face_backend=manager.face_backend,
            person_detector=manager.person_detector,
            on_finished=lambda search_id, target_ids: None,
        )
        session._finished.set()
        session._finished_at = (
            time.monotonic() - 61.0 if index < 2 else time.monotonic()
        )
        sessions.append(session)
    first, second, retained = sessions
    first.clear_evidence = lambda: (_ for _ in ()).throw(RuntimeError("evidence boom"))  # type: ignore[method-assign]
    first.clear_sensitive_state = lambda: (_ for _ in ()).throw(RuntimeError("sensitive boom"))  # type: ignore[method-assign]
    second_calls: list[str] = []
    second.clear_evidence = lambda: second_calls.append("evidence")  # type: ignore[method-assign]

    def clear_second_sensitive() -> None:
        second_calls.append("sensitive")
        second._sensitive_cleared = True

    second.clear_sensitive_state = clear_second_sensitive  # type: ignore[method-assign]
    with manager._lock:
        manager._sessions.update({session.search_id: session for session in sessions})

    manager._prune_sessions()

    assert first.search_id not in manager._sessions
    assert second.search_id not in manager._sessions
    assert retained.search_id in manager._sessions
    assert second_calls == ["evidence", "sensitive"]
    assert manager._prune_timer is not None
    manager.shutdown()


def test_status_reads_reuse_the_existing_prune_timer() -> None:
    settings = Settings(terminal_session_ttl_seconds=60.0)
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), FakePersonDetector())
    target_view = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    target = manager.get_target(target_view.target_id)
    session = SearchSession(
        search_id="timer-reuse",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=settings,
        face_backend=manager.face_backend,
        person_detector=manager.person_detector,
        on_finished=lambda search_id, target_ids: None,
    )
    session._finished_at = time.monotonic()
    session._finished.set()
    with manager._lock:
        manager._sessions[session.search_id] = session
    manager._schedule_prune_timer()
    timer = manager._prune_timer
    assert timer is not None

    assert manager.active_search() is None
    assert manager.get_search(session.search_id).search_id == session.search_id

    assert manager._prune_timer is timer
    manager.shutdown()


def test_shutdown_cancels_prune_timer_and_closes_manager_without_active_search() -> None:
    settings = Settings(terminal_session_ttl_seconds=60.0)
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), FakePersonDetector())
    target_view = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    target = manager.get_target(target_view.target_id)
    session = SearchSession(
        search_id="shutdown-terminal",
        target=target,
        source=SourceConfig(type="camera", device_index=0),
        settings=settings,
        face_backend=manager.face_backend,
        person_detector=manager.person_detector,
        on_finished=lambda search_id, target_ids: None,
    )
    session._finished_at = time.monotonic()
    session._finished.set()
    with manager._lock:
        manager._sessions[session.search_id] = session
    manager._schedule_prune_timer()
    timer = manager._prune_timer
    assert timer is not None

    manager.shutdown()

    assert manager._prune_timer is None
    assert manager._prune_timer_deadline is None
    assert manager._active_search_id is None
    timer.join(timeout=0.5)
    assert not timer.is_alive()
    with pytest.raises(PersonSearchError) as enroll_error:
        manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "拒绝")
    assert enroll_error.value.code == "manager_shutdown"
    with pytest.raises(PersonSearchError) as start_error:
        manager.start_search(target_view.target_id, SourceConfig(type="camera", device_index=0))
    assert start_error.value.code == "manager_shutdown"


def test_shutdown_stops_active_search_and_cancels_timer(monkeypatch) -> None:
    settings = Settings(terminal_session_ttl_seconds=60.0)
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), FakePersonDetector())
    target_view = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    monkeypatch.setattr(SearchSession, "start", lambda session: None)
    search = manager.start_search(
        target_view.target_id, SourceConfig(type="camera", device_index=0)
    )
    active_session = manager.get_session(search.search_id)

    # Add a retained terminal session solely to exercise cancellation while an
    # active worker occupies the singleton slot.
    terminal_target_view = TargetView(
        target_id="shutdown-retained-target",
        name="保留",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake-arcface",
    )
    terminal_target = Target(
        terminal_target_view.target_id,
        np.asarray([1.0, 0.0], dtype=np.float32),
        terminal_target_view,
        terminal_target_view.name,
    )
    terminal_session = SearchSession(
        search_id="shutdown-retained",
        target=terminal_target,
        source=SourceConfig(type="camera", device_index=0),
        settings=settings,
        face_backend=manager.face_backend,
        person_detector=manager.person_detector,
        on_finished=lambda search_id, target_ids: None,
    )
    terminal_session._finished_at = time.monotonic()
    terminal_session._finished.set()
    with manager._lock:
        manager._sessions[terminal_session.search_id] = terminal_session
    manager._schedule_prune_timer()
    timer = manager._prune_timer
    assert timer is not None

    manager.shutdown()

    assert manager._active_search_id is None
    assert active_session.status == SearchStatus.STOPPING
    assert manager._prune_timer is None
    timer.join(timeout=0.5)
    assert not timer.is_alive()


def test_shutdown_defers_sensitive_cleanup_until_a_timed_out_worker_exits(monkeypatch) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    release_worker = threading.Event()

    def start_blocked(session: SearchSession) -> None:
        def worker() -> None:
            release_worker.wait(timeout=2.0)
            session._finished.set()

        thread = threading.Thread(target=worker, daemon=True)
        session._worker = thread
        thread.start()

    monkeypatch.setattr(SearchSession, "start", start_blocked)
    original_stop = SearchSession.stop
    monkeypatch.setattr(
        SearchSession,
        "stop",
        lambda session: original_stop(session, timeout=0),
    )
    search = manager.start_search(
        target.target_id, SourceConfig(type="camera", device_index=0)
    )
    session = manager.get_session(search.search_id)

    manager.shutdown()
    # The worker is still using its private gallery copy; shutdown must not race
    # and zero it underneath the provider.  The deferred watcher handles it once
    # the worker publishes its terminal event.
    assert session._sensitive_cleared is False
    assert session.targets[0].embedding is not None

    release_worker.set()
    deadline = time.monotonic() + 2.0
    while not session._sensitive_cleared and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session._sensitive_cleared is True
    assert all(target.embedding is None for target in session.targets)


def test_safe_model_error_does_not_expose_loader_path() -> None:
    message = _safe_error(ModelUnavailableError("failed to load /srv/secrets/models/arcface.onnx"))

    assert "/srv/secrets" not in message
    assert message == "model unavailable; verify model files and runtime configuration"


def test_source_epoch_reset_wipes_old_debug_face_and_confirmation_state() -> None:
    session, _ = _evidence_session("search-source-epoch-reset")
    face = make_face()
    session._debug_faces = [(face, "person_strict", 0.9)]
    confirmation = next(iter(session._confirmations.values()))
    calls = 0

    def record_reset() -> None:
        nonlocal calls
        calls += 1

    confirmation.clear_sensitive = record_reset  # type: ignore[method-assign]
    embedding = face.embedding

    session._reset_temporal_state()

    assert calls == 1
    assert session._debug_faces == []
    assert embedding is not None
    assert np.all(embedding == 0)
