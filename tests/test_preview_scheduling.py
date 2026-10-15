from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from person_search.config import Settings
from person_search.domain import SearchMetrics, SourceConfig, Track
from person_search.service import SearchSession
from person_search.video import LatestFrameReader


class _Clock:
    now = 0.0


class _ClockEvent:
    def __init__(self, clock, *, stop_at=float("inf")):
        self.clock = clock
        self.stop_at = stop_at
        self.flag = False
        self.waits = []

    def is_set(self):
        return self.flag or self.clock.now >= self.stop_at

    def set(self):
        self.flag = True

    def wait(self, seconds):
        self.waits.append(seconds)
        self.clock.now += seconds
        return self.is_set()


class _ScheduledReader:
    """A source clock independent of both encoding and the unused AI queue."""

    def __init__(self, clock, arrivals):
        self.clock = clock
        self.arrivals = arrivals
        self.wait_calls = 0

    def latest(self):
        arrived = [index for index, at in enumerate(self.arrivals) if at <= self.clock.now]
        if not arrived:
            return None
        index = arrived[-1]
        return SimpleNamespace(frame_id=index, captured_at=self.arrivals[index], frame=index)

    def wait_for_new_latest(self, last_id, timeout):
        self.wait_calls += 1
        packet = self.latest()
        if packet is not None and packet.frame_id != last_id:
            return packet
        future = [at for at in self.arrivals if at > self.clock.now]
        self.clock.now = min(self.clock.now + timeout, future[0] if future else float("inf"))
        packet = self.latest()
        return packet if packet is not None and packet.frame_id != last_id else None


def _preview_session(clock, *, subscribers=True, stop_at=10.0):
    return SimpleNamespace(
        settings=SimpleNamespace(preview_hz=15.0),
        preview=SimpleNamespace(has_subscribers=subscribers),
        _preview_stop=_ClockEvent(clock, stop_at=stop_at),
        _stop=threading.Event(),
        _preview_lock=threading.Lock(),
        _preview_tracks=[],
        _preview_faces=[],
        _preview_debug_faces=[],
        _preview_track_states={},
    )


def _run_schedule(monkeypatch, arrivals, costs):
    clock = _Clock()
    monkeypatch.setattr("person_search.service.time.monotonic", lambda: clock.now)
    session = _preview_session(clock)
    reader = _ScheduledReader(clock, arrivals)
    published = []

    def encode(frame, *_args, **_kwargs):
        published.append((frame, clock.now))
        clock.now += costs[len(published) - 1]
        if len(published) == len(costs):
            session._preview_stop.set()
        return True

    session._publish_preview = encode
    SearchSession._preview_loop(session, reader)
    return published, session, reader


def test_fifteen_hz_preview_waits_for_frames_just_after_its_previous_deadline(monkeypatch):
    arrivals = [0.001 + index / 15 for index in range(8)]
    published, _, _ = _run_schedule(monkeypatch, arrivals, [0.01] * 8)
    assert [frame for frame, _ in published] == list(range(8))
    assert [started for _, started in published] == pytest.approx(arrivals)


def test_bursty_decode_does_not_force_an_empty_polling_period(monkeypatch):
    # Average 15fps, but frames arrive in pairs. A fixed polling phase can
    # collapse both arrivals into one preview even when encoding is fast enough.
    arrivals = [0.001, 0.002, 0.135, 0.136, 0.269, 0.270]
    published, _, _ = _run_schedule(monkeypatch, arrivals, [0.01] * 6)
    assert [frame for frame, _ in published] == list(range(6))
    starts = [started for _, started in published]
    assert starts == pytest.approx(
        [0.001, 0.001 + 1 / 15, 0.135, 0.135 + 1 / 15, 0.269, 0.269 + 1 / 15]
    )


def test_over_budget_encoder_gets_no_extra_sleep(monkeypatch):
    arrivals = [0.001 + index / 60 for index in range(100)]
    published, session, _ = _run_schedule(monkeypatch, arrivals, [0.09] * 4)
    assert [started for _, started in published] == pytest.approx([0.001, 0.091, 0.181, 0.271])
    assert session._preview_stop.waits == []


def test_fast_source_keeps_newest_frame_and_rate_cap_without_catch_up(monkeypatch):
    arrivals = [0.001 + index / 60 for index in range(100)]
    published, _, _ = _run_schedule(monkeypatch, arrivals, [0.2, 0.01, 0.01, 0.01])
    frames, starts = zip(*published)
    assert starts == pytest.approx([0.001, 0.201, 0.201 + 1 / 15, 0.201 + 2 / 15])
    assert frames == (0, 12, 16, 20)


def test_preview_never_republishes_one_frame(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("person_search.service.time.monotonic", lambda: clock.now)
    session = _preview_session(clock, stop_at=0.4)
    reader = _ScheduledReader(clock, [0.0])
    published = []
    session._publish_preview = lambda frame, *_args, **_kwargs: published.append(frame) or True
    SearchSession._preview_loop(session, reader)
    assert published == [0]


def test_preview_keeps_new_frames_but_suppresses_stale_or_wrong_epoch_boxes(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("person_search.service.time.monotonic", lambda: clock.now)
    session = _preview_session(clock)
    session._lock = threading.RLock()
    session.metrics = SearchMetrics()
    session._preview_tracks = [Track(7, np.array([1, 1, 3, 3], np.float32), 0.9)]
    session._preview_frame_id = 0
    session._preview_source_epoch = 0
    session._preview_captured_at = 0.0
    reader = _ScheduledReader(clock, [0.0, 0.2, 0.45])
    published = []

    def encode(frame, tracks, *_args, **_kwargs):
        published.append((frame, len(tracks)))
        if len(published) == 3:
            session._preview_stop.set()
        return True

    session._publish_preview = encode
    SearchSession._preview_loop(session, reader)
    assert published == [(0, 1), (1, 1), (2, 0)]
    assert session.metrics.preview_overlay_stale_frames == 1
    assert list(session.metrics.preview_overlay_ages_ms) == pytest.approx([0.0, 200.0])

    session._preview_source_epoch = 0
    session._preview_captured_at = 0.45
    session._preview_frame_id = 2
    session._preview_stop = _ClockEvent(clock)
    reader.latest = lambda: SimpleNamespace(frame_id=3, captured_at=0.5, source_epoch=1, frame=3)
    reader.wait_for_new_latest = lambda *_args, **_kwargs: reader.latest()
    session._publish_preview = lambda frame, tracks, *_args, **_kwargs: (
        published.append((frame, len(tracks))), session._preview_stop.set(), True
    )[-1]
    SearchSession._preview_loop(session, reader)
    assert published[-1] == (3, 0)


def test_no_subscriber_does_not_encode_or_wait_on_source(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("person_search.service.time.monotonic", lambda: clock.now)
    session = _preview_session(clock, subscribers=False, stop_at=0.3)
    reader = _ScheduledReader(clock, [0.0])
    session._publish_preview = lambda *_args, **_kwargs: pytest.fail("unwatched JPEG encoded")
    SearchSession._preview_loop(session, reader)
    assert reader.wait_calls == 0


def test_ended_reader_exits_preview_instead_of_spinning(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("person_search.service.time.monotonic", lambda: clock.now)
    session = _preview_session(clock)
    ended = threading.Event()
    ended.set()
    reader = SimpleNamespace(ended=ended, wait_for_new_latest=lambda *_args, **_kwargs: None)
    SearchSession._preview_loop(session, reader)
    assert clock.now == 0


def test_nonblocking_stopped_reader_cannot_busy_loop(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("person_search.service.time.monotonic", lambda: clock.now)
    session = _preview_session(clock, stop_at=0.3)
    reader = SimpleNamespace(wait_for_new_latest=lambda *_args, **_kwargs: None)
    SearchSession._preview_loop(session, reader)
    assert session._preview_stop.waits == pytest.approx([0.1, 0.1, 0.1])


class _Capture:
    def __init__(self):
        self.frames = queue.Queue()

    def isOpened(self):
        return True

    def read(self):
        value = self.frames.get()
        return (False, None) if value is None else (True, np.full((2, 2, 3), value, np.uint8))

    def release(self):
        pass

    def interrupt(self):
        self.frames.put(None)


def _controlled_reader(monkeypatch):
    capture = _Capture()
    reader = LatestFrameReader(
        SourceConfig(type="file", uri="/synthetic/video"),
        Settings(frame_queue_size=2),
        lambda *_: None,
        lambda: None,
    )
    monkeypatch.setattr(reader, "_open", lambda: capture)
    reader.start()
    return reader, capture


def test_new_frame_wakes_preview_without_consuming_ai_queue(monkeypatch):
    reader, capture = _controlled_reader(monkeypatch)
    waiting = threading.Event()
    result = []

    def receive():
        waiting.set()
        result.append(reader.wait_for_new_latest(-1, timeout=2))

    waiter = threading.Thread(target=receive)
    try:
        waiter.start()
        assert waiting.wait(1)
        capture.frames.put(7)
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        packet = result[0]
        assert packet.frame_id == 0 and np.all(packet.frame == 7)
        assert reader.get(timeout=1) is packet
        assert reader.wait_for_new_latest(packet.frame_id, timeout=0.001) is None
    finally:
        reader.stop()
        waiter.join(timeout=1)


@pytest.mark.parametrize("finish", ["stop", "eof"])
def test_source_stop_or_end_wakes_waiting_preview(monkeypatch, finish):
    reader, capture = _controlled_reader(monkeypatch)
    waiting = threading.Event()
    result = []

    def receive():
        waiting.set()
        result.append(reader.wait_for_new_latest(-1, timeout=2))

    waiter = threading.Thread(target=receive)
    try:
        waiter.start()
        assert waiting.wait(1)
        if finish == "stop":
            reader.stop()
        else:
            capture.frames.put(None)
        waiter.join(timeout=1)
        assert not waiter.is_alive() and result == [None]
    finally:
        reader.stop()
        waiter.join(timeout=1)


def test_clear_drops_old_preview_and_next_frame_remains_available_to_ai(monkeypatch):
    reader, capture = _controlled_reader(monkeypatch)
    try:
        capture.frames.put(3)
        old = reader.wait_for_new_latest(-1, timeout=1)
        assert old is not None
        assert reader.get(timeout=1) is old
        reader.clear()
        assert reader.wait_for_new_latest(old.frame_id, timeout=0.001) is None
        capture.frames.put(9)
        new = reader.wait_for_new_latest(old.frame_id, timeout=1)
        assert new is not None and new.frame_id > old.frame_id
        assert np.all(new.frame == 9)
        assert reader.get(timeout=1) is new
    finally:
        reader.stop()


def test_preview_thread_stop_is_bounded_while_source_has_no_new_frame(monkeypatch):
    reader = LatestFrameReader(
        SourceConfig(type="file", uri="/synthetic/video"), Settings(), lambda *_: None, lambda: None
    )
    session = _preview_session(_Clock())
    session.search_id = "synthetic-preview"
    session._preview_stop = threading.Event()
    session._preview_thread = None
    session._preview_loop = lambda source: SearchSession._preview_loop(session, source)
    session._publish_preview = lambda *_args, **_kwargs: pytest.fail("encoded without a frame")
    entered = threading.Event()
    wait_latest = reader.wait_for_new_latest

    def wait(last_id, timeout):
        entered.set()
        return wait_latest(last_id, timeout)

    monkeypatch.setattr(reader, "wait_for_new_latest", wait)
    try:
        SearchSession._start_preview_thread(session, reader)
        assert entered.wait(timeout=1)
        thread = session._preview_thread
        SearchSession._stop_preview_thread(session)
        assert not thread.is_alive() and session._preview_thread is None
    finally:
        session._preview_stop.set()
        reader.stop()
        if session._preview_thread is not None:
            session._preview_thread.join(timeout=1)
