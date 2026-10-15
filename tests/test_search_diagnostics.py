from __future__ import annotations

import pytest

from person_search.domain import (
    SearchMetrics,
    SearchStatus,
    SearchView,
    SourceConfig,
    TargetSearchView,
)


def test_search_metrics_snapshot_serializes_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr("time.monotonic", lambda: 14.0)
    metrics = SearchMetrics(
        frame_count=20,
        started_at=10.0,
        latencies_ms=[10.0, 20.0, 30.0],
        face_size_counts={"gte80": 2, "48_63": 4},
        face_source_counts={"roi": 3, "full_frame": 5},
        match_stage_counts={"confirmed": 1, "quality_accepted": 6},
        track_sampling_hz=[2.5, 3.5],
        unconfirmed_gate_counts={"insufficient_samples": 2},
        time_to_confirm_seconds=[0.8],
        track_dwell_seconds=[1.2],
        stage_latencies_ms={
            "face": [10.0, 20.0, 30.0],
            "person": [5.0, 15.0],
            "unused": [],
        },
        stage_call_counts={"face": 8, "person": 4},
    )

    snapshot = metrics.snapshot()

    assert snapshot["processed_fps"] == pytest.approx(5.0)
    assert snapshot["p95_latency_ms"] == pytest.approx(29.0)
    assert snapshot["face_size_counts"] == {"48_63": 4, "gte80": 2}
    assert snapshot["face_source_counts"] == {"full_frame": 5, "roi": 3}
    assert snapshot["match_stage_counts"] == {"confirmed": 1, "quality_accepted": 6}
    view = SearchView(
        search_id="search-1",
        status=SearchStatus.RUNNING,
        source=SourceConfig(type="camera", device_index=0),
        **snapshot,
    )
    assert view.achieved_sampling_hz == pytest.approx(3.0)
    assert view.unconfirmed_gate_counts == {"insufficient_samples": 2}
    assert view.confirmed_tracks == 1
    assert view.time_to_confirm_p50_seconds == pytest.approx(0.8)
    assert snapshot["stage_p95_latency_ms"] == {
        "face": pytest.approx(29.0),
        "person": pytest.approx(14.5),
    }
    assert snapshot["effective_hz"] == {
        "face": pytest.approx(2.0),
        "person": pytest.approx(1.0),
    }


def test_search_view_diagnostic_fields_are_backward_compatible() -> None:
    view = SearchView(
        search_id="search-1",
        status=SearchStatus.RUNNING,
        source=SourceConfig(type="camera", device_index=0),
    )

    assert view.face_size_counts == {}
    assert view.face_source_counts == {}
    assert view.match_stage_counts == {}
    assert view.stage_p95_latency_ms == {}
    assert view.effective_hz == {}
    assert view.preview_fps == 0.0
    assert view.preview_frame_age_p95_ms == 0.0

    target = TargetSearchView(target_id="target-1", name="目标")
    assert target.best_observed_similarity is None
    assert target.last_face_px is None
    assert target.evidence_count == 0
    assert target.required_evidence == 0
    assert target.last_rejection_reason is None


def test_recent_rates_decay_and_stage_statistics_expose_sample_scope(monkeypatch) -> None:
    monkeypatch.setattr('time.monotonic', lambda: 100.0)
    metrics = SearchMetrics(started_at=1.0, frame_count=900,
                            processed_at=[80.0, 91.0, 95.0],
                            preview_published_at=[80.0, 91.0, 95.0],
                            stage_latencies_ms={'face_full': [10.0, 20.0, 90.0]})
    snapshot = metrics.snapshot()
    assert snapshot['processed_fps'] == pytest.approx(0.2)
    assert snapshot['preview_fps'] == pytest.approx(0.2)
    assert snapshot['source_fps'] == 0.0
    assert snapshot['stage_p50_latency_ms']['face_full'] == 20.0
    assert snapshot['stage_max_latency_ms']['face_full'] == 90.0
    assert snapshot['stage_sample_counts']['face_full'] == 3
    monkeypatch.setattr('time.monotonic', lambda: 106.0)
    assert metrics.snapshot()['processed_fps'] == 0.0
    assert metrics.snapshot()['preview_fps'] == 0.0


def test_stage_rates_use_recent_completions_and_expire_after_source_loss(monkeypatch) -> None:
    from person_search.domain import _window_fps
    monkeypatch.setattr('time.monotonic', lambda: 100.0)
    metrics = SearchMetrics(started_at=0.0, stage_call_counts={'face_full': 200},
                            stage_called_at={'face_full': [80.0, 91.0, 95.0]})
    # A monotonic origin of zero is valid; do not substitute the first sample.
    assert _window_fps([8.0, 9.0], 10.0, started_at=0.0) == pytest.approx(0.2)
    # SearchMetrics uses started_at=0 as the legacy not-started sentinel.
    metrics.started_at = 1.0
    assert metrics.snapshot()['effective_hz']['face_full'] == pytest.approx(0.2)
    monkeypatch.setattr('time.monotonic', lambda: 106.0)
    assert metrics.snapshot()['effective_hz']['face_full'] == 0.0
