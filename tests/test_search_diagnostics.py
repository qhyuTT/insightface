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

    target = TargetSearchView(target_id="target-1", name="目标")
    assert target.best_observed_similarity is None
    assert target.last_face_px is None
    assert target.evidence_count == 0
    assert target.required_evidence == 0
    assert target.last_rejection_reason is None
