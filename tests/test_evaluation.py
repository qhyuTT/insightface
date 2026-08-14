from __future__ import annotations

import json

import pytest

from person_search.evaluation import (
    aggregate_threshold_results,
    load_manifest,
    recommend_threshold,
    summarize_events,
    threshold_key,
    validate_thresholds,
)


def test_manifest_resolves_paths_and_validates_intervals(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "positive-1",
                        "photo": "target.jpg",
                        "video": "airport.mp4",
                        "target_name": "Alice",
                        "expected_intervals_seconds": [[1.0, 3.0], [5.0, 8.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    case = load_manifest(manifest)[0]
    assert case.photo == tmp_path / "target.jpg"
    assert case.video == tmp_path / "airport.mp4"
    assert case.expected_intervals_seconds == ((1.0, 3.0), (5.0, 8.0))


def test_manifest_rejects_overlapping_intervals(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "bad",
                        "photo": "target.jpg",
                        "video": "airport.mp4",
                        "target_name": "Alice",
                        "expected_intervals_seconds": [[1.0, 3.0], [2.0, 4.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-overlapping"):
        load_manifest(manifest)


def test_summarizes_interval_recall_false_confirmations_and_latency() -> None:
    events = [
        {"state": "confirmed", "timestamp_seconds": 12.0},
        {"state": "confirmed", "timestamp_seconds": 42.0},
        {"state": "confirmed", "timestamp_seconds": 80.0},
    ]
    result = summarize_events(events, ((10.0, 20.0), (40.0, 50.0)), 100.0)
    assert result["interval_recall"] == 1.0
    assert result["false_confirmations"] == 1
    assert result["negative_exposure_seconds"] == 80.0
    assert result["confirmation_latencies_seconds"] == [2.0, 2.0]


def test_adjacent_intervals_do_not_share_boundary_confirmation() -> None:
    events = [{"state": "confirmed", "timestamp_seconds": 20.0}]
    result = summarize_events(events, ((10.0, 20.0), (20.0, 30.0)), 40.0)
    assert result["detected_intervals"] == 1
    assert result["confirmation_latencies_seconds"] == [0.0]


def test_recommends_only_threshold_meeting_recall_rate_and_exposure() -> None:
    thresholds = (0.55, 0.6)
    case_results = [
        {
            "threshold_results": {
                threshold_key(0.55): {
                    "metrics": {
                        "expected_intervals": 10,
                        "detected_intervals": 10,
                        "false_confirmations": 2,
                        "negative_exposure_seconds": 36000.0,
                        "confirmation_latencies_seconds": [0.5] * 10,
                    }
                },
                threshold_key(0.6): {
                    "metrics": {
                        "expected_intervals": 10,
                        "detected_intervals": 9,
                        "false_confirmations": 1,
                        "negative_exposure_seconds": 36000.0,
                        "confirmation_latencies_seconds": [0.8] * 9,
                    }
                },
            }
        }
    ]
    aggregate = aggregate_threshold_results(case_results, thresholds)
    assert not aggregate[threshold_key(0.55)]["passed"]
    assert aggregate[threshold_key(0.6)]["passed"]
    assert recommend_threshold(aggregate) == 0.6


def test_insufficient_negative_exposure_never_recommends_threshold() -> None:
    case_results = [
        {
            "threshold_results": {
                threshold_key(0.55): {
                    "metrics": {
                        "expected_intervals": 1,
                        "detected_intervals": 1,
                        "false_confirmations": 0,
                        "negative_exposure_seconds": 35999.0,
                        "confirmation_latencies_seconds": [0.5],
                    }
                }
            }
        }
    ]
    aggregate = aggregate_threshold_results(case_results, (0.55,))
    assert recommend_threshold(aggregate) is None
    assert not aggregate[threshold_key(0.55)]["negative_exposure_sufficient"]


def test_thresholds_must_be_unique_and_bounded() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_thresholds([0.55, 0.55])
    with pytest.raises(ValueError, match="between"):
        validate_thresholds([1.1])
