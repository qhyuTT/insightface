from __future__ import annotations

import json

import pytest

from person_search.evaluation import (
    FACE_PX_BUCKETS,
    aggregate_threshold_results,
    face_px_bucket,
    load_manifest,
    recommend_threshold,
    summarize_events,
    summarize_similarity_samples,
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


def test_manifest_v2_loads_bucketed_interval_objects(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "distance-buckets",
                        "photo": "target.jpg",
                        "video": "office.mp4",
                        "target_name": "Alice",
                        "expected_intervals_seconds": [
                            {"start": 1.0, "end": 3.0, "face_px_bucket": "48-55"},
                            {"start": 5.0, "end": 8.0, "face_px_bucket": "64-79"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    case = load_manifest(manifest)[0]
    assert case.expected_intervals_seconds == ((1.0, 3.0), (5.0, 8.0))
    assert case.expected_face_px_buckets == ("48-55", "64-79")


def test_manifest_v2_requires_bucket_and_object_shape(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "bad-v2",
                        "photo": "target.jpg",
                        "video": "office.mp4",
                        "target_name": "Alice",
                        "expected_intervals_seconds": [{"start": 1.0, "end": 3.0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="face_px_bucket"):
        load_manifest(manifest)


def test_manifest_v2_rejects_unknown_face_px_bucket(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "bad-bucket",
                        "photo": "target.jpg",
                        "video": "office.mp4",
                        "target_name": "Alice",
                        "expected_intervals_seconds": [
                            {"start": 1.0, "end": 3.0, "face_px_bucket": "tiny"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be one of"):
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


def test_summarizes_and_aggregates_metrics_by_face_px_bucket() -> None:
    events = [
        {"state": "confirmed", "timestamp_seconds": 12.0},
        {"state": "confirmed", "timestamp_seconds": 45.0},
    ]
    metrics = summarize_events(
        events,
        ((10.0, 20.0), (30.0, 40.0), (40.0, 50.0)),
        100.0,
        ("48-55", "48-55", "64-79"),
    )

    small = metrics["face_px_buckets"]["48-55"]
    assert small["expected_intervals"] == 2
    assert small["detected_intervals"] == 1
    assert small["interval_recall"] == 0.5
    assert small["mean_confirmation_latency_seconds"] == 2.0
    assert small["p95_confirmation_latency_seconds"] == 2.0

    key = threshold_key(0.6)
    aggregate = aggregate_threshold_results(
        [{"threshold_results": {key: {"metrics": metrics}}}], (0.6,)
    )[key]
    assert aggregate["face_px_buckets"]["48-55"]["interval_recall"] == 0.5
    assert aggregate["face_px_buckets"]["64-79"]["mean_confirmation_latency_seconds"] == 5.0


def test_below_floor_interval_requires_rejection_and_counts_confirmation_as_false() -> None:
    metrics = summarize_events(
        [
            {"state": "confirmed", "timestamp_seconds": 12.0},
            {"state": "confirmed", "timestamp_seconds": 32.0},
        ],
        ((10.0, 20.0), (30.0, 40.0)),
        100.0,
        ("<48", "48-55"),
    )

    assert metrics["expected_intervals"] == 1
    assert metrics["detected_intervals"] == 1
    assert metrics["interval_recall"] == 1.0
    assert metrics["false_confirmations"] == 1
    assert metrics["negative_exposure_seconds"] == 90.0
    assert metrics["face_px_buckets"]["<48"] == {
        "expected_intervals": 1,
        "unexpected_confirmations": 1,
        "passed": False,
    }


def test_shadow_events_are_isolated_from_production_metrics() -> None:
    events = [
        {"state": "shadow_confirmed", "timestamp_seconds": 12.0},
        {"state": "shadow_lost", "timestamp_seconds": 15.0},
        {"state": "confirmed", "timestamp_seconds": 32.0},
    ]
    intervals = ((10.0, 20.0), (30.0, 40.0))
    buckets = ("48-55", ">=80")

    production = summarize_events(events, intervals, 100.0, buckets)
    shadow = summarize_events(
        events,
        intervals,
        100.0,
        buckets,
        confirmation_state="shadow_confirmed",
    )

    assert production["detected_intervals"] == 1
    assert production["confirmation_latencies_seconds"] == [2.0]
    assert shadow["detected_intervals"] == 1
    assert shadow["confirmation_latencies_seconds"] == [2.0]


def test_shadow_aggregate_cannot_influence_production_recommendation() -> None:
    key = threshold_key(0.6)
    common = {
        "expected_intervals": 10,
        "false_confirmations": 0,
        "negative_exposure_seconds": 360000.0,
        "false_confirmations_per_hour": 0.0,
        "face_px_buckets": {},
    }
    case_results = [
        {
            "threshold_results": {
                key: {
                    "metrics": {
                        **common,
                        "detected_intervals": 0,
                        "interval_recall": 0.0,
                        "confirmation_latencies_seconds": [],
                    },
                    "shadow_metrics": {
                        **common,
                        "detected_intervals": 10,
                        "interval_recall": 1.0,
                        "confirmation_latencies_seconds": [0.5] * 10,
                    },
                }
            }
        }
    ]

    production = aggregate_threshold_results(case_results, (0.6,))
    shadow = aggregate_threshold_results(case_results, (0.6,), metrics_key="shadow_metrics")

    assert recommend_threshold(production) is None
    assert shadow[key]["passed"]


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
                        "negative_exposure_seconds": 360000.0,
                        "confirmation_latencies_seconds": [0.5] * 10,
                    }
                },
                threshold_key(0.6): {
                    "metrics": {
                        "expected_intervals": 10,
                        "detected_intervals": 9,
                        "false_confirmations": 0,
                        "negative_exposure_seconds": 360000.0,
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
                        "negative_exposure_seconds": 359999.0,
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


def test_similarity_distribution_derives_labels_from_expected_intervals() -> None:
    """Interval-labelled footage has no per-face truth, so the labels are derived."""
    samples = [
        # Inside the interval: the highest-scoring face of the frame is the target,
        # the other one is a bystander and is left out rather than guessed at.
        {"frame_id": 0, "timestamp_seconds": 0.5, "similarity": 0.71, "face_px_bucket": "48-55"},
        {"frame_id": 0, "timestamp_seconds": 0.5, "similarity": 0.18, "face_px_bucket": "48-55"},
        {"frame_id": 1, "timestamp_seconds": 0.9, "similarity": 0.66, "face_px_bucket": "48-55"},
        # Outside every interval the target is absent, so every face is an impostor.
        {"frame_id": 9, "timestamp_seconds": 5.0, "similarity": 0.33, "face_px_bucket": "64-79"},
        {"frame_id": 9, "timestamp_seconds": 5.0, "similarity": 0.29, "face_px_bucket": "64-79"},
    ]

    summary = summarize_similarity_samples(samples, ((0.0, 1.0),))

    assert summary["labelling"] == "derived_from_expected_intervals"
    assert summary["observations"] == 5
    assert summary["genuine"]["count"] == 2
    assert summary["genuine"]["min"] == pytest.approx(0.66)
    assert summary["impostor"]["count"] == 2
    assert summary["impostor"]["max"] == pytest.approx(0.33)
    assert summary["by_face_px_bucket"]["64-79"]["observations"] == 2


def test_similarity_distribution_reports_when_it_cannot_label() -> None:
    samples = [
        {"frame_id": 0, "timestamp_seconds": 0.5, "similarity": 0.71, "face_px_bucket": "48-55"}
    ]

    summary = summarize_similarity_samples(samples, None)

    # Saying "unlabelled" is the point: a distribution nobody can split into
    # genuine and impostor cannot justify a threshold.
    assert summary["labelling"] == "unlabelled"
    assert summary["genuine"] == {"count": 0}
    assert summary["impostor"] == {"count": 0}
    assert summary["by_face_px_bucket"]["48-55"]["observations"] == 1


def test_face_px_bucket_uses_the_manifest_vocabulary() -> None:
    assert face_px_bucket(47) == "<48"
    assert face_px_bucket(48) == "48-55"
    assert face_px_bucket(56) == "56-63"
    assert face_px_bucket(64) == "64-79"
    assert face_px_bucket(80) == ">=80"
    assert face_px_bucket(80) in FACE_PX_BUCKETS
