from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from conftest import make_face

from person_search import cli
from person_search.cli import (
    _associate_search_faces,
    _confirmation_input_counts,
    _face_policies,
    _is_matchable_face,
    _offline_decision_state,
)
from person_search.config import Settings
from person_search.domain import Track
from person_search.face_tracking import FaceTracker


def test_offline_face_policies_match_live_small_and_fallback_paths() -> None:
    settings = Settings(
        min_search_face_px=64,
        preferred_search_face_px=80,
        similarity_threshold=0.55,
        small_face_similarity_threshold=0.60,
        evidence_required=3,
        small_face_evidence_required=4,
    )
    faces = [
        make_face(bbox=(10, 10, 74, 74)),
        make_face(bbox=(100, 10, 180, 90)),
        make_face(bbox=(200, 10, 280, 90)),
    ]

    policies = _face_policies(
        faces,
        {0: "person_strict", 1: "person_relaxed", 2: "face_fallback"},
        settings,
    )

    assert policies[0].threshold == 0.60
    assert policies[0].evidence_required == 4
    assert policies[0].suppress_candidate
    assert policies[1].threshold == 0.60
    assert policies[1].suppress_candidate
    assert policies[2].threshold == 0.60
    assert policies[2].suppress_candidate


def test_offline_association_uses_relaxed_person_then_face_fallback() -> None:
    settings = Settings(fallback_face_detection_threshold=0.55)
    relaxed_face = make_face(bbox=(20, 91, 84, 155))
    fallback_face = make_face(bbox=(140, 10, 204, 74))
    tracks = [
        Track(
            track_id=7,
            bbox=np.asarray([0, 0, 100, 160], dtype=np.float32),
            score=0.9,
        )
    ]

    all_tracks, associations, modes = _associate_search_faces(
        [relaxed_face, fallback_face],
        tracks,
        settings=settings,
        face_tracker=FaceTracker(),
        timestamp=1.0,
    )

    assert associations[0] == 7
    assert modes[0] == "person_relaxed"
    assert associations[1] < 0
    assert modes[1] == "face_fallback"
    assert {track.track_id for track in all_tracks} == {7, associations[1]}


def test_offline_tiny_face_requires_strict_person_association_without_fallback() -> None:
    settings = Settings(tiny_face_enabled=True)
    relaxed = make_face(bbox=(20, 65, 68, 113))
    unassociated = make_face(bbox=(140, 10, 188, 58))
    tracks = [
        Track(
            track_id=7,
            bbox=np.asarray([0, 0, 100, 120], dtype=np.float32),
            score=0.9,
        )
    ]

    all_tracks, associations, modes = _associate_search_faces(
        [relaxed, unassociated],
        tracks,
        settings=settings,
        face_tracker=FaceTracker(),
        timestamp=1.0,
    )

    assert associations == {}
    assert modes == {}
    assert [track.track_id for track in all_tracks] == [7]


def test_offline_matchable_face_enforces_hard_lower_bound() -> None:
    settings = Settings(tiny_face_enabled=True)

    assert not _is_matchable_face(make_face(bbox=(0, 0, 47, 47)), settings)
    assert _is_matchable_face(make_face(bbox=(0, 0, 48, 48)), settings)


def test_offline_stage_counts_distinguish_eligibility_from_threshold() -> None:
    settings = Settings(tiny_face_enabled=True)
    face = make_face(embedding=(0.5, 0.8660254), bbox=(0, 0, 48, 48))
    policies = _face_policies([face], {0: "person_strict"}, settings)

    counts = _confirmation_input_counts(
        [face],
        {0: 7},
        policies,
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert counts == {"above_threshold": 0, "evidence_eligible": 1}


def test_offline_normal_low_similarity_face_is_not_evidence_eligible() -> None:
    settings = Settings()
    face = make_face(embedding=(0.5, 0.8660254), bbox=(0, 0, 80, 80))
    policies = _face_policies([face], {0: "person_strict"}, settings)

    counts = _confirmation_input_counts(
        [face],
        {0: 7},
        policies,
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert counts == {"above_threshold": 0, "evidence_eligible": 0}


def test_offline_shadow_decision_states_are_distinct_from_production() -> None:
    assert _offline_decision_state("confirmed", shadow=False) == "confirmed"
    assert _offline_decision_state("confirmed", shadow=True) == "shadow_confirmed"
    assert _offline_decision_state("lost", shadow=True) == "shadow_lost"


def test_manifest_report_v2_keeps_shadow_out_of_recommendation(monkeypatch, tmp_path) -> None:
    case = SimpleNamespace(
        case_id="shadow-case",
        photo=Path("target.jpg"),
        video=Path("video.mp4"),
        target_name="Alice",
        expected_intervals_seconds=((0.0, 1.0),),
        expected_face_px_buckets=("48-55",),
    )
    common = {
        "expected_intervals": 1,
        "false_confirmations": 0,
        "negative_exposure_seconds": 360000.0,
        "false_confirmations_per_hour": 0.0,
        "face_px_buckets": {},
    }

    monkeypatch.setattr(cli, "load_manifest", lambda path: [case])
    monkeypatch.setattr(
        cli,
        "run_offline",
        lambda *args, **kwargs: {
            "threshold_results": {
                "0.6": {
                    "metrics": {
                        **common,
                        "detected_intervals": 0,
                        "interval_recall": 0.0,
                        "confirmation_latencies_seconds": [],
                    },
                    "shadow_metrics": {
                        **common,
                        "detected_intervals": 1,
                        "interval_recall": 1.0,
                        "confirmation_latencies_seconds": [0.5],
                    },
                }
            }
        },
    )

    cli.run_manifest(tmp_path / "manifest.json", tmp_path / "output", (0.6,))

    report = json.loads((tmp_path / "output" / "report.json").read_text())
    assert report["schema_version"] == 2
    assert report["recommended_similarity_threshold"] is None
    assert not report["aggregate"]["0.6"]["passed"]
    assert report["shadow_aggregate"]["0.6"]["passed"]
