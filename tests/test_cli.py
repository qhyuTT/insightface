from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from conftest import FakeFaceBackend, FakePersonDetector, make_face

from person_search import cli
from person_search.cli import (
    _associate_search_faces,
    _confirmation_input_counts,
    _face_policies,
    _flush_offline_confirmations,
    _is_matchable_face,
    _offline_decision_state,
    _safe_similarity,
    _summarize_track_outcomes,
)
from person_search.config import Settings
from person_search.confirmation import (
    GATE_INSUFFICIENT_SAMPLES,
    TrackConfirmation,
    tiny_face_match_policy,
)
from person_search.domain import Detection, MatchState, Target, TargetView, Track
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


def test_offline_tiny_face_keeps_a_relaxed_association_but_never_a_face_only_track() -> None:
    """Seated and truncated far faces are unambiguous; face-only tracks are not."""
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

    assert associations == {0: 7}
    assert modes == {0: "person_relaxed"}
    # No fallback face track was created for the unassociated far face.
    assert [track.track_id for track in all_tracks] == [7]
    # The relaxed path must not swap the far tier's policy for the looser
    # small-face one: it is weaker body evidence, not a bigger face.
    policies = _face_policies([relaxed, unassociated], modes, settings)
    assert policies[0] == tiny_face_match_policy(settings)


def test_offline_tiny_face_can_still_be_held_to_strict_person_association() -> None:
    settings = Settings(tiny_face_enabled=True, tiny_face_allow_relaxed_association=False)
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


def test_confirmation_skips_missing_or_malformed_face_embeddings() -> None:
    settings = Settings(evidence_required=1)
    target_view = TargetView(
        target_id="target-1",
        face_width=80,
        face_height=80,
        detection_score=0.99,
        quality_score=0.9,
        model="fake",
    )
    target = Target(
        "target-1",
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
    )
    track = Track(
        track_id=7,
        bbox=np.asarray([0, 0, 100, 160], dtype=np.float32),
        score=0.9,
    )
    missing = make_face()
    missing.embedding = None
    malformed = make_face()
    malformed.embedding = np.asarray([1.0, np.nan], dtype=np.float32)

    confirmation = TrackConfirmation(settings)
    result = confirmation.process_with_stats(
        frame_id=0,
        timestamp=0.0,
        frame_shape=(160, 100, 3),
        tracks=[track],
        faces=[missing, malformed],
        target=target,
        associations={0: 7, 1: 7},
        association_modes={0: "person_strict", 1: "person_strict"},
    )

    assert result.decisions == []
    assert result.evidence_collected == 0
    assert confirmation.track_progress(target) == {}


def test_confirmation_uses_valid_face_when_higher_quality_peer_is_malformed() -> None:
    settings = Settings(evidence_required=1)
    target_view = TargetView(
        target_id="target-1",
        face_width=80,
        face_height=80,
        detection_score=0.99,
        quality_score=0.9,
        model="fake",
    )
    target = Target(
        "target-1", np.asarray([1.0, 0.0], dtype=np.float32), target_view
    )
    track = Track(
        track_id=7,
        bbox=np.asarray([0, 0, 100, 160], dtype=np.float32),
        score=0.9,
    )
    malformed = make_face(quality=1.0)
    malformed.embedding = np.asarray([1.0, np.nan], dtype=np.float32)
    valid = make_face(embedding=(1.0, 0.0), quality=0.5)

    result = TrackConfirmation(settings).process_with_stats(
        frame_id=0,
        timestamp=0.0,
        frame_shape=(160, 100, 3),
        tracks=[track],
        faces=[malformed, valid],
        target=target,
        associations={0: 7, 1: 7},
        association_modes={0: "person_strict", 1: "person_strict"},
    )

    assert result.evidence_collected == 1
    assert [decision.state for decision in result.decisions] == [MatchState.CANDIDATE, MatchState.CONFIRMED]


def test_offline_policy_hysteresis_uses_the_associated_track_tier() -> None:
    settings = Settings(tiny_face_enabled=True, face_tier_hysteresis_px=4)
    # 65px normally enters the small tier. A track already held in the tiny tier
    # must cross 68px before changing tier, exactly as the live session does.
    face = make_face(bbox=(0, 0, 65, 65))

    policies = _face_policies(
        [face],
        {0: "person_strict"},
        settings,
        track_tiers={7: "tiny"},
        associations={0: 7},
    )

    assert policies[0].tier == "tiny"
    assert _face_policies([face], {0: "person_strict"}, settings)[0].tier == "small"


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


def test_offline_stage_counts_skip_faces_without_embeddings() -> None:
    settings = Settings()
    face = make_face(bbox=(0, 0, 80, 80))
    face.embedding = None
    policies = _face_policies([face], {0: "person_strict"}, settings)

    counts = _confirmation_input_counts(
        [face],
        {0: 7},
        policies,
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert counts == {"above_threshold": 0, "evidence_eligible": 0}


def test_offline_similarity_helpers_skip_malformed_embeddings() -> None:
    settings = Settings()
    malformed = make_face()
    malformed.embedding = np.asarray([1.0, np.nan], dtype=np.float32)
    policies = _face_policies([malformed], {0: "person_strict"}, settings)

    assert _safe_similarity(np.asarray([1.0, 0.0], dtype=np.float32), None) is None
    assert _safe_similarity(np.asarray([1.0, 0.0], dtype=np.float32), np.asarray([1.0])) is None
    assert _safe_similarity(np.asarray([1.0, 0.0], dtype=np.float32), malformed.embedding) is None
    assert _confirmation_input_counts(
        [malformed], {0: 7}, policies, np.asarray([1.0, 0.0], dtype=np.float32)
    ) == {"above_threshold": 0, "evidence_eligible": 0}


def test_cli_similarity_helper_is_defined_before_module_entrypoint() -> None:
    source = (Path(cli.__file__)).read_text()
    assert source.index("def _similarity_sample(") < source.index('if __name__ == "__main__":')


def test_offline_eof_flush_reports_an_unfinished_track() -> None:
    settings = Settings(
        similarity_threshold=0.60,
        evidence_required=3,
        evidence_window_seconds=1.5,
        confirmed_track_grace_seconds=2.0,
    )
    confirmation = TrackConfirmation(settings)
    target_view = TargetView(
        target_id="target-1",
        face_width=80,
        face_height=80,
        detection_score=0.99,
        quality_score=0.9,
        model="fake",
    )
    target = Target(
        "target-1",
        np.asarray([1.0, 0.0], dtype=np.float32),
        target_view,
    )
    track = Track(
        track_id=7,
        bbox=np.asarray([0, 0, 100, 160], dtype=np.float32),
        score=0.9,
    )
    for frame_id in range(2):
        confirmation.process_with_stats(
            frame_id=frame_id,
            timestamp=frame_id * 0.25,
            frame_shape=(160, 100, 3),
            tracks=[track],
            faces=[make_face(bbox=(10, 10, 90, 90))],
            target=target,
        )

    results = _flush_offline_confirmations(
        {"0.6": confirmation},
        target=target,
        frame_id=2,
        fps=4.0,
        frame_shape=(160, 100, 3),
        settings=settings,
    )

    result = results["0.6"]
    assert result.decisions == []
    assert len(result.outcomes) == 1
    assert result.outcomes[0].blocking_gate == GATE_INSUFFICIENT_SAMPLES
    assert _summarize_track_outcomes(result.outcomes)["tracks"] == 1
    assert confirmation.track_progress(target) == {}


def test_offline_embedding_uses_live_budget_and_microbatch_path(monkeypatch, tmp_path) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    enrollment_face = make_face(bbox=(20, 20, 120, 120))
    tracked_face = make_face(bbox=(30, 20, 130, 120))
    unassociated_face = make_face(bbox=(200, 20, 300, 120))

    class Backend(FakeFaceBackend):
        def __init__(self) -> None:
            super().__init__([enrollment_face, tracked_face, unassociated_face])
            self.batch_inputs: list[int] = []

        def detect_faces(self, frame, *, enrollment=False, detection_size=None):
            self.detect_calls += 1
            if enrollment:
                return [enrollment_face]
            return [
                type(face)(
                    bbox=face.bbox.copy(),
                    detection_score=face.detection_score,
                    embedding=None,
                    quality=face.quality,
                    landmarks=face.landmarks,
                    accepted=face.accepted,
                    rejection_reasons=face.rejection_reasons,
                    blur_variance=face.blur_variance,
                )
                for face in (tracked_face, unassociated_face)
            ]

        def embed_faces(self, frame, faces):
            self.batch_inputs.append(len(faces))
            return [
                type(face)(
                    bbox=face.bbox.copy(),
                    detection_score=face.detection_score,
                    embedding=(
                        tracked_face.embedding
                        if tuple(face.bbox.tolist()) == tuple(tracked_face.bbox.tolist())
                        else unassociated_face.embedding
                    ),
                    quality=face.quality,
                    landmarks=face.landmarks,
                    accepted=face.accepted,
                    rejection_reasons=face.rejection_reasons,
                    blur_variance=face.blur_variance,
                )
                for face in faces
            ]

    class Capture:
        def __init__(self, _path):
            self.frames = [frame.copy()]

        def isOpened(self):
            return True

        def get(self, prop):
            import cv2

            return {
                cv2.CAP_PROP_FPS: 4.0,
                cv2.CAP_PROP_FRAME_WIDTH: frame.shape[1],
                cv2.CAP_PROP_FRAME_HEIGHT: frame.shape[0],
            }.get(prop, 0.0)

        def read(self):
            return (True, self.frames.pop(0)) if self.frames else (False, None)

        def release(self):
            return None

    class Writer:
        def __init__(self, *_args):
            pass

        def isOpened(self):
            return True

        def write(self, _frame):
            return None

        def release(self):
            return None

    backend = Backend()
    detector = FakePersonDetector(
        [
            Detection(
                bbox=np.asarray([0, 0, 160, 220], dtype=np.float32),
                score=0.9,
            )
        ]
    )
    settings = Settings(
        face_detection_hz_cpu=4.0,
        person_detection_hz_cpu=4.0,
        roi_face_detection_hz_cpu=0.0,
        max_faces_per_frame=1,
        arcface_micro_batch_size=1,
        evidence_required=2,
        evidence_window_seconds=1.0,
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "InsightFaceBackend", lambda _settings: backend)
    monkeypatch.setattr(cli, "YoloXOnnxDetector", lambda _settings: detector)
    monkeypatch.setattr(cli.cv2, "imread", lambda _path: frame.copy())
    monkeypatch.setattr(cli.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(cli.cv2, "VideoWriter", Writer)

    summary = cli.run_offline(
        Path("target.jpg"),
        Path("clip.mp4"),
        tmp_path,
        print_summary=False,
    )

    diagnostics = summary["quality_diagnostics"]
    assert diagnostics["embedding_candidates"] == 2
    assert diagnostics["faces_dropped_by_budget"] == 1
    assert diagnostics["embedding_batch_count"] == 1
    # The enrollment photo is embedded once before the live clip; the bounded
    # replay path contributes the second single-row call.
    assert backend.batch_inputs == [1, 1]


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
