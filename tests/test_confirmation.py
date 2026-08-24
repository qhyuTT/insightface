from __future__ import annotations

import numpy as np
import pytest
from conftest import make_face

from person_search.config import Settings
from person_search.confirmation import (
    TrackConfirmation,
    associate_faces_to_tracks,
    associate_faces_to_tracks_detailed,
    default_face_match_policy,
)
from person_search.domain import MatchState, Target, TargetView, Track


def _target() -> Target:
    view = TargetView(
        target_id="target-1",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake",
    )
    return Target("target-1", np.asarray([1.0, 0.0], dtype=np.float32), view)


def _track(track_id: int = 7) -> Track:
    return Track(track_id, np.asarray([0, 0, 100, 200], dtype=np.float32), 0.9)


def _face_with_similarity(short_side: int, similarity: float):
    return make_face(
        embedding=(similarity, float(np.sqrt(1.0 - similarity**2))),
        bbox=(10, 10, 10 + short_side, 10 + short_side),
    )


def test_three_separated_frames_confirm_then_face_timeout_loses() -> None:
    settings = Settings(
        similarity_threshold=0.8,
        evidence_required=3,
        evidence_window_seconds=1.5,
        confirmed_track_grace_seconds=2.0,
        candidate_emit_interval_seconds=0.5,
    )
    matcher = TrackConfirmation(settings)
    face = make_face()
    track = _track()

    states: list[MatchState] = []
    for frame_id, timestamp in enumerate((0.0, 0.2, 0.4)):
        decisions = matcher.process(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=(200, 100, 3),
            tracks=[track],
            faces=[face],
            target=_target(),
        )
        states.extend(item.state for item in decisions)
    assert states.count(MatchState.CONFIRMED) == 1
    assert states[-1] == MatchState.CONFIRMED

    before_timeout = matcher.process(
        frame_id=3,
        timestamp=2.39,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[],
        target=_target(),
    )
    assert not any(item.state == MatchState.LOST for item in before_timeout)
    at_timeout = matcher.process(
        frame_id=4,
        timestamp=2.4,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[],
        target=_target(),
    )
    assert [item.state for item in at_timeout] == [MatchState.LOST]


def test_duplicate_or_too_close_frames_do_not_accumulate() -> None:
    settings = Settings(similarity_threshold=0.8, evidence_required=2)
    matcher = TrackConfirmation(settings)
    track = _track()
    face = make_face()
    first = matcher.process(
        frame_id=1,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[face],
        target=_target(),
    )
    second = matcher.process(
        frame_id=1,
        timestamp=0.3,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[face],
        target=_target(),
    )
    assert first[0].evidence_count == 1
    assert not any(item.state == MatchState.CONFIRMED for item in second)


def test_face_associates_to_smallest_upper_body_box() -> None:
    face = make_face(bbox=(40, 20, 60, 50))
    large = Track(1, np.asarray([0, 0, 100, 200]), 0.9)
    small = Track(2, np.asarray([30, 10, 70, 100]), 0.9)
    assert associate_faces_to_tracks([face], [large, small]) == {0: 2}


def test_face_below_upper_body_is_not_associated() -> None:
    face = make_face(bbox=(40, 150, 60, 180))
    assert associate_faces_to_tracks([face], [_track()]) == {}


def test_strict_association_includes_the_upper_sixty_percent_boundary() -> None:
    at_boundary = make_face(bbox=(40, 110, 60, 130))
    below_boundary = make_face(bbox=(40, 111, 60, 131))

    assert associate_faces_to_tracks([at_boundary], [_track()]) == {0: 7}
    assert associate_faces_to_tracks([below_boundary], [_track()]) == {}


def test_seated_face_uses_relaxed_association_for_one_containing_track() -> None:
    face = make_face(bbox=(40, 150, 60, 170))

    assert associate_faces_to_tracks_detailed([face], [_track()]) == {
        0: (7, "person_relaxed")
    }


def test_relaxed_association_rejects_multiple_containing_tracks() -> None:
    face = make_face(bbox=(40, 150, 60, 170))
    tracks = [
        _track(track_id=7),
        Track(8, np.asarray([20, 0, 80, 220], dtype=np.float32), 0.8),
    ]

    assert associate_faces_to_tracks_detailed([face], tracks) == {}


@pytest.mark.parametrize("short_side", [64, 79])
def test_small_face_requires_four_frames_at_point_six_without_candidate(
    short_side: int,
) -> None:
    settings = Settings(
        similarity_threshold=0.55,
        evidence_required=3,
        small_face_similarity_threshold=0.60,
        small_face_evidence_required=4,
        small_face_evidence_window_seconds=2.0,
    )
    matcher = TrackConfirmation(settings)
    face = _face_with_similarity(short_side, 0.60)
    policy = default_face_match_policy(face, settings)

    assert policy.threshold == 0.60
    assert policy.evidence_required == 4
    assert policy.suppress_candidate

    decisions = []
    for frame_id, timestamp in enumerate((0.0, 0.25, 0.5, 0.75)):
        frame_decisions = matcher.process(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=(200, 100, 3),
            tracks=[_track()],
            faces=[face],
            target=_target(),
            face_policies={0: policy},
        )
        if frame_id < 3:
            assert not any(item.state == MatchState.CONFIRMED for item in frame_decisions)
            assert matcher.active_track_states() == {}
        decisions.extend(frame_decisions)

    assert not any(item.state == MatchState.CANDIDATE for item in decisions)
    confirmed = [item for item in decisions if item.state == MatchState.CONFIRMED]
    assert len(confirmed) == 1
    assert confirmed[0].evidence_count == 4


def test_small_face_rejects_similarity_below_point_six() -> None:
    settings = Settings(
        similarity_threshold=0.55,
        small_face_similarity_threshold=0.60,
        small_face_evidence_required=4,
    )
    matcher = TrackConfirmation(settings)
    face = _face_with_similarity(64, 0.59)
    policy = default_face_match_policy(face, settings)

    decisions = []
    for frame_id, timestamp in enumerate((0.0, 0.25, 0.5, 0.75)):
        decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[face],
                target=_target(),
                face_policies={0: policy},
            )
        )

    assert decisions == []
    assert matcher.active_track_states() == {}


def test_preferred_size_face_uses_normal_three_frame_policy() -> None:
    settings = Settings(
        similarity_threshold=0.55,
        evidence_required=3,
        small_face_similarity_threshold=0.60,
        small_face_evidence_required=4,
    )
    matcher = TrackConfirmation(settings)
    face = _face_with_similarity(80, 0.56)
    policy = default_face_match_policy(face, settings)

    assert policy.threshold == 0.55
    assert policy.evidence_required == 3
    assert not policy.suppress_candidate

    decisions = []
    for frame_id, timestamp in enumerate((0.0, 0.25, 0.5)):
        frame_decisions = matcher.process(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=(200, 100, 3),
            tracks=[_track()],
            faces=[face],
            target=_target(),
            face_policies={0: policy},
        )
        if frame_id < 2:
            assert not any(item.state == MatchState.CONFIRMED for item in frame_decisions)
        decisions.extend(frame_decisions)

    assert any(item.state == MatchState.CANDIDATE for item in decisions)
    confirmed = [item for item in decisions if item.state == MatchState.CONFIRMED]
    assert len(confirmed) == 1
    assert confirmed[0].evidence_count == 3


def test_candidate_state_disappears_when_evidence_expires() -> None:
    matcher = TrackConfirmation(
        Settings(similarity_threshold=0.8, evidence_required=3, evidence_window_seconds=1.0)
    )
    track = _track()
    matcher.process(
        frame_id=1,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[make_face()],
        target=_target(),
    )
    assert matcher.active_track_states()[track.track_id][0] == MatchState.CANDIDATE

    matcher.process(
        frame_id=2,
        timestamp=1.01,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[],
        target=_target(),
    )
    assert matcher.active_track_states() == {}
