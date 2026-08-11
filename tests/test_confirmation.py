from __future__ import annotations

import numpy as np
from conftest import make_face

from person_search.config import Settings
from person_search.confirmation import TrackConfirmation, associate_faces_to_tracks
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
