from __future__ import annotations

from conftest import make_face

from person_search.face_tracking import FaceTracker


def test_same_face_keeps_negative_track_id_across_frames() -> None:
    tracker = FaceTracker(iou_threshold=0.25, buffer_seconds=1.0)

    first = tracker.update([make_face(bbox=(10, 10, 50, 50))], timestamp=0.0)[0]
    second = tracker.update([make_face(bbox=(12, 11, 52, 51))], timestamp=0.2)[0]

    assert first is not None
    assert second is not None
    assert first.track_id < 0
    assert second.track_id == first.track_id


def test_different_faces_receive_independent_track_ids() -> None:
    tracker = FaceTracker(iou_threshold=0.25, buffer_seconds=1.0)
    faces = [
        make_face(bbox=(10, 10, 50, 50)),
        make_face(bbox=(110, 10, 150, 50)),
    ]

    first = tracker.update(faces, timestamp=0.0)
    second = tracker.update(faces, timestamp=0.2)

    first_ids = [track.track_id for track in first if track is not None]
    second_ids = [track.track_id for track in second if track is not None]
    assert len(set(first_ids)) == 2
    assert all(track_id < 0 for track_id in first_ids)
    assert second_ids == first_ids


def test_expired_face_receives_a_new_track_id() -> None:
    tracker = FaceTracker(iou_threshold=0.25, buffer_seconds=1.0)
    face = make_face(bbox=(10, 10, 50, 50))

    first = tracker.update([face], timestamp=0.0)[0]
    after_expiry = tracker.update([face], timestamp=1.01)[0]

    assert first is not None
    assert after_expiry is not None
    assert after_expiry.track_id < 0
    assert after_expiry.track_id != first.track_id
