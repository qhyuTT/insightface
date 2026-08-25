from __future__ import annotations

import numpy as np
import pytest

from person_search.domain import Detection
from person_search.tracker import ByteTracker


def detection(box: tuple[int, int, int, int], score: float) -> Detection:
    return Detection(np.asarray(box, dtype=np.float32), score)


def test_low_score_detection_keeps_existing_track() -> None:
    tracker = ByteTracker(track_buffer=2)
    first = tracker.update([detection((0, 0, 100, 200), 0.9)])
    second = tracker.update([detection((2, 0, 102, 200), 0.3)])
    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_low_score_detection_does_not_create_new_track() -> None:
    tracker = ByteTracker()
    assert tracker.update([detection((0, 0, 100, 200), 0.3)]) == []


def _id_owning(tracks: list, box: tuple[int, int, int, int]) -> int | None:
    """Return the id of the track that actually took this detection."""
    wanted = np.asarray(box, dtype=np.float32)
    for track in tracks:
        if np.allclose(track.bbox, wanted):
            return track.track_id
    return None


def test_camera_motion_keeps_one_person_on_one_track_id() -> None:
    """A panning robot displaces every box at once; IoU alone reads that as a loss.

    The person stands still in the world and moves 90px per frame in the image,
    which is more than a 100px-wide box can overlap through.
    """
    compensated = ByteTracker()
    naive = ByteTracker()
    boxes = [(0, 0, 100, 200), (90, 0, 190, 200), (180, 0, 280, 200)]
    motion = np.asarray([90.0, 0.0], dtype=np.float32)

    compensated_ids = []
    naive_ids = []
    for index, box in enumerate(boxes):
        compensated_ids.append(
            _id_owning(
                compensated.update([detection(box, 0.9)], motion=None if index == 0 else motion),
                box,
            )
        )
        naive_ids.append(_id_owning(naive.update([detection(box, 0.9)]), box))

    assert len(set(compensated_ids)) == 1
    # Without compensation the same footage hands the person a new id on the second
    # frame, and a new id restarts the evidence window from zero.
    assert naive_ids[0] != naive_ids[1]


def test_velocity_is_measured_between_observations_not_against_the_prediction() -> None:
    """Measuring against the already-advanced box yields a residual that decays to 0."""
    tracker = ByteTracker()
    tracker.update([detection((0, 0, 100, 200), 0.9)])
    tracker.update([detection((10, 0, 110, 200), 0.9)])
    memory = next(iter(tracker._tracks.values()))

    assert memory.velocity[0] == pytest.approx(10.0)

    # Coasting for two frames then re-acquiring must not inflate the velocity by
    # the number of frames the track spent unmatched.
    tracker.update([])
    tracker.update([])
    tracker.update([detection((60, 0, 160, 200), 0.9)])
    memory = next(iter(tracker._tracks.values()))

    assert memory.velocity[0] == pytest.approx((60.0 - 10.0) / 3.0)
