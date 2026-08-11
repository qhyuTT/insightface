from __future__ import annotations

import numpy as np

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
