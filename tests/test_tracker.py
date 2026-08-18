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


def test_reset_drops_tracks_and_restarts_ids() -> None:
    tracker = ByteTracker()
    assert tracker.update([detection((0, 0, 100, 200), 0.9)])[0].track_id == 1
    tracker.reset()
    tracks = tracker.update([detection((10, 10, 110, 210), 0.9)])
    assert [item.track_id for item in tracks] == [1]


def test_association_does_not_let_invalid_edge_block_valid_match() -> None:
    # The first track has one valid candidate; the second candidate is below
    # threshold.  Assignment must retain the valid edge even when the invalid
    # edge would otherwise be considered by a global Hungarian solve.
    tracker = ByteTracker(first_iou_threshold=0.5, track_buffer=2)
    first = tracker.update(
        [
            detection((0, 0, 100, 100), 0.9),
            detection((200, 0, 300, 100), 0.9),
        ]
    )
    second = tracker.update(
        [
            detection((0, 0, 100, 100), 0.9),
            detection((102, 0, 202, 100), 0.3),
        ]
    )
    assert {item.track_id for item in second} == {item.track_id for item in first}
