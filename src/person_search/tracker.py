from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import Detection, Track


@dataclass(slots=True)
class _TrackMemory:
    track_id: int
    bbox: np.ndarray
    score: float
    velocity: np.ndarray
    missed: int = 0


class ByteTracker:
    """Small runtime implementation of ByteTrack's high/low score association idea."""

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        first_iou_threshold: float = 0.3,
        second_iou_threshold: float = 0.2,
        track_buffer: int = 30,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.first_iou_threshold = first_iou_threshold
        self.second_iou_threshold = second_iou_threshold
        self.track_buffer = track_buffer
        self._tracks: dict[int, _TrackMemory] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Drop all track state and restart track IDs from one.

        A reader can be reconnected without constructing a new tracker.  The
        explicit reset hook is also useful when a stream seek/discontinuity is
        detected: predicted boxes from the old timeline must not be associated
        with frames from the new one.
        """

        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        memories = list(self._tracks.values())
        for memory in memories:
            # Prediction is an in-place add; allocating a new four-element
            # array for every live track on every update is surprisingly costly
            # at the frame rates used by edge cameras.
            np.add(memory.bbox, memory.velocity, out=memory.bbox)
            memory.missed += 1

        high = [item for item in detections if item.score >= self.high_threshold]
        low = [
            item
            for item in detections
            if self.low_threshold <= item.score < self.high_threshold
        ]
        matched_track_ids: set[int] = set()
        matched_high: set[int] = set()

        first_matches = _associate(memories, high, self.first_iou_threshold)
        for track_index, detection_index in first_matches:
            self._apply_match(memories[track_index], high[detection_index])
            matched_track_ids.add(memories[track_index].track_id)
            matched_high.add(detection_index)

        remaining = [item for item in memories if item.track_id not in matched_track_ids]
        for track_index, detection_index in _associate(remaining, low, self.second_iou_threshold):
            self._apply_match(remaining[track_index], low[detection_index])
            matched_track_ids.add(remaining[track_index].track_id)

        for index, detection in enumerate(high):
            if index in matched_high:
                continue
            memory = _TrackMemory(
                track_id=self._next_id,
                bbox=np.array(detection.bbox, dtype=np.float32, copy=True),
                score=detection.score,
                velocity=np.zeros(4, dtype=np.float32),
            )
            self._tracks[memory.track_id] = memory
            matched_track_ids.add(memory.track_id)
            self._next_id += 1

        expired = [track_id for track_id, item in self._tracks.items() if item.missed > self.track_buffer]
        for track_id in expired:
            del self._tracks[track_id]

        return [
            Track(track_id=item.track_id, bbox=item.bbox.copy(), score=item.score)
            for item in self._tracks.values()
            if item.missed <= self.track_buffer
        ]

    @staticmethod
    def _apply_match(memory: _TrackMemory, detection: Detection) -> None:
        # Detection boxes are normally float32 already.  Reuse the velocity
        # buffer and copy the coordinates into the track-owned box so callers
        # can safely reuse/mutate their Detection objects.
        if memory.velocity.shape != detection.bbox.shape:
            memory.velocity = np.empty_like(memory.bbox, dtype=np.float32)
        np.subtract(
            detection.bbox,
            memory.bbox,
            out=memory.velocity,
            dtype=np.float32,
            casting="unsafe",
        )
        np.copyto(memory.bbox, detection.bbox, casting="unsafe")
        memory.score = detection.score
        memory.missed = 0


def _associate(
    tracks: list[_TrackMemory], detections: list[Detection], threshold: float
) -> list[tuple[int, int]]:
    if not tracks or not detections:
        return []

    # Compute all pairwise IoUs with broadcasting.  The old nested Python loops
    # became a measurable bottleneck once several people were visible in a
    # 1080p stream; the resulting matrix is tiny compared with detector work
    # and is much cheaper to build in NumPy.
    track_boxes = np.stack([track.bbox for track in tracks]).astype(np.float32, copy=False)
    detection_boxes = np.stack([item.bbox for item in detections]).astype(
        np.float32, copy=False
    )
    top_left = np.maximum(track_boxes[:, None, :2], detection_boxes[None, :, :2])
    bottom_right = np.minimum(track_boxes[:, None, 2:], detection_boxes[None, :, 2:])
    wh = np.maximum(0.0, bottom_right - top_left)
    intersection = wh[..., 0] * wh[..., 1]
    track_wh = np.maximum(0.0, track_boxes[:, 2:] - track_boxes[:, :2])
    detection_wh = np.maximum(0.0, detection_boxes[:, 2:] - detection_boxes[:, :2])
    track_area = track_wh[:, 0] * track_wh[:, 1]
    detection_area = detection_wh[:, 0] * detection_wh[:, 1]
    ious = intersection / np.maximum(
        track_area[:, None] + detection_area[None, :] - intersection, 1e-6
    )

    # Only valid edges should participate in assignment.  Assigning all edges
    # first and filtering by threshold afterwards can discard a valid match
    # when an invalid, higher-cost edge was selected by Hungarian assignment.
    valid = ious >= threshold
    try:
        from scipy.optimize import linear_sum_assignment

        # Keep invalid edges prohibitively expensive.  The subsequent mask
        # still protects against the rectangular-matrix edge case where the
        # solver has to return an invalid pair.
        cost = 1.0 - ious
        cost = np.where(valid, cost, 1e6)
        rows, columns = linear_sum_assignment(cost)
        pairs = zip(rows.tolist(), columns.tolist(), strict=True)
    except ImportError:
        # Deterministic greedy fallback for installations without SciPy.  Sort
        # only candidate edges, avoiding a copy/mutation of the full IoU matrix.
        candidates = np.argwhere(valid)
        if candidates.size == 0:
            return []
        candidate_scores = ious[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(-candidate_scores, kind="stable")
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        pairs_list: list[tuple[int, int]] = []
        for candidate_index in order.tolist():
            row, column = (int(value) for value in candidates[candidate_index])
            if row in used_rows or column in used_columns:
                continue
            used_rows.add(row)
            used_columns.add(column)
            pairs_list.append((row, column))
        return pairs_list
    return [(row, column) for row, column in pairs if valid[row, column]]


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return float(intersection / max(area_a + area_b - intersection, 1e-6))
