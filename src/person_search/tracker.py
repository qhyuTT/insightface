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
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        memories = list(self._tracks.values())
        for memory in memories:
            memory.bbox = memory.bbox + memory.velocity
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
                bbox=detection.bbox.copy(),
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
        memory.velocity = detection.bbox.astype(np.float32) - memory.bbox.astype(np.float32)
        memory.bbox = detection.bbox.copy()
        memory.score = detection.score
        memory.missed = 0


def _associate(
    tracks: list[_TrackMemory], detections: list[Detection], threshold: float
) -> list[tuple[int, int]]:
    if not tracks or not detections:
        return []
    ious = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            ious[track_index, detection_index] = _iou(track.bbox, detection.bbox)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(1.0 - ious)
        pairs = zip(rows.tolist(), columns.tolist(), strict=True)
    except ImportError:
        pairs_list: list[tuple[int, int]] = []
        work = ious.copy()
        while work.size and float(work.max()) >= threshold:
            row, column = np.unravel_index(int(work.argmax()), work.shape)
            pairs_list.append((int(row), int(column)))
            work[row, :] = -1
            work[:, column] = -1
        pairs = pairs_list
    return [(row, column) for row, column in pairs if ious[row, column] >= threshold]


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return float(intersection / max(area_a + area_b - intersection, 1e-6))
