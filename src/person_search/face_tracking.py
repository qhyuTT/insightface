from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import FaceObservation, Track


@dataclass(slots=True)
class _FaceTrackMemory:
    track_id: int
    bbox: np.ndarray
    last_seen: float


class FaceTracker:
    """Small IoU tracker used only when no unambiguous person track owns a face."""

    def __init__(self, iou_threshold: float = 0.25, buffer_seconds: float = 1.0):
        self.iou_threshold = iou_threshold
        self.buffer_seconds = buffer_seconds
        self._tracks: dict[int, _FaceTrackMemory] = {}
        self._next_id = -1

    def update(self, faces: list[FaceObservation], timestamp: float) -> list[Track | None]:
        self._expire(timestamp)
        remaining = set(range(len(faces)))
        assignments: dict[int, int] = {}
        by_face: dict[int, Track] = {}
        pairs: list[tuple[float, int, int]] = []
        for track_id, memory in self._tracks.items():
            for face_index, face in enumerate(faces):
                pairs.append((_iou(memory.bbox, face.bbox), track_id, face_index))
        for iou, track_id, face_index in sorted(pairs, reverse=True):
            if iou < self.iou_threshold or face_index not in remaining or track_id in assignments:
                continue
            assignments[track_id] = face_index
            remaining.remove(face_index)
        for track_id, face_index in assignments.items():
            memory = self._tracks[track_id]
            memory.bbox = faces[face_index].bbox.copy()
            memory.last_seen = timestamp
            by_face[face_index] = Track(
                track_id=track_id, bbox=memory.bbox.copy(), score=1.0
            )
        for face_index in remaining:
            face = faces[face_index]
            track_id = self._next_id
            self._tracks[track_id] = _FaceTrackMemory(
                track_id=self._next_id,
                bbox=face.bbox.copy(),
                last_seen=timestamp,
            )
            by_face[face_index] = Track(
                track_id=track_id, bbox=face.bbox.copy(), score=1.0
            )
            self._next_id -= 1
        return [by_face.get(face_index) for face_index in range(len(faces))]

    def _expire(self, timestamp: float) -> None:
        expired = [
            track_id
            for track_id, memory in self._tracks.items()
            if timestamp - memory.last_seen > self.buffer_seconds
        ]
        for track_id in expired:
            del self._tracks[track_id]


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return float(intersection / max(area_first + area_second - intersection, 1e-6))
