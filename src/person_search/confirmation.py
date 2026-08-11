from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .domain import FaceObservation, MatchState, Target, Track


@dataclass(frozen=True, slots=True)
class MatchDecision:
    state: MatchState
    track_id: int
    bbox: np.ndarray
    similarity: float
    quality: float
    evidence_count: int


@dataclass(slots=True)
class _Evidence:
    frame_id: int
    timestamp: float
    similarity: float
    quality: float


@dataclass(slots=True)
class _TrackState:
    evidence: deque[_Evidence] = field(default_factory=deque)
    confirmed: bool = False
    last_track_seen: float = 0.0
    last_face_seen: float = 0.0
    last_candidate_emit: float = -1e9
    last_bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_similarity: float = -1.0
    last_quality: float = 0.0


class TrackConfirmation:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._states: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._states.clear()

    def process(
        self,
        *,
        frame_id: int,
        timestamp: float,
        frame_shape: tuple[int, ...],
        tracks: list[Track],
        faces: list[FaceObservation],
        target: Target,
    ) -> list[MatchDecision]:
        decisions: list[MatchDecision] = []
        tracks_by_id = {track.track_id: track for track in tracks}
        for track in tracks:
            state = self._states.setdefault(track.track_id, _TrackState())
            state.last_track_seen = timestamp
            state.last_bbox = track.bbox.copy()

        associations = associate_faces_to_tracks(faces, tracks)
        # At most one face contributes to a track in a frame: keep the highest-quality face.
        best_by_track: dict[int, FaceObservation] = {}
        for face_index, track_id in associations.items():
            face = faces[face_index]
            if not face.accepted:
                continue
            previous = best_by_track.get(track_id)
            if previous is None or face.quality > previous.quality:
                best_by_track[track_id] = face

        for track_id, face in best_by_track.items():
            state = self._states[track_id]
            similarity = float(np.dot(target.embedding, face.embedding))
            state.last_similarity = similarity
            state.last_quality = face.quality
            if similarity < self.settings.similarity_threshold:
                continue
            state.last_face_seen = timestamp
            self._expire_evidence(state, timestamp)
            duplicate_frame = any(item.frame_id == frame_id for item in state.evidence)
            separated = not state.evidence or timestamp - state.evidence[-1].timestamp >= 0.2
            if not duplicate_frame and separated:
                state.evidence.append(
                    _Evidence(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        similarity=similarity,
                        quality=face.quality,
                    )
                )

            if not state.confirmed:
                if timestamp - state.last_candidate_emit >= self.settings.candidate_emit_interval_seconds:
                    decisions.append(
                        self._decision(MatchState.CANDIDATE, track_id, state)
                    )
                    state.last_candidate_emit = timestamp
                if self._is_confirmed(state):
                    state.confirmed = True
                    decisions.append(
                        self._decision(MatchState.CONFIRMED, track_id, state)
                    )

        grace = self.settings.confirmed_track_grace_seconds
        for track_id, state in list(self._states.items()):
            self._expire_evidence(state, timestamp)
            if state.confirmed:
                track_missing = track_id not in tracks_by_id
                track_expired = track_missing and timestamp - state.last_track_seen >= grace
                face_expired = timestamp - state.last_face_seen >= grace
                if track_expired or face_expired:
                    decisions.append(self._decision(MatchState.LOST, track_id, state))
                    del self._states[track_id]
            elif (
                not state.evidence
                and track_id not in tracks_by_id
                and timestamp - state.last_track_seen >= grace
            ):
                del self._states[track_id]

        return decisions

    def _expire_evidence(self, state: _TrackState, timestamp: float) -> None:
        while (
            state.evidence
            and timestamp - state.evidence[0].timestamp > self.settings.evidence_window_seconds
        ):
            state.evidence.popleft()

    def _is_confirmed(self, state: _TrackState) -> bool:
        if len(state.evidence) < self.settings.evidence_required:
            return False
        similarities = [item.similarity for item in state.evidence]
        return float(np.median(similarities)) >= self.settings.similarity_threshold

    @staticmethod
    def _decision(state_name: MatchState, track_id: int, state: _TrackState) -> MatchDecision:
        return MatchDecision(
            state=state_name,
            track_id=track_id,
            bbox=state.last_bbox.copy(),
            similarity=state.last_similarity,
            quality=state.last_quality,
            evidence_count=len(state.evidence),
        )


def associate_faces_to_tracks(
    faces: list[FaceObservation], tracks: list[Track]
) -> dict[int, int]:
    associations: dict[int, int] = {}
    for face_index, face in enumerate(faces):
        center_x = float((face.bbox[0] + face.bbox[2]) / 2.0)
        center_y = float((face.bbox[1] + face.bbox[3]) / 2.0)
        candidates: list[tuple[float, int]] = []
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            upper_limit = y1 + 0.6 * (y2 - y1)
            if x1 <= center_x <= x2 and y1 <= center_y <= upper_limit:
                area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                candidates.append((area, track.track_id))
        if candidates:
            # The smallest containing person box is the least ambiguous assignment.
            associations[face_index] = min(candidates)[1]
    return associations


def normalize_bbox(bbox: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    return (
        float(np.clip(x1 / width, 0.0, 1.0)),
        float(np.clip(y1 / height, 0.0, 1.0)),
        float(np.clip(x2 / width, 0.0, 1.0)),
        float(np.clip(y2 / height, 0.0, 1.0)),
    )
