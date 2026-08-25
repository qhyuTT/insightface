from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from person_search.domain import Detection, FaceObservation


class FakeFaceBackend:
    model_name = "fake-arcface"
    provider_name = "CPUExecutionProvider"
    detection_provider_name = "CPUExecutionProvider"
    recognition_provider_name = "CPUExecutionProvider"

    def __init__(self, observations: list[FaceObservation] | None = None):
        self.observations = observations or []
        self.calls = 0
        self.detect_calls = 0
        self.embed_calls = 0
        self.embedded_faces = 0
        self.detection_sizes: list[int | Sequence[int] | None] = []

    def detect_faces(
        self,
        frame: np.ndarray,
        *,
        enrollment: bool = False,
        detection_size: int | Sequence[int] | None = None,
    ) -> list[FaceObservation]:
        self.detect_calls += 1
        self.detection_sizes.append(detection_size)
        # Detections carry no embedding; the session must ask for one explicitly.
        return [replace(face, embedding=None) for face in self.observations]

    def embed_faces(
        self, frame: np.ndarray, faces: list[FaceObservation]
    ) -> list[FaceObservation]:
        self.embed_calls += 1
        self.embedded_faces += len(faces)
        by_bbox = {tuple(face.bbox.tolist()): face.embedding for face in self.observations}
        embedded = []
        for face in faces:
            embedding = by_bbox.get(tuple(face.bbox.tolist()))
            if embedding is None:
                continue
            embedded.append(replace(face, embedding=embedding))
        return embedded

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        self.calls += 1
        return self.embed_faces(frame, self.detect_faces(frame, enrollment=enrollment))


class FakePersonDetector:
    provider_name = "CPUExecutionProvider"

    def __init__(self, detections: list[Detection] | None = None):
        self.detections = detections or []
        self.ready = False

    def ensure_ready(self) -> None:
        self.ready = True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self.detections)


def make_face(
    embedding: tuple[float, ...] = (1.0, 0.0),
    bbox: tuple[float, float, float, float] = (20, 20, 80, 80),
    *,
    accepted: bool = True,
    quality: float = 0.9,
    blur_variance: float = 0.0,
) -> FaceObservation:
    vector = np.asarray(embedding, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return FaceObservation(
        bbox=np.asarray(bbox, dtype=np.float32),
        detection_score=0.99,
        embedding=vector,
        quality=quality,
        accepted=accepted,
        rejection_reasons=() if accepted else ("face_blurry",),
        blur_variance=blur_variance,
    )
