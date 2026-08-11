from __future__ import annotations

import numpy as np

from person_search.domain import Detection, FaceObservation


class FakeFaceBackend:
    model_name = "fake-arcface"
    provider_name = "CPUExecutionProvider"

    def __init__(self, observations: list[FaceObservation] | None = None):
        self.observations = observations or []
        self.calls = 0

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        self.calls += 1
        return list(self.observations)


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
    )
