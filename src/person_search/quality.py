from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np

from .config import Settings


@dataclass(frozen=True, slots=True)
class QualityResult:
    accepted: bool
    score: float
    reasons: tuple[str, ...]
    face_width: int
    face_height: int
    blur_variance: float
    brightness: float
    roll_degrees: float | None
    yaw_proxy: float | None


def normalize_embedding(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("embedding must be real-valued")
    embedding = np.asarray(raw, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.isfinite(embedding).all():
        raise ValueError("embedding must contain only finite values")
    magnitude = float(np.linalg.norm(embedding.astype(np.float64, copy=False)))
    if not np.isfinite(magnitude) or magnitude <= 1e-12:
        raise ValueError("embedding magnitude is zero")
    normalized = embedding / magnitude
    if not np.isfinite(normalized).all() or not np.any(normalized):
        raise ValueError("embedding normalization failed")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def assess_face(
    frame: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray | None,
    detection_score: float,
    settings: Settings,
    *,
    enrollment: bool,
) -> QualityResult:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(width, int(x2)), min(height, int(y2))
    face_width, face_height = max(0, x2i - x1i), max(0, y2i - y1i)
    crop = frame[y1i:y2i, x1i:x2i]

    reasons: list[str] = []
    minimum_size = (
        settings.min_enrollment_face_px if enrollment else settings.effective_search_min_face_px
    )
    minimum_blur = (
        settings.min_enrollment_blur_variance if enrollment else settings.min_search_blur_variance
    )
    if min(face_width, face_height) < minimum_size:
        reasons.append("face_too_small")
    minimum_detection_score = (
        max(settings.face_detection_threshold, settings.min_enrollment_detection_score)
        if enrollment
        else (
            max(settings.face_detection_threshold, settings.tiny_face_detection_threshold)
            if settings.tiny_face_enabled
            and min(face_width, face_height) < settings.min_search_face_px
            else settings.face_detection_threshold
        )
    )
    if detection_score < minimum_detection_score:
        reasons.append("detection_score_low")

    blur = 0.0
    brightness = 0.0
    if crop.size == 0:
        reasons.append("invalid_face_crop")
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        if blur < minimum_blur:
            reasons.append("face_blurry")
        if brightness < settings.min_brightness or brightness > settings.max_brightness:
            reasons.append("face_exposure_bad")

    roll, yaw = _pose_proxies(landmarks)
    if enrollment:
        if roll is not None and abs(roll) > settings.max_abs_roll_degrees:
            reasons.append("face_roll_too_large")
        if yaw is not None and abs(yaw) > settings.max_yaw_proxy:
            reasons.append("face_yaw_too_large")

    size_score = min(1.0, min(face_width, face_height) / max(minimum_size * 1.5, 1))
    blur_score = min(1.0, blur / max(minimum_blur * 2.0, 1.0))
    exposure_score = max(0.0, 1.0 - abs(brightness - 130.0) / 130.0)
    pose_score = 1.0
    if roll is not None:
        pose_score *= max(0.0, 1.0 - abs(roll) / 45.0)
    if yaw is not None:
        pose_score *= max(0.0, 1.0 - abs(yaw))
    score = float(
        np.clip(
            0.25 * detection_score
            + 0.25 * size_score
            + 0.27 * blur_score
            + 0.15 * exposure_score
            + 0.08 * pose_score,
            0.0,
            1.0,
        )
    )
    return QualityResult(
        accepted=not reasons,
        score=score,
        reasons=tuple(reasons),
        face_width=face_width,
        face_height=face_height,
        blur_variance=blur,
        brightness=brightness,
        roll_degrees=roll,
        yaw_proxy=yaw,
    )


def _pose_proxies(landmarks: np.ndarray | None) -> tuple[float | None, float | None]:
    if landmarks is None:
        return None, None
    points = np.asarray(landmarks, dtype=float)
    if points.shape[0] < 3 or points.shape[1] < 2:
        return None, None
    left_eye, right_eye, nose = points[0], points[1], points[2]
    eye_dx = float(right_eye[0] - left_eye[0])
    eye_dy = float(right_eye[1] - left_eye[1])
    eye_distance = max(float(np.hypot(eye_dx, eye_dy)), 1e-6)
    roll = degrees(atan2(eye_dy, eye_dx))
    eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
    yaw = float((nose[0] - eye_mid_x) / (eye_distance / 2.0))
    return roll, yaw
