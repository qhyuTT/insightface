from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .backends import InsightFaceBackend
from .config import Settings
from .confirmation import TrackConfirmation, normalize_bbox
from .detector import YoloXOnnxDetector
from .domain import MatchState
from .service import SearchManager
from .tracker import ByteTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run person search on a prerecorded video")
    parser.add_argument("--photo", type=Path, required=True, help="single-face target photo")
    parser.add_argument("--name", default=None, help="display name for the target")
    parser.add_argument("--video", type=Path, required=True, help="input video")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    run_offline(args.photo, args.video, args.output_dir, args.threshold, args.name)


def run_offline(
    photo_path: Path,
    video_path: Path,
    output_dir: Path,
    threshold: float | None = None,
    name: str | None = None,
) -> None:
    settings = Settings()
    if threshold is not None:
        settings.similarity_threshold = threshold
    output_dir.mkdir(parents=True, exist_ok=True)
    photo = cv2.imread(str(photo_path))
    if photo is None:
        raise SystemExit(f"cannot read target photo: {photo_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")

    face_backend = InsightFaceBackend(settings)
    detector = YoloXOnnxDetector(settings)
    manager = SearchManager(settings, face_backend=face_backend, person_detector=detector)
    target_view = manager.enroll(photo, name or photo_path.stem)
    target = manager.get_target(target_view.target_id)
    detector.ensure_ready()

    fps = capture.get(cv2.CAP_PROP_FPS) or settings.input_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    tracker = ByteTracker()
    confirmation = TrackConfirmation(settings)
    events: list[dict[str, object]] = []
    tracks = []
    frame_id = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_id / max(fps, 1.0)
            if frame_id % max(1, round(fps / settings.person_detection_hz_cpu)) == 0:
                tracks = tracker.update(detector.detect(frame))
            faces = []
            if frame_id % max(1, round(fps / settings.face_detection_hz_cpu)) == 0:
                faces = face_backend.analyze(frame, enrollment=False)
            decisions = confirmation.process(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_shape=frame.shape,
                tracks=tracks,
                faces=faces,
                target=target,
            )
            for track in tracks:
                x1, y1, x2, y2 = track.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 0), 2)
                cv2.putText(
                    frame,
                    f"track {track.track_id}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 0),
                    1,
                )
            for decision in decisions:
                color = (0, 255, 0) if decision.state == MatchState.CONFIRMED else (0, 165, 255)
                x1, y1, x2, y2 = decision.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    f"{decision.state.value} {decision.similarity:.3f}",
                    (x1, min(height - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                events.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_seconds": timestamp,
                        "state": decision.state.value,
                        "track_id": decision.track_id,
                        "bbox": normalize_bbox(decision.bbox, frame.shape),
                        "similarity": decision.similarity,
                        "quality": decision.quality,
                        "evidence_count": decision.evidence_count,
                    }
                )
            writer.write(frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        manager.shutdown()

    summary = {
        "photo": str(photo_path),
        "video": str(video_path),
        "model": face_backend.model_name,
        "target_name": target.name,
        "provider": face_backend.provider_name,
        "similarity_threshold": settings.similarity_threshold,
        "frames": frame_id,
        "elapsed_seconds": time.monotonic() - started,
        "confirmed_events": sum(item["state"] == "confirmed" for item in events),
        "events": events,
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "events"}, indent=2))


if __name__ == "__main__":
    main()
