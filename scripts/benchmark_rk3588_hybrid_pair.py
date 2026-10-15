"""Benchmark independent RPP face and RKNN person detection on one RK3588 frame."""

from __future__ import annotations

import argparse
import faulthandler
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from person_search.config import Settings
from person_search.hybrid_backends import RppScrfdRknnFaceBackend
from person_search.rknn_backends import RknnFaceBackend, RknnYoloXDetector


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--face-backend", choices=("rknn", "rpp"), default="rpp")
    parser.add_argument("--start-at", type=float)
    args = parser.parse_args()
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        parser.error(f"cannot read image: {args.image}")
    if args.width < 32 or args.height < 32 or args.repeats < 3:
        parser.error("width/height must be >=32 and repeats must be >=3")
    frame = cv2.resize(frame, (args.width, args.height))

    settings = Settings()
    person = RknnYoloXDetector(settings)
    face = (
        RppScrfdRknnFaceBackend(settings)
        if args.face_backend == "rpp"
        else RknnFaceBackend(settings)
    )
    try:
        person.ensure_ready()
        face.ensure_ready()
        for _ in range(2):
            person.detect(frame)
            face.detect_faces(frame)
        if args.start_at is not None:
            time.sleep(max(0.0, args.start_at - time.time()))

        serial_ms = []
        parallel_ms = []
        person_count = face_count = 0
        with ThreadPoolExecutor(max_workers=2) as executor:
            for _ in range(args.repeats):
                started = time.perf_counter()
                people = person.detect(frame)
                faces = face.detect_faces(frame)
                serial_ms.append((time.perf_counter() - started) * 1000)
                person_count, face_count = len(people), len(faces)

                started = time.perf_counter()
                person_future = executor.submit(person.detect, frame)
                face_future = executor.submit(face.detect_faces, frame)
                parallel_people = person_future.result()
                parallel_faces = face_future.result()
                parallel_ms.append((time.perf_counter() - started) * 1000)
                if (len(parallel_people), len(parallel_faces)) != (
                    person_count,
                    face_count,
                ):
                    raise RuntimeError("serial and parallel detection counts differ")

        print(json.dumps({
            "frame_size": [args.width, args.height],
            "repeats": args.repeats,
            "providers": [person.provider_name, face.detection_provider_name],
            "person_count": person_count,
            "face_count": face_count,
            "serial_p50_ms": round(statistics.median(serial_ms), 2),
            "parallel_p50_ms": round(statistics.median(parallel_ms), 2),
            "speedup": round(statistics.median(serial_ms) / statistics.median(parallel_ms), 3),
            "serial_p95_ms": round(float(np.percentile(serial_ms, 95)), 2),
            "parallel_p95_ms": round(float(np.percentile(parallel_ms, 95)), 2),
        }, ensure_ascii=False), flush=True)
    finally:
        face.close()
        person.close()


if __name__ == "__main__":
    main()
