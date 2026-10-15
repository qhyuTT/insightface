"""Compare RKNN and RPP SCRFD on the same image and the same RK3588 box.

Run this on the box after both accelerators are healthy. It measures the
complete detect_faces call, including transfers and CPU preprocessing/NMS.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from person_search.config import Settings
from person_search.hybrid_backends import RppScrfdRknnFaceBackend
from person_search.rknn_backends import RknnFaceBackend
from person_search.rpp_backends import _decode_scrfd, _preprocess_scrfd


def _measure(backend, frame: np.ndarray, *, enrollment: bool, repeats: int) -> dict:
    backend.ensure_ready()
    for _ in range(2):
        backend.detect_faces(frame, enrollment=enrollment)

    samples_ms = []
    observations = []
    for _ in range(repeats):
        start = time.perf_counter()
        observations = backend.detect_faces(frame, enrollment=enrollment)
        samples_ms.append((time.perf_counter() - start) * 1000)

    return {
        "provider": backend.detection_provider_name,
        "p50_ms": round(statistics.median(samples_ms), 2),
        "p95_ms": round(float(np.percentile(samples_ms, 95)), 2),
        "faces": [
            {"bbox": [round(float(v), 2) for v in face.bbox],
             "score": round(float(face.detection_score), 4)}
            for face in observations
        ],
    }


def _profile_rpp(backend, frame: np.ndarray, *, enrollment: bool, repeats: int) -> dict:
    detector = backend._enrollment_detector if enrollment else backend._search_detector
    assert detector is not None
    input_hw = (
        backend.enrollment_static_input_hw if enrollment else backend.static_input_hw
    )
    stages = {name: [] for name in ("preprocess", "engine_run", "decode")}
    for _ in range(repeats):
        start = time.perf_counter()
        tensor, ratio = _preprocess_scrfd(frame, input_hw)
        preprocess_end = time.perf_counter()
        outputs = detector.run(tensor)
        run_end = time.perf_counter()
        _decode_scrfd(
            frame,
            outputs,
            backend._detection_output_names,
            input_hw,
            ratio,
            threshold=backend.settings.face_detection_threshold,
        )
        decode_end = time.perf_counter()
        for name, elapsed in (
            ("preprocess", preprocess_end - start),
            ("engine_run", run_end - preprocess_end),
            ("decode", decode_end - run_end),
        ):
            stages[name].append(elapsed * 1000)
    return {name + "_p50_ms": round(statistics.median(values), 2)
            for name, values in stages.items()}


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--enrollment", action="store_true")
    parser.add_argument("--backend", choices=("rknn", "rpp", "both"), default="both")
    parser.add_argument("--profile-rpp", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        parser.error(f"cannot read image: {args.image}")

    settings = Settings()
    results = {}
    for name, backend_type in (
        ("rknn", RknnFaceBackend),
        ("rpp", RppScrfdRknnFaceBackend),
    ):
        if args.backend not in (name, "both"):
            continue
        print(f"{name}: initialize", file=sys.stderr, flush=True)
        backend = backend_type(settings)
        try:
            results[name] = _measure(
                backend, frame, enrollment=args.enrollment, repeats=args.repeats
            )
            if name == "rpp" and args.profile_rpp:
                results[name]["stages"] = _profile_rpp(
                    backend, frame, enrollment=args.enrollment, repeats=args.repeats
                )
            print(f"{name}: measured", file=sys.stderr, flush=True)
            print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)
        finally:
            print(f"{name}: close", file=sys.stderr, flush=True)
            backend.close()
            print(f"{name}: closed", file=sys.stderr, flush=True)

    summary = {
        "image_size": [int(frame.shape[1]), int(frame.shape[0])],
        "enrollment": args.enrollment,
        "repeats": args.repeats,
        **results,
    }
    if len(results) == 2:
        summary["detection_p50_speedup"] = round(
            results["rknn"]["p50_ms"] / results["rpp"]["p50_ms"], 3
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
