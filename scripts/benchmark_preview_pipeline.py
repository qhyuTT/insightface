"""Synthetic NV12/slow-AI/JPEG benchmark; no camera, model, identity or network.

Run the same command against old/new reader files to isolate queue policy.
This measures server JPEG publication, not real decoder or browser performance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from person_search.config import Settings
from person_search.domain import SourceConfig
from person_search.video import LatestFrameReader


class SyntheticCapture:
    def __init__(self, width: int, height: int, fps: float):
        self.nv12 = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
        self.nv12[:height] = np.linspace(16, 235, width, dtype=np.uint8)
        self.interval = 1.0 / fps
        self.next_at = time.monotonic()
        self.closed = threading.Event()
        self.reads = 0
        self.grabs = 0

    def isOpened(self):
        return not self.closed.is_set()

    def _tick(self):
        self.next_at += self.interval
        return not self.closed.wait(max(0.0, self.next_at - time.monotonic()))

    def read(self):
        if not self._tick():
            return False, None
        self.reads += 1
        return True, cv2.cvtColor(self.nv12, cv2.COLOR_YUV2BGR_NV12)

    def grab(self):
        if not self._tick():
            return False
        self.grabs += 1
        return True

    def release(self):
        self.closed.set()

    interrupt = release


def run_source(reader_type, width, height, source_hz, preview_hz, ai_ms, seconds):
    stop = threading.Event()
    capture = SyntheticCapture(width, height, source_hz)
    drops = []
    processed = []
    published = []
    ages = []
    encode_ms = []
    reader = reader_type(
        SourceConfig(type="camera", device_index=0),
        Settings(frame_queue_size=2), lambda *_: None, lambda: drops.append(1),
    )
    reader._open = lambda: capture

    def ai_worker():
        while not stop.is_set():
            packet = reader.get(timeout=0.1)
            if packet is not None and not stop.wait(ai_ms / 1000.0):
                processed.append(time.monotonic())

    def preview_worker():
        last_id = -1
        while not stop.is_set():
            started = time.monotonic()
            packet = reader.latest()
            if packet is not None and packet.frame_id != last_id:
                canvas = cv2.resize(packet.frame, (960, round(height * 960 / width)),
                                    interpolation=cv2.INTER_AREA)
                ok, _ = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    now = time.monotonic()
                    published.append(now)
                    ages.append((now - packet.captured_at) * 1000.0)
                    encode_ms.append((now - started) * 1000.0)
                    last_id = packet.frame_id
            stop.wait(max(0.0, 1.0 / preview_hz - (time.monotonic() - started)))

    workers = [threading.Thread(target=target) for target in (ai_worker, preview_worker)]
    started = time.monotonic()
    reader.start()
    for worker in workers:
        worker.start()
    stop.wait(seconds)
    stop.set()
    reader.stop()
    for worker in workers:
        worker.join(timeout=2.0)
        if worker.is_alive():
            raise RuntimeError("benchmark worker did not stop")
    elapsed = time.monotonic() - started
    return {
        "resolution": [width, height], "source_hz": source_hz,
        "preview_target_hz": preview_hz, "ai_sleep_ms": ai_ms,
        "duration_seconds": elapsed, "bgr_reads": capture.reads,
        "skipped_bgr_reads": capture.grabs, "ai_frames": len(processed),
        "jpeg_frames": len(published), "jpeg_fps": len(published) / elapsed,
        "ai_fps": len(processed) / elapsed, "ai_drops": len(drops),
        "queue_length": reader.frames.qsize(),
        "jpeg_frame_age_p95_ms": float(np.percentile(ages, 95)) if ages else 0.0,
        "jpeg_encode_p95_ms": float(np.percentile(encode_ms, 95)) if encode_ms else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-file", type=Path)
    parser.add_argument("--preview-hz", type=float, default=15.0)
    parser.add_argument("--ai-ms", type=float, default=200.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not 0 < args.preview_hz <= 240 or not 0 < args.ai_ms <= 10000 or args.seconds < 2:
        parser.error("preview/AI rates must be positive; duration must be >= 2 seconds")
    reader_type = LatestFrameReader
    if args.reader_file:
        spec = importlib.util.spec_from_file_location("person_search.benchmark_reader", args.reader_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        reader_type = module.LatestFrameReader
    results = []
    threads = [threading.Thread(target=lambda size=size: results.append(
        run_source(reader_type, *size, args.preview_hz, args.ai_ms, args.seconds)
    )) for size in [(3840, 2160, 20.0), (1280, 720, 15.0)]]
    cpu_started = time.process_time()
    wall_started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if len(results) != 2:
        raise RuntimeError("one synthetic source failed")
    print(json.dumps({
        "kind": "synthetic_nv12_slow_ai_jpeg", "reader": str(args.reader_file or "installed"),
        "process_cpu_percent": 100 * (time.process_time() - cpu_started) / (time.monotonic() - wall_started),
        "sources": sorted(results, key=lambda item: item["resolution"][0], reverse=True),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
