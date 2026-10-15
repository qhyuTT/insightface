"""Read existing HTTP metrics without opening video or exposing credentials."""

import argparse
import configparser
import json
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

METRICS = (
    "status",
    "processed_fps",
    "source_fps",
    "decode_fps",
    "decoded_frames",
    "processed_frames",
    "queue_length",
    "queue_capacity",
    "last_decoded_frame_age_ms",
    "capture_p50_latency_ms",
    "capture_p95_latency_ms",
    "capture_max_latency_ms",
    "dropped_frames",
    "drop_rate",
    "preview_fps",
    "preview_frame_age_p95_ms",
    "preview_encode_p95_latency_ms",
    "end_to_end_p95_latency_ms",
    "p95_latency_ms",
    "stage_p95_latency_ms",
    "stage_p50_latency_ms",
    "stage_max_latency_ms",
    "stage_sample_counts",
    "latency_scope",
    "effective_hz",
    "frame_width",
    "frame_height",
    "roi_calls_per_frame",
    "budget_skips",
    "embedding_batch_count",
    "embedding_failures",
    "accepted_faces",
    "face_observations",
    "achieved_sampling_hz",
)
CONFIG = (
    "preview_hz",
    "person_detection_hz",
    "face_detection_hz",
    "gstreamer_nv12_converter_active",
    "gstreamer_nv12_converter_fallback_reason",
    "gstreamer_latency_ms",
    "rknn_core_mask_applied",
)


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--zlm-container", default="zlmediakit")
    parser.add_argument("--skip-media", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.interval <= 0:
        parser.error("samples and interval must be positive")
    media_url = None
    if not args.skip_media:
        try:
            config = configparser.ConfigParser(interpolation=None)
            config.read_string(
                subprocess.check_output(
                    ["docker", "exec", args.zlm_container, "cat", "/opt/media/conf/config.ini"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            )
            media_url = (
                "http://127.0.0.1:"
                + config.get("http", "port", fallback="80")
                + "/index/api/getMediaList?"
                + urllib.parse.urlencode({"secret": config.get("api", "secret", fallback="")})
            )
        except (OSError, ValueError, configparser.Error, subprocess.SubprocessError) as exc:
            print(json.dumps({"media_setup_error": type(exc).__name__}), flush=True)
    previous = {}
    for index in range(args.samples):
        started = time.monotonic()
        # The box uses Python 3.8, which does not provide datetime.UTC.
        row = {"sample": index, "utc": datetime.now(timezone.utc).isoformat()}  # noqa: UP017
        try:
            active = fetch(args.base_url.rstrip("/") + "/v1/searches/active")
            row["active"] = active is not None
            if isinstance(active, dict):
                row["metrics"] = {key: active[key] for key in METRICS if key in active}
                row["sources"] = [
                    {
                        key: source[key]
                        for key in (
                            "source_id",
                        ) + METRICS
                        if key in source
                    }
                    for source in active.get("sources", [])
                ]
                effective = active.get("effective_config", {})
                row["config"] = {key: effective[key] for key in CONFIG if key in effective}
        except (OSError, ValueError, TypeError) as exc:
            row["api_error"] = type(exc).__name__
        if media_url:
            try:
                media = fetch(media_url)
                received_at = time.monotonic()
                row["media_code"] = media.get("code")
                streams = []
                for source in media.get("data", []):
                    if source.get("schema") != "rtsp":
                        continue
                    for track in source.get("tracks", []):
                        if track.get("codec_type") != 0:
                            continue
                        name = source.get("stream")
                        key = (source.get("vhost"), source.get("app"), name)
                        frames = track.get("frames")
                        item = {
                            "stream": name,
                            "bytes_per_second": source.get("bytesSpeed"),
                            "readers": source.get("readerCount"),
                        }
                        item.update(
                            {
                                key: track.get(key)
                                for key in (
                                    "codec_id_name",
                                    "width",
                                    "height",
                                    "fps",
                                    "frames",
                                    "gop_interval_ms",
                                )
                            }
                        )
                        old = previous.get(key)
                        if isinstance(frames, (int, float)):
                            if old and frames >= old[1] and received_at > old[0]:
                                item["media_frame_counter_delta_hz"] = (frames - old[1]) / (
                                    received_at - old[0]
                                )
                            previous[key] = (received_at, frames)
                        streams.append(item)
                row["media"] = streams
            except (OSError, ValueError, TypeError) as exc:
                row["media_error"] = type(exc).__name__
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if index + 1 < args.samples:
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
