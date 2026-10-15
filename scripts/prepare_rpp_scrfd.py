"""Prepare fixed-shape SCRFD graphs for RPP before deploying the service."""

from __future__ import annotations

import argparse
from pathlib import Path

from person_search.rpp_backends import _scrfd_static_model_path, _static_scrfd_model


def _shape(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shape must be WIDTHxHEIGHT") from exc
    if min(width, height) <= 0 or width % 32 or height % 32:
        raise argparse.ArgumentTypeError("shape dimensions must be positive multiples of 32")
    return height, width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model",
        type=Path,
        default=Path("~/.insightface/models/buffalo_l/det_10g.onnx"),
        help="upstream InsightFace SCRFD ONNX file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/rpp-static"),
        help="directory for static RPP model artifacts",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_shape,
        metavar="WIDTHxHEIGHT",
        help="static input shape (repeatable; defaults to 1280x736 and 640x640)",
    )
    args = parser.parse_args()
    source_path = args.source_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_path.is_file():
        parser.error(f"SCRFD source model does not exist: {source_path}")

    source_bytes = source_path.read_bytes()
    shapes = args.shape or [(736, 1280), (640, 640)]
    for input_hw in dict.fromkeys(shapes):
        _static_scrfd_model(source_bytes, input_hw=input_hw, cache_dir=output_dir)
        artifact = _scrfd_static_model_path(
            source_bytes, input_hw=input_hw, model_dir=output_dir
        )
        print(f"prepared {artifact} ({artifact.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
