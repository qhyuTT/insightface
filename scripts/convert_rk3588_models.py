#!/usr/bin/env python3
"""Convert the project's ONNX models to static RK3588 RKNN models.

This script is run in a conversion environment containing Rockchip's
``rknn-toolkit2``.  The runtime box only needs the resulting ``.rknn`` files and
``librknnrt.so``; it does not need the Toolkit2 Python package.

The face models use the same RGB preprocessing semantics as InsightFace:
SCRFD is normalized with mean 127.5 / std 128 and ArcFace with mean/std 127.5.
YOLOX keeps raw 0-255 BGR pixels, matching the project's existing ONNX
preprocessing after letterbox.  FP models are the default because they are the
safest first parity target; INT8 conversion requires a representative
calibration dataset and is opt-in.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The conversion venv installs Toolkit2 but not the application package. Keep
# this script runnable directly from a clean checkout without requiring a second
# package install just for the SHA-256 helper.
project_src = Path(__file__).resolve().parents[1] / "src"
if str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

from person_search.model_assets import sha256


@dataclass(frozen=True, slots=True)
class ConversionSpec:
    name: str
    source: Path
    input_name: str
    input_size: tuple[int, int, int, int]
    mean_values: list[list[float]]
    std_values: list[list[float]]
    output_name: str
    # The toolkit reads calibration images as RGB.  YOLOX consumes raw BGR at
    # runtime, so its calibration set must be swapped back; the face models take
    # RGB and must not be.
    quant_rgb2bgr: bool = False
    # Calibration list for this spec (``--quantize``); one file per input shape,
    # because the images must already be letterboxed exactly as the runtime does.
    dataset: Path | None = None


def main() -> int:
    args = parse_args()
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        print(
            "rknn-toolkit2 is not installed; run scripts/prepare_rknn_toolkit.sh "
            "or activate the vendor conversion environment first",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    specs = build_specs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "target_platform": args.platform,
        "quantized": args.quantize,
        "calibration_dataset": str(args.dataset) if args.dataset else None,
        "scrfd_size": list(args.scrfd_size),
        "yolox_size": list(args.yolox_size),
        "models": [],
    }
    for spec in specs:
        record = convert_one(RKNN, spec, args)
        records = manifest["models"]
        assert isinstance(records, list)
        records.append(record)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote RKNN conversion manifest: {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolox", type=Path, default=Path("models/yolox_tiny.onnx"))
    parser.add_argument(
        "--insightface-root",
        type=Path,
        default=Path("~/.insightface").expanduser(),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/rk3588")
    )
    parser.add_argument("--platform", default="rk3588")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="build INT8 models; requires --dataset with representative images",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="calibration list used for every model unless a per-model list is given",
    )
    parser.add_argument("--scrfd-dataset", type=Path, default=None)
    parser.add_argument("--yolox-dataset", type=Path, default=None)
    parser.add_argument(
        "--scrfd-size",
        type=_parse_size,
        default=(640, 640),
        help="static SCRFD input as WxH, multiples of 32 (default 640x640). A 16:9 "
        "camera wastes 44%% of a square input on padding; 1280x736 sees a 4K face at "
        "twice the pixels of 640x640 for 2.3x the compute",
    )
    parser.add_argument(
        "--yolox-size",
        type=_parse_size,
        default=(416, 416),
        help="static YOLOX input as WxH, multiples of 32 (default 416x416)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="appended to output file names, e.g. '_int8'; sizes other than the "
        "defaults are always encoded in the name",
    )
    parser.add_argument(
        "--only",
        choices=("all", "yolox", "face"),
        default="all",
        help="convert one model family while iterating",
    )
    args = parser.parse_args()
    for name in ("dataset", "scrfd_dataset", "yolox_dataset"):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            parser.error(f"calibration dataset does not exist: {value}")
    if args.quantize:
        needs_face = args.only in ("all", "face") and (args.scrfd_dataset or args.dataset) is None
        needs_yolox = args.only in ("all", "yolox") and (args.yolox_dataset or args.dataset) is None
        if needs_face or needs_yolox:
            parser.error("--quantize requires --dataset or a per-model --scrfd-dataset/--yolox-dataset")
    return args


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x")
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WxH, got {value!r}") from exc
    if width % 32 or height % 32 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive multiples of 32")
    return width, height


def _sized_name(stem: str, size: tuple[int, int], default: tuple[int, int], suffix: str) -> str:
    label = "" if size == default else f"_{size[0]}x{size[1]}"
    return f"{stem}{label}{suffix}.rknn"


def build_specs(args: argparse.Namespace) -> list[ConversionSpec]:
    face_root = args.insightface_root.expanduser() / "models" / "buffalo_l"
    suffix = args.suffix
    specs: list[ConversionSpec] = []
    if args.only in ("all", "yolox"):
        width, height = args.yolox_size
        specs.append(
            ConversionSpec(
                name="yolox_tiny",
                source=args.yolox,
                input_name="images",
                input_size=(1, 3, height, width),
                mean_values=[[0.0, 0.0, 0.0]],
                std_values=[[1.0, 1.0, 1.0]],
                output_name=_sized_name("yolox_tiny", args.yolox_size, (416, 416), suffix),
                quant_rgb2bgr=True,
                dataset=args.yolox_dataset or args.dataset,
            )
        )
    if args.only in ("all", "face"):
        width, height = args.scrfd_size
        specs.extend(
            [
                ConversionSpec(
                    name="scrfd_10g",
                    source=face_root / "det_10g.onnx",
                    input_name="input.1",
                    input_size=(1, 3, height, width),
                    mean_values=[[127.5, 127.5, 127.5]],
                    std_values=[[128.0, 128.0, 128.0]],
                    output_name=(
                        f"scrfd_10g_640{suffix}.rknn"
                        if args.scrfd_size == (640, 640)
                        else f"scrfd_10g_{width}x{height}{suffix}.rknn"
                    ),
                    dataset=args.scrfd_dataset or args.dataset,
                ),
                ConversionSpec(
                    name="w600k_r50",
                    source=face_root / "w600k_r50.onnx",
                    input_name="input.1",
                    input_size=(1, 3, 112, 112),
                    mean_values=[[127.5, 127.5, 127.5]],
                    std_values=[[127.5, 127.5, 127.5]],
                    output_name="w600k_r50.rknn",
                    # ArcFace stays FP16: an INT8 embedding shifts every similarity
                    # and would silently invalidate the thresholds.
                    dataset=None,
                ),
            ]
        )
    return specs


def convert_one(rknn_factory: Any, spec: ConversionSpec, args: argparse.Namespace) -> dict[str, object]:
    if not spec.source.is_file():
        raise SystemExit(f"source ONNX model does not exist: {spec.source}")
    output = args.output_dir / spec.output_name
    print(f"converting {spec.name}: {spec.source} -> {output}")
    quantize = bool(args.quantize) and spec.dataset is not None
    rknn = rknn_factory(verbose=False)
    try:
        check(rknn.config(
            mean_values=spec.mean_values,
            std_values=spec.std_values,
            target_platform=args.platform,
            quant_img_RGB2BGR=spec.quant_rgb2bgr,
        ), "rknn.config")
        check(
            rknn.load_onnx(
                model=str(spec.source),
                inputs=[spec.input_name],
                input_size_list=[list(spec.input_size)],
            ),
            "rknn.load_onnx",
        )
        check(
            rknn.build(
                do_quantization=quantize,
                dataset=str(spec.dataset) if quantize else None,
            ),
            "rknn.build",
        )
        check(rknn.export_rknn(str(output)), "rknn.export_rknn")
    finally:
        rknn.release()
    record = {
        "name": spec.name,
        "source": str(spec.source),
        "input_name": spec.input_name,
        "source_sha256": sha256(spec.source),
        "output": str(output),
        "output_sha256": sha256(output),
        "input_size": list(spec.input_size),
        "mean_values": spec.mean_values,
        "std_values": spec.std_values,
        "quantized": quantize,
        "calibration_dataset": str(spec.dataset) if quantize else None,
    }
    print(f"converted {spec.name}: {record['output_sha256']}")
    return record


def check(result: Any, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} returned RKNN error code {result}")


if __name__ == "__main__":
    raise SystemExit(main())
