#!/usr/bin/env python3
"""Convert the project's ONNX models to static RK3588 RKNN models.

This script is run in a conversion environment containing Rockchip's
``rknn-toolkit2``.  The runtime box only needs the resulting ``.rknn`` files and
``librknnrt.so``; it does not need the Toolkit2 Python package.

The face models use the same RGB preprocessing semantics as InsightFace:
SCRFD is normalized with mean 127.5 / std 128 and ArcFace with mean/std 127.5.
YOLOX keeps raw 0-255 pixels, matching the project's existing ONNX preprocessing
after letterbox and RGB conversion.  FP models are the default because they
are the safest first parity target; INT8 conversion requires a representative
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
    input_size: tuple[int, int, int, int]
    mean_values: list[list[float]]
    std_values: list[list[float]]
    output_name: str


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
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument(
        "--only",
        choices=("all", "yolox", "face"),
        default="all",
        help="convert one model family while iterating",
    )
    args = parser.parse_args()
    if args.quantize and args.dataset is None:
        parser.error("--quantize requires --dataset")
    if args.dataset is not None and not args.dataset.is_file():
        parser.error(f"calibration dataset does not exist: {args.dataset}")
    return args


def build_specs(args: argparse.Namespace) -> list[ConversionSpec]:
    face_root = args.insightface_root.expanduser() / "models" / "buffalo_l"
    specs: list[ConversionSpec] = []
    if args.only in ("all", "yolox"):
        specs.append(
            ConversionSpec(
                name="yolox_tiny",
                source=args.yolox,
                input_size=(1, 3, 416, 416),
                mean_values=[[0.0, 0.0, 0.0]],
                std_values=[[1.0, 1.0, 1.0]],
                output_name="yolox_tiny.rknn",
            )
        )
    if args.only in ("all", "face"):
        specs.extend(
            [
                ConversionSpec(
                    name="scrfd_10g_640",
                    source=face_root / "det_10g.onnx",
                    input_size=(1, 3, 640, 640),
                    mean_values=[[127.5, 127.5, 127.5]],
                    std_values=[[128.0, 128.0, 128.0]],
                    output_name="scrfd_10g_640.rknn",
                ),
                ConversionSpec(
                    name="w600k_r50",
                    source=face_root / "w600k_r50.onnx",
                    input_size=(1, 3, 112, 112),
                    mean_values=[[127.5, 127.5, 127.5]],
                    std_values=[[127.5, 127.5, 127.5]],
                    output_name="w600k_r50.rknn",
                ),
            ]
        )
    return specs


def convert_one(rknn_factory: Any, spec: ConversionSpec, args: argparse.Namespace) -> dict[str, object]:
    if not spec.source.is_file():
        raise SystemExit(f"source ONNX model does not exist: {spec.source}")
    output = args.output_dir / spec.output_name
    print(f"converting {spec.name}: {spec.source} -> {output}")
    rknn = rknn_factory(verbose=False)
    try:
        check(rknn.config(
            mean_values=spec.mean_values,
            std_values=spec.std_values,
            target_platform=args.platform,
        ), "rknn.config")
        check(
            rknn.load_onnx(
                model=str(spec.source),
                input_size_list=[list(spec.input_size)],
            ),
            "rknn.load_onnx",
        )
        check(
            rknn.build(
                do_quantization=bool(args.quantize),
                dataset=str(args.dataset) if args.dataset else None,
            ),
            "rknn.build",
        )
        check(rknn.export_rknn(str(output)), "rknn.export_rknn")
    finally:
        rknn.release()
    record = {
        "name": spec.name,
        "source": str(spec.source),
        "source_sha256": sha256(spec.source),
        "output": str(output),
        "output_sha256": sha256(output),
        "input_size": list(spec.input_size),
        "mean_values": spec.mean_values,
        "std_values": spec.std_values,
    }
    print(f"converted {spec.name}: {record['output_sha256']}")
    return record


def check(result: Any, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} returned RKNN error code {result}")


if __name__ == "__main__":
    raise SystemExit(main())
