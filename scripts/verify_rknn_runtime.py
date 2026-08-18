#!/usr/bin/env python3
"""Smoke-test RKNN Lite and one or more converted model artifacts on a board."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from contextlib import suppress
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help=".rknn model files to load")
    parser.add_argument(
        "--core-mask",
        type=int,
        choices=range(8),
        default=None,
        metavar="0..7",
    )
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"python: {sys.version.split()[0]}")
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        print(f"RKNN Lite import failed: {exc}", file=sys.stderr)
        return 2

    for path in args.models:
        if not path.is_file():
            print(f"missing model: {path}", file=sys.stderr)
            return 3
        runtime = RKNNLite()
        try:
            load_result = runtime.load_rknn(str(path))
            if load_result not in (None, 0):
                print(f"{path}: load_rknn failed with {load_result!r}", file=sys.stderr)
                return 4
            kwargs = {} if args.core_mask is None else {"core_mask": args.core_mask}
            try:
                init_result = runtime.init_runtime(**kwargs)
            except TypeError:
                if not kwargs:
                    raise
                print(
                    f"{path}: runtime does not accept core_mask; retrying in auto mode",
                    file=sys.stderr,
                )
                init_result = runtime.init_runtime()
            print(
                f"{path}: load={load_result!r} init={init_result!r} "
                f"sha256={sha256(path)}"
            )
            if init_result not in (None, 0):
                return 4
        except Exception as exc:  # noqa: BLE001 - vendor runtime errors vary by release
            print(f"{path}: RKNN runtime check failed: {exc}", file=sys.stderr)
            return 4
        finally:
            with suppress(Exception):
                runtime.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
