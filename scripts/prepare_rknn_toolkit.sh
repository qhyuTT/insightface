#!/usr/bin/env bash
set -euo pipefail

# RKNN Toolkit2 is distributed by Rockchip rather than PyPI.  Point this helper
# at the wheel unpacked from the vendor package; the resulting conversion venv is
# intentionally separate from the service's Python 3.11 venv.
toolkit_wheel="${RKNN_TOOLKIT_WHEEL:?set RKNN_TOOLKIT_WHEEL to the vendor rknn-toolkit2 wheel}"
rknn_python="${RKNN_PYTHON:-python3.10}"
rknn_venv="${RKNN_VENV:-.rknn-venv}"

command -v uv >/dev/null || {
  echo "uv is required; install it before preparing the RKNN conversion venv" >&2
  exit 1
}
test -f "${toolkit_wheel}" || {
  echo "RKNN toolkit wheel not found: ${toolkit_wheel}" >&2
  exit 1
}

uv venv --allow-existing --python "${rknn_python}" "${rknn_venv}"
uv pip install --python "${rknn_venv}/bin/python" \
  'cmake>=3.22' \
  'setuptools==69.5.1'

# onnxoptimizer 0.3.8 currently has no manylinux aarch64 wheel.  Its bundled
# ONNX CMake project declares an obsolete minimum policy, so use a user-local
# CMake and opt into the compatibility policy rather than requiring sudo to
# upgrade the box's system CMake.
conversion_path="${rknn_venv}/bin:${PATH}"
conversion_cmake_args="${RKNN_CMAKE_ARGS:--DCMAKE_POLICY_VERSION_MINIMUM=3.5}"
conversion_max_jobs="${RKNN_MAX_JOBS:-4}"
PATH="${conversion_path}" CMAKE_ARGS="${conversion_cmake_args}" MAX_JOBS="${conversion_max_jobs}" \
uv pip install --python "${rknn_venv}/bin/python" \
  "${toolkit_wheel}" \
  'numpy<2' \
  'onnx<1.17' \
  'onnxoptimizer==0.3.8' \
  'Pillow>=10.0.1'

"${rknn_venv}/bin/python" -c \
  'from rknn.api import RKNN; print("RKNN Toolkit2 import OK:", RKNN)'
echo "conversion environment ready: ${rknn_venv}"
