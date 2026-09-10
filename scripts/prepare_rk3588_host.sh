#!/usr/bin/env bash
set -euo pipefail

# Install uv and its managed CPython 3.11 into the current user's directories.
# No sudo, apt changes, Docker, or system Python replacement is performed.
python_version="${RK3588_PYTHON_VERSION:-3.11}"
uv_bin="$(command -v uv || true)"
if [[ -z "${uv_bin}" ]]; then
  command -v curl >/dev/null || {
    echo "curl is required to bootstrap uv" >&2
    exit 1
  }
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh | sh
  uv_bin="${RK3588_UV_BIN:-${HOME}/.local/bin/uv}"
fi
test -x "${uv_bin}" || {
  echo "uv was not found at ${uv_bin}; set RK3588_UV_BIN to its path" >&2
  exit 1
}

"${uv_bin}" python install "${python_version}"
"${uv_bin}" python find "${python_version}"
echo "RK3588 user-local Python ready: ${python_version}"
