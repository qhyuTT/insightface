#!/usr/bin/env bash
set -euo pipefail

# Native deployment for the EIS100 RK3588 box. This script never stops or
# replaces existing services and does not require Docker or sudo. It uploads
# source/configuration, prepares a Python 3.11 venv with uv, and optionally
# starts a per-user systemd service on port 18080.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
ssh_target="${RK3588_SSH_TARGET:-EIS100RK3588@192.168.11.203}"
remote_dir="${RK3588_REMOTE_DIR:-/ssd/person-search}"
expected_branch="${RK3588_EXPECTED_BRANCH:-eis100}"
remote_python="${RK3588_PYTHON:-3.11}"
start_service="${RK3588_START_SERVICE:-0}"
allow_dirty="${RK3588_ALLOW_DIRTY:-0}"
bootstrap_host="${RK3588_BOOTSTRAP_HOST:-0}"
remote_runtime_source="${RK3588_REMOTE_RUNTIME_SOURCE:-}"

current_branch="$(git -C "${repo_dir}" branch --show-current)"
if [[ "${current_branch}" != "${expected_branch}" ]]; then
  echo "expected branch ${expected_branch}, got ${current_branch}" >&2
  exit 1
fi
if [[ "${allow_dirty}" != "1" ]] && [[ -n "$(git -C "${repo_dir}" status --short)" ]]; then
  echo "refusing to deploy a dirty checkout; set RK3588_ALLOW_DIRTY=1 only for an explicit experiment" >&2
  exit 1
fi

commit="$(git -C "${repo_dir}" rev-parse HEAD)"
echo "deploying ${commit} from ${current_branch} to ${ssh_target}:${remote_dir}"

command -v ssh >/dev/null || { echo "ssh is required" >&2; exit 1; }
command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 1; }

remote_dir_q="$(printf '%q' "${remote_dir}")"
ssh "${ssh_target}" "mkdir -p -- ${remote_dir_q}"
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.rknn-venv/' \
  --exclude 'models/*' \
  --exclude 'artifacts/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "${repo_dir}/" "${ssh_target}:${remote_dir%/}/"

if [[ -n "${RK3588_ENV_FILE:-}" ]]; then
  test -f "${RK3588_ENV_FILE}" || {
    echo "RK3588_ENV_FILE does not exist: ${RK3588_ENV_FILE}" >&2
    exit 1
  }
  scp "${RK3588_ENV_FILE}" "${ssh_target}:${remote_dir%/}/.env"
else
  echo "preserving remote .env; use deploy/rk3588.env.example as the starting point"
fi

if [[ -n "${RK3588_MODEL_DIR:-}" ]]; then
  test -d "${RK3588_MODEL_DIR}" || {
    echo "RK3588_MODEL_DIR does not exist: ${RK3588_MODEL_DIR}" >&2
    exit 1
  }
  rsync -az "${RK3588_MODEL_DIR%/}/" "${ssh_target}:${remote_dir%/}/models/rk3588/"
fi

if [[ -n "${RK3588_RUNTIME_LIBRARY:-}" ]]; then
  test -f "${RK3588_RUNTIME_LIBRARY}" || {
    echo "RK3588_RUNTIME_LIBRARY does not exist: ${RK3588_RUNTIME_LIBRARY}" >&2
    exit 1
  }
  ssh "${ssh_target}" "mkdir -p -- ${remote_dir_q}/runtime"
  scp "${RK3588_RUNTIME_LIBRARY}" \
    "${ssh_target}:${remote_dir%/}/runtime/librknnrt.so"
elif [[ -n "${remote_runtime_source}" ]]; then
  remote_runtime_source_q="$(printf '%q' "${remote_runtime_source}")"
  ssh "${ssh_target}" "
    set -eu
    test -f ${remote_runtime_source_q}
    mkdir -p -- ${remote_dir_q}/runtime
    install -m 0644 ${remote_runtime_source_q} ${remote_dir_q}/runtime/librknnrt.so
  "
fi

remote_python_q="$(printf '%q' "${remote_python}")"
ssh "${ssh_target}" "
  set -eu
  cd -- ${remote_dir_q}
  if [ \"${bootstrap_host}\" = 1 ]; then
    bash scripts/prepare_rk3588_host.sh
  fi
  uv_bin=\"\$(command -v uv || true)\"
  if [ -z \"\${uv_bin}\" ]; then
    uv_bin=\"\${RK3588_UV_BIN:-\${HOME}/.local/bin/uv}\"
  fi
  test -x \"\${uv_bin}\"
  \"\${uv_bin}\" venv --python ${remote_python_q} .venv
  \"\${uv_bin}\" sync --python .venv/bin/python --no-dev
  mkdir -p \"\${XDG_CONFIG_HOME:-\${HOME}/.config}/systemd/user\"
  install -m 0644 deploy/person-search-rk3588.service \"\${XDG_CONFIG_HOME:-\${HOME}/.config}/systemd/user/person-search-rk3588.service\"
  systemctl --user daemon-reload
"

if [[ "${start_service}" == "1" ]]; then
  ssh "${ssh_target}" "systemctl --user enable --now person-search-rk3588.service"
  echo "waiting for native service on ${ssh_target}:18080"
  for _ in {1..20}; do
    if ssh "${ssh_target}" "curl --fail --silent http://127.0.0.1:18080/healthz"; then
      echo
      echo "native service is healthy; commit=${commit}"
      exit 0
    fi
    sleep 1
  done
  ssh "${ssh_target}" "systemctl --user --no-pager status person-search-rk3588.service || true"
  exit 1
fi

echo "uploaded commit=${commit}; service not started (set RK3588_START_SERVICE=1 after filling .env)"
