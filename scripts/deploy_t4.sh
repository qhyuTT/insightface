#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

T4_IMAGE_NAME="${T4_IMAGE_NAME:-person-search:t4}"
T4_CONTAINER_NAME="${T4_CONTAINER_NAME:-person-search}"
T4_MODEL_VOLUME="${T4_MODEL_VOLUME:-person-search-models}"
T4_CUDA_IMAGE="${T4_CUDA_IMAGE:-docker.m.daocloud.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
T4_PIP_INDEX_URL="${T4_PIP_INDEX_URL:-}"
# Tried in order when T4_PIP_INDEX_URL is unset; the first reachable one wins.
T4_PIP_INDEX_CANDIDATES="${T4_PIP_INDEX_CANDIDATES-https://mirrors.aliyun.com/pypi/simple https://mirrors.ustc.edu.cn/pypi/simple https://pypi.org/simple}"
T4_ORT_VERSION="${T4_ORT_VERSION:-1.23.2}"
T4_BIND_HOST="${T4_BIND_HOST:-127.0.0.1}"
T4_PRELOAD_INSIGHTFACE="${T4_PRELOAD_INSIGHTFACE:-0}"
T4_YOLOX_MODEL_URL="${T4_YOLOX_MODEL_URL:-}"
T4_PREFETCH_YOLOX="${T4_PREFETCH_YOLOX:-1}"
# Prefix-style GitHub proxies, tried in order before the upstream URL itself.
T4_YOLOX_MIRRORS="${T4_YOLOX_MIRRORS-https://gh-proxy.com/ https://ghfast.top/ https://ghproxy.net/}"

log() {
  printf '[deploy-t4] %s\n' "$*"
}

fail() {
  printf '[deploy-t4] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_command docker
require_command nvidia-smi

docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable for the current user"
nvidia-smi -L >/dev/null 2>&1 || fail "NVIDIA driver cannot see the T4 GPU"

cd "${PROJECT_ROOT}"

# Keep this in sync with YOLOX_MODEL_SHA256 in the Dockerfile.
YOLOX_SHA256="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"
YOLOX_UPSTREAM_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
YOLOX_PATH="models/yolox_tiny.onnx"

yolox_checksum_ok() {
  [[ -f "$1" ]] || return 1
  [[ "$(sha256sum "$1" | awk '{ print $1 }')" == "${YOLOX_SHA256}" ]]
}

# Downloading the YOLOX weight inside `docker build` is the step most likely to
# fail on a slow or proxied connection, and a failure there throws away every
# cached layer above it. Fetch it on the host instead, where a partial transfer
# can be resumed and mirrors can be retried; the Dockerfile then COPYs it out of
# the build context. Set T4_PREFETCH_YOLOX=0 to download during the build.
prefetch_yolox() {
  if yolox_checksum_ok "${YOLOX_PATH}"; then
    log "YOLOX weight already present and verified: ${YOLOX_PATH}"
    return 0
  fi

  local source_url="${T4_YOLOX_MODEL_URL:-${YOLOX_UPSTREAM_URL}}"
  local -a candidates=()
  # Mirrors only apply to GitHub URLs; a custom T4_YOLOX_MODEL_URL is used as-is.
  if [[ -z "${T4_YOLOX_MODEL_URL}" ]]; then
    local mirror
    for mirror in ${T4_YOLOX_MIRRORS}; do
      candidates+=("${mirror}${source_url}")
    done
  fi
  candidates+=("${source_url}")

  mkdir -p "$(dirname "${YOLOX_PATH}")"
  local partial="${YOLOX_PATH}.part"
  local round url status
  for round in 1 2; do
    for url in "${candidates[@]}"; do
      log "Fetching YOLOX weight (round ${round}) via $(printf '%s' "${url}" | awk -F/ '{ print $3 }')"
      # Resume only when there is something to resume from: -C - on a
      # zero-length file makes curl fail before it sends the request.
      local -a resume=()
      if [[ -s "${partial}" ]]; then
        resume=(--continue-at -)
      fi
      status=0
      # --http1.1 avoids the HTTP/2 INTERNAL_ERROR some GitHub proxies return
      # mid-transfer. --speed-limit gives up on a stalled mirror instead of
      # sitting on it until --max-time. The ${a[@]+"${a[@]}"} form expands an
      # empty array safely under `set -u` in Bash 3.2.
      curl --fail --location --http1.1 ${resume[@]+"${resume[@]}"} \
        --retry 3 --retry-delay 5 --retry-connrefused \
        --connect-timeout 20 --max-time 900 \
        --speed-limit 30720 --speed-time 60 \
        --output "${partial}" "${url}" || status=$?

      if [[ "${status}" -eq 0 ]]; then
        if yolox_checksum_ok "${partial}"; then
          mv "${partial}" "${YOLOX_PATH}"
          log "YOLOX weight verified: ${YOLOX_PATH}"
          return 0
        fi
        # A complete but wrong file is usually a proxy error page, so the bytes
        # are not worth resuming from.
        log "Checksum mismatch, discarding the download"
        rm -f "${partial}"
      else
        # An interrupted transfer is kept so the next candidate can resume it.
        log "Transfer failed (curl ${status}), keeping $(wc -c <"${partial}" 2>/dev/null || echo 0) bytes for resume"
      fi
    done
  done

  rm -f "${partial}"
  log "Could not prefetch the YOLOX weight; falling back to downloading during the build"
  log "To supply it manually: place a verified yolox_tiny.onnx at ${PROJECT_ROOT}/${YOLOX_PATH}"
  return 1
}

if [[ "${T4_PREFETCH_YOLOX}" == "1" ]]; then
  require_command curl
  require_command sha256sum
  prefetch_yolox || true
fi

# A mirror that rate-limits or blocks this host fails the build several layers
# in, so pick one that answers now rather than trusting a hardcoded default.
select_pip_index() {
  local url code
  for url in ${T4_PIP_INDEX_CANDIDATES}; do
    code="$(
      curl --silent --output /dev/null --write-out '%{http_code}' \
        --connect-timeout 10 --max-time 20 "${url}/pip/" 2>/dev/null || true
    )"
    if [[ "${code}" == "200" ]]; then
      T4_PIP_INDEX_URL="${url}"
      log "Using Python package index ${url}"
      return 0
    fi
    log "Skipping ${url} (HTTP ${code:-no response})"
  done
  return 1
}

if [[ -z "${T4_PIP_INDEX_URL}" ]]; then
  require_command curl
  if ! select_pip_index; then
    fail "No usable Python package index found; set T4_PIP_INDEX_URL explicitly"
  fi
fi

build_args=(
  --build-arg "CUDA_IMAGE=${T4_CUDA_IMAGE}"
  --build-arg "PIP_INDEX_URL=${T4_PIP_INDEX_URL}"
  --build-arg "ONNXRUNTIME_GPU_VERSION=${T4_ORT_VERSION}"
)
if [[ -n "${T4_YOLOX_MODEL_URL}" ]]; then
  build_args+=(--build-arg "YOLOX_MODEL_URL=${T4_YOLOX_MODEL_URL}")
fi

log "Building ${T4_IMAGE_NAME}"
docker build "${build_args[@]}" --tag "${T4_IMAGE_NAME}" .

log "Checking NVIDIA Container Toolkit"
docker run --rm --gpus all --entrypoint nvidia-smi "${T4_IMAGE_NAME}" >/dev/null

existing_container_id="$(
  docker ps --all --quiet --filter "name=^/${T4_CONTAINER_NAME}$"
)"
if [[ -n "${existing_container_id}" ]]; then
  log "Replacing existing container ${T4_CONTAINER_NAME}"
  docker rm --force "${existing_container_id}" >/dev/null
fi

docker volume create "${T4_MODEL_VOLUME}" >/dev/null

log "Starting ${T4_CONTAINER_NAME} with host networking"
docker run --detach \
  --gpus all \
  --restart unless-stopped \
  --network host \
  --name "${T4_CONTAINER_NAME}" \
  --volume "${T4_MODEL_VOLUME}:/models" \
  --env "PERSON_SEARCH_HOST=${T4_BIND_HOST}" \
  --env "PERSON_SEARCH_PORT=8000" \
  --env "PERSON_SEARCH_PREFER_CUDA=true" \
  "${T4_IMAGE_NAME}" >/dev/null

log "Waiting for API health check"
ready=0
for _ in $(seq 1 30); do
  if docker exec "${T4_CONTAINER_NAME}" python -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=2).read()' \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "${ready}" != "1" ]]; then
  docker logs --tail 100 "${T4_CONTAINER_NAME}" >&2 || true
  fail "API did not become healthy within 60 seconds"
fi

log "Verifying CUDAExecutionProvider with the YOLOX model"
docker exec "${T4_CONTAINER_NAME}" python -c \
  'from person_search.config import Settings; from person_search.detector import YoloXOnnxDetector; detector = YoloXOnnxDetector(Settings()); detector.ensure_ready(); print(detector.provider_name); assert detector.provider_name == "CUDAExecutionProvider"'

if [[ "${T4_PRELOAD_INSIGHTFACE}" == "1" ]]; then
  log "Downloading and preloading the InsightFace model"
  docker exec "${T4_CONTAINER_NAME}" python -c \
    'from person_search.backends import InsightFaceBackend; from person_search.config import Settings; backend = InsightFaceBackend(Settings()); backend.ensure_ready(); print(backend.provider_name); assert backend.provider_name == "CUDAExecutionProvider"'
fi

log "Deployment completed"
log "Monitor: http://${T4_BIND_HOST}:8000/monitor"
if [[ "${T4_BIND_HOST}" == "127.0.0.1" ]]; then
  log "Open an SSH tunnel from your computer: ssh -L 8000:127.0.0.1:8000 user@t4-server"
fi
log "Follow logs: docker logs -f ${T4_CONTAINER_NAME}"
