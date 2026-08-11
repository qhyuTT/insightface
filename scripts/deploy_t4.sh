#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

T4_IMAGE_NAME="${T4_IMAGE_NAME:-person-search:t4}"
T4_CONTAINER_NAME="${T4_CONTAINER_NAME:-person-search}"
T4_MODEL_VOLUME="${T4_MODEL_VOLUME:-person-search-models}"
T4_CUDA_IMAGE="${T4_CUDA_IMAGE:-docker.m.daocloud.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
T4_PIP_INDEX_URL="${T4_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
T4_ORT_VERSION="${T4_ORT_VERSION:-1.23.2}"
T4_BIND_HOST="${T4_BIND_HOST:-127.0.0.1}"
T4_PRELOAD_INSIGHTFACE="${T4_PRELOAD_INSIGHTFACE:-0}"
T4_YOLOX_MODEL_URL="${T4_YOLOX_MODEL_URL:-}"

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
