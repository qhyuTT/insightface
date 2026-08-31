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
T4_PORT="${T4_PORT:-8000}"
# A candidate uses a separate host-network port so the currently serving
# container can stay up while the new image is loaded and checked.
T4_CANARY_PORT="${T4_CANARY_PORT:-18000}"
T4_STOP_TIMEOUT_SECONDS="${T4_STOP_TIMEOUT_SECONDS:-30}"
# Deployments are expected to come from a reproducible, committed checkout.
# Set T4_ALLOW_DIRTY=1 only for an explicitly marked local experiment.
T4_ALLOW_DIRTY="${T4_ALLOW_DIRTY:-0}"
T4_EXPECTED_BRANCH="${T4_EXPECTED_BRANCH:-main}"
T4_EXPECTED_COMMIT="${T4_EXPECTED_COMMIT:-}"
T4_EXPECTED_REMOTE_URL="${T4_EXPECTED_REMOTE_URL:-}"
# Tiny-face confirmation is diagnostic-only until its calibration has been
# accepted. Two explicit switches are required before it can drive actions.
T4_TINY_FACE_ENABLED="${T4_TINY_FACE_ENABLED:-true}"
T4_TINY_FACE_SHADOW_MODE="${T4_TINY_FACE_SHADOW_MODE:-true}"
T4_ALLOW_PHYSICAL_ACTIONS="${T4_ALLOW_PHYSICAL_ACTIONS:-false}"
# Optional ORT controls are passed through when set; leaving them empty keeps
# ONNX Runtime's native defaults.
T4_ORT_INTRA_OP_NUM_THREADS="${T4_ORT_INTRA_OP_NUM_THREADS:-}"
T4_ORT_INTER_OP_NUM_THREADS="${T4_ORT_INTER_OP_NUM_THREADS:-}"
T4_ORT_CUDA_DEVICE_ID="${T4_ORT_CUDA_DEVICE_ID:-}"
T4_HEALTH_ATTEMPTS="${T4_HEALTH_ATTEMPTS:-30}"
T4_HEALTH_INTERVAL_SECONDS="${T4_HEALTH_INTERVAL_SECONDS:-2}"
# Enables GET /v1/searches/{id}/evidence/{id}. Must match the dispatch platform's
# DISPATCH_INSIGHTFACE_EVIDENCE_API_KEY, or every confirmed hit loses its face crop.
T4_EVIDENCE_API_KEY="${T4_EVIDENCE_API_KEY:-}"
T4_PRELOAD_INSIGHTFACE="${T4_PRELOAD_INSIGHTFACE:-0}"
# conservative keeps the calibrated far-face bar; responsive trades false-accept
# headroom for a faster verdict and should wait for a calibrated threshold.
# transit additionally shortens the far-face window and the gap between samples,
# for a hall where the subject walks past instead of lingering. None of them move a
# similarity threshold: that needs a measured distribution, not a scene name.
T4_MATCH_PROFILE="${T4_MATCH_PROFILE:-conservative}"
# Judge a track once on its way out when it ran out of frames rather than out of
# similarity, and report it on the shadow channel (tiny_shadow_confirmed) as a lead
# rather than as a production confirmation. Off unless the operator asks for it:
# it is a deliberate move toward false accepts.
T4_DEPARTURE_ADJUDICATION="${T4_DEPARTURE_ADJUDICATION:-false}"
# One 1280 pass rather than Auto's 128+640: measured on a T4 it is both cheaper
# (57ms against 108ms) and better at every face size, because each extra ONNX input
# shape costs ~30ms of re-planning -- far more than the scale's own inference. The
# extra-scale mechanism stays available but has nothing to add on top of 1280.
T4_FACE_DETECTION_SIZE="${T4_FACE_DETECTION_SIZE:-1280}"
T4_FACE_DETECTION_EXTRA_SCALE="${T4_FACE_DETECTION_EXTRA_SCALE:-0}"
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

normalize_bool() {
  local normalized
  normalized="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in
    1|true|yes) printf 'true' ;;
    0|false|no) printf 'false' ;;
    *) fail "${2}: expected true/false (or 1/0), got '${1}'" ;;
  esac
}

require_positive_integer() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a positive integer"
  (( 10#${value} > 0 )) || fail "${name} must be a positive integer"
}

require_port() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be an integer from 1 to 65535"
  (( 10#${value} >= 1 && 10#${value} <= 65535 )) || fail \
    "${name} must be an integer from 1 to 65535"
}

require_grid_size() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a non-negative multiple of 32"
  (( 10#${value} == 0 || 10#${value} % 32 == 0 )) || fail \
    "${name} must be a non-negative multiple of 32"
}

require_version() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+(\.[0-9]+){2}([.-][0-9A-Za-z]+)*$ ]] || fail \
    "${name} must be a semantic version such as 1.23.2"
}

require_nonnegative_integer() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a non-negative integer"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

sanitize_remote() {
  # Git remotes are normally credential-free, but an operator may have used a
  # temporary HTTPS token or an ``user@host:path`` SSH form.  Keep the raw value
  # for exact comparisons while redacting the user-info portion in logs/errors.
  printf '%s' "$1" | sed -E \
    -e 's#(://)[^/]*@#\1<redacted>@#g' \
    -e 's#^[^/@[:space:]]+@#<redacted>@#'
}

# Normalize all boolean switches before any conditional uses them.  In
# particular, ``true``/``false`` should behave the same as ``1``/``0`` for the
# prefetch and preload paths; silently skipping an explicitly enabled check is a
# deployment footgun.
T4_ALLOW_DIRTY="$(normalize_bool "${T4_ALLOW_DIRTY}" T4_ALLOW_DIRTY)"
T4_TINY_FACE_ENABLED="$(normalize_bool "${T4_TINY_FACE_ENABLED}" T4_TINY_FACE_ENABLED)"
T4_TINY_FACE_SHADOW_MODE="$(normalize_bool "${T4_TINY_FACE_SHADOW_MODE}" T4_TINY_FACE_SHADOW_MODE)"
T4_ALLOW_PHYSICAL_ACTIONS="$(normalize_bool "${T4_ALLOW_PHYSICAL_ACTIONS}" T4_ALLOW_PHYSICAL_ACTIONS)"
T4_PREFETCH_YOLOX="$(normalize_bool "${T4_PREFETCH_YOLOX}" T4_PREFETCH_YOLOX)"
T4_PRELOAD_INSIGHTFACE="$(normalize_bool "${T4_PRELOAD_INSIGHTFACE}" T4_PRELOAD_INSIGHTFACE)"
T4_DEPARTURE_ADJUDICATION="$(normalize_bool "${T4_DEPARTURE_ADJUDICATION}" T4_DEPARTURE_ADJUDICATION)"

require_command docker
require_command nvidia-smi
require_command git

docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable for the current user"
nvidia-smi -L >/dev/null 2>&1 || fail "NVIDIA driver cannot see the T4 GPU"

cd "${PROJECT_ROOT}"

# Print and verify the exact source that is about to be built. This catches a
# copied release directory whose .git remote still points at an older checkout.
SOURCE_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null)" || fail "not a git checkout"
SOURCE_BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || printf 'detached')"
SOURCE_REMOTE="$(git remote get-url origin 2>/dev/null || true)"
SOURCE_REMOTE_DISPLAY="$(sanitize_remote "${SOURCE_REMOTE}")"
SOURCE_DIRTY="$(git status --porcelain --untracked-files=all)"
log "Source commit: ${SOURCE_COMMIT}"
log "Source branch: ${SOURCE_BRANCH}"
if [[ -n "${SOURCE_REMOTE}" ]]; then
  log "Source origin: ${SOURCE_REMOTE_DISPLAY}"
else
  log "Source origin: <none>"
fi
if [[ -n "${T4_EXPECTED_COMMIT}" ]]; then
  [[ "${SOURCE_COMMIT}" == "${T4_EXPECTED_COMMIT}"* ]] || fail \
    "HEAD ${SOURCE_COMMIT} does not match T4_EXPECTED_COMMIT=${T4_EXPECTED_COMMIT}"
fi
if [[ -n "${T4_EXPECTED_BRANCH}" ]]; then
  [[ "${SOURCE_BRANCH}" == "${T4_EXPECTED_BRANCH}" ]] || fail \
    "branch ${SOURCE_BRANCH} does not match T4_EXPECTED_BRANCH=${T4_EXPECTED_BRANCH}"
fi
if [[ -n "${T4_EXPECTED_REMOTE_URL}" ]]; then
  [[ "${SOURCE_REMOTE}" == "${T4_EXPECTED_REMOTE_URL}" ]] || fail \
    "origin ${SOURCE_REMOTE_DISPLAY:-<none>} does not match T4_EXPECTED_REMOTE_URL=$(sanitize_remote "${T4_EXPECTED_REMOTE_URL}")"
fi
if [[ -n "${SOURCE_DIRTY}" && "$(normalize_bool "${T4_ALLOW_DIRTY}" T4_ALLOW_DIRTY)" != "true" ]]; then
  fail "working tree is dirty; commit/stash changes or set T4_ALLOW_DIRTY=1 for an explicit experiment"
fi

require_port T4_PORT "${T4_PORT}"
require_port T4_CANARY_PORT "${T4_CANARY_PORT}"
require_positive_integer T4_STOP_TIMEOUT_SECONDS "${T4_STOP_TIMEOUT_SECONDS}"
require_positive_integer T4_HEALTH_ATTEMPTS "${T4_HEALTH_ATTEMPTS}"
require_positive_integer T4_HEALTH_INTERVAL_SECONDS "${T4_HEALTH_INTERVAL_SECONDS}"
[[ "${T4_PORT}" != "${T4_CANARY_PORT}" ]] || fail "T4_CANARY_PORT must differ from T4_PORT"
[[ -n "${T4_BIND_HOST}" && "${T4_BIND_HOST}" != *[[:space:]]* ]] || fail \
  "T4_BIND_HOST must be a non-empty host without whitespace"
require_version T4_ORT_VERSION "${T4_ORT_VERSION}"
require_grid_size T4_FACE_DETECTION_SIZE "${T4_FACE_DETECTION_SIZE}"
require_grid_size T4_FACE_DETECTION_EXTRA_SCALE "${T4_FACE_DETECTION_EXTRA_SCALE}"
case "${T4_MATCH_PROFILE}" in
  conservative|responsive|transit) ;;
  *) fail "T4_MATCH_PROFILE must be conservative, responsive, or transit" ;;
esac
if [[ "${T4_TINY_FACE_ENABLED}" == "true" \
  && "${T4_TINY_FACE_SHADOW_MODE}" != "true" \
  && "${T4_ALLOW_PHYSICAL_ACTIONS}" != "true" ]]; then
  fail "tiny-face production confirmation requires T4_ALLOW_PHYSICAL_ACTIONS=true"
fi
if [[ -n "${T4_ORT_INTRA_OP_NUM_THREADS}" ]]; then
  require_nonnegative_integer T4_ORT_INTRA_OP_NUM_THREADS "${T4_ORT_INTRA_OP_NUM_THREADS}"
fi
if [[ -n "${T4_ORT_INTER_OP_NUM_THREADS}" ]]; then
  require_nonnegative_integer T4_ORT_INTER_OP_NUM_THREADS "${T4_ORT_INTER_OP_NUM_THREADS}"
fi
if [[ -n "${T4_ORT_CUDA_DEVICE_ID}" ]]; then
  require_nonnegative_integer T4_ORT_CUDA_DEVICE_ID "${T4_ORT_CUDA_DEVICE_ID}"
fi

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

if [[ "${T4_PREFETCH_YOLOX}" == "true" ]]; then
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
  --build-arg "VCS_REF=${SOURCE_COMMIT}"
)
if [[ -n "${T4_YOLOX_MODEL_URL}" ]]; then
  build_args+=(--build-arg "YOLOX_MODEL_URL=${T4_YOLOX_MODEL_URL}")
fi

log "Building ${T4_IMAGE_NAME}"
docker build "${build_args[@]}" --tag "${T4_IMAGE_NAME}" .

image_revision="$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
  "${T4_IMAGE_NAME}" 2>/dev/null || true)"
[[ "${image_revision}" == "${SOURCE_COMMIT}" ]] || fail \
  "image revision ${image_revision:-<missing>} does not match source ${SOURCE_COMMIT}"

log "Checking NVIDIA Container Toolkit"
docker run --rm --gpus all --entrypoint nvidia-smi "${T4_IMAGE_NAME}" >/dev/null

docker volume create "${T4_MODEL_VOLUME}" >/dev/null

if [[ -z "${T4_EVIDENCE_API_KEY}" ]]; then
  log "WARNING: T4_EVIDENCE_API_KEY is empty; the confirmed-hit evidence endpoint will return 503"
fi

# Keep environment construction in one place. The candidate deliberately binds
# loopback only; production uses T4_BIND_HOST after the candidate passes checks.
container_env_args() {
  local port="$1" bind_host="${2:-${T4_BIND_HOST}}"
  CONTAINER_ENV_ARGS=(
    --env "PERSON_SEARCH_HOST=${bind_host}"
    --env "PERSON_SEARCH_PORT=${port}"
    --env "PERSON_SEARCH_PREFER_CUDA=true"
    --env "PERSON_SEARCH_TINY_FACE_ENABLED=${T4_TINY_FACE_ENABLED}"
    --env "PERSON_SEARCH_TINY_FACE_SHADOW_MODE=${T4_TINY_FACE_SHADOW_MODE}"
    --env "PERSON_SEARCH_FACE_DETECTION_SIZE=${T4_FACE_DETECTION_SIZE}"
    --env "PERSON_SEARCH_FACE_DETECTION_EXTRA_SCALE_CUDA=${T4_FACE_DETECTION_EXTRA_SCALE}"
    --env "PERSON_SEARCH_MATCH_PROFILE=${T4_MATCH_PROFILE}"
    --env "PERSON_SEARCH_DEPARTURE_ADJUDICATION_ENABLED=${T4_DEPARTURE_ADJUDICATION}"
    --env "PERSON_SEARCH_EVIDENCE_API_KEY=${T4_EVIDENCE_API_KEY}"
  )
  if [[ -n "${T4_ORT_INTRA_OP_NUM_THREADS}" ]]; then
    CONTAINER_ENV_ARGS+=(--env "PERSON_SEARCH_ORT_INTRA_OP_NUM_THREADS=${T4_ORT_INTRA_OP_NUM_THREADS}")
  fi
  if [[ -n "${T4_ORT_INTER_OP_NUM_THREADS}" ]]; then
    CONTAINER_ENV_ARGS+=(--env "PERSON_SEARCH_ORT_INTER_OP_NUM_THREADS=${T4_ORT_INTER_OP_NUM_THREADS}")
  fi
  if [[ -n "${T4_ORT_CUDA_DEVICE_ID}" ]]; then
    CONTAINER_ENV_ARGS+=(--env "PERSON_SEARCH_ORT_CUDA_DEVICE_ID=${T4_ORT_CUDA_DEVICE_ID}")
  fi
}

start_container() {
  local name="$1" port="$2" restart_policy="$3" bind_host="${4:-${T4_BIND_HOST}}"
  container_env_args "${port}" "${bind_host}"
  docker run --detach \
    --gpus all \
    --restart "${restart_policy}" \
    --stop-timeout "${T4_STOP_TIMEOUT_SECONDS}" \
    --network host \
    --name "${name}" \
    --volume "${T4_MODEL_VOLUME}:/models" \
    "${CONTAINER_ENV_ARGS[@]}" \
    "${T4_IMAGE_NAME}" >/dev/null
}

container_exists() {
  docker container inspect "$1" >/dev/null 2>&1
}

wait_for_health() {
  local name="$1" port="$2" label="$3" ready=0 running
  log "Waiting for ${label} API health check"
  for _ in $(seq 1 "${T4_HEALTH_ATTEMPTS}"); do
    running="$(docker inspect --format '{{.State.Running}}' "${name}" 2>/dev/null || true)"
    if [[ "${running}" != "true" ]]; then
      break
    fi
    if docker exec "${name}" python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/healthz', timeout=2).read()" \
      >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep "${T4_HEALTH_INTERVAL_SECONDS}"
  done
  if [[ "${ready}" != "1" ]]; then
    docker logs --tail 100 "${name}" >&2 || true
    fail "${label} API did not become healthy within $((10#${T4_HEALTH_ATTEMPTS} * 10#${T4_HEALTH_INTERVAL_SECONDS})) seconds"
  fi
}

verify_provider() {
  local name="$1"
  log "Verifying CUDAExecutionProvider with the YOLOX model (${name})"
  docker exec "${name}" python -c \
    'from person_search.config import Settings; from person_search.detector import YoloXOnnxDetector; detector = YoloXOnnxDetector(Settings()); detector.ensure_ready(); print(detector.provider_name); assert detector.provider_name == "CUDAExecutionProvider"'
}

verify_provider_best_effort() {
  local name="$1"
  docker exec "${name}" python -c \
    'from person_search.config import Settings; from person_search.detector import YoloXOnnxDetector; detector = YoloXOnnxDetector(Settings()); detector.ensure_ready(); assert detector.provider_name == "CUDAExecutionProvider"' \
    >/dev/null 2>&1
}

wait_for_rollback() {
  local name="$1" port="$2" ready=0 running
  for _ in $(seq 1 "${T4_HEALTH_ATTEMPTS}"); do
    running="$(docker inspect --format '{{.State.Running}}' "${name}" 2>/dev/null || true)"
    if [[ "${running}" != "true" ]]; then
      break
    fi
    if docker exec "${name}" python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/healthz', timeout=2).read()" \
      >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep "${T4_HEALTH_INTERVAL_SECONDS}"
  done
  if [[ "${ready}" != "1" ]]; then
    return 1
  fi
  verify_provider_best_effort "${name}"
}

if [[ "${T4_PRELOAD_INSIGHTFACE}" == "true" ]]; then
  verify_insightface() {
    local name="$1"
    log "Downloading and preloading the InsightFace model (${name})"
    docker exec "${name}" python -c \
      'from person_search.backends import InsightFaceBackend; from person_search.config import Settings; backend = InsightFaceBackend(Settings()); backend.ensure_ready(); print(backend.provider_name); assert backend.provider_name == "CUDAExecutionProvider"'
  }
else
  verify_insightface() { :; }
fi

# A candidate is loaded on an alternate host port while the old service remains
# untouched. Any failure before the switch removes only the candidate.
candidate_name="${T4_CONTAINER_NAME}.candidate.$$"
previous_name="${T4_CONTAINER_NAME}.previous.$(date +%Y%m%d%H%M%S).$$"
existing_container_id="$(docker ps --all --quiet --filter "name=^/${T4_CONTAINER_NAME}$")"
previous_port="${T4_PORT}"
existing_state=""
if [[ -n "${existing_container_id}" ]]; then
  existing_state="$(docker inspect --format '{{.State.Running}}' "${T4_CONTAINER_NAME}" 2>/dev/null || true)"
  previous_port="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' \
    "${T4_CONTAINER_NAME}" 2>/dev/null | awk -F= '$1 == "PERSON_SEARCH_PORT" { print $2; exit }' || true)"
  [[ -n "${previous_port}" ]] || previous_port="${T4_PORT}"
  # The value is later interpolated into the in-container health probe.  Refuse
  # malformed/oversized values before any stop/rename operation so a stale
  # container cannot turn rollback into a broken or ambiguous check.
  require_port PERSON_SEARCH_PORT "${previous_port}"
  [[ "${existing_state}" != "true" || "${previous_port}" != "${T4_CANARY_PORT}" ]] || fail \
    "T4_CANARY_PORT conflicts with the existing container's PERSON_SEARCH_PORT"
fi
old_was_running=0
old_stopped=0
old_renamed=0
candidate_started=0
replacement_started=0
deployment_ok=0

rollback_on_exit() {
  local status=$?
  set +e
  if [[ "${deployment_ok}" != "1" ]]; then
    if [[ "${replacement_started}" == "1" ]] && container_exists "${T4_CONTAINER_NAME}"; then
      log "Removing failed replacement ${T4_CONTAINER_NAME}"
      docker rm --force "${T4_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
    if [[ "${old_renamed}" == "1" ]] && container_exists "${previous_name}"; then
      log "Restoring previous container ${previous_name} -> ${T4_CONTAINER_NAME}"
      docker rename "${previous_name}" "${T4_CONTAINER_NAME}" >/dev/null 2>&1 || true
      if [[ "${old_was_running}" == "1" ]]; then
        docker start "${T4_CONTAINER_NAME}" >/dev/null 2>&1 || true
        if wait_for_rollback "${T4_CONTAINER_NAME}" "${previous_port}"; then
          log "Previous container passed rollback health/provider checks"
        else
          log "ERROR: previous container was restored but failed rollback health/provider checks" >&2
          [[ "${status}" -eq 0 ]] && status=1
        fi
      fi
    elif [[ "${old_stopped}" == "1" ]] && container_exists "${T4_CONTAINER_NAME}"; then
      docker start "${T4_CONTAINER_NAME}" >/dev/null 2>&1 || true
      if [[ "${old_was_running}" == "1" ]]; then
        if wait_for_rollback "${T4_CONTAINER_NAME}" "${previous_port}"; then
          log "Previous container passed rollback health/provider checks"
        else
          log "ERROR: previous container was restored but failed rollback health/provider checks" >&2
          [[ "${status}" -eq 0 ]] && status=1
        fi
      fi
    fi
  fi
  if [[ "${candidate_started}" == "1" ]] && container_exists "${candidate_name}"; then
    docker rm --force "${candidate_name}" >/dev/null 2>&1 || true
  fi
  if [[ "${status}" -ne 0 && "${old_renamed}" == "1" ]]; then
    log "Deployment failed; previous container was restored when possible"
  fi
  exit "${status}"
}
trap rollback_on_exit EXIT

log "Starting candidate ${candidate_name} on host port ${T4_CANARY_PORT}"
candidate_started=1
start_container "${candidate_name}" "${T4_CANARY_PORT}" no 127.0.0.1
wait_for_health "${candidate_name}" "${T4_CANARY_PORT}" candidate
verify_provider "${candidate_name}"
verify_insightface "${candidate_name}"

# The candidate has completed all checks; stop it before starting the production
# replacement so two model copies do not compete for T4 VRAM during the switch.
log "Stopping validated candidate ${candidate_name}"
docker stop --time "${T4_STOP_TIMEOUT_SECONDS}" "${candidate_name}" >/dev/null
docker rm "${candidate_name}" >/dev/null
candidate_started=0

# Switch only after every candidate check passes. Keep the old container under a
# timestamped name so a failed final start/health check can be rolled back.
if [[ -n "${existing_container_id}" ]]; then
  old_state="${existing_state}"
  [[ "${old_state}" == "true" ]] && old_was_running=1
  if [[ "${old_was_running}" == "1" ]]; then
    log "Stopping current container ${T4_CONTAINER_NAME} (timeout ${T4_STOP_TIMEOUT_SECONDS}s)"
    # Mark this before the command: if Docker returns an error after stopping the
    # process, the EXIT trap can still attempt an idempotent restart.
    old_stopped=1
    docker stop --time "${T4_STOP_TIMEOUT_SECONDS}" "${T4_CONTAINER_NAME}" >/dev/null
  fi
  log "Keeping previous container as ${previous_name}"
  docker rename "${T4_CONTAINER_NAME}" "${previous_name}" >/dev/null
  old_renamed=1
fi

log "Starting replacement ${T4_CONTAINER_NAME} on host port ${T4_PORT}"
replacement_started=1
start_container "${T4_CONTAINER_NAME}" "${T4_PORT}" unless-stopped
wait_for_health "${T4_CONTAINER_NAME}" "${T4_PORT}" replacement
verify_provider "${T4_CONTAINER_NAME}"

deployment_ok=1

log "Deployment completed (source ${SOURCE_COMMIT})"
log "Monitor: http://${T4_BIND_HOST}:${T4_PORT}/monitor"
if [[ "${T4_BIND_HOST}" == "127.0.0.1" ]]; then
  log "Open an SSH tunnel from your computer: ssh -L ${T4_PORT}:127.0.0.1:${T4_PORT} user@t4-server"
fi
log "Follow logs: docker logs -f ${T4_CONTAINER_NAME}"
if [[ "${old_renamed}" == "1" ]]; then
  log "Rollback point retained as ${previous_name}"
fi
