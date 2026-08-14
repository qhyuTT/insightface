#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME="$(basename -- "$0")"
DOMAIN="gui/$(id -u)"
MEDIAMTX_LABEL="com.insightface.mediamtx"
CAMERA_LABEL="com.insightface.camera"
MEDIAMTX_JOB="${DOMAIN}/${MEDIAMTX_LABEL}"
CAMERA_JOB="${DOMAIN}/${CAMERA_LABEL}"

RTSP_PORT="${LOCAL_RTSP_PORT:-8554}"
RTSP_PATH="${LOCAL_RTSP_PATH:-camera}"
CAMERA_DEVICE="${LOCAL_RTSP_CAMERA_DEVICE:-0}"
VIDEO_SIZE="${LOCAL_RTSP_VIDEO_SIZE:-1280x720}"
FRAME_RATE="${LOCAL_RTSP_FRAME_RATE:-30}"
VIDEO_BITRATE="${LOCAL_RTSP_VIDEO_BITRATE:-2500k}"
START_TIMEOUT="${LOCAL_RTSP_START_TIMEOUT:-20}"
STATE_DIR="${LOCAL_RTSP_STATE_DIR:-/tmp/insightface-local-rtsp}"
MEDIAMTX_LOG="${STATE_DIR}/mediamtx.log"
CAMERA_STDOUT_LOG="${STATE_DIR}/camera.stdout.log"
CAMERA_STDERR_LOG="${STATE_DIR}/camera.stderr.log"
RTSP_URL="rtsp://127.0.0.1:${RTSP_PORT}/${RTSP_PATH}"

MEDIAMTX_BIN=""
MEDIAMTX_CONFIG=""
FFMPEG_BIN=""
FFPROBE_BIN=""
STARTED_MEDIAMTX=0
STARTED_CAMERA=0
ROLLBACK_ON_EXIT=0

log() {
  printf '[local-rtsp] %s\n' "$*"
}

warn() {
  printf '[local-rtsp] WARNING: %s\n' "$*" >&2
}

fail() {
  printf '[local-rtsp] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} <command>

Commands:
  start       Start MediaMTX and publish the Mac camera
  stop        Stop the camera publisher and MediaMTX
  restart     Stop and start the local RTSP stream
  status      Show jobs, listener, and stream health
  logs [-f]   Show logs; -f keeps following them
  help        Show this help

Environment overrides:
  LOCAL_RTSP_CAMERA_DEVICE      AVFoundation camera index/name (default: 0)
  LOCAL_RTSP_VIDEO_SIZE         Capture size (default: 1280x720)
  LOCAL_RTSP_FRAME_RATE         Capture FPS (default: 30)
  LOCAL_RTSP_VIDEO_BITRATE      H.264 bitrate (default: 2500k)
  LOCAL_RTSP_PORT               RTSP port (default: 8554)
  LOCAL_RTSP_PATH               RTSP path (default: camera)
  LOCAL_RTSP_START_TIMEOUT      Startup timeout in seconds (default: 20)
  LOCAL_RTSP_STATE_DIR          Log directory (default: /tmp/insightface-local-rtsp)
  LOCAL_RTSP_MEDIAMTX_BIN       Explicit MediaMTX executable
  LOCAL_RTSP_MEDIAMTX_CONFIG    Explicit MediaMTX config file
  LOCAL_RTSP_FFMPEG_BIN         Explicit FFmpeg executable
  LOCAL_RTSP_FFPROBE_BIN        Explicit FFprobe executable
EOF
}

require_macos() {
  [[ "$(uname -s)" == "Darwin" ]] || fail "this script supports macOS only"
}

validate_settings() {
  [[ "${RTSP_PORT}" =~ ^[0-9]+$ ]] || fail "LOCAL_RTSP_PORT must be an integer"
  ((RTSP_PORT >= 1 && RTSP_PORT <= 65535)) || fail "LOCAL_RTSP_PORT must be between 1 and 65535"
  [[ "${FRAME_RATE}" =~ ^[0-9]+$ ]] || fail "LOCAL_RTSP_FRAME_RATE must be a positive integer"
  ((FRAME_RATE > 0)) || fail "LOCAL_RTSP_FRAME_RATE must be greater than zero"
  [[ "${START_TIMEOUT}" =~ ^[0-9]+$ ]] || fail "LOCAL_RTSP_START_TIMEOUT must be a positive integer"
  ((START_TIMEOUT > 0)) || fail "LOCAL_RTSP_START_TIMEOUT must be greater than zero"
  [[ "${VIDEO_SIZE}" =~ ^[0-9]+x[0-9]+$ ]] || fail "LOCAL_RTSP_VIDEO_SIZE must look like 1280x720"
  [[ "${RTSP_PATH}" =~ ^[A-Za-z0-9._~-]+$ ]] || fail "LOCAL_RTSP_PATH contains unsupported characters"
}

resolve_executable() {
  local override="$1"
  local command_name="$2"
  local resolved=""

  if [[ -n "${override}" ]]; then
    [[ -x "${override}" ]] || fail "executable is missing or not executable: ${override}"
    printf '%s\n' "${override}"
    return
  fi

  resolved="$(command -v "${command_name}" 2>/dev/null || true)"
  [[ -n "${resolved}" ]] || fail "${command_name} was not found; install it with: brew install mediamtx ffmpeg"
  printf '%s\n' "${resolved}"
}

resolve_dependencies() {
  MEDIAMTX_BIN="$(resolve_executable "${LOCAL_RTSP_MEDIAMTX_BIN:-}" mediamtx)"
  FFMPEG_BIN="$(resolve_executable "${LOCAL_RTSP_FFMPEG_BIN:-}" ffmpeg)"
  FFPROBE_BIN="$(resolve_executable "${LOCAL_RTSP_FFPROBE_BIN:-}" ffprobe)"

  if ! "${FFMPEG_BIN}" -hide_banner -devices 2>/dev/null | grep -q 'avfoundation'; then
    fail "${FFMPEG_BIN} does not include AVFoundation camera support"
  fi

  resolve_mediamtx_config
}

resolve_ffprobe_if_available() {
  local override="${LOCAL_RTSP_FFPROBE_BIN:-}"
  if [[ -n "${override}" && -x "${override}" ]]; then
    FFPROBE_BIN="${override}"
  else
    FFPROBE_BIN="$(command -v ffprobe 2>/dev/null || true)"
  fi
}

resolve_mediamtx_config() {
  local explicit="${LOCAL_RTSP_MEDIAMTX_CONFIG:-}"
  local prefix=""
  local candidate=""
  local -a candidates=()

  if [[ -n "${explicit}" ]]; then
    [[ -f "${explicit}" ]] || fail "MediaMTX config does not exist: ${explicit}"
    MEDIAMTX_CONFIG="${explicit}"
    return
  fi

  prefix="$(cd -- "$(dirname -- "${MEDIAMTX_BIN}")/.." && pwd)"
  candidates=(
    "${prefix}/etc/mediamtx/mediamtx.yml"
    "${prefix}/etc/mediamtx.yml"
    "/usr/local/etc/mediamtx/mediamtx.yml"
    "/usr/local/etc/mediamtx.yml"
    "/opt/homebrew/etc/mediamtx/mediamtx.yml"
    "/opt/homebrew/etc/mediamtx.yml"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      MEDIAMTX_CONFIG="${candidate}"
      return
    fi
  done

  fail "MediaMTX config was not found; set LOCAL_RTSP_MEDIAMTX_CONFIG"
}

job_exists() {
  launchctl print "$1" >/dev/null 2>&1
}

job_running() {
  launchctl print "$1" 2>/dev/null | grep -q 'state = running'
}

job_pid() {
  launchctl print "$1" 2>/dev/null | awk '/^[[:space:]]*pid = [0-9]+$/ { print $3; exit }'
}

stop_job() {
  local job="$1"
  local name="$2"
  local deadline=0

  if ! job_exists "${job}"; then
    log "${name} is already stopped"
    return
  fi

  log "Stopping ${name}"
  if ! launchctl bootout "${job}" >/dev/null 2>&1; then
    fail "launchctl could not stop ${name} (${job})"
  fi

  deadline=$((SECONDS + 5))
  while job_exists "${job}"; do
    if ((SECONDS >= deadline)); then
      fail "timed out waiting for ${name} to stop"
    fi
    sleep 0.2
  done
}

listener_details() {
  /usr/sbin/lsof -nP -iTCP:"${RTSP_PORT}" -sTCP:LISTEN 2>/dev/null || true
}

port_is_listening() {
  [[ -n "$(listener_details)" ]]
}

port_owned_by_job() {
  local pid=""
  pid="$(job_pid "${MEDIAMTX_JOB}")"
  [[ -n "${pid}" ]] || return 1
  /usr/sbin/lsof -nP -t -a -p "${pid}" -iTCP:"${RTSP_PORT}" -sTCP:LISTEN 2>/dev/null |
    grep -qx "${pid}"
}

wait_for_job_and_port() {
  local deadline=$((SECONDS + START_TIMEOUT))

  while ((SECONDS < deadline)); do
    if job_running "${MEDIAMTX_JOB}" && port_owned_by_job; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

probe_stream() {
  [[ -n "${FFPROBE_BIN}" ]] || return 1
  "${FFPROBE_BIN}" \
    -v error \
    -rtsp_transport tcp \
    -rw_timeout 2000000 \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate \
    -of csv=p=0 \
    "${RTSP_URL}" 2>/dev/null
}

wait_for_stream() {
  local deadline=$((SECONDS + START_TIMEOUT))
  local stream_info=""

  while ((SECONDS < deadline)); do
    if ! job_running "${CAMERA_JOB}"; then
      sleep 0.5
      continue
    fi
    stream_info="$(probe_stream || true)"
    if [[ -n "${stream_info}" ]]; then
      printf '%s\n' "${stream_info}"
      return 0
    fi
    sleep 0.5
  done
  return 1
}

show_failure_logs() {
  local log_file=""
  for log_file in "${CAMERA_STDERR_LOG}" "${MEDIAMTX_LOG}"; do
    if [[ -s "${log_file}" ]]; then
      printf '\n--- %s (last 30 lines) ---\n' "${log_file}" >&2
      tail -n 30 "${log_file}" >&2 || true
    fi
  done
}

rollback_on_failure() {
  local exit_code=$?
  if ((exit_code != 0 && ROLLBACK_ON_EXIT == 1)); then
    warn "startup failed; rolling back jobs created by this invocation"
    if ((STARTED_CAMERA == 1)); then
      launchctl bootout "${CAMERA_JOB}" >/dev/null 2>&1 || true
    fi
    if ((STARTED_MEDIAMTX == 1)); then
      launchctl bootout "${MEDIAMTX_JOB}" >/dev/null 2>&1 || true
    fi
  fi
}

start_mediamtx() {
  local owner=""

  if job_running "${MEDIAMTX_JOB}"; then
    if ! port_owned_by_job; then
      fail "MediaMTX job is running but does not own port ${RTSP_PORT}; run '${SCRIPT_NAME} restart'"
    fi
    log "MediaMTX is already running"
    return
  fi

  if job_exists "${MEDIAMTX_JOB}"; then
    stop_job "${MEDIAMTX_JOB}" MediaMTX
  fi

  owner="$(listener_details)"
  if [[ -n "${owner}" ]]; then
    printf '%s\n' "${owner}" >&2
    fail "TCP port ${RTSP_PORT} is already occupied; refusing to stop an unmanaged process"
  fi

  : >"${MEDIAMTX_LOG}"
  log "Starting MediaMTX on port ${RTSP_PORT}"
  launchctl submit \
    -l "${MEDIAMTX_LABEL}" \
    -o "${MEDIAMTX_LOG}" \
    -e "${MEDIAMTX_LOG}" \
    -- /bin/sh -c \
    'cd "$1" && export MTX_RTSPADDRESS=":$4" && exec "$2" "$3"' \
    local-rtsp "${STATE_DIR}" "${MEDIAMTX_BIN}" "${MEDIAMTX_CONFIG}" "${RTSP_PORT}"
  STARTED_MEDIAMTX=1

  if ! wait_for_job_and_port; then
    show_failure_logs
    fail "MediaMTX did not listen on port ${RTSP_PORT} within ${START_TIMEOUT}s"
  fi
}

start_camera() {
  local keyframe_interval="${FRAME_RATE}"

  if job_running "${CAMERA_JOB}"; then
    log "camera publisher is already running"
    return
  fi

  if job_exists "${CAMERA_JOB}"; then
    stop_job "${CAMERA_JOB}" "camera publisher"
  fi

  : >"${CAMERA_STDOUT_LOG}"
  : >"${CAMERA_STDERR_LOG}"
  log "Starting camera ${CAMERA_DEVICE} at ${VIDEO_SIZE}/${FRAME_RATE}fps"
  launchctl submit \
    -l "${CAMERA_LABEL}" \
    -o "${CAMERA_STDOUT_LOG}" \
    -e "${CAMERA_STDERR_LOG}" \
    -- /usr/bin/nice -n 5 "${FFMPEG_BIN}" \
    -hide_banner -loglevel warning \
    -f avfoundation \
    -framerate "${FRAME_RATE}" \
    -video_size "${VIDEO_SIZE}" \
    -pixel_format nv12 \
    -i "${CAMERA_DEVICE}" \
    -an \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -pix_fmt yuv420p \
    -b:v "${VIDEO_BITRATE}" \
    -maxrate "${VIDEO_BITRATE}" \
    -bufsize 1250k \
    -g "${keyframe_interval}" \
    -keyint_min "${keyframe_interval}" \
    -sc_threshold 0 \
    -bf 0 \
    -rtsp_transport tcp \
    -pkt_size 1200 \
    -f rtsp "${RTSP_URL}"
  STARTED_CAMERA=1
}

print_addresses() {
  local lan_ip=""
  lan_ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
  log "Local stream: ${RTSP_URL}"
  if [[ -n "${lan_ip}" ]]; then
    log "LAN stream:   rtsp://${lan_ip}:${RTSP_PORT}/${RTSP_PATH}"
  fi
}

start_all() {
  local stream_info=""

  require_macos
  validate_settings
  resolve_dependencies
  mkdir -p "${STATE_DIR}"

  ROLLBACK_ON_EXIT=1
  trap rollback_on_failure EXIT

  start_mediamtx
  start_camera

  log "Waiting for a readable H.264 stream"
  stream_info="$(wait_for_stream || true)"
  if [[ -z "${stream_info}" ]]; then
    show_failure_logs
    fail "camera stream was not readable within ${START_TIMEOUT}s; check camera permission and device index"
  fi

  ROLLBACK_ON_EXIT=0
  log "Stream is ready (${stream_info})"
  print_addresses
}

stop_all() {
  require_macos
  validate_settings
  stop_job "${CAMERA_JOB}" "camera publisher"
  stop_job "${MEDIAMTX_JOB}" MediaMTX

  if port_is_listening; then
    warn "port ${RTSP_PORT} is still occupied by an unmanaged process:"
    listener_details >&2
    return 1
  fi
  log "Local RTSP stream is stopped"
}

show_status() {
  local healthy=1
  local stream_info=""

  require_macos
  validate_settings
  resolve_ffprobe_if_available

  if job_running "${MEDIAMTX_JOB}"; then
    log "MediaMTX: running"
  elif job_exists "${MEDIAMTX_JOB}"; then
    log "MediaMTX: loaded but not running"
    healthy=0
  else
    log "MediaMTX: stopped"
    healthy=0
  fi

  if job_running "${CAMERA_JOB}"; then
    log "Camera:   running"
  elif job_exists "${CAMERA_JOB}"; then
    log "Camera:   loaded but not running"
    healthy=0
  else
    log "Camera:   stopped"
    healthy=0
  fi

  if port_owned_by_job; then
    log "Port:     ${RTSP_PORT} is owned by MediaMTX"
  elif port_is_listening; then
    log "Port:     ${RTSP_PORT} is occupied by another process"
    healthy=0
  else
    log "Port:     ${RTSP_PORT} is closed"
    healthy=0
  fi

  if [[ -z "${FFPROBE_BIN}" ]]; then
    warn "ffprobe was not found; skipping stream probe"
    healthy=0
  else
    stream_info="$(probe_stream || true)"
    if [[ -n "${stream_info}" ]]; then
      log "Stream:   healthy (${stream_info})"
      print_addresses
    else
      log "Stream:   unavailable"
      healthy=0
    fi
  fi

  ((healthy == 1))
}

show_logs() {
  local follow="${1:-}"
  local -a tail_args=(-n 80)
  local -a files=()
  local file=""
  local existing=""
  local duplicate=0

  if [[ -n "${follow}" && "${follow}" != "-f" && "${follow}" != "--follow" ]]; then
    fail "logs accepts only -f or --follow"
  fi
  if [[ "${follow}" == "-f" || "${follow}" == "--follow" ]]; then
    tail_args+=(-f)
  fi

  for file in "${MEDIAMTX_LOG}" "${CAMERA_STDOUT_LOG}" "${CAMERA_STDERR_LOG}"; do
    [[ -f "${file}" ]] && files+=("${file}")
  done
  while IFS= read -r file; do
    [[ -f "${file}" ]] || continue
    duplicate=0
    for existing in ${files[@]+"${files[@]}"}; do
      if [[ "${existing}" == "${file}" ]]; then
        duplicate=1
        break
      fi
    done
    ((duplicate == 1)) || files+=("${file}")
  done < <(
    {
      launchctl print "${MEDIAMTX_JOB}" 2>/dev/null || true
      launchctl print "${CAMERA_JOB}" 2>/dev/null || true
    } | awk '/^[[:space:]]*(stdout|stderr) path = / { sub(/^[^=]*= /, ""); print }'
  )
  if ((${#files[@]} == 0)); then
    fail "no logs found in ${STATE_DIR}; start the stream first"
  fi
  tail "${tail_args[@]}" "${files[@]}"
}

command_name="${1:-help}"
shift || true

case "${command_name}" in
  start)
    (($# == 0)) || fail "start does not accept positional arguments"
    start_all
    ;;
  stop)
    (($# == 0)) || fail "stop does not accept positional arguments"
    stop_all
    ;;
  restart)
    (($# == 0)) || fail "restart does not accept positional arguments"
    stop_all
    start_all
    ;;
  status)
    (($# == 0)) || fail "status does not accept positional arguments"
    show_status
    ;;
  logs)
    (($# <= 1)) || fail "logs accepts at most one argument"
    show_logs "${1:-}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    fail "unknown command: ${command_name}"
    ;;
esac
