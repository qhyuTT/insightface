from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_t4.sh"
DOCKERFILE = Path(__file__).parents[1] / "Dockerfile"


def test_t4_deploy_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_t4_deploy_script_checks_candidate_before_switching() -> None:
    source = SCRIPT.read_text()

    candidate_start = source.index('start_container "${candidate_name}"')
    candidate_health = source.index('wait_for_health "${candidate_name}"', candidate_start)
    candidate_readiness = source.index('verify_readiness "${candidate_name}"', candidate_health)
    candidate_stop = source.index(
        'docker stop --time "${T4_STOP_TIMEOUT_SECONDS}" "${candidate_name}"',
        candidate_readiness,
    )
    candidate_remove = source.index('docker rm "${candidate_name}"', candidate_stop)
    old_stop = source.index(
        'docker stop --time "${T4_STOP_TIMEOUT_SECONDS}" "${T4_CONTAINER_NAME}"',
        candidate_remove,
    )
    assert candidate_start < candidate_health < candidate_readiness < candidate_stop
    assert candidate_stop < candidate_remove < old_stop
    assert 'start_container "${candidate_name}" "${T4_CANARY_PORT}" no 127.0.0.1' in source
    assert "--stop-timeout \"${T4_STOP_TIMEOUT_SECONDS}\"" in source
    assert "docker rename \"${T4_CONTAINER_NAME}\" \"${previous_name}\"" in source
    assert "org.opencontainers.image.revision" in source


def test_t4_deploy_script_cleans_failed_replacement_and_verifies_rollback() -> None:
    source = SCRIPT.read_text()
    cleanup = (
        'if [[ "${replacement_started}" == "1" ]] '
        '&& container_exists "${T4_CONTAINER_NAME}"; then'
    )
    assert cleanup in source
    replacement_flag = source.index("replacement_started=1", source.index("Starting replacement"))
    replacement_start = source.index('start_container "${T4_CONTAINER_NAME}"', replacement_flag)
    assert replacement_flag < replacement_start
    assert 'wait_for_rollback "${T4_CONTAINER_NAME}" "${previous_port}"' in source


def test_docker_revision_label_preserves_cache_and_health_uses_runtime_port() -> None:
    source = DOCKERFILE.read_text()
    label = source.index("LABEL org.opencontainers.image.title")
    dependency_install = source.index("uv sync --frozen")
    model_setup = source.index("RUN mkdir -p /models/.insightface")
    assert dependency_install < model_setup < label
    assert "ENV PERSON_SEARCH_BUILD_REVISION=${VCS_REF}" in source
    # The port reference must reach the runtime shell verbatim, and all three
    # ways of "protecting" it break it. Docker expands variables only in ADD /
    # COPY / ENV / EXPOSE / FROM / LABEL / STOPSIGNAL / USER / VOLUME / WORKDIR
    # -- never in HEALTHCHECK -- so a plain ${...} is already stored unexpanded
    # and needs no escape at all. Compose's ``$$`` reaches curl as a literal
    # dollar, and ``\$`` keeps its backslash in the stored command, which sh
    # reads as a literal dollar too: measured in the container, the plain form
    # exits 0 while the escaped form exits 3 on a malformed URL, leaving Docker
    # reporting the container unhealthy while the API itself serves fine.
    # ``CMD-SHELL`` is Compose's spelling of the shell form, which the Dockerfile
    # parser rejects outright -- that one fails the build, not the deployment.
    assert "CMD-SHELL" not in source
    assert "$$" not in source
    assert r"\${PERSON_SEARCH_PORT" not in source
    assert (
        'CMD curl --fail --silent "http://127.0.0.1:${PERSON_SEARCH_PORT:-8000}/healthz"'
    ) in source


def test_t4_deploy_script_defaults_tiny_face_to_shadow_mode() -> None:
    source = SCRIPT.read_text()
    assert 'T4_TINY_FACE_SHADOW_MODE="${T4_TINY_FACE_SHADOW_MODE:-true}"' in source
    assert "T4_ALLOW_PHYSICAL_ACTIONS=true" in source


def test_t4_deploy_script_normalizes_switches_and_bounds_runtime_values() -> None:
    source = SCRIPT.read_text()
    assert "normalize_bool" in source
    assert 'T4_PREFETCH_YOLOX="$(normalize_bool' in source
    assert 'if [[ "${T4_PREFETCH_YOLOX}" == "true" ]]' in source
    assert "T4_PRELOAD_INSIGHTFACE" not in source
    assert "require_port T4_PORT" in source
    assert "require_port T4_CANARY_PORT" in source
    assert "<= 65535" in source
    assert "require_grid_size T4_FACE_DETECTION_SIZE" in source
    assert "require_version T4_ORT_VERSION" in source


def test_t4_readiness_gate_checks_versions_revision_contract_and_all_cuda_providers() -> None:
    source = SCRIPT.read_text()
    readiness = source[source.index("verify_readiness() {") : source.index("wait_for_rollback() {")]
    assert "/readyz" in readiness
    assert 'payload.get("package_version") == "0.1.0"' in readiness
    assert 'payload.get("api_version") == "0.2.0"' in readiness
    assert 'payload.get("build_revision") == expected_revision' in readiness
    assert 'contract.get("contract_id")' in readiness
    assert '"person_detection", "face_detection", "face_recognition"' in readiness
    assert 'providers[name] == "CUDAExecutionProvider"' in readiness

    candidate_gate = source.index('verify_readiness "${candidate_name}"')
    old_stop = source.index(
        'docker stop --time "${T4_STOP_TIMEOUT_SECONDS}" "${T4_CONTAINER_NAME}"'
    )
    replacement_gate = source.index('verify_readiness "${T4_CONTAINER_NAME}"')
    assert candidate_gate < old_stop < replacement_gate


def test_t4_deploy_script_redacts_remote_credentials_in_logs() -> None:
    source = SCRIPT.read_text()
    assert "sanitize_remote()" in source
    assert 'SOURCE_REMOTE_DISPLAY="$(sanitize_remote' in source
    # The raw remote is still used for the exact expected-origin comparison;
    # only the user-facing message is sanitized.
    assert '[[ "${SOURCE_REMOTE}" == "${T4_EXPECTED_REMOTE_URL}" ]]' in source
    assert "<redacted>" in source

    start = source.index("sanitize_remote() {")
    end = source.index("\n}", start) + 2
    function = source[start:end]

    def sanitize(value: str) -> str:
        result = subprocess.run(
            ["bash", "-c", f'{function}\nsanitize_remote "$1"', "bash", value],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert sanitize("https://user:token@example.test/org/repo.git") == (
        "https://<redacted>@example.test/org/repo.git"
    )
    assert sanitize("https://user:p@ss@example.test/org/repo.git") == (
        "https://<redacted>@example.test/org/repo.git"
    )
    assert sanitize("git@example.test:org/repo.git") == (
        "<redacted>@example.test:org/repo.git"
    )
    assert sanitize("https://example.test/org/repo.git") == (
        "https://example.test/org/repo.git"
    )

