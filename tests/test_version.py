from __future__ import annotations

import re
import tomllib
from pathlib import Path

from person_search import API_VERSION, __version__
from person_search.api import create_app
from person_search.config import Settings
from person_search.model_assets import (
    BUFFALO_L_EMBEDDING_MANIFEST,
    YOLOX_TINY_SHA256,
    YOLOX_TINY_URL,
)


def test_package_version_and_api_version_are_explicitly_distinct() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project_version = tomllib.load(stream)["project"]["version"]
    with (root / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    lock_version = next(
        package["version"]
        for package in lock["package"]
        if package["name"] == "robot-person-search-poc"
    )

    # The API contract can evolve independently of the package release.  Keep
    # the packaging metadata and ``__version__`` aligned with one another while
    # FastAPI/health expose the explicitly versioned HTTP contract.
    assert __version__ == "0.1.0"
    assert project_version == __version__
    assert lock_version == __version__
    assert API_VERSION == "0.2.0"


def test_fastapi_metadata_uses_the_same_api_version() -> None:
    class Manager:
        def active_search(self):
            return None

        def shutdown(self):
            pass

    app = create_app(Settings(), Manager())  # type: ignore[arg-type]
    assert app.version == API_VERSION


def test_build_revision_defaults_to_unknown_and_rejects_whitespace() -> None:
    assert Settings().build_revision == "unknown"

    try:
        Settings(build_revision="not a revision")
    except ValueError:
        pass
    else:
        raise AssertionError("build_revision containing whitespace was accepted")


def test_runtime_dependency_and_model_constants_stay_aligned() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text()
    deploy = (root / "scripts" / "deploy_t4.sh").read_text()
    with (root / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)

    docker_ort = re.search(r"ARG ONNXRUNTIME_GPU_VERSION=([^\s]+)", dockerfile)
    deploy_ort = re.search(r'T4_ORT_VERSION="\$\{T4_ORT_VERSION:-([^}]+)\}"', deploy)
    locked_ort = next(
        package["version"] for package in lock["package"] if package["name"] == "onnxruntime"
    )
    assert docker_ort is not None and deploy_ort is not None
    assert docker_ort.group(1) == deploy_ort.group(1) == locked_ort

    hashes = {
        re.search(r"ARG YOLOX_MODEL_SHA256=([0-9a-f]{64})", dockerfile).group(1),  # type: ignore[union-attr]
        re.search(r'YOLOX_SHA256="([0-9a-f]{64})"', deploy).group(1),  # type: ignore[union-attr]
        YOLOX_TINY_SHA256,
    }
    urls = {
        re.search(r"ARG YOLOX_MODEL_URL=([^\s]+)", dockerfile).group(1),  # type: ignore[union-attr]
        re.search(r'YOLOX_UPSTREAM_URL="([^"]+)"', deploy).group(1),  # type: ignore[union-attr]
    }
    assert len(hashes) == 1
    assert len(urls | {YOLOX_TINY_URL}) == 1


def test_docker_build_revision_label_and_runtime_value_share_vcs_ref() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text()
    assert "ARG VCS_REF=unknown" in dockerfile
    assert "ENV PERSON_SEARCH_BUILD_REVISION=${VCS_REF}" in dockerfile
    assert 'org.opencontainers.image.revision="${VCS_REF}"' in dockerfile


def test_buffalo_l_embedding_manifest_pins_the_production_contract() -> None:
    manifest = BUFFALO_L_EMBEDDING_MANIFEST

    assert manifest.model_name == "buffalo_l"
    assert manifest.recognition_filename == "w600k_r50.onnx"
    assert manifest.recognition_sha256 == (
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
    )
    assert manifest.embedding_dimension == 512
    assert manifest.input_size == (112, 112)
