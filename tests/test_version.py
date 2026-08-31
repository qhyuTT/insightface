from __future__ import annotations

import tomllib
from pathlib import Path

from person_search import API_VERSION, __version__
from person_search.api import create_app
from person_search.config import Settings


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
