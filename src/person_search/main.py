from __future__ import annotations

import uvicorn

from .api import create_app
from .config import get_settings
from .privacy import uvicorn_log_config

app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "person_search.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        log_config=uvicorn_log_config(),
        workers=1,
    )


if __name__ == "__main__":
    run()
