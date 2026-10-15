"""Fail the container health check unless the dispatch API contract is ready."""

from __future__ import annotations

import json
import os
import sys
from urllib.error import URLError
from urllib.request import urlopen

_REQUIRED_CAPABILITIES = frozenset(
    {
        "replace_active",
        "active_search",
        "search_timeout",
        "request_lookup",
        "event_replay",
        "confirmed_evidence_v1",
    }
)


def _get_json(base_url: str, path: str) -> dict[str, object]:
    with urlopen(f"{base_url}{path}", timeout=5) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} returned a non-object response")
    return payload


def main() -> int:
    port = os.environ.get("PERSON_SEARCH_PORT", "18080")
    base_url = f"http://127.0.0.1:{port}"
    try:
        health = _get_json(base_url, "/healthz")
        readiness = _get_json(base_url, "/readyz")
    except (OSError, URLError, ValueError, TypeError) as exc:
        print(f"executor health contract unavailable: {type(exc).__name__}", file=sys.stderr)
        return 1

    capabilities = health.get("capabilities")
    advertised = (
        {item for item in capabilities if isinstance(item, str)}
        if isinstance(capabilities, list)
        else set()
    )
    missing = sorted(_REQUIRED_CAPABILITIES - advertised)
    if health.get("status") != "ok" or readiness.get("status") != "ready" or missing:
        if missing:
            print("executor missing required capabilities: " + ", ".join(missing), file=sys.stderr)
        if health.get("status") != "ok":
            print("executor /healthz is not ok", file=sys.stderr)
        if readiness.get("status") != "ready":
            print("executor /readyz is not ready", file=sys.stderr)
        return 1

    print("executor readiness and all 6 required capabilities are healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
