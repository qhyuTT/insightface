"""Copy the dispatch evidence API credential into the RK3588 runtime env file."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_SOURCE_VARIABLE = "DISPATCH_INSIGHTFACE_EVIDENCE_API_KEY"
_TARGET_VARIABLE = "PERSON_SEARCH_EVIDENCE_API_KEY"


def _container_environment(container: str) -> dict[str, str]:
    raw = subprocess.check_output(["docker", "inspect", container], text=True)
    config = json.loads(raw)[0]["Config"]
    pairs = config.get("Env", [])
    return {key: value for key, value in (item.split("=", 1) for item in pairs if "=" in item)}


def _replace_key(path: Path, key: str) -> None:
    old_stat = path.stat() if path.exists() else None
    if old_stat is not None:
        backup_path = path.with_name(
            f"{path.name}.backup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"  # noqa: UP017
        )
        shutil.copy2(path, backup_path)
        os.chown(backup_path, old_stat.st_uid, old_stat.st_gid)
        os.chmod(backup_path, 0o600)
    lines = path.read_text(encoding="utf-8").splitlines() if old_stat is not None else []
    lines = [line for line in lines if not line.startswith(f"{_TARGET_VARIABLE}=")]
    lines.append(f"{_TARGET_VARIABLE}={shlex.quote(key)}")
    content = "\n".join(lines) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        if old_stat is not None:
            os.chown(temporary_path, old_stat.st_uid, old_stat.st_gid)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="airport-robot-dispatch-box-backend-1")
    parser.add_argument("--runtime-env", type=Path, default=Path("/ssd/person-search-rpp/runtime.env"))
    parser.add_argument("--apply", action="store_true", help="write the credential to runtime-env")
    args = parser.parse_args()

    source_key = _container_environment(args.container).get(_SOURCE_VARIABLE)
    if not source_key:
        parser.error("dispatch evidence API key is not configured")
    if any(char.isspace() or not char.isprintable() for char in source_key):
        parser.error("dispatch evidence API key cannot be represented safely in runtime.env")
    if not args.runtime_env.parent.is_dir():
        parser.error("runtime env file parent directory does not exist")
    if not args.apply:
        parser.error("pass --apply to update the runtime env file")

    _replace_key(args.runtime_env, source_key)
    print("evidence API key synchronized; value redacted; runtime env permissions set to 0600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
