#!/usr/bin/env bash
set -euo pipefail

app_python="/app/.venv/bin/python"
python_lib_dir="$("${app_python}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"

export PYTHONPATH="/usr/local/rpp/lib:/app/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/rpp/lib:/usr/local/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:${python_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "$@"
