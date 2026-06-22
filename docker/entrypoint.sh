#!/usr/bin/env bash
set -euo pipefail

site_packages="$('/app/.venv/bin/python' - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

for library_dir in "$site_packages"/nvidia/*/lib "$site_packages"/nvidia/cu13/lib; do
    if [ -d "$library_dir" ]; then
        export LD_LIBRARY_PATH="$library_dir:${LD_LIBRARY_PATH:-}"
    fi
done

exec "$@"
