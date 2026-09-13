#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 0 ]]; then exec "$@"; fi
if [[ -f /app/run_thamudic.py ]]; then exec python3 /app/run_thamudic.py --help; fi
if [[ -f /app/package.json ]]; then npm install --omit=dev >/tmp/npm-install.log 2>&1 || true; if node -e 'let p=require("./package.json"); process.exit(p.scripts?.start?0:1)' 2>/dev/null; then exec npm start; fi; fi
if [[ -f /app/CMakeLists.txt ]]; then cmake -S /app -B /tmp/repo-build -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/repo-build -j"$(nproc)"; bin=$(find /tmp/repo-build -maxdepth 3 -type f -perm -111 -not -name '*.so' | head -n 1 || true); [[ -n "$bin" ]] && exec "$bin"; fi
if [[ -f /app/index.html ]]; then exec python3 -m http.server 8000 --directory /app; fi
if [[ -f /app/requirements.txt ]]; then python3 -m venv /tmp/venv; /tmp/venv/bin/pip install -r /app/requirements.txt >/tmp/pip-install.log 2>&1 || true; fi
for f in /app/main.py /app/app.py /app/main.js; do [[ -f "$f" ]] || continue; case "$f" in *.py) exec python3 "$f";; *.js) exec node "$f";; esac; done
printf '%s\n' 'Container is ready. No unambiguous application entry point was detected.' 'Pass a command explicitly to run a specific application.'
exec bash
