#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${1:-developer}"
cd "$ROOT"
need(){ command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 2; }; }
need git; need python3; need node; need npm
python3 -m venv .venv 2>/dev/null || true
if [[ -x .venv/bin/python && -f python/requirements.txt ]]; then .venv/bin/python -m pip install --upgrade pip; .venv/bin/python -m pip install -r python/requirements.txt; fi
if [[ -f web/package-lock.json ]]; then (cd web && npm ci); elif [[ -f web/package.json ]]; then (cd web && npm install); fi
if [[ "$PROFILE" != minimal && "$PROFILE" != ci ]] && command -v cmake >/dev/null 2>&1; then cmake --version | head -n 1; fi
if [[ -x scripts/database/init-database.sh ]]; then scripts/database/init-database.sh check; fi
echo "Bootstrap completed: $PROFILE"
