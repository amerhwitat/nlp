#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
"$PWD/scripts/check-dependencies.sh"
"$PWD/scripts/build-all-languages.sh"
. .venv/bin/activate
mkdir -p data
python -m uvicorn server.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8010}"
