#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r server/requirements.txt
mkdir -p data
python -m uvicorn server.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8010}"
