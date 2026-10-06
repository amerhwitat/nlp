#!/usr/bin/env sh
# Run the NLP / Ancient Script Studio backend + single-page app.
set -e
cd "$(dirname "$0")"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

PORT="${PORT:-8000}"
echo "Starting NLP / Ancient Script Studio on http://localhost:${PORT}"
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}"
