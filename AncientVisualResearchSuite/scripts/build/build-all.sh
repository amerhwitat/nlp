#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if command -v cmake >/dev/null 2>&1; then
  cmake -S "$ROOT/core/cpp" -B "$ROOT/build/cpp" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$ROOT/build/cpp" --config Release
fi
if command -v python3 >/dev/null 2>&1; then python3 -m compileall "$ROOT/core/python"; fi
if [ -f "$ROOT/web/package.json" ] && command -v npm >/dev/null 2>&1; then
  cd "$ROOT/web"; npm ci; npm run build
fi
echo "AVRS reference build finished where toolchains are installed."
