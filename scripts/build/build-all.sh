#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Release}"
cd "$ROOT"
step(){ echo "== $1 =="; shift; "$@"; }
command -v node >/dev/null && command -v npm >/dev/null
if [[ -f web/package.json ]]; then step "Web build" bash -lc 'cd web && if [[ -f package-lock.json ]]; then npm ci; else npm install; fi && npm run build'; fi
if [[ -x scripts/build/build-wasm.sh ]]; then step "WASM build" scripts/build/build-wasm.sh "$CONFIGURATION"; fi
if [[ -x scripts/build/build-native.sh ]]; then step "Native build" scripts/build/build-native.sh "$CONFIGURATION"; fi
if [[ -d dotnet ]]; then step ".NET build" dotnet build dotnet -c "$CONFIGURATION" --nologo; fi
if [[ -d python ]]; then step "Python validation" python -m compileall -q python; fi
if [[ -x scripts/database/init-database.sh ]]; then step "Database validation" scripts/database/init-database.sh check; fi
echo "Build-all completed: $CONFIGURATION"
