#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRICT="${STRICT_TOOLCHAINS:-0}"
run_optional(){ if command -v "$1" >/dev/null 2>&1; then shift; "$@"; else echo "SKIP: $1 not installed"; [[ "$STRICT" == 1 ]] && exit 2 || true; fi }
cd "$ROOT"
run_optional cmake cmake -S cpp -B cpp/build && run_optional cmake cmake --build cpp/build
run_optional dotnet dotnet build csharp/ThamudicOcr.csproj
run_optional mvn mvn -q -f java/pom.xml package
run_optional go go build -o go/ocr_scan ./go
run_optional cargo cargo build --manifest-path rust/Cargo.toml
run_optional node node javascript/ocr_scan.mjs --help || true
if command -v npm >/dev/null 2>&1; then (cd typescript && npm install && npm run build); else echo 'SKIP: npm not installed'; [[ "$STRICT" == 1 ]] && exit 2 || true; fi
