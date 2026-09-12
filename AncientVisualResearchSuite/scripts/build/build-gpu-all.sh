#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cmake -S "$ROOT/gpu" -B "$ROOT/build/gpu" -DCMAKE_BUILD_TYPE=Release -DAVRS_ENABLE_OPENGL=ON -DAVRS_ENABLE_DIRECTX12=OFF
cmake --build "$ROOT/build/gpu" --config Release --parallel
