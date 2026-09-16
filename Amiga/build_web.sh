#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/fetch_upstream.sh"
# SAE's browser engine is kept under its upstream tree; the Chimera shell loads
# vendor/sae/index.js when present. This build script also records the exact
# upstream commit used for reproducible review.
cd "$ROOT/vendor/sae"
git rev-parse HEAD > "$ROOT/SAE_COMMIT.txt"
printf 'SAE source commit: '; cat "$ROOT/SAE_COMMIT.txt"
