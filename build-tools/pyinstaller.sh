#!/usr/bin/env sh
set -eu
R=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$R/build-tools/build.sh" --only python --onefile "$@"
