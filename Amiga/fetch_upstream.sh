#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "$0")/vendor"
cd "$(dirname "$0")/vendor"
if [ ! -d sae/.git ]; then git clone https://github.com/naTmeg/ScriptedAmigaEmulator.git sae; else git -C sae pull --ff-only; fi
if [ ! -d vamigaweb/.git ]; then git clone https://github.com/vAmigaWeb/vAmigaWeb.git vamigaweb; else git -C vamigaweb pull --ff-only; fi
printf '\nUpstream Amiga sources are available under Amiga/vendor/. Review each upstream license before redistribution.\n'
