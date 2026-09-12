#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v python3 >/dev/null || { echo 'python3 is required'; exit 1; }
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r server/requirements.txt
if [[ "${INSTALL_KRAKEN:-0}" == "1" ]]; then python -m pip install 'kraken>=7,<8'; fi
if command -v tesseract >/dev/null 2>&1; then echo 'Tesseract binary: available'; else echo 'Tesseract binary: not found (optional; install with your OS package manager)'; fi
python - <<'PY'
mods=['PIL','numpy','cv2','pytesseract','fastapi','pypdf','reportlab']
for name in mods:
    __import__(name); print(f'{name}: OK')
try:
    import kraken
    print('kraken: available')
except Exception:
    print('kraken: not installed (optional; set INSTALL_KRAKEN=1 to install)')
PY
