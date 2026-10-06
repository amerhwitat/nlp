#!/usr/bin/env sh
# Install NLP / Ancient Script Studio dependencies into a local virtualenv.
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "Creating virtual environment (.venv)..."
"$PY" -m venv .venv

# shellcheck disable=SC1091
. .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. Dependencies installed into nlp-studio/.venv"
echo
echo "Optional: install a system OCR engine for the lightweight fallback"
echo "  (Debian/Ubuntu):  sudo apt-get install -y tesseract-ocr"
echo
echo "Run the app with:  ./run.sh"
