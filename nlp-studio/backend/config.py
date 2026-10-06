"""Application configuration for NLP / Ancient Script Studio.

The backend reuses the existing ``python/thamudic`` package by adding the
repository ``python/`` directory (and the repository root, so the ``python.*``
namespace package also resolves) to ``sys.path``. Existing code under
``python/thamudic`` is never modified.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "NLP / Ancient Script Studio"
APP_VERSION = "1.0.0"
APP_SLUG = "nlp-studio"

BASE_DIR = Path(__file__).resolve().parent.parent          # nlp-studio/
REPO_ROOT = BASE_DIR.parent                                # repository root (contains python/)
PYTHON_DIR = REPO_ROOT / "python"

# Reuse the existing python/thamudic modules without touching them.
for _path in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- Storage -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("NLP_STUDIO_DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
WEB_DIR = BASE_DIR / "web"

# The existing translation_log module honours this env var.
os.environ.setdefault("THAMUDIC_TRANSLATION_LOG", str(DATA_DIR / "translations.jsonl"))

# --- Upload limits -----------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.environ.get("NLP_STUDIO_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# --- OCR ---------------------------------------------------------------------
DEFAULT_OCR_LANGUAGES = ["en", "ar"]

# --- Translation -------------------------------------------------------------
SUPPORTED_TARGET_LANGUAGES = ("en", "ar")
# Optional remote translation provider hook. It is only enabled when an API key
# is present in the environment; by default the app is fully offline/corpus-only.
TRANSLATION_API_KEY = os.environ.get("ANCIENT_TRANSLATION_API_KEY", "").strip()
TRANSLATION_API_URL = os.environ.get("ANCIENT_TRANSLATION_API_URL", "").strip()
TRANSLATION_MODEL = os.environ.get("ANCIENT_TRANSLATION_MODEL", "").strip()

# --- CORS --------------------------------------------------------------------
def cors_origins() -> list[str]:
    raw = os.environ.get("NLP_STUDIO_CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "null",  # static file:// or sandboxed iframe origins
    ]


def ensure_directories() -> None:
    for directory in (UPLOAD_DIR, RESULT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def provider_enabled() -> bool:
    return bool(TRANSLATION_API_KEY)
