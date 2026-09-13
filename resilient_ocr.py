"""Compatibility shim for the resilient Thamudic OCR runtime.

Keeps root-level launchers working when the repository is executed without
installing ``python/`` as a package root. The canonical implementation remains
``python/thamudic/resilient_ocr.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from thamudic.resilient_ocr import (  # noqa: E402,F401
    DEFAULT_TIMEOUT,
    IMAGE_SUFFIXES,
    MAX_IMAGE_DIMENSION,
    ocr_image,
    pdf_page,
    prepare_image,
    run_worker,
    terminate_worker,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "IMAGE_SUFFIXES",
    "MAX_IMAGE_DIMENSION",
    "ocr_image",
    "pdf_page",
    "prepare_image",
    "run_worker",
    "terminate_worker",
]
