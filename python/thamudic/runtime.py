"""Process-runtime safeguards for optional scientific/OCR dependencies.

Windows installations that combine PyTorch/EasyOCR, NumPy/SciPy/scikit-learn and
other native packages can load more than one Intel OpenMP runtime (libiomp5md.dll).
That can abort the GUI before it is even displayed.

This module is intentionally tiny and must be imported before scientific/OCR packages.
The compatibility fallback is Windows-only and can be disabled with
THAMUDIC_ALLOW_DUPLICATE_OPENMP=0. Thread counts are also capped unless explicitly
configured, which keeps the desktop scanner responsive on constrained systems.
"""
from __future__ import annotations

import os
import sys


def configure_runtime() -> None:
    """Configure environment variables before native ML/OCR libraries are imported."""
    if sys.platform.startswith("win"):
        # EasyOCR/PyTorch and NumPy/SciPy wheels may ship different OpenMP runtimes.
        # Setting this before either runtime is initialized prevents the hard abort.
        # It is a compatibility fallback, not a claim that multiple runtimes are ideal.
        if os.environ.get("THAMUDIC_ALLOW_DUPLICATE_OPENMP", "1") != "0":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Avoid uncontrolled native thread explosions when several numerical libraries
    # participate in one desktop process. Users can override these before launch.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


configure_runtime()

__all__ = ["configure_runtime"]
