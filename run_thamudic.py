#!/usr/bin/env python3
"""Primary launcher for the unified Thamudic + NLP all-in-one workbench.

The implementation lives in ``run_thamudic002.py`` so both entry points expose
the same GUI, resilient OCR, source scanning, transliteration, translation,
voice, history/PDF export, and CLI workflows.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_thamudic002 import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
