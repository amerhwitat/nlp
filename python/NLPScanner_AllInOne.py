#!/usr/bin/env python3
"""NLP Scanner All-In-One compatibility entry point.

Delegates to the canonical unified GUI and resilient OCR runtime. This keeps
NLP and Thamudic scanners on the same GUI while eliminating the historical
180-second native-worker failure mode.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"python"))
from run_thamudic002 import launch

if __name__=="__main__":
    launch("nlp")
