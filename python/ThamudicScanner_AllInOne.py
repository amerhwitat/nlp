#!/usr/bin/env python3
"""Thamudic Scanner All-In-One compatibility entry point.

Uses the canonical unified Thamudic/NLP GUI and resilient OCR runtime so both
applications share one maintained implementation and one timeout policy.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"python"))
from run_thamudic002 import launch

if __name__=="__main__":
    launch("thamudic")
