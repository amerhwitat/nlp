#!/usr/bin/env python3
"""Thamudic Scanner All-In-One entry point using the canonical resilient runtime."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"python"))
from run_thamudic002 import main

if __name__=="__main__":
    raise SystemExit(main(["--mode","thamudic",*sys.argv[1:]]))
