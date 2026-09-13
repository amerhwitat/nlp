#!/usr/bin/env python3
"""Voice-enabled launcher for the Thamudic all-in-one scanner.

The main GUI implementation now lives in ``ThamudicScanner_AllInOne.py`` so both
entry points expose the same run_thammudic-style layout, image/PDF import controls,
transliteration/translation panels, and visible voice controls.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ThamudicScanner_AllInOne import ThamudicScannerApp


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voice-enabled all-in-one Thamudic scanner")
    parser.add_argument("file", nargs="?", help="image/PDF/text file to import")
    parser.add_argument("--text", default="", help="source text")
    parser.add_argument("--script", default="ancient-north-arabian")
    parser.add_argument("--target", default="en", choices=["en", "ar"])
    parser.add_argument("--gui", action="store_true", help="start the full GUI")
    args = parser.parse_args(argv)

    app = ThamudicScannerApp()
    app.script.set(args.script)
    app.target.set(args.target)

    if args.file:
        app.media_path = Path(args.file)
        app.process_media()
    elif args.text:
        app.source.insert("1.0", args.text)
        app.translate()

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
