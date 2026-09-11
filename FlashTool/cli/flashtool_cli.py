#!/usr/bin/env python3
"""FlashTool CLI: inspect and preflight only; destructive execution is intentionally separate."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from python.analyzer import analyze_path


def main() -> int:
    parser = argparse.ArgumentParser(prog="flashtool")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="inspect a local image or OTA package")
    inspect.add_argument("path")

    analyze = sub.add_parser("analyze", help="analyze a local artifact and emit JSON")
    analyze.add_argument("path")
    analyze.add_argument("--pretty", action="store_true")

    args = parser.parse_args()
    if args.command in {"inspect", "analyze"}:
        result = analyze_path(args.path)
        if args.command == "inspect":
            print(f"kind: {result['kind']}")
            print(f"size: {result['size']}")
            print(f"sha256: {result['sha256']}")
            for note in result["notes"]:
                print(f"note: {note}")
        else:
            print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
