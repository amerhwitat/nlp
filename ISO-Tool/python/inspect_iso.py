#!/usr/bin/env python3
"""Command-line entry point for safe, read-only ISO inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iso_tool.advanced_inspect import inspection_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ISO structure without mounting or executing it")
    parser.add_argument("image", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = inspection_dict(args.image)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("ISO:", report["path"])
        print("SHA-256:", report["sha256"])
        print("Size:", report["size"])
        print("ISO 9660:", report["iso9660"])
        print("Joliet:", report["joliet"])
        print("Rock Ridge hint:", report["rock_ridge_hint"])
        print("UDF:", report["udf"])
        print("MBR:", report["mbr"], "GPT:", report["gpt"])
        print("Boot entries:", len(report["boot_entries"]))
        for warning in report["warnings"]:
            print("WARNING:", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
