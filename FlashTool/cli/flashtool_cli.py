#!/usr/bin/env python3
"""FlashTool CLI: inspect, analyze, preflight and dry-run only.

No command in this module performs a destructive device operation.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from python.analyzer import analyze_path
from python.flashtool import DeviceInfo, FlashPlan, Slot, Transport, preflight


def _transport(value):
    return Transport(value)


def _build_plan(args):
    return DeviceInfo(
        serial=args.serial,
        product=args.product,
        bootloader_unlocked=args.unlocked,
        transport=_transport(args.transport),
        slot=Slot(args.slot),
    ), FlashPlan(
        partition=args.partition,
        image_path=args.path,
        dry_run=True,
        verify=True,
        target_slot=Slot(args.slot),
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="flashtool")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="inspect a local image or OTA package")
    inspect.add_argument("path")

    analyze = sub.add_parser("analyze", help="analyze a local artifact and emit JSON")
    analyze.add_argument("path")
    analyze.add_argument("--pretty", action="store_true")

    for name in ("preflight", "dry-run"):
        cmd = sub.add_parser(name, help="perform a non-destructive compatibility preflight")
        cmd.add_argument("path")
        cmd.add_argument("--partition", required=True)
        cmd.add_argument("--transport", choices=[t.value for t in Transport], default=Transport.FASTBOOT.value)
        cmd.add_argument("--slot", choices=[s.value for s in Slot], default=Slot.UNKNOWN.value)
        cmd.add_argument("--serial", default="")
        cmd.add_argument("--product", default="")
        cmd.add_argument("--unlocked", action="store_true")
        cmd.add_argument("--partition-size", type=int, default=0)

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

    device, plan = _build_plan(args)
    result = preflight(device, plan, args.partition_size)
    result["command"] = args.command
    result["analysis"] = analyze_path(args.path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
