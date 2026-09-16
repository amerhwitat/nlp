#!/usr/bin/env python3
"""Analyze a public Bitcoin-address list without searching private-key space."""
from __future__ import annotations
import argparse
import re
from collections import Counter

BASE58 = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def classify(address: str) -> str:
    if not address or any(ch not in BASE58 for ch in address):
        return "invalid"
    if address.startswith("1"):
        return "legacy-p2pkh-candidate"
    if address.startswith("3"):
        return "p2sh-candidate"
    if address.startswith("bc1q"):
        return "segwit-v0-candidate"
    if address.startswith("bc1p"):
        return "taproot-candidate"
    return "other"


def analyze(path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        raw = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    unique = list(dict.fromkeys(raw))
    counts = Counter(classify(a) for a in unique)
    print(f"Input lines: {len(raw)}")
    print(f"Unique addresses: {len(unique)}")
    for kind, count in sorted(counts.items()):
        print(f"{kind}: {count}")
    invalid = [a for a in unique if classify(a) == "invalid"]
    if invalid:
        print("Invalid/unsupported entries:")
        for a in invalid:
            print(f"  {a}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address_file")
    analyze(parser.parse_args().address_file)
