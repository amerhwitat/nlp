"""Generate a compact, deterministic Unicode registry from a pinned UCD directory.

Expected input: Unicode Public/<version>/ucd with UnicodeData.txt, Scripts.txt,
Blocks.txt and DerivedBidiClass.txt. The generator intentionally records Unicode
properties only; language/script associations are supplied by the application's
separate language registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _parse_range_file(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ";" not in line:
            continue
        span, value = [part.strip() for part in line.split(";", 1)]
        if ".." in span:
            start, end = (int(part, 16) for part in span.split(".."))
        else:
            start = end = int(span, 16)
        for cp in range(start, end + 1):
            values[cp] = value
    return values


def generate_snapshot(ucd_root: str, output: str, version: str) -> None:
    root = Path(ucd_root)
    unicode_data = root / "UnicodeData.txt"
    scripts = _parse_range_file(root / "Scripts.txt")
    bidi = _parse_range_file(root / "extracted" / "DerivedBidiClass.txt")
    blocks = _parse_range_file(root / "Blocks.txt")
    records = []
    for raw in unicode_data.read_text(encoding="utf-8").splitlines():
        fields = raw.split(";")
        if len(fields) < 15:
            continue
        cp = int(fields[0], 16)
        script = scripts.get(cp)
        if not script or script in {"Common", "Inherited", "Unknown"}:
            continue
        records.append({
            "codepoint": f"U+{cp:04X}",
            "name": fields[1],
            "category": fields[2],
            "combining_class": fields[3],
            "bidi_class": bidi.get(cp, fields[4]),
            "script": script,
            "block": blocks.get(cp),
            "utf8": chr(cp).encode("utf-8").hex(" ").upper(),
        })
    payload = {
        "unicode_version": version,
        "source": "Unicode Character Database",
        "source_sha256": hashlib.sha256(unicode_data.read_bytes()).hexdigest(),
        "records": records,
    }
    Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucd-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    generate_snapshot(args.ucd_root, args.output, args.version)


if __name__ == "__main__":
    main()
