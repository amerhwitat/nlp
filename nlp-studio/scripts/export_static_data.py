#!/usr/bin/env python3
"""Export corpus + registry + lookup tables to JSON for the no-backend static app.

Run from anywhere::

    python nlp-studio/scripts/export_static_data.py

It reuses the existing ``python/thamudic`` modules (Old North Arabian character
table, alphabet registry, OCIANA seed corpus, and Unicode source-language
profiles) and writes four JSON files into ``nlp-studio/web/data/`` that the
client-side build loads directly, so the static app stays in sync with the
Python sources.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]          # repository root (contains python/)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from python.thamudic.old_north_arabian import CHARACTERS, FIRST, LAST, VARIANT_FORMS  # noqa: E402
from python.thamudic.ancient_alphabet_registry import load_alphabet_registry  # noqa: E402
from python.thamudic.ancient_translation import OCIANA_SEED, entry_to_dict  # noqa: E402
from python.thamudic.source_language_scanner import load_profiles  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "web" / "data"


def write(name: str, payload) -> Path:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> int:
    ona = {
        "first": FIRST,
        "last": LAST,
        "variant_forms": list(VARIANT_FORMS),
        "characters": [
            {
                "codepoint": item["codepoint"],
                "character": item["character"],
                "name": item["name"],
                "transliteration": item["transliteration"],
                "utf8_hex": item["utf8_hex"],
            }
            for item in CHARACTERS
        ],
        "by_character": {item["character"]: item["transliteration"] for item in CHARACTERS},
    }
    corpus = [entry_to_dict(entry) for entry in OCIANA_SEED]

    # The generic Unicode profiles (ancient_classical_unicode.json) do not list
    # Old North Arabian, so synthesize its profile from the ONA registry so the
    # static app can detect those characters too.
    profiles = load_profiles()
    profiles["ancient-north-arabian"] = {
        "id": "ancient-north-arabian",
        "language": "Ancient North Arabian",
        "iso639": [],
        "scripts": ["Old North Arabian"],
        "direction": "rtl",
        "ranges": [f"U+{FIRST:04X}-U+{LAST:04X}"],
        "notes": "Old North Arabian Unicode block; synthesized for the static app from python/thamudic.old_north_arabian.",
    }

    written = [
        write("ona_chars.json", ona),
        write("alphabet_registry.json", load_alphabet_registry()),
        write("translation_corpus.json", corpus),
        write("source_language_profiles.json", profiles),
    ]
    for path in written:
        print(f"  wrote {path}")
    print(f"Static data exported ({len(written)} files) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
