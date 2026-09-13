"""Unicode-aware scanner for requested ancient/classical source-language profiles.

This module detects characters by Unicode script/range profiles and always exposes
UTF-8 bytes. It does not claim that a Unicode script uniquely identifies a language;
Chinese/Japanese Han overlap is intentionally reported as an ambiguity when useful.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REGISTRY = Path(__file__).resolve().parents[2] / "data" / "source_languages" / "ancient_classical_unicode.json"


@dataclass(frozen=True)
class UnicodeRange:
    start: int
    end: int

    def contains(self, codepoint: int) -> bool:
        return self.start <= codepoint <= self.end


def _parse_range(value: str) -> UnicodeRange:
    start, end = value.removeprefix("U+").split("-U+")
    return UnicodeRange(int(start, 16), int(end, 16))


def load_profiles() -> dict[str, dict[str, Any]]:
    data = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data["profiles"]}


def supported_source_languages() -> tuple[str, ...]:
    return tuple(load_profiles())


def _profile_matches(ch: str, profile: dict[str, Any]) -> bool:
    cp = ord(ch)
    return any(_parse_range(item).contains(cp) for item in profile["ranges"])


def _utf8_hex(ch: str) -> str:
    return " ".join(f"{byte:02X}" for byte in ch.encode("utf-8"))


def scan_source_language(text: str, language: str | None = None) -> dict[str, Any]:
    profiles = load_profiles()
    if language:
        key = language.casefold().replace(" ", "-")
        aliases = {
            "egyptian": "ancient-egyptian",
            "ancient-egyptian": "ancient-egyptian",
            "chinese": "chinese",
            "japanese": "japanese",
            "greek": "greek",
            "ancient-greek": "greek",
            "latin": "latin",
            "classical-latin": "latin",
        }
        key = aliases.get(key, key)
        if key not in profiles:
            raise ValueError(f"unsupported source language: {language}")
        candidate_ids = (key,)
    else:
        candidate_ids = tuple(profiles)

    characters: list[dict[str, Any]] = []
    counts: dict[str, int] = {key: 0 for key in candidate_ids}
    for index, ch in enumerate(text):
        matches = [key for key in candidate_ids if _profile_matches(ch, profiles[key])]
        if matches:
            for key in matches:
                counts[key] += 1
            characters.append({
                "index": index,
                "character": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "decimal": ord(ch),
                "name": unicodedata.name(ch, "UNNAMED"),
                "utf8": _utf8_hex(ch),
                "utf8_bytes": list(ch.encode("utf-8")),
                "normalized_nfc": unicodedata.normalize("NFC", ch),
                "matches": matches,
            })

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    detected = [key for key, count in ranked if count]
    ambiguous = any(len(item["matches"]) > 1 for item in characters)
    return {
        "text": text,
        "encoding": "UTF-8",
        "unicode_normalization": "NFC",
        "requested_language": language,
        "detected_languages": detected,
        "ambiguous_script_overlap": ambiguous,
        "counts": counts,
        "characters": characters,
        "matched_character_count": len(characters),
        "supported_source_languages": list(profiles),
    }


def encode_utf8(text: str) -> dict[str, Any]:
    raw = text.encode("utf-8")
    return {"text": text, "encoding": "UTF-8", "bytes": list(raw), "hex": " ".join(f"{b:02X}" for b in raw)}
