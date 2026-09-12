"""Unicode/UTF-8 registry primitives shared by ancient-language adapters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def codepoint_to_utf8(codepoint: int) -> bytes:
    if not 0 <= codepoint <= 0x10FFFF:
        raise ValueError("Unicode code point out of range")
    return chr(codepoint).encode("utf-8")


def utf8_to_codepoints(value: bytes) -> list[int]:
    return [ord(char) for char in value.decode("utf-8")]


@dataclass(frozen=True)
class UnicodeCharacter:
    codepoint: str
    name: str
    script: str
    utf8: str
    block: str | None = None
    category: str | None = None
    bidi_class: str | None = None


class UnicodeRegistry:
    def __init__(self, payload: dict, version: str):
        self.payload = payload
        self.version = version
        self._records = {item["codepoint"]: item for item in payload.get("records", [])}

    def lookup(self, codepoint: int) -> UnicodeCharacter | None:
        key = f"U+{codepoint:04X}"
        item = self._records.get(key)
        return UnicodeCharacter(**item) if item else None

    def script_for(self, text: str) -> str | None:
        chars = list(text)
        if not chars:
            return None
        item = self.lookup(ord(chars[0]))
        return item.script if item else None


def load_unicode_registry(path: str, version: str) -> UnicodeRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = payload.get("unicode_version")
    if actual != version:
        raise ValueError(f"registry version {actual!r} does not match requested {version!r}")
    return UnicodeRegistry(payload, version)
