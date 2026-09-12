"""Hebrew historical-stage and alphabet registry helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HebrewCharacter:
    name: str
    char: str
    codepoint: str
    utf8: str
    transliteration: str = ""


class HebrewRegistry:
    def __init__(self, language: dict, alphabet_table: dict) -> None:
        self.language = language
        self.alphabet_table = alphabet_table

    def stages(self) -> list[str]:
        return [stage["id"] for stage in self.language["stages"]]

    def alphabet(self, stage_id: str) -> list[HebrewCharacter]:
        table = self.alphabet_table["alphabets"].get(stage_id)
        if table is None:
            table = self.alphabet_table["alphabets"].get("hebrew.modern")
        if table is None:
            return []
        members = [HebrewCharacter(**entry) for entry in table["members"]]
        members.extend(HebrewCharacter(**entry, transliteration="") for entry in table.get("final_forms", []))
        return members

    def lookup(self, char: str) -> dict | None:
        for stage_id in self.stages():
            for item in self.alphabet(stage_id):
                if item.char == char:
                    return item.__dict__
        return None


def load_hebrew_registry(path: str) -> HebrewRegistry:
    root = Path(path)
    with (root / "hebrew.json").open(encoding="utf-8") as stream:
        language = json.load(stream)
    with (root / "alphabet_tables.json").open(encoding="utf-8") as stream:
        alphabet = json.load(stream)
    return HebrewRegistry(language, alphabet)


def codepoint_to_utf8(codepoint: int) -> bytes:
    return chr(codepoint).encode("utf-8")
