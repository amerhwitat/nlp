"""Pronunciation profile lookup for transliteration and translation speech."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PronunciationResult:
    text: str
    profile_id: str
    language_stage: str
    pronunciation_type: str
    locale: str
    backend: str
    label: str


def resolve_pronunciation(text: str, language_stage: str, profile_id: str, path: str = "data/ancient_languages/pronunciation.json") -> PronunciationResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profile = next((p for p in payload["profiles"] if p["id"] == profile_id), None)
    if profile is None:
        raise KeyError(profile_id)
    if profile["language_stage"] != language_stage:
        raise ValueError("pronunciation profile does not match language stage")
    return PronunciationResult(
        text=text,
        profile_id=profile["id"],
        language_stage=profile["language_stage"],
        pronunciation_type=profile["type"],
        locale=profile["locale"],
        backend=profile["backend"],
        label=profile["label"],
    )
