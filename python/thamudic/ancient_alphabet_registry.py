"""Registry access for ancient/classical language alphabets and variants.

The registry is deliberately metadata-first: it records historical variants,
Unicode script blocks and supported translation directions without pretending that
an alphabet alone provides a translation model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REGISTRY = Path(__file__).resolve().parents[2] / "data" / "source_languages" / "ancient_language_alphabets.json"


def load_alphabet_registry() -> dict[str, dict[str, Any]]:
    data = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data["languages"]}


def supported_alphabet_languages() -> tuple[str, ...]:
    return tuple(load_alphabet_registry())


def language_profile(language: str) -> dict[str, Any]:
    profiles = load_alphabet_registry()
    key = language.casefold().replace(" ", "-")
    aliases = {
        "egyptian": "ancient-egyptian",
        "ancient-egyptian": "ancient-egyptian",
        "akkadian": "akkadian",
        "sumerian": "sumerian",
        "ugaritic": "ugaritic",
        "phoenician": "phoenician",
        "hebrew": "ancient-hebrew",
        "ancient-hebrew": "ancient-hebrew",
        "aramaic": "aramaic",
        "ancient-north-arabian": "ancient-north-arabian",
        "greek": "greek",
        "ancient-greek": "greek",
        "latin": "latin",
        "chinese": "chinese",
        "japanese": "japanese",
        "old-persian": "old-persian",
        "sanskrit": "sanskrit",
        "coptic": "coptic",
        "hittite": "hittite",
        "luwian": "luwian",
        "etruscan": "etruscan",
        "gothic": "gothic",
        "old-turkic": "old-turkic",
        "linear-b-greek": "linear-b-greek",
        "cypro-minoan": "cypro-minoan",
    }
    key = aliases.get(key, key)
    if key not in profiles:
        raise ValueError(f"unsupported alphabet language: {language}")
    return profiles[key]


def variations(language: str) -> tuple[str, ...]:
    return tuple(language_profile(language).get("variations", ()))


def translation_capabilities(language: str) -> tuple[str, ...]:
    return tuple(language_profile(language).get("translation_modes", ()))


def translation_directions(language: str) -> dict[str, bool]:
    modes = set(translation_capabilities(language))
    return {
        "source_to_transliteration": any("script-to-transliteration" in m or "cuneiform-to-transliteration" in m or "hieroglyph-to-transliteration" in m for m in modes),
        "transliteration_to_translation": "transliteration-to-translation" in modes,
        "source_to_translation": any("classical-text-to-translation" in m or "script-to-translation" in m for m in modes),
        "translation_to_source_retrieval": any("translation-to-script" in m or "translation-to-hieroglyph" in m or "translation-to-han" in m for m in modes),
    }
