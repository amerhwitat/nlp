"""Registry access for ancient/classical language alphabets, variants and historical metadata."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _ROOT / "data" / "source_languages" / "ancient_language_alphabets.json"
_METADATA = _ROOT / "data" / "source_languages" / "ancient_script_metadata.json"


def load_alphabet_registry() -> dict[str, dict[str, Any]]:
    data = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    metadata = json.loads(_METADATA.read_text(encoding="utf-8")).get("languages", {}) if _METADATA.exists() else {}
    profiles: dict[str, dict[str, Any]] = {}
    for item in data["languages"]:
        profile = dict(item); profile.update(metadata.get(item["id"], {}))
        direction = profile.get("direction", "")
        profile.setdefault("writing_direction_description", {
            "ltr": "Primarily written from left to right.",
            "rtl": "Primarily written from right to left.",
            "rtl-or-ltr": "Historical inscriptions may be written in more than one direction depending on corpus and layout.",
        }.get(direction, "Direction varies by historical corpus or remains uncertain."))
        profiles[item["id"]] = profile
    return profiles


def supported_alphabet_languages() -> tuple[str, ...]: return tuple(load_alphabet_registry())


def language_profile(language: str) -> dict[str, Any]:
    profiles = load_alphabet_registry(); key = language.casefold().replace(" ", "-")
    aliases = {"egyptian":"ancient-egyptian","ancient-egyptian":"ancient-egyptian","akkadian":"akkadian","sumerian":"sumerian","ugaritic":"ugaritic","phoenician":"phoenician","hebrew":"ancient-hebrew","ancient-hebrew":"ancient-hebrew","aramaic":"aramaic","ancient-north-arabian":"ancient-north-arabian","old-south-arabian":"old-south-arabian","greek":"greek","ancient-greek":"greek","latin":"latin","chinese":"chinese","japanese":"japanese","old-persian":"old-persian","sanskrit":"sanskrit","coptic":"coptic","hittite":"hittite","luwian":"luwian","etruscan":"etruscan","gothic":"gothic","old-turkic":"old-turkic","linear-b-greek":"linear-b-greek","linear-b":"linear-b-greek","cypro-minoan":"cypro-minoan"}
    key = aliases.get(key, key)
    if key not in profiles: raise ValueError(f"unsupported alphabet language: {language}")
    return profiles[key]


def variations(language: str) -> tuple[str, ...]: return tuple(language_profile(language).get("variations", ()))
def translation_capabilities(language: str) -> tuple[str, ...]: return tuple(language_profile(language).get("translation_modes", ()))


def translation_directions(language: str) -> dict[str, bool]:
    modes = set(translation_capabilities(language))
    source_to_translit = any("script-to-transliteration" in m or "cuneiform-to-transliteration" in m or "hieroglyph-to-transliteration" in m or "han-to-reading" in m for m in modes)
    translit_to_translation = "transliteration-to-translation" in modes
    source_to_translation = any("classical-text-to-translation" in m or "script-to-translation" in m for m in modes) or (source_to_translit and translit_to_translation)
    return {
        "source_to_transliteration": source_to_translit,
        "transliteration_to_translation": translit_to_translation,
        "source_to_translation": source_to_translation,
        "translation_to_source_retrieval": any("translation-to-script" in m or "translation-to-hieroglyph" in m or "translation-to-han" in m for m in modes),
    }
