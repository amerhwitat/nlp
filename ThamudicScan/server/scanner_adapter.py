from __future__ import annotations

from python.thamudic import (
    BY_CHARACTER, extract, is_old_north_arabian, scan_source_language, translate, transliterate,
    supported_alphabet_languages, language_profile, variations, translation_capabilities,
    translation_directions, translate_ancient, translation_matrix,
)


def _matches_keywords(text: str, transliterated: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    haystacks = (text.casefold(), transliterated.casefold())
    return any(keyword.strip().casefold() in haystack for keyword in keywords if keyword.strip() for haystack in haystacks)


def scan_text(text: str, keywords: list[str] | None = None) -> dict:
    keywords = keywords or []
    extracted = extract(text)
    transliterated = transliterate(extracted)
    matched = bool(extracted) and _matches_keywords(extracted, transliterated, keywords)
    return {
        "text": extracted,
        "transliteration": transliterated,
        "keywords": keywords,
        "matched": matched,
        "confidence": 1.0 if extracted else 0.0,
        "language": "Old North Arabian",
        "script_variant": "Dadanitic",
        "codepoints": [ord(ch) for ch in extracted],
    }


def translate_text(text: str, script: str = "Dadanitic", target_language: str = "en") -> dict:
    extracted = extract(text)
    if not extracted:
        extracted = text
    return translate(extracted, script=script, target_language=target_language)


def translate_ancient_text(text: str, source_language: str, target_language: str, source_form: str = "script") -> dict:
    return translate_ancient(text, source_language, target_language, source_form=source_form).as_dict()


def scan_source_language_text(text: str, language: str | None = None) -> dict:
    """Scan Unicode source text using the universal source-language registry."""
    return scan_source_language(text, language=language)


def alphabet_languages() -> tuple[str, ...]:
    return supported_alphabet_languages()


def alphabet_profile(language: str) -> dict:
    return language_profile(language)


def alphabet_variations(language: str) -> tuple[str, ...]:
    return variations(language)


def translation_modes(language: str) -> tuple[str, ...]:
    return translation_capabilities(language)


def translation_directions_for(language: str) -> dict[str, bool]:
    return translation_directions(language)


def all_translation_directions() -> dict[str, dict[str, bool]]:
    return translation_matrix()


def validate_text(text: str) -> dict:
    chars = [ch for ch in text if is_old_north_arabian(ch)]
    return {
        "count": len(chars),
        "characters": chars,
        "codepoints": [ord(ch) for ch in chars],
        "known": [BY_CHARACTER[ch] for ch in chars],
        "unicode_range": "U+10A80-U+10A9F",
    }
