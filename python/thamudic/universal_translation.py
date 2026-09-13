"""Provider-oriented translation facade for the ancient-language registry.

This module intentionally separates *capability* from *availability*: a registered
language can expose a translation direction while a local provider may not yet have
an attested parallel corpus/model for a particular fragment. In that case the API
returns ``not_available`` instead of fabricating a translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

from .ancient_alphabet_registry import language_profile, translation_directions


@dataclass(frozen=True)
class TranslationResult:
    source: str
    source_language: str
    source_form: str
    target_language: str
    target_form: str | None
    transliteration: str | None
    status: str
    confidence: float
    provider: str
    provenance: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_language": self.source_language,
            "source_form": self.source_form,
            "target_language": self.target_language,
            "target_form": self.target_form,
            "transliteration": self.transliteration,
            "status": self.status,
            "confidence": self.confidence,
            "provider": self.provider,
            "provenance": self.provenance,
        }


class TranslationProvider(Protocol):
    name: str

    def translate(self, source: str, source_language: str, source_form: str, target_language: str) -> TranslationResult | None: ...


def _basic_transliteration(text: str) -> str:
    """Return a conservative scholarly placeholder for Unicode source text.

    The universal layer does not invent phonetic values. Characters are retained
    unless a language-specific provider supplies a real transliteration.
    """
    return text


def translate_ancient(
    source: str,
    source_language: str,
    target_language: str,
    *,
    source_form: str = "script",
    provider: TranslationProvider | None = None,
) -> TranslationResult:
    if not source.strip():
        raise ValueError("source is required")

    profile = language_profile(source_language)
    target = target_language.casefold()
    directions = translation_directions(source_language)
    direction_key = {
        "script": "source_to_translation",
        "transliteration": "transliteration_to_translation",
        "translation": "translation_to_source_retrieval",
    }.get(source_form.casefold())
    if direction_key is None:
        raise ValueError("source_form must be script, transliteration, or translation")

    if provider is not None:
        result = provider.translate(source, profile["id"], source_form, target)
        if result is not None:
            return result

    # Metadata-backed capability without a model/corpus is not a translation.
    capable = directions.get(direction_key, False)
    return TranslationResult(
        source=source,
        source_language=profile["id"],
        source_form=source_form,
        target_language=target,
        target_form=None,
        transliteration=_basic_transliteration(source) if source_form != "translation" else None,
        status="provider_required" if capable else "direction_not_registered",
        confidence=0.0,
        provider="none",
        provenance=None,
    )


def translation_matrix() -> dict[str, dict[str, bool]]:
    """Return registered directional capabilities for every alphabet profile."""
    return {language_id: translation_directions(language_id) for language_id in __import__(
        "python.thamudic.ancient_alphabet_registry", fromlist=["supported_alphabet_languages"]
    ).supported_alphabet_languages()}
