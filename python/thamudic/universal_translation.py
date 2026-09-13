"""Provider-oriented translation facade for the ancient-language registry.

This module separates capability from availability and never fabricates a
translation when an attested corpus/model is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

from .ancient_alphabet_registry import language_profile, translation_directions
from .translation_log import append_record, make_record


@dataclass(frozen=True)
class TranslationResult:
    source: str
    source_language: str
    source_form: str
    target_language: str
    target_form: str | None
    transliteration: str | None
    translation: str | None
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
            "translation": self.translation,
            "status": self.status,
            "confidence": self.confidence,
            "provider": self.provider,
            "provenance": self.provenance,
        }


class TranslationProvider(Protocol):
    name: str

    def translate(self, source: str, source_language: str, source_form: str, target_language: str) -> TranslationResult | None: ...


def _basic_transliteration(text: str) -> str:
    return text


def _log_result(result: TranslationResult, request_metadata: dict[str, Any] | None = None) -> None:
    # Local import avoids a module cycle: script_summary itself uses this facade
    # to build complete script reports.
    from .script_summary import build_script_summary
    metadata = build_script_summary(result.source_language)
    record = make_record(
        source=result.source, source_language=result.source_language,
        source_form=result.source_form, target_language=result.target_language,
        target_form=result.target_form, transliteration=result.transliteration,
        translation=result.translation, status=result.status,
        confidence=result.confidence, provider=result.provider,
        provenance=result.provenance, script_metadata=metadata,
        request_metadata=request_metadata,
    )
    append_record(record)


def translate_ancient(
    source: str,
    source_language: str,
    target_language: str,
    *,
    source_form: str = "script",
    provider: TranslationProvider | None = None,
    request_metadata: dict[str, Any] | None = None,
    log: bool = True,
) -> TranslationResult:
    if not source.strip():
        raise ValueError("source is required")
    profile = language_profile(source_language)
    target = target_language.casefold()
    directions = translation_directions(source_language)
    direction_key = {"script": "source_to_translation", "transliteration": "transliteration_to_translation", "translation": "translation_to_source_retrieval"}.get(source_form.casefold())
    if direction_key is None:
        raise ValueError("source_form must be script, transliteration, or translation")

    result = None
    if provider is not None:
        result = provider.translate(source, profile["id"], source_form, target)
    if result is None:
        capable = directions.get(direction_key, False)
        result = TranslationResult(
            source=source, source_language=profile["id"], source_form=source_form,
            target_language=target, target_form=None,
            transliteration=_basic_transliteration(source) if source_form != "translation" else None,
            translation=None, status="provider_required" if capable else "direction_not_registered",
            confidence=0.0, provider="none", provenance=None,
        )
    if log:
        _log_result(result, request_metadata)
    return result


def translation_matrix() -> dict[str, dict[str, bool]]:
    from .ancient_alphabet_registry import supported_alphabet_languages
    return {language_id: translation_directions(language_id) for language_id in supported_alphabet_languages()}
