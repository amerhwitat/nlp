"""Evidence-first transliteration and translation services for Ancient North Arabian.

The module deliberately separates:
1. source inscription text;
2. scholarly transliteration;
3. target-language translation;
4. confidence/provenance.

It does not invent a translation when the corpus has no supported parallel reading.
Corpus-backed examples are based on published OCIANA records; additional corpora can
be supplied through CorpusEntry objects or a provider implementing lookup().
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Iterable, Protocol

from .old_north_arabian import transliterate as ona_transliterate


@dataclass(frozen=True)
class CorpusEntry:
    identifier: str
    script: str
    transliteration: str
    translations: dict[str, str]
    source_url: str
    confidence: str = "scholarly"


class TranslationProvider(Protocol):
    def lookup(self, transliteration: str, script: str) -> CorpusEntry | None: ...


# Seed records are exact corpus-backed examples used for deterministic offline behavior.
# They are not presented as a general Thamudic grammar or unrestricted MT model.
OCIANA_SEED: tuple[CorpusEntry, ...] = (
    CorpusEntry(
        "TIJ 503", "Safaitic", "ytm bn ʿbny w wgm ʿl- ḫll -h",
        {"en": "Ytm son ʿbny and he grieved for his friend",
         "ar": "يتم بن عبني وحزن على صديقه"},
        "https://ociana.osu.edu/inscriptions/2400",
    ),
    CorpusEntry(
        "AH 311", "Dadanitic", "bḏkrh wdd ḏ{h}k",
        {"en": "Bḏkrh loves {Ḏhk}", "ar": "بذَكرَه يحب {ذهك}"},
        "https://ociana.osu.edu/inscriptions/13954",
    ),
    CorpusEntry(
        "Is.H 806", "Thamudic B", "l ḍtm h- s¹fr w h- frs¹",
        {"en": "By Ḍtm are the inscription and the horse",
         "ar": "لِضَتم النقش والحصان"},
        "https://ociana.osu.edu/inscriptions/5826",
    ),
    CorpusEntry(
        "GETham 2", "Thamudic B", "l (l)hn wdd ns²l ḏ ʿtq",
        {"en": "By Ḏ son of . (Llhn) greets Ns²l who was freed",
         "ar": "من Ḏ بن . (للهن) يحيّي نس²ل الذي أُعتق"},
        "https://ociana.osu.edu/inscriptions/44105",
    ),
)


def _norm(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _latinize_for_lookup(value: str) -> str:
    """Normalize common punctuation/spacing without destroying scholarly signs."""
    value = value.replace("–", "-").replace("—", "-")
    return _norm(value)


class InMemoryCorpus:
    def __init__(self, entries: Iterable[CorpusEntry] = OCIANA_SEED):
        self.entries = tuple(entries)
        self._index = {(_latinize_for_lookup(e.transliteration), e.script.casefold()): e for e in self.entries}

    def lookup(self, transliteration: str, script: str = "") -> CorpusEntry | None:
        key = (_latinize_for_lookup(transliteration), script.casefold())
        if key in self._index:
            return self._index[key]
        # Script may be uncertain; exact transliteration remains preferable to guessing.
        candidates = [e for (t, _), e in self._index.items() if t == key[0]]
        return candidates[0] if len(candidates) == 1 else None


def transliterate_source(text: str) -> str:
    return ona_transliterate(text)


def translate(
    text: str,
    *,
    script: str = "Dadanitic",
    target_language: str = "en",
    provider: TranslationProvider | None = None,
) -> dict:
    """Return transliteration + evidence-backed target translation.

    If no corpus entry matches, translation is explicitly marked unavailable rather
    than fabricated. This is important for fragmentary Ancient North Arabian texts.
    """
    target = target_language.casefold().replace("_", "-")
    if target.startswith("ar"):
        target = "ar"
    elif target.startswith("en"):
        target = "en"
    transliteration = transliterate_source(text)
    provider = provider or InMemoryCorpus()
    entry = provider.lookup(transliteration, script)
    translation = entry.translations.get(target) if entry else None
    return {
        "source_text": text,
        "script": script,
        "transliteration": transliteration,
        "target_language": target,
        "translation": translation,
        "translation_status": "corpus_match" if translation else "not_available",
        "confidence": entry.confidence if entry else "unknown",
        "corpus_id": entry.identifier if entry else None,
        "provenance": entry.source_url if entry else None,
    }


def supported_targets() -> tuple[str, ...]:
    return ("en", "ar")


def entry_to_dict(entry: CorpusEntry) -> dict:
    return asdict(entry)
