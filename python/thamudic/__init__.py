"""Python facade for Thamudic and Ancient North Arabian text processing."""
from .old_north_arabian import BY_CHARACTER, BY_CODEPOINT, CHARACTERS, FIRST, LAST, VARIANT_FORMS, is_old_north_arabian, transliterate as transliterate_ona, utf8_bytes
from .ancient_translation import CorpusEntry, InMemoryCorpus, OCIANA_SEED, supported_targets, translate, transliterate_source

is_thamudic = is_old_north_arabian

def extract(text: str) -> str:
    return ''.join(ch for ch in text if is_old_north_arabian(ch))

def transliterate(text: str, mapping: dict[int, str] | None = None) -> str:
    if mapping is not None:
        return ''.join(mapping.get(ord(ch), '?' if is_old_north_arabian(ch) else ch) for ch in text)
    return transliterate_ona(text)

__all__ = [
    "BY_CHARACTER", "BY_CODEPOINT", "CHARACTERS", "FIRST", "LAST", "VARIANT_FORMS",
    "CorpusEntry", "InMemoryCorpus", "OCIANA_SEED", "extract", "is_old_north_arabian",
    "is_thamudic", "supported_targets", "translate", "transliterate", "transliterate_source",
    "utf8_bytes",
]
