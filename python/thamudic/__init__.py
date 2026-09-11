"""Python facade for Thamudic and Ancient North Arabian text processing."""
from .old_north_arabian import BY_CHARACTER, BY_CODEPOINT, CHARACTERS, FIRST, LAST, VARIANT_FORMS, is_old_north_arabian, transliterate as transliterate_ona, utf8_bytes

is_thamudic = is_old_north_arabian

def extract(text: str) -> str:
    return ''.join(ch for ch in text if is_old_north_arabian(ch))

def transliterate(text: str, mapping: dict[int, str] | None = None) -> str:
    if mapping is not None:
        return ''.join(mapping.get(ord(ch), '?' if is_old_north_arabian(ch) else ch) for ch in text)
    return transliterate_ona(text)
