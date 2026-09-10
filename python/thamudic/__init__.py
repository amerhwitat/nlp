"""Language-neutral Python facade for the Thamudic scanner family.

Legacy root scripts remain supported for backward compatibility; new language
implementations mirror this API instead of depending on GUI-specific code.
"""
FIRST, LAST = 0x10A80, 0x10A9F
def is_thamudic(codepoint: int) -> bool:
    return FIRST <= codepoint <= LAST
def extract(text: str) -> str:
    return ''.join(ch for ch in text if is_thamudic(ord(ch)))
def transliterate(text: str, mapping: dict[int,str]) -> str:
    return ''.join(mapping.get(ord(ch), '?' if is_thamudic(ord(ch)) else (' ' if ch.isspace() else '')) for ch in text)
