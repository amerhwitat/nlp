"""Canonical Old North Arabian (Ancient North Arabian) Unicode registry.

The Unicode repertoire is U+10A80..U+10A9F. Unicode encodes the repertoire
using Dadanitic forms; Safaitic, Hismaic, Taymanitic, Minaic and Thamudic B
have historically attested variant glyph forms that should be represented by
font/variant metadata rather than invented Unicode code points.
"""

FIRST, LAST = 0x10A80, 0x10A9F
VARIANT_FORMS = ("Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Minaic", "Thamudic B")
_NAMES = [
    "HEH", "LAM", "HAH", "MEEM", "QAF", "WAW", "ES-2", "REH", "BEH", "TEH", "ES-1", "KAF", "NOON", "KHAH", "SAD", "ES-3",
    "FEH", "ALEF", "AIN", "DAD", "GEEM", "DAL", "GHAIN", "TAH", "ZAIN", "THAL", "YEH", "THEH", "ZAH",
]
_TRANSLITERATION = ["h", "l", "ḥ", "m", "q", "w", "s2", "r", "b", "t", "s1", "k", "n", "ḫ", "ṣ", "s3", "f", "ʼ", "ʽ", "ḍ", "g", "d", "ġ", "ṭ", "z", "ḏ", "y", "ṯ", "ẓ"]

CHARACTERS = tuple({
    "codepoint": cp,
    "character": chr(cp),
    "name": f"OLD NORTH ARABIAN LETTER {name}",
    "transliteration": transliteration,
    "utf8": chr(cp).encode("utf-8"),
    "utf8_hex": chr(cp).encode("utf-8").hex(" ").upper(),
} for cp, name, transliteration in zip(range(FIRST, 0x10A9D), _NAMES, _TRANSLITERATION)) + tuple({
    "codepoint": cp,
    "character": chr(cp),
    "name": f"OLD NORTH ARABIAN NUMBER {label}",
    "transliteration": value,
    "utf8": chr(cp).encode("utf-8"),
    "utf8_hex": chr(cp).encode("utf-8").hex(" ").upper(),
} for cp, label, value in ((0x10A9D, "ONE", "1"), (0x10A9E, "TEN", "10"), (0x10A9F, "TWENTY", "20")))

BY_CODEPOINT = {item["codepoint"]: item for item in CHARACTERS}
BY_CHARACTER = {item["character"]: item for item in CHARACTERS}

def is_old_north_arabian(value: int | str) -> bool:
    cp = ord(value) if isinstance(value, str) else value
    return FIRST <= cp <= LAST

def utf8_bytes(value: int | str) -> bytes:
    cp = ord(value) if isinstance(value, str) else value
    return chr(cp).encode("utf-8")

def transliterate(text: str) -> str:
    return "".join(BY_CHARACTER[ch]["transliteration"] if ch in BY_CHARACTER else ch for ch in text)
