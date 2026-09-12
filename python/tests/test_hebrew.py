from pathlib import Path

from ancient_languages.hebrew import load_hebrew_registry, codepoint_to_utf8


ROOT = Path(__file__).parents[2] / "data" / "ancient_languages"


def test_hebrew_historical_stages():
    registry = load_hebrew_registry(str(ROOT))
    assert {"hebrew.ancient", "hebrew.second_temple", "hebrew.masoretic", "hebrew.medieval", "hebrew.modern"}.issubset(set(registry.stages()))


def test_hebrew_alphabet_and_final_forms():
    registry = load_hebrew_registry(str(ROOT))
    members = registry.alphabet("hebrew.modern")
    assert len([m for m in members if m.name in {"Alef", "Bet", "Gimel", "Dalet", "He", "Waw", "Zayin", "Het", "Tet", "Yod", "Kaf", "Lamed", "Mem", "Nun", "Samekh", "Ayin", "Pe", "Tsadi", "Qof", "Resh", "Shin", "Tav"}]) == 22
    assert registry.lookup("ך")["codepoint"] == "U+05DA"


def test_stored_utf8_matches_unicode_encoding():
    registry = load_hebrew_registry(str(ROOT))
    for member in registry.alphabet("hebrew.modern"):
        expected = codepoint_to_utf8(int(member.codepoint[2:], 16)).hex(" ").upper()
        assert member.utf8 == expected


def test_hebrew_is_rtl_and_preserves_combining_capability():
    registry = load_hebrew_registry(str(ROOT))
    stage = next(s for s in registry.language["stages"] if s["id"] == "hebrew.masoretic")
    assert stage["direction"] == "rtl"
    assert stage["combining_marks"] is True
