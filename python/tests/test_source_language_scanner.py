from python.thamudic import encode_utf8, scan_source_language, supported_source_languages


def test_supported_source_languages():
    assert set(supported_source_languages()) == {
        "ancient-egyptian", "chinese", "japanese", "greek", "latin"
    }


def test_ancient_egyptian_unicode_and_utf8():
    result = scan_source_language("𓀀", language="ancient-egyptian")
    assert result["matched_character_count"] == 1
    item = result["characters"][0]
    assert item["codepoint"] == "U+13000"
    assert item["utf8"] == "F0 93 80 80"


def test_greek_extended():
    result = scan_source_language("ἄ", language="greek")
    assert result["matched_character_count"] == 1
    assert result["detected_languages"] == ["greek"]


def test_japanese_kana_and_chinese_han_overlap_is_visible():
    result = scan_source_language("日本語かな")
    assert result["counts"]["japanese"] == 5
    assert result["counts"]["chinese"] == 3
    assert result["ambiguous_script_overlap"] is True


def test_latin_utf8():
    result = scan_source_language("Cicero", language="latin")
    assert result["matched_character_count"] == 6
    encoded = encode_utf8("é")
    assert encoded["hex"] == "C3 A9"
