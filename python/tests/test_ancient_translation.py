from python.thamudic import translate, supported_targets


def test_translation_returns_transliteration_and_english():
    result = translate("ytm bn ʿbny w wgm ʿl- ḫll -h", script="Safaitic", target_language="en")
    assert result["transliteration"] == "ytm bn ʿbny w wgm ʿl- ḫll -h"
    assert result["translation"]
    assert result["translation_status"] == "corpus_match"
    assert result["corpus_id"] == "TIJ 503"


def test_arabic_target_is_populated_for_known_entry():
    result = translate("bḏkrh wdd ḏ{h}k", script="Dadanitic", target_language="ar")
    assert result["translation"]
    assert result["target_language"] == "ar"


def test_unknown_text_is_not_fabricated():
    result = translate("unknown ancient fragment", script="Thamudic B", target_language="en")
    assert result["translation"] is None
    assert result["translation_status"] == "not_available"
    assert result["confidence"] == "unknown"


def test_supported_targets():
    assert supported_targets() == ("en", "ar")
