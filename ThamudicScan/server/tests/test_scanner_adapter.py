from ThamudicScan.server.scanner_adapter import scan_text, validate_text


def test_scanner_adapter_uses_existing_thamudic_api():
    result = scan_text("𐪀𐪁", ["h"])
    assert result["text"] == "𐪀𐪁"
    assert result["transliteration"] == "hl"
    assert result["confidence"] == 1.0


def test_validate_uses_canonical_unicode_registry():
    result = validate_text("x𐪀𐪁y")
    assert result["count"] == 2
    assert result["codepoints"] == [0x10A80, 0x10A81]
    assert result["characters"] == ["𐪀", "𐪁"]
