from ThamudicScan.server.exporter import export_results_csv, export_results_json


def test_export_csv_has_stable_columns():
    csv_text = export_results_csv([{
        "id": "1", "source": "test", "text": "𐪀",
        "transliteration": "h", "confidence": 0.9,
        "language": "Old North Arabian", "script_variant": "Dadanitic",
        "codepoints": [0x10A80],
    }])
    assert csv_text.splitlines()[0].startswith("id,session_id,source,text,transliteration,confidence")
    assert "𐪀" in csv_text


def test_export_json_preserves_unicode():
    payload = export_results_json([{"text": "𐪀", "confidence": 0.9}])
    assert "𐪀" in payload


def test_export_csv_empty_results_still_has_header():
    assert export_results_csv([]).startswith("id,session_id,source,text,transliteration,confidence")
