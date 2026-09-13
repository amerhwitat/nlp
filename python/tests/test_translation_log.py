from pathlib import Path

from python.thamudic.translation_log import append_record, make_record, read_records, verify_records, export_records


def test_translation_log_round_trip(tmp_path: Path):
    path = tmp_path / "translations.jsonl"
    record = make_record(
        source="𐪀𐪁",
        source_language="ancient-north-arabian",
        source_form="script",
        target_language="en",
        transliteration="ʾb",
        translation=None,
        status="provider_required",
        confidence=0,
        provider="none",
        provenance=None,
        script_metadata={"writing_direction": "rtl"},
        request_metadata={"api": "/translate_ancient"},
    )
    append_record(record, path)
    records = read_records(path)
    assert len(records) == 1
    assert verify_records(records)["invalid"] == 0
    text, media, filename = export_records("txt", path)
    assert "provider_required" in text
    assert media.startswith("text/plain")
    assert filename == "translation-log.txt"


def test_tampering_is_detected(tmp_path: Path):
    path = tmp_path / "translations.jsonl"
    record = make_record(source="abc", source_language="latin", source_form="script", target_language="en")
    append_record(record, path)
    records = read_records(path)
    records[0]["source"] = "tampered"
    assert verify_records(records)["invalid"] == 1
