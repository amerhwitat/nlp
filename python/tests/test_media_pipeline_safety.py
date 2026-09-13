from pathlib import Path

from thamudic.media_pipeline import extract_media_text


def test_text_import_remains_in_process(tmp_path: Path):
    source = tmp_path / "sample.txt"
    source.write_text("𐪀 test", encoding="utf-8")
    text, meta = extract_media_text(source)
    assert text == "𐪀 test"
    assert meta["provider"] == "utf8"


def test_image_ocr_failure_is_non_fatal(monkeypatch, tmp_path: Path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-a-real-image")

    def fail(*args, **kwargs):
        raise RuntimeError("simulated native OCR failure")

    monkeypatch.setattr("thamudic.media_pipeline._easyocr_image", fail)
    text, meta = extract_media_text(image)
    assert text == ""
    assert meta["ocr_available"] is False
    assert "simulated native OCR failure" in meta["ocr_error"]
