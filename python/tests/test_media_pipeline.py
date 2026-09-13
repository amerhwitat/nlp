from pathlib import Path

from thamudic.media_pipeline import extract_media_text, scan_translate_media


def test_text_media_roundtrip(tmp_path: Path):
    p = tmp_path / "sample.txt"
    p.write_text("ytm bn ʿbny w wgm ʿl- ḫll -h", encoding="utf-8")
    text, meta = extract_media_text(p)
    assert text.startswith("ytm bn")
    assert meta["provider"] == "utf8"


def test_media_pipeline_preserves_translation_status(tmp_path: Path):
    p = tmp_path / "sample.txt"
    p.write_text("ytm bn ʿbny w wgm ʿl- ḫll -h", encoding="utf-8")
    result = scan_translate_media(p, script="Safaitic", target_language="en")
    assert result["transliteration"]
    assert result["translation"]
    assert result["translation_ready"] is True
    assert result["media"]["provider"] == "utf8"
