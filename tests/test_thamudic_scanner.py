from pathlib import Path

from PIL import Image, ImageDraw

from thamudic_scanner import ONA, scan_image


def test_unicode_table_contains_old_north_arabian_block():
    assert 0x10A80 in ONA
    assert 0x10A9F in ONA
    assert ONA[0x10A8C] == "n"


def test_scan_detects_candidate_components(tmp_path: Path):
    image = Image.new("L", (160, 80), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 45, 55), fill=0)
    draw.line((70, 20, 100, 55), fill=0, width=5)
    source = tmp_path / "inscription.png"
    image.save(source)
    result = scan_image(source, threshold=150, min_area=10, scale=1)
    assert result["schema"] == "thamudic-scanner/v1"
    assert result["unicode_range"] == "U+10A80-U+10A9F"
    assert len(result["glyphs"]) >= 2
    assert result["recognition_status"] == "segmentation_only"
