from io import BytesIO

from PIL import Image

from server.ocr_scanner import scan_image


def _png() -> bytes:
    image = Image.new('L', (640, 320), 255)
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def test_scanner_returns_provenance_and_quality_metadata():
    result = scan_image(_png(), engine='quality-only')
    assert result.source_sha256
    assert result.image_quality['width'] == 640
    assert result.image_quality['height'] == 320
    assert result.engine == 'quality-only'
    assert result.preprocessing


def test_scanner_rejects_oversized_input():
    try:
        scan_image(b'1234', max_bytes=3)
    except ValueError as exc:
        assert 'byte limit' in str(exc)
    else:
        raise AssertionError('expected byte limit error')
