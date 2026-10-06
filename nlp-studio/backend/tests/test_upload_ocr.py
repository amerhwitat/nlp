import os

import pytest

from fastapi.testclient import TestClient  # noqa: F401  (used by fixture type hints)


def _font_candidates():
    return [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
    ]


def _make_text_image(tmp_path, text="ANCIENT SCRIPT"):
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in _font_candidates():
        if os.path.exists(candidate):
            font = ImageFont.truetype(candidate, 44)
            break
    image = Image.new("RGB", (720, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 38), text, fill="black", font=font)
    path = tmp_path / "sample.png"
    image.save(path)
    return path


def _engine_available(client):
    return client.get("/api/health").json()["ocr"]


def test_reject_non_image_extension(client):
    response = client.post(
        "/api/upload",
        files={"file": ("evil.txt", b"print('do not run')", "text/plain")},
    )
    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"]


def test_reject_oversize(client):
    big = b"0" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/upload",
        files={"file": ("big.png", big, "image/png")},
    )
    assert response.status_code == 400
    assert "too large" in response.json()["detail"]


def test_reject_mislabeled_pdf(client):
    response = client.post(
        "/api/upload",
        files={"file": ("fake.pdf", b"not a real pdf", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_image_ocr_smoke(client, tmp_path):
    engines = _engine_available(client)
    if not (engines.get("easyocr") or engines.get("tesseract")):
        pytest.skip("no OCR engine available in this environment")

    image_path = _make_text_image(tmp_path)
    with open(image_path, "rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": ("sample.png", handle, "image/png")},
            data={"script": "Dadanitic", "source_language": "ancient-north-arabian", "target_language": "en"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result_id"]
    assert body["upload"]["original_name"] == "sample.png"
    assert body["ocr"]["ok"] is True
    assert "ancient" in (body["ocr"]["text"] or "").lower()
    assert body["script_information"]["name"] == "Ancient North Arabian"

    # The result must be saved and exportable.
    result_id = body["result_id"]
    exported = client.get(f"/api/export/result/{result_id}?format=json")
    assert exported.status_code == 200
    assert exported.json()["result_id"] == result_id


def test_upload_text_pdf(client, tmp_path):
    """A text PDF is extracted via the existing pypdf path without any OCR engine."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    pdf_path = tmp_path / "blank.pdf"
    with open(pdf_path, "wb") as handle:
        writer.write(handle)

    with open(pdf_path, "rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": ("blank.pdf", handle, "application/pdf")},
            data={"script": "Dadanitic", "source_language": "ancient-north-arabian", "target_language": "en"},
        )
    assert response.status_code == 200
    assert response.json()["result_id"]
