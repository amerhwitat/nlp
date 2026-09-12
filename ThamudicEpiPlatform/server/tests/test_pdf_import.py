from __future__ import annotations

import io

import pytest
from reportlab.pdfgen import canvas

from server.pdf_import import import_pdf


def make_pdf(text: str) -> bytes:
    out = io.BytesIO()
    c = canvas.Canvas(out)
    c.drawString(72, 720, text)
    c.save()
    return out.getvalue()


def test_import_pdf_preserves_page_provenance():
    result = import_pdf(make_pdf("Historical object report"))
    assert result.page_count == 1
    assert result.total_chars > 0
    assert len(result.sha256) == 64
    assert result.pages[0].page_number == 1


def test_import_pdf_rejects_non_pdf():
    with pytest.raises(ValueError, match="not a PDF"):
        import_pdf(b"not-a-pdf")


def test_import_pdf_enforces_size_limit():
    with pytest.raises(ValueError, match="size limit"):
        import_pdf(make_pdf("x"), max_bytes=4)
