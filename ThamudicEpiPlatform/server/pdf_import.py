from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


@dataclass
class PdfPage:
    page_number: int
    text: str
    char_count: int


@dataclass
class PdfImportResult:
    filename: str
    sha256: str
    page_count: int
    pages: list[PdfPage] = field(default_factory=list)
    total_chars: int = 0
    warnings: list[str] = field(default_factory=list)


def import_pdf(source: str | Path | bytes | BinaryIO, *, max_pages: int = 250, max_bytes: int = 50 * 1024 * 1024) -> PdfImportResult:
    """Extract text from a bounded PDF while preserving page-level provenance.

    Image/OCR processing is intentionally not implicit: callers can attach OCR results
    separately so that machine-generated text is never confused with source PDF text.
    """
    if max_pages < 1 or max_bytes < 1:
        raise ValueError("max_pages and max_bytes must be positive")

    filename = "document.pdf"
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = path.name
        raw = path.read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        filename = getattr(source, "name", filename)
        raw = source.read()

    if len(raw) > max_bytes:
        raise ValueError(f"PDF exceeds configured size limit of {max_bytes} bytes")
    if not raw.startswith(b"%PDF-"):
        raise ValueError("input is not a PDF")

    digest = hashlib.sha256(raw).hexdigest()
    reader = PdfReader(io.BytesIO(raw), strict=False)
    if len(reader.pages) > max_pages:
        raise ValueError(f"PDF exceeds configured page limit of {max_pages}")

    pages: list[PdfPage] = []
    warnings: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # defensive isolation of a damaged page
            text = ""
            warnings.append(f"page {number}: extraction failed: {type(exc).__name__}")
        pages.append(PdfPage(number, text, len(text)))

    return PdfImportResult(
        filename=filename,
        sha256=digest,
        page_count=len(pages),
        pages=pages,
        total_chars=sum(p.char_count for p in pages),
        warnings=warnings,
    )
