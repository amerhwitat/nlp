"""OCR service that reuses the existing ``python/thamudic/media_pipeline`` module.

The primary path calls ``media_pipeline.extract_media_text`` which handles:

* text PDFs in-process via pypdf;
* images and scanned PDFs via EasyOCR inside an isolated worker subprocess
  (so a native crash can never take down the web server).

When the heavy EasyOCR engine is not installed, the service degrades to a
lightweight Tesseract fallback so the app still works on a minimal machine.
OCR failures are always reported as structured metadata -- never a fabricated
reading and never a process crash.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from . import config
from python.thamudic.media_pipeline import extract_media_text

_IMAGE_SUFFIXES = config.ALLOWED_IMAGE_SUFFIXES


def _easyocr_present() -> bool:
    try:
        return importlib.util.find_spec("easyocr") is not None
    except Exception:
        return False


def _tesseract_present() -> bool:
    try:
        import pytesseract  # type: ignore
        return bool(pytesseract.get_tesseract_version())
    except Exception:
        return False


def _tesseract_confidence(data: dict) -> float:
    values = [int(v) for v in data.get("conf", []) if isinstance(v, (int, float)) and int(v) > 0]
    if not values:
        return 0.0
    return round(sum(values) / len(values) / 100.0, 6)


def _tesseract_image(path: Path) -> tuple[str, float] | None:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
        image = Image.open(path)
        text = (pytesseract.image_to_string(image) or "").strip()
        if not text:
            return None
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        return text, _tesseract_confidence(data)
    except Exception:
        return None


def _tesseract_pdf(path: Path) -> tuple[str, float] | None:
    try:
        import pymupdf  # type: ignore
        import pytesseract  # type: ignore
        from PIL import Image
        document = pymupdf.open(path)
        pages: list[str] = []
        confidences: list[float] = []
        for page in document:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            text = (pytesseract.image_to_string(image) or "").strip()
            if text:
                pages.append(text)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences.append(_tesseract_confidence(data))
        joined = "\n".join(pages).strip()
        if not joined:
            return None
        return joined, (round(sum(confidences) / len(confidences), 6) if confidences else 0.0)
    except Exception:
        return None


def ocr_file(path: str | Path, languages: list[str] | None = None) -> dict[str, Any]:
    """OCR an uploaded image or PDF and return a structured, crash-safe result.

    ``ok`` is True only when non-empty text was extracted. Metadata always
    includes the engine/provider used so the UI can show provenance.
    """
    p = Path(path)
    suffix = p.suffix.casefold()
    media_type = "pdf" if suffix == ".pdf" else "image"
    languages = languages or config.DEFAULT_OCR_LANGUAGES

    meta: dict[str, Any] = {"media_type": media_type, "source_file": str(p)}

    # Primary: the existing media pipeline (pypdf / EasyOCR-in-subprocess).
    if suffix == ".pdf" or _easyocr_present():
        try:
            text, media = extract_media_text(p, ocr_languages=languages)
        except Exception as exc:  # pragma: no cover - defensive
            text, media = "", {"ocr_error": f"{type(exc).__name__}: {exc}", "ocr_available": False}
        meta.update(media)
        if text and text.strip():
            return {
                "ok": True,
                "text": text.strip(),
                "engine": str(media.get("provider", "pypdf")),
                "confidence": float(media.get("ocr_confidence", 0.0) or 0.0),
                "meta": meta,
            }

    # Fallback: lightweight Tesseract when the native engine is unavailable.
    fallback = None
    if suffix in _IMAGE_SUFFIXES and _tesseract_present():
        fallback = _tesseract_image(p)
    elif suffix == ".pdf" and _tesseract_present():
        fallback = _tesseract_pdf(p)

    if fallback and fallback[0].strip():
        text, confidence = fallback
        meta.update({"provider": "tesseract", "ocr_confidence": confidence, "ocr_available": True})
        return {"ok": True, "text": text, "engine": "tesseract", "confidence": confidence, "meta": meta}

    return {
        "ok": False,
        "text": "",
        "engine": "unavailable",
        "confidence": 0.0,
        "error": str(meta.get("ocr_error") or "No OCR engine is available for this file"),
        "meta": meta,
    }


def ocr_availability() -> dict[str, Any]:
    """Describe which OCR engines are available (used by /api/health)."""
    return {
        "easyocr": _easyocr_present(),
        "tesseract": _tesseract_present(),
        "pypdf": importlib.util.find_spec("pypdf") is not None,
        "pymupdf": importlib.util.find_spec("pymupdf") is not None or importlib.util.find_spec("fitz") is not None,
    }
