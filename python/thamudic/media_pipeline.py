"""Import image/PDF media, extract text, then scan/transliterate/translate it.

OCR is provider based: PDF text is extracted locally with pypdf when available;
images and scanned PDFs can use EasyOCR when installed. The pipeline never invents
ancient readings. OCR confidence and provider are retained in the result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ancient_translation import translate
from .source_language_scanner import scan_source_language


def _pdf_text(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF text extraction requires pypdf") from exc
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, 1):
        pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    return text, {"provider": "pypdf", "pages": len(reader.pages), "scanned_pages": not bool(text)}


def _easyocr_image(path: Path, languages: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    try:
        import easyocr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Image OCR requires optional dependency easyocr") from exc
    langs = languages or ["en", "ar"]
    reader = easyocr.Reader(langs, gpu=False, verbose=False)
    rows = reader.readtext(str(path), detail=1, paragraph=False)
    rows = sorted(rows, key=lambda x: (min(p[1] for p in x[0]), min(p[0] for p in x[0])))
    text = "\n".join(str(x[1]) for x in rows).strip()
    confidence = sum(float(x[2]) for x in rows) / len(rows) if rows else 0.0
    return text, {"provider": "easyocr", "detections": len(rows), "ocr_confidence": round(confidence, 6)}


def extract_media_text(path: str | Path, ocr_languages: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.casefold()
    if suffix == ".pdf":
        text, meta = _pdf_text(p)
        if text:
            return text, {"source_file": str(p), "media_type": "pdf", **meta}
        # Scanned PDFs: render pages and OCR through pypdfium2 if available.
        try:
            import pypdfium2 as pdfium  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Scanned PDF OCR requires pypdfium2 and easyocr") from exc
        reader = pdfium.PdfDocument(str(p))
        all_text = []
        confidences = []
        for i in range(len(reader)):
            page = reader[i]
            bitmap = page.render(scale=2.0)
            image_path = p.with_name(f".{p.stem}.page-{i+1}.png")
            bitmap.to_pil().save(image_path)
            try:
                page_text, ocr_meta = _easyocr_image(image_path, ocr_languages)
                all_text.append(page_text)
                if "ocr_confidence" in ocr_meta:
                    confidences.append(ocr_meta["ocr_confidence"])
            finally:
                image_path.unlink(missing_ok=True)
        meta = {"source_file": str(p), "media_type": "pdf", "provider": "pypdfium2+easyocr", "pages": len(reader), "ocr_confidence": round(sum(confidences)/len(confidences), 6) if confidences else 0.0, "scanned_pages": True}
        return "\n".join(x for x in all_text if x).strip(), meta
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        text, meta = _easyocr_image(p, ocr_languages)
        return text, {"source_file": str(p), "media_type": "image", **meta}
    if suffix in {".txt", ".md", ".csv"}:
        return p.read_text(encoding="utf-8"), {"source_file": str(p), "media_type": "text", "provider": "utf8"}
    raise ValueError(f"unsupported media type: {suffix}")


def scan_translate_media(path: str | Path, script: str = "Dadanitic", target_language: str = "en", ocr_languages: list[str] | None = None) -> dict[str, Any]:
    text, media = extract_media_text(path, ocr_languages)
    scan = scan_source_language(text, language="ancient-north-arabian" if script else None)
    result = translate(text, script=script, target_language=target_language)
    result["media"] = media
    result["scan"] = scan
    result["translation_ready"] = bool(result.get("translation"))
    result["workflow"] = "import -> extract/OCR -> scan -> transliterate -> corpus translation"
    return result
