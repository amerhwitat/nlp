"""Isolated worker for native OCR/PDF rendering dependencies.

EasyOCR/PyTorch and pypdfium2 can load native DLLs that conflict with a Tkinter
parent. This worker keeps those native operations in a child process so a native
failure cannot terminate the GUI. Results are JSON on stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ocr_image(path: Path, languages: list[str]) -> dict:
    import easyocr  # type: ignore

    reader = easyocr.Reader(languages, gpu=False, verbose=False)
    rows = reader.readtext(str(path), detail=1, paragraph=False)
    rows = sorted(rows, key=lambda x: (min(p[1] for p in x[0]), min(p[0] for p in x[0])))
    text = "\n".join(str(x[1]) for x in rows).strip()
    confidence = sum(float(x[2]) for x in rows) / len(rows) if rows else 0.0
    return {"text": text, "provider": "easyocr-subprocess", "detections": len(rows), "ocr_confidence": round(confidence, 6)}


def _render_pdf(path: Path, page_number: int, output: Path) -> dict:
    import pypdfium2 as pdfium  # type: ignore

    document = pdfium.PdfDocument(str(path))
    if page_number < 0 or page_number >= len(document):
        raise IndexError(f"PDF page index out of range: {page_number}")
    page = document[page_number]
    bitmap = page.render(scale=2.0)
    bitmap.to_pil().save(output)
    return {"output": str(output), "provider": "pypdfium2-subprocess", "page": page_number + 1}


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated EasyOCR/pypdfium2 worker")
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--languages", default="en,ar")
    parser.add_argument("--render-pdf", type=Path)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.render_pdf:
            if not args.output:
                raise ValueError("--output is required with --render-pdf")
            if not args.render_pdf.exists():
                raise FileNotFoundError(args.render_pdf)
            result = _render_pdf(args.render_pdf, args.page, args.output)
        else:
            if not args.image:
                raise ValueError("an image path is required")
            if not args.image.exists():
                raise FileNotFoundError(args.image)
            languages = [x.strip() for x in args.languages.split(",") if x.strip()] or ["en", "ar"]
            result = _ocr_image(args.image, languages)
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
